import wave
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from diarization import preprocess as preproc


def write_tone(path, duration_s=1.0, sample_rate=16000, channels=1):
    frames = int(sample_rate * duration_s)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(bytes(frames * channels * 2))


def write_voice_like(path, duration_s=8.0, sample_rate=16000, seed=0):
    rng = np.random.default_rng(seed)
    n = int(sample_rate * duration_s)
    times = np.arange(n) / sample_rate
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 2.5 * times)
    voice = (
        0.35 * np.sin(2 * np.pi * 120.0 * times)
        + 0.18 * np.sin(2 * np.pi * 240.0 * times)
        + 0.08 * np.sin(2 * np.pi * 360.0 * times)
    )
    noise = 0.04 * rng.standard_normal(n)
    samples = np.clip((voice * envelope + noise) * 0.9, -1.0, 1.0).astype(np.float32)
    sf.write(str(path), samples, sample_rate, subtype="PCM_16")


def test_techniques_preserve_duration(tmp_path):
    source = tmp_path / "in.wav"
    write_tone(source, duration_s=1.25)
    input_duration = preproc.duration(source)
    for name in ("loudnorm", "highpass80", "afftdn"):
        destination = tmp_path / name / "out.wav"
        preproc.process(name, source, destination, reuse=False)
        assert destination.is_file()
        assert preproc.meta_path(destination).is_file()
        assert preproc.duration(destination) == pytest.approx(input_duration, abs=0.01)
        rate, channels, _ = preproc.wav_info(destination)
        assert rate == 16000
        assert channels == 1


def test_reuse_skips_ffmpeg_and_reads_shift(tmp_path, monkeypatch):
    source = tmp_path / "in.wav"
    write_voice_like(source, duration_s=1.0)
    destination = tmp_path / "loudnorm" / "ev99.wav"
    preproc.process("loudnorm", source, destination, reuse=False)
    meta = preproc.read_shift(destination)
    assert meta is not None
    calls = []

    def boom(*args, **kwargs):
        calls.append("ffmpeg")
        raise AssertionError("ffmpeg no debería ejecutarse")

    def boom_align(*args, **kwargs):
        calls.append("align")
        raise AssertionError("no debería realinear en reutilización")

    monkeypatch.setattr(preproc, "_run_ffmpeg_filter", boom)
    monkeypatch.setattr(preproc, "align_processed", boom_align)
    again = preproc.process("loudnorm", source, destination, reuse=True)
    assert again == destination
    assert calls == []
    assert preproc.read_shift(destination) == meta


def test_duration_mismatch_raises_and_removes(tmp_path, monkeypatch):
    source = tmp_path / "in.wav"
    write_tone(source, duration_s=1.0)
    destination = tmp_path / "broken.wav"

    def short_output(src, dst, filter_expr, sample_rate, channels):
        write_tone(dst, duration_s=0.5, sample_rate=sample_rate, channels=channels)

    monkeypatch.setattr(preproc, "_run_ffmpeg_filter", short_output)
    with pytest.raises(ValueError, match="Duración distinta"):
        preproc.process("highpass80", source, destination, reuse=False)
    assert not destination.exists()
    assert not preproc.meta_path(destination).exists()


def test_register_extends_techniques(tmp_path):
    source = tmp_path / "in.wav"
    write_tone(source, duration_s=0.5)
    destination = tmp_path / "copy.wav"
    seen = []

    def copy_apply(src, dst):
        seen.append((Path(src), Path(dst)))
        Path(dst).write_bytes(Path(src).read_bytes())

    preproc.register("copia_test", copy_apply)
    try:
        assert "copia_test" in preproc.TECHNIQUES
        preproc.process("copia_test", source, destination, reuse=False)
        assert destination.is_file()
        assert len(seen) == 1
        assert seen[0][0] == source
        assert seen[0][1].name.startswith(".preproc-raw-")
    finally:
        preproc.TECHNIQUES.pop("copia_test", None)


def test_fake_delay_is_compensated(tmp_path):
    sample_rate = 16000
    source = tmp_path / "in.wav"
    write_voice_like(source, duration_s=2.0, sample_rate=sample_rate)
    delay = 400

    def delay_apply(src, dst):
        data, rate = sf.read(str(src), dtype="float32", always_2d=True)
        delayed = np.concatenate(
            [np.zeros((delay, data.shape[1]), dtype=np.float32), data[: data.shape[0] - delay]],
            axis=0,
        )
        sf.write(str(dst), delayed, rate, subtype="PCM_16")

    preproc.register("retraso400", delay_apply)
    try:
        destination = tmp_path / "out.wav"
        preproc.process("retraso400", source, destination, reuse=False)
        reference, rate = sf.read(str(source), dtype="float32", always_2d=True)
        aligned, _ = sf.read(str(destination), dtype="float32", always_2d=True)
        remaining = preproc.estimate_lag_samples(reference, aligned, rate)
        assert remaining is not None
        assert abs(remaining) <= max(1, int(round(0.001 * rate)))
        meta = preproc.read_shift(destination)
        assert meta["desfase_muestras"] == delay
        assert meta["compensado"] is True
    finally:
        preproc.TECHNIQUES.pop("retraso400", None)


def test_afftdn_voice_like_is_aligned(tmp_path):
    source = tmp_path / "voice.wav"
    write_voice_like(source, duration_s=8.0)
    destination = tmp_path / "afftdn" / "out.wav"
    preproc.process("afftdn", source, destination, reuse=False)
    reference, rate = sf.read(str(source), dtype="float32", always_2d=True)
    aligned, _ = sf.read(str(destination), dtype="float32", always_2d=True)
    remaining = preproc.estimate_lag_samples(reference, aligned, rate)
    assert remaining is not None
    assert abs(remaining) <= max(1, int(round(0.001 * rate)))
    meta = preproc.read_shift(destination)
    assert meta is not None
    assert meta["omitido"] is False
