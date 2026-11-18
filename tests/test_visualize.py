import wave
import re
import shutil
import subprocess
import sys
import time

import imageio_ffmpeg
import numpy as np
import pytest
from pyannote.core import Annotation, Segment

from diarization.rttm import write_rttm
from diarization.visualize import (
    F0_EDGECOLORS,
    F0_LINEWIDTHS,
    F0_POINT_SIZE,
    SPEC_CMAP,
    crop_position,
    disagreement_spans,
    load_mono,
    render,
    spectrogram,
)

PNG = b"\x89PNG\r\n\x1a\n"


def write_tone(path, sample_rate=16000):
    silence = int(sample_rate * 0.5)
    count = int(sample_rate * 2.0)
    times = np.arange(count) / sample_rate
    tone = (0.5 * np.sin(2 * np.pi * 200.0 * times) * 32767).astype(np.int16)
    samples = np.concatenate([
        np.zeros(silence, dtype=np.int16), tone, np.zeros(silence, dtype=np.int16),
    ])
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(samples.tobytes())


def write_noise(path, duration_s=5.0, sample_rate=16000, seed=0):
    count = int(sample_rate * duration_s)
    samples = (0.2 * np.random.default_rng(seed).standard_normal(count) * 32767)
    samples = np.clip(samples, -32767, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(samples.tobytes())


def write_spans(path, spans, uri="tono"):
    annotation = Annotation(uri=uri)
    for index, (start, end, label) in enumerate(spans):
        annotation[Segment(start, end), index] = label
    write_rttm(annotation, path)


def test_synthetic_tone_png_and_pitch(tmp_path):
    audio, rttm, png = tmp_path / "tono.wav", tmp_path / "tono.rttm", tmp_path / "tono.png"
    write_tone(audio)
    write_spans(rttm, [(0.0, 3.0, "SPEAKER_00")])
    started = time.perf_counter()
    summary = render(audio, rttm, png)
    print(f"tono sintético de 3 s: {time.perf_counter() - started:.2f} s (incluye imagen)")
    voiced = np.isfinite(summary["hz"])
    assert png.read_bytes()[:8] == PNG
    assert voiced.any()
    assert np.median(summary["hz"][voiced]) == pytest.approx(200.0, rel=0.05)
    assert not voiced[summary["times"] < 0.35].any()
    assert not voiced[summary["times"] > 2.65].any()
    render(audio, rttm, png)
    assert list(tmp_path.glob("*.png")) == [png]
    assert png.read_bytes()[:8] == PNG


def test_dry_run_does_not_write(tmp_path):
    audio, rttm, png = tmp_path / "tono.wav", tmp_path / "tono.rttm", tmp_path / "out" / "tono.png"
    write_tone(audio)
    write_spans(rttm, [(0.5, 2.5, "SPEAKER_00")])
    summary = render(audio, rttm, png, dry_run=True)
    assert summary["hablantes"] == 1
    assert summary["turnos"] == 1
    assert summary["duracion_s"] == pytest.approx(3.0, abs=0.01)
    assert "hz" not in summary
    assert not png.exists()


def test_disagreement_spans_cover_label_mismatch():
    hypothesis = Annotation(uri="ev99")
    hypothesis[Segment(0.0, 1.0), 0] = "A"
    reference = Annotation(uri="ev99")
    reference[Segment(0.0, 0.4), 0] = "A"
    reference[Segment(0.4, 1.0), 1] = "B"
    spans = disagreement_spans(hypothesis, reference, 0.0, 1.0)
    assert len(spans) == 1
    start, duration = spans[0]
    assert start == pytest.approx(0.0, abs=0.01)
    assert start + duration == pytest.approx(0.4, abs=0.01)


def test_reference_png(tmp_path):
    audio, hyp, ref, png = (
        tmp_path / "tono.wav", tmp_path / "hyp.rttm", tmp_path / "ref.rttm", tmp_path / "tono.png",
    )
    write_tone(audio)
    write_spans(hyp, [(0.5, 2.5, "SPEAKER_00")])
    write_spans(ref, [(0.5, 1.5, "SPEAKER_00"), (1.5, 2.5, "SPEAKER_01")])
    result = subprocess.run([sys.executable, "-m", "diarization.visualize", "--audio", str(audio),
                             "--rttm", str(hyp), "--ref", str(ref), "--ref-label", "variante",
                             "--out", str(png)], capture_output=True, text=True)
    assert result.returncode == 0
    assert png.read_bytes()[:8] == PNG


def test_tone_start_under_cursor():
    fps, pixels_per_second, window_pixels = 30, 100, 2000
    tone_start = 0.5
    frame = round(tone_start * fps)
    tone_pixel = window_pixels / 2 + tone_start * pixels_per_second
    position = tone_pixel - crop_position(frame / fps, pixels_per_second)
    assert abs(position - window_pixels / 2) <= pixels_per_second / fps


def test_synthetic_video_has_synchronized_tracks(tmp_path):
    audio, rttm = tmp_path / "tono.wav", tmp_path / "tono.rttm"
    png, video = tmp_path / "tono.png", tmp_path / "tono.mp4"
    write_tone(audio)
    write_spans(rttm, [(0.5, 2.5, "SPEAKER_00")])
    render(audio, rttm, png, video=video)
    ffmpeg = shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe()
    result = subprocess.run([ffmpeg, "-i", str(video)], capture_output=True, text=True)
    assert "Video: h264" in result.stderr
    assert "Audio: aac" in result.stderr
    duration = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", result.stderr)
    assert duration is not None
    hours, minutes, seconds = map(float, duration.groups())
    assert hours * 3600 + minutes * 60 + seconds == pytest.approx(3.0, abs=0.1)


def test_mapping_and_all_disagreements():
    hypothesis = Annotation(uri="ev99")
    reference = Annotation(uri="ev99")
    hypothesis[Segment(0, 14), 0] = "A"
    reference[Segment(0, 14), 0] = "X"
    for index in range(7):
        reference[Segment(index * 2, index * 2 + 0.5), index + 1] = "Y"
    spans = disagreement_spans(hypothesis, reference, 0, 14)
    assert len(spans) == 7
    assert all(duration == pytest.approx(0.5) for _, duration in spans)


def test_force_regenerates_video(tmp_path):
    audio, rttm = tmp_path / "tono.wav", tmp_path / "tono.rttm"
    png, video = tmp_path / "tono.png", tmp_path / "tono.mp4"
    write_tone(audio)
    write_spans(rttm, [(0, 3, "A")])
    video.write_bytes(b"existente")
    with pytest.raises(ValueError, match="ya existe"):
        render(audio, rttm, png, video=video)
    assert video.read_bytes() == b"existente"
    render(audio, rttm, png, video=video, force=True, window=2, px_per_s=20, fps=5)
    assert video.stat().st_size > len(b"existente")


def test_spectrogram_rows_capped(tmp_path):
    audio = tmp_path / "tono.wav"
    write_tone(audio)
    waveform, sample_rate = load_mono(audio)
    _, frequencies, spec = spectrogram(waveform, sample_rate, 0.0)
    assert spec.shape[0] == len(frequencies) <= 256
    assert frequencies[0] >= 50.0


def test_spectrogram_no_empty_log_bands(tmp_path):
    audio = tmp_path / "ruido.wav"
    write_noise(audio, duration_s=5.0)
    waveform, sample_rate = load_mono(audio)
    _, _, spec = spectrogram(waveform, sample_rate, 0.0)
    floor = float(np.min(spec))
    assert spec.shape[0] <= 256
    # Bandas vacías quedaban enteras en el piso de top_db (−80 dB).
    assert not np.any(np.all(np.isclose(spec, floor, atol=1e-5), axis=1))


def test_f0_contrast_over_gray_spectrogram(tmp_path, monkeypatch):
    assert SPEC_CMAP != "magma"
    assert SPEC_CMAP == "gray"
    assert F0_EDGECOLORS == "black"
    assert F0_LINEWIDTHS > 0
    assert F0_POINT_SIZE >= 10

    captured = {"cmaps": [], "scatters": []}
    import matplotlib.pyplot as plt

    original_pcolormesh = plt.Axes.pcolormesh
    original_scatter = plt.Axes.scatter

    def capture_pcolormesh(self, *args, **kwargs):
        captured["cmaps"].append(kwargs.get("cmap"))
        return original_pcolormesh(self, *args, **kwargs)

    def capture_scatter(self, *args, **kwargs):
        captured["scatters"].append(kwargs)
        return original_scatter(self, *args, **kwargs)

    monkeypatch.setattr(plt.Axes, "pcolormesh", capture_pcolormesh)
    monkeypatch.setattr(plt.Axes, "scatter", capture_scatter)

    audio, hyp, ref, png = (
        tmp_path / "tono.wav",
        tmp_path / "hyp.rttm",
        tmp_path / "ref.rttm",
        tmp_path / "tono.png",
    )
    write_tone(audio)
    write_spans(hyp, [(0.5, 2.5, "SPEAKER_00")])
    write_spans(ref, [(0.5, 1.5, "SPEAKER_00"), (1.5, 2.5, "SPEAKER_01")])
    render(audio, hyp, png, ref=ref)
    assert png.read_bytes()[:8] == PNG
    assert captured["cmaps"]
    assert all(cmap == SPEC_CMAP and cmap != "magma" for cmap in captured["cmaps"])
    assert captured["scatters"]
    for kwargs in captured["scatters"]:
        assert kwargs.get("edgecolors") == F0_EDGECOLORS
        assert kwargs.get("linewidths") == F0_LINEWIDTHS
        assert kwargs.get("s") == F0_POINT_SIZE
