import io
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from pyannote.core import Annotation, Segment

from diarization import diagnose as diag
from diarization.rttm import write_rttm


SR = 16000


def write_wav(path: Path, audio: np.ndarray, sample_rate: int = SR) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio.astype(np.float32), sample_rate, subtype="PCM_16")
    return path


def white_noise(n: int, seed: int = 0, scale: float = 0.1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (scale * rng.standard_normal(n)).astype(np.float64)


def test_hum_50hz_within_one_db(tmp_path):
    """Zumbido agregado calibrado a un exceso conocido (±1 dB)."""
    n = SR * 20
    times = np.arange(n) / SR
    noise = white_noise(n, seed=1, scale=0.1)
    target_db = 25.0
    lo, hi = 1e-4, 0.5
    audio = noise
    for _ in range(40):
        mid = (lo + hi) / 2.0
        candidate = noise + mid * np.sin(2 * np.pi * 50.0 * times)
        value = diag.measure_hum(candidate, SR)["hum50_db"]
        if value < target_db:
            lo = mid
        else:
            hi = mid
        audio = candidate
    put_value = diag.measure_hum(audio, SR)["hum50_db"]
    assert put_value == pytest.approx(target_db, abs=0.5)
    path = write_wav(tmp_path / "hum.wav", audio)
    measured = diag.diagnose_path(path, already_wav_16k=True)["hum50_db"]
    assert measured == pytest.approx(put_value, abs=1.0)


def test_clip_ratio_clipped_sine(tmp_path):
    n = SR * 4
    times = np.arange(n) / SR
    audio = np.clip(1.5 * np.sin(2 * np.pi * 440.0 * times), -1.0, 1.0)
    path = write_wav(tmp_path / "clip.wav", audio)
    measures = diag.diagnose_path(path, already_wav_16k=True)
    assert measures["clip_ratio"] > 0.05
    assert measures["clip_plateau"] == 1.0


def test_clean_noise_no_false_alarms(tmp_path):
    audio = white_noise(SR * 8, seed=3, scale=0.2)
    path = write_wav(tmp_path / "clean.wav", audio)
    measures = diag.diagnose_path(path, already_wav_16k=True)
    assert measures["clip_ratio"] == pytest.approx(0.0, abs=1e-4)
    assert measures["clip_plateau"] == 0.0
    assert measures["hum50_db"] < 6.0


def test_cutoff_lowpass_4khz(tmp_path):
    n = SR * 12
    noise = white_noise(n, seed=4, scale=0.5)
    spectrum = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(n, d=1.0 / SR)
    spectrum[freqs > 4000.0] = 0.0
    filtered = np.fft.irfft(spectrum, n=n)
    peak = np.max(np.abs(filtered)) + 1e-12
    filtered = 0.8 * filtered / peak
    path = write_wav(tmp_path / "lp.wav", filtered)
    measured = diag.diagnose_path(path, already_wav_16k=True)["cutoff_hz"]
    assert measured == pytest.approx(4000.0, rel=0.10)


def test_snr_known_mask(tmp_path):
    n = SR * 10
    mask = np.zeros(n, dtype=bool)
    mask[SR * 2 : SR * 7] = True
    speech_amp = 0.2
    noise_amp = 0.02
    rng = np.random.default_rng(5)
    audio = noise_amp * rng.standard_normal(n)
    audio[mask] += speech_amp * rng.standard_normal(np.count_nonzero(mask))
    designed = 10.0 * np.log10((speech_amp ** 2 + noise_amp ** 2) / (noise_amp ** 2))
    path = write_wav(tmp_path / "snr.wav", audio)
    measured = diag.snr_db(audio, mask)
    assert measured == pytest.approx(designed, abs=1.0)
    annotation = Annotation(uri="ev99")
    annotation[Segment(2.0, 7.0)] = "SPEAKER_00"
    rttm = tmp_path / "ev99.rttm"
    write_rttm(annotation, rttm)
    via = diag.diagnose_path(path, rttm_path=rttm, already_wav_16k=True)["snr_db"]
    assert via == pytest.approx(designed, abs=1.0)


def test_lufs_known_sine(tmp_path):
    n = SR * 5
    times = np.arange(n) / SR
    amplitude = 0.5
    audio = amplitude * np.sin(2 * np.pi * 1000.0 * times)
    path = write_wav(tmp_path / "sine.wav", audio)
    expected = -3.0 + 20.0 * np.log10(amplitude)
    measured = diag.diagnose_path(path, already_wav_16k=True)["lufs"]
    assert measured == pytest.approx(expected, abs=0.5)


def test_pico_bajo_63hz(tmp_path):
    n = SR * 16
    times = np.arange(n) / SR
    audio = white_noise(n, seed=7, scale=0.05) + 0.08 * np.sin(2 * np.pi * 63.0 * times)
    path = write_wav(tmp_path / "pico63.wav", audio)
    measured = diag.diagnose_path(path, already_wav_16k=True)
    assert measured["pico_bajo_hz"] == pytest.approx(63.0, abs=1.0)
    assert measured["pico_bajo_db"] > 10.0


def test_clicks_harmonic_bursts_low(tmp_path):
    """Ráfagas armónicas con ataque brusco no deben contar como clics."""
    n = SR * 12
    rng = np.random.default_rng(8)
    audio = 0.02 * rng.standard_normal(n)
    for start_s in np.arange(0.0, 11.0, 0.5):
        i0 = int(start_s * SR)
        i1 = min(n, i0 + int(0.15 * SR))
        t = np.arange(i1 - i0) / SR
        audio[i0:i1] += 0.35 * (
            np.sin(2 * np.pi * 120.0 * t)
            + 0.4 * np.sin(2 * np.pi * 240.0 * t)
        )
    path = write_wav(tmp_path / "bursts.wav", audio)
    rate = diag.diagnose_path(path, already_wav_16k=True)["clicks_por_min"]
    assert rate < 5.0


def test_clicks_thirty_inserted(tmp_path):
    n = SR * 20
    rng = np.random.default_rng(9)
    audio = 0.05 * rng.standard_normal(n)
    positions = np.linspace(SR, n - SR, 30, dtype=int)
    for pos in positions:
        audio[pos] = 0.95
    path = write_wav(tmp_path / "clicks30.wav", audio)
    rate = diag.diagnose_path(path, already_wav_16k=True)["clicks_por_min"]
    expected_per_min = 30 / (n / SR / 60.0)
    assert rate == pytest.approx(expected_per_min, rel=0.20)


def test_manifest_and_inputs_cli(tmp_path, monkeypatch):
    audio = white_noise(SR * 3, seed=6, scale=0.15)
    sensitive = tmp_path / "NombrePersona_entrevista.wav"
    wav = write_wav(sensitive, audio)
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "uri,ruta_audio,duracion_s,n_hablantes_esperado,tipo,notas\n"
        f"ev99,{wav.as_posix()},3,,,test\n",
        encoding="utf-8",
    )
    out_m = tmp_path / "diag-m"
    diag.main(["--manifest", str(manifest), "--out", str(out_m), "--rttm-dir", str(tmp_path)])
    csv_m = (out_m / "diagnostico.csv").read_text(encoding="utf-8").splitlines()
    assert "hum100_db" in csv_m[0]
    assert "pico_bajo_hz" in csv_m[0]
    assert "ev99" in csv_m[1]
    assert (out_m / "ev99_ltas.png").is_file()

    listing = tmp_path / "lista.txt"
    listing.write_text(f"{wav.as_posix()}\n", encoding="utf-8")
    out_i = tmp_path / "diag-i"
    monkeypatch.setattr(diag, "_is_git_ignored", lambda path: True)
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        diag.main(["--inputs", str(listing), "--out", str(out_i), "--max-minutes", "1"])
    combined = stdout.getvalue() + stderr.getvalue()
    assert "NombrePersona" not in combined
    assert "1/1: ok" in combined
    assert (out_i / "1_ltas.png").is_file()
    assert not any(out_i.glob("*NombrePersona*"))
    csv_i = (out_i / "diagnostico.csv").read_text(encoding="utf-8")
    assert wav.as_posix() in csv_i


def test_inputs_rejects_non_ignored_out(tmp_path, monkeypatch):
    audio = white_noise(SR * 2, seed=10, scale=0.1)
    wav = write_wav(tmp_path / "NombrePersona_x.wav", audio)
    listing = tmp_path / "lista.txt"
    listing.write_text(f"{wav.as_posix()}\n", encoding="utf-8")
    out_i = tmp_path / "not-ignored"
    monkeypatch.setattr(diag, "_is_git_ignored", lambda path: False)
    with pytest.raises(SystemExit) as caught:
        diag.main(["--inputs", str(listing), "--out", str(out_i)])
    assert caught.value.code == 1
    assert not (out_i / "diagnostico.csv").exists()
    assert list(out_i.glob("*")) == [] or not out_i.exists()
