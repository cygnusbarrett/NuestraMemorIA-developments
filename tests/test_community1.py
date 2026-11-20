"""Tests del backend community1 (subproceso aislado) y su cableado en evaluate/compare."""

from __future__ import annotations

import csv
import json
import os
import wave
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from pyannote.core import Annotation, Segment

from diarization import compare as compare_mod
from diarization import evaluate
from diarization.backends import community1
from diarization.rttm import write_rttm


def _write_rttm_text(path: Path, uri: str = "ev01") -> None:
    path.write_text(
        f"SPEAKER {uri} 1 0.000000 1.000000 <NA> <NA> SPEAKER_00 <NA> <NA>\n"
        f"SPEAKER {uri} 1 0.500000 1.000000 <NA> <NA> SPEAKER_01 <NA> <NA>\n",
        encoding="utf-8",
    )


def test_community1_command_and_rttm_to_annotation(tmp_path, monkeypatch):
    audio = tmp_path / "ev01.wav"
    with wave.open(str(audio), "wb") as handle:
        handle.setparams((1, 2, 16000, 16000, "NONE", "not compressed"))
        handle.writeframes(bytes(32000))
    seen = []

    def fake_run(command, check=False, capture_output=True, text=True, env=None):
        seen.append(list(command))
        assert env is not None
        assert "PATH" in env
        rttm = Path(command[command.index(str(community1.COMMUNITY1_SCRIPT)) + 2])
        # El script recibe: ... run.py <entrada> <salida> ...
        assert command[command.index(str(community1.COMMUNITY1_SCRIPT)) + 1] == str(audio)
        _write_rttm_text(rttm, uri="ev01")
        return type("Result", (), {"returncode": 0, "stderr": "", "stdout": ""})()

    monkeypatch.setattr(community1.subprocess, "run", fake_run)
    result = community1.diarize(
        audio, device="cpu", num_speakers=2, min_speakers=1, max_speakers=3,
    )
    assert isinstance(result, Annotation)
    assert result.uri == "ev01"
    assert set(result.labels()) == {"SPEAKER_00", "SPEAKER_01"}
    assert len(seen) == 1
    command = seen[0]
    assert command[:3] == ["uv", "run", "--frozen"]
    assert command[command.index("--project") + 1] == str(community1.COMMUNITY1_PROJECT)
    assert str(community1.COMMUNITY1_SCRIPT) in command
    assert "--device" in command and "cpu" in command
    assert command[command.index("--num-speakers") + 1] == "2"
    assert command[command.index("--min-speakers") + 1] == "1"
    assert command[command.index("--max-speakers") + 1] == "3"
    assert not any(Path(part).suffix == ".rttm" and Path(part).exists() for part in command)


def test_community1_subprocess_error_is_clear(tmp_path, monkeypatch):
    audio = tmp_path / "ev01.wav"
    audio.write_bytes(b"x")

    def fake_run(command, check=False, capture_output=True, text=True, env=None):
        return type(
            "Result",
            (),
            {"returncode": 2, "stderr": "Falta HUGGING_FACE_TOKEN en el entorno\n", "stdout": ""},
        )()

    monkeypatch.setattr(community1.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="Backend community1 falló"):
        community1.Backend(device="cpu").diarize(audio)


def test_community1_empty_rttm_returns_empty_annotation(tmp_path, monkeypatch):
    audio = tmp_path / "ev01.wav"
    audio.write_bytes(b"x")
    read_calls = []

    def fake_run(command, check=False, capture_output=True, text=True, env=None):
        rttm = Path(command[command.index(str(community1.COMMUNITY1_SCRIPT)) + 2])
        assert not rttm.exists()
        rttm.write_text("", encoding="utf-8")
        return type("Result", (), {"returncode": 0, "stderr": "", "stdout": ""})()

    monkeypatch.setattr(community1.subprocess, "run", fake_run)
    monkeypatch.setattr(
        community1, "read_rttm", lambda path: read_calls.append(path) or Annotation(uri="wrong"),
    )
    result = community1.Backend(device="cpu").diarize(audio)
    assert isinstance(result, Annotation)
    assert result.uri == "ev01"
    assert list(result.itertracks()) == []
    assert read_calls == []


def test_community1_missing_rttm_raises(tmp_path, monkeypatch):
    audio = tmp_path / "ev01.wav"
    audio.write_bytes(b"x")

    def fake_run(command, check=False, capture_output=True, text=True, env=None):
        rttm = Path(command[command.index(str(community1.COMMUNITY1_SCRIPT)) + 2])
        assert not rttm.exists()
        return type("Result", (), {"returncode": 0, "stderr": "", "stdout": ""})()

    monkeypatch.setattr(community1.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="no escribió RTTM"):
        community1.Backend(device="cpu").diarize(audio)


def test_evaluate_selects_backend_and_leaderboard_prefix(tmp_path):
    manifest = tmp_path / "manifest.csv"
    audio = tmp_path / "ev01.wav"
    with wave.open(str(audio), "wb") as handle:
        handle.setparams((1, 2, 16000, 16000, "NONE", "not compressed"))
        handle.writeframes(bytes(32000))
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["uri", "ruta_audio", "n_hablantes_esperado"])
        writer.writeheader()
        writer.writerow({"uri": "ev01", "ruta_audio": str(audio), "n_hablantes_esperado": ""})

    class Backend:
        def __init__(self, *, device):
            self.device = device

        def diarize(self, path, **options):
            result = Annotation(uri=path.stem)
            result[Segment(0, 1), "t"] = "A"
            return result

    board = tmp_path / "LEADERBOARD.md"
    out = tmp_path / "community1" / "auto__none"
    evaluate.evaluate(
        manifest, out, backend=Backend, backend_name="community1",
        reference_dir=tmp_path / "refs", leaderboard_path=board,
    )
    data = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert data["backend"] == "community1"
    assert data["modelo"] == "pyannote/speaker-diarization-community-1"
    assert "| community1/auto__none |" in board.read_text(encoding="utf-8")
    assert evaluate.BACKENDS["community1"]["module"] is community1
    assert evaluate.BACKENDS["pyannote31"]["module"].Backend is not None


def test_leaderboard_keeps_existing_pyannote31_figures(tmp_path):
    board = tmp_path / "LEADERBOARD.md"
    board.write_text(
        "# Leaderboard\n\n## Variantes v1\n\n"
        "| variante | versión | dispositivo | cobertura | hablantes por fragmento | acierto | turnos cortos | hablantes mínimos | inestabilidad | degenerados | referencia | DER@0 |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
        "| auto__none | v1 provisional | cuda | 5 | ev01: 5 | 1/1 | 0.1882 | 0 | 0.0328 | 0 | sin referencia | sin referencia |\n",
        encoding="utf-8",
    )
    rows = [{
        "uri": "ev01", "hablantes_estimados": 2, "acierto_hablantes": "sin referencia",
        "hablantes": {"A": {"turnos": 1, "tiempo_s": 1.0}}, "turnos_cortos": 0,
        "hablantes_minimos": 0, "inestabilidad_ganancia": "sin referencia",
        "inestabilidad_silencio": "sin referencia", "degenerada": False,
        "der_025": "sin referencia",
    }]
    aggregate = dict.fromkeys(
        ("der_025", "der_0", "jer", "missed_s", "false_alarm_s", "confusion_s"),
        "sin referencia",
    )
    evaluate.leaderboard(
        board, "community1", rows, aggregate, variant="auto__none",
        device="cpu", run_version="v1 provisional", backend_name="community1",
    )
    text = board.read_text(encoding="utf-8")
    assert "| pyannote31/auto__none |" in text
    assert "| 0.1882 |" in text
    assert "| community1/auto__none |" in text


def test_compare_resolves_backend_prefix(tmp_path):
    root = tmp_path / "results"
    for backend, variant, label in (
        ("pyannote31", "auto__none", "A"),
        ("community1", "auto__loudnorm", "B"),
    ):
        hyp = root / backend / variant / "hypotheses"
        hyp.mkdir(parents=True)
        ann = Annotation(uri="ev01")
        ann[Segment(0, 1), "t"] = label
        write_rttm(ann, hyp / "ev01.rttm")

    left = compare_mod.resolve_variant_path("auto__none", root=root)
    right = compare_mod.resolve_variant_path("community1/auto__loudnorm", root=root)
    assert left == (root / "pyannote31" / "auto__none").resolve()
    assert right == (root / "community1" / "auto__loudnorm").resolve()
    result = compare_mod.compare(left, right)
    assert "ev01" in result
    with pytest.raises(ValueError, match="sale de eval/results|inválido"):
        compare_mod.resolve_variant_path("../fuera", root=root)
    with pytest.raises(ValueError):
        compare_mod.resolve_variant_path("a/b/c", root=root)


def _hf_token_available():
    if os.environ.get("HUGGING_FACE_TOKEN"):
        return True
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    except Exception:
        return False
    return bool(os.environ.get("HUGGING_FACE_TOKEN"))


def _community1_weights_cached():
    hubs = []
    if os.environ.get("HUGGINGFACE_HUB_CACHE"):
        hubs.append(Path(os.environ["HUGGINGFACE_HUB_CACHE"]))
    if os.environ.get("HF_HOME"):
        hubs.append(Path(os.environ["HF_HOME"]) / "hub")
    hubs.append(Path.home() / ".cache" / "huggingface" / "hub")
    marker = "models--pyannote--speaker-diarization-community-1"
    for hub in hubs:
        root = hub / marker
        if root.is_dir() and any(root.rglob("config.yaml")):
            return True
    return False


def _community1_env_ready():
    venv_python = community1.COMMUNITY1_PROJECT / ".venv" / "bin" / "python"
    if not venv_python.is_file() or not community1.COMMUNITY1_SCRIPT.is_file():
        return False
    if not _hf_token_available() or not _community1_weights_cached():
        return False
    try:
        import subprocess

        probe = subprocess.run(
            [
                "uv", "run", "--frozen", "--project", str(community1.COMMUNITY1_PROJECT),
                "python", "-c", "import pyannote.audio, soundfile, torch, numpy",
            ],
            check=False,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
        return probe.returncode == 0
    except OSError:
        return False


def _two_voice_turns(duration_s=30, rate=16000, turn_s=5.0):
    """Dos voces armónicas que se turnan (110 Hz y 220 Hz con vibrato leve)."""
    times = np.arange(rate * duration_s) / rate
    samples = np.zeros_like(times, dtype=np.float32)
    fundamentals = (110.0, 220.0)
    for index, start in enumerate(np.arange(0.0, duration_s, turn_s)):
        end = min(start + turn_s, duration_s)
        mask = (times >= start) & (times < end)
        t = times[mask]
        f0 = fundamentals[index % 2]
        vibrato = f0 * (1.0 + 0.01 * np.sin(2 * np.pi * 5.0 * t))
        voice = (
            0.35 * np.sin(2 * np.pi * vibrato * t)
            + 0.18 * np.sin(2 * np.pi * 2 * vibrato * t)
            + 0.08 * np.sin(2 * np.pi * 3 * vibrato * t)
        )
        # Ventana corta en los bordes del turno para evitar clics.
        fade = min(int(0.02 * rate), max(1, voice.size // 8))
        if voice.size >= 2 * fade:
            window = np.ones(voice.size, dtype=np.float32)
            window[:fade] = np.linspace(0, 1, fade, dtype=np.float32)
            window[-fade:] = np.linspace(1, 0, fade, dtype=np.float32)
            voice = voice * window
        samples[mask] = voice.astype(np.float32)
    return samples, rate


@pytest.mark.skipif(not _community1_env_ready(), reason="community1 no instalado, sin token o sin pesos en caché")
def test_community1_real_synthetic_smoke(tmp_path):
    import time

    samples, rate = _two_voice_turns(duration_s=30, rate=16000, turn_s=5.0)
    audio = tmp_path / "ev99.wav"
    sf.write(str(audio), samples, rate, subtype="PCM_16")
    start = time.perf_counter()
    annotation = community1.diarize(audio, device="cpu")
    elapsed = time.perf_counter() - start
    assert isinstance(annotation, Annotation)
    assert annotation.uri == "ev99"
    assert elapsed > 0
    print(f"community1 smoke CPU 30s audio: {elapsed:.1f}s")
