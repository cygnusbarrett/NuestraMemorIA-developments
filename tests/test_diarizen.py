"""Tests del backend diarizen (subproceso aislado) y su cableado en evaluate."""

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

from diarization import evaluate
from diarization.backends import diarizen


def _write_rttm_text(path: Path, uri: str = "ev01") -> None:
    path.write_text(
        f"SPEAKER {uri} 1 0.000000 1.000000 <NA> <NA> SPEAKER_00 <NA> <NA>\n"
        f"SPEAKER {uri} 1 0.500000 1.000000 <NA> <NA> SPEAKER_01 <NA> <NA>\n",
        encoding="utf-8",
    )


def test_resolve_speaker_bounds_maps_num_speakers():
    assert diarizen.resolve_speaker_bounds(3, None, None) == (3, 3)
    assert diarizen.resolve_speaker_bounds(None, 1, 4) == (1, 4)
    assert diarizen.resolve_speaker_bounds(None, None, None) == (None, None)
    with pytest.raises(ValueError, match="no se combina"):
        diarizen.resolve_speaker_bounds(2, 1, None)
    with pytest.raises(ValueError, match="no se combina"):
        diarizen.resolve_speaker_bounds(2, None, 5)


def test_diarizen_command_shape(tmp_path, monkeypatch):
    audio = tmp_path / "ev01.wav"
    with wave.open(str(audio), "wb") as handle:
        handle.setparams((1, 2, 16000, 16000, "NONE", "not compressed"))
        handle.writeframes(bytes(32000))
    seen = []

    def fake_run(command, check=False, capture_output=True, text=True, env=None):
        seen.append(list(command))
        assert env is not None
        rttm = Path(command[command.index(str(diarizen.DIARIZEN_SCRIPT)) + 2])
        assert command[command.index(str(diarizen.DIARIZEN_SCRIPT)) + 1] == str(audio)
        assert not rttm.exists()
        _write_rttm_text(rttm, uri="ev01")
        return type("Result", (), {"returncode": 0, "stderr": "", "stdout": ""})()

    monkeypatch.setattr(diarizen.subprocess, "run", fake_run)
    result = diarizen.diarize(audio, device="cpu", min_speakers=1, max_speakers=4)
    assert isinstance(result, Annotation)
    assert result.uri == "ev01"
    assert set(result.labels()) == {"SPEAKER_00", "SPEAKER_01"}
    command = seen[0]
    assert command[:3] == ["uv", "run", "--no-sync"]
    assert command[command.index("--project") + 1] == str(diarizen.DIARIZEN_PROJECT)
    assert str(diarizen.DIARIZEN_SCRIPT) in command
    assert command[command.index("--min-speakers") + 1] == "1"
    assert command[command.index("--max-speakers") + 1] == "4"

    seen.clear()
    diarizen.diarize(audio, device="cpu", num_speakers=2)
    command = seen[0]
    assert command[command.index("--num-speakers") + 1] == "2"
    assert "--min-speakers" not in command
    assert "--max-speakers" not in command


def test_diarizen_empty_rttm_returns_empty_annotation(tmp_path, monkeypatch):
    audio = tmp_path / "ev01.wav"
    audio.write_bytes(b"x")
    read_calls = []

    def fake_run(command, check=False, capture_output=True, text=True, env=None):
        rttm = Path(command[command.index(str(diarizen.DIARIZEN_SCRIPT)) + 2])
        assert not rttm.exists()
        rttm.write_text("", encoding="utf-8")
        return type("Result", (), {"returncode": 0, "stderr": "", "stdout": ""})()

    monkeypatch.setattr(diarizen.subprocess, "run", fake_run)
    monkeypatch.setattr(
        diarizen, "read_rttm", lambda path: read_calls.append(path) or Annotation(uri="wrong"),
    )
    result = diarizen.Backend(device="cpu").diarize(audio)
    assert isinstance(result, Annotation)
    assert result.uri == "ev01"
    assert list(result.itertracks()) == []
    assert read_calls == []


def test_diarizen_missing_rttm_raises(tmp_path, monkeypatch):
    audio = tmp_path / "ev01.wav"
    audio.write_bytes(b"x")

    def fake_run(command, check=False, capture_output=True, text=True, env=None):
        return type("Result", (), {"returncode": 0, "stderr": "", "stdout": ""})()

    monkeypatch.setattr(diarizen.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="no escribió RTTM"):
        diarizen.Backend(device="cpu").diarize(audio)


def test_evaluate_selects_diarizen_backend(tmp_path):
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
    out = tmp_path / "diarizen" / "auto__none"
    evaluate.evaluate(
        manifest, out, backend=Backend, backend_name="diarizen",
        reference_dir=tmp_path / "refs", leaderboard_path=board,
    )
    data = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert data["backend"] == "diarizen"
    assert data["modelo"] == "BUT-FIT/diarizen-wavlm-large-s80-md-v2"
    assert data["diarizen"] == diarizen.DIARIZEN_REV
    assert data["diarizen_model_revision"] == diarizen.MODEL_REVISION
    assert data["diarizen_embedding_revision"] == diarizen.EMBEDDING_REVISION
    assert "| diarizen/auto__none |" in board.read_text(encoding="utf-8")
    assert evaluate.BACKENDS["diarizen"]["module"] is diarizen


def _hf_token_available():
    if os.environ.get("HUGGING_FACE_TOKEN"):
        return True
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    except Exception:
        return False
    return bool(os.environ.get("HUGGING_FACE_TOKEN"))


def _diarizen_weights_cached():
    hubs = []
    if os.environ.get("HUGGINGFACE_HUB_CACHE"):
        hubs.append(Path(os.environ["HUGGINGFACE_HUB_CACHE"]))
    if os.environ.get("HF_HOME"):
        hubs.append(Path(os.environ["HF_HOME"]) / "hub")
    hubs.append(Path.home() / ".cache" / "huggingface" / "hub")
    markers = (
        "models--BUT-FIT--diarizen-wavlm-large-s80-md-v2",
        "models--pyannote--wespeaker-voxceleb-resnet34-LM",
    )
    found = {marker: False for marker in markers}
    for hub in hubs:
        for marker in markers:
            root = hub / marker
            if root.is_dir() and any(root.iterdir()):
                found[marker] = True
    return all(found.values())


def _diarizen_env_ready():
    venv_python = diarizen.DIARIZEN_PROJECT / ".venv" / "bin" / "python"
    if not venv_python.is_file() or not diarizen.DIARIZEN_SCRIPT.is_file():
        return False
    if not _diarizen_weights_cached():
        return False
    try:
        import subprocess

        probe = subprocess.run(
            [
                "uv", "run", "--no-sync", "--project", str(diarizen.DIARIZEN_PROJECT),
                "python", "-c", "from diarizen.pipelines.inference import DiariZenPipeline",
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
        fade = min(int(0.02 * rate), max(1, voice.size // 8))
        if voice.size >= 2 * fade:
            window = np.ones(voice.size, dtype=np.float32)
            window[:fade] = np.linspace(0, 1, fade, dtype=np.float32)
            window[-fade:] = np.linspace(1, 0, fade, dtype=np.float32)
            voice = voice * window
        samples[mask] = voice.astype(np.float32)
    return samples, rate


@pytest.mark.skipif(not _diarizen_env_ready(), reason="diarizen no instalado o sin pesos en caché")
def test_diarizen_real_synthetic_smoke(tmp_path):
    import time

    samples, rate = _two_voice_turns(duration_s=30, rate=16000, turn_s=5.0)
    audio = tmp_path / "ev99.wav"
    sf.write(str(audio), samples, rate, subtype="PCM_16")
    # Carga dotenv por si el embedding de pyannote pide token.
    if not os.environ.get("HUGGING_FACE_TOKEN"):
        _hf_token_available()
    start = time.perf_counter()
    annotation = diarizen.diarize(audio, device="cpu")
    elapsed = time.perf_counter() - start
    assert isinstance(annotation, Annotation)
    assert annotation.uri == "ev99"
    assert elapsed > 0
    print(f"diarizen smoke CPU 30s audio: {elapsed:.1f}s")
