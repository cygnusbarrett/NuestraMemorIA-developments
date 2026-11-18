import csv
import json
import sys
import wave

import pytest
from pyannote.core import Annotation, Segment

from diarization import evaluate
from diarization.rttm import write_rttm


def annotation(uri, end):
    result = Annotation(uri=uri)
    result[Segment(0, end), "track"] = "A"
    return result


def test_known_der():
    result = evaluate.metrics(annotation("ev01", 4), annotation("ev01", 3))
    assert result["der_0"] == pytest.approx(0.25, abs=0.001)
    assert result["missed_s"] > 0
    assert result["jer"] > 0


def test_no_reference_and_single_pipeline_load(tmp_path):
    manifest = tmp_path / "manifest.csv"
    audios = []
    for uri in ("ev01", "ev02"):
        path = tmp_path / f"{uri}.wav"
        with wave.open(str(path), "wb") as audio:
            audio.setparams((1, 2, 16000, 64000, "NONE", "not compressed"))
            audio.writeframes(bytes(128000))
        audios.append(path)
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["uri", "ruta_audio", "n_hablantes_esperado"])
        writer.writeheader()
        writer.writerows({"uri": p.stem, "ruta_audio": str(p), "n_hablantes_esperado": ""} for p in audios)
    calls = []

    class Backend:
        def __init__(self, *, device):
            calls.append(("load", device))

        def diarize(self, path):
            calls.append(("run", path.stem))
            result = annotation(path.stem, 1)
            result[Segment(2, 4), "second"] = "A"
            return result

    out = tmp_path / "out"
    leaderboard = tmp_path / "LEADERBOARD.md"
    rows = evaluate.evaluate(manifest, out, backend=Backend, reference_dir=tmp_path / "refs", leaderboard_path=leaderboard)
    assert calls == [("load", "cpu"), ("run", "ev01"), ("run", "ev02")]
    assert rows[0]["der_025"] == rows[0]["jer"] == "sin referencia"
    assert rows[0]["hablantes"]["A"] == {
        "tiempo_s": 3, "turnos": 2, "duracion_min_s": 1,
        "duracion_mediana_s": 1.5, "duracion_max_s": 2,
    }
    assert json.loads((out / "run_manifest.json").read_text())["seed"] == 0
    assert (out / "hypotheses" / "ev01.rttm").exists()
    assert "sin referencia" in leaderboard.read_text()
    evaluate.evaluate(manifest, out, backend=Backend, reference_dir=tmp_path / "refs", leaderboard_path=leaderboard)
    assert leaderboard.read_text().count("| pyannote31 · sin preprocesamiento · auto |") == 1


def test_reference_generates_metrics(tmp_path):
    manifest = tmp_path / "manifest.csv"
    audio_path = tmp_path / "ev01.wav"
    with wave.open(str(audio_path), "wb") as audio:
        audio.setparams((1, 2, 16000, 64000, "NONE", "not compressed"))
        audio.writeframes(bytes(128000))
    manifest.write_text(f"uri,ruta_audio,n_hablantes_esperado\nev01,{audio_path},1\n")
    refs = tmp_path / "refs"
    refs.mkdir()
    write_rttm(annotation("ev01", 4), refs / "ev01.rttm")

    class Backend:
        def __init__(self, *, device):
            pass

        def diarize(self, path):
            return annotation(path.stem, 3)

    board = tmp_path / "LEADERBOARD.md"
    rows = evaluate.evaluate(manifest, tmp_path / "out", backend=Backend, reference_dir=refs, leaderboard_path=board)
    assert rows[0]["der_0"] == pytest.approx(0.25, abs=0.001)
    assert "0.2500" in board.read_text()


def test_cli_loads_dotenv_before_evaluation(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(evaluate, "load_dotenv", lambda path: calls.append("dotenv"))
    monkeypatch.setattr(evaluate, "evaluate", lambda *args, **kwargs: calls.append("evaluate"))
    monkeypatch.setattr(sys, "argv", ["evaluate", "--backend", "pyannote31", "--manifest", "m.csv", "--out", str(tmp_path)])
    evaluate.main()
    assert calls == ["dotenv", "evaluate"]
