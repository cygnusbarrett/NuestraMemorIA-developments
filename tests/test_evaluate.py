import csv
import json
import sys
import wave
from pathlib import Path

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
    manifest_json = json.loads((out / "run_manifest.json").read_text())
    assert manifest_json["seed"] == 0
    assert manifest_json["backend"] == "pyannote31"
    assert (out / "hypotheses" / "ev01.rttm").exists()
    assert "sin referencia" in leaderboard.read_text()
    evaluate.evaluate(manifest, out, backend=Backend, reference_dir=tmp_path / "refs", leaderboard_path=leaderboard)
    assert leaderboard.read_text().count("| pyannote31/auto__none |") == 1


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


def test_variants_and_validation():
    assert evaluate.variant_id() == "auto__none"
    assert evaluate.variant_id(oracle=True) == "oraculo__none"
    assert evaluate.variant_id(min_speakers=2, max_speakers=8) == "min2-max8__none"
    assert evaluate.variant_id(num_speakers=3) == "num3__none"
    assert evaluate.variant_id(preprocess="loudnorm") == "auto__loudnorm"
    assert evaluate.variant_id(preprocess="highpass80+loudnorm") == "auto__highpass80+loudnorm"
    with pytest.raises(ValueError):
        evaluate.variant_id(num_speakers=2, oracle=True)
    with pytest.raises(ValueError):
        evaluate.variant_id(min_speakers=5, max_speakers=2)
    with pytest.raises(ValueError):
        evaluate.variant_id(preprocess="desconocido")
    with pytest.raises(ValueError):
        evaluate.variant_id(preprocess="loudnorm+fantasma")


def test_fragmentation_and_degenerate():
    result = Annotation(uri="ev01")
    result[Segment(0, 10), "a"] = "A"
    result[Segment(11, 11.1), "b"] = "B"
    stats = evaluate.fragmentation(result)
    assert stats["proporcion_turnos_cortos"] == 0.5
    assert stats["hablantes_minimos"] == 1
    assert not stats["degenerada"]
    assert evaluate.fragmentation(annotation("ev01", 1))["degenerada"]


def test_stability_removes_silence(tmp_path):
    path = tmp_path / "ev01.wav"
    with wave.open(str(path), "wb") as audio:
        audio.setparams((1, 2, 16000, 16000, "NONE", "not compressed"))
        audio.writeframes(bytes(32000))

    class Backend:
        def diarize(self, path, **kwargs):
            with wave.open(str(path), "rb") as audio:
                duration = audio.getnframes() / audio.getframerate()
            result = Annotation(uri="ev01")
            offset = 0.05 if duration > 1 else 0
            result[Segment(offset, 1 + offset), "a"] = "A"
            return result

    gain, silence = evaluate.stability(Backend(), path, annotation("ev01", 1), 1, {})
    assert gain == pytest.approx(0)
    assert silence == pytest.approx(0)


def test_compare_known_disagreement():
    from diarization.compare import compare_annotations

    left = annotation("ev01", 4)
    right = annotation("ev01", 3)
    result = compare_annotations(left, right)
    assert result["discrepancia_der"] == pytest.approx(0.25)
    assert result["tramos"] == [{"inicio": 3, "fin": 4, "a": ["A"], "b": []}]


def test_oracle_passes_known_counts_and_auto_for_unknown(tmp_path):
    manifest = tmp_path / "manifest.csv"
    for uri in ("ev01", "ev02"):
        with wave.open(str(tmp_path / f"{uri}.wav"), "wb") as audio:
            audio.setparams((1, 2, 16000, 16000, "NONE", "not compressed"))
            audio.writeframes(bytes(32000))
    manifest.write_text(f"uri,ruta_audio,n_hablantes_esperado\nev01,{tmp_path / 'ev01.wav'},2\nev02,{tmp_path / 'ev02.wav'},\n")
    calls = []

    class Backend:
        def __init__(self, *, device):
            pass

        def diarize(self, path, **options):
            calls.append(options)
            return annotation(path.stem, 1)

    rows = evaluate.evaluate(manifest, tmp_path / "out", backend=Backend, oracle=True,
                             reference_dir=tmp_path / "refs", leaderboard_path=tmp_path / "board")
    assert calls == [{"num_speakers": 2}, {}]
    assert [row["acierto_hablantes"] for row in rows] == ["forzado", "sin referencia"]
    with (tmp_path / "out" / "metrics.csv").open() as stream:
        assert [row["acierto_hablantes"] for row in csv.DictReader(stream)] == ["forzado", "sin referencia"]
    assert "Acierto de hablantes: forzado." in (tmp_path / "out" / "summary.md").read_text()
    assert "| forzado |" in (tmp_path / "board").read_text()
    assert json.loads((tmp_path / "out" / "run_manifest.json").read_text())["parametros"]["variante"] == "oraculo__none"


def test_forced_accuracy_does_not_outweigh_fragmentation(tmp_path):
    audio_path = tmp_path / "ev01.wav"
    with wave.open(str(audio_path), "wb") as audio:
        audio.setparams((1, 2, 16000, 16000, "NONE", "not compressed"))
        audio.writeframes(bytes(32000))
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(f"uri,ruta_audio,n_hablantes_esperado\nev01,{audio_path},2\n")

    class Backend:
        def __init__(self, *, device):
            pass

        def diarize(self, path, **options):
            if not options:
                return annotation(path.stem, 1)
            result = Annotation(uri=path.stem)
            for i in range(10):
                result[Segment(i / 10, (i + 1) / 10), str(i)] = str(i % 2)
            return result

    board = tmp_path / "board"
    common = {"backend": Backend, "reference_dir": tmp_path / "refs", "leaderboard_path": board}
    evaluate.evaluate(manifest, tmp_path / "oracle", oracle=True, **common)
    evaluate.evaluate(manifest, tmp_path / "auto", **common)
    rows = evaluate.evaluate(manifest, tmp_path / "fixed", num_speakers=2, **common)
    assert rows[0]["acierto_hablantes"] == "forzado"
    assert "forzado" in (tmp_path / "fixed" / "metrics.csv").read_text()
    assert "Acierto de hablantes: forzado." in (tmp_path / "fixed" / "summary.md").read_text()
    lines = board.read_text().splitlines()
    auto = next(i for i, line in enumerate(lines) if line.startswith("| pyannote31/auto__none |"))
    oracle = next(i for i, line in enumerate(lines) if line.startswith("| pyannote31/oraculo__none |"))
    assert "| 0/1 |" in lines[auto]
    assert "| forzado |" in lines[oracle]
    assert auto < oracle


def test_compare_variants_uses_matching_labels(tmp_path):
    from diarization.compare import compare

    for name in ("a", "b"):
        (tmp_path / name / "hypotheses").mkdir(parents=True)
    left = annotation("ev01", 4)
    right = Annotation(uri="ev01")
    right[Segment(0, 3), "x"] = "different_name"
    write_rttm(left, tmp_path / "a" / "hypotheses" / "ev01.rttm")
    write_rttm(right, tmp_path / "b" / "hypotheses" / "ev01.rttm")
    result = compare(tmp_path / "a", tmp_path / "b")["ev01"]
    assert result["discrepancia_der"] == pytest.approx(0.25)
    assert result["tramos"][0]["inicio"] == 3


def test_evaluate_preprocess_loudnorm_uses_cache_path(tmp_path):
    import numpy as np
    import soundfile as sf

    audio_path = tmp_path / "ev01.wav"
    rate = 16000
    times = np.arange(rate) / rate
    samples = (0.2 * np.sin(2 * np.pi * 180.0 * times)).astype(np.float32)
    sf.write(str(audio_path), samples, rate, subtype="PCM_16")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        f"uri,ruta_audio,n_hablantes_esperado\nev01,{audio_path},\n", encoding="utf-8",
    )
    cache_root = tmp_path / "cache"
    seen = []

    class Backend:
        def __init__(self, *, device):
            pass

        def diarize(self, path, **options):
            seen.append(Path(path).resolve())
            return annotation(Path(path).stem, 1)

    evaluate.evaluate(
        manifest, tmp_path / "out", backend=Backend, preprocess="loudnorm",
        reference_dir=tmp_path / "refs", leaderboard_path=tmp_path / "board",
        cache_root=cache_root,
    )
    cached = (cache_root / "loudnorm" / "ev01.wav").resolve()
    assert cached.is_file()
    assert seen == [cached]
    manifest_json = json.loads((tmp_path / "out" / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest_json["parametros"]["variante"] == "auto__loudnorm"
    assert "ev01" in manifest_json["audio_sha256"]
    assert "ev01" in manifest_json["audio_procesado_sha256"]
    assert "ev01" in manifest_json["desfase_compensado_muestras"]
    assert isinstance(manifest_json["desfase_compensado_muestras"]["ev01"], int)
