import csv
import wave

import pytest

from diarization.fragments import MANIFEST_FIELDS, SOURCE_FIELDS, run, wav_duration


def write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_fragment_is_idempotent_and_dry_run_does_not_write(tmp_path):
    original = tmp_path / "source.wav"
    with wave.open(str(original), "wb") as audio:
        audio.setparams((1, 2, 16000, 32000, "NONE", "not compressed"))
        audio.writeframes(bytes(64000))
    sources = tmp_path / "sources.csv"
    manifest = tmp_path / "manifest.csv"
    audio_dir = tmp_path / "audio"
    write_csv(sources, SOURCE_FIELDS, [{
        "uri": "ev01", "archivo_original": str(original), "inicio_s": "0.25", "fin_s": "1.5",
    }])
    write_csv(manifest, MANIFEST_FIELDS, [])
    before = manifest.read_bytes()
    assert run(sources, manifest, audio_dir, dry_run=True) == (1, 0)
    assert not audio_dir.exists()
    assert manifest.read_bytes() == before
    assert run(sources, manifest, audio_dir) == (1, 0)
    fragment = audio_dir / "ev01.wav"
    assert wav_duration(fragment) == pytest.approx(1.25, abs=0.01)
    modified = fragment.stat().st_mtime_ns
    assert run(sources, manifest, audio_dir) == (0, 1)
    assert fragment.stat().st_mtime_ns == modified
    with manifest.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["uri"] == "ev01"
    assert float(rows[0]["duracion_s"]) == 1.25


def test_existing_incompatible_fragment_is_not_overwritten(tmp_path):
    original = tmp_path / "source.wav"
    with wave.open(str(original), "wb") as audio:
        audio.setparams((1, 2, 16000, 16000, "NONE", "not compressed"))
        audio.writeframes(bytes(32000))
    sources, manifest, audio_dir = tmp_path / "sources.csv", tmp_path / "manifest.csv", tmp_path / "audio"
    write_csv(sources, SOURCE_FIELDS, [{
        "uri": "ev01", "archivo_original": str(original), "inicio_s": "0", "fin_s": "1",
    }])
    write_csv(manifest, MANIFEST_FIELDS, [])
    audio_dir.mkdir()
    fragment = audio_dir / "ev01.wav"
    fragment.write_bytes(b"invalid")
    with pytest.raises(ValueError):
        run(sources, manifest, audio_dir)
    assert fragment.read_bytes() == b"invalid"
