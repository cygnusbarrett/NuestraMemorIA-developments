import sys
from types import ModuleType, SimpleNamespace

from pyannote.core import Annotation

from diarization.backends.pyannote31 import diarize


def test_backend_forwards_settings_and_preserves_overlap(tmp_path, monkeypatch):
    calls = {}

    class Pipeline:
        def __init__(self, **kwargs):
            calls["init"] = kwargs

        def __call__(self, audio, **kwargs):
            calls["audio"] = audio
            calls["options"] = kwargs
            return SimpleNamespace(itertuples=lambda **_: iter([
                SimpleNamespace(start=0.0, end=1.0, speaker="A"),
                SimpleNamespace(start=0.0, end=1.0, speaker="B"),
            ]))

    module = ModuleType("whisperx.diarize")
    module.DiarizationPipeline = Pipeline
    monkeypatch.setitem(sys.modules, "whisperx.diarize", module)
    monkeypatch.setenv("HUGGING_FACE_TOKEN", "synthetic-token")
    audio = tmp_path / "ev01.wav"
    result = diarize(audio, device="cpu")
    assert isinstance(result, Annotation)
    assert result.uri == "ev01"
    assert set(result.labels()) == {"A", "B"}
    assert calls["init"] == {"use_auth_token": "synthetic-token", "device": "cpu"}
    assert calls["audio"] == str(audio)
    assert calls["options"] == {"num_speakers": None, "min_speakers": None, "max_speakers": None}
