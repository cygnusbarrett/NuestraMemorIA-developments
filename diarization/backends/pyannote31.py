"""Backend de diarización equivalente al pipeline existente."""

import os
from pathlib import Path

from pyannote.core import Annotation, Segment


class Backend:
    def __init__(self, *, device="cpu", use_auth_token=None):
        from whisperx.diarize import DiarizationPipeline

        token = use_auth_token or os.getenv("HUGGING_FACE_TOKEN")
        self.pipeline = DiarizationPipeline(use_auth_token=token, device=device)

    def diarize(self, audio_path: Path, *, num_speakers=None, min_speakers=None, max_speakers=None):
        segments = self.pipeline(
            str(audio_path), num_speakers=num_speakers,
            min_speakers=min_speakers, max_speakers=max_speakers,
        )
        annotation = Annotation(uri=audio_path.stem)
        for track, row in enumerate(segments.itertuples(index=False)):
            annotation[Segment(row.start, row.end), track] = row.speaker
        return annotation


def diarize(audio_path: Path, *, num_speakers=None, min_speakers=None, max_speakers=None, device="cpu", **kwargs):
    backend = Backend(device=device, **kwargs)
    return backend.diarize(
        audio_path, num_speakers=num_speakers, min_speakers=min_speakers, max_speakers=max_speakers,
    )
