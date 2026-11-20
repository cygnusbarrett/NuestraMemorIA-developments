"""Backend de diarización vía entorno aislado DiariZen.

``num_speakers=N`` se traduce a ``min_speakers = max_speakers = N`` en
``run.py``. No se combina con min/max explícitos.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

from pyannote.core import Annotation

from diarization.rttm import read_rttm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DIARIZEN_PROJECT = REPO_ROOT / "envs" / "diarizen"
DIARIZEN_SCRIPT = DIARIZEN_PROJECT / "run.py"
DIARIZEN_REV = "844f5555b0a98acd0931511fc641a8c5b8ba92c7"
MODEL_REVISION = "027b3e221b9d81dce2be794084f1dbd3ba2da403"
EMBEDDING_REVISION = "837717ddb9ff5507820346191109dc79c958d614"


def resolve_speaker_bounds(num_speakers, min_speakers, max_speakers):
    """Traduce num_speakers a min=max=N; error si se combina con min/max.

    Misma regla que envs/diarizen/run.py (mantener alineadas).
    """
    if num_speakers is not None and (min_speakers is not None or max_speakers is not None):
        raise ValueError(
            "num_speakers no se combina con min_speakers ni max_speakers"
        )
    if num_speakers is not None:
        return num_speakers, num_speakers
    return min_speakers, max_speakers


def diarizen_command(
    audio_path: Path,
    rttm_path: Path,
    *,
    device: str = "cpu",
    num_speakers=None,
    min_speakers=None,
    max_speakers=None,
):
    # --no-sync: con deps git, --frozen puede bloquearse esperando red.
    command = [
        "uv", "run", "--no-sync", "--project", str(DIARIZEN_PROJECT),
        "python", str(DIARIZEN_SCRIPT),
        str(audio_path), str(rttm_path),
        "--device", device,
    ]
    if num_speakers is not None:
        command.extend(["--num-speakers", str(num_speakers)])
    if min_speakers is not None:
        command.extend(["--min-speakers", str(min_speakers)])
    if max_speakers is not None:
        command.extend(["--max-speakers", str(max_speakers)])
    return command


class Backend:
    def __init__(self, *, device="cpu"):
        self.device = device

    def diarize(self, audio_path: Path, *, num_speakers=None, min_speakers=None, max_speakers=None):
        audio_path = Path(audio_path)
        with tempfile.TemporaryDirectory(prefix=".diarizen-") as directory:
            temporary = Path(directory) / "out.rttm"
            command = diarizen_command(
                audio_path,
                temporary,
                device=self.device,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                env=os.environ.copy(),
            )
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout or "").strip()
                message = f"Backend diarizen falló (código {completed.returncode})"
                if detail:
                    message = f"{message}: {detail.splitlines()[-1]}"
                raise RuntimeError(message)
            if not temporary.is_file():
                raise RuntimeError("Backend diarizen no escribió RTTM")
            if temporary.stat().st_size == 0:
                return Annotation(uri=audio_path.stem)
            return read_rttm(temporary)


def diarize(audio_path: Path, *, num_speakers=None, min_speakers=None, max_speakers=None, device="cpu", **kwargs):
    del kwargs
    backend = Backend(device=device)
    return backend.diarize(
        audio_path,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )
