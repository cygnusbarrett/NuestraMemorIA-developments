"""Backend de diarización vía entorno aislado community-1."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

from pyannote.core import Annotation

from diarization.rttm import read_rttm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
COMMUNITY1_PROJECT = REPO_ROOT / "envs" / "community1"
COMMUNITY1_SCRIPT = COMMUNITY1_PROJECT / "run.py"


def community1_command(
    audio_path: Path,
    rttm_path: Path,
    *,
    device: str = "cpu",
    num_speakers=None,
    min_speakers=None,
    max_speakers=None,
):
    command = [
        "uv", "run", "--frozen", "--project", str(COMMUNITY1_PROJECT),
        "python", str(COMMUNITY1_SCRIPT),
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


def isolated_pyannote_version():
    """Versión de pyannote.audio en envs/community1, o None si no se puede leer."""
    try:
        completed = subprocess.run(
            [
                "uv", "run", "--frozen", "--project", str(COMMUNITY1_PROJECT),
                "python", "-c", "import importlib.metadata as m; print(m.version('pyannote.audio'))",
            ],
            check=False,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    value = (completed.stdout or "").strip()
    return value or None


class Backend:
    def __init__(self, *, device="cpu"):
        self.device = device

    def diarize(self, audio_path: Path, *, num_speakers=None, min_speakers=None, max_speakers=None):
        audio_path = Path(audio_path)
        with tempfile.TemporaryDirectory(prefix=".community1-") as directory:
            temporary = Path(directory) / "out.rttm"
            command = community1_command(
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
                message = f"Backend community1 falló (código {completed.returncode})"
                if detail:
                    # Solo la última línea: evita volcar traceback o el entorno.
                    message = f"{message}: {detail.splitlines()[-1]}"
                raise RuntimeError(message)
            if not temporary.is_file():
                raise RuntimeError("Backend community1 no escribió RTTM")
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
