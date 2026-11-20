#!/usr/bin/env python3
"""Diariza un WAV con pyannote speaker-diarization-community-1 y escribe RTTM."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from pyannote.audio import Pipeline


MODEL_ID = "pyannote/speaker-diarization-community-1"


def _write_rttm(path: Path, uri: str, turns) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as target:
        for turn, speaker in turns:
            duration = turn.end - turn.start
            if duration <= 0:
                continue
            label = str(speaker)
            if not label or any(c.isspace() for c in label):
                raise RuntimeError("Etiqueta de hablante inválida en la salida del pipeline")
            target.write(
                f"SPEAKER {uri} 1 {turn.start:.6f} {duration:.6f} "
                f"<NA> <NA> {label} <NA> <NA>\n"
            )


def diarize_file(
    source: Path,
    destination: Path,
    *,
    num_speakers: int | None,
    min_speakers: int | None,
    max_speakers: int | None,
    device: str,
) -> None:
    token = os.environ.get("HUGGING_FACE_TOKEN")
    if not token:
        raise SystemExit("Falta HUGGING_FACE_TOKEN en el entorno")

    samples, sample_rate = sf.read(str(source), dtype="float32", always_2d=True)
    # soundfile: (muestras, canales) → pyannote: (canales, muestras)
    waveform = torch.from_numpy(np.ascontiguousarray(samples.T))

    pipeline = Pipeline.from_pretrained(MODEL_ID, token=token)
    if pipeline is None:
        raise SystemExit(f"No se pudo cargar el pipeline {MODEL_ID}")

    resolved = "cpu"
    if device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("Se pidió cuda pero torch.cuda.is_available() es falso")
        resolved = "cuda"
    pipeline.to(torch.device(resolved))

    options = {
        key: value
        for key, value in {
            "num_speakers": num_speakers,
            "min_speakers": min_speakers,
            "max_speakers": max_speakers,
        }.items()
        if value is not None
    }
    output = pipeline({"waveform": waveform, "sample_rate": int(sample_rate)}, **options)
    _write_rttm(destination, source.stem, output.speaker_diarization)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("entrada", type=Path)
    parser.add_argument("salida", type=Path)
    parser.add_argument("--num-speakers", type=int)
    parser.add_argument("--min-speakers", type=int)
    parser.add_argument("--max-speakers", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)

    if not args.entrada.is_file():
        raise SystemExit(f"No existe el audio de entrada: {args.entrada}")
    diarize_file(
        args.entrada,
        args.salida,
        num_speakers=args.num_speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        device=args.device,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
