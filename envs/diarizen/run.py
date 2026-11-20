#!/usr/bin/env python3
"""Diariza un WAV con DiariZen (BUT-FIT) y escribe RTTM."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import soundfile as sf
import torch
from huggingface_hub import hf_hub_download, snapshot_download

# PyTorch >= 2.6 usa weights_only=True por defecto; el fork pyannote 3.1 de
# DiariZen / lightning pueden pasar True y fallan al cargar el embedding.
_torch_load = torch.load


def _torch_load_compat(*args, **kwargs):
    kwargs["weights_only"] = False
    return _torch_load(*args, **kwargs)


torch.load = _torch_load_compat

from diarizen.pipelines.inference import DiariZenPipeline  # noqa: E402

MODEL_ID = "BUT-FIT/diarizen-wavlm-large-s80-md-v2"
MODEL_REVISION = "027b3e221b9d81dce2be794084f1dbd3ba2da403"
EMBEDDING_REPO = "pyannote/wespeaker-voxceleb-resnet34-LM"
EMBEDDING_FILE = "pytorch_model.bin"
EMBEDDING_REVISION = "837717ddb9ff5507820346191109dc79c958d614"


def resolve_speaker_bounds(num_speakers, min_speakers, max_speakers):
    """Traduce num_speakers a min=max=N; error si se combina con min/max."""
    if num_speakers is not None and (min_speakers is not None or max_speakers is not None):
        raise ValueError(
            "num_speakers no se combina con min_speakers ni max_speakers"
        )
    if num_speakers is not None:
        return num_speakers, num_speakers
    return min_speakers, max_speakers


def _write_rttm_from_annotation(path: Path, uri: str, annotation) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as target:
        for turn, _, speaker in annotation.itertracks(yield_label=True):
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


def _load_pipeline(rttm_out_dir: str) -> DiariZenPipeline:
    diarizen_hub = Path(
        snapshot_download(MODEL_ID, revision=MODEL_REVISION)
    ).expanduser().absolute()
    embedding_model = hf_hub_download(
        EMBEDDING_REPO,
        EMBEDDING_FILE,
        revision=EMBEDDING_REVISION,
    )
    return DiariZenPipeline(
        diarizen_hub=diarizen_hub,
        embedding_model=embedding_model,
        rttm_out_dir=rttm_out_dir,
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
    try:
        min_speakers, max_speakers = resolve_speaker_bounds(
            num_speakers, min_speakers, max_speakers,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    # Valida el WAV con soundfile; el pipeline carga por ruta (torchaudio).
    samples, sample_rate = sf.read(str(source), dtype="float32", always_2d=True)
    if samples.size == 0 or sample_rate <= 0:
        raise SystemExit(f"Audio inválido: {source}")

    resolved = "cpu"
    if device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("Se pidió cuda pero torch.cuda.is_available() es falso")
        resolved = "cuda"

    with tempfile.TemporaryDirectory(prefix=".diarizen-rttm-") as directory:
        # Si la librería escribe RTTM propio, queda solo en este temporal.
        pipeline = _load_pipeline(directory)
        if min_speakers is not None:
            pipeline.min_speakers = min_speakers
        if max_speakers is not None:
            pipeline.max_speakers = max_speakers

        target = torch.device(resolved)
        if hasattr(pipeline, "to"):
            pipeline.to(target)
        else:
            pipeline.device = target

        # Con CUDA_VISIBLE_DEVICES el índice local es 0; no lo reescribimos.
        result = pipeline(str(source), sess_name=source.stem)

    _write_rttm_from_annotation(destination, source.stem, result)


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
