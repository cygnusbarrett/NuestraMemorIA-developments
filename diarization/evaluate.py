"""Ejecuta y describe una evaluación de diarización local."""

import argparse
import csv
import hashlib
import json
import random
import re
import statistics
import time
import wave
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate

from diarization.backends import pyannote31
from diarization.rttm import read_rttm, write_rttm

MODEL = "pyannote/speaker-diarization-3.1"
MISSING = "sin referencia"
FIELDS = [
    "uri", "duracion_s", "hablantes_esperados", "hablantes_estimados", "hablantes",
    "tiempo_s", "rtf", "der_025", "der_0", "missed_s", "false_alarm_s",
    "confusion_s", "referencia_s", "jer",
]


def describe(annotation):
    durations = {}
    for segment, _, label in annotation.itertracks(yield_label=True):
        durations.setdefault(label, []).append(segment.duration)
    return {
        label: {
            "tiempo_s": sum(values), "turnos": len(values),
            "duracion_min_s": min(values), "duracion_mediana_s": statistics.median(values),
            "duracion_max_s": max(values),
        }
        for label, values in sorted(durations.items())
    }


def metrics(reference, hypothesis):
    if reference is None:
        return dict.fromkeys(("der_025", "der_0", "missed_s", "false_alarm_s", "confusion_s", "referencia_s", "jer"), MISSING)
    primary = DiarizationErrorRate(collar=0.25, skip_overlap=False)(reference, hypothesis, detailed=True)
    strict = DiarizationErrorRate(collar=0, skip_overlap=False)(reference, hypothesis)
    jer = JaccardErrorRate(collar=0.25, skip_overlap=False)(reference, hypothesis)
    return {
        "der_025": primary["diarization error rate"], "der_0": strict,
        "missed_s": primary["missed detection"], "false_alarm_s": primary["false alarm"],
        "confusion_s": primary["confusion"], "referencia_s": primary["total"], "jer": jer,
    }


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def wav_duration(path):
    with wave.open(str(path), "rb") as source:
        return source.getnframes() / source.getframerate()


def leaderboard(path, config, rows, aggregate):
    headers = ["configuración", "versión", "cobertura", "referencia", "hablantes", "tiempo_habla_s", "turnos", "rtf",
               "DER@0.25", "DER@0", "missed_s", "false_alarm_s", "confusion_s", "JER"]
    marker = f"| {config} |"
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else [
        "# Leaderboard", "", "| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines = [line for line in lines if not line.startswith(marker)]
    speech = sum(sum(v["tiempo_s"] for v in row["hablantes"].values()) for row in rows)
    turns = sum(sum(v["turnos"] for v in row["hablantes"].values()) for row in rows)
    rtf = sum(row["tiempo_s"] for row in rows) / sum(row["duracion_s"] for row in rows)
    reference_count = sum(row["der_025"] != MISSING for row in rows)
    status = f"{reference_count}/{len(rows)} con referencia" if reference_count else MISSING
    values = [aggregate[key] for key in ("der_025", "der_0", "missed_s", "false_alarm_s", "confusion_s", "jer")]
    metric_cells = " | ".join(f"{value:.4f}" if isinstance(value, float) else str(value) for value in values)
    lines.append(f"| {config} | v0 provisional | {len(rows)} fragmento(s) | {status} | "
                 f"{sum(row['hablantes_estimados'] for row in rows)} | {speech:.3f} | {turns} | {rtf:.4f} | {metric_cells} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate(manifest, out, *, backend, device="cpu", seed=0, reference_dir=Path("eval/reference"), leaderboard_path=Path("eval/results/LEADERBOARD.md")):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    with manifest.open(encoding="utf-8", newline="") as stream:
        entries = list(csv.DictReader(stream))
    if not entries:
        raise ValueError("El manifiesto está vacío")
    uris = [entry["uri"] for entry in entries]
    if any(not re.fullmatch(r"ev\d{2,}", uri) for uri in uris) or len(uris) != len(set(uris)):
        raise ValueError("URI inválido o repetido")
    out.mkdir(parents=True, exist_ok=True)
    hypotheses = out / "hypotheses"
    hypotheses.mkdir(exist_ok=True)
    engine = backend(device=device)
    rows, hashes = [], {}
    primary = DiarizationErrorRate(collar=0.25, skip_overlap=False)
    strict = DiarizationErrorRate(collar=0, skip_overlap=False)
    jer = JaccardErrorRate(collar=0.25, skip_overlap=False)
    reference_count = 0
    for entry in entries:
        uri = entry["uri"]
        audio = Path(entry["ruta_audio"])
        duration = wav_duration(audio)
        if duration <= 0:
            raise ValueError(f"Duración inválida para {uri}")
        hashes[uri] = digest(audio)
        start = time.perf_counter()
        hypothesis = engine.diarize(audio)
        elapsed = time.perf_counter() - start
        hypothesis.uri = uri
        write_rttm(hypothesis, hypotheses / f"{uri}.rttm")
        ref_path = reference_dir / f"{uri}.rttm"
        reference = read_rttm(ref_path) if ref_path.exists() else None
        if reference is not None and reference.uri != uri:
            raise ValueError(f"URI de referencia incompatible para {uri}")
        if reference is not None:
            primary(reference, hypothesis)
            strict(reference, hypothesis)
            jer(reference, hypothesis)
            reference_count += 1
        row = {
            "uri": uri, "duracion_s": duration, "hablantes_esperados": entry.get("n_hablantes_esperado", ""),
            "hablantes_estimados": len(hypothesis.labels()), "hablantes": describe(hypothesis),
            "tiempo_s": elapsed, "rtf": elapsed / duration,
        }
        row.update(metrics(reference, hypothesis))
        rows.append(row)
    with (out / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({**row, "hablantes": json.dumps(row["hablantes"], ensure_ascii=False, sort_keys=True)} for row in rows)
    summary = ["# Evaluación v0 provisional", "", f"Cobertura: {len(rows)} fragmento(s).", ""]
    for row in rows:
        summary.append(f"## {row['uri']} — {row['hablantes_estimados']} hablante(s), RTF {row['rtf']:.4f}")
        summary.append(f"Referencia: {'disponible' if row['der_025'] != MISSING else MISSING}.")
        if row["der_025"] != MISSING:
            summary.append(f"DER@0.25: {row['der_025']:.4f}; DER@0: {row['der_0']:.4f}; JER: {row['jer']:.4f}.")
        for label, stats in row["hablantes"].items():
            summary.append(f"- {label}: {stats['tiempo_s']:.3f} s, {stats['turnos']} turnos; "
                           f"duración mínima/mediana/máxima: {stats['duracion_min_s']:.3f}/"
                           f"{stats['duracion_mediana_s']:.3f}/{stats['duracion_max_s']:.3f} s.")
        summary.append("")
    (out / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    run_manifest = {
        "modelo": MODEL, "pyannote_audio": version("pyannote.audio"), "whisperx": version("whisperx"),
        "backend": "pyannote31", "device": device, "seed": seed,
        "parametros": {"num_speakers": None, "min_speakers": None, "max_speakers": None, "preprocesamiento": "ninguno"},
        "audio_sha256": hashes,
    }
    (out / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    config = "pyannote31 · sin preprocesamiento · auto"
    if reference_count:
        aggregate = {"der_025": abs(primary), "der_0": abs(strict), "jer": abs(jer),
                     "missed_s": primary.accumulated_["missed detection"],
                     "false_alarm_s": primary.accumulated_["false alarm"],
                     "confusion_s": primary.accumulated_["confusion"]}
    else:
        aggregate = dict.fromkeys(("der_025", "der_0", "jer", "missed_s", "false_alarm_s", "confusion_s"), MISSING)
    leaderboard(leaderboard_path, config, rows, aggregate)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["pyannote31"], required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    evaluate(args.manifest, args.out, backend=pyannote31.Backend, device=args.device, seed=args.seed)


if __name__ == "__main__":
    main()
