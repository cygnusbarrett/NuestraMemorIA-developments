"""Ejecuta y describe una evaluación de diarización local."""

import argparse
import csv
import hashlib
import json
import random
import re
import statistics
import time
import tempfile
import wave
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
import soundfile as sf
from pyannote.core import Annotation, Segment
from dotenv import load_dotenv
from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate

from diarization.backends import community1, pyannote31
from diarization.compare import compare_annotations
from diarization import preprocess as preproc
from diarization.rttm import read_rttm, write_rttm

BACKENDS = {
    "pyannote31": {
        "module": pyannote31,
        "modelo": "pyannote/speaker-diarization-3.1",
    },
    "community1": {
        "module": community1,
        "modelo": "pyannote/speaker-diarization-community-1",
    },
}
MISSING = "sin referencia"
FORCED = "forzado"
FIELDS = [
    "uri", "duracion_s", "hablantes_esperados", "hablantes_estimados", "hablantes",
    "tiempo_s", "rtf", "der_025", "der_0", "missed_s", "false_alarm_s",
    "confusion_s", "referencia_s", "jer", "acierto_hablantes", "turnos_cortos",
    "proporcion_turnos_cortos", "hablantes_minimos", "degenerada", "inestabilidad_ganancia", "inestabilidad_silencio",
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


def variant_id(num_speakers=None, oracle=False, min_speakers=None, max_speakers=None, preprocess="none"):
    try:
        preproc.parse_preprocess(preprocess)
    except ValueError as error:
        raise ValueError(str(error)) from error
    values = (num_speakers, min_speakers, max_speakers)
    if any(value is not None and value < 1 for value in values):
        raise ValueError("El número de hablantes debe ser positivo")
    if (oracle or num_speakers is not None) and (sum((oracle, num_speakers is not None)) > 1 or min_speakers is not None or max_speakers is not None):
        raise ValueError("Opciones de hablantes incompatibles")
    if min_speakers is not None and max_speakers is not None and min_speakers > max_speakers:
        raise ValueError("El mínimo supera el máximo")
    mode = "oraculo" if oracle else (f"num{num_speakers}" if num_speakers is not None else
        "-".join(part for part in (f"min{min_speakers}" if min_speakers else "", f"max{max_speakers}" if max_speakers else "") if part) or "auto")
    return f"{mode}__{preprocess}"


def fragmentation(annotation):
    stats = describe(annotation)
    durations = [segment.duration for segment, _, _ in annotation.itertracks(yield_label=True)]
    total = sum(item["tiempo_s"] for item in stats.values())
    short = sum(duration < 0.5 for duration in durations)
    return {"turnos_cortos": short, "proporcion_turnos_cortos": short / len(durations) if durations else 0,
            "hablantes_minimos": sum(item["tiempo_s"] < total * 0.02 for item in stats.values()),
            "degenerada": len(stats) == 1}


def shift_back(annotation, offset, duration):
    result = Annotation(uri=annotation.uri)
    for segment, track, label in annotation.itertracks(yield_label=True):
        start, end = max(0, segment.start - offset), min(duration, segment.end - offset)
        if end > start:
            result[Segment(start, end), track] = label
    return result


def _prepare_stability_audio(source, destination, preprocess):
    if preprocess == "none":
        return source
    return preproc.process(preprocess, source, destination, reuse=False)


def stability(engine, audio, hypothesis, duration, options, preprocess="none"):
    samples, rate = sf.read(str(audio), dtype="float32", always_2d=True)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        path = root / "perturbacion.wav"
        sf.write(str(path), samples * (10 ** (-6 / 20)), rate)
        gain_audio = _prepare_stability_audio(path, root / "ganancia.wav", preprocess)
        gain = engine.diarize(gain_audio, **options)
        silence = round(rate * 0.05)
        sf.write(str(path), np.concatenate((np.zeros((silence, samples.shape[1]), dtype=np.float32), samples)), rate)
        delayed_audio = _prepare_stability_audio(path, root / "silencio.wav", preprocess)
        delayed = shift_back(engine.diarize(delayed_audio, **options), silence / rate, duration)
    return (compare_annotations(hypothesis, gain)["discrepancia_der"],
            compare_annotations(hypothesis, delayed)["discrepancia_der"])


def _leaderboard_variant_key(line):
    cells = [cell.strip() for cell in line.split("|")[1:-1]]
    return cells[0] if cells else ""


def _normalize_leaderboard_row(line, default_backend="pyannote31"):
    """Prefija filas antiguas sin backend para no confundirlas con community1."""
    key = _leaderboard_variant_key(line)
    if not key or "/" in key:
        return line
    return line.replace(f"| {key} |", f"| {default_backend}/{key} |", 1)


def leaderboard(path, config, rows, aggregate, *, variant, device, run_version, backend_name="pyannote31"):
    # El histórico v0 se conserva con su esquema original y se identifica explícitamente.
    del config
    board_variant = f"{backend_name}/{variant}"
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    if "v0 histórico" not in old and "pyannote31 · sin preprocesamiento · auto" in old:
        old = old.replace("v0 provisional", "v0 histórico (CPU)")
    marker = "\n## Variantes v1\n"
    historic = old.split(marker)[0].rstrip() if old else "# Leaderboard"
    existing = old.split(marker, 1)[1].splitlines() if marker in old else []
    existing = [
        _normalize_leaderboard_row(line)
        for line in existing
        if line.startswith("| ") and not line.startswith("| variante ") and not line.startswith("| ---")
    ]
    existing = [line for line in existing if not line.startswith(f"| {board_variant} |")]
    known = [r["acierto_hablantes"] for r in rows if isinstance(r["acierto_hablantes"], int)]
    forced = any(r["acierto_hablantes"] == FORCED for r in rows)
    accuracy = FORCED if forced else (f"{sum(known)}/{len(known)}" if known else "sin datos")
    turns = sum(sum(v["turnos"] for v in r["hablantes"].values()) for r in rows)
    short = sum(r["turnos_cortos"] for r in rows) / turns if turns else 0
    small = sum(r["hablantes_minimos"] for r in rows)
    stable = [r[k] for r in rows for k in ("inestabilidad_ganancia", "inestabilidad_silencio") if r[k] != MISSING]
    instability = statistics.mean(stable) if stable else None
    def show(value):
        return "sin datos" if value is None else f"{value:.4f}"
    counts = ", ".join(f"{r['uri']}: {r['hablantes_estimados']}" for r in rows)
    reference = sum(r["der_025"] != MISSING for r in rows)
    ref_status = f"{reference}/{len(rows)} con referencia" if reference else MISSING
    der = aggregate["der_0"]
    der_text = f"{der:.4f}" if isinstance(der, float) else str(der)
    line = f"| {board_variant} | {run_version} | {device} | {len(rows)} | {counts} | {accuracy} | {short:.4f} | {small} | {show(instability)} | {sum(r['degenerada'] for r in rows)} | {ref_status} | {der_text} |"
    existing.append(line)
    def ranking(line):
        cells = [cell.strip() for cell in line.split("|")[1:-1]]
        def number(value, missing):
            try:
                return float(value)
            except ValueError:
                return missing
        accuracy_parts = cells[5].split("/")
        if len(accuracy_parts) == 2:
            numerator = number(accuracy_parts[0], 0)
            denominator = number(accuracy_parts[1], 0)
            accuracy = numerator / denominator if denominator else 0
        else:
            accuracy = number(cells[5], 0)
        return (-accuracy, number(cells[6], float("inf")), number(cells[7], float("inf")), number(cells[8], float("inf")), cells[0])
    existing.sort(key=ranking)
    headers = "| variante | versión | dispositivo | cobertura | hablantes por fragmento | acierto | turnos cortos | hablantes mínimos | inestabilidad | degenerados | referencia | DER@0 |"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(historic + marker + "\n" + headers + "\n| " + " | ".join(["---"] * 12) + " |\n" + "\n".join(existing) + "\n", encoding="utf-8")


def evaluate(manifest, out, *, backend, device="cpu", seed=0, reference_dir=Path("eval/reference"), leaderboard_path=Path("eval/results/LEADERBOARD.md"), num_speakers=None, oracle=False, min_speakers=None, max_speakers=None, preprocess="none", run_version="v1 provisional", estabilidad=False, cache_root=preproc.CACHE_ROOT, backend_name="pyannote31"):
    variant = variant_id(num_speakers, oracle, min_speakers, max_speakers, preprocess)
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
    rows, hashes, processed_hashes, shift_samples = [], {}, {}, {}
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
        diarize_audio = audio
        if preprocess != "none":
            diarize_audio = preproc.ensure_cached(preprocess, audio, uri, cache_root=cache_root)
            processed_hashes[uri] = digest(diarize_audio)
            meta = preproc.read_shift(diarize_audio) or {}
            shift_samples[uri] = int(meta.get("desfase_muestras", 0))
        start = time.perf_counter()
        expected = entry.get("n_hablantes_esperado", "").strip()
        options = {"num_speakers": int(expected) if oracle and expected else num_speakers, "min_speakers": min_speakers, "max_speakers": max_speakers}
        options = {key: value for key, value in options.items() if value is not None}
        hypothesis = engine.diarize(diarize_audio, **options)
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
        row.update(fragmentation(hypothesis))
        row["acierto_hablantes"] = (FORCED if "num_speakers" in options else
                                    int(len(hypothesis.labels()) == int(expected)) if expected else MISSING)
        row["inestabilidad_ganancia"], row["inestabilidad_silencio"] = (
            stability(engine, audio, hypothesis, duration, options, preprocess)
            if estabilidad else (MISSING, MISSING)
        )
        row.update(metrics(reference, hypothesis))
        rows.append(row)
    with (out / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({**row, "hablantes": json.dumps(row["hablantes"], ensure_ascii=False, sort_keys=True)} for row in rows)
    summary = [f"# Evaluación {run_version} · {variant}", "", f"Cobertura: {len(rows)} fragmento(s).", ""]
    for row in rows:
        summary.append(f"## {row['uri']} — {row['hablantes_estimados']} hablante(s), RTF {row['rtf']:.4f}")
        summary.append(f"Fragmentación: {row['proporcion_turnos_cortos']:.4f} turnos cortos, {row['hablantes_minimos']} hablantes mínimos; degenerada: {row['degenerada']}. Acierto de hablantes: {row['acierto_hablantes']}. Inestabilidad (ganancia/silencio): {row['inestabilidad_ganancia']} / {row['inestabilidad_silencio']}.")
        summary.append(f"Referencia: {'disponible' if row['der_025'] != MISSING else MISSING}.")
        if row["der_025"] != MISSING:
            summary.append(f"DER@0.25: {row['der_025']:.4f}; DER@0: {row['der_0']:.4f}; JER: {row['jer']:.4f}.")
        for label, stats in row["hablantes"].items():
            summary.append(f"- {label}: {stats['tiempo_s']:.3f} s, {stats['turnos']} turnos; "
                           f"duración mínima/mediana/máxima: {stats['duracion_min_s']:.3f}/"
                           f"{stats['duracion_mediana_s']:.3f}/{stats['duracion_max_s']:.3f} s.")
        summary.append("")
    known = [row["acierto_hablantes"] for row in rows if isinstance(row["acierto_hablantes"], int)]
    forced = any(row["acierto_hablantes"] == FORCED for row in rows)
    turns = sum(sum(stats["turnos"] for stats in row["hablantes"].values()) for row in rows)
    short = sum(row["turnos_cortos"] for row in rows)
    instability = [row[key] for row in rows for key in ("inestabilidad_ganancia", "inestabilidad_silencio") if row[key] != MISSING]
    accuracy = FORCED if forced else (f"{sum(known)}/{len(known)} conocidos" if known else "sin datos")
    summary.extend(["## Agregado", "", f"Acierto de hablantes: {accuracy}.",
                    f"Turnos cortos: {short}/{turns}; hablantes mínimos: {sum(row['hablantes_minimos'] for row in rows)}; fragmentos degenerados: {sum(row['degenerada'] for row in rows)}.",
                    f"Inestabilidad media: {statistics.mean(instability):.4f}." if instability else "Inestabilidad: sin datos.", ""])
    (out / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    meta = BACKENDS.get(backend_name, {"modelo": backend_name})
    run_manifest = {
        "modelo": meta["modelo"],
        "backend": backend_name,
        "device": device,
        "seed": seed,
        "parametros": {
            "num_speakers": num_speakers, "oracle": oracle, "min_speakers": min_speakers,
            "max_speakers": max_speakers, "preprocesamiento": preprocess,
            "estabilidad": estabilidad, "version": run_version, "variante": variant,
        },
        "audio_sha256": hashes,
        "audio_procesado_sha256": processed_hashes,
        "desfase_compensado_muestras": shift_samples,
    }
    if backend_name == "pyannote31":
        run_manifest["pyannote_audio"] = version("pyannote.audio")
        run_manifest["whisperx"] = version("whisperx")
    elif backend_name == "community1":
        isolated = community1.isolated_pyannote_version()
        if isolated is not None:
            run_manifest["pyannote_audio"] = isolated
    (out / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    config = f"{backend_name} · sin preprocesamiento · auto"
    if reference_count:
        aggregate = {"der_025": abs(primary), "der_0": abs(strict), "jer": abs(jer),
                     "missed_s": primary.accumulated_["missed detection"],
                     "false_alarm_s": primary.accumulated_["false alarm"],
                     "confusion_s": primary.accumulated_["confusion"]}
    else:
        aggregate = dict.fromkeys(("der_025", "der_0", "jer", "missed_s", "false_alarm_s", "confusion_s"), MISSING)
    leaderboard(
        leaderboard_path, config, rows, aggregate, variant=variant, device=device,
        run_version=run_version, backend_name=backend_name,
    )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=sorted(BACKENDS), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--num-speakers", type=int)
    parser.add_argument("--oracle-num-speakers", action="store_true")
    parser.add_argument("--min-speakers", type=int)
    parser.add_argument("--max-speakers", type=int)
    parser.add_argument("--preprocess", default="none",
                        help="none, una técnica o cadena con + (p. ej. highpass80+loudnorm)")
    parser.add_argument("--version", default="v1 provisional")
    parser.add_argument("--estabilidad", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    try:
        variant = variant_id(args.num_speakers, args.oracle_num_speakers, args.min_speakers, args.max_speakers, args.preprocess)
    except ValueError as error:
        parser.error(str(error))
    out_root = args.out if args.out is not None else Path("eval/results") / args.backend
    engine = BACKENDS[args.backend]["module"].Backend
    evaluate(
        args.manifest, out_root / variant, backend=engine, device=args.device, seed=args.seed,
        num_speakers=args.num_speakers, oracle=args.oracle_num_speakers,
        min_speakers=args.min_speakers, max_speakers=args.max_speakers,
        preprocess=args.preprocess, run_version=args.version, estabilidad=args.estabilidad,
        backend_name=args.backend,
    )


if __name__ == "__main__":
    main()
