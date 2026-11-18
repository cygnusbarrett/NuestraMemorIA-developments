"""Recorta fuentes locales en fragmentos de evaluación."""

import argparse
import csv
import math
import re
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

import imageio_ffmpeg

SOURCE_FIELDS = ("uri", "archivo_original", "inicio_s", "fin_s")
MANIFEST_FIELDS = ("uri", "ruta_audio", "duracion_s", "n_hablantes_esperado", "tipo", "notas")


def read_csv(path, fields):
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(fields):
            raise ValueError("Cabecera CSV inválida")
        rows = list(reader)
    uris = [row["uri"] for row in rows]
    if any(not re.fullmatch(r"ev\d{2,}", uri) for uri in uris) or len(uris) != len(set(uris)):
        raise ValueError("URI inválido o duplicado")
    return rows


def wav_duration(path):
    try:
        with wave.open(str(path), "rb") as audio:
            if audio.getframerate() != 16000 or audio.getnchannels() != 1:
                raise ValueError("Formato WAV incorrecto")
            return audio.getnframes() / audio.getframerate()
    except (wave.Error, EOFError) as error:
        raise ValueError("WAV inválido") from error


def run(sources_path, manifest_path, audio_dir, *, dry_run=False):
    sources = read_csv(sources_path, SOURCE_FIELDS)
    manifest = read_csv(manifest_path, MANIFEST_FIELDS)
    rows = {row["uri"]: row for row in manifest}
    created = reused = 0
    for source in sources:
        uri = source["uri"]
        start, end = float(source["inicio_s"]), float(source["fin_s"])
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
            raise ValueError(f"Intervalo inválido para {uri}")
        original = Path(source["archivo_original"])
        if not source["archivo_original"] or not original.is_file():
            raise ValueError(f"Fuente no disponible para {uri}")
        output = audio_dir / f"{uri}.wav"
        duration = end - start
        if output.exists():
            if abs(wav_duration(output) - duration) > 0.01:
                raise ValueError(f"El WAV existente de {uri} no coincide con el tramo")
            reused += 1
        else:
            if not dry_run:
                audio_dir.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(dir=audio_dir) as temp:
                    target = Path(temp) / "fragment.wav"
                    command = [
                        shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe(),
                        "-nostdin", "-v", "error", "-ss", str(start), "-i", str(original),
                        "-t", str(duration), "-vn", "-ac", "1", "-ar", "16000",
                        "-c:a", "pcm_s16le", "-n", str(target),
                    ]
                    result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    if result.returncode or not target.exists() or abs(wav_duration(target) - duration) > 0.01:
                        raise RuntimeError(f"No se pudo generar el fragmento {uri}")
                    target.rename(output)
            created += 1
        row = rows.setdefault(uri, dict.fromkeys(MANIFEST_FIELDS, ""))
        row.update(uri=uri, ruta_audio=output.as_posix(), duracion_s=f"{duration:g}")
    if not dry_run:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=manifest_path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            try:
                writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
                writer.writeheader()
                writer.writerows(rows.values())
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        temporary.replace(manifest_path)
    return created, reused


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=Path("eval/fuentes.local.csv"))
    parser.add_argument("--manifest", type=Path, default=Path("eval/manifest.csv"))
    parser.add_argument("--audio-dir", type=Path, default=Path("eval/audio"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        created, reused = run(args.sources, args.manifest, args.audio_dir, dry_run=args.dry_run)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Error: {type(error).__name__}; revise las fuentes y el manifiesto locales.\n")
    print(f"{'Simulación: ' if args.dry_run else ''}{created} por crear o creados; {reused} reutilizados")


if __name__ == "__main__":
    main()
