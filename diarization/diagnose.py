"""Diagnóstico de degradaciones acústicas por archivo (LTAS, zumbido, corte, etc.).

En --inputs no hay máscara de voz: snr_db queda vacío y cutoff/tilt se miden
sobre todo el archivo (música y silencios incluidos). En --manifest, el SNR
usa la voz del RTTM; los tramos sin etiqueta pueden incluir música o
multitud, así que el valor puede salir bajo sin ser un error de cálculo.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal
from scipy.ndimage import median_filter

from diarization.preprocess import ffmpeg_bin
from diarization.rttm import read_rttm

VERSION = "v0"
TARGET_SR = 16000
NPERSEG = 8192
NPERSEG_HUM = 32768
HUM_HALF_BW_HZ = 2.0
HUM_NEIGH_HZ = 15.0
HUM_HARMONICS = range(1, 7)
PICO_BAJO_LO_HZ = 30.0
PICO_BAJO_HI_HZ = 300.0
CLIP_THRESHOLD = 0.99
CLICK_Z = 8.0
CLICK_WINDOW_S = 0.020
CLICK_MAX_RUN = 2
CLICK_NEIGH_RATIO = 1.5
CLICK_NEIGH_RADIUS = 5
MANIFEST_FIELDS = ("uri", "ruta_audio", "duracion_s", "n_hablantes_esperado", "tipo", "notas")
MEASURE_FIELDS = (
    "hum50_db",
    "hum100_db",
    "hum_max_db",
    "pico_bajo_hz",
    "pico_bajo_db",
    "cutoff_hz",
    "tilt_db_oct",
    "clip_ratio",
    "clip_plateau",
    "lufs",
    "lra_lu",
    "snr_db",
    "clicks_por_min",
)
CSV_FIELDS_MANIFEST = ("version", "uri", *MEASURE_FIELDS)
CSV_FIELDS_INPUTS = ("version", "ruta", *MEASURE_FIELDS)
DEFAULT_MANIFEST_OUT = Path("eval/results/diagnostico")
DEFAULT_RTTM_DIR = Path("eval/results/pyannote31/auto__loudnorm/hypotheses")
_I_RE = re.compile(r"^\s*I:\s*(-?\d+(?:\.\d+)?)\s*LUFS", re.MULTILINE)
_LRA_RE = re.compile(r"^\s*LRA:\s*(-?\d+(?:\.\d+)?)\s*LU", re.MULTILINE)


def _is_git_ignored(path: Path) -> bool:
    """True si git check-ignore marca la ruta (0). False si no o si no hay repo."""
    result = subprocess.run(
        ["git", "check-ignore", "-q", "--", str(path)],
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def load_mono(path: Path, *, sample_rate: int | None = None) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = np.mean(audio, axis=1)
    audio = np.asarray(audio, dtype=np.float64)
    if sample_rate is not None and sr != sample_rate:
        raise ValueError(f"Se esperaba {sample_rate} Hz, se obtuvo {sr}")
    return audio, int(sr)


def voice_mask_from_rttm(rttm_path: Path, n_samples: int, sample_rate: int) -> np.ndarray:
    annotation = read_rttm(rttm_path)
    mask = np.zeros(n_samples, dtype=bool)
    for segment, _, _ in annotation.itertracks(yield_label=True):
        start = max(0, int(segment.start * sample_rate))
        end = min(n_samples, int(np.ceil(segment.end * sample_rate)))
        if end > start:
            mask[start:end] = True
    return mask


def welch_db(audio: np.ndarray, sample_rate: int, *, nperseg: int) -> tuple[np.ndarray, np.ndarray]:
    nperseg = min(nperseg, max(8, len(audio)))
    freqs, power = signal.welch(audio, sample_rate, nperseg=nperseg)
    return freqs, 10.0 * np.log10(power + 1e-20)


def hum_db_at(
    freqs: np.ndarray,
    power_db: np.ndarray,
    center_hz: float,
    *,
    half_bw_hz: float = HUM_HALF_BW_HZ,
    neigh_hz: float = HUM_NEIGH_HZ,
) -> float:
    central = (freqs >= center_hz - half_bw_hz) & (freqs <= center_hz + half_bw_hz)
    neighbor = (
        (freqs >= center_hz - neigh_hz)
        & (freqs <= center_hz + neigh_hz)
        & ~central
    )
    if not np.any(central) or not np.any(neighbor):
        return float("nan")
    return float(np.max(power_db[central]) - np.median(power_db[neighbor]))


def pico_bajo(freqs: np.ndarray, power_db: np.ndarray) -> tuple[float, float]:
    """Pico estrecho más alto entre 30 y 300 Hz (incluye múltiplos de 50)."""
    band = (freqs >= PICO_BAJO_LO_HZ) & (freqs <= PICO_BAJO_HI_HZ)
    if not np.any(band):
        return float("nan"), float("nan")
    best_hz = float("nan")
    best_db = float("-inf")
    for center in freqs[band]:
        value = hum_db_at(freqs, power_db, float(center))
        if np.isfinite(value) and value > best_db:
            best_db = value
            best_hz = float(center)
    if not np.isfinite(best_db):
        return float("nan"), float("nan")
    return best_hz, float(best_db)


def measure_hum(audio: np.ndarray, sample_rate: int) -> dict[str, float]:
    freqs, power_db = welch_db(audio, sample_rate, nperseg=NPERSEG_HUM)
    values = {
        f"hum{k * 50}_db": hum_db_at(freqs, power_db, float(k * 50))
        for k in HUM_HARMONICS
    }
    harmonics = [values[f"hum{k * 50}_db"] for k in HUM_HARMONICS]
    finite = [v for v in harmonics if np.isfinite(v)]
    values["hum_max_db"] = float(max(finite)) if finite else float("nan")
    peak_hz, peak_db = pico_bajo(freqs, power_db)
    values["pico_bajo_hz"] = peak_hz
    values["pico_bajo_db"] = peak_db
    return values


def cutoff_hz(freqs: np.ndarray, power_db: np.ndarray) -> float:
    band = (freqs > 200.0) & (freqs < 4000.0)
    if not np.any(band):
        return float("nan")
    peak = float(np.max(power_db[band]))
    above = freqs[(power_db > peak - 30.0) & (freqs > 1000.0)]
    if above.size == 0:
        return float("nan")
    return float(np.max(above))


def tilt_db_oct(freqs: np.ndarray, power_db: np.ndarray) -> float:
    mask = (freqs > 300.0) & (freqs < 3400.0)
    if np.count_nonzero(mask) < 2:
        return float("nan")
    return float(np.polyfit(np.log2(freqs[mask]), power_db[mask], 1)[0])


def clip_metrics(audio: np.ndarray) -> tuple[float, float]:
    """Proporción cerca de escala completa y meseta en extremos del histograma."""
    if audio.size == 0:
        return 0.0, 0.0
    abs_x = np.abs(audio)
    clip_ratio = float(np.mean(abs_x >= CLIP_THRESHOLD))
    hist, _edges = np.histogram(audio, bins=64, range=(-1.0, 1.0))
    if hist.sum() == 0:
        return clip_ratio, 0.0
    left, right = int(hist[0]), int(hist[-1])
    inner_left, inner_right = int(hist[1]), int(hist[-2])
    plateau = 0.0
    if left >= 5 and left >= 3 * max(inner_left, 1):
        plateau = 1.0
    if right >= 5 and right >= 3 * max(inner_right, 1):
        plateau = 1.0
    if plateau == 0.0 and clip_ratio >= 0.001:
        body = hist[2:-2]
        if body.size and max(left, right) >= 5 * max(int(np.median(body)), 1):
            plateau = 1.0
    return clip_ratio, plateau


def clicks_per_minute(audio: np.ndarray, sample_rate: int) -> float:
    """Impulsos cortos: error de predicción vs MAD local (~20 ms) y entorno."""
    if audio.size < 5:
        return 0.0
    err = np.abs(audio[1:-1] - 0.5 * (audio[:-2] + audio[2:]))
    window = max(5, int(round(sample_rate * CLICK_WINDOW_S)))
    if window % 2 == 0:
        window += 1
    local_med = median_filter(err, size=window, mode="nearest")
    local_mad = median_filter(np.abs(err - local_med), size=window, mode="nearest") + 1e-12
    above = err > (local_med + CLICK_Z * 1.4826 * local_mad)
    peak = np.zeros(err.size, dtype=bool)
    peak[1:-1] = (err[1:-1] > err[:-2]) & (err[1:-1] > err[2:])
    above &= peak
    radius = CLICK_NEIGH_RADIUS
    for index in np.flatnonzero(above):
        lo = max(0, index - radius)
        hi = min(err.size, index + radius + 1)
        surround = np.concatenate([err[lo:index], err[index + 1 : hi]])
        if surround.size and err[index] < CLICK_NEIGH_RATIO * float(np.max(surround)):
            above[index] = False
    padded = np.concatenate(([False], above, [False]))
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:])
    count = int(np.count_nonzero((ends - starts) <= CLICK_MAX_RUN))
    minutes = audio.size / sample_rate / 60.0
    return float(count / minutes) if minutes > 0 else float(count)


def snr_db(audio: np.ndarray, voice_mask: np.ndarray) -> float:
    if voice_mask is None or voice_mask.size != audio.size:
        return float("nan")
    speech = audio[voice_mask]
    silence = audio[~voice_mask]
    if speech.size == 0 or silence.size == 0:
        return float("nan")
    ps = float(np.mean(speech ** 2))
    pn = float(np.mean(silence ** 2)) + 1e-20
    return float(10.0 * np.log10(ps / pn))


def ebur128_loudness(path: Path) -> tuple[float, float]:
    command = [
        ffmpeg_bin(),
        "-nostdin",
        "-i",
        str(path),
        "-af",
        "ebur128=peak=true",
        "-f",
        "null",
        "-",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    text = (result.stderr or "") + (result.stdout or "")
    if result.returncode != 0:
        raise RuntimeError("ffmpeg ebur128 falló")
    match_i = _I_RE.search(text)
    match_lra = _LRA_RE.search(text)
    if match_i is None or match_lra is None:
        raise RuntimeError("No se pudo leer I/LRA de ebur128")
    return float(match_i.group(1)), float(match_lra.group(1))


def decode_to_wav(
    source: Path,
    destination: Path,
    *,
    sample_rate: int = TARGET_SR,
    start_s: float | None = None,
    duration_s: float | None = None,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [ffmpeg_bin(), "-nostdin", "-v", "error", "-y"]
    if start_s is not None:
        command.extend(["-ss", f"{start_s:.6f}"])
    command.extend(["-i", str(source)])
    if duration_s is not None:
        command.extend(["-t", f"{duration_s:.6f}"])
    command.extend(
        [
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            str(destination),
        ]
    )
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not destination.is_file():
        raise RuntimeError("No se pudo decodificar")


def media_duration_s(path: Path) -> float:
    command = [
        ffmpeg_bin(),
        "-nostdin",
        "-i",
        str(path),
        "-f",
        "null",
        "-",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    text = result.stderr or ""
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", text)
    if match is None:
        raise RuntimeError("No se pudo leer la duración")
    hours, minutes, seconds = int(match.group(1)), int(match.group(2)), float(match.group(3))
    return hours * 3600 + minutes * 60 + seconds


def sample_windows(duration_s: float, max_minutes: float | None) -> list[tuple[float, float]]:
    if duration_s <= 0:
        raise ValueError("Duración inválida")
    if max_minutes is None or max_minutes * 60 >= duration_s:
        return [(0.0, duration_s)]
    total = float(max_minutes) * 60.0
    n_chunks = max(1, int(round(total / 60.0)))
    chunk = total / n_chunks
    if n_chunks == 1:
        start = max(0.0, (duration_s - chunk) / 2.0)
        return [(start, min(chunk, duration_s - start))]
    span = duration_s - chunk
    return [(i * span / (n_chunks - 1), chunk) for i in range(n_chunks)]


def load_analysis_audio(
    source: Path,
    *,
    max_minutes: float | None = None,
    already_wav_16k: bool = False,
) -> tuple[np.ndarray, int, Path | None]:
    """Devuelve (audio, sr, wav_para_ebur128_o_None_si_es_source)."""
    if already_wav_16k and max_minutes is None:
        audio, sr = load_mono(source, sample_rate=TARGET_SR)
        return audio, sr, None

    if already_wav_16k:
        duration = float(sf.info(str(source)).duration)
    else:
        duration = media_duration_s(source)
    windows = sample_windows(duration, max_minutes)
    if len(windows) == 1 and windows[0][0] == 0.0 and abs(windows[0][1] - duration) < 1e-6:
        if already_wav_16k:
            audio, sr = load_mono(source, sample_rate=TARGET_SR)
            return audio, sr, None
        temp_dir = Path(tempfile.mkdtemp(prefix="diagnose-"))
        try:
            combined = temp_dir / "combined.wav"
            decode_to_wav(source, combined)
            audio, sr = load_mono(combined, sample_rate=TARGET_SR)
            return audio, sr, combined
        except Exception:
            for path in temp_dir.glob("*"):
                path.unlink(missing_ok=True)
            temp_dir.rmdir()
            raise

    temp_dir = Path(tempfile.mkdtemp(prefix="diagnose-"))
    try:
        chunks: list[np.ndarray] = []
        full = None
        if already_wav_16k:
            full, _ = load_mono(source, sample_rate=TARGET_SR)
        for index, (start, length) in enumerate(windows):
            if already_wav_16k:
                i0 = int(start * TARGET_SR)
                i1 = min(len(full), int(round((start + length) * TARGET_SR)))
                chunks.append(full[i0:i1])
            else:
                piece = temp_dir / f"chunk-{index:03d}.wav"
                decode_to_wav(source, piece, start_s=start, duration_s=length)
                data, _ = load_mono(piece, sample_rate=TARGET_SR)
                chunks.append(data)
        audio = np.concatenate(chunks)
        combined = temp_dir / "combined.wav"
        sf.write(str(combined), audio.astype(np.float32), TARGET_SR, subtype="PCM_16")
        return audio, TARGET_SR, combined
    except Exception:
        for path in temp_dir.glob("*"):
            path.unlink(missing_ok=True)
        temp_dir.rmdir()
        raise


def analyze_array(
    audio: np.ndarray,
    sample_rate: int,
    *,
    voice_mask: np.ndarray | None = None,
    wav_for_loudness: Path | None = None,
) -> dict[str, float]:
    if sample_rate != TARGET_SR:
        raise ValueError(f"Se espera {TARGET_SR} Hz")
    freqs, power_db = welch_db(audio, sample_rate, nperseg=NPERSEG)
    voice_power = None
    if voice_mask is not None and np.any(voice_mask):
        speech = audio[voice_mask]
        if speech.size >= NPERSEG // 4:
            voice_power = welch_db(speech, sample_rate, nperseg=NPERSEG)

    hum = measure_hum(audio, sample_rate)
    spectrum_for_cutoff = voice_power if voice_power is not None else (freqs, power_db)
    spectrum_for_tilt = voice_power if voice_power is not None else (freqs, power_db)

    clip_ratio, clip_plateau = clip_metrics(audio)
    if wav_for_loudness is not None:
        lufs, lra = ebur128_loudness(wav_for_loudness)
    else:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            temp_wav = Path(handle.name)
        try:
            sf.write(str(temp_wav), audio.astype(np.float32), sample_rate, subtype="PCM_16")
            lufs, lra = ebur128_loudness(temp_wav)
        finally:
            temp_wav.unlink(missing_ok=True)

    return {
        "hum50_db": hum["hum50_db"],
        "hum100_db": hum["hum100_db"],
        "hum_max_db": hum["hum_max_db"],
        "pico_bajo_hz": hum["pico_bajo_hz"],
        "pico_bajo_db": hum["pico_bajo_db"],
        "cutoff_hz": cutoff_hz(*spectrum_for_cutoff),
        "tilt_db_oct": tilt_db_oct(*spectrum_for_tilt),
        "clip_ratio": clip_ratio,
        "clip_plateau": clip_plateau,
        "lufs": lufs,
        "lra_lu": lra,
        "snr_db": snr_db(audio, voice_mask) if voice_mask is not None else float("nan"),
        "clicks_por_min": clicks_per_minute(audio, sample_rate),
    }


def _pyplot():
    mpl_dir = os.path.join(tempfile.gettempdir(), "mplconfig-diarization")
    os.makedirs(mpl_dir, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", mpl_dir)
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def save_spectrum_plot(
    freqs: np.ndarray,
    power_db: np.ndarray,
    destination: Path,
    *,
    title: str,
) -> None:
    plt = _pyplot()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogx(freqs[1:], power_db[1:], color="black", linewidth=0.8)
    ax.set_xlabel("Hz")
    ax.set_ylabel("dB")
    ax.set_title(title)
    ax.set_xlim(20, freqs[-1])
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(destination, dpi=120)
    plt.close(fig)


def _cleanup_temp(path: Path | None) -> None:
    if path is None:
        return
    parent = path.parent
    path.unlink(missing_ok=True)
    if parent.name.startswith("diagnose-"):
        for child in parent.glob("*"):
            child.unlink(missing_ok=True)
        parent.rmdir()


def diagnose_path(
    source: Path,
    *,
    rttm_path: Path | None = None,
    max_minutes: float | None = None,
    already_wav_16k: bool = False,
    spectrum_out: Path | None = None,
    label: str = "",
) -> dict[str, float]:
    audio, sr, temp_wav = load_analysis_audio(
        source, max_minutes=max_minutes, already_wav_16k=already_wav_16k,
    )
    try:
        mask = None
        if rttm_path is not None and rttm_path.is_file() and max_minutes is None:
            mask = voice_mask_from_rttm(rttm_path, len(audio), sr)
        loudness_wav = temp_wav if temp_wav is not None else (source if already_wav_16k else None)
        measures = analyze_array(audio, sr, voice_mask=mask, wav_for_loudness=loudness_wav)
        if spectrum_out is not None:
            freqs, power_db = welch_db(audio, sr, nperseg=NPERSEG)
            save_spectrum_plot(freqs, power_db, spectrum_out, title=label or "ltas")
        return measures
    finally:
        _cleanup_temp(temp_wav)


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(MANIFEST_FIELDS):
            raise ValueError("Cabecera de manifiesto inválida")
        return list(reader)


def read_inputs_list(path: Path) -> list[Path]:
    lines = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(Path(line))
    if not lines:
        raise ValueError("La lista de entradas está vacía")
    return lines


def write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent, delete=False,
    ) as handle:
        temporary = Path(handle.name)
        try:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                out = {key: row.get(key, "") for key in fieldnames}
                writer.writerow(out)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


def _format_measure(value: float) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return ""
    return f"{float(value):.6g}"


def _row_from_measures(measures: dict[str, float], *, key: str, key_value: str, fields: tuple[str, ...]) -> dict:
    entry = {"version": VERSION, key: key_value}
    for name in fields:
        if name in ("version", key):
            continue
        entry[name] = _format_measure(measures[name])
    return entry


def run_manifest(
    manifest_path: Path,
    out_dir: Path,
    *,
    rttm_dir: Path,
) -> list[dict]:
    rows_out: list[dict] = []
    for row in read_manifest(manifest_path):
        uri = row["uri"]
        audio_path = Path(row["ruta_audio"])
        if "cache" in audio_path.parts:
            raise ValueError(f"El diagnóstico no usa audio de caché: {uri}")
        if not audio_path.is_file():
            raise FileNotFoundError(f"Falta el audio de {uri}")
        rttm_path = rttm_dir / f"{uri}.rttm"
        spectrum_path = out_dir / f"{uri}_ltas.png"
        measures = diagnose_path(
            audio_path,
            rttm_path=rttm_path if rttm_path.is_file() else None,
            already_wav_16k=True,
            spectrum_out=spectrum_path,
            label=uri,
        )
        rows_out.append(
            _row_from_measures(measures, key="uri", key_value=uri, fields=CSV_FIELDS_MANIFEST)
        )
        print(f"{uri}: ok")
    write_csv(out_dir / "diagnostico.csv", CSV_FIELDS_MANIFEST, rows_out)
    return rows_out


def run_inputs(
    inputs_path: Path,
    out_dir: Path,
    *,
    max_minutes: float | None,
) -> list[dict]:
    if out_dir.resolve() == DEFAULT_MANIFEST_OUT.resolve():
        raise ValueError(
            "Con --inputs use otra carpeta de salida; "
            f"{DEFAULT_MANIFEST_OUT} queda para el modo --manifest"
        )
    if not _is_git_ignored(out_dir):
        raise ValueError("La carpeta --out debe estar ignorada por git (.gitignore)")
    sources = read_inputs_list(inputs_path)
    total = len(sources)
    rows_out: list[dict] = []
    for index, source in enumerate(sources, 1):
        label = f"{index}/{total}"
        if not source.is_file():
            raise FileNotFoundError(f"{label}: no existe")
        spectrum_path = out_dir / f"{index}_ltas.png"
        try:
            measures = diagnose_path(
                source,
                max_minutes=max_minutes,
                already_wav_16k=False,
                spectrum_out=spectrum_path,
                label=str(index),
            )
        except RuntimeError as error:
            raise RuntimeError(f"{label}: error al decodificar") from error
        rows_out.append(
            _row_from_measures(
                measures, key="ruta", key_value=source.as_posix(), fields=CSV_FIELDS_INPUTS,
            )
        )
        print(f"{label}: ok")
    write_csv(out_dir / "diagnostico.csv", CSV_FIELDS_INPUTS, rows_out)
    return rows_out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--manifest", type=Path, help="Manifiesto de evaluación (URI evNN)")
    mode.add_argument("--inputs", type=Path, help="Lista de rutas a audio/video (una por línea)")
    parser.add_argument(
        "--rttm-dir",
        type=Path,
        default=DEFAULT_RTTM_DIR,
        help="RTTM para máscara de voz (solo --manifest)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Carpeta de salida (obligatoria con --inputs; por defecto diagnostico/ con --manifest)",
    )
    parser.add_argument(
        "--max-minutes",
        type=float,
        help="Con --inputs, minutos totales a muestrear a lo largo del archivo",
    )
    args = parser.parse_args(argv)

    try:
        if args.manifest is not None:
            out_dir = args.out if args.out is not None else DEFAULT_MANIFEST_OUT
            if args.max_minutes is not None:
                parser.error("--max-minutes solo aplica con --inputs")
            run_manifest(args.manifest, out_dir, rttm_dir=args.rttm_dir)
        else:
            if args.out is None:
                parser.error("--out es obligatorio con --inputs")
            run_inputs(args.inputs, args.out, max_minutes=args.max_minutes)
    except (OSError, ValueError, RuntimeError, FileNotFoundError) as error:
        parser.exit(1, f"Error: {type(error).__name__}: {error}\n")


if __name__ == "__main__":
    main()
