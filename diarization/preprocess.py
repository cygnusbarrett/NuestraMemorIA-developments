"""Preprocesamiento acústico para diarización (ffmpeg y, en 2.5, subprocesos)."""

import json
import os
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import soundfile as sf
from scipy import signal

DURATION_TOLERANCE_S = 0.01
MAX_LAG_S = 0.5
ALIGN_TOLERANCE_S = 0.001
CACHE_ROOT = Path("eval/cache")


def ffmpeg_bin():
    return shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe()


def wav_info(path):
    with wave.open(str(path), "rb") as source:
        channels = source.getnchannels()
        sample_rate = source.getframerate()
        frames = source.getnframes()
    if channels < 1 or sample_rate <= 0 or frames < 0:
        raise ValueError("Audio inválido")
    return sample_rate, channels, frames / sample_rate


def duration(path):
    return wav_info(path)[2]


def cached_path(technique, uri, cache_root=CACHE_ROOT):
    return Path(cache_root) / technique / f"{uri}.wav"


def meta_path(wav_path):
    return Path(wav_path).with_suffix(".json")


def read_shift(wav_path):
    path = meta_path(wav_path)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _run_ffmpeg_filter(source, destination, filter_expr, sample_rate, channels):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".preproc-", suffix=".wav", dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    command = [
        ffmpeg_bin(), "-nostdin", "-v", "error", "-y",
        "-i", str(source),
        "-af", filter_expr,
        "-ar", str(sample_rate),
        "-ac", str(channels),
        "-c:a", "pcm_s16le",
        str(temporary),
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        temporary.replace(destination)
    except subprocess.CalledProcessError as error:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg falló para {destination.name}") from error
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _ffmpeg_technique(filter_expr):
    def apply(source, destination):
        sample_rate, channels, _ = wav_info(source)
        _run_ffmpeg_filter(source, destination, filter_expr, sample_rate, channels)

    return apply


# Registro por nombre: cada entrada es callable(source, destination) -> None.
# La 2.5 agregará técnicas que llamen a subprocesos en envs/preproc/.
TECHNIQUES = {
    "loudnorm": _ffmpeg_technique("loudnorm=I=-16:TP=-1.5:LRA=11"),
    "highpass80": _ffmpeg_technique("highpass=f=80"),
    "afftdn": _ffmpeg_technique("afftdn"),
}


def register(name, apply):
    if not name or name == "none":
        raise ValueError("Nombre de técnica inválido")
    TECHNIQUES[name] = apply


def known_names():
    return ("none", *sorted(TECHNIQUES))


def _mono(samples):
    if samples.ndim == 1:
        return samples.astype(np.float32, copy=False)
    return samples.mean(axis=1).astype(np.float32, copy=False)


def estimate_lag_samples(reference, estimate, sample_rate, max_lag_s=MAX_LAG_S):
    """Desfase en muestras: positivo si `estimate` llega tarde respecto de `reference`."""
    reference = _mono(reference)
    estimate = _mono(estimate)
    if reference.size == 0 or estimate.size == 0:
        return None
    if float(np.max(np.abs(reference))) < 1e-8 or float(np.max(np.abs(estimate))) < 1e-8:
        return None
    max_lag = max(1, int(round(max_lag_s * sample_rate)))
    correlation = signal.correlate(estimate, reference, mode="full", method="fft")
    lags = signal.correlation_lags(estimate.size, reference.size, mode="full")
    window = (lags >= -max_lag) & (lags <= max_lag)
    if not np.any(window):
        return None
    return int(lags[window][np.argmax(correlation[window])])


def _shift_samples(samples, lag):
    if lag == 0:
        return samples
    channels = 1 if samples.ndim == 1 else samples.shape[1]
    flat = samples.reshape(-1, channels)
    length = flat.shape[0]
    if lag > 0:
        pad = np.zeros((lag, channels), dtype=flat.dtype)
        shifted = np.concatenate([flat[lag:], pad], axis=0)
    else:
        pad = np.zeros((-lag, channels), dtype=flat.dtype)
        shifted = np.concatenate([pad, flat[: length + lag]], axis=0)
    return shifted.reshape(samples.shape) if samples.ndim == 1 and channels == 1 else shifted


def _write_wav_atomic(path, samples, sample_rate):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".preproc-", suffix=".wav", dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        sf.write(str(temporary), samples, sample_rate, subtype="PCM_16")
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _write_meta(wav_path, meta):
    meta_path(wav_path).write_text(
        json.dumps(meta, ensure_ascii=False) + "\n", encoding="utf-8",
    )


def align_processed(source, processed, destination):
    """Compensa el desfase de `processed` respecto de `source` y escribe en `destination`."""
    reference, sample_rate = sf.read(str(source), dtype="float32", always_2d=True)
    estimate, out_rate = sf.read(str(processed), dtype="float32", always_2d=True)
    if sample_rate != out_rate:
        raise ValueError("Frecuencia de muestreo distinta tras el preprocesamiento")
    if reference.shape[0] != estimate.shape[0]:
        raise ValueError("Número de muestras distinto tras el preprocesamiento")
    lag = estimate_lag_samples(reference, estimate, sample_rate)
    if lag is None:
        meta = {"desfase_muestras": 0, "compensado": False, "omitido": True}
        _write_wav_atomic(destination, estimate, sample_rate)
        _write_meta(destination, meta)
        return meta
    aligned = _shift_samples(estimate, lag)
    remaining = estimate_lag_samples(reference, aligned, sample_rate)
    limit = max(1, int(round(ALIGN_TOLERANCE_S * sample_rate)))
    if remaining is not None and abs(remaining) > limit:
        raise ValueError(
            f"Desfase residual tras compensar: {remaining} muestras "
            f"(límite {limit})"
        )
    meta = {
        "desfase_muestras": int(lag),
        "compensado": lag != 0,
        "omitido": False,
    }
    _write_wav_atomic(destination, aligned, sample_rate)
    _write_meta(destination, meta)
    return meta


def process(technique, source, destination, *, reuse=True):
    """Aplica una técnica a `source` y deja el WAV alineado en `destination`."""
    if technique == "none":
        raise ValueError("none no produce archivo procesado")
    apply = TECHNIQUES.get(technique)
    if apply is None:
        raise ValueError(f"Preprocesamiento desconocido: {technique}")
    source = Path(source)
    destination = Path(destination)
    input_duration = duration(source)
    sidecar = meta_path(destination)
    if reuse and destination.is_file() and sidecar.is_file():
        if abs(duration(destination) - input_duration) <= DURATION_TOLERANCE_S:
            return destination
        destination.unlink(missing_ok=True)
        sidecar.unlink(missing_ok=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_name = tempfile.mkstemp(
        prefix=".preproc-raw-", suffix=".wav", dir=destination.parent,
    )
    os.close(descriptor)
    raw = Path(raw_name)
    try:
        apply(source, raw)
        if not raw.is_file():
            raise RuntimeError(f"No se generó salida para {technique}")
        output_duration = duration(raw)
        if abs(output_duration - input_duration) > DURATION_TOLERANCE_S:
            raise ValueError(
                f"Duración distinta tras {technique}: "
                f"entrada {input_duration:.3f} s, salida {output_duration:.3f} s"
            )
        align_processed(source, raw, destination)
    except Exception:
        destination.unlink(missing_ok=True)
        sidecar.unlink(missing_ok=True)
        raise
    finally:
        raw.unlink(missing_ok=True)
    return destination


def ensure_cached(technique, source, uri, cache_root=CACHE_ROOT):
    destination = cached_path(technique, uri, cache_root)
    return process(technique, source, destination, reuse=True)
