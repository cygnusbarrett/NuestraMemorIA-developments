#!/usr/bin/env python3
"""Aplica una técnica de preprocesamiento: noisereduce | wpe | demucs."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

# WPE en dominio STFT (valores típicos del ejemplo oficial de nara_wpe).
WPE_STFT_SIZE = 512
WPE_STFT_SHIFT = 128
WPE_TAPS = 10
WPE_DELAY = 3
WPE_ITERATIONS = 5

TECHNIQUES = ("noisereduce", "wpe", "demucs")


def _read_wav(path: Path):
    samples, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    return samples, int(sample_rate)


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), samples, sample_rate, subtype="PCM_16")


def _match_length(samples: np.ndarray, length: int) -> np.ndarray:
    if samples.shape[0] == length:
        return samples
    if samples.shape[0] > length:
        return samples[:length]
    pad = np.zeros((length - samples.shape[0], samples.shape[1]), dtype=samples.dtype)
    return np.concatenate([samples, pad], axis=0)


def apply_noisereduce(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    import noisereduce as nr

    channels = []
    for index in range(samples.shape[1]):
        channels.append(nr.reduce_noise(y=samples[:, index], sr=sample_rate))
    return np.stack(channels, axis=1).astype(np.float32, copy=False)


def apply_wpe(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    from nara_wpe.utils import istft, stft
    from nara_wpe.wpe import wpe

    del sample_rate  # la STFT no depende de la tasa aquí
    # Entrada (canales, muestras) → STFT → (frecuencias, canales, marcos).
    observation = samples.T.astype(np.float64, copy=False)
    spectrum = stft(observation, size=WPE_STFT_SIZE, shift=WPE_STFT_SHIFT)
    spectrum = spectrum.transpose(2, 0, 1)
    enhanced = wpe(
        spectrum,
        taps=WPE_TAPS,
        delay=WPE_DELAY,
        iterations=WPE_ITERATIONS,
    )
    restored = istft(
        enhanced.transpose(1, 2, 0),
        size=WPE_STFT_SIZE,
        shift=WPE_STFT_SHIFT,
    )
    return restored.T.astype(np.float32, copy=False)


def _resample(waveform: torch.Tensor, orig_freq: int, new_freq: int) -> torch.Tensor:
    if orig_freq == new_freq:
        return waveform
    import torchaudio

    return torchaudio.functional.resample(waveform, orig_freq, new_freq)


def apply_demucs(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model("htdemucs")
    model.to(device)
    model.eval()

    waveform = torch.from_numpy(samples.T.copy())  # (canales, tiempo)
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    elif waveform.shape[0] > 2:
        waveform = waveform[:2]

    waveform = _resample(waveform, sample_rate, model.samplerate)
    ref = waveform.mean(0)
    std = ref.std().clamp_min(1e-8)
    waveform = (waveform - ref.mean()) / std
    with torch.no_grad():
        sources = apply_model(model, waveform[None], device=device)[0]
    vocals_index = list(model.sources).index("vocals")
    vocals = sources[vocals_index] * std + ref.mean()
    vocals = _resample(vocals.cpu(), model.samplerate, sample_rate)

    channels = samples.shape[1]
    if channels == 1:
        out = vocals.mean(0, keepdim=True)
    else:
        out = vocals[:channels]
        if out.shape[0] < channels:
            pad = out[:1].repeat(channels - out.shape[0], 1)
            out = torch.cat([out, pad], dim=0)
    return out.T.contiguous().numpy().astype(np.float32, copy=False)


def process(technique: str, source: Path, destination: Path) -> None:
    if technique not in TECHNIQUES:
        raise SystemExit(f"Técnica desconocida: {technique}")
    samples, sample_rate = _read_wav(source)
    length = samples.shape[0]
    if technique == "noisereduce":
        output = apply_noisereduce(samples, sample_rate)
    elif technique == "wpe":
        output = apply_wpe(samples, sample_rate)
    else:
        output = apply_demucs(samples, sample_rate)
    output = _match_length(np.asarray(output, dtype=np.float32), length)
    if output.shape[1] != samples.shape[1]:
        raise RuntimeError(
            f"Canales distintos tras {technique}: "
            f"entrada {samples.shape[1]}, salida {output.shape[1]}"
        )
    _write_wav(destination, output, sample_rate)


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 3:
        raise SystemExit(
            "Uso: run.py <noisereduce|wpe|demucs> <entrada.wav> <salida.wav>"
        )
    technique, source_name, destination_name = args
    process(technique, Path(source_name), Path(destination_name))


if __name__ == "__main__":
    main()
