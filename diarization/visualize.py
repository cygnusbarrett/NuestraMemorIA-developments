"""Dibuja espectrograma, F0 y franjas de hablante."""

import argparse
import ast
import math
import os
import re
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path

_MPLCONFIG = os.path.join(tempfile.gettempdir(), "mplconfig-diarization")
os.makedirs(_MPLCONFIG, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", _MPLCONFIG)
os.environ.setdefault("MPLBACKEND", "Agg")

import librosa  # noqa: E402
import numpy as np  # noqa: E402
import imageio_ffmpeg  # noqa: E402
import torchaudio  # noqa: E402
from pyannote.core import Segment  # noqa: E402

from diarization.compare import compare_annotations  # noqa: E402
from diarization.rttm import read_rttm  # noqa: E402

F0_LOW = 50
F0_HIGH = 500
FRAME_TIME = 0.01
MEL_FMIN = 50.0
MEL_FMAX = 8000.0
N_FFT = 4096
SPEC_BANDS = 256
SPEC_CMAP = "gray"
F0_POINT_SIZE = 10
F0_EDGECOLORS = "black"
F0_LINEWIDTHS = 0.3
URI_CODE = re.compile(r"ev\d{2,}")


def speaker_palette():
    # El módulo importa moviepy, que no está en este entorno.
    source = Path(__file__).resolve().parents[1] / "audio_transcription/scripts/make_diarized_videos.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "SPEAKER_PALETTE":
                colors = ast.literal_eval(node.value)
                if colors and all(len(color) == 3 for color in colors):
                    return [tuple(int(channel) for channel in color) for color in colors]
    raise RuntimeError("SPEAKER_PALETTE no encontrada")


def load_mono(path):
    path = Path(path)
    if not path.is_file():
        raise OSError("Audio no disponible")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
        waveform, sample_rate = torchaudio.load(path)
    if waveform.ndim != 2 or waveform.shape[-1] == 0 or sample_rate <= 0:
        raise ValueError("Audio inválido")
    if waveform.shape[0] != 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.contiguous(), int(sample_rate)


def clip_bounds(n_samples, sample_rate, start, end):
    duration = n_samples / sample_rate
    t0 = 0.0 if start is None else float(start)
    t1 = duration if end is None else float(end)
    if not math.isfinite(t0) or not math.isfinite(t1) or t0 < 0 or t1 <= t0 or t0 >= duration:
        raise ValueError("Ventana inválida")
    i0 = min(n_samples, max(0, int(round(t0 * sample_rate))))
    i1 = min(n_samples, max(0, int(round(min(t1, duration) * sample_rate))))
    if i1 <= i0:
        raise ValueError("Ventana inválida")
    return i0, i1


def tracks_of(annotation):
    return [
        (segment.start, segment.end, label)
        for segment, _, label in annotation.itertracks(yield_label=True)
    ]


def label_order(tracks):
    order = []
    for _, _, label in tracks:
        if label not in order:
            order.append(label)
    return order


def speakers_at(tracks, time):
    return {label for start, end, label in tracks if start <= time < end}


def color_map(labels, palette):
    return {
        label: tuple(channel / 255 for channel in palette[index % len(palette)])
        for index, label in enumerate(labels)
    }


def pitch_series(waveform, sample_rate, tracks, offset, fmin=F0_LOW, fmax=F0_HIGH):
    hop = max(1, round(sample_rate * 0.02))
    values, voiced, _ = librosa.pyin(
        waveform.squeeze(0).detach().cpu().numpy(), fmin=fmin, fmax=fmax,
        sr=sample_rate, hop_length=hop,
    )
    times = offset + librosa.frames_to_time(np.arange(len(values)), sr=sample_rate, hop_length=hop)
    hz = np.full(len(values), np.nan)
    labels = [None] * len(values)
    limit = offset + waveform.shape[-1] / sample_rate
    for index, time in enumerate(times):
        active = speakers_at(tracks, float(time))
        if time < limit and voiced[index] and len(active) == 1 and np.isfinite(values[index]):
            hz[index] = values[index]
            labels[index] = next(iter(active))
    return times, hz, labels


def disagreement_spans(hypothesis, reference, t0, t1):
    intervals = compare_annotations(hypothesis, reference, max_intervals=None)["tramos"]
    return [(max(item["inicio"], t0), min(item["fin"], t1) - max(item["inicio"], t0))
            for item in intervals if item["fin"] > t0 and item["inicio"] < t1]


def spectrogram(waveform, sample_rate, t0, n_bands=SPEC_BANDS):
    hop = max(1, round(sample_rate * FRAME_TIME))
    samples = waveform.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
    spectrum = np.abs(librosa.stft(samples, n_fft=N_FFT, hop_length=hop))
    frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=N_FFT)
    f_max = min(MEL_FMAX, sample_rate / 2.0)
    selected = (frequencies >= MEL_FMIN) & (frequencies <= f_max)
    magnitude = spectrum[selected]
    del spectrum
    freqs = frequencies[selected]
    bands = max(1, min(int(n_bands), magnitude.shape[0]))
    edges = np.geomspace(MEL_FMIN, f_max, bands + 1)
    band_mags = np.zeros((bands, magnitude.shape[1]), dtype=np.float32)
    band_freqs = np.empty(bands, dtype=np.float64)
    empty = []
    for index in range(bands):
        if index + 1 == bands:
            mask = (freqs >= edges[index]) & (freqs <= edges[index + 1])
        else:
            mask = (freqs >= edges[index]) & (freqs < edges[index + 1])
        band_freqs[index] = math.sqrt(edges[index] * edges[index + 1])
        if mask.any():
            band_mags[index] = magnitude[mask].mean(axis=0)
        else:
            empty.append(index)
    if empty:
        # Bandas sin bins STFT: magnitud en el centro log, cuadro a cuadro.
        centers = band_freqs[np.asarray(empty, dtype=int)]
        right = np.clip(np.searchsorted(freqs, centers, side="left"), 1, len(freqs) - 1)
        left = right - 1
        span = freqs[right] - freqs[left]
        weight = np.where(span > 0, (centers - freqs[left]) / span, 0.0)
        band_mags[empty] = (
            (1.0 - weight)[:, None] * magnitude[left] + weight[:, None] * magnitude[right]
        ).astype(np.float32, copy=False)
    del magnitude
    spec = librosa.amplitude_to_db(band_mags, ref=np.max, top_db=80)
    times = t0 + librosa.frames_to_time(np.arange(spec.shape[1]), sr=sample_rate, hop_length=hop)
    return times, band_freqs, spec


def _pyplot():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _spans(tracks, label, t0, t1):
    spans = []
    for start, end, current in tracks:
        if current != label:
            continue
        start, end = max(start, t0), min(end, t1)
        if end > start:
            spans.append((start, end - start))
    return spans


def draw(path, waveform, sample_rate, t0, hypothesis, reference, times, hz, labels,
         spec_data, ref_label="referencia"):
    plt = _pyplot()
    spec_times, frequencies, spec = spec_data
    t1 = t0 + waveform.shape[-1] / sample_rate

    hyp_tracks = tracks_of(hypothesis)
    ref_tracks = tracks_of(reference) if reference is not None else []
    order = label_order(hyp_tracks)
    for label in label_order(ref_tracks):
        if label not in order:
            order.append(label)
    colors = color_map(order, speaker_palette())
    rows = list(label_order(hyp_tracks))
    if reference is not None:
        rows.append(ref_label)
    width = min(16.0, max(8.0, (t1 - t0) / 15.0))
    height = 4.4 + 0.42 * max(len(rows), 1)
    fig = plt.figure(figsize=(width, height), layout="constrained")
    try:
        if rows:
            grid = fig.add_gridspec(2, 1, height_ratios=[4.4, 0.42 * len(rows)])
            spec_ax = fig.add_subplot(grid[0])
            stripe_ax = fig.add_subplot(grid[1], sharex=spec_ax)
        else:
            spec_ax = fig.add_subplot(111)
            stripe_ax = None
        spec_ax.pcolormesh(
            spec_times, frequencies, spec, shading="nearest", cmap=SPEC_CMAP, rasterized=True,
        )
        spec_ax.set_yscale("log")
        spec_ax.set_ylim(MEL_FMIN, min(MEL_FMAX, sample_rate / 2))
        spec_ax.set_ylabel("Hz")
        voiced = np.isfinite(hz)
        if voiced.any():
            spec_ax.scatter(
                times[voiced], hz[voiced],
                c=[colors[label] for label, keep in zip(labels, voiced) if keep],
                s=F0_POINT_SIZE, edgecolors=F0_EDGECOLORS, linewidths=F0_LINEWIDTHS,
                rasterized=True, zorder=3,
            )
        if URI_CODE.fullmatch(hypothesis.uri or ""):
            spec_ax.set_title(hypothesis.uri)
        spec_ax.set_xlim(t0, t1)
        if stripe_ax is None:
            spec_ax.set_xlabel("tiempo (s)")
        else:
            spec_ax.tick_params(labelbottom=False)
            for index, name in enumerate(rows):
                if reference is not None and index == len(rows) - 1:
                    for label in label_order(ref_tracks):
                        spans = _spans(ref_tracks, label, t0, t1)
                        if spans:
                            stripe_ax.broken_barh(
                                spans, (index + 0.15, 0.7), facecolors=colors[label],
                            )
                    for start, duration in disagreement_spans(hypothesis, reference, t0, t1):
                        stripe_ax.broken_barh(
                            [(start, duration)], (index + 0.15, 0.7),
                            facecolors=(0.8, 0.0, 0.0), zorder=3,
                        )
                else:
                    spans = _spans(hyp_tracks, name, t0, t1)
                    if spans:
                        stripe_ax.broken_barh(spans, (index + 0.15, 0.7), facecolors=colors[name])
            stripe_ax.set_ylim(len(rows), 0)
            stripe_ax.set_yticks([index + 0.5 for index in range(len(rows))])
            stripe_ax.set_yticklabels(rows)
            stripe_ax.set_xlabel("tiempo (s)")
            stripe_ax.set_xlim(t0, t1)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=".viz-", suffix=".png", dir=path.parent)
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            fig.savefig(temporary, dpi=120, facecolor="white")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    finally:
        plt.close(fig)


def crop_position(time, px_per_s):
    return time * px_per_s


def draw_video_images(directory, waveform, sample_rate, t0, hypothesis, reference,
                      times, hz, labels, px_per_s, spec_data, ref_label="referencia"):
    plt = _pyplot()
    duration = waveform.shape[-1] / sample_rate
    width = max(1, math.ceil(duration * px_per_s))
    spec_times, frequencies, spec = spec_data
    hyp_tracks = tracks_of(hypothesis)
    ref_tracks = tracks_of(reference) if reference is not None else []
    rows = label_order(hyp_tracks)
    order = rows + [name for name in label_order(ref_tracks) if name not in rows]
    colors = color_map(order, speaker_palette())
    if reference is not None:
        rows = rows + [ref_label]
    fig = plt.figure(figsize=(width / 100, 7.2), dpi=100, facecolor="black")
    try:
        ax = fig.add_axes([0, 0.34, 1, 0.64])
        ax.pcolormesh(spec_times, frequencies, spec,
                      shading="nearest", cmap=SPEC_CMAP, rasterized=True)
        ax.set_yscale("log")
        ax.set_ylim(MEL_FMIN, min(MEL_FMAX, sample_rate / 2))
        ax.set_xlim(t0, t0 + duration)
        ax.set_axis_off()
        voiced = np.isfinite(hz)
        if voiced.any():
            ax.scatter(times[voiced], hz[voiced],
                       c=[colors[label] for label, keep in zip(labels, voiced) if keep],
                       s=F0_POINT_SIZE, edgecolors=F0_EDGECOLORS, linewidths=F0_LINEWIDTHS,
                       rasterized=True)
        stripes = fig.add_axes([0, 0.08, 1, 0.23])
        for index, name in enumerate(rows):
            is_ref = reference is not None and index == len(rows) - 1
            tracks = ref_tracks if is_ref else hyp_tracks
            names = label_order(tracks) if is_ref else [name]
            for label in names:
                stripes.broken_barh(_spans(tracks, label, t0, t0 + duration),
                                    (index + 0.15, 0.7), facecolors=colors[label])
            if reference is not None and index == len(rows) - 1:
                stripes.broken_barh(disagreement_spans(hypothesis, reference, t0, t0 + duration),
                                    (index + 0.15, 0.7), facecolors="red")
        stripes.set_xlim(t0, t0 + duration)
        stripes.set_ylim(max(len(rows), 1), 0)
        stripes.set_axis_off()
        for second in range(math.ceil(t0 / 5) * 5, math.ceil(t0 + duration), 5):
            x = (second - t0) / duration
            fig.add_artist(plt.Line2D([x, x], [0.055, 0.075], transform=fig.transFigure, color="white"))
            fig.text(x, 0.025, f"{second // 60:02d}:{second % 60:02d}",
                     color="white", fontsize=10, ha="center")
        fig.savefig(directory / "content.png", dpi=100, facecolor="black")
    finally:
        plt.close(fig)

    side = plt.figure(figsize=(2, 7.2), dpi=100, facecolor="black")
    try:
        for frequency in (50, 60, 100, 200, 400, 1000, 2000, 4000, 8000):
            if frequency > sample_rate / 2:
                continue
            y = 0.34 + 0.64 * math.log(frequency / MEL_FMIN) / math.log(min(MEL_FMAX, sample_rate / 2) / MEL_FMIN)
            side.text(0.95, y, f"{frequency} Hz", color="white", ha="right", va="center", fontsize=9)
        for index, name in enumerate(rows):
            y = 0.31 - 0.23 * (index + 0.5) / max(len(rows), 1)
            side.text(0.95, y, name, color=colors.get(name, "white"), ha="right", va="center", fontsize=9)
        side.savefig(directory / "side.png", dpi=100, facecolor="black")
    finally:
        plt.close(side)


def render_video(audio, target, waveform, sample_rate, t0, hypothesis, reference,
                 times, hz, labels, spec_data, window=20, px_per_s=100, fps=30,
                 force=False, ref_label="referencia"):
    if not all(math.isfinite(value) and value > 0 for value in (window, px_per_s, fps)):
        raise ValueError("Parámetros de video inválidos")
    window_pixels = int(round(window * px_per_s / 2)) * 2
    if window_pixels < 2:
        raise ValueError("Ventana de video demasiado pequeña")
    target = Path(target)
    if target.exists() and not force:
        raise ValueError("El video ya existe")
    target.parent.mkdir(parents=True, exist_ok=True)
    duration = waveform.shape[-1] / sample_rate
    with tempfile.TemporaryDirectory(prefix="diarization-video-") as temp:
        directory = Path(temp)
        draw_video_images(directory, waveform, sample_rate, t0, hypothesis, reference,
                          times, hz, labels, px_per_s, spec_data, ref_label)
        content_width = max(1, math.ceil(duration * px_per_s))
        filters = (
            f"[0:v]pad={content_width + window_pixels}:720:{window_pixels // 2}:0:black,"
            f"crop={window_pixels}:720:x='t*{crop_position(1, px_per_s)}':y=0[scroll];"
            f"[1:v][scroll]hstack=inputs=2,"
            f"drawbox=x={200 + window_pixels // 2}:y=0:w=2:h=720:color=white:t=fill,"
            "scale=w='max(1280,iw)':h=-2,format=yuv420p[v]"
        )
        descriptor, temp_video = tempfile.mkstemp(prefix=".viz-", suffix=".mp4", dir=target.parent)
        os.close(descriptor)
        temporary = Path(temp_video)
        command = [shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe(),
                   "-nostdin", "-v", "error", "-loop", "1", "-framerate", str(fps),
                   "-i", str(directory / "content.png"), "-loop", "1", "-framerate", str(fps),
                   "-i", str(directory / "side.png"), "-ss", str(t0), "-i", str(audio),
                   "-filter_complex", filters, "-map", "[v]", "-map", "2:a:0", "-t", str(duration),
                   "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                   "-c:a", "aac", "-movflags", "+faststart",
                   "-y", str(temporary)]
        try:
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as error:
            raise RuntimeError("No se pudo generar el video") from error
        else:
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)


def render(audio, rttm, out, ref=None, start=None, end=None, dry_run=False,
           video=None, window=20, px_per_s=100, fps=30, force=False, ref_label="referencia"):
    waveform, sample_rate = load_mono(audio)
    i0, i1 = clip_bounds(waveform.shape[-1], sample_rate, start, end)
    t0, t1 = i0 / sample_rate, i1 / sample_rate
    segment = Segment(t0, t1)
    hypothesis = read_rttm(rttm).crop(segment)
    reference = read_rttm(ref).crop(segment) if ref else None
    hyp_tracks = tracks_of(hypothesis)
    summary = {
        "duracion_s": t1 - t0,
        "hablantes": len(label_order(hyp_tracks)),
        "turnos": len(hyp_tracks),
    }
    if dry_run:
        return summary
    if video is not None and Path(video).exists() and not force:
        raise ValueError("El video ya existe")
    clip = waveform[:, i0:i1]
    times, hz, labels = pitch_series(clip, sample_rate, hyp_tracks, t0)
    spec_data = spectrogram(clip, sample_rate, t0)
    draw(out, clip, sample_rate, t0, hypothesis, reference, times, hz, labels,
         spec_data, ref_label)
    if video is not None:
        render_video(audio, video, clip, sample_rate, t0, hypothesis, reference,
                     times, hz, labels, spec_data, window, px_per_s, fps, force, ref_label)
    summary["times"] = times
    summary["hz"] = hz
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--rttm", type=Path, required=True)
    parser.add_argument("--ref", type=Path)
    parser.add_argument("--ref-label", default="referencia")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start", type=float)
    parser.add_argument("--end", type=float)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--video", type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--window", type=float, default=20)
    parser.add_argument("--px-per-s", type=float, default=100)
    parser.add_argument("--fps", type=float, default=30)
    args = parser.parse_args()
    try:
        summary = render(
            args.audio, args.rttm, args.out, ref=args.ref,
            start=args.start, end=args.end, dry_run=args.dry_run,
            video=args.video, window=args.window, px_per_s=args.px_per_s, fps=args.fps,
            force=args.force, ref_label=args.ref_label,
        )
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Error: {type(error).__name__}; revise el audio y los RTTM.\n")
    prefix = "simulacion " if args.dry_run else ""
    print(
        f"{prefix}duracion_s={summary['duracion_s']:.3f} "
        f"hablantes={summary['hablantes']} turnos={summary['turnos']}"
    )


if __name__ == "__main__":
    main()
