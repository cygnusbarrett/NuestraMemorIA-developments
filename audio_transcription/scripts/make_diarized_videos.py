#!/usr/bin/env python3
"""Build black-background subtitle videos colored by diarization speaker."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy import AudioFileClip, ColorClip, ImageClip, CompositeVideoClip

# Distinct colors on black (RGB)
SPEAKER_PALETTE = [
    (0, 229, 229),    # calipso
    (255, 107, 107),  # coral
    (255, 217, 61),   # yellow
    (107, 203, 119),  # green
    (255, 140, 200),  # pink
    (195, 177, 225),  # lavender
    (255, 159, 67),   # orange
    (116, 185, 255),  # sky
]
UNKNOWN_COLOR = (200, 200, 200)

W, H = 1920, 1080
FONT_SIZE = 64
MAX_TEXT_W = int(W * 0.82)
PAD_X, PAD_Y = 80, 80


def load_font(size: int = FONT_SIZE) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def speaker_color(speaker: str | None, mapping: dict[str, tuple[int, int, int]]) -> tuple[int, int, int]:
    if not speaker or speaker in {"UNKNOWN", "HABLANTE_DESCONOCIDO"}:
        return UNKNOWN_COLOR
    if speaker not in mapping:
        mapping[speaker] = SPEAKER_PALETTE[len(mapping) % len(SPEAKER_PALETTE)]
    return mapping[speaker]


def wrap_text(text: str, font, max_width: int) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines, cur = [], words[0]
    for w in words[1:]:
        trial = f"{cur} {w}"
        bbox = font.getbbox(trial)
        if bbox[2] - bbox[0] <= max_width:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return lines


def render_subtitle(text: str, color_rgb: tuple[int, int, int], font) -> np.ndarray:
    fill = (*color_rgb, 255)
    lines = wrap_text(text, font, MAX_TEXT_W)
    ascent, descent = font.getmetrics()
    line_gap = 18
    line_h = ascent + descent + line_gap

    tmp_w = MAX_TEXT_W + 2 * PAD_X + 200
    tmp_h = line_h * len(lines) + 2 * PAD_Y + 200
    tmp = Image.new("RGBA", (tmp_w, tmp_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(tmp)
    y_base = PAD_Y + ascent + 40
    for line in lines:
        bbox = font.getbbox(line)
        lw = bbox[2] - bbox[0]
        x = (tmp_w - lw) // 2 - bbox[0]
        draw.text((x, y_base), line, font=font, fill=fill, anchor="ls")
        y_base += line_h

    arr = np.array(tmp)
    alpha = arr[:, :, 3]
    ys, xs = np.where(alpha > 0)
    if len(xs) == 0:
        return arr
    y0, y1 = max(0, ys.min() - PAD_Y), min(tmp_h, ys.max() + PAD_Y + 1)
    x0, x1 = max(0, xs.min() - PAD_X), min(tmp_w, xs.max() + PAD_X + 1)
    return arr[y0:y1, x0:x1]


def cues_from_json(data: dict) -> list[tuple[float, float, str, str]]:
    cues = []
    for seg in data.get("segments", []):
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        start = float(seg.get("start") or 0.0)
        end = float(seg.get("end") or start)
        if end <= start:
            continue
        speaker = seg.get("speaker") or "UNKNOWN"
        # Prefer majority word-level speaker if present
        words = seg.get("words") or []
        word_speakers = [w.get("speaker") for w in words if w.get("speaker")]
        if word_speakers:
            speaker = max(set(word_speakers), key=word_speakers.count)
        cues.append((start, end, text, speaker))
    return cues


def write_txt_exports(cues, out_stem: Path, color_map: dict[str, tuple[int, int, int]]) -> None:
    plain = out_stem.with_name(out_stem.name + ".txt")
    ts = out_stem.with_name(out_stem.name + "_timestamps.txt")
    legend = out_stem.with_name(out_stem.name + "_speakers.txt")

    def fmt(t: float) -> str:
        ms = int(round(t * 1000))
        h, rem = divmod(ms, 3_600_000)
        m, rem = divmod(rem, 60_000)
        s, ms = divmod(rem, 1000)
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

    plain.write_text(
        "\n".join(f"[{spk}] {text}" for _, _, text, spk in cues) + "\n",
        encoding="utf-8",
    )
    ts.write_text(
        "\n".join(
            f"[{fmt(a)} --> {fmt(b)}] [{spk}] {text}" for a, b, text, spk in cues
        )
        + "\n",
        encoding="utf-8",
    )
    lines = []
    for spk, rgb in sorted(color_map.items()):
        hexcol = f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"
        lines.append(f"{spk}\t{hexcol}")
    legend.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def build_video(audio_path: Path, json_path: Path, out_mp4: Path, font) -> None:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    cues = cues_from_json(data)
    color_map: dict[str, tuple[int, int, int]] = {}
    for *_, spk in cues:
        speaker_color(spk, color_map)

    write_txt_exports(cues, out_mp4.with_suffix(""), color_map)

    audio = AudioFileClip(str(audio_path))
    duration = audio.duration
    bg = ColorClip(size=(W, H), color=(0, 0, 0)).with_duration(duration)
    clips = [bg]
    for start, end, text, spk in cues:
        rgb = speaker_color(spk, color_map)
        arr = render_subtitle(text, rgb, font)
        dur = min(max(0.05, end - start), max(0.05, duration - start))
        clips.append(
            ImageClip(arr, duration=dur).with_start(start).with_position("center")
        )

    video = CompositeVideoClip(clips, size=(W, H)).with_audio(audio)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    video.write_videofile(
        str(out_mp4),
        fps=24,
        codec="libx264",
        audio_codec="aac",
        preset="medium",
        threads=4,
        logger="bar",
    )
    audio.close()
    video.close()
    print(f"✅ video: {out_mp4.name} | speakers={sorted(color_map.keys())}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", required=True)
    parser.add_argument("--json_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    json_dir = Path(args.json_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    font = load_font()

    jsons = sorted(json_dir.glob("*_full.json"))
    if not jsons:
        raise SystemExit(f"No *_full.json in {json_dir}")

    for jp in jsons:
        stem = re.sub(r"_full$", "", jp.stem)
        # match audio: exact stem or stem without trailing markers
        candidates = [
            audio_dir / f"{stem}.wav",
            audio_dir / f"{stem}.mp3",
            audio_dir / f"{stem}.ogg",
        ]
        audio = next((c for c in candidates if c.exists()), None)
        if audio is None:
            print(f"⚠️  skip {jp.name}: audio not found for stem={stem}")
            continue
        out_mp4 = output_dir / f"{stem}_subtitulos.mp4"
        print(f"🎬 {audio.name} + {jp.name}")
        build_video(audio, jp, out_mp4, font)


if __name__ == "__main__":
    main()
