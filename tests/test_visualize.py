import wave
import re
import shutil
import subprocess

import imageio_ffmpeg
import numpy as np
import pytest
from pyannote.core import Annotation, Segment

from diarization.rttm import write_rttm
from diarization.visualize import crop_position, disagreement_spans, render

PNG = b"\x89PNG\r\n\x1a\n"


def write_tone(path, sample_rate=16000):
    silence = int(sample_rate * 0.5)
    count = int(sample_rate * 2.0)
    times = np.arange(count) / sample_rate
    tone = (0.5 * np.sin(2 * np.pi * 200.0 * times) * 32767).astype(np.int16)
    samples = np.concatenate([
        np.zeros(silence, dtype=np.int16), tone, np.zeros(silence, dtype=np.int16),
    ])
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(samples.tobytes())


def write_spans(path, spans, uri="tono"):
    annotation = Annotation(uri=uri)
    for index, (start, end, label) in enumerate(spans):
        annotation[Segment(start, end), index] = label
    write_rttm(annotation, path)


def test_synthetic_tone_png_and_pitch(tmp_path):
    audio, rttm, png = tmp_path / "tono.wav", tmp_path / "tono.rttm", tmp_path / "tono.png"
    write_tone(audio)
    write_spans(rttm, [(0.5, 2.5, "SPEAKER_00")])
    summary = render(audio, rttm, png)
    voiced = np.isfinite(summary["hz"])
    assert png.read_bytes()[:8] == PNG
    assert voiced.any()
    assert np.median(summary["hz"][voiced]) == pytest.approx(200.0, rel=0.05)
    assert summary["times"][voiced].min() >= 0.5 - 0.02
    assert summary["times"][voiced].max() <= 2.5 + 0.02
    render(audio, rttm, png)
    assert list(tmp_path.glob("*.png")) == [png]
    assert png.read_bytes()[:8] == PNG


def test_dry_run_does_not_write(tmp_path):
    audio, rttm, png = tmp_path / "tono.wav", tmp_path / "tono.rttm", tmp_path / "out" / "tono.png"
    write_tone(audio)
    write_spans(rttm, [(0.5, 2.5, "SPEAKER_00")])
    summary = render(audio, rttm, png, dry_run=True)
    assert summary["hablantes"] == 1
    assert summary["turnos"] == 1
    assert summary["duracion_s"] == pytest.approx(3.0, abs=0.01)
    assert "hz" not in summary
    assert not png.exists()


def test_disagreement_spans_cover_label_mismatch():
    hypothesis = Annotation(uri="ev99")
    hypothesis[Segment(0.0, 1.0), 0] = "A"
    reference = Annotation(uri="ev99")
    reference[Segment(0.0, 0.4), 0] = "A"
    reference[Segment(0.4, 1.0), 1] = "B"
    spans = disagreement_spans(hypothesis, reference, 0.0, 1.0, step=0.01)
    assert len(spans) == 1
    start, duration = spans[0]
    assert start == pytest.approx(0.4, abs=0.01)
    assert start + duration == pytest.approx(1.0, abs=0.01)


def test_reference_png(tmp_path):
    audio, hyp, ref, png = (
        tmp_path / "tono.wav", tmp_path / "hyp.rttm", tmp_path / "ref.rttm", tmp_path / "tono.png",
    )
    write_tone(audio)
    write_spans(hyp, [(0.5, 2.5, "SPEAKER_00")])
    write_spans(ref, [(0.5, 1.5, "SPEAKER_00"), (1.5, 2.5, "SPEAKER_01")])
    render(audio, hyp, png, ref=ref)
    assert png.read_bytes()[:8] == PNG


def test_tone_start_under_cursor():
    fps, pixels_per_second, window_pixels = 30, 100, 2000
    tone_start = 0.5
    frame = round(tone_start * fps)
    tone_pixel = window_pixels / 2 + tone_start * pixels_per_second
    position = tone_pixel - crop_position(frame / fps, pixels_per_second)
    assert abs(position - window_pixels / 2) <= pixels_per_second / fps


def test_synthetic_video_has_synchronized_tracks(tmp_path):
    audio, rttm = tmp_path / "tono.wav", tmp_path / "tono.rttm"
    png, video = tmp_path / "tono.png", tmp_path / "tono.mp4"
    write_tone(audio)
    write_spans(rttm, [(0.5, 2.5, "SPEAKER_00")])
    render(audio, rttm, png, video=video)
    ffmpeg = shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe()
    result = subprocess.run([ffmpeg, "-i", str(video)], capture_output=True, text=True)
    assert "Video: h264" in result.stderr
    assert "Audio: aac" in result.stderr
    duration = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", result.stderr)
    assert duration is not None
    hours, minutes, seconds = map(float, duration.groups())
    assert hours * 3600 + minutes * 60 + seconds == pytest.approx(3.0, abs=0.1)
