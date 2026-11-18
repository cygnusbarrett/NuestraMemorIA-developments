#!/usr/bin/env python3
"""Dual-GPU WhisperX batch worker for Juicio por la Verdad AV corpus.

Claims files atomically from a shared queue; writes JSON, CSV, and two
1-minute TXT outputs (with / without diarization). Never writes empty
success stubs on failure.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import gc
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
SCRIPTS_DIR = CURRENT_FILE.parent
MODULE_DIR = SCRIPTS_DIR.parent
PROJECT_ROOT = MODULE_DIR.parent

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

import pandas as pd
import torch
import whisper
import whisperx
from whisperx.diarize import DiarizationPipeline, assign_word_speakers

sys.path.insert(0, str(SCRIPTS_DIR))
from transcribe import (  # noqa: E402
    setup_ffmpeg_wrapper,
    setup_hardware_precision,
    setup_local_nltk,
)


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def setup_logger(worker_id: int, log_path: Path) -> logging.Logger:
    logger = logging.getLogger(f"jxv-worker-{worker_id}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter(
        f"%(asctime)s [W{worker_id}] %(levelname)s %(message)s"
    )
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def fmt_mmss_minute(minute: int) -> str:
    return f"[{minute:02d}:00]"


def segments_to_minute_blocks(segments: list, with_speakers: bool) -> str:
    """Bucket segment text into 1-minute windows."""
    buckets: dict[int, list[str]] = {}
    for seg in segments or []:
        start = seg.get("start")
        if start is None:
            continue
        minute = int(float(start) // 60)
        text = (seg.get("text") or "").strip()
        if not text:
            words = seg.get("words") or []
            text = " ".join((w.get("word") or "").strip() for w in words).strip()
        if not text:
            continue
        if with_speakers:
            spk = seg.get("speaker") or "SPEAKER_UNKNOWN"
            if spk == "SPEAKER_UNKNOWN" and seg.get("words"):
                for w in seg["words"]:
                    if w.get("speaker"):
                        spk = w["speaker"]
                        break
            line = f"{spk}: {text}"
        else:
            line = text
        buckets.setdefault(minute, []).append(line)

    if not buckets:
        return ""

    lines_out: list[str] = []
    for minute in sorted(buckets):
        lines_out.append(fmt_mmss_minute(minute))
        lines_out.append(" ".join(buckets[minute]).strip())
        lines_out.append("")
    return "\n".join(lines_out).rstrip() + "\n"


def rel_out_dir(src_root: Path, media: Path, out_root: Path) -> Path:
    try:
        rel = media.relative_to(src_root)
    except ValueError:
        rel = Path(media.name)
    dest = out_root / rel.parent / media.stem
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def outputs_complete(dest: Path) -> bool:
    base = dest.name
    required = [
        dest / f"{base}_full.json",
        dest / f"{base}_words.csv",
        dest / f"{base}_por_minuto.txt",
        dest / f"{base}_por_minuto_diarizado.txt",
    ]
    for p in required:
        if not p.exists() or p.stat().st_size == 0:
            return False
    try:
        data = json.loads(required[0].read_text(encoding="utf-8"))
    except Exception:
        return False
    if data.get("status") == "no_speech":
        return True
    if not (data.get("segments") or []):
        return False
    return True


def update_progress(batch_dir: Path, event: dict) -> None:
    status_path = batch_dir / "status.json"
    progress_log = batch_dir / "progress.log"
    lock_path = batch_dir / "status.lock"
    with open(lock_path, "a+", encoding="utf-8") as lf:
        fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
        try:
            if status_path.exists():
                status = json.loads(status_path.read_text(encoding="utf-8"))
            else:
                status = {
                    "total": 0,
                    "done": 0,
                    "failed": 0,
                    "skipped": 0,
                    "in_progress": {},
                    "started_at": utcnow(),
                }
            kind = event.get("kind")
            if kind == "init":
                status["total"] = event["total"]
                status["started_at"] = status.get("started_at") or utcnow()
            elif kind == "start":
                status.setdefault("in_progress", {})[event["worker"]] = {
                    "file": event["file"],
                    "started_at": utcnow(),
                }
            elif kind == "done":
                status["done"] = int(status.get("done", 0)) + 1
                status.get("in_progress", {}).pop(event["worker"], None)
            elif kind == "skip":
                status["skipped"] = int(status.get("skipped", 0)) + 1
                status.get("in_progress", {}).pop(event["worker"], None)
            elif kind == "fail":
                status["failed"] = int(status.get("failed", 0)) + 1
                status.get("in_progress", {}).pop(event["worker"], None)
                fails = status.setdefault("failures", [])
                fails.append(
                    {
                        "file": event["file"],
                        "error": event.get("error", ""),
                        "at": utcnow(),
                    }
                )
            total = max(1, int(status.get("total", 1)))
            finished = int(status.get("done", 0)) + int(status.get("skipped", 0))
            pct = 100.0 * finished / total
            status["percent"] = round(pct, 2)
            status["updated_at"] = utcnow()
            status_path.write_text(
                json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            line = (
                f"{utcnow()} pct={status['percent']:.2f}% "
                f"done={status.get('done', 0)} skip={status.get('skipped', 0)} "
                f"fail={status.get('failed', 0)} total={total} "
                f"event={kind} worker={event.get('worker')} file={event.get('file', '')}"
            )
            if event.get("error"):
                line += f" error={event['error'][:200]}"
            with progress_log.open("a", encoding="utf-8") as pf:
                pf.write(line + "\n")
        finally:
            fcntl.flock(lf.fileno(), fcntl.LOCK_UN)


def claim_next(queue_path: Path, claimed_path: Path, worker: str) -> Path | None:
    lock_path = queue_path.with_suffix(".lock")
    with open(lock_path, "a+", encoding="utf-8") as lf:
        fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
        try:
            lines = [
                ln.strip()
                for ln in queue_path.read_text(encoding="utf-8").splitlines()
                if ln.strip()
            ]
            if not lines:
                return None
            nxt = lines[0]
            rest = lines[1:]
            queue_path.write_text(
                ("\n".join(rest) + "\n") if rest else "", encoding="utf-8"
            )
            with claimed_path.open("a", encoding="utf-8") as cf:
                cf.write(f"{utcnow()}\t{worker}\t{nxt}\n")
            return Path(nxt)
        finally:
            fcntl.flock(lf.fileno(), fcntl.LOCK_UN)


def extract_wav(media: Path, wav_path: Path, logger: logging.Logger) -> float:
    """Extract mono 16k wav; return duration seconds."""
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(media),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(wav_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not wav_path.exists() or wav_path.stat().st_size < 100:
        raise RuntimeError(
            f"ffmpeg extract failed: {proc.stderr.strip() or proc.stdout.strip() or 'empty wav'}"
        )
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            str(wav_path),
        ],
        capture_output=True,
        text=True,
    )
    try:
        dur = float(probe.stdout.strip())
    except Exception:
        dur = 0.0
    logger.info(f"extracted wav duration={dur:.2f}s size={wav_path.stat().st_size}")
    return dur


class BatchPipeline:
    def __init__(self, args, logger: logging.Logger):
        self.args = args
        self.logger = logger
        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.hf_token = os.getenv("HUGGING_FACE_TOKEN")
        if not self.hf_token:
            raise RuntimeError("HUGGING_FACE_TOKEN missing — diarization required")
        self.logger.info(
            f"device={self.device} model={args.model_size} lang={args.language}"
        )

    def _clear_vram(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def transcribe(self, audio_path: str):
        self.logger.info("[1/3] ASR")
        model = whisper.load_model(self.args.model_size, device=self.device)
        try:
            result = model.transcribe(
                audio_path,
                temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                best_of=5,
                beam_size=5,
                patience=1.0,
                length_penalty=1.0,
                verbose=False,
                fp16=False,
                word_timestamps=True,
                language=self.args.language,
            )
        finally:
            del model
            self._clear_vram()
        return result

    def align(self, asr_result, audio_path: str):
        self.logger.info("[2/3] align")
        audio = whisperx.load_audio(audio_path)
        # Drop empty / near-zero segments that crash whisperx (trellis.size==1 → /0)
        segs = []
        for s in asr_result.get("segments") or []:
            text = (s.get("text") or "").strip()
            try:
                start = float(s.get("start") or 0.0)
                end = float(s.get("end") or 0.0)
            except (TypeError, ValueError):
                continue
            if not text or end - start < 0.02:
                continue
            segs.append(s)
        if not segs:
            self.logger.warning("no alignable segments — using empty aligned result")
            return {"segments": []}, audio

        align_model, align_metadata = whisperx.load_align_model(
            language_code=self.args.language, device=self.device
        )
        try:
            try:
                aligned = whisperx.align(
                    segs,
                    align_model,
                    align_metadata,
                    audio,
                    self.device,
                    return_char_alignments=False,
                )
            except ZeroDivisionError:
                # Known whisperx bug on pathological short trellis; keep ASR timestamps
                self.logger.warning(
                    "whisperx.align ZeroDivisionError — falling back to ASR word timestamps"
                )
                aligned = {"segments": segs}
        finally:
            del align_model, align_metadata
            self._clear_vram()
        return aligned, audio

    def diarize(self, aligned_result, audio):
        self.logger.info("[3/3] diarize")
        diarize_model = DiarizationPipeline(
            use_auth_token=self.hf_token, device=self.device
        )
        try:
            diarize_segments = diarize_model(audio)
            final = assign_word_speakers(diarize_segments, aligned_result)
        finally:
            del diarize_model
            self._clear_vram()
        return final

    def save(self, final_result, raw_whisper, dest: Path, duration: float, no_speech: bool = False):
        base = dest.name
        json_path = dest / f"{base}_full.json"
        csv_path = dest / f"{base}_words.csv"
        txt_path = dest / f"{base}_por_minuto.txt"
        txt_d_path = dest / f"{base}_por_minuto_diarizado.txt"

        if no_speech:
            payload = {
                "status": "no_speech",
                "duration": duration,
                "segments": [],
                "note": "Audio extractable but no speech segments from ASR",
            }
            json_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            pd.DataFrame(
                columns=[
                    "start",
                    "end",
                    "word",
                    "alignment_score",
                    "probability",
                    "speaker",
                ]
            ).to_csv(csv_path, index=False)
            marker = f"[00:00]\n[sin habla detectable — duración {duration:.1f}s]\n"
            txt_path.write_text(marker, encoding="utf-8")
            txt_d_path.write_text(marker, encoding="utf-8")
            return

        raw_words_flat = [
            w
            for s in raw_whisper.get("segments", [])
            for w in s.get("words", [])
        ]
        raw_idx = 0
        for seg in final_result.get("segments", []):
            for w in seg.get("words", []):
                if raw_idx < len(raw_words_flat):
                    w["probability"] = raw_words_flat[raw_idx].get("probability", 0.0)
                    raw_idx += 1
                else:
                    w["probability"] = 0.0
                if "score" in w:
                    w["alignment_score"] = w.pop("score")

        final_result = dict(final_result)
        final_result["status"] = "ok"
        final_result["duration"] = duration

        tmp_json = json_path.with_suffix(".json.tmp")
        tmp_json.write_text(
            json.dumps(final_result, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        rows = []
        for seg in final_result.get("segments", []):
            spk = seg.get("speaker", "UNKNOWN")
            for w in seg.get("words", []):
                rows.append(
                    {
                        "start": w.get("start"),
                        "end": w.get("end"),
                        "word": (w.get("word") or "").strip(),
                        "alignment_score": f"{w.get('alignment_score', 0.0):.4f}",
                        "probability": f"{w.get('probability', 0.0):.4f}",
                        "speaker": spk,
                    }
                )
        tmp_csv = csv_path.with_suffix(".csv.tmp")
        pd.DataFrame(rows).to_csv(tmp_csv, index=False)

        txt_plain = segments_to_minute_blocks(
            final_result.get("segments", []), with_speakers=False
        )
        txt_diar = segments_to_minute_blocks(
            final_result.get("segments", []), with_speakers=True
        )
        if not txt_plain.strip():
            raise RuntimeError(
                "empty por_minuto transcript after successful ASR — refusing empty output"
            )
        if not txt_diar.strip():
            txt_diar = segments_to_minute_blocks(
                [
                    {**s, "speaker": s.get("speaker") or "SPEAKER_UNKNOWN"}
                    for s in final_result.get("segments", [])
                ],
                with_speakers=True,
            )

        tmp_txt = txt_path.with_suffix(".txt.tmp")
        tmp_txtd = txt_d_path.with_suffix(".txt.tmp")
        tmp_txt.write_text(txt_plain, encoding="utf-8")
        tmp_txtd.write_text(txt_diar, encoding="utf-8")

        os.replace(tmp_json, json_path)
        os.replace(tmp_csv, csv_path)
        os.replace(tmp_txt, txt_path)
        os.replace(tmp_txtd, txt_d_path)

        for p in (json_path, csv_path, txt_path, txt_d_path):
            if p.stat().st_size == 0:
                raise RuntimeError(f"empty output written: {p}")

    def process_one(self, media: Path, src_root: Path, out_root: Path):
        dest = rel_out_dir(src_root, media, out_root)
        if outputs_complete(dest):
            return "skip"
        for p in dest.glob("*.tmp"):
            p.unlink(missing_ok=True)

        with tempfile.TemporaryDirectory(prefix="jxv_wav_") as td:
            wav = Path(td) / "audio.wav"
            duration = extract_wav(media, wav, self.logger)
            if duration < 0.3:
                self.save({}, {"segments": []}, dest, duration, no_speech=True)
                return "done"

            asr = self.transcribe(str(wav))
            segs = asr.get("segments") or []
            if not segs:
                self.save({}, asr, dest, duration, no_speech=True)
                return "done"

            asr_for_align = copy.deepcopy(asr)
            aligned, audio_obj = self.align(asr_for_align, str(wav))
            try:
                final = self.diarize(aligned, audio_obj)
            except Exception as e:
                self.logger.error(
                    f"diarization failed, writing aligned-only with UNKNOWN speakers: {e}"
                )
                final = aligned
                for seg in final.get("segments", []):
                    seg.setdefault("speaker", "SPEAKER_UNKNOWN")

            self.save(final, asr, dest, duration, no_speech=False)
        return "done"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--src-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--batch-dir", required=True)
    parser.add_argument(
        "--model-size", default=os.getenv("DEFAULT_MODEL_SIZE", "large-v3")
    )
    parser.add_argument(
        "--language", default=os.getenv("DEFAULT_LANGUAGE", "es")
    )
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)

    setup_local_nltk()
    setup_ffmpeg_wrapper()
    setup_hardware_precision()

    batch_dir = Path(args.batch_dir)
    log_path = batch_dir / f"worker_{args.worker_id}.log"
    logger = setup_logger(args.worker_id, log_path)
    queue_path = batch_dir / "queue.txt"
    claimed_path = batch_dir / "claimed.txt"
    src_root = Path(args.src_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    pipe = BatchPipeline(args, logger)
    worker = f"gpu{args.gpu_index}-w{args.worker_id}"
    logger.info(f"worker start {worker}")

    while True:
        media = claim_next(queue_path, claimed_path, worker)
        if media is None:
            logger.info("queue empty — exiting")
            break
        update_progress(
            batch_dir, {"kind": "start", "worker": worker, "file": str(media)}
        )
        t0 = time.time()
        try:
            if not media.exists():
                raise FileNotFoundError(str(media))
            result = pipe.process_one(media, src_root, out_root)
            elapsed = time.time() - t0
            if result == "skip":
                update_progress(
                    batch_dir,
                    {"kind": "skip", "worker": worker, "file": str(media)},
                )
                logger.info(f"SKIP {media} ({elapsed:.1f}s)")
            else:
                update_progress(
                    batch_dir,
                    {"kind": "done", "worker": worker, "file": str(media)},
                )
                logger.info(f"DONE {media} ({elapsed:.1f}s)")
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            logger.error(f"FAIL {media}: {err}\n{traceback.format_exc()}")
            update_progress(
                batch_dir,
                {
                    "kind": "fail",
                    "worker": worker,
                    "file": str(media),
                    "error": err,
                },
            )
            pipe._clear_vram()

    logger.info("worker finished")


if __name__ == "__main__":
    main()
