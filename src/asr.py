from __future__ import annotations

import json
import os
from typing import List, Optional, Tuple

from faster_whisper import WhisperModel

from .models import TranscriptSegment, transcript_to_jsonable


def _ensure_dir(file_path: str) -> None:
    folder = os.path.dirname(file_path)
    if folder:
        os.makedirs(folder, exist_ok=True)


def _config_signature(
    model_id: str,
    device: str,
    compute_type: str,
    beam_size: int,
    language: Optional[str],
    task: str,
) -> str:
    lang = language or ""
    return f"{model_id}|{device}|{compute_type}|beam={beam_size}|lang={lang}|task={task}"


def _read_cached_segments(transcript_path: str) -> Tuple[Optional[str], Optional[List[TranscriptSegment]]]:
    if not os.path.exists(transcript_path):
        return None, None

    with open(transcript_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, list):
        return None, [TranscriptSegment.from_dict(x) for x in raw]

    if isinstance(raw, dict) and "segments" in raw:
        meta = raw.get("meta", {})
        sig = meta.get("signature")
        segments_raw = raw.get("segments", [])
        return sig, [TranscriptSegment.from_dict(x) for x in segments_raw]

    return None, None


def run_asr(
    audio_path: str,
    transcript_path: str,
    model_id: str = "large-v3-turbo",
    device: str = "cuda",
    compute_type: str = "int8",
    beam_size: int = 5,
    language: Optional[str] = None,
    task: str = "transcribe",
    force: bool = False,
    progress_cb=None,
) -> List[TranscriptSegment]:
    _ensure_dir(transcript_path)

    requested_sig = _config_signature(
        model_id=model_id,
        device=device,
        compute_type=compute_type,
        beam_size=beam_size,
        language=language,
        task=task,
    )

    cached_sig, cached_segments = _read_cached_segments(transcript_path)
    if not force and cached_segments is not None:
        if cached_sig is None or cached_sig == requested_sig:
            if progress_cb is not None:
                progress_cb(1, 1, "ASR: cached transcript loaded")
            print(f"Transcript file already exists: {transcript_path}")
            print(f"Loaded {len(cached_segments)} transcript segments.")
            return cached_segments

        print("Transcript cache does not match requested ASR configuration.")
        print(f"Cached:    {cached_sig}")
        print(f"Requested: {requested_sig}")
        print("Recomputing transcript...")

    print("Loading faster-whisper model …")
    model = WhisperModel(model_id, device=device, compute_type=compute_type)

    print(f"Transcribing audio: {audio_path}")
    segments_iter, info = model.transcribe(
        str(audio_path),
        beam_size=beam_size,
        language=language,
        task=task,
    )

    total_seconds = float(getattr(info, "duration", 0.0) or 0.0)

    segments: List[TranscriptSegment] = []
    for seg in segments_iter:
        segments.append(
            TranscriptSegment(
                start=float(seg.start),
                end=float(seg.end),
                text=seg.text.strip(),
            )
        )

        if progress_cb is not None:
            current = float(seg.end)
            total = total_seconds if total_seconds > 0 else 1.0
            if current > total:
                current = total
            progress_cb(current, total, f"ASR: {current:.1f}s / {total:.1f}s")

    payload = {
        "meta": {
            "signature": requested_sig,
            "model_id": model_id,
            "device": device,
            "compute_type": compute_type,
            "beam_size": beam_size,
            "language": language,
            "task": task,
        },
        "segments": transcript_to_jsonable(segments),
    }

    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    if progress_cb is not None:
        progress_cb(1, 1, "ASR: done")

    print(f"Saved transcript to: {transcript_path}")
    print(f"Number of transcript segments: {len(segments)}")

    return segments


def preview_transcript(segments: List[TranscriptSegment], n: int = 5) -> None:
    print(f"Total segments: {len(segments)}")
    for seg in segments[:n]:
        print(f"[{seg.start:.2f} -> {seg.end:.2f}] {seg.text}")
