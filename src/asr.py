from __future__ import annotations

import json
import os
from typing import List

from faster_whisper import WhisperModel

from .models import TranscriptSegment, transcript_to_jsonable


def run_asr(
    audio_path: str,
    transcript_path: str,
    model_size: str = "small",
    device: str = "cuda",
    compute_type: str = "int8",
) -> List[TranscriptSegment]:
    transcript_dir = os.path.dirname(transcript_path)
    if transcript_dir:
        os.makedirs(transcript_dir, exist_ok=True)

    if os.path.exists(transcript_path):
        print(f"Transcript file already exists: {transcript_path}")
        with open(transcript_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        segments = [TranscriptSegment.from_dict(x) for x in raw]
        print(f"Loaded {len(segments)} transcript segments.")
        return segments

    print("Loading faster-whisper model …")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    print(f"Transcribing audio: {audio_path}")
    segments_iter, info = model.transcribe(str(audio_path), beam_size=5)

    segments: List[TranscriptSegment] = []
    for seg in segments_iter:
        segments.append(
            TranscriptSegment(
                start=float(seg.start),
                end=float(seg.end),
                text=seg.text.strip(),
            )
        )

    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump(transcript_to_jsonable(segments), f, ensure_ascii=False, indent=2)

    print(f"Saved transcript to: {transcript_path}")
    print(f"Number of transcript segments: {len(segments)}")

    return segments


def preview_transcript(segments: List[TranscriptSegment], n: int = 5) -> None:
    print(f"Total segments: {len(segments)}")
    for seg in segments[:n]:
        print(f"[{seg.start:.2f} -> {seg.end:.2f}] {seg.text}")
