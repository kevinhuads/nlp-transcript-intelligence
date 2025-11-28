from __future__ import annotations

import json
import os
from typing import List

from .models import TranscriptSegment, OCRRecord, Segment, segments_to_jsonable


def _precompute_ocr_times(ocr_records: List[OCRRecord]) -> List[float]:
    return [rec.time for rec in ocr_records]


def _find_nearest_ocr_index(
    ocr_times: List[float],
    target_time: float,
) -> int:
    if not ocr_times:
        return -1
    best_idx = 0
    best_diff = abs(ocr_times[0] - target_time)
    for i, t in enumerate(ocr_times[1:], start=1):
        diff = abs(t - target_time)
        if diff < best_diff:
            best_diff = diff
            best_idx = i
    return best_idx


def align_transcript_and_ocr(
    transcript_segments: List[TranscriptSegment],
    ocr_records: List[OCRRecord],
    segments_path: str,
) -> List[Segment]:
    segments_dir = os.path.dirname(segments_path)
    if segments_dir:
        os.makedirs(segments_dir, exist_ok=True)

    ocr_times = _precompute_ocr_times(ocr_records)
    segments_merged: List[Segment] = []

    for seg in transcript_segments:
        mid = 0.5 * (seg.start + seg.end)
        ocr_idx = _find_nearest_ocr_index(ocr_times, mid)

        if ocr_idx == -1:
            ocr_text = ""
            ocr_time = None
            ocr_frame = None
        else:
            ocr_rec = ocr_records[ocr_idx]
            ocr_text = ocr_rec.text
            ocr_time = ocr_rec.time
            ocr_frame = ocr_rec.frame

        segments_merged.append(
            Segment(
                start=seg.start,
                end=seg.end,
                mid=mid,
                speech=seg.text,
                slide_text=ocr_text,
                slide_time=ocr_time,
                slide_frame=ocr_frame,
            )
        )

    with open(segments_path, "w", encoding="utf-8") as f:
        json.dump(segments_to_jsonable(segments_merged), f, ensure_ascii=False, indent=2)

    print(f"Saved aligned multimodal segments to: {segments_path}")
    print(f"Total segments: {len(segments_merged)}")

    return segments_merged


def preview_segments(segments: List[Segment], n: int = 5) -> None:
    print(f"Aligned segments: {len(segments)}")
    for seg in segments[:n]:
        print(
            f"[{seg.start:.2f} -> {seg.end:.2f}] "
            f"(slide at {seg.slide_time}) speech='{seg.speech[:60]}...'"
        )
        print(f"Slide text: {seg.slide_text[:200]}")
        print("-" * 40)
