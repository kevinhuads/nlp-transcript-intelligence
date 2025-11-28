import json
import os

from src.align import (
    _precompute_ocr_times,
    _find_nearest_ocr_index,
    align_transcript_and_ocr,
    preview_segments,
)
from src.models import TranscriptSegment, OCRRecord


def test_precompute_ocr_times_returns_times_in_order():
    records = [
        OCRRecord(time=1.0, frame="f1.jpg", text="A"),
        OCRRecord(time=2.5, frame="f2.jpg", text="B"),
        OCRRecord(time=4.0, frame="f3.jpg", text="C"),
    ]
    times = _precompute_ocr_times(records)
    assert times == [1.0, 2.5, 4.0]


def test_find_nearest_ocr_index_normal_case():
    ocr_times = [1.0, 3.0, 10.0]

    # Target 2.1 is closer to 3.0 than to 1.0
    idx = _find_nearest_ocr_index(ocr_times, 2.1)
    assert idx == 1

    # Target very close to first
    idx2 = _find_nearest_ocr_index(ocr_times, 1.1)
    assert idx2 == 0

    # Target very close to last
    idx3 = _find_nearest_ocr_index(ocr_times, 9.9)
    assert idx3 == 2



def test_find_nearest_ocr_index_empty_list():
    assert _find_nearest_ocr_index([], 2.0) == -1


def test_align_transcript_and_ocr_creates_segments_and_file(tmp_path):
    base_dir = str(tmp_path)
    segments_path = os.path.join(base_dir, "out", "segments.json")

    transcript_segments = [
        TranscriptSegment(start=0.0, end=4.0, text="first segment"),
        TranscriptSegment(start=4.0, end=8.0, text="second segment"),
    ]
    ocr_records = [
        OCRRecord(time=0.0, frame="frame_00000.jpg", text="slide 0"),
        OCRRecord(time=3.0, frame="frame_00001.jpg", text="slide 1"),
        OCRRecord(time=7.0, frame="frame_00002.jpg", text="slide 2"),
    ]

    segments = align_transcript_and_ocr(
        transcript_segments=transcript_segments,
        ocr_records=ocr_records,
        segments_path=segments_path,
    )

    assert os.path.exists(segments_path)
    assert len(segments) == 2

    # Check midpoints and alignment
    assert segments[0].mid == 2.0
    # Nearest time to 2.0 is 3.0 (index 1)
    assert segments[0].slide_text == "slide 1"
    assert segments[0].slide_time == 3.0
    assert segments[0].slide_frame == "frame_00001.jpg"

    assert segments[1].mid == 6.0
    # Nearest time to 6.0 is 7.0 (index 2)
    assert segments[1].slide_text == "slide 2"
    assert segments[1].slide_time == 7.0
    assert segments[1].slide_frame == "frame_00002.jpg"

    with open(segments_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list)
    assert len(data) == 2
    assert data[0]["speech"] == "first segment"


def test_preview_segments_prints_expected_lines(capsys):
    from src.models import Segment

    segments = [
        Segment(
            start=0.0,
            end=2.0,
            mid=1.0,
            speech="some speech text",
            slide_text="slide text",
            slide_time=0.0,
            slide_frame="frame_00000.jpg",
        )
    ]
    preview_segments(segments, n=1)
    captured = capsys.readouterr().out
    assert "Aligned segments: 1" in captured
    assert "[0.00 -> 2.00]" in captured
    assert "some speech text" in captured
    assert "Slide text: slide text" in captured
