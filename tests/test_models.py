import json

from src.models import (
    TranscriptSegment,
    OCRRecord,
    Segment,
    segments_to_jsonable,
    transcript_to_jsonable,
    ocr_to_jsonable,
)


def test_transcriptsegment_to_from_dict_roundtrip():
    seg = TranscriptSegment(start=0.5, end=2.5, text="hello")
    data = seg.to_dict()
    seg2 = TranscriptSegment.from_dict(data)
    assert seg2 == seg
    assert isinstance(data, dict)
    assert data["start"] == 0.5
    assert data["end"] == 2.5
    assert data["text"] == "hello"


def test_ocrrecord_to_from_dict_roundtrip():
    rec = OCRRecord(time=3.0, frame="frame_00001.jpg", text="Slide text")
    data = rec.to_dict()
    rec2 = OCRRecord.from_dict(data)
    assert rec2 == rec
    assert data["time"] == 3.0
    assert data["frame"] == "frame_00001.jpg"
    assert data["text"] == "Slide text"


def test_segment_to_from_dict_roundtrip():
    seg = Segment(
        start=1.0,
        end=3.0,
        mid=2.0,
        speech="speech",
        slide_text="slide",
        slide_time=5.0,
        slide_frame="frame_00002.jpg",
    )
    data = seg.to_dict()
    seg2 = Segment.from_dict(data)
    assert seg2 == seg
    assert data["mid"] == 2.0
    assert data["slide_time"] == 5.0
    assert data["slide_frame"] == "frame_00002.jpg"


def test_segments_to_jsonable():
    items = [
        Segment(
            start=0.0,
            end=1.0,
            mid=0.5,
            speech="a",
            slide_text="",
            slide_time=None,
            slide_frame=None,
        ),
        Segment(
            start=1.0,
            end=2.0,
            mid=1.5,
            speech="b",
            slide_text="",
            slide_time=None,
            slide_frame=None,
        ),
    ]
    as_jsonable = segments_to_jsonable(items)
    assert isinstance(as_jsonable, list)
    assert len(as_jsonable) == 2
    assert all(isinstance(x, dict) for x in as_jsonable)
    assert as_jsonable[0]["speech"] == "a"


def test_transcript_to_jsonable_and_back():
    items = [
        TranscriptSegment(start=0.0, end=1.0, text="first"),
        TranscriptSegment(start=1.0, end=2.0, text="second"),
    ]
    jsonable = transcript_to_jsonable(items)
    encoded = json.dumps(jsonable)
    decoded = json.loads(encoded)
    restored = [TranscriptSegment.from_dict(d) for d in decoded]
    assert restored == items


def test_ocr_to_jsonable_and_back():
    items = [
        OCRRecord(time=0.0, frame="frame_00000.jpg", text="A"),
        OCRRecord(time=3.0, frame="frame_00001.jpg", text="B"),
    ]
    jsonable = ocr_to_jsonable(items)
    encoded = json.dumps(jsonable)
    decoded = json.loads(encoded)
    restored = [OCRRecord.from_dict(d) for d in decoded]
    assert restored == items
