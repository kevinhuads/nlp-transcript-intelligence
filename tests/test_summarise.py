from src.summarise import _chunk_text, summarise_segments, summarise_segments_and_save
from src.models import Segment
import json


def test_chunk_text_splits_when_max_chars_exceeded():
    text = "a\nb\nc\nd"
    chunks = _chunk_text(text, max_chars=3)
    # Each line is "a" + newline etc, so each should be separated
    assert chunks == ["a", "b", "c", "d"]


def test_chunk_text_keeps_multiple_lines_when_under_limit():
    text = "first\nsecond\nthird"
    # Enough to keep "first\nsecond" together but not "third"
    max_chars = len("first") + 1 + len("second") + 1  # +1 for newline each time
    chunks = _chunk_text(text, max_chars=max_chars)
    assert len(chunks) == 2
    assert chunks[0] == "first\nsecond"
    assert chunks[1] == "third"


def test_summarise_segments_calls_pipeline_for_each_chunk(monkeypatch, capsys):
    calls = []

    class DummySummariser:
        def __init__(self, task, model, device):
            self.task = task
            self.model = model
            self.device = device

        def __call__(self, text, max_length, min_length, do_sample):
            calls.append(
                {
                    "text": text,
                    "max_length": max_length,
                    "min_length": min_length,
                    "do_sample": do_sample,
                }
            )
            return [{"summary_text": "SUMMARY: " + text[:10]}]

    def fake_pipeline(task, model, device):
        return DummySummariser(task, model, device)

    monkeypatch.setattr("src.summarise.pipeline", fake_pipeline)

    segments = [
        Segment(
            start=0.0,
            end=1.0,
            mid=0.5,
            speech="First part of the transcript.",
            slide_text="",
            slide_time=None,
            slide_frame=None,
        ),
        Segment(
            start=1.0,
            end=2.0,
            mid=1.5,
            speech="Second part of the transcript.",
            slide_text="",
            slide_time=None,
            slide_frame=None,
        ),
    ]

    summary = summarise_segments(
        segments,
        model_name="facebook/bart-large-cnn",
        device=-1,
        max_chunk_chars=30,  # force multiple chunks
        max_length=50,
        min_length=10,
    )

    out = capsys.readouterr().out
    assert "Total transcript length (characters)" in out
    assert "Number of chunks for summarisation" in out
    assert "=== GLOBAL SUMMARY ===" in out

    # With small max_chunk_chars we expect more than one chunk
    assert len(calls) >= 2
    for call in calls:
        assert call["max_length"] == 50
        assert call["min_length"] == 10
        assert call["do_sample"] is False

    # Global summary is the concatenation of per-chunk summaries
    assert "SUMMARY:" in summary


def test_summarise_segments_and_save_writes_summary_json(tmp_path, monkeypatch):
    calls = []

    class DummySummariser:
        def __init__(self, task, model, device):
            self.task = task
            self.model = model
            self.device = device

        def __call__(self, text, max_length, min_length, do_sample):
            calls.append(text)
            return [{"summary_text": "SUMMARY: " + text[:10]}]

    def fake_pipeline(task, model, device):
        return DummySummariser(task, model, device)

    monkeypatch.setattr("src.summarise.pipeline", fake_pipeline)

    segments = [
        Segment(
            start=0.0,
            end=1.0,
            mid=0.5,
            speech="First part of the transcript.",
            slide_text="",
            slide_time=None,
            slide_frame=None,
        ),
        Segment(
            start=1.0,
            end=2.0,
            mid=1.5,
            speech="Second part of the transcript.",
            slide_text="",
            slide_time=None,
            slide_frame=None,
        ),
    ]

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    summary_obj = summarise_segments_and_save(
        segments=segments,
        video_path="example_video.mp4",
        output_dir=str(out_dir),
        model_name="facebook/bart-large-cnn",
        device=-1,
        max_chunk_chars=30,
        max_length=50,
        min_length=10,
    )

    # Ensure the structured object looks reasonable
    assert summary_obj.video_id == "example_video"
    assert summary_obj.summary_text.startswith("SUMMARY:")
    assert summary_obj.stats["num_segments"] == len(segments)

    # Check that the file was written
    summary_path = out_dir / "summary.json"
    assert summary_path.exists()

    with summary_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["video_id"] == "example_video"
    assert "summary_text" in data
    assert "stats" in data
    assert "chunks" in data
    assert len(data["chunks"]) >= 1
