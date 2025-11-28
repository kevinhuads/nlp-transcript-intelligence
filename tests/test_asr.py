import json
import os

import pytest

from src.asr import run_asr, preview_transcript
from src.models import TranscriptSegment


def test_run_asr_uses_cached_transcript_if_exists(tmp_path, monkeypatch, capsys):
    base_dir = str(tmp_path)
    audio_path = os.path.join(base_dir, "audio.wav")
    transcript_path = os.path.join(base_dir, "transcript", "segments.json")

    os.makedirs(os.path.dirname(transcript_path), exist_ok=True)

    cached = [
        {"start": 0.0, "end": 1.0, "text": "hello"},
        {"start": 1.0, "end": 2.0, "text": "world"},
    ]
    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump(cached, f)

    # If WhisperModel is called the test should fail
    def fake_whisper_model(*args, **kwargs):
        raise AssertionError("WhisperModel should not be constructed when cache exists")

    monkeypatch.setattr("src.asr.WhisperModel", fake_whisper_model)

    segments = run_asr(
        audio_path=audio_path,
        transcript_path=transcript_path,
        model_size="tiny",
        device="cpu",
        compute_type="int8",
    )

    captured = capsys.readouterr().out
    assert "Transcript file already exists" in captured
    assert "Loaded 2 transcript segments." in captured

    assert len(segments) == 2
    assert isinstance(segments[0], TranscriptSegment)
    assert segments[0].text == "hello"
    assert segments[1].text == "world"


def test_run_asr_creates_transcript_when_missing(tmp_path, monkeypatch):
    base_dir = str(tmp_path)
    audio_path = os.path.join(base_dir, "audio.wav")
    transcript_path = os.path.join(base_dir, "out", "segments.json")

    # Simulate that the audio file exists
    os.makedirs(os.path.dirname(audio_path), exist_ok=True)
    with open(audio_path, "wb") as f:
        f.write(b"\x00")

    created_model_args = {}

    class DummySeg:
        def __init__(self, start, end, text):
            self.start = start
            self.end = end
            self.text = text

    class DummyModel:
        def __init__(self, model_size, device, compute_type):
            created_model_args["model_size"] = model_size
            created_model_args["device"] = device
            created_model_args["compute_type"] = compute_type

        def transcribe(self, audio_path_str, beam_size):
            assert isinstance(audio_path_str, str)
            assert beam_size == 5
            segments_iter = [
                DummySeg(0.0, 1.0, " first "),
                DummySeg(1.0, 2.0, " second"),
            ]
            return segments_iter, {"info": "dummy"}

    monkeypatch.setattr("src.asr.WhisperModel", DummyModel)

    segments = run_asr(
        audio_path=audio_path,
        transcript_path=transcript_path,
        model_size="small",
        device="cuda",
        compute_type="int8",
    )

    assert created_model_args["model_size"] == "small"
    assert created_model_args["device"] == "cuda"
    assert created_model_args["compute_type"] == "int8"

    assert len(segments) == 2
    assert segments[0].start == 0.0
    assert segments[0].end == 1.0
    assert segments[0].text == "first"
    assert segments[1].text == "second"

    assert os.path.exists(transcript_path)
    with open(transcript_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert len(data) == 2
    assert data[0]["text"] == "first"
    assert data[1]["text"] == "second"


def test_preview_transcript_prints_expected_format(capsys):
    segments = [
        TranscriptSegment(start=0.0, end=1.2345, text="hello"),
        TranscriptSegment(start=1.2345, end=2.5, text="world"),
    ]
    preview_transcript(segments, n=2)
    out = capsys.readouterr().out
    assert "Total segments: 2" in out
    assert "[0.00 -> 1.23] hello" in out
    assert "[1.23 -> 2.50] world" in out
