import json
import os

from src.ocr import run_ocr_on_frames, preview_ocr
from src.models import OCRRecord


def test_run_ocr_on_frames_uses_cached_output_if_exists(tmp_path, monkeypatch, capsys):
    base_dir = str(tmp_path)
    frame_dir = os.path.join(base_dir, "frames")
    ocr_output_path = os.path.join(base_dir, "ocr", "ocr.json")

    os.makedirs(frame_dir, exist_ok=True)
    os.makedirs(os.path.dirname(ocr_output_path), exist_ok=True)

    cached = [
        {"time": 0.0, "frame": "frame_00000.jpg", "text": "hello"},
        {"time": 3.0, "frame": "frame_00002.jpg", "text": "world"},
    ]
    with open(ocr_output_path, "w", encoding="utf-8") as f:
        json.dump(cached, f)

    # If PIL or pytesseract are used, test should fail
    def fail_open(*args, **kwargs):
        raise AssertionError("Image.open should not be used when cache exists")

    def fail_ocr(*args, **kwargs):
        raise AssertionError("pytesseract.image_to_string should not be used when cache exists")

    monkeypatch.setattr("src.ocr.Image.open", fail_open)
    monkeypatch.setattr("src.ocr.pytesseract.image_to_string", fail_ocr)

    records = run_ocr_on_frames(
        frame_dir=frame_dir,
        ocr_output_path=ocr_output_path,
        frame_interval_seconds=3,
        ocr_frame_stride=2,
    )

    out = capsys.readouterr().out
    assert "OCR output already exists" in out
    assert "Loaded OCR text for 2 frames." in out

    assert len(records) == 2
    assert isinstance(records[0], OCRRecord)
    assert records[0].text == "hello"
    assert records[1].frame == "frame_00002.jpg"


def test_run_ocr_on_frames_processes_sampled_frames(tmp_path, monkeypatch, capsys):
    base_dir = str(tmp_path)
    frame_dir = os.path.join(base_dir, "frames")
    ocr_output_path = os.path.join(base_dir, "ocr.json")

    os.makedirs(frame_dir, exist_ok=True)

    # Create some dummy frame files
    names = ["frame_00000.jpg", "frame_00001.jpg", "frame_00002.jpg"]
    for name in names:
        with open(os.path.join(frame_dir, name), "wb") as f:
            f.write(b"\x00")

    class DummyImage:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

    def fake_open(path):
        # Return a context manager compatible object
        assert path.endswith(".jpg")
        return DummyImage()

    def fake_ocr(img):
        return "DETECTED_TEXT"

    monkeypatch.setattr("src.ocr.Image.open", fake_open)
    monkeypatch.setattr("src.ocr.pytesseract.image_to_string", fake_ocr)

    records = run_ocr_on_frames(
        frame_dir=frame_dir,
        ocr_output_path=ocr_output_path,
        frame_interval_seconds=3,
        ocr_frame_stride=2,
    )

    out = capsys.readouterr().out
    assert "Running OCR on sampled frames" in out
    assert "Saved OCR output for" in out

    # With stride 2, indices 0 and 2 should be processed
    assert len(records) == 2
    assert records[0].frame == "frame_00000.jpg"
    assert records[1].frame == "frame_00002.jpg"
    assert records[0].time == 0.0
    assert records[1].time == 2 * 3  # idx * frame_interval_seconds
    assert records[0].text == "DETECTED_TEXT"

    # Check file contents
    assert os.path.exists(ocr_output_path)
    with open(ocr_output_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert len(data) == 2
    assert data[0]["frame"] == "frame_00000.jpg"


def test_preview_ocr_prints_expected_format(capsys):
    records = [
        OCRRecord(time=0.0, frame="frame_00000.jpg", text="A"),
        OCRRecord(time=3.0, frame="frame_00002.jpg", text="B"),
    ]
    preview_ocr(records, n=1)
    out = capsys.readouterr().out
    assert "OCR records: 2" in out
    assert "[t ~ 0.0s] frame=frame_00000.jpg" in out
    assert "A" in out
