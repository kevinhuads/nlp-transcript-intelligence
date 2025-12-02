import os

import pytest

from src.ingest import inspect_video, extract_audio, extract_frames


def test_inspect_video_file_not_found(tmp_path):
    base_dir = str(tmp_path)
    video_path = os.path.join(base_dir, "missing.mp4")
    with pytest.raises(FileNotFoundError):
        inspect_video(video_path)


def test_inspect_video_happy_path(tmp_path, monkeypatch):
    base_dir = str(tmp_path)
    video_path = os.path.join(base_dir, "video.mp4")

    with open(video_path, "wb") as f:
        f.write(b"\x00" * 10)

    class DummyClip:
        def __init__(self, path):
            assert path == video_path
            self.duration = 120.0
            self.fps = 30.0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

    def fake_getsize(path):
        assert path == video_path
        return 5 * 1024 * 1024  # 5 MB

    monkeypatch.setattr("src.ingest.VideoFileClip", DummyClip)
    monkeypatch.setattr("src.ingest.os.path.getsize", fake_getsize)

    info = inspect_video(video_path)

    assert info["filename"] == "video.mp4"
    assert info["path"] == video_path
    assert info["size_mb"] == 5.00
    assert info["duration_seconds"] == 120.00
    assert info["duration_minutes"] == 2.00
    assert info["fps"] == 30.0



def test_extract_audio_returns_early_if_audio_exists(tmp_path, monkeypatch):
    base_dir = str(tmp_path)
    video_path = os.path.join(base_dir, "video.mp4")
    audio_dir = os.path.join(base_dir, "audio")
    audio_path = os.path.join(audio_dir, "audio.wav")

    os.makedirs(audio_dir, exist_ok=True)
    with open(audio_path, "wb") as f:
        f.write(b"\x00")

    def fake_video_file_clip(*args, **kwargs):
        raise AssertionError("VideoFileClip should not be used when audio exists")

    monkeypatch.setattr("src.ingest.VideoFileClip", fake_video_file_clip)

    extract_audio(video_path, audio_path)
    # If we reach here, no exception was raised and VideoFileClip was not used


def test_extract_audio_uses_video_clip_and_writes_audio(tmp_path, monkeypatch, capsys):
    base_dir = str(tmp_path)
    video_path = os.path.join(base_dir, "video.mp4")
    audio_path = os.path.join(base_dir, "out", "audio.wav")

    # Write dummy video file only so path is plausible
    os.makedirs(base_dir, exist_ok=True)
    with open(video_path, "wb") as f:
        f.write(b"\x00")

    created = {}

    class DummyAudio:
        def __init__(self):
            self.written_to = None

        def write_audiofile(self, path):
            self.written_to = path
            # Simulate creating the file
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                f.write(b"\x00\x01")

    class DummyClip:
        def __init__(self, path):
            assert path == video_path
            self.audio = DummyAudio()
            created["clip"] = self

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

    monkeypatch.setattr("src.ingest.VideoFileClip", DummyClip)

    extract_audio(video_path, audio_path)
    out = capsys.readouterr().out
    assert "Extracting audio track" in out
    assert "Saved audio to" in out

    assert "clip" in created
    assert created["clip"].audio.written_to == audio_path
    assert os.path.exists(audio_path)


def test_extract_frames_creates_expected_frames(tmp_path, monkeypatch, capsys):
    base_dir = str(tmp_path)
    video_path = os.path.join(base_dir, "video.mp4")
    frame_dir = os.path.join(base_dir, "frames")

    with open(video_path, "wb") as f:
        f.write(b"\x00")

    frame_times = []

    class DummyClip:
        def __init__(self, path):
            assert path == video_path
            self.duration = 7.5  # seconds

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def get_frame(self, t):
            frame_times.append(t)
            return "dummy_frame_at_%s" % t

    class DummyImage:
        def __init__(self, frame):
            self.frame = frame

        def save(self, path, format=None):
            # Just create an empty file where the frame would be saved
            with open(path, "wb") as f:
                f.write(b"\x00")

    def fake_fromarray(frame):
        return DummyImage(frame)

    # Pretend audio exists so "Audio present" is True
    audio_path = os.path.join(base_dir, "audio.wav")
    with open(audio_path, "wb") as f:
        f.write(b"\x00")

    monkeypatch.setattr("src.ingest.VideoFileClip", DummyClip)
    monkeypatch.setattr("src.ingest.Image.fromarray", fake_fromarray)

    extract_frames(video_path, frame_dir, interval_seconds=3)
    out = capsys.readouterr().out

    # Duration 7.5 with interval 3 gives times [0, 3, 6]
    assert set(frame_times) == {0, 3, 6}
    assert "Planned number of frames: 3" in out
    assert "Saving frames to" in out

    frame_files = sorted(
        f for f in os.listdir(frame_dir) if f.startswith("frame_") and f.endswith(".jpg")
    )
    assert len(frame_files) == 3
    assert frame_files[0] == "frame_00000.jpg"
    assert frame_files[-1] == "frame_00002.jpg"
    assert "Audio present: True" in out
    assert "Number of frame files: 3" in out
