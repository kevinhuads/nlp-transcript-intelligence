# src/ingest.py

import os
from moviepy import VideoFileClip
from PIL import Image
from tqdm import tqdm


def inspect_video(video_path: str) -> None:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    file_size_mb = os.path.getsize(video_path) / (1024 * 1024)

    # Ensure the reader is closed when we are done
    with VideoFileClip(video_path) as clip:
        duration = clip.duration
        fps = clip.fps

    print(f"Found video file: {os.path.basename(video_path)}")
    print(f"Size: {file_size_mb:.2f} MB")
    print(f"Duration: {duration:.2f} seconds ({duration / 60:.2f} minutes)")
    print(f"Frame rate (fps): {fps}")


def extract_audio(video_path: str, audio_path: str) -> None:
    os.makedirs(os.path.dirname(audio_path), exist_ok=True)

    if os.path.exists(audio_path):
        print(f"Audio file already exists: {audio_path}")
        return

    # Open, write audio, then close
    with VideoFileClip(video_path) as clip:
        print("Extracting audio track ...")
        clip.audio.write_audiofile(audio_path)
        print(f"Saved audio to: {audio_path}")


def extract_frames(
    video_path: str,
    frame_dir: str,
    interval_seconds: int = 3,
) -> None:
    os.makedirs(frame_dir, exist_ok=True)

    # Open the video only inside this context
    with VideoFileClip(video_path) as clip:
        duration = clip.duration

        n_frames = int(duration // interval_seconds) + 1
        print(f"Planned number of frames: {n_frames}")
        print(f"Saving frames to: {frame_dir}")

        for i, t in enumerate(tqdm(range(0, int(duration) + 1, interval_seconds))):
            frame_time = min(t, duration)
            frame = clip.get_frame(frame_time)
            frame_path = os.path.join(frame_dir, f"frame_{i:05d}.jpg")
            if os.path.exists(frame_path):
                continue
            img = Image.fromarray(frame)
            img.save(frame_path, format="JPEG")

    audio_path = os.path.join(os.path.dirname(frame_dir), "audio.wav")
    audio_present = os.path.exists(audio_path)
    frame_files = sorted(
        f for f in os.listdir(frame_dir) if f.startswith("frame_") and f.endswith(".jpg")
    )

    print(f"Audio present: {audio_present}")
    print(f"Number of frame files: {len(frame_files)}")
    print("First few frame files:")
    for f in frame_files[:5]:
        print(" -", f)
