# src/ingest.py

import os
from moviepy import VideoFileClip
from PIL import Image
from tqdm import tqdm


def inspect_video(video_path: str) -> dict:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    file_size_mb = os.path.getsize(video_path) / (1024 * 1024)

    # Ensure the reader is closed when we are done
    with VideoFileClip(video_path) as clip:
        duration = clip.duration
        fps = clip.fps

    # Return metadata instead of printing it
    return {
        "filename": os.path.basename(video_path),
        "path": video_path,
        "size_mb": round(file_size_mb, 2),
        "duration_seconds": round(duration, 2),
        "duration_minutes": round(duration / 60, 2),
        "fps": fps,
    }


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
    progress_cb=None,
) -> None:
    os.makedirs(frame_dir, exist_ok=True)

    with VideoFileClip(video_path) as clip:
        duration = float(clip.duration)

        times = list(range(0, int(duration) + 1, interval_seconds))
        if not times:
            times = [0]
        if times[-1] < duration:
            times.append(int(duration))

        total = len(times)
        print(f"Planned number of frames: {total}")
        print(f"Saving frames to: {frame_dir}")

        for i, t in enumerate(tqdm(times, total=total)):
            frame_time = min(float(t), duration)
            frame_path = os.path.join(frame_dir, f"frame_{i:05d}.jpg")

            if not os.path.exists(frame_path):
                frame = clip.get_frame(frame_time)
                img = Image.fromarray(frame)
                img.save(frame_path, format="JPEG")

            if progress_cb is not None:
                progress_cb(i + 1, total, f"Extracting frames: {i + 1}/{total}")

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
