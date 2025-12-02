from __future__ import annotations

import argparse
import os

from .ingest import inspect_video, extract_audio, extract_frames
from .asr import run_asr, preview_transcript
from .ocr import run_ocr_on_frames, preview_ocr
from .align import align_transcript_and_ocr, preview_segments
from .summarise import summarise_segments


def run_full_pipeline(
    video_path: str,
    output_dir: str,
    frame_interval_seconds: int = 3,
    ocr_frame_stride: int = 2,
    device_summariser: int = 0,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    frame_dir = os.path.join(output_dir, "frames")
    os.makedirs(frame_dir, exist_ok=True)

    audio_path = os.path.join(output_dir, "audio.wav")
    transcript_path = os.path.join(output_dir, "transcript_segments.json")
    ocr_output_path = os.path.join(output_dir, "ocr_frames.json")
    segments_path = os.path.join(output_dir, "segments.json")

    print("=== Inspect video ===")
    inspect_video(video_path)

    print("\n=== Extract audio ===")
    extract_audio(video_path, audio_path)

    print("\n=== Extract frames ===")
    extract_frames(video_path, frame_dir, interval_seconds=frame_interval_seconds)

    print("\n=== ASR ===")
    transcript_segments = run_asr(audio_path, transcript_path)
    preview_transcript(transcript_segments, n=5)

    print("\n=== OCR ===")
    ocr_records = run_ocr_on_frames(
        frame_dir=frame_dir,
        ocr_output_path=ocr_output_path,
        frame_interval_seconds=frame_interval_seconds,
        ocr_frame_stride=ocr_frame_stride,
    )
    preview_ocr(ocr_records, n=5)

    print("\n=== Alignment ===")
    segments_merged = align_transcript_and_ocr(
        transcript_segments=transcript_segments,
        ocr_records=ocr_records,
        segments_path=segments_path,
    )
    preview_segments(segments_merged, n=5)

    print("\n=== Global summarisation ===")
    summarise_segments(
        segments_merged,
        model_name="facebook/bart-large-cnn",
        device=device_summariser,
        max_chunk_chars=3000,
        max_length=500,
        min_length=40,
    )

    print("\nPipeline completed.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multimodal NLP pipeline: video -> audio, frames, ASR, OCR, alignment, summary.",
    )
    parser.add_argument(
        "--video-path",
        type=str,
        required=True,
        help="Path to input MP4 video.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where artefacts will be written.",
    )
    parser.add_argument(
        "--frame-interval-seconds",
        type=int,
        default=3,
        help="Interval in seconds between extracted frames.",
    )
    parser.add_argument(
        "--ocr-frame-stride",
        type=int,
        default=2,
        help="Stride for selecting frames for OCR (every nth frame).",
    )
    parser.add_argument(
        "--summariser-device",
        type=int,
        default=0,
        help="Device index for the summarisation model (use -1 for CPU).",
    )

    args = parser.parse_args()

    run_full_pipeline(
        video_path=args.video_path,
        output_dir=args.output_dir,
        frame_interval_seconds=args.frame_interval_seconds,
        ocr_frame_stride=args.ocr_frame_stride,
        device_summariser=args.summariser_device,
    )


if __name__ == "__main__":
    main()
