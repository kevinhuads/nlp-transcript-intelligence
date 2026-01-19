from __future__ import annotations

from typing import List, Optional
import json
import os
from datetime import datetime

from transformers import pipeline

from .models import Segment, SummaryChunk, VideoSummary


def _chunk_text(text: str, max_chars: int) -> List[str]:
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    for line in text.split("\n"):
        line_len = len(line) + 1
        if current_len + line_len > max_chars and current:
            chunks.append("\n".join(current))
            current = [line]
            current_len = line_len
        else:
            current.append(line)
            current_len += line_len
    if current:
        chunks.append("\n".join(current))
    return chunks


def summarise_segments(
    segments: List[Segment],
    model_name: str = "facebook/bart-large-cnn",
    device: int = 0,
    max_chunk_chars: int = 3000,
    max_length: int = 500,
    min_length: int = 40,
    video_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    summary_filename: Optional[str] = None,
    progress_cb=None,
) -> str:
    full_transcript_text = "\n".join(seg.speech for seg in segments)
    print(f"Total transcript length (characters): {len(full_transcript_text)}")

    chunks = _chunk_text(full_transcript_text, max_chunk_chars)
    print(f"Number of chunks for summarisation: {len(chunks)}")

    if progress_cb is not None:
        progress_cb(0, max(len(chunks), 1), "Summarise: loading model")

    summariser = pipeline(
        "summarization",
        model=model_name,
        device=device,
    )

    summaries: List[str] = []
    summary_chunks: List[SummaryChunk] = []

    total = max(len(chunks), 1)
    for idx, ch in enumerate(chunks):
        print(f"Summarising chunk {idx + 1}/{len(chunks)} …")

        if progress_cb is not None:
            progress_cb(idx, total, f"Summarise: chunk {idx + 1}/{len(chunks)}")

        out = summariser(
            ch,
            max_length=max_length,
            min_length=min_length,
            do_sample=False,
        )
        summary_text = out[0]["summary_text"]
        summaries.append(summary_text)

        if summary_filename is not None:
            summary_chunks.append(
                SummaryChunk(
                    chunk_index=idx,
                    num_chars=len(ch),
                    summary=summary_text,
                )
            )

        if progress_cb is not None:
            progress_cb(idx + 1, total, f"Summarise: chunk {idx + 1}/{len(chunks)}")

    global_summary = "\n".join(summaries)

    print("\n=== GLOBAL SUMMARY ===\n")
    print(global_summary)

    if summary_filename is not None:
        if video_path is None or output_dir is None:
            raise ValueError("video_path and output_dir must be provided when summary_filename is set.")

        os.makedirs(output_dir, exist_ok=True)

        generated_at = datetime.utcnow().isoformat() + "Z"
        video_id = VideoSummary.default_video_id(video_path)

        parameters = {
            "max_chunk_chars": max_chunk_chars,
            "max_length": max_length,
            "min_length": min_length,
        }

        stats = {
            "num_segments": len(segments),
            "num_chunks": len(chunks),
            "total_input_chars": len(full_transcript_text),
            "total_summary_chars": len(global_summary),
        }

        summary_obj = VideoSummary(
            video_id=video_id,
            video_path=os.path.abspath(video_path),
            output_dir=os.path.abspath(output_dir),
            generated_at=generated_at,
            model_name=model_name,
            parameters=parameters,
            stats=stats,
            summary_text=global_summary,
            chunks=summary_chunks,
        )

        summary_path = os.path.join(output_dir, summary_filename)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_obj.to_dict(), f, ensure_ascii=False, indent=2)

        print(f"Structured summary written to {summary_path}")

    if progress_cb is not None:
        progress_cb(1, 1, "Summarise: done")

    return global_summary
