from __future__ import annotations

from typing import List

from transformers import pipeline

from .models import Segment


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
) -> str:
    full_transcript_text = "\n".join(seg.speech for seg in segments)
    print(f"Total transcript length (characters): {len(full_transcript_text)}")

    chunks = _chunk_text(full_transcript_text, max_chunk_chars)
    print(f"Number of chunks for summarisation: {len(chunks)}")

    summariser = pipeline(
        "summarization",
        model=model_name,
        device=device,  # use -1 for CPU
    )

    summaries: List[str] = []
    for idx, ch in enumerate(chunks):
        print(f"Summarising chunk {idx + 1}/{len(chunks)} …")
        out = summariser(
            ch,
            max_length=max_length,
            min_length=min_length,
            do_sample=False,
        )
        summaries.append(out[0]["summary_text"])

    global_summary = "\n".join(summaries)

    print("\n=== GLOBAL SUMMARY ===\n")
    print(global_summary)

    return global_summary
