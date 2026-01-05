from __future__ import annotations

import csv
import os
import re
from typing import Dict, List, Tuple

import jiwer


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text, flags=re.UNICODE).strip()
    return text


def load_manifest_csv(manifest_path: str) -> List[dict]:
    with open(manifest_path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def shorten_audio_path_for_csv(audio_path: str, dataset: str) -> str:
    p = os.path.normpath(audio_path)
    parts = p.split(os.sep)
    dataset_norm = dataset.strip().strip("/\\")
    idx = -1
    for i, part in enumerate(parts):
        if part == dataset_norm:
            idx = i
    if idx >= 0 and idx < len(parts):
        return os.sep.join(parts[idx:])
    return os.path.basename(p)


def compute_measures(ref: str, hyp: str) -> Dict[str, float]:
    out = jiwer.process_words(ref, hyp)
    return {
        "wer": float(out.wer),
        "hits": float(out.hits),
        "substitutions": float(out.substitutions),
        "deletions": float(out.deletions),
        "insertions": float(out.insertions),
    }


def measures_to_counts(m: Dict[str, float]) -> Tuple[int, int, int, int]:
    s = int(m["substitutions"])
    d = int(m["deletions"])
    i = int(m["insertions"])
    hits = int(m["hits"])
    n = s + d + hits
    return s, d, i, n
