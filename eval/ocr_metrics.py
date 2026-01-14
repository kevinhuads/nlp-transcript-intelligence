from __future__ import annotations

import re
from typing import Dict, Tuple


def normalize(text: str, lowercase: bool = True, strip_punct: bool = True) -> str:
    s = str(text)
    if lowercase:
        s = s.lower()
    if strip_punct:
        s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s, flags=re.UNICODE).strip()
    return s


def _require_jiwer():
    try:
        import jiwer
    except Exception as e:
        raise RuntimeError("jiwer is required for OCR CER/WER computation") from e
    return jiwer


def compute_word_measures(ref: str, hyp: str) -> Dict[str, float]:
    jiwer = _require_jiwer()
    out = jiwer.process_words(ref, hyp)
    return {
        "wer": float(out.wer),
        "hits": float(out.hits),
        "substitutions": float(out.substitutions),
        "deletions": float(out.deletions),
        "insertions": float(out.insertions),
    }


def compute_char_measures(ref: str, hyp: str) -> Dict[str, float]:
    jiwer = _require_jiwer()
    ref_chars = " ".join(list(ref))
    hyp_chars = " ".join(list(hyp))
    out = jiwer.process_words(ref_chars, hyp_chars)
    return {
        "cer": float(out.wer),
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
