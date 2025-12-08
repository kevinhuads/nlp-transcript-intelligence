from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Callable, Sequence

from dataclasses import dataclass

import numpy as np
from sklearn.neighbors import NearestNeighbors
from .models import Segment


@dataclass
class SearchResult:
    """One retrieval result."""
    rank: int
    score: float                    # similarity (higher is better)
    text: str                       # text used for embedding (for display)
    segment: Any                    # original payload (Segment or dict)
    extra: Dict[str, Any]           # optional extra metadata


class SemanticIndex:
    """
    Simple semantic index over a list of items (segments).

    Responsibilities:
    - Hold embeddings and corresponding payload objects.
    - Perform nearest-neighbour search on embeddings.
    - Return a list of SearchResult with similarity scores and payloads.
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        segments: Sequence[Any],
        texts: Sequence[str],
        metric: str = "cosine",
    ) -> None:
        if len(embeddings) != len(segments) or len(segments) != len(texts):
            raise ValueError("embeddings, segments and texts must have the same length")

        self.embeddings = np.asarray(embeddings, dtype="float32")
        self.segments: List[Any] = list(segments)
        self.texts: List[str] = list(texts)
        self.metric: str = metric

        # Build nearest neighbours index
        self._nn = NearestNeighbors(
            metric=metric,
            algorithm="auto",
        )
        self._nn.fit(self.embeddings)

    @classmethod
    def from_segments(
        cls,
        segments: Sequence[Segment],
        model: Any,
        text_fn: Optional[Callable[[Segment], str]] = None,
        batch_size: int = 32,
        normalize: bool = True,
        metric: str = "cosine",
        encode_kwargs: Optional[Dict[str, Any]] = None,
    ) -> "SemanticIndex":
        """
        Build a SemanticIndex from Segment dataclass instances using an embedding model.

        Parameters
        ----------
        segments:
            Iterable of Segment objects.
        model:
            Embedding model with an .encode(List[str], ...) -> np.ndarray API
            (for example sentence-transformers SentenceTransformer).
        text_fn:
            Function mapping a Segment -> text to embed. Defaults to segment.speech.
        batch_size:
            Batch size for model.encode.
        normalize:
            Whether to L2-normalise embeddings (recommended for cosine similarity).
        metric:
            Distance metric for NearestNeighbors ("cosine" or "euclidean" etc.).
        encode_kwargs:
            Extra keyword arguments passed to model.encode.
        """
        if text_fn is None:
            text_fn = lambda seg: seg.speech  # type: ignore[assignment]

        texts = [text_fn(seg) for seg in segments]

        if encode_kwargs is None:
            encode_kwargs = {}

        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            **encode_kwargs,
        )

        embeddings = np.asarray(embeddings, dtype="float32")
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
            embeddings = embeddings / norms

        return cls(
            embeddings=embeddings,
            segments=list(segments),
            texts=texts,
            metric=metric,
        )

    @classmethod
    def from_records(
        cls,
        records: Sequence[Dict[str, Any]],
        model: Any,
        text_fn: Callable[[Dict[str, Any]], str],
        batch_size: int = 32,
        normalize: bool = True,
        metric: str = "cosine",
        encode_kwargs: Optional[Dict[str, Any]] = None,
    ) -> "SemanticIndex":
        """
        Build a SemanticIndex from plain dict records (for example loaded from segments.json).

        Parameters
        ----------
        records:
            Iterable of dict records.
        model:
            Embedding model with an .encode(List[str], ...) -> np.ndarray API.
        text_fn:
            Function mapping a record -> text to embed (for example build_segment_text).
        The remaining parameters are identical to from_segments.
        """
        texts = [text_fn(rec) for rec in records]

        if encode_kwargs is None:
            encode_kwargs = {}

        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            **encode_kwargs,
        )

        embeddings = np.asarray(embeddings, dtype="float32")
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
            embeddings = embeddings / norms

        return cls(
            embeddings=embeddings,
            segments=list(records),
            texts=texts,
            metric=metric,
        )

    @classmethod
    def from_precomputed_embeddings(
        cls,
        segments: Sequence[Any],
        embeddings: np.ndarray,
        texts: Sequence[str],
        metric: str = "cosine",
    ) -> "SemanticIndex":
        """
        Build a SemanticIndex when embeddings have already been computed and saved.

        Parameters
        ----------
        segments:
            Payload items (Segment dataclasses or dict records).
        embeddings:
            Array of shape (n_items, dim).
        texts:
            Text used for each embedding, same order as segments.
        metric:
            Distance metric for NearestNeighbors.
        """
        embeddings = np.asarray(embeddings, dtype="float32")
        return cls(
            embeddings=embeddings,
            segments=list(segments),
            texts=list(texts),
            metric=metric,
        )

    def search(
        self,
        query: str,
        model: Any,
        top_k: int = 5,
        encode_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Search for the segments most semantically similar to the query.

        Parameters
        ----------
        query:
            User query string.
        model:
            Same embedding model used to build the index (or a compatible one).
        top_k:
            Maximum number of results to return.
        encode_kwargs:
            Extra kwargs passed to model.encode for the query.

        Returns
        -------
        List[SearchResult]
        """
        if encode_kwargs is None:
            encode_kwargs = {}

        query_emb = model.encode(
            [query],
            convert_to_numpy=True,
            **encode_kwargs,
        )
        query_emb = np.asarray(query_emb, dtype="float32")

        # Normalise query embedding for cosine similarity
        norms = np.linalg.norm(query_emb, axis=1, keepdims=True) + 1e-12
        query_emb = query_emb / norms

        n_results = min(top_k, len(self.segments))
        distances, indices = self._nn.kneighbors(
            query_emb,
            n_neighbors=n_results,
        )

        distances = distances[0]
        indices = indices[0]

        results: List[SearchResult] = []
        for rank, (idx, dist) in enumerate(zip(indices, distances), start=1):
            if self.metric == "cosine":
                # sklearn returns distance = 1 - cosine_similarity
                score = 1.0 - float(dist)
            else:
                score = float(-dist)

            seg = self.segments[int(idx)]
            text = self.texts[int(idx)]
            extra: Dict[str, Any] = {
                "index": int(idx),
                "distance": float(dist),
            }

            results.append(
                SearchResult(
                    rank=rank,
                    score=score,
                    text=text,
                    segment=seg,
                    extra=extra,
                )
            )

        return results




def load_segments(path: str) -> List[Dict[str, Any]]:
    """
    Load multimodal segments from a JSON file.

    The file is expected to contain a JSON list of dict-like records.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Segments file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a list of segments in {path}, got {type(data)}")

    return data


def build_segment_text(segment: Dict[str, Any]) -> str:
    """
    Build a text representation for a segment suitable for embeddings.

    Spoken text is preferred, with slide/OCR text appended when available.

    Priority for spoken text:
        combined_text, text, transcript, asr_text, speech

    Priority for slide text:
        slide_text, ocr_text

    When both are present the result is:

        <spoken>

        [SLIDE]
        <slide text>
    """
    speech_text: Optional[str] = None
    for key in ("combined_text", "text", "transcript", "asr_text", "speech"):
        value = segment.get(key)
        if isinstance(value, str) and value.strip():
            speech_text = value.strip()
            break

    slide_text: Optional[str] = None
    for key in ("slide_text", "ocr_text"):
        value = segment.get(key)
        if isinstance(value, str) and value.strip():
            slide_text = value.strip()
            break

    if speech_text and slide_text:
        combined = speech_text
        combined += "\n\n[SLIDE]\n" + slide_text
        return combined

    if speech_text:
        return speech_text

    if slide_text:
        return slide_text

    generic = segment.get("content")
    return generic.strip() if isinstance(generic, str) else ""


def get_segment_text(
    segment: Dict[str, Any],
    max_chars: Optional[int] = None,
) -> str:
    """
    Human-readable segment text, suitable for UI and logging.

    Uses the same logic as build_segment_text, with optional truncation.
    """
    text = build_segment_text(segment)
    if not text:
        text = "[no text found]"

    if max_chars is not None and len(text) > max_chars:
        return text[:max_chars] + "..."

    return text


def get_segment_time_range(segment: Dict[str, Any]) -> str:
    """
    Format time information from a segment, if available.

    Supports both (start, end) and (start_sec, end_sec) naming schemes.
    """
    start = segment.get("start")
    end = segment.get("end")

    if start is None or end is None:
        start = segment.get("start_sec")
        end = segment.get("end_sec")

    if start is None or end is None:
        return "[no timestamps]"

    try:
        start_f = float(start)
        end_f = float(end)
    except (TypeError, ValueError):
        return "[invalid timestamps]"

    return f"{start_f:.2f} s → {end_f:.2f} s"


def pretty_print_results(results: List[SearchResult], max_chars: int = 260) -> None:
    """Pretty print search results in a concise format."""
    for r in results:
        seg = r.segment
        rank = r.rank
        score = r.score

        # Handle dict-based segments (for example records from segments.json)
        if isinstance(seg, dict):
            time_range = get_segment_time_range(seg)
            seg_id = seg.get("segment_id") or seg.get("id") or "[no id]"
            text = get_segment_text(seg)
        else:
            # Fallback for Segment dataclass or similar payloads
            start = getattr(seg, "start", None)
            end = getattr(seg, "end", None)
            if start is not None and end is not None:
                try:
                    time_range = f"{float(start):.2f} s → {float(end):.2f} s"
                except (TypeError, ValueError):
                    time_range = "[invalid timestamps]"
            else:
                time_range = "[no timestamps]"

            seg_id = getattr(seg, "segment_id", "[no id]")

            parts: List[str] = []
            speech = getattr(seg, "speech", None)
            if isinstance(speech, str) and speech.strip():
                parts.append(speech.strip())
            slide = getattr(seg, "slide_text", None)
            if isinstance(slide, str) and slide.strip():
                parts.append("[SLIDE]\n" + slide.strip())

            text = "\n\n".join(parts) if parts else "[no text found]"

        if len(text) > max_chars:
            text_display = text[:max_chars] + "..."
        else:
            text_display = text

        print(f"Rank {rank} | score={score:.3f} | {time_range} | id={seg_id}")
        print(text_display)
        print("-" * 80)


def semantic_search(
    index: SemanticIndex,
    model: Any,
    query: str,
    top_k: int = 5,
    max_chars: int = 260,
) -> None:
    """Wrapper around index.search with pretty-printing."""
    print(f"Query: {query}\n")
    results = index.search(query=query, model=model, top_k=top_k)
    pretty_print_results(results, max_chars=max_chars)
