from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors  # you may not need this anymore

from src.utils import load_segments, build_segment_text, SemanticIndex


@dataclass
class EmbeddingConfig:
    """
    Configuration for segment-level embeddings.
    """

    model_name: str = "sentence-transformers/all-MiniLM-L12-v2"
    batch_size: int = 32
    device: str = "cpu"  # set to "cuda" if GPU is available
    normalize_embeddings: bool = True

    # Name of the embedding artefact, relative to the output directory.
    embeddings_filename: str = "segment_embeddings.npy"


def load_embedding_model(config: EmbeddingConfig) -> SentenceTransformer:
    """
    Load the sentence-transformers embedding model according to the config.
    """
    model = SentenceTransformer(config.model_name, device=config.device)
    return model


def compute_segment_embeddings(
    segments: Sequence[Dict[str, Any]],
    model: SentenceTransformer,
    config: EmbeddingConfig,
) -> np.ndarray:
    """
    Compute embeddings for a sequence of segments.

    Returns an array of shape (num_segments, dim).
    """
    texts = [build_segment_text(seg) for seg in segments]

    embeddings = model.encode(
        texts,
        batch_size=config.batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    if config.normalize_embeddings:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        embeddings = embeddings / norms

    return embeddings.astype("float32")


def save_embeddings(embeddings: np.ndarray, output_path: str) -> None:
    """
    Save embeddings to a .npy file.
    """
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    np.save(output_path, embeddings)


def load_embeddings(embeddings_path: str) -> np.ndarray:
    """
    Load embeddings from a .npy file.
    """
    return np.load(embeddings_path)

def build_index_from_output_dir(
    output_dir: str,
    config: Optional[EmbeddingConfig] = None,
    model: Optional[SentenceTransformer] = None,
    force_recompute: bool = False,
) -> SemanticIndex:
    """
    Build a SemanticIndex from artefacts in an output directory.

    Expected files:
      - segments.json
      - segment_embeddings.npy (created if missing or force_recompute is True)

    The index payloads are the raw segment dicts loaded from segments.json.
    """
    if config is None:
        config = EmbeddingConfig()

    segments_path = os.path.join(output_dir, "segments.json")
    embeddings_path = os.path.join(output_dir, config.embeddings_filename)

    segments = load_segments(segments_path)

    if os.path.exists(embeddings_path) and not force_recompute:
        embeddings = load_embeddings(embeddings_path)
    else:
        if model is None:
            model = load_embedding_model(config)
        embeddings = compute_segment_embeddings(segments, model, config)
        save_embeddings(embeddings, embeddings_path)

    texts = [build_segment_text(seg) for seg in segments]

    index = SemanticIndex.from_precomputed_embeddings(
        segments=segments,
        embeddings=embeddings,
        texts=texts,
        metric="cosine",
    )
    return index
