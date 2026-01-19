from __future__ import annotations

import hashlib
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

from langchain_ollama import ChatOllama, OllamaEmbeddings


def _ensure_dir(file_path: str) -> None:
    folder = os.path.dirname(file_path)
    if folder:
        os.makedirs(folder, exist_ok=True)


def _config_signature(
    embed_model: str,
    chunk_chars: int,
    chunk_overlap: int,
    text_mode: str,
) -> str:
    return f"{embed_model}|chunk={chunk_chars}|overlap={chunk_overlap}|mode={text_mode}"


def _read_cached_index(index_path: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    if not os.path.exists(index_path):
        return None, None

    with open(index_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        return None, None

    meta = raw.get("meta", {})
    sig = meta.get("signature")
    return sig, raw


def _hash_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def _chunk_text(text: str, chunk_chars: int, chunk_overlap: int) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []

    chunk_chars = max(int(chunk_chars), 1)
    chunk_overlap = max(int(chunk_overlap), 0)
    if chunk_overlap >= chunk_chars:
        chunk_overlap = max(chunk_chars - 1, 0)

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_chars, len(text))
        part = text[start:end].strip()
        if part:
            chunks.append(part)
        if end >= len(text):
            break
        start = max(end - chunk_overlap, 0)

    return chunks


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _segment_to_text(seg: Any, text_mode: str) -> str:
    if text_mode == "aligned":
        asr_text = (_get(seg, "asr_text", None) or _get(seg, "text", "")).strip()
        ocr_text = (_get(seg, "ocr_text", "") or "").strip()
        if ocr_text:
            return f"{asr_text}\n\n[OCR]\n{ocr_text}".strip()
        return asr_text
    return (_get(seg, "text", None) or _get(seg, "asr_text", "")).strip()


def _segment_meta(seg: Any) -> Dict[str, Any]:
    return {
        "start": _get(seg, "start", None),
        "end": _get(seg, "end", None),
        "frame": _get(seg, "frame", None),
    }


def _cosine(a: List[float], b: List[float]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    denom = math.sqrt(na) * math.sqrt(nb)
    if denom == 0.0:
        return 0.0
    return dot / denom


def _normalize_base_url(host: str) -> str:
    base = (host or "").strip().rstrip("/")
    if base.endswith("/api"):
        base = base[:-4]
    if base.endswith("/v1"):
        base = base[:-3]
    return base


def _make_embeddings(embed_model: str, host: str, timeout_s: int) -> OllamaEmbeddings:
    base_url = _normalize_base_url(host)
    return OllamaEmbeddings(
        model=embed_model,
        base_url=base_url,
        client_kwargs={"timeout": timeout_s},
    )


def _make_chat(llm_model: str, host: str, timeout_s: int) -> ChatOllama:
    base_url = _normalize_base_url(host)
    return ChatOllama(
        model=llm_model,
        temperature=0,
        base_url=base_url,
        client_kwargs={"timeout": timeout_s},
    )


def _batches(items: List[str], batch_size: int) -> List[List[str]]:
    batch_size = max(int(batch_size), 1)
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def build_qa_index(
    segments: List[Any],
    index_path: str,
    embed_model: str = "nomic-embed-text",
    chunk_chars: int = 900,
    chunk_overlap: int = 150,
    text_mode: str = "aligned",
    force: bool = False,
    host: str = "http://localhost:11434",
    progress_cb=None,
) -> Dict[str, Any]:
    _ensure_dir(index_path)

    requested_sig = _config_signature(
        embed_model=embed_model,
        chunk_chars=chunk_chars,
        chunk_overlap=chunk_overlap,
        text_mode=text_mode,
    )

    cached_sig, cached = _read_cached_index(index_path)
    if not force and cached is not None and cached_sig == requested_sig:
        if progress_cb is not None:
            progress_cb(1, 1, "RAG index: cached index loaded")
        return cached

    chunks: List[Dict[str, Any]] = []
    for seg in segments:
        base_text = _segment_to_text(seg, text_mode=text_mode)
        if not base_text:
            continue
        seg_meta = _segment_meta(seg)
        for part in _chunk_text(base_text, chunk_chars=chunk_chars, chunk_overlap=chunk_overlap):
            cid = _hash_id(f"{seg_meta.get('start')}|{seg_meta.get('end')}|{part}")
            chunks.append(
                {
                    "id": cid,
                    "text": part,
                    "meta": seg_meta,
                    "embedding": None,
                }
            )

    total = len(chunks)
    if total == 0:
        payload = {
            "meta": {
                "signature": requested_sig,
                "embed_model": embed_model,
                "chunk_chars": chunk_chars,
                "chunk_overlap": chunk_overlap,
                "text_mode": text_mode,
                "host": host,
            },
            "chunks": [],
        }
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        if progress_cb is not None:
            progress_cb(1, 1, "RAG index: done (no chunks)")
        return payload

    embeddings = _make_embeddings(embed_model=embed_model, host=host, timeout_s=120)

    batch_size = 32
    done = 0
    texts = [ch["text"] for ch in chunks]
    for batch in _batches(texts, batch_size=batch_size):
        embs = embeddings.embed_documents(batch)
        for emb in embs:
            chunks[done]["embedding"] = list(emb)
            done += 1
            if progress_cb is not None:
                progress_cb(done, total, f"RAG index: embedding {done}/{total}")

    payload = {
        "meta": {
            "signature": requested_sig,
            "embed_model": embed_model,
            "chunk_chars": chunk_chars,
            "chunk_overlap": chunk_overlap,
            "text_mode": text_mode,
            "host": host,
        },
        "chunks": chunks,
    }

    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    if progress_cb is not None:
        progress_cb(1, 1, "RAG index: done")

    return payload


def answer_question(
    question: str,
    index_path: str,
    llm_model: str = "llama3.1",
    embed_model: str = "nomic-embed-text",
    top_k: int = 5,
    host: str = "http://localhost:11434",
    progress_cb=None,
) -> Dict[str, Any]:
    if not os.path.exists(index_path):
        raise FileNotFoundError(index_path)

    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    chunks = index.get("chunks", []) or []
    q = (question or "").strip()
    if not q:
        return {"answer": "", "sources": []}

    if progress_cb is not None:
        progress_cb(0, 1, "Q/A: embedding question")

    embeddings = _make_embeddings(embed_model=embed_model, host=host, timeout_s=120)
    q_emb = list(embeddings.embed_query(q))

    scored: List[Tuple[float, Dict[str, Any]]] = []
    for ch in chunks:
        emb = ch.get("embedding")
        if not emb:
            continue
        score = _cosine(q_emb, emb)
        scored.append((score, ch))

    scored.sort(key=lambda x: x[0], reverse=True)
    hits = [ch for _score, ch in scored[: max(int(top_k), 1)]]

    context_lines: List[str] = []
    for i, ch in enumerate(hits, start=1):
        meta = ch.get("meta", {}) or {}
        start = meta.get("start")
        end = meta.get("end")
        label = f"[S{i}]"
        time_part = ""
        if start is not None and end is not None:
            time_part = f" t={start:.2f}-{end:.2f}"
        context_lines.append(f"{label}{time_part}\n{ch.get('text','')}".strip())

    context = "\n\n---\n\n".join(context_lines).strip()

    prompt = (
        "Context:\n"
        f"{context}\n\n"
        "Question:\n"
        f"{q}\n\n"
        "Answer:"
    )

    if progress_cb is not None:
        progress_cb(0, 1, "Q/A: generating answer")

    llm = _make_chat(llm_model=llm_model, host=host, timeout_s=300)
    system_text = (
        "Answer using only the provided context. "
        "If the context is insufficient, say so. "
        "When possible, cite sources using [S#] markers."
    )

    msg = llm.invoke([("system", system_text), ("human", prompt)])
    answer = (getattr(msg, "content", None) or str(msg)).strip()

    if progress_cb is not None:
        progress_cb(1, 1, "Q/A: done")

    sources: List[Dict[str, Any]] = []
    for i, ch in enumerate(hits, start=1):
        sources.append(
            {
                "source_id": f"S{i}",
                "chunk_id": ch.get("id"),
                "meta": ch.get("meta", {}),
                "text": ch.get("text", ""),
            }
        )

    return {
        "answer": answer,
        "sources": sources,
    }
