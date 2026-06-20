from __future__ import annotations

import hashlib
import re

from langchain_core.documents import Document


def char_grams(text: str, n: int) -> set[str]:
    if len(text) < n:
        return {text} if text else set()
    return {text[index : index + n] for index in range(0, len(text) - n + 1)}


def normalize_for_lexical(text: str) -> str:
    return re.sub(r"[^\u4e00-\u9fffA-Za-z0-9·]+", "", text).lower()


def too_similar(left: str, right: str) -> bool:
    left_grams = char_grams(normalize_for_lexical(left[:1200]), n=3)
    right_grams = char_grams(normalize_for_lexical(right[:1200]), n=3)
    if not left_grams or not right_grams:
        return False
    overlap = len(left_grams & right_grams) / min(len(left_grams), len(right_grams))
    return overlap > 0.72


def document_key(doc: Document) -> str:
    metadata = doc.metadata or {}
    chunk_id = metadata.get("chunk_id")
    if chunk_id:
        return str(chunk_id)
    source = metadata.get("source", "")
    chunk_index = metadata.get("chunk_index")
    if source and chunk_index is not None:
        return f"{source}:{chunk_index}"
    return hashlib.sha1(doc.page_content.encode("utf-8", errors="ignore")).hexdigest()


def format_distance(distance: float | None) -> str:
    return "n/a" if distance is None else f"{distance:.4f}"

