from __future__ import annotations

from typing import Any

from langchain_core.documents import Document

from retrieval import RankedChunk, format_context
from retrieval.metadata import metadata_list, restore_runtime_metadata


def build_followup_context_options(chunks: list[RankedChunk]) -> list[dict[str, Any]]:
    chunks = _merge_contiguous_evidence_chunks(chunks)
    options: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks, start=1):
        metadata = restore_runtime_metadata(dict(chunk.document.metadata or {}))
        source = str(metadata.get("doc_name") or metadata.get("source", "unknown")).strip() or "unknown"
        chapter = str(metadata.get("chapter") or "Unknown Chapter").strip()
        chunk_index = metadata.get("chunk_index", "?")
        context_role = "expanded_neighbor" if chunk.is_context_expansion else "primary_evidence"
        context_role_label = "Expanded Neighbor" if chunk.is_context_expansion else "Primary Evidence"
        expanded_context_text = str(metadata.get("expanded_context_text") or "").strip()
        if expanded_context_text:
            context_role = "expanded_context"
            context_role_label = "Expanded Context"
        if str(metadata.get("context_merge_reason") or "") == "continuous_retrieval_chunks":
            context_role = "merged_context"
            context_role_label = "Merged Evidence"
        label = f"{source} / {chapter} / chunk {chunk_index}"
        payload_text = expanded_context_text or chunk.document.page_content
        preview = payload_text.strip().replace("\n", " ")
        options.append(
            {
                "option_id": str(metadata.get("chunk_id") or f"chunk-{index}"),
                "label": label,
                "preview": preview[:220],
                "page_content": payload_text,
                "original_page_content": chunk.document.page_content,
                "metadata": metadata,
                "score": float(chunk.score),
                "distance": _coerce_float(chunk.distance),
                "dense_score": float(chunk.dense_score),
                "lexical_score": float(chunk.lexical_score),
                "metadata_score": float(chunk.metadata_score),
                "summary_score": float(chunk.summary_score),
                "relation_score": float(chunk.relation_score),
                "position_score": float(chunk.position_score),
                "is_context_expansion": bool(chunk.is_context_expansion),
                "has_expanded_context": bool(expanded_context_text),
                "context_role": context_role,
                "context_role_label": context_role_label,
                "expansion_reason": str(metadata.get("expansion_reason", "") or ""),
                "context_merge_reason": str(metadata.get("context_merge_reason", "") or ""),
                "merged_chunk_start_index": metadata.get("merged_chunk_start_index"),
                "merged_chunk_end_index": metadata.get("merged_chunk_end_index"),
                "merged_chunk_ids": metadata_list(metadata, "merged_chunk_ids"),
                "persons": metadata_list(metadata, "persons"),
                "events": metadata_list(metadata, "events"),
                "chapter": chapter,
                "source": source,
                "chunk_index": str(chunk_index),
                "prev_chunk_id": str(metadata.get("prev_chunk_id") or ""),
                "next_chunk_id": str(metadata.get("next_chunk_id") or ""),
            }
        )
    return options


def _merge_contiguous_evidence_chunks(chunks: list[RankedChunk]) -> list[RankedChunk]:
    sortable: list[dict[str, Any]] = []
    passthrough: list[tuple[int, RankedChunk]] = []
    for rank, chunk in enumerate(chunks):
        metadata = restore_runtime_metadata(dict(chunk.document.metadata or {}))
        chunk_index = _coerce_int(metadata.get("chunk_index"))
        if chunk_index is None:
            passthrough.append((rank, chunk))
            continue
        start_index = chunk_index
        end_index = chunk_index
        expanded_text = str(metadata.get("expanded_context_text") or "").strip()
        if expanded_text:
            if str(metadata.get("expanded_prev_chunk_id") or "").strip():
                start_index = min(start_index, chunk_index - 1)
            if str(metadata.get("expanded_next_chunk_id") or "").strip():
                end_index = max(end_index, chunk_index + 1)
        sortable.append(
            {
                "rank": rank,
                "chunk": chunk,
                "metadata": metadata,
                "key": _evidence_group_key(metadata),
                "start": start_index,
                "end": end_index,
                "text": expanded_text or chunk.document.page_content,
            }
        )

    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for item in sortable:
        groups.setdefault(item["key"], []).append(item)

    merged: list[tuple[int, RankedChunk]] = list(passthrough)
    for items in groups.values():
        items.sort(key=lambda item: (item["start"], item["end"], item["rank"]))
        current: list[dict[str, Any]] = []
        current_end: int | None = None
        for item in items:
            if not current:
                current = [item]
                current_end = int(item["end"])
                continue
            if int(item["start"]) <= int(current_end or item["end"]) + 1:
                current.append(item)
                current_end = max(int(current_end or item["end"]), int(item["end"]))
                continue
            merged.append(_merge_evidence_group(current))
            current = [item]
            current_end = int(item["end"])
        if current:
            merged.append(_merge_evidence_group(current))

    merged.sort(key=lambda pair: pair[0])
    return [chunk for _rank, chunk in merged]


def _merge_evidence_group(items: list[dict[str, Any]]) -> tuple[int, RankedChunk]:
    if len(items) == 1 and int(items[0]["start"]) == int(items[0]["end"]):
        return int(items[0]["rank"]), items[0]["chunk"]

    items = sorted(items, key=lambda item: (int(item["start"]), int(item["end"]), int(item["rank"])))
    first = items[0]
    best = max(items, key=lambda item: float(item["chunk"].score))
    base_chunk: RankedChunk = best["chunk"]
    metadata = restore_runtime_metadata(dict(base_chunk.document.metadata or {}))
    start = min(int(item["start"]) for item in items)
    end = max(int(item["end"]) for item in items)
    merged_text = ""
    chunk_ids: list[str] = []
    persons: list[str] = []
    events: list[str] = []
    reasons: list[str] = []

    for item in items:
        item_metadata = restore_runtime_metadata(dict(item["metadata"] or {}))
        text = str(item.get("text") or "").strip()
        if text:
            merged_text = _append_without_overlap(merged_text, text) if merged_text else text
            merged_text = _collapse_adjacent_duplicate_lines(merged_text)
        chunk_id = str(item_metadata.get("chunk_id") or "").strip()
        if chunk_id and chunk_id not in chunk_ids:
            chunk_ids.append(chunk_id)
        for value in metadata_list(item_metadata, "persons"):
            if value not in persons:
                persons.append(value)
        for value in metadata_list(item_metadata, "events"):
            if value not in events:
                events.append(value)
        reason = str(item_metadata.get("expansion_reason") or "").strip()
        if reason and reason not in reasons:
            reasons.append(reason)

    label_index = str(start) if start == end else f"{start}-{end}"
    metadata["chunk_index"] = label_index
    metadata["merged_chunk_start_index"] = start
    metadata["merged_chunk_end_index"] = end
    metadata["merged_chunk_ids"] = chunk_ids
    metadata["expanded_context_text"] = merged_text
    metadata["has_expanded_context"] = True
    metadata["context_merge_reason"] = "continuous_retrieval_chunks"
    if reasons:
        metadata["expansion_reason"] = "；".join(reasons)
    if persons:
        metadata["persons"] = persons[:12]
    if events:
        metadata["events"] = events[:10]
    if chunk_ids:
        metadata["chunk_id"] = f"{chunk_ids[0]}..{chunk_ids[-1]}" if len(chunk_ids) > 1 else chunk_ids[0]

    return (
        min(int(item["rank"]) for item in items),
        RankedChunk(
            document=Document(page_content=merged_text, metadata=metadata),
            distance=_min_distance([item["chunk"].distance for item in items]),
            dense_score=max(float(item["chunk"].dense_score) for item in items),
            lexical_score=max(float(item["chunk"].lexical_score) for item in items),
            metadata_score=max(float(item["chunk"].metadata_score) for item in items),
            summary_score=max(float(item["chunk"].summary_score) for item in items),
            relation_score=max(float(item["chunk"].relation_score) for item in items),
            position_score=max(float(item["chunk"].position_score) for item in items),
            score=max(float(item["chunk"].score) for item in items),
            is_context_expansion=False,
        ),
    )


def _evidence_group_key(metadata: dict[str, Any]) -> tuple[str, ...]:
    return (
        str(metadata.get("corpus_name") or "").strip(),
        str(metadata.get("doc_id") or "").strip(),
        str(metadata.get("source") or metadata.get("doc_name") or "").strip(),
        str(metadata.get("chapter_id") or metadata.get("chapter") or "").strip(),
    )


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _min_distance(values: list[float | None]) -> float | None:
    concrete = [float(value) for value in values if value is not None]
    return min(concrete) if concrete else None


def _append_without_overlap(left: str, right: str, *, min_overlap: int = 4, max_scan: int = 300) -> str:
    left = str(left or "").strip()
    right = str(right or "").strip()
    if not left:
        return right
    if not right:
        return left
    if right in left:
        return left
    if left in right:
        return right
    max_overlap = min(len(left), len(right), max_scan)
    min_overlap = min(min_overlap, max(1, max_overlap))
    best = 0
    for size in range(max_overlap, min_overlap - 1, -1):
        if left[-size:] == right[:size]:
            best = size
            break
    if best:
        return left + right[best:]
    return left + "\n" + right


def _collapse_adjacent_duplicate_lines(text: str) -> str:
    lines = [line.strip() for line in str(text or "").splitlines()]
    collapsed: list[str] = []
    for line in lines:
        if not line:
            continue
        if collapsed and collapsed[-1] == line:
            continue
        collapsed.append(line)
    return "\n".join(collapsed)


def format_selected_contexts(selected_contexts: list[dict[str, Any] | object] | None) -> str:
    chunks = normalize_selected_contexts(selected_contexts)
    if not chunks:
        return ""
    return format_context(chunks)


def normalize_selected_contexts(selected_contexts: list[dict[str, Any] | object] | None) -> list[RankedChunk]:
    normalized: list[RankedChunk] = []
    if not selected_contexts:
        return normalized

    for item in selected_contexts:
        if isinstance(item, RankedChunk):
            normalized.append(_clone_ranked_chunk(item))
            continue
        if isinstance(item, dict):
            chunk = _ranked_chunk_from_payload(item)
            if chunk is not None:
                normalized.append(chunk)
    return normalized


def _clone_ranked_chunk(chunk: RankedChunk) -> RankedChunk:
    metadata = restore_runtime_metadata(dict(chunk.document.metadata or {}))
    metadata["selected_for_followup"] = True
    return RankedChunk(
        document=Document(page_content=chunk.document.page_content, metadata=metadata),
        distance=chunk.distance,
        dense_score=chunk.dense_score,
        lexical_score=chunk.lexical_score,
        metadata_score=chunk.metadata_score,
        summary_score=chunk.summary_score,
        relation_score=chunk.relation_score,
        position_score=chunk.position_score,
        score=chunk.score,
        is_context_expansion=chunk.is_context_expansion,
    )


def _ranked_chunk_from_payload(payload: dict[str, Any]) -> RankedChunk | None:
    page_content = str(payload.get("page_content", "") or "").strip()
    metadata = payload.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    metadata = restore_runtime_metadata(dict(metadata))
    if not page_content:
        return None

    metadata["selected_for_followup"] = True
    is_context_expansion = bool(payload.get("is_context_expansion"))
    return RankedChunk(
        document=Document(page_content=page_content, metadata=metadata),
        distance=_coerce_float(payload.get("distance")),
        dense_score=_coerce_float(payload.get("dense_score"), default=0.0),
        lexical_score=_coerce_float(payload.get("lexical_score"), default=0.0),
        metadata_score=_coerce_float(payload.get("metadata_score"), default=0.0),
        summary_score=_coerce_float(payload.get("summary_score"), default=0.0),
        relation_score=_coerce_float(payload.get("relation_score"), default=0.0),
        position_score=_coerce_float(payload.get("position_score"), default=0.0),
        score=_coerce_float(payload.get("score"), default=0.0),
        is_context_expansion=is_context_expansion,
    )


def _coerce_float(value: Any, *, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        return default
