from __future__ import annotations

from pathlib import Path

from .metadata import metadata_list, restore_runtime_metadata
from .models import RankedChunk
from .utils import format_distance


def format_context(chunks: list[RankedChunk]) -> str:
    parts = []
    for idx, item in enumerate(chunks, start=1):
        doc = item.document
        metadata = restore_runtime_metadata(doc.metadata or {})
        source = metadata.get("doc_name") or Path(metadata.get("source", "unknown")).name
        chapter = metadata.get("chapter") or "Unknown Chapter"
        chunk_index = metadata.get("chunk_index", "?")
        persons = ", ".join(metadata_list(metadata, "persons")[:4]) or "-"
        events = ", ".join(metadata_list(metadata, "events")[:3]) or "-"
        relation_types = ", ".join(metadata_list(metadata, "relation_types")[:3]) or "-"
        has_expanded_context = bool(metadata.get("has_expanded_context"))
        context_role = "Expanded Context Evidence" if has_expanded_context else ("Expanded Neighbor" if item.is_context_expansion else "Primary Evidence")
        expansion_reason = str(metadata.get("expansion_reason", "")).strip()
        expansion_flag = " | expanded-neighbor" if item.is_context_expansion else ""
        if has_expanded_context:
            expansion_flag = " | expanded-context"
        header = (
            f"[Chunk {idx} | {source} | {chapter} | chunk={chunk_index} | "
            f"score={item.score:.3f} | distance={format_distance(item.distance)} | "
            f"persons={persons} | events={events} | relations={relation_types}{expansion_flag}]"
        )
        block_lines = [header, f"[Context Role] {context_role}"]
        if expansion_reason:
            block_lines.append(f"[Expansion Reason] {expansion_reason}")
        block_lines.append(str(metadata.get("expanded_context_text") or doc.page_content).strip())
        parts.append("\n".join(block_lines))
    return "\n\n".join(parts)


def format_debug_table(chunks: list[RankedChunk]) -> list[dict[str, object]]:
    rows = []
    for idx, item in enumerate(chunks, start=1):
        metadata = restore_runtime_metadata(item.document.metadata or {})
        rows.append(
            {
                "rank": idx,
                "score": round(item.score, 4),
                "dense": round(item.dense_score, 4),
                "lexical": round(item.lexical_score, 4),
                "metadata": round(item.metadata_score, 4),
                "summary": round(item.summary_score, 4),
                "relation": round(item.relation_score, 4),
                "position": round(item.position_score, 4),
                "volume_score": metadata.get("volume_score"),
                "placeholder_penalty": metadata.get("placeholder_penalty"),
                "character_list_bonus": metadata.get("character_list_bonus"),
                "distance": None if item.distance is None else round(item.distance, 4),
                "hit_keywords": ", ".join(metadata_list(metadata, "hit_keywords")),
                "persons": ", ".join(metadata_list(metadata, "persons")[:4]),
                "relation_persons": ", ".join(metadata_list(metadata, "relation_persons")[:4]),
                "relation_types": ", ".join(metadata_list(metadata, "relation_types")[:3]),
                "events": ", ".join(metadata_list(metadata, "events")[:3]),
                "synthetic_role_index": bool(metadata.get("is_synthetic_role_index")),
                "female_characters": ", ".join(metadata_list(metadata, "female_characters")[:4]),
                "volume": metadata.get("volume_label"),
                "is_context_expansion": item.is_context_expansion,
                "has_expanded_context": bool(metadata.get("has_expanded_context")),
                "expanded_prev_chunk_id": metadata.get("expanded_prev_chunk_id", ""),
                "expanded_next_chunk_id": metadata.get("expanded_next_chunk_id", ""),
                "expansion_reason": metadata.get("expansion_reason", ""),
                "chapter": metadata.get("chapter", "Unknown Chapter"),
                "chunk": metadata.get("chunk_index", "?"),
            }
        )
    return rows
