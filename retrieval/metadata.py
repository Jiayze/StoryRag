from __future__ import annotations

from typing import Iterable


LIST_SEPARATOR = " | "
LIST_LIKE_METADATA_KEYS = (
    "persons",
    "locations",
    "events",
    "objects",
    "aliases",
    "keywords",
    "chapter_persons",
    "chapter_locations",
    "chapter_events",
    "chapter_objects",
    "chapter_keywords",
    "relation_persons",
    "relation_types",
    "hit_keywords",
    "hit_entities",
    "query_keywords",
    "target_roles",
    "female_characters",
    "male_characters",
    "important_relationships",
    "source_chunk_ids",
    "role_relation_types",
)


def metadata_list(metadata: dict, key: str) -> list[str]:
    return coerce_string_list(metadata.get(key))


def coerce_string_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        parts = raw.split(LIST_SEPARATOR) if LIST_SEPARATOR in raw else [raw]
        return _dedupe_strings(parts)
    if isinstance(value, (list, tuple, set)):
        return _dedupe_strings(value)
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray, dict)):
        return _dedupe_strings(value)
    cleaned = str(value).strip()
    return [cleaned] if cleaned else []


def restore_runtime_metadata(metadata: dict) -> dict:
    for key in LIST_LIKE_METADATA_KEYS:
        if key in metadata:
            metadata[key] = coerce_string_list(metadata.get(key))
    return metadata


def _dedupe_strings(values: Iterable[object]) -> list[str]:
    items: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = str(value).strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        items.append(cleaned)
    return items
