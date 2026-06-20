from __future__ import annotations

import json
from typing import Any

from .config import PROCESSED_DIR
from .utils import normalize_for_lexical


def load_alias_entries_for_corpora(corpus_names: list[str] | None) -> list[dict[str, str]]:
    names = [str(name).strip() for name in corpus_names or [] if str(name).strip()]
    if not names:
        return []
    aliases_path = PROCESSED_DIR / "aliases.json"
    if not aliases_path.exists():
        return []
    try:
        payload = json.loads(aliases_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    entries: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for corpus_name in names:
        for item in payload.get(corpus_name, []) or []:
            if not isinstance(item, dict):
                continue
            alias = str(item.get("alias", "")).strip()
            canonical = str(item.get("canonical", "")).strip()
            note = str(item.get("note", "")).strip()
            if not alias or not canonical:
                continue
            key = (corpus_name, alias, canonical)
            if key in seen:
                continue
            seen.add(key)
            entries.append(
                {
                    "corpus_name": corpus_name,
                    "alias": alias,
                    "canonical": canonical,
                    "note": note,
                }
            )
    return entries


def relevant_alias_entries(
    text: str,
    corpus_names: list[str] | None,
    *,
    include_all_if_empty: bool = False,
    limit: int = 24,
) -> list[dict[str, str]]:
    entries = load_alias_entries_for_corpora(corpus_names)
    normalized_text = normalize_for_lexical(text)
    relevant: list[dict[str, str]] = []
    for item in entries:
        alias = item.get("alias", "")
        canonical = item.get("canonical", "")
        if (
            normalize_for_lexical(alias) in normalized_text
            or normalize_for_lexical(canonical) in normalized_text
        ):
            relevant.append(item)
            if len(relevant) >= limit:
                return relevant
    if include_all_if_empty and not relevant:
        return entries[:limit]
    return relevant


def alias_pairs_for_query(query: str, corpus_names: list[str] | None) -> list[tuple[str, str]]:
    return [
        (item["alias"], item["canonical"])
        for item in relevant_alias_entries(query, corpus_names)
        if item.get("alias") and item.get("canonical")
    ]


def render_alias_hints(entries: list[dict[str, Any]] | None) -> str:
    if not entries:
        return "None"
    lines: list[str] = []
    for item in entries:
        alias = str(item.get("alias", "")).strip()
        canonical = str(item.get("canonical", "")).strip()
        corpus_name = str(item.get("corpus_name", "")).strip()
        note = str(item.get("note", "")).strip()
        if not alias or not canonical:
            continue
        suffix = f" ({note})" if note else ""
        prefix = f"[{corpus_name}] " if corpus_name else ""
        lines.append(f"- {prefix}{alias} -> {canonical}{suffix}")
    return "\n".join(lines) if lines else "None"
