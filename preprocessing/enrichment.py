from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai import OpenAI

from env_loader import load_project_env
from llm.client import DEEPSEEK_MODEL, normalize_deepseek_model
from core.config import (
    DEEPSEEK_CHAPTER_CHAR_LIMIT,
    DEEPSEEK_CHUNK_CHAR_LIMIT,
    DEEPSEEK_PREPROCESS_ENABLED,
    DEEPSEEK_ROLE_INDEX_CHAR_LIMIT,
    PREPROCESS_CACHE_DIR,
)


load_project_env()


# DeepSeek 凭证与预处理模型名保留在此(凭证/模型不集中,见 core/config.py 说明);
# 其余预处理 LLM 旋钮(开关、字符上限、缓存目录)已收口至 core.config。
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_BASE = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
DEEPSEEK_PREPROCESS_MODEL = os.getenv("RAG_PREPROCESS_DEEPSEEK_MODEL", DEEPSEEK_MODEL)
RELATION_TYPES = ("family", "friend", "enemy", "mentor", "helper")


@dataclass(slots=True)
class LLMEnrichment:
    summary: str = ""
    keywords: list[str] = field(default_factory=list)
    metadata: dict[str, list[str]] = field(default_factory=dict)
    relations: list[dict[str, Any]] = field(default_factory=list)
    used_llm: bool = False


@dataclass(slots=True)
class RoleIndexEnrichment:
    summary: str = ""
    major_characters: list[str] = field(default_factory=list)
    female_characters: list[str] = field(default_factory=list)
    male_characters: list[str] = field(default_factory=list)
    important_relationships: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    used_llm: bool = False


class DeepSeekEnricher:
    def __init__(
        self,
        *,
        enabled: bool | None = None,
        model: str = DEEPSEEK_PREPROCESS_MODEL,
        cache_dir: Path = PREPROCESS_CACHE_DIR,
    ) -> None:
        self.enabled = DEEPSEEK_PREPROCESS_ENABLED if enabled is None else enabled
        self.model = normalize_deepseek_model(model)
        self.cache_dir = cache_dir
        self._client: OpenAI | None = None
        self._request_count = 0
        self._cache_hit_count = 0
        self._last_heartbeat = time.monotonic()
        # 建库时增强调用会被多个线程并发触发,用锁保护计数器与客户端惰性初始化
        self._lock = threading.Lock()

    def is_available(self) -> bool:
        return bool(self.enabled and DEEPSEEK_API_KEY)

    def enrich_chapter(
        self,
        *,
        title: str,
        text: str,
        fallback_summary: str,
        fallback_keywords: list[str],
        fallback_metadata: dict[str, list[str]],
    ) -> LLMEnrichment:
        if not self.is_available():
            return LLMEnrichment()
        prompt_payload = {
            "scope": "chapter",
            "title": title,
            "text": text[:DEEPSEEK_CHAPTER_CHAR_LIMIT],
            "fallback_summary": fallback_summary,
            "fallback_keywords": fallback_keywords,
            "fallback_metadata": fallback_metadata,
        }
        return self._request_enrichment(prompt_payload)

    def enrich_chunk(
        self,
        *,
        chapter_title: str,
        text: str,
        fallback_keywords: list[str],
        fallback_metadata: dict[str, list[str]],
    ) -> LLMEnrichment:
        if not self.is_available():
            return LLMEnrichment()
        prompt_payload = {
            "scope": "chunk",
            "chapter_title": chapter_title,
            "text": text[:DEEPSEEK_CHUNK_CHAR_LIMIT],
            "fallback_keywords": fallback_keywords,
            "fallback_metadata": fallback_metadata,
        }
        return self._request_enrichment(prompt_payload)

    def enrich_role_index(
        self,
        *,
        title: str,
        scope_label: str,
        evidence_text: str,
        fallback_major_characters: list[str],
        fallback_relationships: list[str],
        fallback_keywords: list[str],
    ) -> RoleIndexEnrichment:
        if not self.is_available():
            return RoleIndexEnrichment()
        prompt_payload = {
            "scope": "role_index",
            "title": title,
            "scope_label": scope_label,
            "evidence_text": evidence_text[:DEEPSEEK_ROLE_INDEX_CHAR_LIMIT],
            "fallback_major_characters": fallback_major_characters,
            "fallback_relationships": fallback_relationships,
            "fallback_keywords": fallback_keywords,
        }
        return self._request_role_index(prompt_payload)

    def _request_enrichment(self, payload: dict[str, Any]) -> LLMEnrichment:
        cache_path = self._cache_path(payload)
        cached = _load_cached_enrichment(cache_path)
        if cached is not None:
            with self._lock:
                self._cache_hit_count += 1
            self._maybe_log_heartbeat(scope=str(payload.get("scope", "unknown")), from_cache=True)
            return cached

        with self._lock:
            self._request_count += 1
            request_no = self._request_count
        scope = str(payload.get("scope", "unknown"))
        print(
            f"[INFO] DeepSeek enrichment request #{request_no} started for scope={scope}."
        )
        try:
            response = self._client_instance().chat.completions.create(
                model=self.model,
                temperature=0.1,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _system_prompt()},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
            )
            raw_content = response.choices[0].message.content or "{}"
            parsed = json.loads(raw_content)
            enrichment = _normalize_enrichment(parsed)
        except Exception:
            enrichment = LLMEnrichment()

        if enrichment.used_llm:
            _write_cached_enrichment(cache_path, enrichment)
        self._maybe_log_heartbeat(scope=scope, from_cache=False)
        return enrichment

    def _client_instance(self) -> OpenAI:
        with self._lock:
            if self._client is None:
                self._client = OpenAI(
                    api_key=DEEPSEEK_API_KEY,
                    base_url=DEEPSEEK_API_BASE,
                )
            return self._client

    def _cache_path(self, payload: dict[str, Any]) -> Path:
        serialized = json.dumps(
            {
                "model": self.model,
                "payload": payload,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        digest = hashlib.sha1(serialized.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def _request_role_index(self, payload: dict[str, Any]) -> RoleIndexEnrichment:
        cache_path = self._cache_path(payload)
        cached = _load_cached_role_index_enrichment(cache_path)
        if cached is not None:
            with self._lock:
                self._cache_hit_count += 1
            self._maybe_log_heartbeat(scope="role_index", from_cache=True)
            return cached

        with self._lock:
            self._request_count += 1
            request_no = self._request_count
        print(f"[INFO] DeepSeek role index request #{request_no} started.")
        try:
            response = self._client_instance().chat.completions.create(
                model=self.model,
                temperature=0.1,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _role_index_system_prompt()},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
            )
            raw_content = response.choices[0].message.content or "{}"
            parsed = json.loads(raw_content)
            enrichment = _normalize_role_index_enrichment(parsed)
        except Exception:
            enrichment = RoleIndexEnrichment()

        if enrichment.used_llm:
            _write_cached_role_index_enrichment(cache_path, enrichment)
        self._maybe_log_heartbeat(scope="role_index", from_cache=False)
        return enrichment

    def _maybe_log_heartbeat(self, *, scope: str, from_cache: bool) -> None:
        now = time.monotonic()
        with self._lock:
            if now - self._last_heartbeat < 30:
                return
            self._last_heartbeat = now
            requests = self._request_count
            cache_hits = self._cache_hit_count
        source = "cache" if from_cache else "api"
        print(
            "[INFO] DeepSeek enrichment heartbeat: "
            f"requests={requests}, cache_hits={cache_hits}, "
            f"latest_scope={scope}, latest_source={source}."
        )


def build_enricher(enabled: bool | None = None, *, model: str | None = None) -> DeepSeekEnricher | None:
    enricher = DeepSeekEnricher(enabled=enabled, model=model or DEEPSEEK_PREPROCESS_MODEL)
    if not enricher.is_available():
        return None
    return enricher


def merge_keywords(*lists: list[str], limit: int) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for values in lists:
        for value in values:
            cleaned = str(value).strip()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            merged.append(cleaned)
            if len(merged) >= limit:
                return merged
    return merged


def merge_metadata(
    *,
    base: dict[str, object],
    enriched: dict[str, object],
    limit: int = 8,
) -> dict[str, object]:
    merged = {key: _copy_metadata_value(value) for key, value in base.items()}
    for key, values in enriched.items():
        if not isinstance(values, list):
            continue
        existing = merged.get(key, [])
        existing_list = existing if isinstance(existing, list) else []
        merged[key] = merge_keywords(values, existing_list, limit=limit)
    return merged


def _copy_metadata_value(value: object) -> object:
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    return value


def _system_prompt() -> str:
    return (
        "You are a strict Chinese fiction/script preprocessing engine.\n"
        "Return only JSON.\n"
        "Be conservative: extract only what is explicit or strongly supported by the input text.\n"
        "Use short canonical Chinese names when possible.\n"
        "Allowed relation_type values: family, friend, enemy, mentor, helper.\n"
        "JSON schema:\n"
        "{"
        "\"summary\": string,"
        "\"keywords\": string[],"
        "\"persons\": string[],"
        "\"locations\": string[],"
        "\"events\": string[],"
        "\"objects\": string[],"
        "\"aliases\": string[],"
        "\"relations\": ["
        "{\"person_a\": string, \"person_b\": string, \"relation_type\": string, \"confidence\": number, \"evidence\": string}"
        "]"
        "}"
    )


def _role_index_system_prompt() -> str:
    return (
        "You are a strict Chinese fiction character roster aggregation engine.\n"
        "Return only JSON.\n"
        "Infer only from the provided evidence. Be conservative.\n"
        "If the evidence is insufficient, leave the list empty instead of guessing.\n"
        "Use short canonical Chinese names when possible.\n"
        "JSON schema:\n"
        "{"
        "\"summary\": string,"
        "\"major_characters\": string[],"
        "\"female_characters\": string[],"
        "\"male_characters\": string[],"
        "\"important_relationships\": string[],"
        "\"keywords\": string[]"
        "}"
    )


def _normalize_enrichment(payload: dict[str, Any]) -> LLMEnrichment:
    metadata = {
        "persons": _normalize_string_list(payload.get("persons"), limit=8),
        "locations": _normalize_string_list(payload.get("locations"), limit=8),
        "events": _normalize_string_list(payload.get("events"), limit=8),
        "objects": _normalize_string_list(payload.get("objects"), limit=8),
        "aliases": _normalize_string_list(payload.get("aliases"), limit=8),
        "keywords": _normalize_string_list(payload.get("keywords"), limit=12),
    }
    relations = _normalize_relations(payload.get("relations"))
    return LLMEnrichment(
        summary=str(payload.get("summary", "")).strip(),
        keywords=metadata["keywords"],
        metadata=metadata,
        relations=relations,
        used_llm=bool(metadata["keywords"] or metadata["persons"] or relations or str(payload.get("summary", "")).strip()),
    )


def _normalize_role_index_enrichment(payload: dict[str, Any]) -> RoleIndexEnrichment:
    summary = str(payload.get("summary", "")).strip()
    major_characters = _normalize_string_list(payload.get("major_characters"), limit=12)
    female_characters = _normalize_string_list(payload.get("female_characters"), limit=10)
    male_characters = _normalize_string_list(payload.get("male_characters"), limit=10)
    important_relationships = _normalize_string_list(payload.get("important_relationships"), limit=12)
    keywords = _normalize_string_list(payload.get("keywords"), limit=16)
    return RoleIndexEnrichment(
        summary=summary,
        major_characters=major_characters,
        female_characters=female_characters,
        male_characters=male_characters,
        important_relationships=important_relationships,
        keywords=keywords,
        used_llm=bool(
            summary
            or major_characters
            or female_characters
            or male_characters
            or important_relationships
            or keywords
        ),
    )


def _normalize_relations(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    relations: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        person_a = str(item.get("person_a", "")).strip()
        person_b = str(item.get("person_b", "")).strip()
        relation_type = str(item.get("relation_type", "")).strip().lower()
        if not person_a or not person_b or person_a == person_b:
            continue
        if relation_type not in RELATION_TYPES:
            continue
        try:
            confidence = float(item.get("confidence", 0.0) or 0.0)
        except Exception:
            confidence = 0.0
        evidence = str(item.get("evidence", "")).strip()
        relations.append(
            {
                "person_a": person_a,
                "person_b": person_b,
                "relation_type": relation_type,
                "confidence": max(0.0, min(1.0, confidence)),
                "evidence": evidence,
            }
        )
    return relations[:12]


def _normalize_string_list(value: Any, *, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    items: list[str] = []
    seen: set[str] = set()
    for raw in value:
        cleaned = str(raw).strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        items.append(cleaned)
        if len(items) >= limit:
            break
    return items


def _load_cached_enrichment(path: Path) -> LLMEnrichment | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return _normalize_enrichment(payload)


def _load_cached_role_index_enrichment(path: Path) -> RoleIndexEnrichment | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return _normalize_role_index_enrichment(payload)


def _write_cached_enrichment(path: Path, enrichment: LLMEnrichment) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": enrichment.summary,
        "keywords": enrichment.keywords,
        "persons": enrichment.metadata.get("persons", []),
        "locations": enrichment.metadata.get("locations", []),
        "events": enrichment.metadata.get("events", []),
        "objects": enrichment.metadata.get("objects", []),
        "aliases": enrichment.metadata.get("aliases", []),
        "relations": enrichment.relations,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_cached_role_index_enrichment(path: Path, enrichment: RoleIndexEnrichment) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": enrichment.summary,
        "major_characters": enrichment.major_characters,
        "female_characters": enrichment.female_characters,
        "male_characters": enrichment.male_characters,
        "important_relationships": enrichment.important_relationships,
        "keywords": enrichment.keywords,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
