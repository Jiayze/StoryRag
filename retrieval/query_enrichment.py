from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai import OpenAI

from env_loader import load_project_env
from core import get_logger
from llm.client import DEEPSEEK_MODEL, create_deepseek_client, normalize_deepseek_model
from core.config import QUERY_ENRICHMENT_CACHE_DIR, QUERY_ENRICHMENT_ENABLED


load_project_env()


logger = get_logger(__name__)


# 查询增强模型名保留在此(默认回退到 DEEPSEEK_MODEL,属模型层);
# 开关与缓存目录已收口至 core.config。
QUERY_ENRICHMENT_MODEL = os.getenv("RAG_QUERY_PREPROCESS_DEEPSEEK_MODEL", DEEPSEEK_MODEL)
QUERY_ENRICHMENT_CACHE_VERSION = "v3"


@dataclass(slots=True)
class QueryEnrichment:
    rewritten_query: str = ""
    core_question: str = ""
    retrieval_focus: str = ""
    premise_claims: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    persons: list[str] = field(default_factory=list)
    locations: list[str] = field(default_factory=list)
    events: list[str] = field(default_factory=list)
    objects: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    relation_intents: list[str] = field(default_factory=list)
    query_modes: list[str] = field(default_factory=list)
    is_open_ended: bool = False
    used_llm: bool = False


class DeepSeekQueryEnricher:
    def __init__(
        self,
        *,
        enabled: bool | None = None,
        model: str = QUERY_ENRICHMENT_MODEL,
        cache_dir: Path = QUERY_ENRICHMENT_CACHE_DIR,
    ) -> None:
        self.enabled = QUERY_ENRICHMENT_ENABLED if enabled is None else enabled
        self.model = normalize_deepseek_model(model)
        self.cache_dir = cache_dir
        self._client: OpenAI | None = None

    def is_available(self) -> bool:
        return self.enabled and bool(os.getenv("DEEPSEEK_API_KEY"))

    def enrich_query(
        self,
        *,
        query: str,
        heuristic_payload: dict[str, Any],
    ) -> QueryEnrichment:
        if not self.is_available():
            return QueryEnrichment()

        payload = {
            "query": query,
            "heuristic_payload": heuristic_payload,
        }
        cache_path = self._cache_path(payload)
        cached = _load_cached_query_enrichment(cache_path)
        if cached is not None:
            return cached

        logger.info(f"DeepSeek query preprocessing started for query: {query[:80]}")
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
            enrichment = _normalize_query_enrichment(parsed)
        except Exception:
            enrichment = QueryEnrichment()

        if enrichment.used_llm:
            _write_cached_query_enrichment(cache_path, enrichment)
        return enrichment

    def _client_instance(self) -> OpenAI:
        if self._client is None:
            self._client = create_deepseek_client()
        return self._client

    def _cache_path(self, payload: dict[str, Any]) -> Path:
        serialized = json.dumps(
            {"version": QUERY_ENRICHMENT_CACHE_VERSION, "model": self.model, "payload": payload},
            ensure_ascii=False,
            sort_keys=True,
        )
        digest = hashlib.sha1(serialized.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"


def build_query_enricher(enabled: bool | None = None, *, model: str | None = None) -> DeepSeekQueryEnricher | None:
    enricher = DeepSeekQueryEnricher(enabled=enabled, model=model or QUERY_ENRICHMENT_MODEL)
    if not enricher.is_available():
        return None
    return enricher


def _system_prompt() -> str:
    return (
        "You are a strict Chinese fiction retrieval query preprocessing engine.\n"
        "Return only JSON.\n"
        "Your job is to preserve rare terms, aliases, slurs, titles, and proper nouns exactly as they appear in the query when useful.\n"
        "The user payload may contain heuristic_payload.user_alias_hints entries like alias -> canonical. Treat these as user-maintained lexicon hints for understanding and retrieval.\n"
        "When a user alias appears in the query, include both the alias and canonical name in keywords/aliases/persons when useful. Do not reinterpret the alias as a generic common word.\n"
        "Do not split a single Chinese name or term into characters.\n"
        "If the query contains a domain-specific term like 泥巴种, keep it as one keyword.\n"
        "Be conservative and useful for retrieval.\n"
        "Separate the retrieval target from the answer request. The retrieval target should be the concrete scene/event/entity to find in the text; the answer request can ask for volume/chapter/location.\n"
        "For chapter_locator questions like '动画最后的场景（老八和温水坐摩天轮）在小说中是第几卷第几章', set core_question/retrieval_focus to the concrete scene only, e.g. '老八/八奈见和温水坐摩天轮的场景', and put the locator request in query_modes, not keywords.\n"
        "Do not include generic locator/request words in keywords, such as 动画最后, 小说中, 第几卷, 第几章, 在哪里, 哪一章, 查找, 场景所在. Keep keywords focused on entities and scene terms.\n"
        "Keywords must be complete semantic terms, not arbitrary character n-grams.\n"
        "Never output broken fragments such as 第八卷结, 尾天爱星, 样的一颗, or partial phrase splices.\n"
        "For volume hints, output clean terms like 第八卷, not text crossing into the next word.\n"
        "Allowed relation_intents: family, friend, enemy, mentor, helper.\n"
        "Allowed query_modes: fact, relation, causal, chapter_locator, first_appearance, character_list, open_ended.\n"
        "Set is_open_ended=true when the user asks for interpretation, analysis, motivations, themes,评价,看法,原因, or other broad explanatory questions that may need synthesis across evidence.\n"
        "Treat the user's narrative setup as claims to be verified, not as ground truth.\n"
        "Extract the smallest neutral core question for retrieval, and separate any factual premises the user assumed.\n"
        "If you cannot reliably infer a field, return an empty string or empty list. Never output placeholders like unknown, none, or n/a.\n"
        "JSON schema:\n"
        "{"
        "\"rewritten_query\": string,"
        "\"core_question\": string,"
        "\"retrieval_focus\": string,"
        "\"premise_claims\": string[],"
        "\"keywords\": string[],"
        "\"persons\": string[],"
        "\"locations\": string[],"
        "\"events\": string[],"
        "\"objects\": string[],"
        "\"aliases\": string[],"
        "\"relation_intents\": string[],"
        "\"query_modes\": string[],"
        "\"is_open_ended\": boolean"
        "}"
    )


def _normalize_query_enrichment(payload: dict[str, Any]) -> QueryEnrichment:
    rewritten_query = str(payload.get("rewritten_query", "")).strip()
    core_question = str(payload.get("core_question", "")).strip()
    retrieval_focus = str(payload.get("retrieval_focus", "")).strip()
    premise_claims = _normalize_string_list(payload.get("premise_claims"), limit=8)
    keywords = _normalize_string_list(payload.get("keywords"), limit=12)
    persons = _normalize_string_list(payload.get("persons"), limit=8)
    locations = _normalize_string_list(payload.get("locations"), limit=8)
    events = _normalize_string_list(payload.get("events"), limit=8)
    objects = _normalize_string_list(payload.get("objects"), limit=8)
    aliases = _normalize_string_list(payload.get("aliases"), limit=8)
    relation_intents = [
        value
        for value in _normalize_string_list(payload.get("relation_intents"), limit=5)
        if value in {"family", "friend", "enemy", "mentor", "helper"}
    ]
    query_modes = [
        value
        for value in _normalize_string_list(payload.get("query_modes"), limit=5)
        if value in {"fact", "relation", "causal", "chapter_locator", "first_appearance", "character_list", "open_ended"}
    ]
    is_open_ended = bool(payload.get("is_open_ended", False)) or "open_ended" in query_modes
    if is_open_ended and "open_ended" not in query_modes:
        query_modes.append("open_ended")
    used_llm = any(
        [
            rewritten_query,
            keywords,
            persons,
            locations,
            events,
            objects,
            aliases,
            relation_intents,
            query_modes,
            is_open_ended,
        ]
    )
    return QueryEnrichment(
        rewritten_query=rewritten_query,
        core_question=core_question,
        retrieval_focus=retrieval_focus,
        premise_claims=premise_claims,
        keywords=keywords,
        persons=persons,
        locations=locations,
        events=events,
        objects=objects,
        aliases=aliases,
        relation_intents=relation_intents,
        query_modes=query_modes,
        is_open_ended=is_open_ended,
        used_llm=used_llm,
    )


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


def _load_cached_query_enrichment(path: Path) -> QueryEnrichment | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return _normalize_query_enrichment(payload)


def _write_cached_query_enrichment(path: Path, enrichment: QueryEnrichment) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "rewritten_query": enrichment.rewritten_query,
        "core_question": enrichment.core_question,
        "retrieval_focus": enrichment.retrieval_focus,
        "premise_claims": enrichment.premise_claims,
        "keywords": enrichment.keywords,
        "persons": enrichment.persons,
        "locations": enrichment.locations,
        "events": enrichment.events,
        "objects": enrichment.objects,
        "aliases": enrichment.aliases,
        "relation_intents": enrichment.relation_intents,
        "query_modes": enrichment.query_modes,
        "is_open_ended": enrichment.is_open_ended,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
