from __future__ import annotations

import json
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from .config import (
    BGE_QUERY_PREFIX,
    CHUNKS_PATH,
    DEFAULT_MAX_KEYWORDS,
    EVENT_HINTS,
    LOCATION_SUFFIXES,
    MIN_KNOWN_PERSON_FREQUENCY,
    OBJECT_HINTS,
    PERSON_NAME_BLACKLIST,
    PERSON_QUERY_NOISE,
    PERSON_TITLE_SUFFIXES,
    QUERY_PERSON_STOPWORDS,
    QUESTION_PHRASES,
    RELATION_INTENT_PATTERNS,
    RELATION_QUERY_HINTS,
    RELATIONS_PATH,
    STOP_KEYWORDS,
    PROCESSED_DIR,
)
from .aliases import alias_pairs_for_query, relevant_alias_entries
from .models import QueryPlan
from .query_enrichment import build_query_enricher
from .utils import normalize_for_lexical


QUERY_TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fffA-Za-z0-9路]+")
ENGLISH_NAME_PATTERN = re.compile(r"[A-Z][a-z]{1,20}(?:\s+[A-Z][a-z]{1,20})?")
CHINESE_PERSON_PATTERN = re.compile(r"[\u4e00-\u9fff]{2,4}")
VOLUME_PATTERN = re.compile(r"第\s*([0-9零〇一二两三四五六七八九十百]+)\s*([卷册部])")
QUERY_SPLIT_PATTERN = re.compile(
    r"(?:请问|告诉我|介绍一下|解释一下|是什么|是谁|在哪里|为什么|怎么样|怎么|如何|多少|哪一个|哪位|哪个|什么关系|关系|和|与|及|以及|还有|关于|有关|相关)"
)

CHARACTER_LIST_MARKERS = (
    "有哪些角色",
    "角色有谁",
    "人物有谁",
    "有谁",
    "是谁",
    "名单",
    "主要人物",
    "主要角色",
    "登场人物",
    "主角",
    "女主",
    "男主",
)
ROLE_HINTS = {
    "女主角": ("女主角", "女主", "女主人公"),
    "男主角": ("男主角", "男主", "男主人公"),
    "主角": ("主角", "主人公"),
    "主要角色": ("主要角色", "主要人物", "核心角色"),
    "角色": ("角色", "人物", "登场人物"),
}
ROLE_EXPANSIONS = {
    "女主角": ["女主角", "女主", "主角", "角色", "人物"],
    "男主角": ["男主角", "男主", "主角", "角色", "人物"],
    "主角": ["主角", "主人公", "角色", "人物"],
    "主要角色": ["主要角色", "主要人物", "角色", "人物"],
    "角色": ["角色", "人物", "登场人物"],
}
CN_NUMERAL_MAP = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}


def build_retrieval_query(query: str) -> str:
    query = query.strip()
    if not BGE_QUERY_PREFIX or query.startswith(BGE_QUERY_PREFIX):
        return query
    return f"{BGE_QUERY_PREFIX}{query}"


def analyze_query(
    query: str,
    max_keywords: int = DEFAULT_MAX_KEYWORDS,
    *,
    model: str | None = None,
    corpus_names: list[str] | None = None,
) -> QueryPlan:
    query = query.strip()
    relevant_aliases = relevant_alias_entries(query, corpus_names or [])
    alias_terms = [(item["alias"], item["canonical"]) for item in relevant_aliases]
    alias_values = [value for pair in alias_terms for value in pair]
    known_names = load_known_person_names()
    core_question = query
    retrieval_focus = query
    premise_claims: list[str] = []
    target_roles = _extract_target_roles(query)
    target_volume, target_volume_index = _extract_target_volume(query)

    persons = _merge_terms(_extract_query_persons(query), alias_values, limit=8)
    locations = _extract_query_locations(query)
    events = _extract_query_hints(query, EVENT_HINTS, max_items=4)
    objects = _extract_query_hints(query, OBJECT_HINTS, max_items=4)
    aliases: list[str] = []

    keywords = extract_keywords(
        query,
        max_keywords=max_keywords,
        reserved_terms=[*alias_values, *persons, *locations, *events, *objects],
    )
    if target_roles:
        keywords = _merge_terms(_expand_role_keywords(target_roles), keywords, limit=max_keywords)
    if target_volume:
        keywords = _merge_terms([target_volume], keywords, limit=max_keywords)

    relation_intents = tuple(sorted(_relation_intent_types_from_text(query, persons)))
    query_modes = tuple(_detect_query_modes(query, persons, relation_intents))
    retrieval_focus = _augment_retrieval_focus(retrieval_focus, target_roles, target_volume)
    locator_scene_hint = _extract_locator_scene_hint(query)

    heuristic_payload = {
        "persons": persons,
        "locations": locations,
        "events": events,
        "objects": objects,
        "aliases": aliases,
        "keywords": keywords,
        "relation_intents": list(relation_intents),
        "query_modes": list(query_modes),
        "target_roles": target_roles,
        "target_volume": target_volume,
        "user_alias_hints": relevant_aliases,
    }
    query_enricher = build_query_enricher(model=model)
    enriched = query_enricher.enrich_query(query=query, heuristic_payload=heuristic_payload) if query_enricher else None

    if enriched and enriched.used_llm:
        if _is_useful_llm_text(enriched.core_question):
            core_question = enriched.core_question.strip()
        if _is_useful_llm_text(enriched.retrieval_focus):
            retrieval_focus = enriched.retrieval_focus.strip()
        premise_claims = _merge_terms(premise_claims, enriched.premise_claims, limit=8)
        persons = _merge_terms(persons, enriched.persons, limit=8)
        locations = _merge_terms(locations, enriched.locations, limit=6)
        events = _merge_terms(events, enriched.events, limit=6)
        objects = _merge_terms(objects, enriched.objects, limit=6)
        aliases = _merge_terms(aliases, enriched.aliases, alias_values, limit=8)
        llm_keywords = _filter_keyword_candidates(
            enriched.keywords,
            original_query=query,
            known_entities=[*persons, *locations, *events, *objects, *aliases],
        )
        if llm_keywords:
            keywords = _merge_terms(
                llm_keywords,
                [*persons, *locations, *events, *objects, *aliases],
                limit=max_keywords,
            )
        else:
            keywords = _filter_keyword_candidates(
                _merge_terms(
                    [*alias_values, *persons, *locations, *events, *objects, *aliases],
                    keywords,
                    limit=max_keywords,
                ),
                original_query=query,
                known_entities=[*persons, *locations, *events, *objects, *aliases],
            )
        relation_intents = tuple(sorted(set(relation_intents) | set(enriched.relation_intents)))
        query_modes = tuple(_merge_terms(list(query_modes), enriched.query_modes, limit=8))

    if "chapter_locator" in set(query_modes):
        if locator_scene_hint:
            retrieval_focus = _merge_locator_focus(locator_scene_hint, persons, aliases)
            core_question = _merge_locator_focus(locator_scene_hint, persons, aliases)
        else:
            retrieval_focus = _clean_locator_text(retrieval_focus)
            core_question = _clean_locator_text(core_question)

    persons = _filter_persons(persons, known_names=known_names, llm_persons=[*(enriched.persons if enriched else []), *alias_values])
    if target_roles:
        query_modes = tuple(_merge_terms(list(query_modes), ["character_list"], limit=8))
    keywords = _filter_keyword_candidates(
        keywords,
        original_query=query,
        known_entities=[*alias_values, *persons, *locations, *events, *objects, *aliases],
        limit=max_keywords,
    )
    if "chapter_locator" in set(query_modes):
        keywords = _filter_locator_keywords(keywords, known_entities=[*alias_values, *persons, *locations, *events, *objects, *aliases])
    if enriched and enriched.used_llm:
        print(f"[INFO] Query preprocessing used LLM. keywords={keywords}")
    if not premise_claims:
        premise_claims = _fallback_premise_claims(query)

    rewritten_query = (
        enriched.rewritten_query.strip()
        if enriched and enriched.used_llm and enriched.rewritten_query.strip()
        else query
    )
    retrieval_focus = _augment_retrieval_focus(retrieval_focus or core_question, target_roles, target_volume)
    retrieval_seed = retrieval_focus or core_question or rewritten_query or query
    retrieval_query = build_retrieval_query(retrieval_seed)

    return QueryPlan(
        original_query=query,
        core_question=core_question,
        retrieval_focus=retrieval_focus or core_question,
        premise_claims=premise_claims,
        retrieval_query=retrieval_query,
        keywords=keywords,
        persons=persons,
        locations=locations,
        events=events,
        objects=objects,
        aliases=aliases,
        query_modes=query_modes,
        relation_intents=relation_intents,
        target_roles=target_roles,
        target_volume=target_volume,
        target_volume_index=target_volume_index,
        used_llm_enrichment=bool(enriched and enriched.used_llm),
    )


def extract_keywords(
    query: str,
    max_keywords: int = DEFAULT_MAX_KEYWORDS,
    reserved_terms: list[str] | None = None,
) -> list[str]:
    keywords: list[str] = []
    seen: set[str] = set()

    for reserved in reserved_terms or []:
        normalized_reserved = reserved.strip()
        if not normalized_reserved or normalized_reserved in seen:
            continue
        seen.add(normalized_reserved)
        keywords.append(normalized_reserved)
        if len(keywords) >= max_keywords:
            return keywords

    for raw_phrase in QUERY_TOKEN_PATTERN.findall(query):
        phrase = _strip_question_phrases(raw_phrase.strip())
        if not phrase:
            continue

        for candidate in _keyword_candidates(phrase):
            normalized_candidate = candidate.strip()
            if (
                len(normalized_candidate) < 2
                or normalized_candidate in STOP_KEYWORDS
                or normalized_candidate in QUERY_PERSON_STOPWORDS
                or normalized_candidate in seen
            ):
                continue
            seen.add(normalized_candidate)
            keywords.append(normalized_candidate)
            if len(keywords) >= max_keywords:
                return keywords

    return keywords


def load_relation_index() -> dict[str, dict[str, object]]:
    if not RELATIONS_PATH.exists():
        return {}
    mtime_ns = RELATIONS_PATH.stat().st_mtime_ns
    return _load_relation_index_cached(str(RELATIONS_PATH), mtime_ns)


def load_known_person_names() -> list[str]:
    if not CHUNKS_PATH.exists():
        return []
    mtime_ns = CHUNKS_PATH.stat().st_mtime_ns
    return _load_known_person_names_cached(str(CHUNKS_PATH), mtime_ns)


def _alias_expansions(query: str, corpus_names: list[str]) -> list[tuple[str, str]]:
    return alias_pairs_for_query(query, corpus_names)


def is_relation_query(query_plan: QueryPlan) -> bool:
    if "relation" in query_plan.query_modes:
        return True
    normalized_query = normalize_for_lexical(query_plan.original_query)
    if not normalized_query:
        return False
    return any(normalize_for_lexical(hint) in normalized_query for hint in RELATION_QUERY_HINTS)


def relation_intent_types(query_plan: QueryPlan) -> set[str]:
    if query_plan.relation_intents:
        return set(query_plan.relation_intents)
    return _relation_intent_types_from_text(query_plan.original_query, query_plan.persons)


def looks_like_query_person_name(text: str) -> bool:
    cleaned = strip_person_title(text.strip())
    if len(cleaned) < 2:
        return False
    if cleaned in PERSON_NAME_BLACKLIST or cleaned in QUERY_PERSON_STOPWORDS:
        return False
    if any(noise in cleaned for noise in PERSON_QUERY_NOISE):
        return False
    if any(cleaned.endswith(suffix) for suffix in LOCATION_SUFFIXES):
        return False
    if ENGLISH_NAME_PATTERN.fullmatch(cleaned):
        return True
    if re.fullmatch(r"[\u4e00-\u9fff路]{2,8}", cleaned) is None:
        return False
    if "路" in cleaned:
        parts = [part for part in cleaned.split("路") if part]
        return bool(parts) and all(1 <= len(part) <= 4 for part in parts)
    return len(cleaned) <= 4


def strip_person_title(text: str) -> str:
    cleaned = text.strip()
    for suffix in PERSON_TITLE_SUFFIXES:
        if cleaned.endswith(suffix) and len(cleaned) > len(suffix):
            return cleaned[: -len(suffix)]
    return cleaned


def strip_question_phrases(text: str) -> str:
    return _strip_question_phrases(text)


@lru_cache(maxsize=4)
def _load_relation_index_cached(relations_path: str, _: int) -> dict[str, dict[str, object]]:
    chunk_to_relations: dict[str, list[dict[str, object]]] = defaultdict(list)
    for raw_line in Path(relations_path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue

        relation = {
            "persons": [str(payload.get("person_a", "")), str(payload.get("person_b", ""))],
            "relation_type": str(payload.get("relation_type", "")),
            "confidence": float(payload.get("confidence", 0.0) or 0.0),
        }
        for chunk_id in payload.get("evidence_chunk_ids", []) or []:
            if chunk_id:
                chunk_to_relations[str(chunk_id)].append(relation)

    index: dict[str, dict[str, object]] = {}
    for chunk_id, relations in chunk_to_relations.items():
        persons = sorted({person for relation in relations for person in relation["persons"] if person})
        relation_types = sorted(
            {str(relation["relation_type"]) for relation in relations if relation.get("relation_type")}
        )
        max_confidence = max((float(relation.get("confidence", 0.0)) for relation in relations), default=0.0)
        index[chunk_id] = {
            "relations": relations,
            "persons": persons,
            "relation_types": relation_types,
            "max_confidence": max_confidence,
        }
    return index


@lru_cache(maxsize=4)
def _load_known_person_names_cached(chunks_path: str, _: int) -> list[str]:
    counter: dict[str, int] = defaultdict(int)
    for raw_line in Path(chunks_path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue

        metadata = payload.get("metadata", {}) or {}
        for person in metadata.get("persons", []) or []:
            if not isinstance(person, str):
                continue
            cleaned = person.strip()
            if len(cleaned) < 2:
                continue
            counter[cleaned] += 1

    names = [
        name
        for name, count in counter.items()
        if count >= MIN_KNOWN_PERSON_FREQUENCY and looks_like_query_person_name(name)
    ]
    names.sort(key=lambda name: (-len(name), -counter[name], name))
    return names


def _detect_query_modes(query: str, persons: list[str], relation_intents: tuple[str, ...]) -> list[str]:
    modes: list[str] = []
    normalized_query = normalize_for_lexical(query)
    open_ended_markers = (
        "为什么",
        "为什么要",
        "为了什么",
        "为啥",
        "原因",
        "目的",
        "目的是",
        "契机",
        "怎么看",
        "如何评价",
        "分析",
        "理解",
        "动机",
        "意义",
        "主题",
        "象征",
        "说明了什么",
        "体现了",
        "关系变化",
    )
    first_mention_markers = ("第一次", "首次", "初次", "第一次出现", "首次出现")
    chapter_markers = ("第几章", "哪一章", "哪个章节", "在哪个章节", "在哪一章", "章节")
    causal_markers = ("为什么", "原因", "导致", "怎么会", "为何")

    if any(marker in query for marker in open_ended_markers):
        modes.append("open_ended")
    if any(marker in query for marker in first_mention_markers):
        modes.append("first_appearance")
    if any(marker in query for marker in chapter_markers):
        modes.append("chapter_locator")
    if any(marker in query for marker in causal_markers):
        modes.append("causal")
    if relation_intents or any(normalize_for_lexical(hint) in normalized_query for hint in RELATION_QUERY_HINTS):
        modes.append("relation")
    elif len(persons) >= 2 and "关系" in query:
        modes.append("relation")
    if _is_character_list_query(query):
        modes.append("character_list")
    if not modes:
        modes.append("fact")
    return _dedupe_strings(modes)


def _relation_intent_types_from_text(query: str, persons: list[str]) -> set[str]:
    normalized_query = normalize_for_lexical(query)
    if not normalized_query:
        return set()

    intent_types: set[str] = set()
    for relation_type, hints in RELATION_INTENT_PATTERNS.items():
        if any(
            normalize_for_lexical(hint) in normalized_query
            and not any(person and normalize_for_lexical(person) == normalize_for_lexical(hint) for person in persons)
            for hint in hints
        ):
            intent_types.add(relation_type)
    return intent_types


def _keyword_candidates(text: str) -> list[str]:
    text = re.sub(r"\s+", "", text)
    if len(text) < 2:
        return []

    candidates: list[str] = []
    if 2 <= len(text) <= 12:
        candidates.append(text)

    for size in (5, 4, 3, 2):
        if len(text) < size:
            continue
        for start in range(0, len(text) - size + 1):
            gram = text[start : start + size]
            if gram not in candidates:
                candidates.append(gram)
    return candidates


def _extract_query_persons(query: str, max_items: int = 6) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()

    known_names = load_known_person_names()
    for name in known_names:
        if name in query and name not in seen:
            seen.add(name)
            candidates.append(name)
            if len(candidates) >= max_items:
                return candidates
            stripped = strip_person_title(name)
            if stripped != name and stripped in query and stripped not in seen:
                seen.add(stripped)
                candidates.append(stripped)
                if len(candidates) >= max_items:
                    return candidates

    for token in ENGLISH_NAME_PATTERN.findall(query):
        cleaned = token.strip()
        if len(cleaned) < 2 or cleaned in seen:
            continue
        seen.add(cleaned)
        candidates.append(cleaned)
        if len(candidates) >= max_items:
            return candidates

    split_parts = [part.strip() for part in QUERY_SPLIT_PATTERN.split(query) if part.strip()]
    for part in split_parts:
        normalized_part = _strip_question_phrases(part)
        if not normalized_part:
            continue
        for match in CHINESE_PERSON_PATTERN.findall(normalized_part):
            cleaned = strip_person_title(match)
            if cleaned in seen:
                continue
            if not looks_like_query_person_name(cleaned):
                continue
            seen.add(cleaned)
            candidates.append(cleaned)
            if len(candidates) >= max_items:
                return candidates

    return candidates


def _extract_query_locations(query: str, max_items: int = 4) -> list[str]:
    values = []
    seen: set[str] = set()
    for token in QUERY_TOKEN_PATTERN.findall(query):
        cleaned = _strip_question_phrases(token.strip())
        if not cleaned:
            continue
        if not any(cleaned.endswith(suffix) for suffix in LOCATION_SUFFIXES):
            continue
        if cleaned in seen:
            continue
        seen.add(cleaned)
        values.append(cleaned)
        if len(values) >= max_items:
            break
    return values


def _extract_query_hints(query: str, hints: tuple[str, ...], max_items: int) -> list[str]:
    values = []
    seen: set[str] = set()
    for hint in hints:
        if hint in query and hint not in seen:
            seen.add(hint)
            values.append(hint)
            if len(values) >= max_items:
                break
    return values


def _is_character_list_query(query: str) -> bool:
    normalized = query.replace(" ", "")
    if not normalized:
        return False
    if not any(marker in normalized for marker in CHARACTER_LIST_MARKERS):
        return False
    return any(alias in normalized for aliases in ROLE_HINTS.values() for alias in aliases)


def _extract_target_roles(query: str, max_items: int = 3) -> list[str]:
    normalized = query.replace(" ", "")
    matched: list[str] = []
    matched_aliases: list[str] = []
    for role, aliases in ROLE_HINTS.items():
        alias_hit = next((alias for alias in aliases if alias in normalized), None)
        if not alias_hit:
            continue
        if any(alias_hit in existing for existing in matched_aliases):
            continue
        if any(existing in alias_hit for existing in matched_aliases):
            continue
        if role in matched:
            continue
        if alias_hit:
            matched.append(role)
            matched_aliases.append(alias_hit)
            if len(matched) >= max_items:
                break
    return matched


def _expand_role_keywords(target_roles: list[str]) -> list[str]:
    expanded: list[str] = []
    seen: set[str] = set()
    for role in target_roles:
        for value in ROLE_EXPANSIONS.get(role, [role]):
            if value not in seen:
                seen.add(value)
                expanded.append(value)
    return expanded


def _augment_retrieval_focus(base_text: str, target_roles: list[str], target_volume: str | None) -> str:
    merged: list[str] = []
    seen: set[str] = set()
    for term in [base_text, target_volume, *_expand_role_keywords(target_roles)]:
        cleaned = str(term or "").strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        merged.append(cleaned)
    return " ".join(merged)


def _extract_locator_scene_hint(query: str) -> str:
    text = str(query or "").strip()
    if not text:
        return ""
    candidates = re.findall(r"[（(]([^（）()]{2,80})[）)]", text)
    for candidate in candidates:
        cleaned = _clean_locator_text(candidate)
        if cleaned and not _is_locator_request_only(cleaned):
            return cleaned
    cleaned = _clean_locator_text(text)
    return cleaned if cleaned != text else ""


def _merge_locator_focus(scene_hint: str, persons: list[str], aliases: list[str]) -> str:
    terms = [scene_hint]
    for value in [*persons, *aliases]:
        cleaned = _clean_locator_text(str(value or ""))
        if cleaned and cleaned not in scene_hint and not _is_locator_request_only(cleaned):
            terms.append(cleaned)
    return _dedupe_join_terms(terms)


def _clean_locator_text(text: str) -> str:
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""
    remove_phrases = (
        "动画最后的场景",
        "动画最后场景",
        "动画最后",
        "动画中",
        "动画里",
        "小说中",
        "小说里",
        "原作中",
        "原作里",
        "在小说中",
        "在小说里",
        "是第几卷第几章",
        "第几卷第几章",
        "第几卷",
        "第几章",
        "哪一卷哪一章",
        "哪一卷",
        "哪一章",
        "哪个章节",
        "章节",
        "所在的卷和章节",
        "所在卷和章节",
        "场景所在",
        "查找",
        "寻找",
    )
    for phrase in remove_phrases:
        cleaned = cleaned.replace(phrase, " ")
    cleaned = re.sub(r"[？?，,。；;：:\s]+", " ", cleaned)
    cleaned = cleaned.strip(" 的在是")
    return cleaned.strip()


def _filter_locator_keywords(keywords: list[str], *, known_entities: list[str]) -> list[str]:
    known = {str(value).strip() for value in known_entities if str(value).strip()}
    filtered: list[str] = []
    seen: set[str] = set()
    for keyword in keywords:
        cleaned = _clean_locator_text(str(keyword or ""))
        if not cleaned or cleaned in seen:
            continue
        if _is_locator_request_only(cleaned) and cleaned not in known:
            continue
        if cleaned not in known and ("坐摩" in cleaned and "摩天轮" not in cleaned):
            continue
        if cleaned not in known and cleaned.endswith(("坐摩", "见到温", "时候他们", "动画最后")):
            continue
        if any(cleaned != other and cleaned in other and len(cleaned) <= 5 for other in keywords):
            continue
        seen.add(cleaned)
        filtered.append(cleaned)
    return filtered


def _is_locator_request_only(text: str) -> bool:
    compact = re.sub(r"\s+", "", str(text or ""))
    if not compact:
        return True
    request_terms = (
        "动画最后",
        "小说中",
        "小说里",
        "第几卷",
        "第几章",
        "哪一章",
        "哪一卷",
        "章节",
        "场景",
        "所在",
        "查找",
        "寻找",
    )
    return any(term in compact for term in request_terms) and not any(
        entity in compact for entity in ("老八", "八奈见", "温水", "摩天轮")
    )


def _dedupe_join_terms(terms: list[str]) -> str:
    values: list[str] = []
    seen: set[str] = set()
    for term in terms:
        cleaned = str(term or "").strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        values.append(cleaned)
    return " ".join(values)


def _extract_target_volume(query: str) -> tuple[str | None, int | None]:
    match = VOLUME_PATTERN.search(query)
    if not match:
        return None, None
    raw_number = match.group(1)
    unit = match.group(2)
    volume_index = _parse_volume_number(raw_number)
    if volume_index is None:
        return None, None
    return f"第{raw_number}{unit}", volume_index


def _parse_volume_number(raw: str) -> int | None:
    cleaned = str(raw).strip()
    if not cleaned:
        return None
    if cleaned.isdigit():
        value = int(cleaned)
        return value if value > 0 else None

    if cleaned == "十":
        return 10

    total = 0
    current = 0
    saw_unit = False
    for char in cleaned:
        if char in CN_NUMERAL_MAP:
            current = CN_NUMERAL_MAP[char]
            continue
        if char == "十":
            saw_unit = True
            total += (current or 1) * 10
            current = 0
            continue
        if char == "百":
            saw_unit = True
            total += (current or 1) * 100
            current = 0
            continue
        return None
    value = total + current
    if value <= 0 and saw_unit:
        return None
    return value or None


def _strip_question_phrases(text: str) -> str:
    cleaned = text
    for phrase in QUESTION_PHRASES:
        cleaned = cleaned.replace(phrase, "")
    return cleaned.strip()


def _merge_terms(*sources: list[str], limit: int) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for source in sources:
        for raw in source:
            cleaned = str(raw).strip()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            values.append(cleaned)
            if len(values) >= limit:
                return values
    return values


def _filter_keyword_candidates(
    candidates: list[str],
    *,
    original_query: str,
    known_entities: list[str],
    limit: int | None = None,
) -> list[str]:
    query_compact = re.sub(r"\s+", "", original_query or "")
    known = {str(value).strip() for value in known_entities if str(value).strip()}
    filtered: list[str] = []
    seen: set[str] = set()
    bad_prefixes = ("的", "了", "是", "在", "和", "与", "把", "被", "这", "那")
    bad_suffixes = ("的", "了", "是", "在", "和", "与", "把", "被", "一", "颗", "个", "结")
    suspicious_prefixes = ("尾",)
    broken_question_fragments = (
        "是不",
        "是不是",
        "是否",
        "有没有",
        "有没",
        "没去",
        "没去过",
        "去过谁",
        "哪几",
        "哪些",
    )

    for raw in candidates:
        keyword = str(raw).strip()
        compact = re.sub(r"\s+", "", keyword)
        if not compact or compact in seen:
            continue
        if len(compact) < 2:
            continue
        if compact in STOP_KEYWORDS or compact in QUERY_PERSON_STOPWORDS:
            continue
        volume_match = re.fullmatch(r"第[一二三四五六七八九十0-9]+卷结?", compact)
        if volume_match:
            clean_volume = compact[:-1] if compact.endswith("结") else compact
            if clean_volume and clean_volume not in seen:
                seen.add(clean_volume)
                filtered.append(clean_volume)
                if limit is not None and len(filtered) >= limit:
                    break
            continue
        if compact not in known and compact not in query_compact:
            continue
        if compact not in known:
            if any(fragment in compact for fragment in broken_question_fragments):
                continue
            if compact.startswith(suspicious_prefixes):
                continue
            if compact.startswith(bad_prefixes) or compact.endswith(bad_suffixes):
                continue
            if len(compact) <= 3 and not any(compact == entity for entity in known):
                continue
        seen.add(compact)
        filtered.append(compact)
        if limit is not None and len(filtered) >= limit:
            break
    return filtered


def _filter_persons(persons: list[str], *, known_names: list[str], llm_persons: list[str]) -> list[str]:
    if not persons:
        return []

    known_normalized = {normalize_for_lexical(name) for name in known_names if name}
    llm_normalized = {normalize_for_lexical(name) for name in llm_persons if name}

    filtered: list[str] = []
    seen: set[str] = set()
    for person in persons:
        cleaned = str(person).strip()
        normalized = normalize_for_lexical(cleaned)
        if not cleaned or not normalized or normalized in seen:
            continue
        if llm_normalized:
            keep = normalized in llm_normalized or normalized in known_normalized
        else:
            keep = normalized in known_normalized
        if not keep:
            continue
        seen.add(normalized)
        filtered.append(cleaned)
    return filtered


def _is_useful_llm_text(text: str) -> bool:
    cleaned = str(text or "").strip()
    if not cleaned:
        return False
    normalized = normalize_for_lexical(cleaned)
    if not normalized:
        return False
    if normalized in {"unknown", "none", "null", "na"}:
        return False
    return True


def _fallback_premise_claims(query: str) -> list[str]:
    text = query.strip()
    if not text:
        return []
    for marker in ("请问", "那么", "那", "是否", "是不是", "能否"):
        if marker in text:
            prefix = text.split(marker, 1)[0].strip("，。；？！ ")
            if len(prefix) >= 8:
                return [prefix]
    for punctuation in ("？", "?", "吗", "么"):
        if punctuation in text:
            prefix = text.split(punctuation, 1)[0].strip("，。；？！ ")
            if len(prefix) >= 8:
                return [prefix]
    return []


def _dedupe_strings(values: list[str]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        cleaned.append(value)
    return cleaned
