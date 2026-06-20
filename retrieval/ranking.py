from __future__ import annotations

import re

from langchain_chroma import Chroma
from langchain_core.documents import Document

from core import get_logger

from .config import (
    DEFAULT_DENSE_WEIGHT,
    DEFAULT_EXACT_KEYWORD_FIRST,
    DEFAULT_FETCH_K,
    DEFAULT_KEYWORD_FETCH_K,
    DEFAULT_LEXICAL_WEIGHT,
    DEFAULT_MAX_DISTANCE,
    DEFAULT_MAX_KEYWORDS,
    DEFAULT_METADATA_WEIGHT,
    DEFAULT_MIN_HYBRID_SCORE,
    DEFAULT_POSITION_WEIGHT,
    DEFAULT_RELATION_WEIGHT,
    DEFAULT_SUMMARY_WEIGHT,
    DEFAULT_TOP_K,
)
from .context_expansion import decide_chunk_expansions
from .models import QueryPlan, RankedChunk, RankingWeights, RetrievalResult
from .metadata import metadata_list, restore_runtime_metadata
from .query import (
    analyze_query,
    extract_keywords,
    is_relation_query,
    load_relation_index,
    relation_intent_types,
)
from .utils import char_grams, document_key, normalize_for_lexical, too_similar
from .vectorstore import (
    embed_query_once,
    similarity_search_by_embedding,
    similarity_search_with_score_by_embedding,
)

logger = get_logger(__name__)


def retrieve_context(
    db: Chroma,
    query: str,
    corpus_names: list[str] | None = None,
    search_scope: dict | None = None,
    top_k: int = DEFAULT_TOP_K,
    fetch_k: int = DEFAULT_FETCH_K,
    keyword_fetch_k: int = DEFAULT_KEYWORD_FETCH_K,
    max_keywords: int = DEFAULT_MAX_KEYWORDS,
    dense_weight: float = DEFAULT_DENSE_WEIGHT,
    lexical_weight: float = DEFAULT_LEXICAL_WEIGHT,
    metadata_weight: float = DEFAULT_METADATA_WEIGHT,
    summary_weight: float = DEFAULT_SUMMARY_WEIGHT,
    relation_weight: float = DEFAULT_RELATION_WEIGHT,
    position_weight: float = DEFAULT_POSITION_WEIGHT,
    min_hybrid_score: float = DEFAULT_MIN_HYBRID_SCORE,
    max_distance: float | None = DEFAULT_MAX_DISTANCE,
    exact_keyword_first: bool = DEFAULT_EXACT_KEYWORD_FIRST,
    expand_neighbors: bool = True,
    model: str | None = None,
) -> RetrievalResult:
    scope_corpora = _scope_corpus_names(search_scope) or (corpus_names or [])
    query_plan = analyze_query(query, max_keywords=max_keywords, model=model, corpus_names=scope_corpora)
    collection_count = _collection_count(db)
    if collection_count <= 0:
        return RetrievalResult(
            query=query,
            retrieval_query=query_plan.retrieval_query,
            keywords=query_plan.keywords,
            query_plan=query_plan,
            chunks=[],
        )

    safe_fetch_k = min(max(fetch_k, top_k), collection_count)
    safe_keyword_fetch_k = min(keyword_fetch_k, collection_count)
    candidates: dict[str, tuple[Document, float | None]] = {}
    metadata_filter = _search_scope_filter(search_scope) or _corpus_filter(corpus_names or [])
    has_manual_volume_filter = _has_manual_volume_filter(search_scope)
    query_embedding = embed_query_once(db, query_plan.retrieval_query)

    dense_results = similarity_search_with_score_by_embedding(
        db,
        query_embedding,
        k=safe_fetch_k,
        filter=metadata_filter or None,
    )
    _merge_candidates(candidates, dense_results)

    volume_filter = None if has_manual_volume_filter else _target_volume_filter(query_plan, metadata_filter)
    if volume_filter:
        try:
            volume_results = similarity_search_with_score_by_embedding(
                db,
                query_embedding,
                k=min(max(top_k * 4, 12), collection_count),
                filter=volume_filter,
            )
            _merge_candidates(candidates, volume_results)
        except Exception:
            logger.warning("分卷过滤检索失败,跳过该路召回", exc_info=True)

    for keyword in query_plan.keywords:
        try:
            if metadata_filter:
                keyword_results = _filtered_keyword_search(
                    db,
                    query_embedding,
                    keyword,
                    safe_keyword_fetch_k,
                    metadata_filter,
                )
            else:
                keyword_results = similarity_search_with_score_by_embedding(
                    db,
                    query_embedding,
                    k=safe_keyword_fetch_k,
                    where_document={"$contains": keyword},
                )
        except Exception:
            logger.warning("关键词 %r 检索失败,跳过", keyword, exc_info=True)
            continue
        _merge_candidates(candidates, keyword_results)

    ranked = _rank_candidates(
        candidates.values(),
        query_plan=query_plan,
        weights=_select_ranking_weights(
            query_plan,
            dense_weight,
            lexical_weight,
            metadata_weight,
            summary_weight,
            relation_weight,
            position_weight,
        ),
        max_distance=max_distance,
    )
    ranked = [item for item in ranked if item.score >= min_hybrid_score]
    if exact_keyword_first:
        ranked = _prioritize_keyword_hits(ranked, query_plan)
    ranked = _select_diverse(ranked, top_k=top_k)
    if expand_neighbors:
        ranked = _expand_with_adjacent_chunks(db, ranked, query=query, top_k=top_k, model=model)

    return RetrievalResult(
        query=query,
        retrieval_query=query_plan.retrieval_query,
        keywords=query_plan.keywords,
        query_plan=query_plan,
        chunks=ranked,
    )


def _rank_candidates(
    docs_and_scores,
    *,
    query_plan: QueryPlan,
    weights: RankingWeights,
    max_distance: float | None,
) -> list[RankedChunk]:
    docs_and_scores = list(docs_and_scores)
    if max_distance is not None:
        docs_and_scores = [
            (doc, distance)
            for doc, distance in docs_and_scores
            if distance is None or distance <= max_distance
        ]

    distances = [distance for _, distance in docs_and_scores if distance is not None]
    min_distance = min(distances) if distances else 0.0
    max_seen_distance = max(distances) if distances else min_distance
    distance_span = max(max_seen_distance - min_distance, 1e-9)

    ranked: list[RankedChunk] = []
    relation_index = load_relation_index()
    for doc, distance in docs_and_scores:
        if distance is None:
            dense_score = 0.0
        else:
            dense_score = 1.0 - ((distance - min_distance) / distance_span)
            dense_score = max(0.0, min(1.0, dense_score))

        lexical_score = _lexical_score(query_plan.original_query, doc.page_content)
        metadata = restore_runtime_metadata(doc.metadata or {})
        metadata_score = _metadata_score(query_plan, metadata)
        summary_score = _summary_score(query_plan, metadata)
        relation_score = _relation_score(query_plan, metadata, relation_index, doc.page_content)
        position_score = _position_score(query_plan, metadata)
        volume_score = _volume_score(query_plan, metadata)
        placeholder_penalty = _placeholder_penalty(query_plan, metadata, doc.page_content)
        character_list_bonus = _character_list_bonus(query_plan, metadata, doc.page_content)
        score = (
            weights.dense * dense_score
            + weights.lexical * lexical_score
            + weights.metadata * metadata_score
            + weights.summary * summary_score
            + weights.relation * relation_score
            + weights.position * position_score
        )
        score += 0.18 * volume_score + character_list_bonus - placeholder_penalty
        score = max(0.0, min(1.0, score))

        metadata["query_keywords"] = query_plan.keywords
        metadata["relation_score"] = relation_score
        metadata["volume_score"] = round(volume_score, 4)
        metadata["placeholder_penalty"] = round(placeholder_penalty, 4)
        metadata["character_list_bonus"] = round(character_list_bonus, 4)
        relation_payload = relation_index.get(str(metadata.get("chunk_id", "")), {})
        if relation_payload:
            metadata["relation_persons"] = relation_payload.get("persons", [])
            metadata["relation_types"] = relation_payload.get("relation_types", [])
        ranked.append(
            RankedChunk(
                document=doc,
                distance=distance,
                dense_score=dense_score,
                lexical_score=lexical_score,
                metadata_score=metadata_score,
                summary_score=summary_score,
                relation_score=relation_score,
                position_score=position_score,
                score=score,
            )
        )

    ranked.sort(key=lambda item: item.score, reverse=True)
    return ranked


def _select_ranking_weights(
    query_plan: QueryPlan,
    dense_weight: float,
    lexical_weight: float,
    metadata_weight: float,
    summary_weight: float,
    relation_weight: float,
    position_weight: float,
) -> RankingWeights:
    weights = RankingWeights(
        dense=dense_weight,
        lexical=lexical_weight,
        metadata=metadata_weight,
        summary=summary_weight,
        relation=relation_weight,
        position=position_weight,
    )
    modes = set(query_plan.query_modes)
    if "relation" in modes:
        weights = RankingWeights(
            dense=weights.dense * 0.9,
            lexical=weights.lexical * 1.0,
            metadata=weights.metadata * 1.15,
            summary=weights.summary * 0.7,
            relation=max(weights.relation, 0.22),
            position=max(weights.position, 0.02),
        )
    elif "first_appearance" in modes or "chapter_locator" in modes:
        weights = RankingWeights(
            dense=weights.dense * 0.85,
            lexical=weights.lexical * 1.0,
            metadata=weights.metadata * 1.15,
            summary=weights.summary * 0.85,
            relation=weights.relation * 0.4,
            position=max(weights.position, 0.20),
        )
    elif "causal" in modes:
        weights = RankingWeights(
            dense=weights.dense * 0.95,
            lexical=weights.lexical * 1.0,
            metadata=weights.metadata * 1.0,
            summary=weights.summary * 1.25,
            relation=weights.relation * 1.1,
            position=weights.position,
        )
    elif len(query_plan.persons) >= 2:
        weights = RankingWeights(
            dense=weights.dense * 0.92,
            lexical=weights.lexical * 1.0,
            metadata=weights.metadata * 1.12,
            summary=weights.summary * 0.95,
            relation=max(weights.relation, 0.16),
            position=max(weights.position, 0.02),
        )
    total = weights.dense + weights.lexical + weights.metadata + weights.summary + weights.relation + weights.position
    if total <= 0:
        return RankingWeights(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return RankingWeights(
        dense=weights.dense / total,
        lexical=weights.lexical / total,
        metadata=weights.metadata / total,
        summary=weights.summary / total,
        relation=weights.relation / total,
        position=weights.position / total,
    )


def _metadata_score(query_plan: QueryPlan, metadata: dict) -> float:
    scores = [
        _overlap_ratio(query_plan.persons, metadata_list(metadata, "persons"), metadata_list(metadata, "chapter_persons")),
        _overlap_ratio(query_plan.locations, metadata_list(metadata, "locations"), metadata_list(metadata, "chapter_locations")),
        _overlap_ratio(query_plan.events, metadata_list(metadata, "events"), metadata_list(metadata, "chapter_events")),
        _overlap_ratio(query_plan.objects, metadata_list(metadata, "objects"), metadata_list(metadata, "chapter_objects")),
        _overlap_ratio(query_plan.keywords, metadata_list(metadata, "keywords"), metadata_list(metadata, "chapter_keywords")),
    ]
    if len(query_plan.persons) >= 2:
        scores.append(
            _overlap_ratio(
                query_plan.persons,
                metadata_list(metadata, "relation_persons"),
                metadata_list(metadata, "chapter_persons"),
            )
        )
    if query_plan.aliases:
        scores.append(
            _overlap_ratio(
                query_plan.aliases,
                metadata_list(metadata, "aliases"),
                metadata_list(metadata, "chapter_aliases"),
            )
        )
    if query_plan.target_roles:
        role_overlap = _role_metadata_score(query_plan, metadata)
        if role_overlap > 0:
            scores.append(role_overlap)
    usable_scores = [score for score in scores if score > 0]
    if not usable_scores:
        return 0.0
    base = sum(usable_scores) / len(usable_scores)
    if "relation" in set(query_plan.query_modes):
        relation_entity_bonus = _overlap_ratio(
            [*query_plan.persons, *query_plan.aliases],
            metadata_list(metadata, "relation_persons"),
            metadata_list(metadata, "chapter_persons"),
        )
        return max(base, min(1.0, 0.75 * base + 0.25 * relation_entity_bonus))
    return base


def _summary_score(query_plan: QueryPlan, metadata: dict) -> float:
    summary = str(metadata.get("chapter_summary", ""))
    if not summary:
        return 0.0
    return _lexical_score(query_plan.original_query, summary)


def _position_score(query_plan: QueryPlan, metadata: dict) -> float:
    modes = set(query_plan.query_modes)
    if "first_appearance" not in modes and "chapter_locator" not in modes:
        return 0.0
    chapter_index = metadata.get("chapter_index")
    chunk_index = metadata.get("chunk_index")
    try:
        chapter_value = max(0, int(chapter_index))
    except Exception:
        chapter_value = 999
    try:
        chunk_value = max(0, int(chunk_index))
    except Exception:
        chunk_value = 999
    return 1.0 / (1.0 + chapter_value + chunk_value / 10.0)


def _relation_score(
    query_plan: QueryPlan,
    metadata: dict,
    relation_index: dict[str, dict[str, object]],
    text: str = "",
) -> float:
    if not query_plan.persons:
        return 0.0

    chunk_id = metadata.get("chunk_id")
    query_persons = {normalize_for_lexical(person) for person in query_plan.persons if person}
    if not query_persons:
        return 0.0

    relation_payload = relation_index.get(str(chunk_id)) if chunk_id else None
    relation_persons = {
        normalize_for_lexical(person)
        for person in (relation_payload or {}).get("persons", [])
        if person
    }
    matched_persons = query_persons & relation_persons
    matched_ratio = len(matched_persons) / len(query_persons)
    max_confidence = float((relation_payload or {}).get("max_confidence", 0.0) or 0.0)
    relation_types = {str(value) for value in (relation_payload or {}).get("relation_types", []) if value}
    type_hint_score = _relation_type_hint_score(query_plan, relation_types)
    relation_query = is_relation_query(query_plan)
    intent_types = relation_intent_types(query_plan)
    type_alignment = _relation_type_alignment(relation_types, intent_types)

    chapter_persons = {
        normalize_for_lexical(person)
        for person in metadata_list(metadata, "chapter_persons")
        if person
    }
    chapter_overlap = len(query_persons & chapter_persons) / len(query_persons) if chapter_persons else 0.0

    if len(query_persons) >= 2:
        if not relation_query:
            return min(0.22, 0.18 * chapter_overlap)
        if len(matched_persons) >= 2:
            base = 0.58 + 0.18 * type_hint_score + 0.14 * type_alignment
        else:
            base = _implicit_relation_score(query_plan, metadata, text)
    else:
        if not relation_query:
            return min(0.16, 0.14 * chapter_overlap)
        if relation_persons:
            base = 0.18 * matched_ratio
            if type_hint_score > 0:
                base += 0.18 * type_hint_score
            if type_alignment > 0:
                base += 0.16 * type_alignment
        else:
            alias_overlap = _overlap_ratio(query_plan.aliases, metadata_list(metadata, "aliases"), metadata_list(metadata, "chapter_aliases"))
            base = 0.10 * chapter_overlap + 0.06 * alias_overlap

    score = 0.7 * base + 0.3 * max_confidence
    return max(0.0, min(1.0, score))


def _relation_type_hint_score(query_plan: QueryPlan, relation_types: set[str]) -> float:
    if not relation_types:
        return 0.0

    normalized_query = normalize_for_lexical(query_plan.original_query)
    if not normalized_query:
        return 0.0

    from .config import RELATION_TYPE_HINTS

    matched = 0
    for relation_type in relation_types:
        for hint in RELATION_TYPE_HINTS.get(relation_type, set()):
            if normalize_for_lexical(hint) in normalized_query:
                matched += 1
                break

    if not matched:
        return 0.0
    return matched / len(relation_types)


def _relation_type_alignment(relation_types: set[str], intent_types: set[str]) -> float:
    if not relation_types or not intent_types:
        return 0.0
    overlap = relation_types & intent_types
    if not overlap:
        return 0.0
    return len(overlap) / len(intent_types)


def _implicit_relation_score(query_plan: QueryPlan, metadata: dict, text: str) -> float:
    if len(query_plan.persons) < 2 or not is_relation_query(query_plan):
        return 0.0

    entity_haystack = {
        normalize_for_lexical(value)
        for value in [
            *metadata_list(metadata, "persons"),
            *metadata_list(metadata, "aliases"),
        ]
        if value
    }
    query_persons = [normalize_for_lexical(person) for person in query_plan.persons if person]
    if not query_persons or not all(person in entity_haystack for person in query_persons):
        return 0.0

    normalized_text = normalize_for_lexical(text)
    normalized_query = normalize_for_lexical(query_plan.original_query)
    if not normalized_text or not all(person in normalized_text for person in query_persons):
        return 0.0
    if not _persons_are_close_in_text(normalized_text, query_persons, max_gap=120):
        return 0.0

    from .config import RELATION_INTENT_PATTERNS, RELATION_QUERY_HINTS

    intent_types = relation_intent_types(query_plan)
    score = 0.22
    if any(normalize_for_lexical(hint) in normalized_text for hint in RELATION_QUERY_HINTS):
        score += 0.12

    matched_intents = 0
    for relation_type in intent_types:
        if any(normalize_for_lexical(hint) in normalized_text for hint in RELATION_INTENT_PATTERNS.get(relation_type, ())):
            matched_intents += 1
    if intent_types:
        score += 0.14 * (matched_intents / len(intent_types))

    if normalized_query and normalized_text:
        overlap = len(char_grams(normalized_query, n=2) & char_grams(normalized_text[:1200], n=2))
        if overlap > 0:
            score += 0.05

    return max(0.0, min(0.45, score))


def _persons_are_close_in_text(text: str, persons: list[str], max_gap: int) -> bool:
    if len(persons) < 2:
        return False

    positions: list[tuple[int, int]] = []
    for person in persons:
        idx = text.find(person)
        if idx < 0:
            return False
        positions.append((idx, idx + len(person)))

    positions.sort()
    left_end = positions[0][1]
    right_start = positions[-1][0]
    return max(0, right_start - left_end) <= max_gap


def _overlap_ratio(left: list[str], primary: list[str], secondary: list[str]) -> float:
    if not left:
        return 0.0

    haystack = {normalize_for_lexical(value) for value in [*primary, *secondary] if value}
    if not haystack:
        return 0.0

    matches = 0
    for value in left:
        normalized = normalize_for_lexical(value)
        if normalized and normalized in haystack:
            matches += 1
    return matches / len(left)


def _prioritize_keyword_hits(ranked: list[RankedChunk], query_plan: QueryPlan) -> list[RankedChunk]:
    if not query_plan.keywords:
        return ranked

    hits = []
    misses = []
    for item in ranked:
        metadata = restore_runtime_metadata(item.document.metadata or {})
        metadata["hit_keywords"] = _hit_keywords(item.document.page_content, query_plan.keywords)
        metadata["hit_entities"] = _hit_entities(query_plan, metadata)
        if metadata["hit_keywords"] or metadata["hit_entities"]:
            hits.append(item)
        else:
            misses.append(item)

    if not hits:
        return ranked

    hits.sort(
        key=lambda item: (
            len(item.document.metadata.get("hit_entities", [])),
            len(item.document.metadata.get("hit_keywords", [])),
            item.score,
        ),
        reverse=True,
    )
    return hits + misses


def _hit_entities(query_plan: QueryPlan, metadata: dict) -> list[str]:
    haystack = set()
    for key in ("persons", "locations", "events", "objects", "aliases"):
        haystack.update(metadata_list(metadata, key))
        haystack.update(metadata_list(metadata, f"chapter_{key}"))

    entities = [*query_plan.persons, *query_plan.locations, *query_plan.events, *query_plan.objects, *query_plan.aliases]
    return [entity for entity in entities if entity in haystack]


def _select_diverse(ranked: list[RankedChunk], top_k: int) -> list[RankedChunk]:
    selected: list[RankedChunk] = []
    deferred: list[RankedChunk] = []

    for item in ranked:
        if len(selected) >= top_k:
            break
        if any(too_similar(item.document.page_content, chosen.document.page_content) for chosen in selected):
            deferred.append(item)
            continue
        selected.append(item)

    if len(selected) < top_k:
        for item in deferred:
            if len(selected) >= top_k:
                break
            selected.append(item)
    return selected


def _lexical_score(query: str, text: str) -> float:
    keywords = extract_keywords(query, max_keywords=12)
    keyword_score = 0.0
    if keywords:
        keyword_score = sum(1 for keyword in keywords if keyword in text) / len(keywords)

    q_grams = char_grams(normalize_for_lexical(query), n=2)
    t_grams = char_grams(normalize_for_lexical(text[:2500]), n=2)
    overlap_score = 0.0
    if q_grams and t_grams:
        overlap_score = len(q_grams & t_grams) / len(q_grams)

    return max(0.0, min(1.0, 0.65 * keyword_score + 0.35 * overlap_score))


def _volume_score(query_plan: QueryPlan, metadata: dict) -> float:
    target_index = query_plan.target_volume_index
    if target_index is None:
        return 0.0
    try:
        candidate_index = int(metadata.get("volume_index"))
    except Exception:
        return 0.0

    if candidate_index == target_index:
        return 1.0
    if candidate_index < target_index:
        distance = target_index - candidate_index
        if distance == 1:
            return 0.58
        if distance == 2:
            return 0.34
        if distance == 3:
            return 0.18
        return 0.04
    distance = candidate_index - target_index
    if distance == 1:
        return 0.10
    return 0.0


def _placeholder_penalty(query_plan: QueryPlan, metadata: dict, text: str) -> float:
    normalized_query = normalize_for_lexical(query_plan.original_query)
    if any(term in normalized_query for term in ("插图", "彩页", "封面", "扉页", "设定图")):
        return 0.0

    chapter = str(metadata.get("chapter", ""))
    preview = re.sub(r"\s+", "", text[:160])
    chapter_compact = re.sub(r"\s+", "", chapter)
    placeholder_like = bool(metadata.get("is_placeholder_chunk"))
    placeholder_like = placeholder_like or any(term in chapter_compact for term in ("插图", "彩页", "封面", "扉页"))
    placeholder_like = placeholder_like or any(term in preview for term in ("卷首插图", "卷末插图", "人物设定图", "彩页"))
    if not placeholder_like:
        return 0.0

    entity_count = sum(
        len(metadata_list(metadata, key))
        for key in ("persons", "locations", "events", "objects", "aliases")
    )
    if entity_count > 0:
        return 0.10
    return 0.28


def _character_list_bonus(query_plan: QueryPlan, metadata: dict, text: str) -> float:
    if "character_list" not in set(query_plan.query_modes):
        return 0.0
    if metadata.get("is_synthetic_role_index"):
        role_score = _role_metadata_score(query_plan, metadata)
        return min(0.30, 0.18 + 0.10 * role_score)
    persons = metadata_list(metadata, "persons")
    chapter_persons = metadata_list(metadata, "chapter_persons")
    combined = {normalize_for_lexical(value) for value in [*persons, *chapter_persons] if value}
    if not combined:
        return -0.08

    role_score = _role_metadata_score(query_plan, metadata)
    text_hits = 0
    for role in query_plan.target_roles:
        for cue in _role_cues(role):
            if cue in text:
                text_hits += 1
                break
    base = min(0.18, 0.03 * len(combined))
    return min(0.24, base + 0.08 * role_score + 0.03 * text_hits)


def _role_metadata_score(query_plan: QueryPlan, metadata: dict) -> float:
    if not query_plan.target_roles:
        return 0.0
    text_fields = [
        str(metadata.get("chapter", "")),
        " ".join(metadata_list(metadata, "keywords")),
        " ".join(metadata_list(metadata, "chapter_keywords")),
        " ".join(metadata_list(metadata, "female_characters")),
        " ".join(metadata_list(metadata, "male_characters")),
    ]
    joined = " ".join(text_fields)
    matches = 0
    for role in query_plan.target_roles:
        if any(cue in joined for cue in _role_cues(role)):
            matches += 1
    return matches / len(query_plan.target_roles) if query_plan.target_roles else 0.0


def _role_cues(role: str) -> tuple[str, ...]:
    mapping = {
        "女主角": ("女主角", "女主", "女主人公"),
        "男主角": ("男主角", "男主", "男主人公"),
        "主角": ("主角", "主人公"),
        "主要角色": ("主要角色", "主要人物", "核心角色"),
        "角色": ("角色", "人物", "登场人物"),
    }
    return mapping.get(role, (role,))


def _merge_candidates(
    candidates: dict[str, tuple[Document, float | None]],
    docs_and_scores: list[tuple[Document, float]],
) -> None:
    for doc, distance in docs_and_scores:
        key = document_key(doc)
        current = candidates.get(key)
        if current is None or (distance is not None and (current[1] is None or distance < current[1])):
            candidates[key] = (doc, distance)


def _expand_with_adjacent_chunks(
    db: Chroma,
    ranked: list[RankedChunk],
    *,
    query: str,
    top_k: int,
    model: str | None = None,
) -> list[RankedChunk]:
    if not ranked:
        return ranked

    anchor_count = min(3, len(ranked))
    anchor_items = ranked[:anchor_count]
    decisions = decide_chunk_expansions(
        question=query,
        candidates=[
            {
                "chunk_id": str(restore_runtime_metadata(item.document.metadata or {}).get("chunk_id", "")).strip(),
                "chapter": str(restore_runtime_metadata(item.document.metadata or {}).get("chapter", "")).strip(),
                "text": item.document.page_content,
            }
            for item in anchor_items
        ],
        max_candidates=anchor_count,
        model=model,
    )
    print(
        f"[INFO] Context expansion decisions: anchors={anchor_count}, decisions={len(decisions)}."
    )
    expansion_requests: dict[str, dict[str, str]] = {}
    neighbor_ids: list[str] = []

    for item in anchor_items:
        metadata = restore_runtime_metadata(item.document.metadata or {})
        chunk_id = str(metadata.get("chunk_id", "")).strip()
        decision = decisions.get(chunk_id)
        if decision is None:
            # fallback: if no DS decision, keep the old conservative expansion
            prev_flag = True
            next_flag = True
            expansion_reason = ""
        else:
            if not decision.is_high_value:
                continue
            prev_flag = decision.need_prev_chunk
            next_flag = decision.need_next_chunk
            expansion_reason = decision.reason

        if prev_flag:
            neighbor = str(metadata.get("prev_chunk_id") or "").strip()
            if neighbor:
                expansion_requests.setdefault(chunk_id, {})["prev"] = neighbor
                neighbor_ids.append(neighbor)
        if next_flag:
            neighbor = str(metadata.get("next_chunk_id") or "").strip()
            if neighbor:
                expansion_requests.setdefault(chunk_id, {})["next"] = neighbor
                neighbor_ids.append(neighbor)
        if expansion_reason:
            expansion_requests.setdefault(chunk_id, {})["reason"] = expansion_reason

    if not neighbor_ids:
        print("[INFO] Context expansion selected 0 neighbor chunks.")
        return ranked

    fetched = _fetch_chunks_by_ids(db, sorted(set(neighbor_ids)))
    if not fetched:
        print(f"[INFO] Context expansion selected {len(neighbor_ids)} neighbors but fetched 0 chunks.")
        return ranked
    print(f"[INFO] Context expansion fetched {len(fetched)} neighbor chunks.")

    fetched_by_id = {
        str(restore_runtime_metadata(doc.metadata or {}).get("chunk_id", "")).strip(): doc
        for doc in fetched
    }
    expanded_count = 0
    for item in ranked:
        metadata = restore_runtime_metadata(item.document.metadata or {})
        chunk_id = str(metadata.get("chunk_id", "")).strip()
        request = expansion_requests.get(chunk_id)
        if not request:
            continue
        prev_doc = fetched_by_id.get(request.get("prev", ""))
        next_doc = fetched_by_id.get(request.get("next", ""))
        merged_text = merge_adjacent_texts(
            prev_doc.page_content if prev_doc else "",
            item.document.page_content,
            next_doc.page_content if next_doc else "",
        )
        if not merged_text or merged_text.strip() == item.document.page_content.strip():
            continue
        metadata["expanded_context_text"] = merged_text
        metadata["expanded_prev_chunk_id"] = request.get("prev", "")
        metadata["expanded_next_chunk_id"] = request.get("next", "")
        metadata["expansion_reason"] = request.get("reason", "")
        metadata["has_expanded_context"] = True
        item.document.metadata = metadata
        expanded_count += 1

    print(f"[INFO] Context expansion merged adjacent text into {expanded_count} primary chunks.")
    return ranked


def merge_adjacent_texts(prev_text: str = "", current_text: str = "", next_text: str = "") -> str:
    pieces = [str(prev_text or "").strip(), str(current_text or "").strip(), str(next_text or "").strip()]
    merged = ""
    for piece in pieces:
        if not piece:
            continue
        merged = _append_without_overlap(merged, piece) if merged else piece
    return merged


def _append_without_overlap(left: str, right: str, *, min_overlap: int = 20, max_scan: int = 300) -> str:
    left = str(left or "").strip()
    right = str(right or "").strip()
    if not left:
        return right
    if not right:
        return left
    max_overlap = min(len(left), len(right), max_scan)
    best = 0
    for size in range(max_overlap, min_overlap - 1, -1):
        if left[-size:] == right[:size]:
            best = size
            break
    if best:
        return left + right[best:]
    return left + "\n" + right


def _fetch_chunks_by_ids(db: Chroma, chunk_ids: list[str]) -> list[Document]:
    cleaned = [chunk_id for chunk_id in chunk_ids if str(chunk_id).strip()]
    if not cleaned:
        return []
    try:
        payload = db._collection.get(ids=cleaned, include=["documents", "metadatas"])
    except Exception:
        return []

    ids = payload.get("ids", []) if isinstance(payload, dict) else []
    documents = payload.get("documents", []) if isinstance(payload, dict) else []
    metadatas = payload.get("metadatas", []) if isinstance(payload, dict) else []
    results: list[Document] = []
    for chunk_id, text, metadata in zip(ids, documents, metadatas):
        if not text:
            continue
        doc = Document(page_content=text, metadata=metadata or {}, id=chunk_id)
        results.append(doc)
    return results


def _hit_keywords(text: str, keywords: list[str]) -> list[str]:
    return [keyword for keyword in keywords if keyword and keyword in text]


def _collection_count(db: Chroma) -> int:
    try:
        return int(db._collection.count())
    except Exception:
        return 0


def _corpus_filter(corpus_names: list[str]) -> dict | None:
    cleaned = [name.strip() for name in corpus_names if str(name).strip()]
    if not cleaned:
        return None
    if len(cleaned) == 1:
        return {"corpus_name": cleaned[0]}
    return {"$or": [{"corpus_name": name} for name in cleaned]}


def _search_scope_filter(search_scope: dict | None) -> dict | None:
    if not isinstance(search_scope, dict):
        return None
    corpora = [str(name).strip() for name in search_scope.get("corpora", []) or [] if str(name).strip()]
    volumes_by_corpus = search_scope.get("volumes", {}) or {}
    if not corpora:
        return None

    clauses = []
    for corpus_name in corpora:
        corpus_clause = {"corpus_name": corpus_name}
        raw_volumes = volumes_by_corpus.get(corpus_name, []) if isinstance(volumes_by_corpus, dict) else []
        volume_indices = []
        for value in raw_volumes or []:
            try:
                volume_indices.append(int(value))
            except Exception:
                continue
        volume_indices = sorted(set(volume_indices))
        if volume_indices:
            volume_clause = _or_filter([{"volume_index": index} for index in volume_indices])
            clauses.append(_and_filter([corpus_clause, volume_clause]))
        else:
            clauses.append(corpus_clause)

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$or": clauses}


def _scope_corpus_names(search_scope: dict | None) -> list[str]:
    if not isinstance(search_scope, dict):
        return []
    return [str(name).strip() for name in search_scope.get("corpora", []) or [] if str(name).strip()]


def _has_manual_volume_filter(search_scope: dict | None) -> bool:
    if not isinstance(search_scope, dict):
        return False
    volumes = search_scope.get("volumes", {})
    if not isinstance(volumes, dict):
        return False
    for values in volumes.values():
        if values:
            return True
    return False


def _target_volume_filter(query_plan: QueryPlan, base_filter: dict | None) -> dict | None:
    target_index = query_plan.target_volume_index
    if target_index is None:
        return None

    allowed_indices = [target_index]
    for offset in (1, 2):
        if target_index - offset > 0:
            allowed_indices.append(target_index - offset)

    volume_clause = _or_filter([{"volume_index": index} for index in allowed_indices])
    if not base_filter:
        return volume_clause
    return _and_filter([base_filter, volume_clause])


def _or_filter(clauses: list[dict]) -> dict:
    cleaned = [clause for clause in clauses if clause]
    if not cleaned:
        return {}
    if len(cleaned) == 1:
        return cleaned[0]
    return {"$or": cleaned}


def _and_filter(clauses: list[dict]) -> dict:
    cleaned = [clause for clause in clauses if clause]
    if not cleaned:
        return {}
    if len(cleaned) == 1:
        return cleaned[0]
    return {"$and": cleaned}


def _filtered_keyword_search(
    db: Chroma,
    query_embedding: list[float],
    keyword: str,
    k: int,
    metadata_filter: dict,
) -> list[tuple[Document, float]]:
    docs = similarity_search_by_embedding(
        db,
        query_embedding,
        k=max(k * 4, k),
        filter=metadata_filter,
    )
    matches = []
    for doc in docs:
        if keyword and keyword in doc.page_content:
            matches.append((doc, None))
            if len(matches) >= k:
                break
    return matches
