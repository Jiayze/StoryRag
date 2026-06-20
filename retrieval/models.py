from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document


@dataclass
class QueryPlan:
    original_query: str
    core_question: str
    retrieval_focus: str
    premise_claims: list[str]
    retrieval_query: str
    keywords: list[str]
    persons: list[str]
    locations: list[str]
    events: list[str]
    objects: list[str]
    aliases: list[str]
    query_modes: tuple[str, ...]
    relation_intents: tuple[str, ...]
    target_roles: list[str]
    target_volume: str | None
    target_volume_index: int | None
    used_llm_enrichment: bool = False


@dataclass
class RankedChunk:
    document: Document
    distance: float | None
    dense_score: float
    lexical_score: float
    metadata_score: float
    summary_score: float
    relation_score: float
    position_score: float
    score: float
    is_context_expansion: bool = False


@dataclass(frozen=True)
class RankingWeights:
    dense: float
    lexical: float
    metadata: float
    summary: float
    relation: float
    position: float


@dataclass
class RetrievalResult:
    query: str
    retrieval_query: str
    keywords: list[str]
    query_plan: QueryPlan
    chunks: list[RankedChunk]

    @property
    def context_text(self) -> str:
        from .formatting import format_context

        return format_context(self.chunks)
