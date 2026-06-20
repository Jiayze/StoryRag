from __future__ import annotations

from typing import Any

from core import get_logger
from .context import build_followup_context_options
from .decomposition import answer_with_decomposition


logger = get_logger(__name__)


def answer_with_evidence(
    *,
    client,
    vector_db,
    question: str,
    corpus_names: list[str] | None = None,
    search_scope: dict[str, Any] | None = None,
    question_history: list[dict[str, str] | str] | None = None,
    selected_contexts: list[dict[str, Any] | object] | None = None,
    model: str | None = None,
):
    logger.info(f"QA started for question: {question[:120]}")
    result = answer_with_decomposition(
        client=client,
        vector_db=vector_db,
        question=question,
        corpus_names=corpus_names,
        search_scope=search_scope,
        question_history=question_history,
        selected_contexts=selected_contexts,
        model=model,
    )
    logger.info("QA finished.")
    return result


def build_followup_payload_options(retrieval_result) -> list[dict[str, Any]]:
    return build_followup_context_options(retrieval_result.chunks)
