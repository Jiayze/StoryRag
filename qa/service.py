from __future__ import annotations

from typing import Any

from core import get_logger
from llm import build_answer_messages, get_validated_response
from retrieval import retrieve_context
from retrieval.aliases import relevant_alias_entries
from .context import build_followup_context_options, format_selected_contexts
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
    logger.info("QA asks planner whether multi-query RAG is needed.")
    planned_result = answer_with_decomposition(
        client=client,
        vector_db=vector_db,
        question=question,
        corpus_names=corpus_names,
        search_scope=search_scope,
        question_history=question_history,
        selected_contexts=selected_contexts,
        model=model,
    )
    if planned_result is not None:
        logger.info("QA finished through planner path.")
        return planned_result

    logger.info("Retrieval started.")
    retrieval_result = retrieve_context(vector_db, question, corpus_names=corpus_names, search_scope=search_scope, model=model)
    logger.info(f"Retrieval completed with {len(retrieval_result.chunks)} ranked chunks.")
    selected_context_text = format_selected_contexts(selected_contexts)
    scope_corpora = _scope_corpus_names(search_scope) or (corpus_names or [])
    alias_hints = relevant_alias_entries(question, scope_corpora)
    logger.info("Answer prompt assembly started.")
    api_messages = build_answer_messages(
        context_text=retrieval_result.context_text,
        selected_context_text=selected_context_text,
        question=question,
        core_question=retrieval_result.query_plan.core_question,
        retrieval_focus=retrieval_result.query_plan.retrieval_focus,
        premise_claims=retrieval_result.query_plan.premise_claims,
        question_history=question_history,
        alias_hints=alias_hints,
    )
    logger.info("Validation/generation started.")
    allow_open_ended = "open_ended" in set(retrieval_result.query_plan.query_modes)
    logger.info(f"Guardrail mode: allow_open_ended={allow_open_ended}.")
    validated_res = get_validated_response(
        client,
        api_messages,
        allow_open_ended=allow_open_ended,
    )
    if retrieval_result.chunks and validated_res.is_blocked:
        validated_res.is_related = True
    logger.info("QA finished.")
    return retrieval_result, validated_res


def build_followup_payload_options(retrieval_result) -> list[dict[str, Any]]:
    return build_followup_context_options(retrieval_result.chunks)


def _scope_corpus_names(search_scope: dict[str, Any] | None) -> list[str]:
    if not isinstance(search_scope, dict):
        return []
    return [str(name).strip() for name in search_scope.get("corpora", []) or [] if str(name).strip()]
