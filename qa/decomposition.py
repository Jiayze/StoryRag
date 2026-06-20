from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from llm import DEEPSEEK_MODEL, build_answer_messages, get_validated_response, normalize_deepseek_model
from retrieval import RetrievalResult, analyze_query, retrieve_context
from retrieval.aliases import relevant_alias_entries, render_alias_hints

from .context import format_selected_contexts


DECOMPOSITION_MARKERS = (
    "是不是只有",
    "是否只有",
    "只有",
    "分别",
    "逐个",
    "哪些",
    "谁家",
    "哪几家",
    "有没有去过",
    "没去过",
    "去过谁",
    "都去过",
)


@dataclass(slots=True)
class DecomposedSubQuestion:
    label: str
    question: str
    expected_target: str = ""


@dataclass(slots=True)
class DecompositionPlan:
    subject: str
    original_question: str
    intent: str
    should_decompose: bool
    sub_questions: list[DecomposedSubQuestion]


def should_use_decomposition(question: str) -> bool:
    compact = re.sub(r"\s+", "", question or "")
    if not compact:
        return False
    return any(marker in compact for marker in DECOMPOSITION_MARKERS)


def answer_with_decomposition(
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
    active_model = normalize_deepseek_model(model or DEEPSEEK_MODEL)
    print(f"[INFO] Decomposition QA started for question: {question[:120]}")
    scope_corpora = _scope_corpus_names(search_scope) or (corpus_names or [])
    alias_hints = relevant_alias_entries(question, scope_corpora)
    plan = build_decomposition_plan(client=client, question=question, model=active_model, alias_hints=alias_hints)
    if not plan.should_decompose or len(plan.sub_questions) <= 1:
        print("[INFO] Decomposition planner declined multi-query; falling back to single retrieval.")
        retrieval_result = retrieve_context(
            vector_db,
            question,
            corpus_names=corpus_names,
            search_scope=search_scope,
            model=active_model,
        )
        selected_context_text = format_selected_contexts(selected_contexts)
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
        allow_open_ended = "open_ended" in set(retrieval_result.query_plan.query_modes)
        validated_res = get_validated_response(
            client,
            api_messages,
            allow_open_ended=allow_open_ended,
            model=active_model,
        )
        if retrieval_result.chunks and validated_res.is_blocked:
            validated_res.is_related = True
        return retrieval_result, validated_res

    print(f"[INFO] Decomposition produced {len(plan.sub_questions)} sub-questions.")
    evidence_sections: list[str] = []
    sub_results: list[tuple[DecomposedSubQuestion, RetrievalResult]] = []
    selected_context_text = format_selected_contexts(selected_contexts)

    for sub_question in plan.sub_questions:
        print(f"[INFO] Retrieving sub-question: {sub_question.question[:120]}")
        retrieval_result = retrieve_context(
            vector_db,
            sub_question.question,
            corpus_names=corpus_names,
            search_scope=search_scope,
            top_k=4,
            model=active_model,
        )
        sub_results.append((sub_question, retrieval_result))
        evidence_sections.append(
            _render_sub_result_block(
                sub_question=sub_question,
                retrieval_result=retrieval_result,
            )
        )

    merged_retrieval_result = _merge_retrieval_results(
        question=question,
        sub_results=sub_results,
        model=active_model,
    )
    synthesis_question = _build_synthesis_question(plan, sub_results)
    print("[INFO] Decomposition synthesis answer generation started.")
    api_messages = build_answer_messages(
        context_text="\n\n".join(section for section in evidence_sections if section.strip()),
        selected_context_text=selected_context_text,
        question=synthesis_question,
        core_question=question,
        retrieval_focus="Multi-query RAG verification: answer each sub-question, then synthesize.",
        premise_claims=[question],
        question_history=question_history,
        alias_hints=alias_hints,
    )
    validated_res = get_validated_response(client, api_messages, model=active_model)
    if merged_retrieval_result.chunks and validated_res.is_blocked:
        validated_res.is_related = True
    print("[INFO] Decomposition QA finished.")
    return merged_retrieval_result, validated_res


def build_decomposition_plan(
    *,
    client,
    question: str,
    model: str | None = None,
    alias_hints: list[dict[str, str]] | None = None,
) -> DecompositionPlan:
    payload = _request_decomposition_payload(client=client, question=question, model=model, alias_hints=alias_hints)
    subject = str(payload.get("subject", "")).strip()
    intent = str(payload.get("intent", "")).strip() or "multi_query_verification"
    should_decompose = bool(payload.get("should_decompose", False))

    sub_questions: list[DecomposedSubQuestion] = []
    for item in payload.get("sub_questions", []) or []:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).strip()
        sub_question = str(item.get("question", "")).strip()
        expected_target = str(item.get("expected_target", "")).strip()
        if not sub_question:
            continue
        sub_questions.append(
            DecomposedSubQuestion(
                label=label or expected_target or sub_question[:16],
                question=sub_question,
                expected_target=expected_target,
            )
        )

    if should_decompose and not sub_questions:
        sub_questions = _fallback_sub_questions(question)
    if not sub_questions:
        sub_questions = [DecomposedSubQuestion(label="single-query", question=question)]

    return DecompositionPlan(
        subject=subject or _fallback_subject(question),
        original_question=question,
        intent=intent,
        should_decompose=should_decompose,
        sub_questions=sub_questions,
    )


def _request_decomposition_payload(
    *,
    client,
    question: str,
    model: str | None = None,
    alias_hints: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    active_model = normalize_deepseek_model(model or DEEPSEEK_MODEL)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a Chinese fiction RAG query planner.\n"
                "Return only JSON. Do not answer the user's question.\n"
                "Use [User Alias Hints] as a user-maintained lexicon. If an alias appears in the question, understand it as its canonical name and keep both terms useful for retrieval.\n"
                "Decide whether the question needs multiple RAG searches.\n"
                "Use multi-query decomposition for set membership, exclusivity, comparison, exhaustive checks, or questions like 'A是不是只有B没去过'.\n"
                "For '温水是不是只有老八家没去过', produce exhaustive sub-questions for every candidate home/family target you can infer, including the named negative target and comparison targets such as 八奈见家、小鞠家、烧盐家、天爱星家 when relevant.\n"
                "For exclusivity questions, do not stop at the named target; create separate searches for A有没有去过B, A有没有去过C, A有没有去过D, etc.\n"
                "If you are unsure whether the candidate set is complete, still list concrete likely candidates and set intent to include 'candidate_set_may_be_incomplete'.\n"
                "Each sub-question must be directly searchable and mention the subject plus one concrete target.\n"
                "If the question is a normal single fact/why question, set should_decompose=false and return one sub-question equal to the original question.\n"
                "JSON schema:\n"
                "{"
                "\"should_decompose\": boolean,"
                "\"subject\": string,"
                "\"intent\": string,"
                "\"sub_questions\": ["
                "{\"label\": string, \"question\": string, \"expected_target\": string}"
                "]"
                "}"
            ),
        },
        {"role": "user", "content": f"[User Alias Hints]\n{render_alias_hints(alias_hints)}\n\n[Question]\n{question}"},
    ]
    try:
        print(f"[INFO] DeepSeek RAG query planning started with model={active_model}.")
        response = client.chat.completions.create(
            model=active_model,
            messages=messages,
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        payload = json.loads(content)
        if isinstance(payload, dict):
            print("[SUCCESS] DeepSeek RAG query planning completed.")
            return payload
    except Exception as exc:
        print(f"[INFO] RAG query planning fallback due to error: {exc}")
    return {}


def _fallback_sub_questions(question: str) -> list[DecomposedSubQuestion]:
    return [DecomposedSubQuestion(label="single-query", question=question)]


def _render_sub_result_block(
    *,
    sub_question: DecomposedSubQuestion,
    retrieval_result: RetrievalResult,
) -> str:
    lines = [
        f"[Sub-question Label] {sub_question.label}",
        f"[Sub-question] {sub_question.question}",
    ]
    if sub_question.expected_target:
        lines.append(f"[Expected Target] {sub_question.expected_target}")
    lines.append("[Retrieved Evidence]")
    lines.append(retrieval_result.context_text or "No usable chunks found.")
    return "\n".join(lines)


def _build_synthesis_question(
    plan: DecompositionPlan,
    sub_results: list[tuple[DecomposedSubQuestion, RetrievalResult]],
) -> str:
    targets = "、".join(
        sub_question.expected_target or sub_question.label
        for sub_question, _ in sub_results
        if (sub_question.expected_target or sub_question.label)
    )
    return (
        f"{plan.original_question}\n\n"
        "请基于上面的分项检索证据逐项回答。"
        f"需要核对的对象包括：{targets or '见各子问题'}。\n"
        "要求：先给总判断，再逐项说明每个对象是否被证据支持，最后明确原问题中的“只有/是否只有/分别”等判断是否成立。"
        "如果某一项证据不足，要明确写证据不足，不能把没有搜到当作否定事实。"
    )


def _merge_retrieval_results(
    *,
    question: str,
    sub_results: list[tuple[DecomposedSubQuestion, RetrievalResult]],
    model: str | None = None,
) -> RetrievalResult:
    if not sub_results:
        return RetrievalResult(
            query=question,
            retrieval_query=question,
            keywords=[],
            query_plan=analyze_query(question, model=model),
            chunks=[],
        )

    merged_query_plan = analyze_query(question, model=model)
    merged_chunks = []
    seen_keywords: set[str] = set()
    seen_chunks: set[str] = set()
    merged_keywords: list[str] = []
    for _, result in sub_results:
        for keyword in result.keywords:
            if keyword not in seen_keywords:
                seen_keywords.add(keyword)
                merged_keywords.append(keyword)
        for chunk in result.chunks:
            chunk_id = str(chunk.document.metadata.get("chunk_id", ""))
            dedupe_key = chunk_id or chunk.document.page_content[:80]
            if dedupe_key in seen_chunks:
                continue
            seen_chunks.add(dedupe_key)
            merged_chunks.append(chunk)

    return RetrievalResult(
        query=question,
        retrieval_query=question,
        keywords=merged_keywords,
        query_plan=merged_query_plan,
        chunks=merged_chunks,
    )


def _scope_corpus_names(search_scope: dict[str, Any] | None) -> list[str]:
    if not isinstance(search_scope, dict):
        return []
    return [str(name).strip() for name in search_scope.get("corpora", []) or [] if str(name).strip()]


def _fallback_subject(question: str) -> str:
    compact = re.sub(r"\s+", "", question or "")
    match = re.match(r"(.+?)(是不是|是否|有没有|有无|只有)", compact)
    if match:
        return match.group(1)
    return compact[:8] or "该对象"
