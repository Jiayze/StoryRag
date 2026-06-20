from __future__ import annotations


def render_constrained_answer_prompt() -> str:
    return (
        "You are a rigorous Chinese RAG assistant. Answer only from the provided internal document evidence.\n"
        "Hard constraints:\n"
        "1. Return a valid JSON string, with no Markdown code block.\n"
        "2. JSON must contain is_related, answer, reason, evidence_quotes, premise_status, premise_correction, answer_mode.\n"
        "3. Treat the user's narrative setup as claims to be verified, not as ground truth.\n"
        "4. Before answering, check whether the factual premises in the user's question are supported by the provided references.\n"
        "5. If a premise is unsupported or contradicted, explicitly point it out and correct it when the references allow. Do not silently accept a false premise.\n"
        "6. If the references are insufficient for the exact asked detail but enough to correct the premise or provide a partial answer, do that instead of simply saying the information is missing.\n"
        "7. Do not use outside world knowledge to fill gaps, and do not invent people, chapters, plots, or titles.\n"
        "8. evidence_quotes must be directly findable in the provided internal references.\n"
        "9. premise_status must be one of: supported, partially_supported, unsupported, contradicted.\n"
        "10. answer_mode must be one of: direct, corrected, partial, insufficient.\n"
        "11. Some retrieved chunks are marked as [Context Role] Primary Evidence and some as [Context Role] Expanded Neighbor.\n"
        "12. Prefer Primary Evidence for the core factual claim. Use Expanded Neighbor only to restore missing dialogue lead-in, follow-up, or immediate narrative continuity.\n"
        "13. Do not let Expanded Neighbor override a contradictory Primary Evidence chunk unless the combined local context clearly resolves an incomplete sentence or interrupted dialogue.\n"
        "14. When the question asks for a quote, dialogue content, confession content, or what happened immediately before/after, structure the answer in this order when possible: conclusion first, then core evidence, then supporting context.\n"
        "15. In answer text, make it clear which part is the core factual answer and which part is only surrounding context.\n"
        "16. Conversation History is not evidence. Use it only to resolve omitted references, pronouns, or follow-up intent.\n"
        "17. If Conversation History conflicts with Internal References, Internal References always win.\n"
        "18. Prior Selected Evidence comes from previously retrieved internal chunks that the user explicitly kept for follow-up. Treat it as evidence, but still verify it against the current question.\n"
        "19. User Alias Hints are user-maintained lexicon mappings for interpreting nicknames or shorthand in the question. They are not plot evidence by themselves; use them to understand the question, then answer only from Internal References."
    )


def build_answer_messages(
    *,
    context_text: str,
    selected_context_text: str = "",
    question: str,
    core_question: str = "",
    retrieval_focus: str = "",
    premise_claims: list[str] | None = None,
    question_history: list[dict[str, str] | str] | None = None,
    alias_hints: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    reference_sections: list[str] = []
    if selected_context_text.strip():
        reference_sections.append(f"[Prior Selected Evidence]\n{selected_context_text.strip()}")
    if context_text.strip():
        reference_sections.append(f"[Current Retrieval Evidence]\n{context_text.strip()}")
    references = "\n\n".join(reference_sections) if reference_sections else "No usable chunks found."

    premise_block = "\n".join(f"- {claim}" for claim in (premise_claims or []) if claim.strip()) or "- None extracted"
    history_block = _render_question_history(question_history)
    alias_block = _render_alias_hints(alias_hints)
    answer_style_guidance = (
        "[Preferred Answer Structure]\n"
        "1. 先给出一句最核心的结论。\n"
        "2. 再说明哪一段是核心证据。\n"
        "3. 如果使用了 Expanded Neighbor，只把它当作前后文补充，不要把补充前后文说成核心事实来源。\n"
        "4. 对“告白是什么 / 具体怎么说 / 前后发生了什么”这类问题，优先区分“核心内容”和“补充上下文”。"
    )
    user_content = (
        f"[Internal References]\n{references}\n\n"
        f"[Conversation History]\n{history_block}\n\n"
        f"[User Alias Hints]\n{alias_block}\n\n"
        f"[User Question]\n{question}\n\n"
        f"[Core Question]\n{core_question or question}\n\n"
        f"[Retrieval Focus]\n{retrieval_focus or core_question or question}\n\n"
        f"[Extracted Premise Claims]\n{premise_block}\n\n"
        f"{answer_style_guidance}"
    )
    return [
        {"role": "system", "content": render_constrained_answer_prompt()},
        {"role": "user", "content": user_content},
    ]


def _render_question_history(question_history: list[dict[str, str] | str] | None) -> str:
    if not question_history:
        return "None"

    rendered: list[str] = []
    for item in question_history[-6:]:
        if isinstance(item, str):
            text = _truncate_history_text(item.strip())
            if text:
                rendered.append(f"- {text}")
            continue
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "message")).strip() or "message"
        content = _truncate_history_text(str(item.get("content", "")).strip())
        if content:
            rendered.append(f"- {role}: {content}")

    return "\n".join(rendered) if rendered else "None"


def _render_alias_hints(alias_hints: list[dict[str, str]] | None) -> str:
    if not alias_hints:
        return "None"
    lines: list[str] = []
    for item in alias_hints:
        alias = str(item.get("alias", "")).strip()
        canonical = str(item.get("canonical", "")).strip()
        corpus_name = str(item.get("corpus_name", "")).strip()
        note = str(item.get("note", "")).strip()
        if not alias or not canonical:
            continue
        prefix = f"[{corpus_name}] " if corpus_name else ""
        suffix = f" ({note})" if note else ""
        lines.append(f"- {prefix}{alias} -> {canonical}{suffix}")
    return "\n".join(lines) if lines else "None"


def _truncate_history_text(text: str, *, limit: int = 400) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."
