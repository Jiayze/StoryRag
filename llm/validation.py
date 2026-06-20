from __future__ import annotations

import re

from .schemas import ServiceResponse


FALSE_SIGNAL_PATTERNS = (
    "根据全书",
    "全书内容",
    "常识",
    "先验",
    "我知道",
    "我推测",
    "根据哈利波特",
    "结合原著",
    "从原著可知",
    "小说里",
    "在原著中",
    "显然",
)
VALID_PREMISE_STATUS = {"supported", "partially_supported", "unsupported", "contradicted"}
VALID_ANSWER_MODE = {"direct", "corrected", "partial", "insufficient"}


def enforce_grounding(
    validated_data: ServiceResponse,
    api_messages,
    *,
    allow_open_ended: bool = False,
) -> ServiceResponse:
    context_text = extract_context_text(api_messages)
    if not context_text:
        return mark_low_confidence(validated_data, "检索上下文为空，无法提供可校验的证据。")

    validated_data.premise_status = _normalize_choice(
        validated_data.premise_status,
        VALID_PREMISE_STATUS,
        "supported",
    )
    validated_data.answer_mode = _normalize_choice(
        validated_data.answer_mode,
        VALID_ANSWER_MODE,
        "direct",
    )

    if any(signal in validated_data.reason for signal in FALSE_SIGNAL_PATTERNS) or any(
        signal in validated_data.answer for signal in FALSE_SIGNAL_PATTERNS
    ):
        return mark_low_confidence(
            validated_data,
            "回答包含外部常识或先验推断信号，未严格受检索片段约束。",
        )

    quotes = [
        quote.strip()
        for quote in getattr(validated_data, "evidence_quotes", [])
        if quote and quote.strip()
    ]
    if validated_data.answer_mode in {"direct", "corrected", "partial"}:
        if not quotes:
            if not allow_open_ended:
                return mark_low_confidence(
                    validated_data,
                    "模型未提供可在检索片段中直接校验的原文证据。",
                )
            validated_data.reason = _append_guardrail_note(
                validated_data.reason,
                "开放式问题：未提供逐字引用，已按低强度校验放行。",
            )
        elif not quotes_supported_by_context(quotes, context_text):
            if not allow_open_ended:
                return mark_low_confidence(
                    validated_data,
                    "模型提供的证据片段无法在检索上下文中直接校验。",
                )
            validated_data.reason = _append_guardrail_note(
                validated_data.reason,
                "开放式问题：引用未能逐字匹配，已按综合分析问题放宽校验。",
            )

    if (
        validated_data.premise_status in {"unsupported", "contradicted"}
        and not validated_data.premise_correction.strip()
    ):
        return mark_low_confidence(
            validated_data,
            "模型识别出前提有误，但没有给出基于证据的纠正说明。",
        )

    return validated_data


def mark_low_confidence(validated_data: ServiceResponse, reason: str) -> ServiceResponse:
    validated_data.is_blocked = True
    validated_data.raw_answer = validated_data.answer
    validated_data.raw_reason = validated_data.reason
    validated_data.answer = (
        "【低可信度 / 需核对证据】\n"
        f"{reason}\n\n"
        "系统没有把下面内容当作可靠结论，请优先核对右侧检索证据。\n\n"
        f"模型原始回答：{validated_data.raw_answer or '无'}"
    )
    validated_data.reason = reason
    return validated_data


def extract_context_text(api_messages) -> str:
    for message in reversed(api_messages):
        if message.get("role") != "user":
            continue
        content = message.get("content", "")
        marker = "[Internal References]"
        if marker not in content:
            continue
        tail = content.split(marker, 1)[1]
        if "[Conversation History]" in tail:
            return tail.split("[Conversation History]", 1)[0].strip()
        if "[User Question]" in tail:
            return tail.split("[User Question]", 1)[0].strip()
        return tail.strip()
    return ""


def quotes_supported_by_context(quotes, context_text: str) -> bool:
    normalized_context = normalize_text(context_text)
    if not normalized_context:
        return False

    for quote in quotes:
        normalized_quote = normalize_text(quote)
        if not normalized_quote:
            return False
        if normalized_quote not in normalized_context:
            return False
    return True


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _normalize_choice(value: str, allowed: set[str], default: str) -> str:
    cleaned = str(value or "").strip().lower()
    return cleaned if cleaned in allowed else default


def _append_guardrail_note(reason: str, note: str) -> str:
    reason = str(reason or "").strip()
    if not reason:
        return note
    if note in reason:
        return reason
    return f"{reason}\n\nGuardrail note: {note}"
