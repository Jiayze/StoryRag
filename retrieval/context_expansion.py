from __future__ import annotations

import json
from dataclasses import dataclass

from core import get_logger
from llm.client import DEEPSEEK_MODEL, create_deepseek_client, normalize_deepseek_model


logger = get_logger(__name__)


@dataclass(slots=True)
class ExpansionDecision:
    chunk_id: str
    is_high_value: bool = False
    need_prev_chunk: bool = False
    need_next_chunk: bool = False
    reason: str = ""


def decide_chunk_expansions(
    *,
    question: str,
    candidates: list[dict[str, str]],
    max_candidates: int = 3,
    model: str | None = None,
) -> dict[str, ExpansionDecision]:
    active_model = normalize_deepseek_model(model or DEEPSEEK_MODEL)
    trimmed = [
        {
            "chunk_id": str(item.get("chunk_id", "")).strip(),
            "chapter": str(item.get("chapter", "")).strip(),
            "text": str(item.get("text", "")).strip()[:900],
        }
        for item in candidates[:max_candidates]
        if str(item.get("chunk_id", "")).strip()
    ]
    if not trimmed:
        return {}

    try:
        client = create_deepseek_client()
    except Exception:
        return {}

    logger.info(
        f"DeepSeek context-expansion started with model={active_model} for {len(trimmed)} candidate chunks."
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a Chinese fiction retrieval context-expansion judge.\n"
                "Return only JSON.\n"
                "Given a user question and candidate chunks, decide whether each chunk is a high-value anchor that needs adjacent context.\n"
                "Use need_prev_chunk/need_next_chunk when the answer likely depends on preceding or following narration/dialogue.\n"
                "Be conservative. Do not expand everything.\n"
                "JSON schema:\n"
                "{"
                "\"decisions\": ["
                "{\"chunk_id\": string, \"is_high_value\": boolean, \"need_prev_chunk\": boolean, \"need_next_chunk\": boolean, \"reason\": string}"
                "]"
                "}"
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "question": question,
                    "candidates": trimmed,
                },
                ensure_ascii=False,
            ),
        },
    ]

    try:
        response = client.chat.completions.create(
            model=active_model,
            messages=messages,
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        payload = json.loads(response.choices[0].message.content or "{}")
        logger.info("DeepSeek context-expansion completed.")
    except Exception as exc:
        logger.warning(f"DeepSeek context-expansion skipped due to error: {exc}")
        return {}

    decisions: dict[str, ExpansionDecision] = {}
    for item in payload.get("decisions", []) or []:
        if not isinstance(item, dict):
            continue
        chunk_id = str(item.get("chunk_id", "")).strip()
        if not chunk_id:
            continue
        decisions[chunk_id] = ExpansionDecision(
            chunk_id=chunk_id,
            is_high_value=bool(item.get("is_high_value", False)),
            need_prev_chunk=bool(item.get("need_prev_chunk", False)),
            need_next_chunk=bool(item.get("need_next_chunk", False)),
            reason=str(item.get("reason", "")).strip(),
        )
    return decisions
