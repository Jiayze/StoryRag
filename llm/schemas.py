from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class ServiceResponse(BaseModel):
    is_related: bool = Field(description="Whether the retrieved content is relevant to the user question.")
    answer: str = Field(description="Answer grounded only in the provided internal evidence.")
    reason: str = Field(description="Short explanation for the answer and relevance judgment.")
    evidence_quotes: list[str] = Field(
        default_factory=list,
        description="Direct evidence snippets that must be verifiable in retrieved context.",
    )
    premise_status: str = Field(
        default="supported",
        description="Whether the factual premises in the question are supported by the evidence.",
    )
    premise_correction: str = Field(
        default="",
        description="Correction for unsupported or contradicted premises, using only the provided evidence.",
    )
    answer_mode: str = Field(
        default="direct",
        description="One of direct, corrected, partial, insufficient.",
    )
    is_blocked: bool = Field(default=False, description="Whether the answer was downgraded by guardrails.")
    raw_answer: str = Field(default="", description="Original model answer before downgrade.")
    raw_reason: str = Field(default="", description="Original model reasoning before downgrade.")

    @field_validator("answer", "reason", "premise_status", "premise_correction", "answer_mode", "raw_answer", "raw_reason", mode="before")
    @classmethod
    def _coerce_str(cls, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, (list, tuple)):
            return " ".join(str(item).strip() for item in value if str(item).strip())
        if isinstance(value, dict):
            return " ".join(f"{key}:{val}" for key, val in value.items() if str(val).strip())
        return str(value)

    @field_validator("is_related", "is_blocked", mode="before")
    @classmethod
    def _coerce_bool(cls, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        text = str(value).strip().lower()
        return text in {"true", "1", "yes", "y", "是"}

    @field_validator("evidence_quotes", mode="before")
    @classmethod
    def _coerce_quotes(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, tuple):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            cleaned = value.strip()
            return [cleaned] if cleaned else []
        if isinstance(value, dict):
            flattened = [str(item).strip() for item in value.values() if str(item).strip()]
            return flattened
        cleaned = str(value).strip()
        return [cleaned] if cleaned else []
