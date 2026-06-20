from __future__ import annotations

import json
import re

from .schemas import ServiceResponse
from .validation import enforce_grounding
from .client import DEEPSEEK_MODEL, normalize_deepseek_model


def get_validated_response(
    client,
    api_messages,
    retries: int = 1,
    *,
    allow_open_ended: bool = False,
    model: str | None = None,
) -> ServiceResponse:
    active_model = normalize_deepseek_model(model or DEEPSEEK_MODEL)
    for attempt in range(retries + 1):
        try:
            print(f"[INFO] DeepSeek answer generation started with model={active_model} (attempt {attempt + 1}/{retries + 1}).")
            response = client.chat.completions.create(
                model=active_model,
                messages=api_messages,
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            raw_content = (response.choices[0].message.content or "").strip()
            clean_content = re.sub(r"^```json\s*|```$", "", raw_content, flags=re.IGNORECASE).strip()
            try:
                validated_data = ServiceResponse.model_validate_json(clean_content)
            except Exception:
                payload = json.loads(clean_content)
                validated_data = ServiceResponse.model_validate(_normalize_payload(payload))
            validated_data = enforce_grounding(
                validated_data,
                api_messages,
                allow_open_ended=allow_open_ended,
            )
            print("[SUCCESS] DeepSeek answer generation completed.")

            phone_pattern = re.compile(r"(1[3-9]\d)\d{4}(\d{4})")
            validated_data.answer = phone_pattern.sub(r"\1****\2", validated_data.answer)
            validated_data.premise_correction = phone_pattern.sub(r"\1****\2", validated_data.premise_correction)
            return validated_data
        except Exception as exc:
            print(f"[DEBUG] Validation failed on attempt {attempt + 1}: {exc}")
            if attempt == retries:
                return ServiceResponse(
                    is_related=False,
                    answer="对不起，内部服务响应异常，请稍后再试。",
                    reason=f"安全护栏拦截：结构化校验未通过。异常信息: {str(exc)[:80]}",
                    evidence_quotes=[],
                    premise_status="unsupported",
                    premise_correction="",
                    answer_mode="insufficient",
                    is_blocked=True,
                    raw_answer="",
                    raw_reason="",
                )


def _normalize_payload(payload: dict) -> dict:
    normalized = dict(payload or {})
    if "premise_correction" not in normalized or normalized.get("premise_correction") is None:
        normalized["premise_correction"] = ""
    if "evidence_quotes" in normalized and not isinstance(normalized.get("evidence_quotes"), list):
        value = normalized.get("evidence_quotes")
        if value is None:
            normalized["evidence_quotes"] = []
        else:
            normalized["evidence_quotes"] = [str(value)]
    return normalized
