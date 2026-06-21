from __future__ import annotations

from llm.schemas import ServiceResponse
from llm.validation import mark_low_confidence


def test_mark_low_confidence_keeps_answer_without_redundant_labels() -> None:
    response = ServiceResponse(
        is_related=True,
        answer="八奈见是文中的角色。",
        reason="原始理由",
        evidence_quotes=[],
    )

    downgraded = mark_low_confidence(response, "证据无法逐字校验。")

    assert downgraded.is_blocked is True
    assert downgraded.answer == "【低可信度 / 需核对证据】\n八奈见是文中的角色。"
    assert downgraded.reason == "证据无法逐字校验。"
    assert downgraded.raw_answer == "八奈见是文中的角色。"
    assert "模型原始回答" not in downgraded.answer
    assert "系统没有把下面内容当作可靠结论" not in downgraded.answer
    assert "证据无法逐字校验。" not in downgraded.answer
