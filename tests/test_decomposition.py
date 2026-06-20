from __future__ import annotations

import json
from dataclasses import dataclass, field

from llm.schemas import ServiceResponse
from qa import decomposition
from qa.service import answer_with_evidence
from retrieval.models import QueryPlan, RetrievalResult


def _query_plan(query: str, *, modes: tuple[str, ...] = ("fact",)) -> QueryPlan:
    return QueryPlan(
        original_query=query,
        core_question=query,
        retrieval_focus=query,
        premise_claims=[],
        retrieval_query=query,
        keywords=[],
        persons=[],
        locations=[],
        events=[],
        objects=[],
        aliases=[],
        query_modes=modes,
        relation_intents=(),
        target_roles=[],
        target_volume=None,
        target_volume_index=None,
    )


def _retrieval_result(query: str, *, modes: tuple[str, ...] = ("fact",)) -> RetrievalResult:
    return RetrievalResult(
        query=query,
        retrieval_query=query,
        keywords=[],
        query_plan=_query_plan(query, modes=modes),
        chunks=[],
    )


def _service_response() -> ServiceResponse:
    return ServiceResponse(
        is_related=True,
        answer="ok",
        reason="test",
        evidence_quotes=[],
        premise_status="supported",
        premise_correction="",
        answer_mode="direct",
        is_blocked=False,
    )


@dataclass
class CallRecorder:
    planner_calls: int = 0
    retrieve_models: list[str | None] = field(default_factory=list)
    answer_models: list[str | None] = field(default_factory=list)


def _patch_single_query_dependencies(monkeypatch, calls: CallRecorder, *, planner_payload: dict | None = None) -> None:
    def fake_retrieve_context(_vector_db, query, **kwargs):
        calls.retrieve_models.append(kwargs.get("model"))
        return _retrieval_result(query)

    def fake_get_validated_response(_client, _messages, *, allow_open_ended=False, model=None):
        calls.answer_models.append(model)
        return _service_response()

    def fake_request_decomposition_payload(**_kwargs):
        calls.planner_calls += 1
        return planner_payload or {}

    monkeypatch.setattr(decomposition, "retrieve_context", fake_retrieve_context)
    monkeypatch.setattr(decomposition, "get_validated_response", fake_get_validated_response)
    monkeypatch.setattr(decomposition, "_request_decomposition_payload", fake_request_decomposition_payload)
    monkeypatch.setattr(decomposition, "relevant_alias_entries", lambda *_args, **_kwargs: [])


def test_should_use_decomposition_markers() -> None:
    assert not decomposition.should_use_decomposition("")
    assert not decomposition.should_use_decomposition("八奈见是谁")
    assert decomposition.should_use_decomposition("温水是不是只有老八家没去过")
    assert decomposition.should_use_decomposition("分别有哪些证据")
    assert decomposition.should_use_decomposition("哪些人去过")
    assert decomposition.should_use_decomposition("他有没有去过小鞠家")


def test_simple_question_skips_planner_and_preserves_model(monkeypatch) -> None:
    calls = CallRecorder()
    _patch_single_query_dependencies(monkeypatch, calls)

    retrieval_result, response = decomposition.answer_with_decomposition(
        client=object(),
        vector_db=object(),
        question="八奈见是谁",
        model="dsv4flash",
    )

    assert retrieval_result.query == "八奈见是谁"
    assert response.answer == "ok"
    assert calls.planner_calls == 0
    assert calls.retrieve_models == ["deepseek-v4-flash"]
    assert calls.answer_models == ["deepseek-v4-flash"]


def test_marker_question_calls_planner_once_when_planner_declines(monkeypatch) -> None:
    calls = CallRecorder()
    _patch_single_query_dependencies(
        monkeypatch,
        calls,
        planner_payload={
            "should_decompose": False,
            "subject": "温水",
            "intent": "single-query",
            "sub_questions": [{"label": "single", "question": "温水是不是只有老八家没去过"}],
        },
    )

    retrieval_result, response = decomposition.answer_with_decomposition(
        client=object(),
        vector_db=object(),
        question="温水是不是只有老八家没去过",
        model="deepseek-v4-pro",
    )

    assert retrieval_result.query == "温水是不是只有老八家没去过"
    assert response.answer == "ok"
    assert calls.planner_calls == 1
    assert calls.retrieve_models == ["deepseek-v4-pro"]
    assert calls.answer_models == ["deepseek-v4-pro"]


def test_marker_question_planner_request_uses_json_response(monkeypatch) -> None:
    payload = {"should_decompose": False, "sub_questions": [{"question": "分别说明"}]}
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            message = type("Message", (), {"content": json.dumps(payload, ensure_ascii=False)})
            choice = type("Choice", (), {"message": message})
            return type("Response", (), {"choices": [choice]})

    class FakeClient:
        chat = type("Chat", (), {"completions": FakeCompletions()})()

    plan = decomposition.build_decomposition_plan(client=FakeClient(), question="分别说明", model="dsv4pro")

    assert plan.should_decompose is False
    assert [item.question for item in plan.sub_questions] == ["分别说明"]
    assert captured["model"] == "deepseek-v4-pro"
    assert captured["response_format"] == {"type": "json_object"}


def test_answer_with_evidence_directly_delegates_to_decomposition(monkeypatch) -> None:
    sentinel = (_retrieval_result("问题"), _service_response())
    captured = {}

    def fake_answer_with_decomposition(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr("qa.service.answer_with_decomposition", fake_answer_with_decomposition)

    result = answer_with_evidence(
        client="client",
        vector_db="db",
        question="问题",
        corpus_names=["语料"],
        search_scope={"corpora": ["语料"]},
        question_history=[{"role": "user", "content": "上一问"}],
        selected_contexts=[],
        model="dsv4flash",
    )

    assert result is sentinel
    assert captured["client"] == "client"
    assert captured["vector_db"] == "db"
    assert captured["question"] == "问题"
    assert captured["model"] == "dsv4flash"
