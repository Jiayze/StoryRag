from __future__ import annotations

from langchain_core.documents import Document

import app_services
from retrieval.formatting import format_debug_table
from retrieval.models import RankedChunk


def test_format_debug_table_omits_preview() -> None:
    chunk = RankedChunk(
        document=Document(
            page_content="这是一段不应进入调试表 preview 列的正文。",
            metadata={"chapter": "第一章", "chunk_index": 7},
        ),
        distance=0.123456,
        dense_score=0.9,
        lexical_score=0.8,
        metadata_score=0.7,
        summary_score=0.6,
        relation_score=0.5,
        position_score=0.4,
        score=0.88,
    )

    rows = format_debug_table([chunk])

    assert rows
    row = rows[0]
    assert "preview" not in row
    for key in ("rank", "score", "distance", "chapter", "chunk"):
        assert key in row
    assert row["chapter"] == "第一章"
    assert row["chunk"] == 7


def test_debug_panel_setting_defaults_to_hidden(monkeypatch) -> None:
    monkeypatch.delenv("STORYRAG_SHOW_DEBUG_PANEL", raising=False)

    assert app_services.load_runtime_settings()["STORYRAG_SHOW_DEBUG_PANEL"] == "0"


def test_save_runtime_settings_persists_debug_panel_flag(tmp_path, monkeypatch) -> None:
    env_path = tmp_path / ".env"

    def fake_resolve_project_path(_value, _default):
        return env_path

    monkeypatch.setattr(app_services, "resolve_project_path", fake_resolve_project_path)

    app_services.save_runtime_settings({"STORYRAG_SHOW_DEBUG_PANEL": "1"})

    assert "STORYRAG_SHOW_DEBUG_PANEL=1" in env_path.read_text(encoding="utf-8")
    assert app_services.load_runtime_settings()["STORYRAG_SHOW_DEBUG_PANEL"] == "1"

    app_services.save_runtime_settings({"STORYRAG_SHOW_DEBUG_PANEL": "0"})

    assert "STORYRAG_SHOW_DEBUG_PANEL=0" in env_path.read_text(encoding="utf-8")
    assert app_services.load_runtime_settings()["STORYRAG_SHOW_DEBUG_PANEL"] == "0"
