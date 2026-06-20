from __future__ import annotations

import importlib

import pytest

from core import config


def test_env_int_invalid_value_raises_value_error(monkeypatch) -> None:
    monkeypatch.setenv("STORYRAG_TEST_INT", "not-an-int")

    with pytest.raises(ValueError):
        config.env_int("STORYRAG_TEST_INT", 3)


def test_env_float_invalid_value_raises_value_error(monkeypatch) -> None:
    monkeypatch.setenv("STORYRAG_TEST_FLOAT", "not-a-float")

    with pytest.raises(ValueError):
        config.env_float("STORYRAG_TEST_FLOAT", 0.5)


def test_empty_max_distance_is_unbounded(monkeypatch) -> None:
    monkeypatch.setenv("RAG_MAX_DISTANCE", "")
    reloaded = importlib.reload(config)

    assert reloaded.DEFAULT_MAX_DISTANCE is None
