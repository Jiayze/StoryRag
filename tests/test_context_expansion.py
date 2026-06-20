from __future__ import annotations

from retrieval.ranking import merge_adjacent_texts


def test_merge_adjacent_texts_handles_missing_neighbors() -> None:
    assert merge_adjacent_texts("", "middle", "") == "middle"
    assert merge_adjacent_texts("before", "", "after") == "before\nafter"


def test_merge_adjacent_texts_appends_without_overlap() -> None:
    overlap = "shared bridge segment"
    prev_text = "A" * 25 + overlap
    current_text = overlap + "B" * 25

    assert merge_adjacent_texts(prev_text, current_text, "") == "A" * 25 + overlap + "B" * 25


def test_merge_adjacent_texts_adds_newlines_when_no_overlap() -> None:
    assert merge_adjacent_texts("alpha", "beta", "gamma") == "alpha\nbeta\ngamma"
