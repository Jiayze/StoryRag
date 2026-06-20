from __future__ import annotations

from retrieval.query import _filter_locator_keywords, _is_locator_request_only


def test_filter_locator_keywords_keeps_short_known_entities() -> None:
    filtered = _filter_locator_keywords(
        ["月之桥", "星钥", "碎词", "第几章"],
        known_entities=["月之桥", "星钥"],
    )

    assert "月之桥" in filtered
    assert "星钥" in filtered
    assert "碎词" not in filtered
    assert "第几章" not in filtered


def test_filter_locator_keywords_keeps_short_terms_containing_known_entity() -> None:
    filtered = _filter_locator_keywords(
        ["校舍门", "坐摩", "场景"],
        known_entities=["校舍"],
    )

    assert "校舍门" in filtered
    assert "坐摩" not in filtered
    assert "场景" not in filtered


def test_locator_request_only_uses_known_entities_instead_of_fixed_names() -> None:
    assert _is_locator_request_only("小说中第几章", known_entities=[])
    assert not _is_locator_request_only("角色甲在小说中第几章", known_entities=["角色甲"])
