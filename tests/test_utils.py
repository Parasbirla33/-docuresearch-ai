"""Tests for hashing/URL-normalization and text utilities."""

from __future__ import annotations

from docuresearch.utils.hashing import content_hash, new_id, normalize_url
from docuresearch.utils.text import truncate, word_count


def test_new_id_has_prefix_and_is_unique() -> None:
    a, b = new_id("S"), new_id("S")
    assert a.startswith("S-")
    assert b.startswith("S-")
    assert a != b


def test_content_hash_is_stable_and_sensitive_to_change() -> None:
    assert content_hash("hello world") == content_hash("hello world")
    assert content_hash("hello world") != content_hash("hello world!")


def test_normalize_url_dedupes_equivalent_urls() -> None:
    a = normalize_url("HTTPS://Example.com/Article/")
    b = normalize_url("https://example.com/Article")
    assert a == b


def test_normalize_url_strips_fragment_but_keeps_query() -> None:
    result = normalize_url("https://example.com/page?id=1#section-2")
    assert result == "https://example.com/page?id=1"


def test_word_count() -> None:
    assert word_count("one two three") == 3
    assert word_count("") == 0


def test_truncate_breaks_on_word_boundary() -> None:
    text = "The quick brown fox jumps over the lazy dog"
    result = truncate(text, 15)
    assert result.endswith("...")
    assert len(result) <= 15
    assert not result[: -len("...")].endswith(" ")
