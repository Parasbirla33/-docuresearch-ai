"""Tests for the file-based research cache."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from docuresearch.storage.cache import ResearchCache


def test_cache_miss_returns_none(tmp_path: Path) -> None:
    cache = ResearchCache(tmp_path)
    assert cache.get("https://example.com/page") is None


def test_cache_set_then_get_round_trips(tmp_path: Path) -> None:
    cache = ResearchCache(tmp_path)
    cache.set("https://example.com/page", title="Title", content="Body text", metadata={"k": "v"})
    entry = cache.get("https://example.com/page")
    assert entry is not None
    assert entry.title == "Title"
    assert entry.content == "Body text"
    assert entry.metadata == {"k": "v"}


def test_cache_normalizes_equivalent_urls_to_same_entry(tmp_path: Path) -> None:
    cache = ResearchCache(tmp_path)
    cache.set("HTTPS://Example.com/Page/", title="T", content="C")
    entry = cache.get("https://example.com/Page")
    assert entry is not None
    assert entry.title == "T"


def test_cache_expires_stale_entries(tmp_path: Path) -> None:
    cache = ResearchCache(tmp_path, ttl=timedelta(seconds=-1))
    cache.set("https://example.com/page", title="T", content="C")
    assert cache.get("https://example.com/page") is None


def test_cache_invalidate_removes_entry(tmp_path: Path) -> None:
    cache = ResearchCache(tmp_path)
    cache.set("https://example.com/page", title="T", content="C")
    cache.invalidate("https://example.com/page")
    assert cache.get("https://example.com/page") is None
