"""Tests for the web search tool. All HTTP calls are mocked via respx."""

from __future__ import annotations

import httpx
import pytest
import respx

from docuresearch.config.settings import Settings
from docuresearch.tools.web_search import (
    NullSearchProvider,
    TavilySearchProvider,
    get_search_provider,
)

TAVILY_URL = "https://api.tavily.com/search"


@pytest.mark.asyncio
async def test_null_provider_returns_empty_and_never_raises() -> None:
    provider = NullSearchProvider()
    hits = await provider.search("anything")
    assert hits == []


@pytest.mark.asyncio
@respx.mock
async def test_tavily_provider_parses_results() -> None:
    respx.post(TAVILY_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {"title": "A", "url": "https://a.example", "content": "snippet a", "score": 0.9},
                    {"title": "B", "url": "https://b.example", "content": "snippet b", "score": 0.5},
                ]
            },
        )
    )
    provider = TavilySearchProvider(api_key="test-key")
    hits = await provider.search("query", max_results=5)
    assert len(hits) == 2
    assert hits[0].url == "https://a.example"
    assert hits[0].score == 0.9


@pytest.mark.asyncio
@respx.mock
async def test_tavily_provider_skips_results_without_url() -> None:
    respx.post(TAVILY_URL).mock(
        return_value=httpx.Response(200, json={"results": [{"title": "No URL", "content": "x"}]})
    )
    provider = TavilySearchProvider(api_key="test-key")
    hits = await provider.search("query")
    assert hits == []


def test_get_search_provider_falls_back_to_null_without_key() -> None:
    settings = Settings(_env_file=None, search_api_key=None)  # type: ignore[call-arg]
    provider = get_search_provider(settings)
    assert isinstance(provider, NullSearchProvider)


def test_get_search_provider_uses_tavily_with_key() -> None:
    settings = Settings(_env_file=None, search_api_key="test-key")  # type: ignore[call-arg]
    provider = get_search_provider(settings)
    assert isinstance(provider, TavilySearchProvider)
