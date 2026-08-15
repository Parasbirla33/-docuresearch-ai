"""Pluggable web search tool.

Default concrete provider: Tavily (https://tavily.com), chosen because its
API is purpose-built for LLM/agent retrieval (concise snippets, clean source
URLs, no HTML scraping required) and has a usable free tier. `SEARCH_API_KEY`
in settings is currently interpreted as a Tavily key.

To add another provider, implement `SearchProvider` and return it from
`get_search_provider()` based on config - nothing else in the codebase should
need to change, since callers only depend on the protocol.
"""

from __future__ import annotations

from typing import Protocol

import httpx
from pydantic import BaseModel, Field

from docuresearch.config.settings import Settings, get_settings
from docuresearch.utils.logging import get_logger
from docuresearch.utils.retry import with_retry

logger = get_logger(__name__)

TAVILY_SEARCH_URL = "https://api.tavily.com/search"


class SearchHit(BaseModel):
    """A single web search result, prior to full-page extraction."""

    title: str
    url: str
    snippet: str = ""
    score: float = Field(default=0.0, ge=0.0, le=1.0)


class SearchProvider(Protocol):
    async def search(self, query: str, *, max_results: int = 10) -> list[SearchHit]: ...


class TavilySearchProvider:
    """Concrete `SearchProvider` backed by the Tavily search API."""

    def __init__(self, api_key: str, *, timeout: float = 30.0) -> None:
        self._api_key = api_key
        self._timeout = timeout

    async def search(self, query: str, *, max_results: int = 10) -> list[SearchHit]:
        payload = {"api_key": self._api_key, "query": query, "max_results": max_results}
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await self._post(client, payload)
        results = response.json().get("results", [])
        return [
            SearchHit(
                title=r.get("title") or r.get("url", ""),
                url=r["url"],
                snippet=r.get("content", ""),
                score=min(max(r.get("score", 0.0), 0.0), 1.0),
            )
            for r in results
            if r.get("url")
        ]

    @with_retry(attempts=3, exceptions=(httpx.TransportError, httpx.HTTPStatusError))
    async def _post(self, client: httpx.AsyncClient, payload: dict[str, object]) -> httpx.Response:
        response = await client.post(TAVILY_SEARCH_URL, json=payload)
        response.raise_for_status()
        return response


class NullSearchProvider:
    """No-op provider used when no SEARCH_API_KEY is configured.

    Never raises - callers should treat an empty result list as "this source
    type contributed nothing this run," not as an error.
    """

    async def search(self, query: str, *, max_results: int = 10) -> list[SearchHit]:
        logger.warning("web_search_skipped", reason="no_search_api_key", query=query)
        return []


def get_search_provider(settings: Settings | None = None) -> SearchProvider:
    """Resolve the configured search provider, or a graceful no-op if unavailable."""
    settings = settings or get_settings()
    if not settings.has_search_provider:
        return NullSearchProvider()
    assert settings.search_api_key is not None
    return TavilySearchProvider(api_key=settings.search_api_key, timeout=settings.request_timeout)
