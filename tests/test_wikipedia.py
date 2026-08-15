"""Tests for the Wikipedia tool. All HTTP calls are mocked via respx."""

from __future__ import annotations

import httpx
import pytest
import respx

from docuresearch.tools.wikipedia import WikipediaTool

WIKI_API = "https://en.wikipedia.org/w/api.php"


@pytest.mark.asyncio
async def test_disabled_tool_returns_empty_search() -> None:
    tool = WikipediaTool(enabled=False)
    assert await tool.search("Test Topic") == []
    assert await tool.get_page("Test Topic") is None


@pytest.mark.asyncio
@respx.mock
async def test_search_returns_titles() -> None:
    respx.get(WIKI_API).mock(
        return_value=httpx.Response(
            200,
            json={"query": {"search": [{"title": "Telecommunications in India"}, {"title": "Reliance Jio"}]}},
        )
    )
    tool = WikipediaTool(enabled=True)
    titles = await tool.search("Indian telecom revolution")
    assert titles == ["Telecommunications in India", "Reliance Jio"]


@pytest.mark.asyncio
@respx.mock
async def test_get_page_extracts_text_and_references() -> None:
    respx.get(WIKI_API).mock(
        return_value=httpx.Response(
            200,
            json={
                "query": {
                    "pages": [
                        {
                            "pageid": 123,
                            "title": "Telecommunications in India",
                            "extract": "India's telecom sector grew rapidly after liberalization.",
                            "fullurl": "https://en.wikipedia.org/wiki/Telecommunications_in_India",
                            "extlinks": [{"url": "https://trai.gov.in/report"}, {"url": "https://example.com/study"}],
                        }
                    ]
                }
            },
        )
    )
    tool = WikipediaTool(enabled=True)
    page = await tool.get_page("Telecommunications in India")
    assert page is not None
    assert page.page_id == 123
    assert "liberalization" in page.extract
    assert "https://trai.gov.in/report" in page.references
    assert len(page.references) == 2


@pytest.mark.asyncio
@respx.mock
async def test_get_page_returns_none_when_missing() -> None:
    respx.get(WIKI_API).mock(
        return_value=httpx.Response(200, json={"query": {"pages": [{"title": "Nonexistent Page", "missing": True}]}})
    )
    tool = WikipediaTool(enabled=True)
    page = await tool.get_page("Nonexistent Page")
    assert page is None


@pytest.mark.asyncio
@respx.mock
async def test_search_handles_http_error_gracefully() -> None:
    respx.get(WIKI_API).mock(return_value=httpx.Response(500))
    tool = WikipediaTool(enabled=True)
    titles = await tool.search("Anything")
    assert titles == []
