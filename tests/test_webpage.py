"""Tests for the webpage extraction tool: SSRF guard, robots.txt, status handling, size cap.

All network I/O is mocked (respx for HTTP, monkeypatched DNS) - no live network required.
"""

from __future__ import annotations

import socket

import httpx
import pytest
import respx

from docuresearch.models.sources import SourceAvailability
from docuresearch.tools.webpage import WebpageExtractor

PUBLIC_IP_INFO = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))]
PRIVATE_IP_INFO = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.5", 0))]

SAMPLE_HTML = """
<html>
<head><title>Sample Article</title></head>
<body>
<h1>Sample Article</h1>
<p>This is a sufficiently long paragraph of body text so that the extractor
considers it real content rather than an empty or near-empty page, well past
the minimum content character threshold used by the extraction tool.</p>
</body>
</html>
"""


def _mock_getaddrinfo(monkeypatch: pytest.MonkeyPatch, info: list) -> None:
    monkeypatch.setattr("docuresearch.tools.webpage.socket.getaddrinfo", lambda host, port: info)


@pytest.mark.asyncio
async def test_blocks_private_ip_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_getaddrinfo(monkeypatch, PRIVATE_IP_INFO)
    extractor = WebpageExtractor()
    result = await extractor.extract("http://internal.example/secret")
    assert result.availability == SourceAvailability.ERROR
    assert "safety" in (result.error or "").lower()


@pytest.mark.asyncio
async def test_blocks_non_http_scheme(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_getaddrinfo(monkeypatch, PUBLIC_IP_INFO)
    extractor = WebpageExtractor()
    result = await extractor.extract("file:///etc/passwd")
    assert result.availability == SourceAvailability.ERROR


@pytest.mark.asyncio
@respx.mock
async def test_robots_disallowed_blocks_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_getaddrinfo(monkeypatch, PUBLIC_IP_INFO)
    respx.get("https://example.com/robots.txt").mock(
        return_value=httpx.Response(200, text="User-agent: *\nDisallow: /")
    )
    extractor = WebpageExtractor()
    result = await extractor.extract("https://example.com/article")
    assert result.availability == SourceAvailability.ROBOTS_BLOCKED


@pytest.mark.asyncio
@respx.mock
async def test_successful_extraction(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_getaddrinfo(monkeypatch, PUBLIC_IP_INFO)
    respx.get("https://example.com/robots.txt").mock(return_value=httpx.Response(404))
    respx.get("https://example.com/article").mock(
        return_value=httpx.Response(200, html=SAMPLE_HTML)
    )
    extractor = WebpageExtractor()
    result = await extractor.extract("https://example.com/article")
    assert result.availability == SourceAvailability.AVAILABLE
    assert result.text is not None
    assert "sufficiently long paragraph" in result.text


@pytest.mark.asyncio
@respx.mock
async def test_404_marks_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_getaddrinfo(monkeypatch, PUBLIC_IP_INFO)
    respx.get("https://example.com/robots.txt").mock(return_value=httpx.Response(404))
    respx.get("https://example.com/missing").mock(return_value=httpx.Response(404))
    extractor = WebpageExtractor()
    result = await extractor.extract("https://example.com/missing")
    assert result.availability == SourceAvailability.UNAVAILABLE
    assert result.status_code == 404


@pytest.mark.asyncio
@respx.mock
async def test_403_marks_paywalled(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_getaddrinfo(monkeypatch, PUBLIC_IP_INFO)
    respx.get("https://example.com/robots.txt").mock(return_value=httpx.Response(404))
    respx.get("https://example.com/paywalled").mock(return_value=httpx.Response(403))
    extractor = WebpageExtractor()
    result = await extractor.extract("https://example.com/paywalled")
    assert result.availability == SourceAvailability.PAYWALLED


@pytest.mark.asyncio
@respx.mock
async def test_timeout_marks_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_getaddrinfo(monkeypatch, PUBLIC_IP_INFO)
    respx.get("https://example.com/robots.txt").mock(return_value=httpx.Response(404))
    respx.get("https://example.com/slow").mock(side_effect=httpx.TimeoutException("timed out"))
    extractor = WebpageExtractor()
    result = await extractor.extract("https://example.com/slow")
    assert result.availability == SourceAvailability.ERROR
    assert "timed out" in (result.error or "").lower()


@pytest.mark.asyncio
@respx.mock
async def test_oversized_response_marks_partial(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_getaddrinfo(monkeypatch, PUBLIC_IP_INFO)
    respx.get("https://example.com/robots.txt").mock(return_value=httpx.Response(404))
    respx.get("https://example.com/huge").mock(return_value=httpx.Response(200, html=SAMPLE_HTML * 100))
    extractor = WebpageExtractor(max_response_bytes=200)
    result = await extractor.extract("https://example.com/huge")
    assert result.availability == SourceAvailability.PARTIAL
