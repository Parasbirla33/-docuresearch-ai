"""Robust webpage fetch + extraction tool.

Security requirements from spec section 45 are enforced here specifically:
- only http/https URLs are fetched
- the resolved IP must be public (SSRF guard against private/loopback/link-local ranges)
- responses are streamed and capped at a maximum size
- every request has a timeout
- robots.txt is honored; disallowed pages are never fetched, only marked ROBOTS_BLOCKED
- CAPTCHAs/auth/paywalls are never bypassed - a 401/403 is reported, not worked around

If content cannot be accessed, the result is marked unavailable/blocked/error;
nothing is ever fabricated in its place.
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from urllib.parse import urlparse

import httpx
import trafilatura
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

from docuresearch.config.settings import Settings, get_settings
from docuresearch.models.sources import SourceAvailability
from docuresearch.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_USER_AGENT = "DocuResearchAI/0.1 (+research bot; contact: none configured)"
DEFAULT_MAX_RESPONSE_BYTES = 5_000_000  # 5 MB
MIN_CONTENT_CHARS = 50
ALLOWED_SCHEMES = {"http", "https"}


class ExtractionResult(BaseModel):
    """Outcome of trying to fetch and extract one URL. Never fabricates content."""

    availability: SourceAvailability
    final_url: str | None = None
    title: str | None = None
    text: str | None = None
    headings: list[str] = Field(default_factory=list)
    status_code: int | None = None
    error: str | None = None


class _FetchFailure(Exception):
    def __init__(self, availability: SourceAvailability, status_code: int | None, message: str) -> None:
        super().__init__(message)
        self.availability = availability
        self.status_code = status_code
        self.message = message


async def _is_safe_public_url(url: str) -> bool:
    """SSRF guard: only allow http(s) URLs that resolve to public IP addresses."""
    parsed = urlparse(url)
    if parsed.scheme not in ALLOWED_SCHEMES:
        return False
    host = parsed.hostname
    if not host:
        return False
    try:
        infos = await asyncio.to_thread(socket.getaddrinfo, host, None)
    except socket.gaierror:
        return False
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except ValueError:
            return False
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast or ip.is_unspecified:
            return False
    return True


class WebpageExtractor:
    """Fetches a URL and extracts clean, readable text - or reports why it couldn't."""

    def __init__(
        self,
        *,
        timeout: float = 30.0,
        max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
        user_agent: str = DEFAULT_USER_AGENT,
    ) -> None:
        self._timeout = timeout
        self._max_bytes = max_response_bytes
        self._user_agent = user_agent

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> WebpageExtractor:
        settings = settings or get_settings()
        return cls(timeout=settings.request_timeout)

    async def extract(self, url: str) -> ExtractionResult:
        if not await _is_safe_public_url(url):
            logger.warning("webpage_blocked_unsafe_url", url=url)
            return ExtractionResult(
                availability=SourceAvailability.ERROR,
                error="URL failed safety validation (invalid scheme or non-public address).",
            )

        if not await self._robots_allowed(url):
            logger.info("webpage_robots_blocked", url=url)
            return ExtractionResult(
                availability=SourceAvailability.ROBOTS_BLOCKED,
                error="Disallowed by robots.txt.",
            )

        try:
            html, final_url, status_code = await self._fetch(url)
        except _FetchFailure as exc:
            logger.info("webpage_fetch_failed", url=url, availability=exc.availability.value, error=exc.message)
            return ExtractionResult(availability=exc.availability, status_code=exc.status_code, error=exc.message)

        text, title, headings = self._parse(html, final_url)
        if not text or len(text.strip()) < MIN_CONTENT_CHARS:
            return ExtractionResult(
                availability=SourceAvailability.UNAVAILABLE,
                final_url=final_url,
                status_code=status_code,
                error="No extractable content found.",
            )

        return ExtractionResult(
            availability=SourceAvailability.AVAILABLE,
            final_url=final_url,
            title=title,
            text=text,
            headings=headings,
            status_code=status_code,
        )

    async def _robots_allowed(self, url: str) -> bool:
        from urllib.robotparser import RobotFileParser

        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        try:
            async with httpx.AsyncClient(timeout=min(self._timeout, 10.0), headers={"User-Agent": self._user_agent}) as client:
                response = await client.get(robots_url)
        except httpx.HTTPError:
            return True  # robots.txt unreachable - default allow, per common convention

        if response.status_code >= 400:
            return True  # no robots.txt published - default allow

        parser = RobotFileParser()
        parser.parse(response.text.splitlines())
        return parser.can_fetch(self._user_agent, url)

    async def _fetch(self, url: str) -> tuple[str, str, int]:
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout,
                follow_redirects=True,
                max_redirects=5,
                headers={"User-Agent": self._user_agent},
            ) as client, client.stream("GET", url) as response:
                self._raise_for_status(response)
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    body.extend(chunk)
                    if len(body) > self._max_bytes:
                        raise _FetchFailure(
                            SourceAvailability.PARTIAL,
                            response.status_code,
                            f"Response exceeded max size ({self._max_bytes} bytes); truncated.",
                        )
                html = bytes(body).decode(response.encoding or "utf-8", errors="replace")
                return html, str(response.url), response.status_code
        except httpx.TimeoutException as exc:
            raise _FetchFailure(SourceAvailability.ERROR, None, "Request timed out.") from exc
        except httpx.TransportError as exc:
            raise _FetchFailure(SourceAvailability.ERROR, None, f"Network/SSL error: {exc}") from exc

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        if response.status_code in (401, 403):
            raise _FetchFailure(
                SourceAvailability.PAYWALLED,
                response.status_code,
                f"HTTP {response.status_code}: access denied (possibly paywalled/auth-gated).",
            )
        if response.status_code == 404:
            raise _FetchFailure(SourceAvailability.UNAVAILABLE, 404, "404 Not Found.")
        if response.status_code == 429:
            raise _FetchFailure(SourceAvailability.ERROR, 429, "429 Too Many Requests.")
        if response.status_code >= 400:
            raise _FetchFailure(SourceAvailability.ERROR, response.status_code, f"HTTP {response.status_code}.")

    @staticmethod
    def _parse(html: str, final_url: str) -> tuple[str | None, str | None, list[str]]:
        text = trafilatura.extract(html, url=final_url, favor_recall=True, include_comments=False)

        soup = BeautifulSoup(html, "lxml")
        title = soup.title.get_text(strip=True) if soup.title else None
        headings = [h.get_text(strip=True) for h in soup.find_all(["h1", "h2", "h3"]) if h.get_text(strip=True)][:20]

        if not text:
            for tag in soup(["script", "style", "nav", "footer", "aside", "form"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)

        return text, title, headings
