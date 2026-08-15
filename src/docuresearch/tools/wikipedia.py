"""Wikipedia research tool.

Per spec, Wikipedia is used for topic discovery, chronology, terminology and
as a source of candidate references - not as a final authority for important
claims. It needs no API key; `WIKIPEDIA_API_ENABLED` (settings) can disable
it entirely.
"""

from __future__ import annotations

import httpx
from pydantic import BaseModel, Field

from docuresearch.config.settings import Settings, get_settings
from docuresearch.utils.logging import get_logger

logger = get_logger(__name__)


def _user_agent(contact: str) -> str:
    """Build a Wikimedia-policy-compliant User-Agent.

    Wikimedia's API rejects User-Agents that don't contain an actual contact
    URL/email - a descriptive phrase alone (e.g. "research bot") 403s, even
    with a plausible-looking UA string otherwise. Verified against the live
    API: https://meta.wikimedia.org/wiki/User-Agent_policy
    """
    return f"DocuResearchAI/0.1 ({contact})"


class WikipediaPage(BaseModel):
    """A fetched Wikipedia article: plain-text extract plus candidate references."""

    title: str
    page_id: int
    extract: str
    url: str
    references: list[str] = Field(default_factory=list)


class WikipediaTool:
    """Thin async client over the public Wikipedia MediaWiki API."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        language: str = "en",
        timeout: float = 30.0,
        contact: str = "https://github.com/docuresearch-ai/docuresearch-ai",
    ) -> None:
        self._enabled = enabled
        self._language = language
        self._timeout = timeout
        self._api_base = f"https://{language}.wikipedia.org/w/api.php"
        self._user_agent = _user_agent(contact)

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> WikipediaTool:
        settings = settings or get_settings()
        return cls(
            enabled=settings.wikipedia_api_enabled,
            timeout=settings.request_timeout,
            contact=settings.wikipedia_contact,
        )

    async def search(self, query: str, *, max_results: int = 5) -> list[str]:
        """Return candidate article titles for a query."""
        if not self._enabled:
            logger.warning("wikipedia_skipped", reason="disabled_in_settings", query=query)
            return []

        params: dict[str, str | int] = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": max_results,
            "format": "json",
            "formatversion": "2",
        }
        async with httpx.AsyncClient(timeout=self._timeout, headers={"User-Agent": self._user_agent}) as client:
            try:
                response = await client.get(self._api_base, params=params)
                response.raise_for_status()
            except httpx.HTTPError as exc:
                logger.warning("wikipedia_search_failed", query=query, error=str(exc))
                return []

        hits = response.json().get("query", {}).get("search", [])
        return [h["title"] for h in hits]

    async def get_page(self, title: str) -> WikipediaPage | None:
        """Fetch a plain-text extract and candidate reference (external) links for a title."""
        if not self._enabled:
            logger.warning("wikipedia_skipped", reason="disabled_in_settings", title=title)
            return None

        params = {
            "action": "query",
            "titles": title,
            "prop": "extracts|extlinks|info",
            "explaintext": "1",
            "exsectionformat": "plain",
            "ellimit": "200",
            "inprop": "url",
            "format": "json",
            "formatversion": "2",
        }
        async with httpx.AsyncClient(timeout=self._timeout, headers={"User-Agent": self._user_agent}) as client:
            try:
                response = await client.get(self._api_base, params=params)
                response.raise_for_status()
            except httpx.HTTPError as exc:
                logger.warning("wikipedia_get_page_failed", title=title, error=str(exc))
                return None

        pages = response.json().get("query", {}).get("pages", [])
        page = next((p for p in pages if "missing" not in p), None)
        if page is None:
            logger.info("wikipedia_page_not_found", title=title)
            return None

        references = [link["url"] for link in page.get("extlinks", []) if link.get("url")]
        return WikipediaPage(
            title=page.get("title", title),
            page_id=page.get("pageid", 0),
            extract=page.get("extract", ""),
            url=page.get("fullurl", f"https://{self._language}.wikipedia.org/wiki/{title}"),
            references=references,
        )
