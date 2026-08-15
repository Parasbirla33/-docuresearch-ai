"""Source and document models: what the research agent collects and reads."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, HttpUrl


class SourceType(StrEnum):
    """Category of a research source, used for diversity and authority scoring."""

    WEB_SEARCH = "web_search"
    WIKIPEDIA = "wikipedia"
    GOVERNMENT = "government"
    GOVERNMENT_REPORT = "government_report"
    COURT_DOCUMENT = "court_document"
    ACADEMIC_PAPER = "academic_paper"
    RESEARCH_ORGANIZATION = "research_organization"
    NEWS = "news"
    COMPANY_OFFICIAL = "company_official"
    NGO_REPORT = "ngo_report"
    INTERNATIONAL_ORGANIZATION = "international_organization"
    BOOK_PUBLICATION = "book_publication"
    ARCHIVE = "archive"
    PUBLIC_DATASET = "public_dataset"
    REGULATORY_BODY = "regulatory_body"
    UNKNOWN = "unknown"


class SourceAvailability(StrEnum):
    """Whether the source's content was actually retrievable."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    PARTIAL = "partial"
    PAYWALLED = "paywalled"
    ROBOTS_BLOCKED = "robots_blocked"
    ERROR = "error"


class Source(BaseModel):
    """A single research source: a discovered URL/document and its scoring."""

    id: str
    url: HttpUrl | None = None
    title: str
    publisher: str | None = None
    author: str | None = None
    publication_date: datetime | None = None
    accessed_at: datetime | None = None
    source_type: SourceType = SourceType.UNKNOWN
    domain: str | None = None

    text: str | None = None
    summary: str | None = None

    authority_score: float = Field(default=0.0, ge=0.0, le=1.0)
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    credibility_score: float = Field(default=0.0, ge=0.0, le=1.0)

    primary_source: bool = False
    archived: bool = False
    language: str = "en"

    availability: SourceAvailability = SourceAvailability.AVAILABLE
    content_hash: str | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def confidence_score(self) -> float:
        """Normalized 0-1 composite of authority/relevance/credibility."""
        weights = (0.4, 0.25, 0.35)
        scores = (self.authority_score, self.relevance_score, self.credibility_score)
        return round(sum(w * s for w, s in zip(weights, scores)), 4)


class Document(BaseModel):
    """Extracted, cleaned content pulled from a Source (article body etc.)."""

    id: str
    source_id: str
    raw_html: str | None = None
    clean_text: str
    headings: list[str] = Field(default_factory=list)
    word_count: int = 0
    extracted_at: datetime | None = None
    extraction_method: str = "unknown"
    metadata: dict[str, Any] = Field(default_factory=dict)
