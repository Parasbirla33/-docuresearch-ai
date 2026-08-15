"""Claim, evidence, quote and statistic models used by the verification pipeline."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ClaimType(StrEnum):
    FACTUAL = "factual"
    STATISTICAL = "statistical"
    QUOTE = "quote"
    CAUSAL = "causal"
    INTERPRETIVE = "interpretive"
    OPINION = "opinion"
    ALLEGATION = "allegation"
    HISTORICAL_EVENT = "historical_event"


class ClaimImportance(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class VerificationStatus(StrEnum):
    VERIFIED = "verified"
    PARTIALLY_VERIFIED = "partially_verified"
    DISPUTED = "disputed"
    UNVERIFIED = "unverified"
    FALSE_OR_UNSUPPORTED = "false_or_unsupported"


class Claim(BaseModel):
    """A single factual assertion extracted from research, tracked to its sources."""

    claim_id: str
    text: str
    claim_type: ClaimType = ClaimType.FACTUAL
    importance: ClaimImportance = ClaimImportance.MEDIUM

    source_ids: list[str] = Field(default_factory=list)
    supporting_evidence: list[str] = Field(default_factory=list)
    contradicting_evidence: list[str] = Field(default_factory=list)

    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED

    first_party: bool = False
    requires_multiple_sources: bool = False
    notes: str | None = None


class EvidenceMatrixEntry(BaseModel):
    """A row in the evidence matrix: one claim, its supporting/contradicting sources."""

    claim_id: str
    claim_text: str
    supporting_source_ids: list[str] = Field(default_factory=list)
    contradicting_source_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    status: VerificationStatus = VerificationStatus.UNVERIFIED


class Contradiction(BaseModel):
    """A detected conflict between two or more claims/sources on the same topic."""

    id: str
    claim_ids: list[str]
    description: str
    side_a_source_ids: list[str] = Field(default_factory=list)
    side_b_source_ids: list[str] = Field(default_factory=list)
    severity: ClaimImportance = ClaimImportance.MEDIUM
    resolution_notes: str | None = None


class Quote(BaseModel):
    """A direct quotation with full provenance. Never fabricated."""

    quote: str
    speaker: str
    source_id: str
    date: str | None = None
    url_or_page: str | None = None
    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED


class Statistic(BaseModel):
    """A quantitative claim with the context required to use it responsibly."""

    value: str
    unit: str | None = None
    year: str | None = None
    geography: str | None = None
    source_id: str
    source_url: str | None = None
    context: str | None = None
    is_range: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
