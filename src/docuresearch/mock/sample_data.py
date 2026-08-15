"""Deterministic-shape mock research data for `--mock` runs and tests.

Everything produced here is clearly labelled MOCK/placeholder content - it must
never be mistaken for real, fact-checked research output. Used to exercise the
full LangGraph pipeline (Phase 1) before real research tools (Phase 2+) exist.

Per spec (section 47): 5 fake sources, 10 claims, 2 contradictory claims,
1 disputed statistic.
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import HttpUrl

from docuresearch.models.claims import (
    Claim,
    ClaimImportance,
    ClaimType,
    VerificationStatus,
)
from docuresearch.models.sources import Source, SourceType
from docuresearch.utils.hashing import new_id

MOCK_TAG = "[MOCK]"


def build_mock_sources(topic: str) -> list[Source]:
    now = datetime.now(UTC)
    specs = [
        ("Official Government Report", SourceType.GOVERNMENT_REPORT, True, 0.95, 0.9, 0.93),
        ("Reputable News Outlet Coverage", SourceType.NEWS, False, 0.6, 0.85, 0.7),
        ("Academic Paper on the Subject", SourceType.ACADEMIC_PAPER, True, 0.9, 0.8, 0.92),
        ("Wikipedia Background Overview", SourceType.WIKIPEDIA, False, 0.5, 0.75, 0.6),
        ("Industry Body Press Statement", SourceType.COMPANY_OFFICIAL, False, 0.4, 0.7, 0.45),
    ]
    sources: list[Source] = []
    for title, stype, primary, authority, relevance, credibility in specs:
        sources.append(
            Source(
                id=new_id("S"),
                url=HttpUrl(f"https://example.org/mock/{stype.value}"),
                title=f"{MOCK_TAG} {title} - {topic}",
                publisher="Mock Source",
                author="Mock Author",
                publication_date=now,
                accessed_at=now,
                source_type=stype,
                domain="example.org",
                text=f"{MOCK_TAG} Placeholder extracted text discussing {topic}.",
                summary=f"{MOCK_TAG} Placeholder summary about {topic}.",
                authority_score=authority,
                relevance_score=relevance,
                credibility_score=credibility,
                primary_source=primary,
                archived=False,
                language="en",
            )
        )
    return sources


def build_mock_claims(topic: str, sources: list[Source]) -> list[Claim]:
    gov, news, academic, wiki, industry = sources

    claims: list[Claim] = [
        Claim(
            claim_id=new_id("C"),
            text=f"{MOCK_TAG} A major policy change related to {topic} took effect.",
            claim_type=ClaimType.HISTORICAL_EVENT,
            importance=ClaimImportance.HIGH,
            source_ids=[gov.id, news.id],
            confidence=0.9,
            verification_status=VerificationStatus.VERIFIED,
            requires_multiple_sources=True,
        ),
        Claim(
            claim_id=new_id("C"),
            text=f"{MOCK_TAG} Independent researchers documented the effects of {topic}.",
            claim_type=ClaimType.FACTUAL,
            importance=ClaimImportance.HIGH,
            source_ids=[academic.id],
            confidence=0.85,
            verification_status=VerificationStatus.VERIFIED,
        ),
        Claim(
            claim_id=new_id("C"),
            text=f"{MOCK_TAG} Background context on {topic} is widely documented.",
            claim_type=ClaimType.FACTUAL,
            importance=ClaimImportance.LOW,
            source_ids=[wiki.id],
            confidence=0.6,
            verification_status=VerificationStatus.PARTIALLY_VERIFIED,
        ),
        Claim(
            claim_id=new_id("C"),
            text=f"{MOCK_TAG} An industry group claims {topic} drove rapid growth.",
            claim_type=ClaimType.OPINION,
            importance=ClaimImportance.MEDIUM,
            source_ids=[industry.id],
            confidence=0.4,
            verification_status=VerificationStatus.UNVERIFIED,
            notes="Single, non-independent source - self-interested party.",
        ),
        Claim(
            claim_id=new_id("C"),
            text=f"{MOCK_TAG} Regulators alleged irregularities connected to {topic}.",
            claim_type=ClaimType.ALLEGATION,
            importance=ClaimImportance.CRITICAL,
            source_ids=[gov.id],
            confidence=0.55,
            verification_status=VerificationStatus.PARTIALLY_VERIFIED,
            notes="Allegation, not an established finding - keep as alleged.",
        ),
        Claim(
            claim_id=new_id("C"),
            text=f"{MOCK_TAG} A notable public figure played a central role in {topic}.",
            claim_type=ClaimType.FACTUAL,
            importance=ClaimImportance.MEDIUM,
            source_ids=[academic.id, news.id],
            confidence=0.8,
            verification_status=VerificationStatus.VERIFIED,
        ),
        Claim(
            claim_id=new_id("C"),
            text=f"{MOCK_TAG} There is no reliable source confirming a rumored side-effect of {topic}.",
            claim_type=ClaimType.FACTUAL,
            importance=ClaimImportance.LOW,
            source_ids=[],
            confidence=0.1,
            verification_status=VerificationStatus.UNVERIFIED,
            notes="No source found - flagged as unverified, excluded from script as fact.",
        ),
    ]

    # --- 2 explicitly contradictory claims -----------------------------------
    claim_supports = Claim(
        claim_id=new_id("C"),
        text=f"{MOCK_TAG} Market adoption tied to {topic} began in year X (per official report).",
        claim_type=ClaimType.STATISTICAL,
        importance=ClaimImportance.HIGH,
        source_ids=[gov.id],
        supporting_evidence=[gov.id],
        contradicting_evidence=[industry.id],
        confidence=0.5,
        verification_status=VerificationStatus.DISPUTED,
    )
    claim_disputes = Claim(
        claim_id=new_id("C"),
        text=f"{MOCK_TAG} Market adoption tied to {topic} began a full year earlier (per industry statement).",
        claim_type=ClaimType.STATISTICAL,
        importance=ClaimImportance.HIGH,
        source_ids=[industry.id],
        supporting_evidence=[industry.id],
        contradicting_evidence=[gov.id],
        confidence=0.5,
        verification_status=VerificationStatus.DISPUTED,
    )
    claims.extend([claim_supports, claim_disputes])

    # --- 1 disputed statistic --------------------------------------------------
    disputed_stat = Claim(
        claim_id=new_id("C"),
        text=f"{MOCK_TAG} Reported figures on {topic}'s impact range between two conflicting estimates.",
        claim_type=ClaimType.STATISTICAL,
        importance=ClaimImportance.HIGH,
        source_ids=[gov.id, academic.id],
        supporting_evidence=[gov.id],
        contradicting_evidence=[academic.id],
        confidence=0.45,
        verification_status=VerificationStatus.DISPUTED,
        notes="Government figure and academic estimate disagree - preserved as a range, not resolved.",
    )
    claims.append(disputed_stat)

    assert len(claims) == 10, f"mock claim generator must produce 10 claims, got {len(claims)}"
    return claims
