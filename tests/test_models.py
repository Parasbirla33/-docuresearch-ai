"""Validation and scoring-logic tests for the Pydantic models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from docuresearch.models.claims import Claim, ClaimType, VerificationStatus
from docuresearch.models.research import Hook, HookType
from docuresearch.models.script import QualityScore
from docuresearch.models.sources import Source, SourceType


def test_source_confidence_score_is_weighted_average() -> None:
    source = Source(
        id="S-1",
        title="Test Source",
        source_type=SourceType.GOVERNMENT_REPORT,
        authority_score=1.0,
        relevance_score=1.0,
        credibility_score=1.0,
    )
    assert source.confidence_score == 1.0

    zero = Source(id="S-2", title="Zero", source_type=SourceType.UNKNOWN)
    assert zero.confidence_score == 0.0


def test_source_score_bounds_are_enforced() -> None:
    with pytest.raises(ValidationError):
        Source(id="S-3", title="Bad", authority_score=1.5)


def test_claim_defaults_to_unverified() -> None:
    claim = Claim(claim_id="C-1", text="Something happened.")
    assert claim.verification_status == VerificationStatus.UNVERIFIED
    assert claim.claim_type == ClaimType.FACTUAL
    assert claim.source_ids == []


def test_hook_overall_score_penalizes_clickbait_risk() -> None:
    safe_hook = Hook(
        id="H-1",
        hook_type=HookType.QUESTION,
        text="What really happened?",
        curiosity_score=0.8,
        specificity_score=0.8,
        emotional_tension_score=0.8,
        information_gap_score=0.8,
        relevance_score=0.8,
        truthfulness_score=0.8,
        clickbait_risk=0.0,
    )
    risky_hook = safe_hook.model_copy(update={"id": "H-2", "clickbait_risk": 0.9})

    assert safe_hook.overall_score == pytest.approx(0.8)
    assert risky_hook.overall_score < safe_hook.overall_score


def test_quality_score_overall_is_weighted() -> None:
    perfect = QualityScore(
        research_depth=100,
        source_quality=100,
        claim_verification=100,
        narrative_quality=100,
        hook_quality=100,
        script_structure=100,
        citation_completeness=100,
        factual_safety=100,
    )
    assert perfect.overall == 100.0

    zero = QualityScore()
    assert zero.overall == 0.0
