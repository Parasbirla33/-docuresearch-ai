"""Tests for the Phase 1 mock research dataset (spec section 47)."""

from __future__ import annotations

from docuresearch.mock.sample_data import build_mock_claims, build_mock_sources
from docuresearch.models.claims import VerificationStatus


def test_mock_sources_count_and_labelling() -> None:
    sources = build_mock_sources("Test Topic")
    assert len(sources) == 5
    assert all(s.title.startswith("[MOCK]") for s in sources)
    assert all(s.publisher == "Mock Source" for s in sources)


def test_mock_claims_count_and_disputed_content() -> None:
    sources = build_mock_sources("Test Topic")
    claims = build_mock_claims("Test Topic", sources)

    assert len(claims) == 10
    assert all(c.text.startswith("[MOCK]") for c in claims)

    disputed = [c for c in claims if c.verification_status == VerificationStatus.DISPUTED]
    assert len(disputed) == 3  # 2 contradictory + 1 disputed statistic

    contradictory_pair = [c for c in disputed if c.contradicting_evidence]
    assert len(contradictory_pair) >= 2

    stats = [c for c in claims if c.claim_type.value == "statistical"]
    assert len(stats) >= 1
