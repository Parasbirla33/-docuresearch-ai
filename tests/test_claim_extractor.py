"""Tests for LLM-backed claim extraction, using a fake chat model."""

from __future__ import annotations

import pytest

from docuresearch.extraction.claim_extractor import extract_claims
from docuresearch.models.claims import VerificationStatus
from tests.conftest import FakeChatModel

CANNED_CLAIMS = {
    "claims": [
        {
            "text": "The policy took effect in 2016.",
            "claim_type": "factual",
            "importance": "high",
            "requires_multiple_sources": True,
            "first_party": False,
        },
        {
            "text": "  ",  # blank text should be filtered out
            "claim_type": "factual",
            "importance": "low",
            "requires_multiple_sources": False,
            "first_party": False,
        },
    ]
}


@pytest.mark.asyncio
async def test_extract_claims_maps_llm_output_and_assigns_ids() -> None:
    fake_model = FakeChatModel(CANNED_CLAIMS)
    claims = await extract_claims(
        "The policy took effect in 2016 according to official records.",
        source_id="S-abc123",
        topic="Test Topic",
        model=fake_model,
    )

    assert len(claims) == 1  # blank-text claim filtered out
    claim = claims[0]
    assert claim.text == "The policy took effect in 2016."
    assert claim.source_ids == ["S-abc123"]
    assert claim.requires_multiple_sources is True
    assert claim.verification_status == VerificationStatus.UNVERIFIED
    assert claim.confidence == 0.0
    assert claim.claim_id.startswith("C-")


@pytest.mark.asyncio
async def test_extract_claims_returns_empty_for_blank_document() -> None:
    fake_model = FakeChatModel(CANNED_CLAIMS)
    claims = await extract_claims("   ", source_id="S-1", topic="Topic", model=fake_model)
    assert claims == []
