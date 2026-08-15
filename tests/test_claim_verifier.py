"""Tests for LLM-backed claim verification, using a fake chat model."""

from __future__ import annotations

import pytest

from docuresearch.models.claims import Claim, VerificationStatus
from docuresearch.verification.claim_verifier import verify_claim
from tests.conftest import FakeChatModel


@pytest.mark.asyncio
async def test_verify_claim_with_no_excerpts_never_calls_the_model() -> None:
    fake_model = FakeChatModel({})  # would fail validation if actually invoked
    claim = Claim(claim_id="C-1", text="Something happened.")
    outcome = await verify_claim(claim, [], model=fake_model)
    assert outcome.verification_status == VerificationStatus.UNVERIFIED
    assert outcome.confidence == 0.0
    assert outcome.supports is False


@pytest.mark.asyncio
async def test_verify_claim_maps_llm_output() -> None:
    fake_model = FakeChatModel(
        {
            "verification_status": "verified",
            "confidence": 0.9,
            "supports": True,
            "notes": "Directly confirmed by the excerpt.",
        }
    )
    claim = Claim(claim_id="C-2", text="The policy took effect in 2016.")
    outcome = await verify_claim(claim, ["Official records confirm the policy took effect in 2016."], model=fake_model)

    assert outcome.verification_status == VerificationStatus.VERIFIED
    assert outcome.confidence == 0.9
    assert outcome.supports is True
    assert outcome.notes == "Directly confirmed by the excerpt."


@pytest.mark.asyncio
async def test_verify_claim_can_report_disputed() -> None:
    fake_model = FakeChatModel(
        {"verification_status": "disputed", "confidence": 0.5, "supports": False, "notes": "Sources conflict."}
    )
    claim = Claim(claim_id="C-3", text="Adoption began in year X.")
    outcome = await verify_claim(claim, ["Source A says year X.", "Source B says year X-1."], model=fake_model)
    assert outcome.verification_status == VerificationStatus.DISPUTED
