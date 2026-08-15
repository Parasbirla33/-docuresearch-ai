"""Tests for the independent script fact-checker (narration-vs-claim, not claim-vs-source)."""

from __future__ import annotations

import pytest

from docuresearch.models.claims import Claim
from docuresearch.models.script import FactCheckVerdict
from docuresearch.verification.script_fact_checker import fact_check_section
from tests.conftest import FakeChatModel


@pytest.mark.asyncio
async def test_fact_check_section_with_no_cited_claims_never_calls_model() -> None:
    fake_model = FakeChatModel({})  # would fail validation if actually invoked
    findings = await fact_check_section("Some narration.", [], model=fake_model)
    assert len(findings) == 1
    assert findings[0].verdict == FactCheckVerdict.NEEDS_REVISION
    assert findings[0].unsupported_inference is True


@pytest.mark.asyncio
async def test_fact_check_section_maps_llm_output() -> None:
    claim = Claim(claim_id="C-1", text="The policy took effect in 2016.")
    fake_model = FakeChatModel(
        {
            "findings": [
                {
                    "claim_id": "C-1",
                    "verdict": "needs_revision",
                    "exaggeration_detected": True,
                    "missing_context": False,
                    "unsupported_inference": False,
                    "notes": "Narration overstates certainty.",
                }
            ]
        }
    )

    findings = await fact_check_section("The policy definitely transformed everything overnight.", [claim], model=fake_model)

    assert len(findings) == 1
    assert findings[0].claim_id == "C-1"
    assert findings[0].verdict == FactCheckVerdict.NEEDS_REVISION
    assert findings[0].exaggeration_detected is True


@pytest.mark.asyncio
async def test_fact_check_section_drops_invented_claim_id() -> None:
    claim = Claim(claim_id="C-1", text="Real claim.")
    fake_model = FakeChatModel(
        {"findings": [{"claim_id": "C-999", "verdict": "pass", "notes": "ok"}]}
    )
    findings = await fact_check_section("Narration.", [claim], model=fake_model)
    assert findings[0].claim_id is None  # invented ID must not be trusted


@pytest.mark.asyncio
async def test_fact_check_section_handles_empty_findings() -> None:
    claim = Claim(claim_id="C-1", text="Real claim.")
    fake_model = FakeChatModel({"findings": []})
    findings = await fact_check_section("Narration.", [claim], model=fake_model)
    assert len(findings) == 1
    assert findings[0].verdict == FactCheckVerdict.NEEDS_REVISION
