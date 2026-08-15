"""Tests for cross-claim contradiction detection."""

from __future__ import annotations

import pytest

from docuresearch.models.claims import Claim, ClaimType
from docuresearch.verification.contradiction_detector import (
    check_pair_for_contradiction,
    find_candidate_pairs,
)
from tests.conftest import FakeChatModel


def test_find_candidate_pairs_matches_similar_comparable_claims() -> None:
    a = Claim(claim_id="C-1", text="Company X entered the market in 2016.", claim_type=ClaimType.FACTUAL)
    b = Claim(claim_id="C-2", text="Company X entered the market in 2015.", claim_type=ClaimType.FACTUAL)

    pairs = find_candidate_pairs([a, b])
    assert len(pairs) == 1
    assert {pairs[0][0].claim_id, pairs[0][1].claim_id} == {"C-1", "C-2"}


def test_find_candidate_pairs_excludes_dissimilar_claims() -> None:
    a = Claim(claim_id="C-1", text="Company X entered the market in 2016.", claim_type=ClaimType.FACTUAL)
    b = Claim(claim_id="C-2", text="The weather was unusually cold that winter.", claim_type=ClaimType.FACTUAL)

    assert find_candidate_pairs([a, b]) == []


def test_find_candidate_pairs_excludes_non_comparable_types() -> None:
    a = Claim(claim_id="C-1", text="Company X entered the market in 2016.", claim_type=ClaimType.OPINION)
    b = Claim(claim_id="C-2", text="Company X entered the market in 2016.", claim_type=ClaimType.OPINION)

    assert find_candidate_pairs([a, b]) == []


def test_find_candidate_pairs_caps_and_sorts_by_similarity() -> None:
    claims = [
        Claim(claim_id=f"C-{i}", text=f"Company X entered market segment {i} in 2016.", claim_type=ClaimType.FACTUAL)
        for i in range(10)
    ]
    pairs = find_candidate_pairs(claims)
    assert len(pairs) <= 20  # MAX_CANDIDATE_PAIRS


@pytest.mark.asyncio
async def test_check_pair_for_contradiction_maps_llm_output() -> None:
    fake_model = FakeChatModel({"conflicts": True, "description": "Dates disagree."})
    a = Claim(claim_id="C-1", text="Launched in 2016.")
    b = Claim(claim_id="C-2", text="Launched in 2015.")

    verdict = await check_pair_for_contradiction(a, b, model=fake_model)
    assert verdict.conflicts is True
    assert verdict.description == "Dates disagree."


@pytest.mark.asyncio
async def test_check_pair_for_contradiction_can_report_no_conflict() -> None:
    fake_model = FakeChatModel({"conflicts": False, "description": None})
    a = Claim(claim_id="C-1", text="Launched nationally in 2016.")
    b = Claim(claim_id="C-2", text="Launched in the capital city in 2016.")

    verdict = await check_pair_for_contradiction(a, b, model=fake_model)
    assert verdict.conflicts is False
