"""Tests for the LLM-backed hook generation agent."""

from __future__ import annotations

import pytest

from docuresearch.agents.hook_agent import generate_hooks
from docuresearch.models.claims import Claim
from docuresearch.models.research import HookType
from tests.conftest import FakeChatModel

CANNED_HOOKS = {
    "hooks": [
        {
            "hook_type": "question",
            "text": "What really happened here?",
            "supporting_claim_ids": ["C-1", "C-999"],
            "curiosity_score": 0.8,
            "specificity_score": 0.6,
            "emotional_tension_score": 0.5,
            "information_gap_score": 0.7,
            "relevance_score": 0.9,
            "truthfulness_score": 0.95,
            "clickbait_risk": 0.1,
        },
        {
            "hook_type": "shocking_fact",
            "text": "Almost nobody expected this outcome.",
            "supporting_claim_ids": [],
            "curiosity_score": 0.9,
            "specificity_score": 0.5,
            "emotional_tension_score": 0.6,
            "information_gap_score": 0.8,
            "relevance_score": 0.7,
            "truthfulness_score": 0.9,
            "clickbait_risk": 0.3,
        },
    ]
}


@pytest.mark.asyncio
async def test_generate_hooks_maps_llm_output() -> None:
    claims = [Claim(claim_id="C-1", text="Claim one.", confidence=0.9)]
    fake_model = FakeChatModel(CANNED_HOOKS)

    hooks = await generate_hooks("Test Topic", claims, [], model=fake_model)

    assert len(hooks) == 2
    assert hooks[0].hook_type == HookType.QUESTION
    assert hooks[1].hook_type == HookType.SHOCKING_FACT
    assert hooks[0].text == "What really happened here?"
    assert 0.0 <= hooks[0].overall_score <= 1.0


@pytest.mark.asyncio
async def test_generate_hooks_drops_invented_claim_ids() -> None:
    claims = [Claim(claim_id="C-1", text="Claim one.", confidence=0.9)]
    fake_model = FakeChatModel(CANNED_HOOKS)

    hooks = await generate_hooks("Test Topic", claims, [], model=fake_model)

    # C-999 was never a real claim id - must be dropped, not trusted.
    assert hooks[0].supporting_claim_ids == ["C-1"]
    assert hooks[1].supporting_claim_ids == []


@pytest.mark.asyncio
async def test_generate_hooks_handles_empty_batch() -> None:
    fake_model = FakeChatModel({"hooks": []})
    hooks = await generate_hooks("Test Topic", [], [], model=fake_model)
    assert hooks == []
