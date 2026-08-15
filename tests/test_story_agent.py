"""Tests for the LLM-backed adaptive story architecture agent."""

from __future__ import annotations

import pytest

from docuresearch.agents.story_agent import generate_story_architecture
from docuresearch.models.claims import Claim
from tests.conftest import FakeChatModel

CANNED_ARCHITECTURE = {
    "central_question": "What really drove the change, and who paid the price?",
    "theme": "Progress that arrived unevenly.",
    "sections": [
        {"name": "Origins", "purpose": "Set the scene before anything changed.", "key_claim_ids": ["C-1", "C-999"]},
        {"name": "Turning Point", "purpose": "Show the shift as it happened.", "key_claim_ids": ["C-2"]},
    ],
}


@pytest.mark.asyncio
async def test_generate_story_architecture_maps_llm_output() -> None:
    claims = [
        Claim(claim_id="C-1", text="Claim one.", confidence=0.9),
        Claim(claim_id="C-2", text="Claim two.", confidence=0.8),
    ]
    fake_model = FakeChatModel(CANNED_ARCHITECTURE)

    architecture = await generate_story_architecture("Test Topic", claims, [], model=fake_model)

    assert architecture.central_question == "What really drove the change, and who paid the price?"
    assert architecture.theme == "Progress that arrived unevenly."
    assert len(architecture.sections) == 2
    assert architecture.sections[0].name == "Origins"
    assert architecture.sections[0].order == 0
    assert architecture.sections[1].order == 1


@pytest.mark.asyncio
async def test_generate_story_architecture_drops_invented_claim_ids() -> None:
    claims = [Claim(claim_id="C-1", text="Claim one.", confidence=0.9), Claim(claim_id="C-2", text="Claim two.")]
    fake_model = FakeChatModel(CANNED_ARCHITECTURE)

    architecture = await generate_story_architecture("Test Topic", claims, [], model=fake_model)

    # C-999 was never a real claim id - must be dropped, not trusted.
    assert architecture.sections[0].key_claim_ids == ["C-1"]
    assert architecture.sections[1].key_claim_ids == ["C-2"]


@pytest.mark.asyncio
async def test_generate_story_architecture_handles_no_sections() -> None:
    fake_model = FakeChatModel({"central_question": "What happened?", "sections": []})
    architecture = await generate_story_architecture("Test Topic", [], [], model=fake_model)
    assert architecture.sections == []
