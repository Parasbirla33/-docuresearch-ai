"""Tests for the LLM-backed research planner agent, using a fake chat model."""

from __future__ import annotations

import pytest

from docuresearch.agents.research_agent import generate_research_plan
from docuresearch.models.research import ResearchDepth
from tests.conftest import FakeChatModel

CANNED_PLAN = {
    "key_questions": [
        {
            "question": "What existed before the change?",
            "rationale": "Establishes a baseline.",
            "requires_primary_source": True,
            "requires_multiple_sources": False,
        },
        {
            "question": "Who benefited most?",
            "rationale": None,
            "requires_primary_source": False,
            "requires_multiple_sources": True,
        },
    ],
    "relevant_entities": ["Acme Corp", "Regulator X"],
    "relevant_dates": ["2016"],
    "known_controversies": ["Disputed market entry date"],
    "historical_context_needed": "Some background paragraph.",
}


@pytest.mark.asyncio
async def test_generate_research_plan_maps_llm_output_to_research_plan() -> None:
    fake_model = FakeChatModel(CANNED_PLAN)
    plan = await generate_research_plan(
        "Test Topic",
        depth=ResearchDepth.DEEP,
        date_range="2010-2020",
        geographic_focus="India",
        model=fake_model,
    )

    assert plan.topic == "Test Topic"
    assert plan.depth == ResearchDepth.DEEP
    assert len(plan.key_questions) == 2
    assert plan.key_questions[0].question == "What existed before the change?"
    assert plan.key_questions[0].requires_primary_source is True
    assert plan.key_questions[0].id.startswith("Q-")
    assert plan.relevant_entities == ["Acme Corp", "Regulator X"]
    assert plan.known_controversies == ["Disputed market entry date"]
    assert plan.historical_context_needed == "Some background paragraph."


@pytest.mark.asyncio
async def test_generate_research_plan_handles_empty_llm_output() -> None:
    fake_model = FakeChatModel({"key_questions": []})
    plan = await generate_research_plan("Another Topic", model=fake_model)
    assert plan.key_questions == []
    assert plan.relevant_entities == []
