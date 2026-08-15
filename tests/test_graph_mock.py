"""End-to-end test of the compiled LangGraph pipeline in mock mode."""

from __future__ import annotations

import pytest

from docuresearch.graph.workflow import run_research
from docuresearch.models.script import FactCheckVerdict
from docuresearch.models.state import new_initial_state
from docuresearch.utils.hashing import new_id


@pytest.mark.asyncio
async def test_mock_pipeline_runs_end_to_end() -> None:
    initial_state = new_initial_state(
        research_id=new_id("RUN"),
        topic="Test Documentary Topic",
        mock_mode=True,
    )

    result = await run_research(initial_state)

    # Research
    assert len(result["sources"]) == 5
    assert len(result["claims"]) == 10
    assert len(result["verified_claims"]) > 0
    assert len(result["contradictions"]) == 2

    # Story / hooks
    assert result["story_outline"] is not None
    assert len(result["hook_options"]) >= 5
    assert result["selected_hook"] is not None

    # Script
    draft = result["draft_script"]
    assert draft is not None
    assert draft.hook_text
    assert len(draft.sections) > 0

    # Fact check / citations / quality
    fact_check = result["fact_check_results"]
    assert fact_check is not None
    assert fact_check.overall_verdict in (
        FactCheckVerdict.PASS,
        FactCheckVerdict.NEEDS_REVISION,
        FactCheckVerdict.FAIL,
    )
    assert result["citation_map"]
    assert result["quality_score"] is not None

    # Final output
    final = result["final_output"]
    assert final is not None
    assert final.topic == "Test Documentary Topic"
    assert final.script is not None
    assert final.quality_score is not None


@pytest.mark.asyncio
async def test_mock_pipeline_distributes_citations_across_sections() -> None:
    """Regression test: every section must not cite the exact same claim pair."""
    initial_state = new_initial_state(
        research_id=new_id("RUN"),
        topic="Regression Topic",
        mock_mode=True,
    )
    result = await run_research(initial_state)
    draft = result["draft_script"]
    cited_sets = [tuple(sec.cited_claim_ids) for sec in draft.sections]
    assert len(set(cited_sets)) > 1, "all sections cited the identical claim set"


@pytest.mark.asyncio
async def test_mock_pipeline_does_not_finalize_below_iteration_cap_forever() -> None:
    """The graph must terminate - quality control cannot loop past max_iterations."""
    initial_state = new_initial_state(
        research_id=new_id("RUN"),
        topic="Cap Test",
        mock_mode=True,
        max_iterations=2,
    )
    result = await run_research(initial_state)
    assert result["iteration_count"] <= 2
    assert result["final_output"] is not None
