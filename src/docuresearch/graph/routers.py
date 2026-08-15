"""Conditional-edge routing functions for the LangGraph workflow."""

from __future__ import annotations

from typing import Literal

from docuresearch.config.settings import get_settings
from docuresearch.graph.nodes import (
    QUALITY_CITATION_THRESHOLD,
    QUALITY_FACTUAL_SAFETY_THRESHOLD,
    is_weak_important_claim,
)
from docuresearch.models.state import DocuResearchState
from docuresearch.utils.logging import get_logger

logger = get_logger(__name__)


def route_after_quality_control(state: DocuResearchState) -> Literal["final_revision", "final_output"]:
    """Decide whether the script needs another revision pass or is ready to finalize.

    Never finalizes below the factual-safety threshold unless the iteration
    budget is exhausted, in which case we finalize anyway (with warnings) to
    avoid an infinite loop - the quality score in the output makes the gap visible.
    """
    score = state.get("quality_score")
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 3)

    if score is None:
        return "final_output"

    needs_revision = (
        score.factual_safety < QUALITY_FACTUAL_SAFETY_THRESHOLD
        or score.citation_completeness < QUALITY_CITATION_THRESHOLD
    )

    if needs_revision and iteration_count < max_iterations:
        logger.info(
            "route_after_quality_control",
            decision="final_revision",
            factual_safety=score.factual_safety,
            citation_completeness=score.citation_completeness,
            iteration_count=iteration_count,
        )
        return "final_revision"

    if needs_revision:
        logger.warning(
            "route_after_quality_control",
            decision="final_output_despite_low_quality",
            reason="max_iterations_reached",
            factual_safety=score.factual_safety,
        )
    return "final_output"


def route_after_verify_claims(
    state: DocuResearchState,
) -> Literal["expand_weak_claims", "detect_contradictions"]:
    """Adaptive research loop gate (spec section 52).

    If important claims still lack sufficient evidence, chase more of it
    (`expand_weak_claims`) rather than moving straight to contradiction
    detection/story building on thin evidence - bounded by
    MAX_RESEARCH_ITERATIONS so this always terminates. Mock mode and runs
    without an LLM provider never loop here.
    """
    if state.get("mock_mode"):
        return "detect_contradictions"

    settings = get_settings()
    if not settings.has_openai:
        return "detect_contradictions"

    iteration = state.get("research_iteration_count", 0)
    if iteration >= settings.max_research_iterations:
        return "detect_contradictions"

    weak = [c for c in state.get("claims", []) if is_weak_important_claim(c)]
    if not weak:
        return "detect_contradictions"

    logger.info(
        "route_after_verify_claims",
        decision="expand_weak_claims",
        weak_claim_count=len(weak),
        iteration=iteration,
    )
    return "expand_weak_claims"
