"""Tests for the quality-control conditional edge and iteration-cap guard."""

from __future__ import annotations

from docuresearch.graph.routers import route_after_quality_control
from docuresearch.models.script import QualityScore
from docuresearch.models.state import new_initial_state


def _state_with_score(score: QualityScore, iteration_count: int = 0, max_iterations: int = 3):
    state = new_initial_state(research_id="RUN-1", topic="Test", max_iterations=max_iterations)
    state["quality_score"] = score
    state["iteration_count"] = iteration_count
    return state


def test_routes_to_revision_when_factual_safety_low() -> None:
    score = QualityScore(factual_safety=20, citation_completeness=100)
    state = _state_with_score(score, iteration_count=0)
    assert route_after_quality_control(state) == "final_revision"


def test_routes_to_revision_when_citations_incomplete() -> None:
    score = QualityScore(factual_safety=100, citation_completeness=10)
    state = _state_with_score(score, iteration_count=0)
    assert route_after_quality_control(state) == "final_revision"


def test_routes_to_final_output_when_quality_is_high() -> None:
    score = QualityScore(factual_safety=95, citation_completeness=90)
    state = _state_with_score(score, iteration_count=0)
    assert route_after_quality_control(state) == "final_output"


def test_stops_looping_once_max_iterations_reached() -> None:
    """Even with poor quality, the loop must not run forever."""
    score = QualityScore(factual_safety=10, citation_completeness=10)
    state = _state_with_score(score, iteration_count=3, max_iterations=3)
    assert route_after_quality_control(state) == "final_output"


def test_missing_quality_score_finalizes_immediately() -> None:
    state = new_initial_state(research_id="RUN-2", topic="Test")
    assert route_after_quality_control(state) == "final_output"
