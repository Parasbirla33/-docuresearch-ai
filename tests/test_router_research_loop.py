"""Tests for the adaptive-research-loop conditional edge (route_after_verify_claims)."""

from __future__ import annotations

import pytest

from docuresearch.config.settings import Settings
from docuresearch.graph import routers
from docuresearch.models.claims import Claim, ClaimImportance, VerificationStatus
from docuresearch.models.state import new_initial_state


def _settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


def test_mock_mode_never_loops(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(routers, "get_settings", lambda: _settings(openai_api_key="sk-test"))
    state = new_initial_state(research_id="R", topic="Topic", mock_mode=True)
    state["claims"] = [
        Claim(claim_id="C-1", importance=ClaimImportance.CRITICAL, verification_status=VerificationStatus.UNVERIFIED, text="x")
    ]
    assert routers.route_after_verify_claims(state) == "detect_contradictions"


def test_no_openai_key_never_loops(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(routers, "get_settings", lambda: _settings(openai_api_key=None))
    state = new_initial_state(research_id="R", topic="Topic", mock_mode=False)
    state["claims"] = [
        Claim(claim_id="C-1", importance=ClaimImportance.CRITICAL, verification_status=VerificationStatus.UNVERIFIED, text="x")
    ]
    assert routers.route_after_verify_claims(state) == "detect_contradictions"


def test_loops_when_important_claim_is_weak(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(routers, "get_settings", lambda: _settings(openai_api_key="sk-test", max_research_iterations=2))
    state = new_initial_state(research_id="R", topic="Topic", mock_mode=False)
    state["research_iteration_count"] = 0
    state["claims"] = [
        Claim(claim_id="C-1", importance=ClaimImportance.HIGH, verification_status=VerificationStatus.UNVERIFIED, text="x")
    ]
    assert routers.route_after_verify_claims(state) == "expand_weak_claims"


def test_does_not_loop_when_no_weak_claims(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(routers, "get_settings", lambda: _settings(openai_api_key="sk-test"))
    state = new_initial_state(research_id="R", topic="Topic", mock_mode=False)
    state["claims"] = [
        Claim(claim_id="C-1", importance=ClaimImportance.HIGH, verification_status=VerificationStatus.VERIFIED, confidence=0.9, text="x")
    ]
    assert routers.route_after_verify_claims(state) == "detect_contradictions"


def test_stops_looping_once_research_iteration_cap_reached(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(routers, "get_settings", lambda: _settings(openai_api_key="sk-test", max_research_iterations=2))
    state = new_initial_state(research_id="R", topic="Topic", mock_mode=False)
    state["research_iteration_count"] = 2
    state["claims"] = [
        Claim(claim_id="C-1", importance=ClaimImportance.CRITICAL, verification_status=VerificationStatus.UNVERIFIED, text="x")
    ]
    assert routers.route_after_verify_claims(state) == "detect_contradictions"


def test_disputed_claims_do_not_trigger_the_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disputed = conflicting evidence, not insufficient evidence - a different problem."""
    monkeypatch.setattr(routers, "get_settings", lambda: _settings(openai_api_key="sk-test"))
    state = new_initial_state(research_id="R", topic="Topic", mock_mode=False)
    state["claims"] = [
        Claim(claim_id="C-1", importance=ClaimImportance.CRITICAL, verification_status=VerificationStatus.DISPUTED, confidence=0.5, text="x")
    ]
    assert routers.route_after_verify_claims(state) == "detect_contradictions"
