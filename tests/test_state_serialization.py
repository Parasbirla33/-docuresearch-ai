"""Round-trip tests for models/state.py's serialize_state/deserialize_state (Phase 7)."""

from __future__ import annotations

from docuresearch.mock.sample_data import build_mock_claims, build_mock_sources
from docuresearch.models.research import ResearchDepth
from docuresearch.models.state import deserialize_state, new_initial_state, serialize_state


def _populated_state() -> dict:
    topic = "The Indian Telecom Revolution"
    sources = build_mock_sources(topic)
    claims = build_mock_claims(topic, sources)
    state = new_initial_state(
        research_id="RUN-abc123",
        topic=topic,
        language="hinglish",
        research_depth=ResearchDepth.DEEP,
    )
    state["sources"] = sources
    state["claims"] = claims
    state["verified_claims"] = claims[:1]
    state["unverified_claims"] = claims[1:]
    state["warnings"] = ["no SEARCH_API_KEY configured"]
    return state


def test_serialize_state_produces_json_safe_dict() -> None:
    state = _populated_state()
    dumped = serialize_state(state)

    assert isinstance(dumped, dict)
    assert dumped["research_id"] == "RUN-abc123"
    # HttpUrl / datetime / enum fields must all be plain JSON-safe types.
    for source in dumped["sources"]:
        assert isinstance(source["url"], str) or source["url"] is None
        assert isinstance(source["source_type"], str)


def test_deserialize_state_round_trips_nested_models() -> None:
    state = _populated_state()
    restored = deserialize_state(serialize_state(state))

    assert restored["research_id"] == state["research_id"]
    assert restored["topic"] == state["topic"]
    assert restored["research_depth"] == state["research_depth"]
    assert restored["sources"] == state["sources"]
    assert restored["claims"] == state["claims"]
    assert restored["verified_claims"] == state["verified_claims"]
    assert restored["warnings"] == state["warnings"]


def test_round_trip_preserves_empty_optional_fields() -> None:
    state = new_initial_state(research_id="RUN-empty", topic="Empty Topic")
    restored = deserialize_state(serialize_state(state))

    assert restored["research_plan"] is None
    assert restored["draft_script"] is None
    assert restored["sources"] == []
    assert restored["claims"] == []
