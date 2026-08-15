"""LangGraph graph state.

A TypedDict, not a Pydantic model - this is what LangGraph's StateGraph expects.
Nodes never mutate this in place; they return partial dicts that LangGraph merges
into the running state. `errors`/`warnings` use an additive reducer so nodes can
safely append without clobbering each other; every other field uses default
(replace) semantics. Content-bearing lists (sources, claims, ...) are replaced
wholesale by the node responsible for them - see graph/nodes.py.

Extension point: when parallel fan-out nodes (e.g. `parallel_source_discovery`)
are implemented, give any field they write concurrently its own merge reducer
instead of relying on replace semantics.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from pydantic import TypeAdapter

from docuresearch.models.claims import Claim, Contradiction, EvidenceMatrixEntry
from docuresearch.models.research import (
    Hook,
    ResearchDepth,
    ResearchPlan,
    StoryArchitecture,
    VisualSuggestion,
)
from docuresearch.models.script import (
    DraftScript,
    FactCheckResult,
    FinalOutput,
    QualityScore,
    ScriptStyleSpec,
)
from docuresearch.models.sources import Document, Source


class DocuResearchState(TypedDict, total=False):
    # --- Intake ---------------------------------------------------------
    research_id: str
    topic: str
    language: str
    target_audience: str
    script_length: str
    script_template_path: str | None
    script_template_spec: ScriptStyleSpec | None
    tone: str | None
    date_range: str | None
    geographic_focus: str | None
    research_depth: ResearchDepth
    mock_mode: bool

    # --- Research planning ------------------------------------------------
    research_plan: ResearchPlan | None
    research_queries: list[str]

    # --- Source collection / extraction ------------------------------------
    sources: list[Source]
    documents: list[Document]

    # --- Claims / verification ----------------------------------------------
    claims: list[Claim]
    verified_claims: list[Claim]
    unverified_claims: list[Claim]
    contradictions: list[Contradiction]
    evidence: list[EvidenceMatrixEntry]

    # --- Story / hook -----------------------------------------------------
    story_outline: StoryArchitecture | None
    hook_options: list[Hook]
    selected_hook: Hook | None

    # --- Script ------------------------------------------------------------
    draft_script: DraftScript | None
    fact_check_results: FactCheckResult | None
    citation_map: dict[str, str]
    visual_suggestions: list[VisualSuggestion]

    # --- Quality / control flow --------------------------------------------
    quality_score: QualityScore | None
    errors: Annotated[list[str], operator.add]
    warnings: Annotated[list[str], operator.add]
    iteration_count: int
    max_iterations: int
    # Separate counter/budget for the adaptive research loop (expand_weak_claims),
    # distinct from the script-quality revision loop above.
    research_iteration_count: int

    # --- Output --------------------------------------------------------------
    final_output: FinalOutput | None


def new_initial_state(
    *,
    research_id: str,
    topic: str,
    language: str = "english",
    target_audience: str = "general audience",
    script_length: str = "10min",
    script_template_path: str | None = None,
    tone: str | None = None,
    date_range: str | None = None,
    geographic_focus: str | None = None,
    research_depth: ResearchDepth = ResearchDepth.STANDARD,
    mock_mode: bool = False,
    max_iterations: int = 3,
) -> DocuResearchState:
    """Build a fresh state dict for a new research run."""
    return DocuResearchState(
        research_id=research_id,
        topic=topic,
        language=language,
        target_audience=target_audience,
        script_length=script_length,
        script_template_path=script_template_path,
        script_template_spec=None,
        tone=tone,
        date_range=date_range,
        geographic_focus=geographic_focus,
        research_depth=research_depth,
        mock_mode=mock_mode,
        research_plan=None,
        research_queries=[],
        sources=[],
        documents=[],
        claims=[],
        verified_claims=[],
        unverified_claims=[],
        contradictions=[],
        evidence=[],
        story_outline=None,
        hook_options=[],
        selected_hook=None,
        draft_script=None,
        fact_check_results=None,
        citation_map={},
        visual_suggestions=[],
        quality_score=None,
        errors=[],
        warnings=[],
        iteration_count=0,
        max_iterations=max_iterations,
        research_iteration_count=0,
        final_output=None,
    )


_STATE_ADAPTER: TypeAdapter[DocuResearchState] = TypeAdapter(DocuResearchState)


def serialize_state(state: DocuResearchState) -> dict[str, Any]:
    """Dump a state dict to a plain JSON-safe dict (Phase 7 persistence).

    Every field is already a Pydantic model, primitive, or StrEnum, so a
    single TypeAdapter over the TypedDict handles this without per-field
    mapping code.
    """
    return _STATE_ADAPTER.dump_python(state, mode="json")


def deserialize_state(data: dict[str, Any]) -> DocuResearchState:
    """Reconstruct a state dict from `serialize_state` output."""
    return _STATE_ADAPTER.validate_python(data)
