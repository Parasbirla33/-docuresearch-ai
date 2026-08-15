"""Script template analysis, generation, fact-checking and final-output models."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from docuresearch.models.claims import Contradiction
from docuresearch.models.research import Hook, StoryArchitecture, VisualSuggestion


class ScriptStyleSpec(BaseModel):
    """Extracted style/structure specification from a user-provided script template.

    Captures HOW the user writes, never WHAT they wrote - factual content from a
    template must never be copied into the generated script.
    """

    hook_length: str | None = None
    intro_style: str | None = None
    section_pattern: str | None = None
    tone: str | None = None
    language: str = "english"
    transition_style: str | None = None
    sentence_style: str | None = None
    paragraph_length: str | None = None
    narration_style: str | None = None
    cta_style: str | None = None
    uses_questions: bool = False
    uses_statistics: bool = False
    uses_examples: bool = False
    uses_storytelling: bool = False
    citation_style: str | None = None
    raw_template_excerpt: str | None = None


class ScriptSection(BaseModel):
    """One chapter/section of the generated script."""

    id: str
    heading: str
    narration: str
    cited_claim_ids: list[str] = Field(default_factory=list)
    order: int = 0


class DraftScript(BaseModel):
    """A generated (possibly revised) documentary script draft."""

    version: int = 1
    title_suggestions: list[str] = Field(default_factory=list)
    hook_text: str = ""
    introduction: str = ""
    sections: list[ScriptSection] = Field(default_factory=list)
    conclusion: str = ""
    cta: str | None = None
    full_text: str = ""
    generated_at: datetime | None = None
    model_name: str | None = None
    prompt_version: str | None = None


class FactCheckVerdict(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    NEEDS_REVISION = "needs_revision"


class FactCheckFinding(BaseModel):
    """Result of independently re-checking one factual statement in the script."""

    claim_id: str | None = None
    statement: str
    verdict: FactCheckVerdict
    exaggeration_detected: bool = False
    missing_context: bool = False
    unsupported_inference: bool = False
    notes: str | None = None


class FactCheckResult(BaseModel):
    """Aggregate result of the script fact-checking pass."""

    findings: list[FactCheckFinding] = Field(default_factory=list)
    overall_verdict: FactCheckVerdict = FactCheckVerdict.NEEDS_REVISION

    @property
    def fail_count(self) -> int:
        return sum(1 for f in self.findings if f.verdict == FactCheckVerdict.FAIL)


class CitationEntry(BaseModel):
    """A single claim-to-source citation entry, e.g. `[Source C-014]`."""

    claim_id: str
    source_id: str
    source_label: str
    url: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class QualityScore(BaseModel):
    """Weighted multi-dimensional quality score gating final output."""

    research_depth: float = Field(default=0.0, ge=0.0, le=100.0)
    source_quality: float = Field(default=0.0, ge=0.0, le=100.0)
    claim_verification: float = Field(default=0.0, ge=0.0, le=100.0)
    narrative_quality: float = Field(default=0.0, ge=0.0, le=100.0)
    hook_quality: float = Field(default=0.0, ge=0.0, le=100.0)
    script_structure: float = Field(default=0.0, ge=0.0, le=100.0)
    citation_completeness: float = Field(default=0.0, ge=0.0, le=100.0)
    factual_safety: float = Field(default=0.0, ge=0.0, le=100.0)

    _WEIGHTS: ClassVar[dict[str, float]] = {
        "research_depth": 0.15,
        "source_quality": 0.15,
        "claim_verification": 0.15,
        "narrative_quality": 0.10,
        "hook_quality": 0.05,
        "script_structure": 0.10,
        "citation_completeness": 0.15,
        "factual_safety": 0.15,
    }

    @property
    def overall(self) -> float:
        return round(
            sum(getattr(self, dim) * weight for dim, weight in self._WEIGHTS.items()), 2
        )


class FinalOutput(BaseModel):
    """The complete structured deliverable: everything the CLI/API returns."""

    research_id: str
    topic: str
    title_suggestions: list[str] = Field(default_factory=list)
    opening_hook: str = ""
    script: DraftScript | None = None
    story_architecture: StoryArchitecture | None = None
    source_references: list[dict[str, Any]] = Field(default_factory=list)
    claim_to_source_map: list[CitationEntry] = Field(default_factory=list)
    unverified_claims: list[str] = Field(default_factory=list)
    contradictory_claims: list[Contradiction] = Field(default_factory=list)
    research_notes: list[str] = Field(default_factory=list)
    visual_suggestions: list[VisualSuggestion] = Field(default_factory=list)
    hook_options: list[Hook] = Field(default_factory=list)
    fact_check: FactCheckResult | None = None
    quality_score: QualityScore | None = None
    generated_at: datetime | None = None
