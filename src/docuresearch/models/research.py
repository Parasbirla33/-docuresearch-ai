"""Research planning, story architecture, hooks and visual-suggestion models."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ResearchDepth(StrEnum):
    """Configurable research depth. Exact source counts live in config, not here."""

    QUICK = "quick"
    STANDARD = "standard"
    DEEP = "deep"
    INVESTIGATIVE = "investigative"


class ResearchQuestion(BaseModel):
    """A single question the research plan intends to answer."""

    id: str
    question: str
    rationale: str | None = None
    requires_primary_source: bool = False
    requires_multiple_sources: bool = False
    answered: bool = False


class ResearchPlan(BaseModel):
    """Dynamically generated plan describing what needs to be researched and why."""

    topic: str
    depth: ResearchDepth = ResearchDepth.STANDARD
    key_questions: list[ResearchQuestion] = Field(default_factory=list)
    relevant_entities: list[str] = Field(default_factory=list)
    relevant_dates: list[str] = Field(default_factory=list)
    known_controversies: list[str] = Field(default_factory=list)
    historical_context_needed: str | None = None
    notes: str | None = None


class HookType(StrEnum):
    MYSTERY = "mystery"
    SHOCKING_FACT = "shocking_fact"
    QUESTION = "question"
    CONTRADICTION = "contradiction"
    HUMAN_STORY = "human_story"
    HISTORICAL = "historical"
    MONEY_POWER = "money_power"
    LESSER_KNOWN = "lesser_known"


class Hook(BaseModel):
    """A candidate opening hook, scored across retention and safety dimensions."""

    id: str
    hook_type: HookType
    text: str
    supporting_claim_ids: list[str] = Field(default_factory=list)

    curiosity_score: float = Field(default=0.0, ge=0.0, le=1.0)
    specificity_score: float = Field(default=0.0, ge=0.0, le=1.0)
    emotional_tension_score: float = Field(default=0.0, ge=0.0, le=1.0)
    information_gap_score: float = Field(default=0.0, ge=0.0, le=1.0)
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    truthfulness_score: float = Field(default=0.0, ge=0.0, le=1.0)
    clickbait_risk: float = Field(default=0.0, ge=0.0, le=1.0)

    @property
    def overall_score(self) -> float:
        """Weighted score rewarding retention while penalizing clickbait risk."""
        positive = (
            self.curiosity_score
            + self.specificity_score
            + self.emotional_tension_score
            + self.information_gap_score
            + self.relevance_score
            + self.truthfulness_score
        ) / 6
        return round(positive * (1 - self.clickbait_risk), 4)


class StorySection(BaseModel):
    """One beat of the documentary narrative. Sections adapt per topic - not fixed."""

    id: str
    name: str
    purpose: str
    key_claim_ids: list[str] = Field(default_factory=list)
    order: int = 0
    notes: str | None = None


class StoryArchitecture(BaseModel):
    """The documentary's narrative structure, built from research rather than a template."""

    central_question: str
    sections: list[StorySection] = Field(default_factory=list)
    theme: str | None = None
    tone_notes: str | None = None


class VisualSuggestion(BaseModel):
    """Suggested visual assets for one script section. Search terms, not asserted footage."""

    section_id: str
    section_name: str
    visual_ideas: list[str] = Field(default_factory=list)
    broll_keywords: list[str] = Field(default_factory=list)
    archival_search_keywords: list[str] = Field(default_factory=list)
    map_ideas: list[str] = Field(default_factory=list)
    chart_ideas: list[str] = Field(default_factory=list)
    onscreen_text: list[str] = Field(default_factory=list)
    timeline_ideas: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
