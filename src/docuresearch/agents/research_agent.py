"""Research planner agent: turns a topic into a structured research plan.

The LLM never answers its own questions here - it only plans what needs
verifying. See spec section 8.
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field

from docuresearch.llm.factory import get_chat_model
from docuresearch.models.research import ResearchDepth, ResearchPlan, ResearchQuestion
from docuresearch.prompting.research_prompts import research_plan_prompt
from docuresearch.utils.hashing import new_id
from docuresearch.utils.logging import get_logger

logger = get_logger(__name__)


class _LLMResearchQuestion(BaseModel):
    question: str
    rationale: str | None = None
    requires_primary_source: bool = False
    requires_multiple_sources: bool = False


class _LLMResearchPlan(BaseModel):
    """Narrower shape the LLM actually fills in; IDs/topic/depth are ours to set."""

    key_questions: list[_LLMResearchQuestion] = Field(default_factory=list)
    relevant_entities: list[str] = Field(default_factory=list)
    relevant_dates: list[str] = Field(default_factory=list)
    known_controversies: list[str] = Field(default_factory=list)
    historical_context_needed: str | None = None


async def generate_research_plan(
    topic: str,
    *,
    depth: ResearchDepth = ResearchDepth.STANDARD,
    date_range: str | None = None,
    geographic_focus: str | None = None,
    model: BaseChatModel | None = None,
) -> ResearchPlan:
    """Generate a dynamic research plan for `topic` using an LLM.

    `model` is injectable for testing; production callers should leave it
    unset and let `get_chat_model()` resolve the configured provider.
    """
    model = model or get_chat_model()
    structured = model.with_structured_output(_LLMResearchPlan)

    prompt = research_plan_prompt(
        topic=topic, depth=depth.value, date_range=date_range, geographic_focus=geographic_focus
    )
    raw: _LLMResearchPlan = await structured.ainvoke(prompt)  # type: ignore[assignment]

    logger.info("generate_research_plan", topic=topic, question_count=len(raw.key_questions))

    return ResearchPlan(
        topic=topic,
        depth=depth,
        key_questions=[
            ResearchQuestion(
                id=new_id("Q"),
                question=q.question,
                rationale=q.rationale,
                requires_primary_source=q.requires_primary_source,
                requires_multiple_sources=q.requires_multiple_sources,
            )
            for q in raw.key_questions
        ],
        relevant_entities=raw.relevant_entities,
        relevant_dates=raw.relevant_dates,
        known_controversies=raw.known_controversies,
        historical_context_needed=raw.historical_context_needed,
    )
