"""Story architect agent: turns verified claims into an adaptive narrative structure.

Never forces a fixed template (spec section 25) - the LLM proposes however
many sections the material actually supports. Every `key_claim_ids` entry is
validated against the real claim set afterward; anything invented is dropped,
never trusted at face value.
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field

from docuresearch.llm.factory import get_chat_model
from docuresearch.models.claims import Claim, Contradiction
from docuresearch.models.research import StoryArchitecture, StorySection
from docuresearch.prompting.writing_prompts import story_architecture_prompt
from docuresearch.utils.hashing import new_id
from docuresearch.utils.logging import get_logger

logger = get_logger(__name__)

MAX_CLAIMS_IN_PROMPT = 15


class _LLMStorySection(BaseModel):
    name: str
    purpose: str
    key_claim_ids: list[str] = Field(default_factory=list)


class _LLMStoryArchitecture(BaseModel):
    central_question: str
    theme: str | None = None
    sections: list[_LLMStorySection] = Field(default_factory=list)


async def generate_story_architecture(
    topic: str,
    claims: list[Claim],
    contradictions: list[Contradiction],
    *,
    model: BaseChatModel | None = None,
) -> StoryArchitecture:
    """Generate an adaptive narrative structure grounded only in real claim IDs.

    `claims` should already be the verified subset - this function does not
    filter by verification status itself.
    """
    known_ids = {c.claim_id for c in claims}
    ranked_claims = sorted(claims, key=lambda c: c.confidence, reverse=True)[:MAX_CLAIMS_IN_PROMPT]

    model = model or get_chat_model()
    structured = model.with_structured_output(_LLMStoryArchitecture)
    prompt = story_architecture_prompt(topic=topic, claims=ranked_claims, contradictions=contradictions)
    raw: _LLMStoryArchitecture = await structured.ainvoke(prompt)  # type: ignore[assignment]

    sections: list[StorySection] = []
    for i, section in enumerate(raw.sections):
        grounded_ids = [cid for cid in section.key_claim_ids if cid in known_ids]
        dropped = len(section.key_claim_ids) - len(grounded_ids)
        if dropped:
            logger.warning("story_section_dropped_invented_claim_ids", section=section.name, dropped=dropped)
        sections.append(
            StorySection(
                id=new_id("SEC"),
                name=section.name,
                purpose=section.purpose,
                key_claim_ids=grounded_ids,
                order=i,
            )
        )

    logger.info("generate_story_architecture", topic=topic, section_count=len(sections))
    return StoryArchitecture(central_question=raw.central_question, sections=sections, theme=raw.theme)
