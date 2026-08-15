"""Script writer agent (spec section 30).

The LLM writes introduction/sections/conclusion around an already-finalized
hook, strictly grounded in the verified claims it's given - it never
regenerates the hook itself (that was already scored and grounded in
`hook_agent.py`). Every `cited_claim_ids` entry is validated against the
real claim set afterward, same as the story/hook agents.
"""

from __future__ import annotations

from datetime import UTC, datetime

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field

from docuresearch.llm.factory import get_chat_model
from docuresearch.models.claims import Claim
from docuresearch.models.research import StoryArchitecture
from docuresearch.models.script import DraftScript, ScriptSection, ScriptStyleSpec
from docuresearch.prompting.writing_prompts import script_generation_prompt
from docuresearch.utils.hashing import new_id
from docuresearch.utils.logging import get_logger

logger = get_logger(__name__)

MAX_CLAIMS_IN_PROMPT = 20


class _LLMScriptSection(BaseModel):
    heading: str
    narration: str
    cited_claim_ids: list[str] = Field(default_factory=list)


class _LLMScript(BaseModel):
    title_suggestions: list[str] = Field(default_factory=list)
    introduction: str = ""
    sections: list[_LLMScriptSection] = Field(default_factory=list)
    conclusion: str = ""
    cta: str | None = None


async def generate_script(
    topic: str,
    *,
    language: str,
    tone: str | None,
    script_length: str,
    target_audience: str,
    story_architecture: StoryArchitecture | None,
    claims: list[Claim],
    hook_text: str,
    style_spec: ScriptStyleSpec | None = None,
    feedback: list[str] | None = None,
    version: int = 1,
    model_name: str | None = None,
    model: BaseChatModel | None = None,
) -> DraftScript:
    """Write a documentary script grounded only in real, verified claim IDs.

    `feedback` (findings from a previous fact-check/citation-audit pass) lets
    the same function serve the revision loop - it just adds constraints to
    the same grounded generation, rather than being a separate code path.
    """
    known_ids = {c.claim_id for c in claims}
    ranked_claims = sorted(claims, key=lambda c: c.confidence, reverse=True)[:MAX_CLAIMS_IN_PROMPT]

    model = model or get_chat_model()
    structured = model.with_structured_output(_LLMScript)
    prompt = script_generation_prompt(
        topic=topic,
        language=language,
        tone=tone,
        script_length=script_length,
        target_audience=target_audience,
        story_architecture=story_architecture,
        claims=ranked_claims,
        hook_text=hook_text,
        style_spec=style_spec,
        feedback=feedback,
    )
    raw: _LLMScript = await structured.ainvoke(prompt)  # type: ignore[assignment]

    sections: list[ScriptSection] = []
    for i, s in enumerate(raw.sections):
        grounded_ids = [cid for cid in s.cited_claim_ids if cid in known_ids]
        dropped = len(s.cited_claim_ids) - len(grounded_ids)
        if dropped:
            logger.warning("script_section_dropped_invented_claim_ids", heading=s.heading, dropped=dropped)
        sections.append(
            ScriptSection(id=new_id("SS"), heading=s.heading, narration=s.narration, cited_claim_ids=grounded_ids, order=i)
        )

    full_text = "\n\n".join(
        part for part in [hook_text, raw.introduction, *(s.narration for s in sections), raw.conclusion, raw.cta or ""] if part
    )

    logger.info("generate_script", topic=topic, section_count=len(sections), version=version)

    return DraftScript(
        version=version,
        title_suggestions=raw.title_suggestions,
        hook_text=hook_text,
        introduction=raw.introduction,
        sections=sections,
        conclusion=raw.conclusion,
        cta=raw.cta,
        full_text=full_text,
        generated_at=datetime.now(UTC),
        model_name=model_name,
        prompt_version="v1",
    )
