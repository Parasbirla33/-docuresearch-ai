"""Hook generation agent (spec sections 26-27).

Hooks are generated and self-scored by the model, but grounding is never
trusted at face value: every `supporting_claim_ids` entry is checked against
the real, already-verified claim set afterward, so a hook can never lean on
an invented fact even if the model tries to reach for one.
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field

from docuresearch.llm.factory import get_chat_model
from docuresearch.models.claims import Claim, Contradiction
from docuresearch.models.research import Hook, HookType
from docuresearch.prompting.writing_prompts import hook_generation_prompt
from docuresearch.utils.hashing import new_id
from docuresearch.utils.logging import get_logger

logger = get_logger(__name__)

MAX_CLAIMS_IN_PROMPT = 15


class _LLMHook(BaseModel):
    hook_type: HookType
    text: str
    supporting_claim_ids: list[str] = Field(default_factory=list)
    curiosity_score: float = Field(ge=0.0, le=1.0)
    specificity_score: float = Field(ge=0.0, le=1.0)
    emotional_tension_score: float = Field(ge=0.0, le=1.0)
    information_gap_score: float = Field(ge=0.0, le=1.0)
    relevance_score: float = Field(ge=0.0, le=1.0)
    truthfulness_score: float = Field(ge=0.0, le=1.0)
    clickbait_risk: float = Field(ge=0.0, le=1.0)


class _LLMHookBatch(BaseModel):
    hooks: list[_LLMHook] = Field(default_factory=list)


async def generate_hooks(
    topic: str,
    claims: list[Claim],
    contradictions: list[Contradiction],
    *,
    model: BaseChatModel | None = None,
) -> list[Hook]:
    """Generate candidate hooks grounded only in real, verified claim IDs."""
    known_ids = {c.claim_id for c in claims}
    ranked_claims = sorted(claims, key=lambda c: c.confidence, reverse=True)[:MAX_CLAIMS_IN_PROMPT]

    model = model or get_chat_model()
    structured = model.with_structured_output(_LLMHookBatch)
    prompt = hook_generation_prompt(topic=topic, claims=ranked_claims, contradictions=contradictions)
    raw: _LLMHookBatch = await structured.ainvoke(prompt)  # type: ignore[assignment]

    hooks: list[Hook] = []
    for h in raw.hooks:
        grounded_ids = [cid for cid in h.supporting_claim_ids if cid in known_ids]
        dropped = len(h.supporting_claim_ids) - len(grounded_ids)
        if dropped:
            logger.warning("hook_dropped_invented_claim_ids", hook_type=h.hook_type.value, dropped=dropped)
        hooks.append(
            Hook(
                id=new_id("H"),
                hook_type=h.hook_type,
                text=h.text,
                supporting_claim_ids=grounded_ids,
                curiosity_score=h.curiosity_score,
                specificity_score=h.specificity_score,
                emotional_tension_score=h.emotional_tension_score,
                information_gap_score=h.information_gap_score,
                relevance_score=h.relevance_score,
                truthfulness_score=h.truthfulness_score,
                clickbait_risk=h.clickbait_risk,
            )
        )

    logger.info("generate_hooks", topic=topic, count=len(hooks))
    return hooks
