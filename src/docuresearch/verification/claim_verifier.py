"""Verifies a claim against the source excerpts that supposedly back it.

Judges only whether the given excerpts support the claim - never uses the
model's outside/parametric knowledge, so a claim with no real source backing
gets marked unverified rather than quietly passed on the model's say-so.
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field

from docuresearch.llm.factory import get_fast_model
from docuresearch.models.claims import Claim, VerificationStatus
from docuresearch.prompting.verification_prompts import claim_verification_prompt
from docuresearch.utils.logging import get_logger

logger = get_logger(__name__)


class ClaimVerificationOutcome(BaseModel):
    verification_status: VerificationStatus
    confidence: float = Field(ge=0.0, le=1.0)
    supports: bool
    notes: str | None = None


async def verify_claim(
    claim: Claim,
    source_excerpts: list[str],
    *,
    model: BaseChatModel | None = None,
) -> ClaimVerificationOutcome:
    """Check `claim` against `source_excerpts` and return a verdict.

    An empty `source_excerpts` list always yields UNVERIFIED with confidence
    0.0 - no LLM call is made, since there is nothing to check against.
    """
    if not source_excerpts:
        return ClaimVerificationOutcome(
            verification_status=VerificationStatus.UNVERIFIED,
            confidence=0.0,
            supports=False,
            notes="No source excerpts available to check this claim against.",
        )

    model = model or get_fast_model()
    structured = model.with_structured_output(ClaimVerificationOutcome)

    prompt = claim_verification_prompt(claim_text=claim.text, source_excerpts=source_excerpts)
    outcome: ClaimVerificationOutcome = await structured.ainvoke(prompt)  # type: ignore[assignment]

    logger.info(
        "verify_claim",
        claim_id=claim.claim_id,
        status=outcome.verification_status.value,
        confidence=outcome.confidence,
    )
    return outcome
