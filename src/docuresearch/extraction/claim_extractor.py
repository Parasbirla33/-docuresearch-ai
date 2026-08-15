"""Extracts atomic, checkable claims from a single document's text.

The LLM never invents facts not present in the source text (spec section 30);
it only identifies and classifies claims that are already there. `claim_id`
and `source_ids` are assigned here, not by the model.
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field

from docuresearch.llm.factory import get_fast_model
from docuresearch.models.claims import Claim, ClaimImportance, ClaimType, VerificationStatus
from docuresearch.prompting.research_prompts import claim_extraction_prompt
from docuresearch.utils.hashing import new_id
from docuresearch.utils.logging import get_logger
from docuresearch.utils.text import truncate

logger = get_logger(__name__)

# Cost control (spec section 39): cap how much of one document we send per call
# rather than blindly forwarding an entire long article.
MAX_DOCUMENT_CHARS = 6000


class _LLMExtractedClaim(BaseModel):
    text: str
    claim_type: ClaimType = ClaimType.FACTUAL
    importance: ClaimImportance = ClaimImportance.MEDIUM
    requires_multiple_sources: bool = False
    first_party: bool = False


class _LLMClaimExtraction(BaseModel):
    claims: list[_LLMExtractedClaim] = Field(default_factory=list)


async def extract_claims(
    document_text: str,
    *,
    source_id: str,
    topic: str,
    source_label: str = "",
    model: BaseChatModel | None = None,
) -> list[Claim]:
    """Extract claims from `document_text`, attributing them to `source_id`.

    Returns claims with `verification_status=UNVERIFIED` and `confidence=0.0`
    - the verification agent decides those, this step only identifies claims.
    """
    if not document_text.strip():
        return []

    model = model or get_fast_model()
    structured = model.with_structured_output(_LLMClaimExtraction)

    prompt = claim_extraction_prompt(
        topic=topic,
        document_text=truncate(document_text, MAX_DOCUMENT_CHARS),
        source_label=source_label or source_id,
    )
    raw: _LLMClaimExtraction = await structured.ainvoke(prompt)  # type: ignore[assignment]

    logger.info("extract_claims_from_document", source_id=source_id, count=len(raw.claims))

    return [
        Claim(
            claim_id=new_id("C"),
            text=c.text,
            claim_type=c.claim_type,
            importance=c.importance,
            source_ids=[source_id],
            confidence=0.0,
            verification_status=VerificationStatus.UNVERIFIED,
            first_party=c.first_party,
            requires_multiple_sources=c.requires_multiple_sources,
        )
        for c in raw.claims
        if c.text.strip()
    ]
