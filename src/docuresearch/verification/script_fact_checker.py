"""Independent script fact-checking (spec section 34).

Distinct from `claim_verifier.py`: that module checks a claim against its
source excerpts once, before writing. This module checks the *script's
wording* against the claims it cites after writing - catching exaggeration,
dropped caveats, or unsupported inferences the writer introduced even when
citing a real, previously-verified claim.
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field

from docuresearch.llm.factory import get_fast_model
from docuresearch.models.claims import Claim
from docuresearch.models.script import FactCheckFinding, FactCheckVerdict
from docuresearch.prompting.verification_prompts import script_fact_check_prompt
from docuresearch.utils.logging import get_logger

logger = get_logger(__name__)


class _LLMFactCheckFinding(BaseModel):
    claim_id: str | None = None
    verdict: FactCheckVerdict
    exaggeration_detected: bool = False
    missing_context: bool = False
    unsupported_inference: bool = False
    notes: str | None = None


class _LLMSectionFactCheck(BaseModel):
    findings: list[_LLMFactCheckFinding] = Field(default_factory=list)


async def fact_check_section(
    narration: str,
    cited_claims: list[Claim],
    *,
    model: BaseChatModel | None = None,
) -> list[FactCheckFinding]:
    """Check one section's narration against the claims it cites."""
    if not cited_claims:
        return [
            FactCheckFinding(
                statement=narration,
                verdict=FactCheckVerdict.NEEDS_REVISION,
                unsupported_inference=True,
                notes="Section cites no claims.",
            )
        ]

    model = model or get_fast_model()
    structured = model.with_structured_output(_LLMSectionFactCheck)
    prompt = script_fact_check_prompt(narration=narration, cited_claims=cited_claims)
    raw: _LLMSectionFactCheck = await structured.ainvoke(prompt)  # type: ignore[assignment]

    known_ids = {c.claim_id for c in cited_claims}
    findings = [
        FactCheckFinding(
            claim_id=f.claim_id if f.claim_id in known_ids else None,
            statement=narration,
            verdict=f.verdict,
            exaggeration_detected=f.exaggeration_detected,
            missing_context=f.missing_context,
            unsupported_inference=f.unsupported_inference,
            notes=f.notes,
        )
        for f in raw.findings
    ]

    if not findings:
        logger.warning("script_fact_check_empty", claim_count=len(cited_claims))
        return [
            FactCheckFinding(
                statement=narration,
                verdict=FactCheckVerdict.NEEDS_REVISION,
                notes="Fact-check produced no findings for this section.",
            )
        ]
    return findings
