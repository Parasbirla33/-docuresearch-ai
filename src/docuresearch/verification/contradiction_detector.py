"""Detects conflicts between different claims (spec section 14).

Verification (`claim_verifier.py`) checks one claim against its own sources
in isolation - it can't notice that a *different* claim asserts something
incompatible. This module compares claims against each other. A cheap
textual-similarity pre-filter keeps the LLM call count bounded rather than
checking every pair.
"""

from __future__ import annotations

from difflib import SequenceMatcher

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel

from docuresearch.llm.factory import get_fast_model
from docuresearch.models.claims import Claim, ClaimType
from docuresearch.prompting.verification_prompts import contradiction_check_prompt
from docuresearch.utils.logging import get_logger

logger = get_logger(__name__)

SIMILARITY_THRESHOLD = 0.35
MAX_CANDIDATE_PAIRS = 20
_COMPARABLE_TYPES = {ClaimType.FACTUAL, ClaimType.STATISTICAL, ClaimType.HISTORICAL_EVENT}


class ContradictionVerdict(BaseModel):
    conflicts: bool
    description: str | None = None


def find_candidate_pairs(claims: list[Claim]) -> list[tuple[Claim, Claim]]:
    """Cheap, non-LLM pre-filter: claims of a comparable type with similar wording.

    This is a textual heuristic, not semantic understanding - it exists only
    to avoid spending an LLM call on every possible pair.
    """
    comparable = [c for c in claims if c.claim_type in _COMPARABLE_TYPES]
    scored_pairs: list[tuple[float, Claim, Claim]] = []

    for i in range(len(comparable)):
        for j in range(i + 1, len(comparable)):
            a, b = comparable[i], comparable[j]
            similarity = SequenceMatcher(None, a.text.lower(), b.text.lower()).ratio()
            if similarity >= SIMILARITY_THRESHOLD:
                scored_pairs.append((similarity, a, b))

    scored_pairs.sort(key=lambda item: item[0], reverse=True)
    return [(a, b) for _, a, b in scored_pairs[:MAX_CANDIDATE_PAIRS]]


async def check_pair_for_contradiction(
    claim_a: Claim, claim_b: Claim, *, model: BaseChatModel | None = None
) -> ContradictionVerdict:
    """Ask the model whether two claims genuinely conflict."""
    model = model or get_fast_model()
    structured = model.with_structured_output(ContradictionVerdict)
    prompt = contradiction_check_prompt(claim_a=claim_a.text, claim_b=claim_b.text)
    verdict: ContradictionVerdict = await structured.ainvoke(prompt)  # type: ignore[assignment]
    return verdict
