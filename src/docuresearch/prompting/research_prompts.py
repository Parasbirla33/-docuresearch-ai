"""Prompts for research planning and claim extraction.

Kept out of business logic (spec section 56) - agents/extraction modules
import these functions rather than inlining prompt text.
"""

from __future__ import annotations


def research_plan_prompt(
    *,
    topic: str,
    depth: str,
    date_range: str | None,
    geographic_focus: str | None,
) -> str:
    constraints = []
    if date_range:
        constraints.append(f"Focus on the date range: {date_range}.")
    if geographic_focus:
        constraints.append(f"Focus on the geographic area: {geographic_focus}.")
    constraint_text = " ".join(constraints)

    return f"""You are a documentary research planner. Your job is to plan what \
needs to be investigated before anyone writes a single word of narration - \
not to write the documentary itself.

Topic: {topic}
Research depth: {depth}
{constraint_text}

Produce a research plan with:
- key_questions: the specific questions that must be answered to understand
  this topic accurately. Mark a question requires_primary_source=true when
  only a primary source (government data, court records, company filings,
  first-hand accounts) can settle it, and requires_multiple_sources=true
  when a single source - however good - would not be enough to state it as fact.
- relevant_entities: people, organizations, companies, or institutions
  central to the topic.
- relevant_dates: specific dates or date ranges that matter.
- known_controversies: any disputes, competing narratives, or contested
  claims associated with this topic that research must investigate from
  multiple sides rather than assume.
- historical_context_needed: one paragraph on what background a reader with
  no prior knowledge of this topic would need.

Do not answer the questions. Do not state any fact as established. This is a
plan for what to go verify, not a summary of what you already believe."""


def claim_extraction_prompt(*, topic: str, document_text: str, source_label: str) -> str:
    return f"""You are extracting atomic, checkable factual claims from a single \
research source, for a documentary about: {topic}

Source: {source_label}

Extract each distinct factual claim made in the text below as a separate
item. Rules:
- Each claim must be a single, self-contained, checkable statement.
- Do not merge multiple facts into one claim.
- Do not add any fact, number, name, or date that is not present in the text.
- Classify claim_type as one of: factual, statistical, quote, causal,
  interpretive, opinion, allegation, historical_event.
- Set importance (low/medium/high/critical) based on how central the claim
  is to understanding this topic.
- Set requires_multiple_sources=true for any claim a careful journalist would
  not publish on the strength of this one source alone.
- Set first_party=true only if this source is the entity the claim is about,
  speaking for itself (e.g. a company's own announcement about itself).
- If the text contains no extractable factual claims, return an empty list.

SOURCE TEXT:
{document_text}"""
