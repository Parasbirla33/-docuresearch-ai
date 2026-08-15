"""Prompts for story architecture, hook generation, and script template analysis.

Kept out of business logic (spec section 56). Story/hook prompts are always
given a fixed, numbered list of already-verified claims and instructed to
reference only those claim_ids - the agents that call these prompts then
validate the response against that same list, dropping anything invented.
"""

from __future__ import annotations

from docuresearch.models.claims import Claim, Contradiction
from docuresearch.models.research import StoryArchitecture
from docuresearch.models.script import ScriptStyleSpec


def _format_claims(claims: list[Claim]) -> str:
    if not claims:
        return "(no verified claims available)"
    return "\n".join(f"- [{c.claim_id}] ({c.importance.value}) {c.text}" for c in claims)


def _format_contradictions(contradictions: list[Contradiction]) -> str:
    if not contradictions:
        return "(none detected)"
    return "\n".join(f"- {c.description}" for c in contradictions)


def story_architecture_prompt(*, topic: str, claims: list[Claim], contradictions: list[Contradiction]) -> str:
    return f"""You are a documentary story architect. Convert this research into \
a narrative structure - do not just summarize the claims in order.

TOPIC:
{topic}

VERIFIED CLAIMS (reference these by their [ID] - never invent a claim or an ID):
{_format_claims(claims)}

KNOWN CONTRADICTIONS:
{_format_contradictions(contradictions)}

Design a narrative structure that fits what this research actually contains.
Do NOT force a fixed template (cold open/hook/context/... in that exact
order) if the material doesn't support it - the number and names of
sections should adapt to the topic. A simple, evidence-light topic might
need only 3-4 sections; a complex, contested one might need more.

Produce:
- central_question: the single question the documentary is really answering.
- theme: one sentence on the throughline connecting the sections.
- sections: an ordered list, each with:
  - name: a short section heading.
  - purpose: one sentence on what this section accomplishes narratively.
  - key_claim_ids: the claim IDs (from the list above) this section should
    draw on. Only use IDs that appear above - never invent one."""


def hook_generation_prompt(*, topic: str, claims: list[Claim], contradictions: list[Contradiction]) -> str:
    return f"""You are a documentary hook writer. Generate several distinct \
candidate opening hooks for a documentary - never invent a fact, statistic,
name, or event to make a hook punchier. Every hook must be defensible using
only the claims listed below.

TOPIC:
{topic}

VERIFIED CLAIMS (reference by [ID] - never invent a claim or an ID):
{_format_claims(claims)}

KNOWN CONTRADICTIONS:
{_format_contradictions(contradictions)}

Generate 5-8 hooks, drawing on a mix of these angles where the material
supports them: mystery, shocking-but-true fact, direct question,
contradiction/competing narratives, human story, historical framing,
money/power, "here's what most people don't know."

For each hook, provide:
- hook_type: one of mystery, shocking_fact, question, contradiction,
  human_story, historical, money_power, lesser_known.
- text: the hook itself (1-2 sentences, spoken narration style).
- supporting_claim_ids: the claim IDs (from the list above) that make this
  hook true. Only use IDs that appear above - never invent one. Empty is
  fine if the hook is a framing device rather than a specific fact.
- curiosity_score, specificity_score, emotional_tension_score,
  information_gap_score, relevance_score, truthfulness_score: each 0.0-1.0,
  your honest self-assessment.
- clickbait_risk: 0.0-1.0 - how much this hook oversells or implies more
  than the evidence supports. Be honest here; a high-curiosity hook that
  overclaims should score high on this, not low.

Never fabricate a shocking statistic just to raise curiosity_score - an
unsupported hook is worse than a modest, true one."""


def template_analysis_prompt(*, template_excerpt: str) -> str:
    return f"""You are analyzing a documentary script template to learn its \
STYLE and STRUCTURE - not its content. Do not extract, repeat, or summarize
any factual claims, names, dates, or statistics from the template; another
script will be written later using different, independently-verified facts.

TEMPLATE:
{template_excerpt}

Describe, in your own words, purely the writing style and structure:
- hook_length: roughly how long/short the opening hook is.
- intro_style: how the introduction is written.
- section_pattern: how sections/chapters are structured and named.
- tone: the overall tone (e.g. urgent, reflective, investigative, casual).
- language: the language/register used (e.g. "english", "hinglish").
- transition_style: how the writer moves between sections.
- sentence_style: typical sentence length/rhythm.
- paragraph_length: typical paragraph length.
- narration_style: first/second/third person, direct address, etc.
- cta_style: how (if at all) it closes or calls the viewer to action.
- uses_questions / uses_statistics / uses_examples / uses_storytelling:
  true/false, based on whether the template's style leans on these devices.
- citation_style: how sources/citations are indicated, if at all.

If a field cannot be determined from this excerpt, leave it null rather
than guessing."""


def _format_story_outline(architecture: StoryArchitecture | None) -> str:
    if architecture is None or not architecture.sections:
        return "(no story outline available - structure the script sensibly around the claims below)"
    lines = [f"Central question: {architecture.central_question}"]
    if architecture.theme:
        lines.append(f"Theme: {architecture.theme}")
    for s in architecture.sections:
        ids = ", ".join(s.key_claim_ids) or "none suggested"
        lines.append(f"- {s.name}: {s.purpose} (suggested claims: {ids})")
    return "\n".join(lines)


def _format_style_spec(style: ScriptStyleSpec | None) -> str:
    if style is None:
        return ""
    fields = {
        "hook_length": style.hook_length,
        "intro_style": style.intro_style,
        "section_pattern": style.section_pattern,
        "tone": style.tone,
        "transition_style": style.transition_style,
        "sentence_style": style.sentence_style,
        "paragraph_length": style.paragraph_length,
        "narration_style": style.narration_style,
        "cta_style": style.cta_style,
        "citation_style": style.citation_style,
    }
    lines = [f"- {k}: {v}" for k, v in fields.items() if v]
    flags = []
    if style.uses_questions:
        flags.append("uses direct questions")
    if style.uses_statistics:
        flags.append("leans on statistics")
    if style.uses_examples:
        flags.append("uses concrete examples")
    if style.uses_storytelling:
        flags.append("uses narrative storytelling")
    if flags:
        lines.append(f"- devices: {', '.join(flags)}")
    if not lines:
        return ""
    return "\n\nMATCH THIS STYLE (structure/tone only - it carries no facts of its own):\n" + "\n".join(lines)


def script_generation_prompt(
    *,
    topic: str,
    language: str,
    tone: str | None,
    script_length: str,
    target_audience: str,
    story_architecture: StoryArchitecture | None,
    claims: list[Claim],
    hook_text: str,
    style_spec: ScriptStyleSpec | None = None,
    feedback: list[str] | None = None,
) -> str:
    feedback_block = ""
    if feedback:
        feedback_block = "\n\nA PREVIOUS DRAFT HAD THESE ISSUES - FIX THEM THIS TIME:\n" + "\n".join(
            f"- {line}" for line in feedback
        )

    return f"""You are writing documentary narration. Never invent a name, date, \
statistic, quote, event, organization, or study that isn't in the claims
listed below - if a section's evidence is thin, write around it with
appropriately hedged language ("reports suggest...", "it remains unclear
whether...") rather than inventing specifics.

TOPIC:
{topic}
LANGUAGE: {language} (write entirely in this language/register - if
"hinglish", use natural conversational Hinglish, not a literal translation)
TONE: {tone or "neutral, evidence-driven"}
TARGET AUDIENCE: {target_audience}
TARGET LENGTH: approximately {script_length} of spoken narration (a guide,
not a strict requirement)

OPENING HOOK (already finalized - do not repeat it verbatim; write the
introduction to flow naturally from it):
{hook_text}

SUGGESTED STRUCTURE (follow this outline; you may adapt section count/order
if the evidence doesn't actually support a suggested section):
{_format_story_outline(story_architecture)}

VERIFIED CLAIMS AVAILABLE (reference only these by [ID] - never invent a
claim or an ID; every factual sentence should be traceable to one):
{_format_claims(claims)}{_format_style_spec(style_spec)}{feedback_block}

Do not write a hook - that part is already finished. Produce:
- title_suggestions: 3 candidate documentary titles.
- introduction: sets up the story after the hook.
- sections: an ordered list, each with heading, narration, and
  cited_claim_ids (the claim IDs this section's factual statements draw on).
- conclusion: closes out the narrative.
- cta: an optional one-line call to action, or null if none fits."""
