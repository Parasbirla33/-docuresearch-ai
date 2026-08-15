"""Prompts for claim verification against source evidence."""

from __future__ import annotations

from docuresearch.models.claims import Claim


def claim_verification_prompt(*, claim_text: str, source_excerpts: list[str]) -> str:
    excerpts_block = "\n\n".join(f"[Source {i + 1}]\n{excerpt}" for i, excerpt in enumerate(source_excerpts))

    return f"""You are fact-checking a single claim against source excerpts. \
Do not use outside knowledge - judge only whether these excerpts support, \
partially support, contradict, or say nothing about the claim.

CLAIM:
{claim_text}

SOURCE EXCERPTS:
{excerpts_block or "(no source excerpts available)"}

Decide:
- verification_status: "verified" if the excerpts clearly and directly
  support the claim; "partially_verified" if they support part of it or
  support it with caveats; "disputed" if excerpts conflict with each other
  on this claim; "unverified" if the excerpts say nothing relevant;
  "false_or_unsupported" if the excerpts directly contradict the claim.
- confidence: 0.0-1.0, how confident you are in that status given only
  these excerpts.
- supports: true only if the excerpts, on balance, support the claim as stated.
- notes: one sentence explaining the verdict. If evidence is missing or thin,
  say so plainly rather than implying certainty.

Never mark a claim verified because it sounds plausible - only because the
excerpts actually say so."""


def contradiction_check_prompt(*, claim_a: str, claim_b: str) -> str:
    return f"""Two claims from a documentary's research were flagged as \
possibly discussing the same fact. Decide whether they actually conflict.

CLAIM A:
{claim_a}

CLAIM B:
{claim_b}

They conflict only if a reader could not accept both as true at once (e.g.
different dates/numbers/outcomes for the same event, or directly opposing
statements). They do NOT conflict if they are simply about different things,
compatible details of the same event, or one is a more specific version of
the other.

Decide:
- conflicts: true only if the claims genuinely contradict each other.
- description: if conflicts is true, one plain sentence describing the
  conflict (e.g. "Claim A dates the launch to 2016; Claim B dates it to
  2015."). Leave null if conflicts is false."""


def script_fact_check_prompt(*, narration: str, cited_claims: list[Claim]) -> str:
    claims_block = "\n".join(
        f"- [{c.claim_id}] ({c.verification_status.value}, confidence={c.confidence}) {c.text}"
        for c in cited_claims
    )
    return f"""You are an independent fact-checker reviewing documentary script \
narration against the claims it is supposed to be based on. You did not
write this script - check it skeptically, the way a second editor would.

SCRIPT NARRATION:
{narration}

CLAIMS THIS NARRATION IS SUPPOSED TO BE BASED ON:
{claims_block or "(none - this narration cites no claims)"}

For each claim ID above, decide:
- claim_id: the ID from the list above this finding is about.
- verdict: "pass" if the narration accurately and proportionately reflects
  this claim; "needs_revision" if it's technically supported but overstates
  certainty, drops an important caveat, or the claim itself is only
  partially_verified/disputed; "fail" if the narration asserts something
  this claim does not actually support.
- exaggeration_detected: true if the narration states something more
  dramatically or certainly than the claim supports.
- missing_context: true if the narration omits a caveat, dispute, or
  uncertainty that the claim's status carries.
- unsupported_inference: true if the narration draws a conclusion the claim
  doesn't directly support.
- notes: one sentence explaining the verdict.

Return exactly one finding per claim ID listed above - do not invent a
claim ID that wasn't listed, and do not skip any that were."""
