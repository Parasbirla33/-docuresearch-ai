"""Tests for the script-template style analyzer."""

from __future__ import annotations

import pytest

from docuresearch.extraction.template_analyzer import analyze_script_template
from tests.conftest import FakeChatModel

CANNED_SPEC = {
    "hook_length": "short, 1-2 sentences",
    "intro_style": "direct address to the viewer",
    "section_pattern": "numbered chapters",
    "tone": "urgent",
    "language": "english",
    "transition_style": "abrupt cuts",
    "uses_questions": True,
    "uses_statistics": True,
    "uses_examples": False,
    "uses_storytelling": True,
    # A model trying to slip factual content into this field should never win -
    # the real code always overrides it with an excerpt of the actual template.
    "raw_template_excerpt": "The company earned $4 billion in 2019.",
}


@pytest.mark.asyncio
async def test_analyze_script_template_maps_llm_output() -> None:
    fake_model = FakeChatModel(CANNED_SPEC)
    spec = await analyze_script_template("HOOK: Did you know...\n\nCHAPTER 1: ...", model=fake_model)

    assert spec.tone == "urgent"
    assert spec.section_pattern == "numbered chapters"
    assert spec.uses_questions is True
    assert spec.uses_examples is False


@pytest.mark.asyncio
async def test_analyze_script_template_excerpt_is_the_real_template_not_llm_output() -> None:
    fake_model = FakeChatModel(CANNED_SPEC)
    real_template = "HOOK: Did you know this changed everything? CHAPTER 1: ..."

    spec = await analyze_script_template(real_template, model=fake_model)

    assert spec.raw_template_excerpt is not None
    assert "4 billion" not in spec.raw_template_excerpt
    assert spec.raw_template_excerpt.startswith("HOOK: Did you know")
