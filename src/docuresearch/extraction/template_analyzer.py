"""Analyzes a user-provided script template's STYLE, never its factual content.

Spec section 28: extract a "Script Style Specification" the script generator
uses to match structure/tone/pacing later. Factual content in the template
must never be copied into the generated script - only the style is used.
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from docuresearch.llm.factory import get_fast_model
from docuresearch.models.script import ScriptStyleSpec
from docuresearch.prompting.writing_prompts import template_analysis_prompt
from docuresearch.utils.logging import get_logger
from docuresearch.utils.text import truncate

logger = get_logger(__name__)

MAX_TEMPLATE_CHARS = 6000
EXCERPT_PREVIEW_CHARS = 500


async def analyze_script_template(
    template_text: str, *, model: BaseChatModel | None = None
) -> ScriptStyleSpec:
    """Extract a Script Style Specification from a template's structure/style only."""
    model = model or get_fast_model()
    structured = model.with_structured_output(ScriptStyleSpec)
    prompt = template_analysis_prompt(template_excerpt=truncate(template_text, MAX_TEMPLATE_CHARS))
    spec: ScriptStyleSpec = await structured.ainvoke(prompt)  # type: ignore[assignment]

    logger.info("analyze_script_template", tone=spec.tone, section_pattern=spec.section_pattern)
    return spec.model_copy(update={"raw_template_excerpt": truncate(template_text, EXCERPT_PREVIEW_CHARS)})
