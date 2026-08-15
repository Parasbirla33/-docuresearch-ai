"""Small text utilities shared across extraction/prompting code."""

from __future__ import annotations


def word_count(text: str) -> int:
    return len(text.split())


def truncate(text: str, max_chars: int, *, suffix: str = "...") -> str:
    """Truncate text to at most `max_chars`, breaking on the nearest word boundary."""
    if len(text) <= max_chars:
        return text
    cut = text[: max_chars - len(suffix)]
    last_space = cut.rfind(" ")
    if last_space > 0:
        cut = cut[:last_space]
    return cut + suffix
