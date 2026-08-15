"""LLM provider abstraction (spec section 57).

Callers ask for a *role* (`get_chat_model`, `get_fast_model`), never a
provider - today that role is filled by OpenAI, but nothing outside this
module knows that. Swapping in Anthropic/Google/a local model later means
adding a branch here, not touching agents/extraction/verification code.
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import SecretStr

from docuresearch.config.settings import Settings, get_settings


class MissingLLMProviderError(RuntimeError):
    """Raised when a chat model is requested but no provider is configured."""


def get_chat_model(
    *,
    model: str | None = None,
    temperature: float | None = None,
    settings: Settings | None = None,
) -> BaseChatModel:
    """The primary chat model for structured, reasoning-heavy tasks (planning, verification)."""
    settings = settings or get_settings()
    if not settings.has_openai or not settings.openai_api_key:
        raise MissingLLMProviderError(
            "No LLM provider configured - set OPENAI_API_KEY (or run with --mock)."
        )

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model or settings.default_model,
        temperature=settings.default_temperature if temperature is None else temperature,
        api_key=SecretStr(settings.openai_api_key),
        timeout=settings.request_timeout,
    )


def get_fast_model(*, settings: Settings | None = None) -> BaseChatModel:
    """A cheaper/faster model for high-volume tasks (e.g. per-document claim extraction)."""
    settings = settings or get_settings()
    return get_chat_model(model="gpt-4o-mini", temperature=0.0, settings=settings)
