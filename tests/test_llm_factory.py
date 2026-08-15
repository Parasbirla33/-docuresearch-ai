"""Tests for the OpenAI-backed ModelFactory."""

from __future__ import annotations

import pytest

from docuresearch.config.settings import Settings
from docuresearch.llm.factory import MissingLLMProviderError, get_chat_model, get_fast_model


def test_get_chat_model_raises_without_api_key() -> None:
    settings = Settings(_env_file=None, openai_api_key=None)  # type: ignore[call-arg]
    with pytest.raises(MissingLLMProviderError):
        get_chat_model(settings=settings)


def test_get_chat_model_builds_with_configured_model_and_temperature() -> None:
    settings = Settings(  # type: ignore[call-arg]
        _env_file=None, openai_api_key="sk-test", default_model="gpt-4o-mini", default_temperature=0.3
    )
    model = get_chat_model(settings=settings)
    assert model.model_name == "gpt-4o-mini"
    assert model.temperature == 0.3


def test_get_chat_model_explicit_args_override_settings() -> None:
    settings = Settings(_env_file=None, openai_api_key="sk-test", default_model="gpt-4o")  # type: ignore[call-arg]
    model = get_chat_model(model="gpt-4.1-mini", temperature=0.0, settings=settings)
    assert model.model_name == "gpt-4.1-mini"
    assert model.temperature == 0.0


def test_get_fast_model_uses_mini_model_and_zero_temperature() -> None:
    settings = Settings(_env_file=None, openai_api_key="sk-test")  # type: ignore[call-arg]
    model = get_fast_model(settings=settings)
    assert "mini" in model.model_name.lower()
    assert model.temperature == 0.0
