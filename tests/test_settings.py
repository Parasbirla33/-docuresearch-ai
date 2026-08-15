"""Tests for graceful handling of missing optional provider configuration."""

from __future__ import annotations

import pytest

from docuresearch.config.settings import Settings


@pytest.fixture(autouse=True)
def _clear_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "SEARCH_API_KEY", "NEWS_API_KEY"):
        monkeypatch.delenv(key, raising=False)


def test_settings_load_without_any_env_file_or_keys() -> None:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.has_openai is False
    assert settings.has_anthropic is False
    assert settings.has_any_llm_provider is False
    assert settings.has_search_provider is False
    assert settings.has_news_provider is False


def test_settings_have_sane_defaults() -> None:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.database_url.startswith("sqlite:///")
    assert settings.max_research_sources > 0
    assert settings.request_timeout > 0
    assert 0.0 <= settings.default_temperature <= 2.0


def test_settings_pick_up_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.has_openai is True
    assert settings.has_any_llm_provider is True
