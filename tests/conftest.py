"""Shared test fixtures/helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from docuresearch.config.settings import get_settings


@pytest.fixture(autouse=True)
def _isolated_database(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point Settings.database_url at a per-test sqlite file.

    Without this, `ResearchRunRepository()`'s default (`get_settings()`)
    would read/write the real `data/docuresearch.db` during every test run.
    """
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{(tmp_path / 'test.db').as_posix()}")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


class _FakeStructuredRunnable:
    def __init__(self, schema: Any, canned: dict[str, Any]) -> None:
        self._schema = schema
        self._canned = canned

    async def ainvoke(self, prompt: str) -> Any:
        return self._schema.model_validate(self._canned)


class FakeChatModel:
    """Stand-in for a LangChain chat model's `.with_structured_output(...)` interface.

    Lets agent/extraction/verification code be unit-tested without any real
    LLM provider or API key - `canned` is validated against whatever schema
    the code under test asks for, mirroring how structured output actually behaves.
    """

    def __init__(self, canned: dict[str, Any]) -> None:
        self._canned = canned
        self.last_schema: Any = None

    def with_structured_output(self, schema: Any) -> _FakeStructuredRunnable:
        self.last_schema = schema
        return _FakeStructuredRunnable(schema, self._canned)
