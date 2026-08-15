"""Tests for utils/logging.py's configure_stdio_encoding.

pytest's own stdout/stderr capture doesn't reproduce a real Windows console's
cp1252 UnicodeEncodeError, so these test the fix's actual logic (reconfigure
every stream that supports it, skip ones that don't) against fake streams
rather than trying to force a real OS-level encoding failure.
"""

from __future__ import annotations

import sys

import pytest

from docuresearch.utils.logging import configure_stdio_encoding


class _ReconfigurableStream:
    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    def reconfigure(self, **kwargs: str) -> None:
        self.calls.append(kwargs)


class _PlainStream:
    """Stands in for a stream without `.reconfigure()` (e.g. pytest's capsys)."""


def test_configure_stdio_encoding_reconfigures_streams_that_support_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_out, fake_err = _ReconfigurableStream(), _ReconfigurableStream()
    monkeypatch.setattr(sys, "stdout", fake_out)
    monkeypatch.setattr(sys, "stderr", fake_err)

    configure_stdio_encoding()

    assert fake_out.calls == [{"encoding": "utf-8", "errors": "replace"}]
    assert fake_err.calls == [{"encoding": "utf-8", "errors": "replace"}]


def test_configure_stdio_encoding_skips_streams_without_reconfigure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "stdout", _PlainStream())
    monkeypatch.setattr(sys, "stderr", _PlainStream())

    configure_stdio_encoding()  # must not raise AttributeError
