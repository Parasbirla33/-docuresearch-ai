# Contributing

This is a personal, proprietary project (see [LICENSE](LICENSE)) - it isn't
currently open to external pull requests. Bug reports and suggestions are
still welcome via [Issues](https://github.com/Parasbirla33/-docuresearch-ai/issues).
If you're interested in contributing code, open an issue first to discuss it
before spending time on a PR.

The rest of this document is the dev workflow, for reference.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
pip install -e . --no-deps
Copy-Item .env.example .env
```

See the [README](README.md#requirements) for the full setup walkthrough and
Linux/macOS equivalents. All API keys are optional - `pytest` and `--mock`
need none of them.

## Before submitting a change

```powershell
pytest -v
ruff check src tests
mypy src
```

CI (`.github/workflows/ci.yml`) runs the same three checks on Python 3.11
and 3.12 for every push/PR to `master` - a change isn't done until all three
are clean on both versions.

## Conventions this codebase follows

- **Dependency injection for testability**: functions that call an LLM or
  read settings take an optional `model`/`settings` parameter, defaulting to
  the real singleton. Tests inject fakes (`tests/conftest.py`'s
  `FakeChatModel`) instead of mocking imports.
- **Structured LLM output only**: every LLM call uses
  `model.with_structured_output(PydanticSchema)` - never free-text parsing.
- **Never fabricate**: if a provider/capability isn't configured, the
  affected step contributes nothing and logs a warning. It never falls back
  to fake/mock data to paper over a missing key.
- **Non-destructive by default**: don't drop or silently mutate data that
  something else might already reference by ID (see `tools/source_ranker.py`'s
  deduplication - down-score and tag, never delete).
- **Graceful degradation over hard failure**: live-mode LLM/network failures
  fall back to a deterministic template rather than crashing the pipeline;
  the failure is recorded in `state["warnings"]`, not swallowed silently.

## Commit messages

Imperative summary line, blank line, then a body explaining *why* the change
was made (motivation, prior failure mode, tradeoff) rather than restating
the diff. See the existing git history for the house style.
