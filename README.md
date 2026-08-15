# DocuResearch AI

[![CI](https://github.com/Parasbirla33/-docuresearch-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/Parasbirla33/-docuresearch-ai/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![Status: Phase 7 / V1 complete](https://img.shields.io/badge/status-phase%207%20%2F%20V1%20complete-brightgreen)](#status)
[![License: Proprietary](https://img.shields.io/badge/license-proprietary-lightgrey)](#)

**Topics:** [langgraph](https://github.com/topics/langgraph) ·
[llm](https://github.com/topics/llm) ·
[ai-agent](https://github.com/topics/ai-agent) ·
[langchain](https://github.com/topics/langchain) ·
[openai](https://github.com/topics/openai) ·
[pydantic](https://github.com/topics/pydantic) ·
[python](https://github.com/topics/python) ·
[documentary](https://github.com/topics/documentary) ·
[script-generation](https://github.com/topics/script-generation) ·
[content-generation](https://github.com/topics/content-generation) ·
[sqlalchemy](https://github.com/topics/sqlalchemy)

A modular, LangGraph-based documentary research & script generation agent.
Given a topic, it runs a research pipeline (planning → source discovery →
extraction → claim verification → contradiction detection → evidence
matrix → story architecture → hook generation → script generation → fact
checking → citation audit → quality control) and produces a structured,
source-backed documentary script.

**Status:**
- **Phase 1 (done):** the full graph is wired and runs end-to-end in
  **mock mode** using clearly-labelled placeholder data — this proves out
  the architecture, state model, and control flow (including the
  quality-control revision loop) before real research tools are built.
- **Phase 2 (done):** standalone, independently-tested research tools —
  web search (Tavily), Wikipedia, webpage extraction, and a file-based
  research cache (`src/docuresearch/tools/`, `src/docuresearch/storage/cache.py`).
- **Phase 3 (done):** `--topic <real topic>` (without `--mock`) now runs a
  genuinely live pipeline: an OpenAI-backed research planner
  (`agents/research_agent.py`), real source discovery/collection via the
  Phase 2 tools, LLM-backed claim extraction (`extraction/claim_extractor.py`),
  and LLM-backed claim verification against actual source text
  (`verification/claim_verifier.py`). If a capability's provider isn't
  configured (no `OPENAI_API_KEY` / no `SEARCH_API_KEY`), that capability
  contributes nothing and logs a warning — it never falls back to the fake
  mock dataset, since mixing fabricated placeholder facts into real output
  would violate the project's core "never fabricate" rule.
- **Phase 4 (done):** evidence intelligence for real (non-mock) research -
  heuristic source scoring + non-destructive duplicate/syndication detection
  (`tools/source_ranker.py`), real cross-claim contradiction detection
  (`verification/contradiction_detector.py`, wired back onto the affected
  claims), and an adaptive research loop (`graph/nodes.py: expand_weak_claims`)
  that chases additional targeted evidence for important-but-under-supported
  claims before moving on, bounded by `MAX_RESEARCH_ITERATIONS`.
- **Phase 5 (done):** the story engine. `build_story_architecture` and
  `generate_hooks` are now LLM-backed in live mode (`agents/story_agent.py`,
  `agents/hook_agent.py`) - the narrative structure adapts to what the
  research actually contains instead of forcing a fixed template, and hooks
  are grounded in real verified claims rather than generic string templates.
  Both validate every claim-ID reference the model returns against the real
  claim set and drop anything invented; both fall back to the original fixed
  templates on LLM failure/empty output/no key, so the pipeline never breaks.
  `--template <file>` is now actually used: `analyze_script_template`
  (`extraction/template_analyzer.py`) extracts a Script Style Specification
  from it (tone, structure, pacing) for the script generator to follow -
  explicitly never copying the template's factual content.
- **Phase 6 (done) - the script engine, completing spec V1.**
  `generate_script` (`agents/script_agent.py`) writes real narration in the
  requested language/tone/style, strictly grounded in verified claims
  (claim-ID validated, same as Phases 5's agents) and built around the
  already-finalized hook rather than regenerating it. `fact_check_script`
  (`verification/script_fact_checker.py`) is a genuinely independent second
  pass: it compares the *script's wording* against the claims it cites,
  catching exaggeration/dropped caveats/unsupported inferences the writer
  introduced even when citing a real claim - distinct from Phase 3's
  claim-vs-source check. `citation_audit` now also flags citations that
  don't resolve to a real source and known contradictions the script never
  acknowledges. `final_revision` was fixed to actually regenerate the
  script with feedback about what failed in live mode, instead of
  re-running the fact-checker on an unchanged draft and reproducing the
  same failure until the iteration budget ran out. All four fall back to
  their original deterministic behavior on LLM failure/no key, and mock
  mode is untouched throughout.

This completes the spec's V1 scope (topic intake through fact-checked,
cited script output).
- **Phase 7 (done) - SQLite/SQLAlchemy persistence + resumable runs.**
  `storage/repository.py: ResearchRunRepository` checkpoints a run's full
  state (`models/state.py: serialize_state`/`deserialize_state`, a single
  `pydantic.TypeAdapter` over the `DocuResearchState` TypedDict - every field
  is already a Pydantic model/primitive/enum, so no hand-written per-field
  mapping is needed) at the one point in the graph where it's actually useful
  to pause: after the research phase (`build_evidence_matrix`), before the
  creative/script phase (`build_story_architecture` onward) begins. The graph
  itself is split accordingly (`graph/workflow.py: build_research_graph`/
  `build_script_graph`, sharing the same node-wiring helpers as the
  unmodified full-pipeline `build_graph`) - not LangGraph's built-in
  checkpointer, since node-level crash recovery isn't the valuable case here;
  re-doing expensive LLM/search-backed research after a trivial script tweak
  is. `--research-only` now runs the research phase, persists, and stops
  (printing a `Resume with: --resume <id> --script-only` hint);
  `--resume RESEARCH_ID [--script-only]` reloads that checkpoint in a later,
  separate process and (re-)runs just the story/hook/script/fact-check/
  quality phase against it; `--list-runs` shows persisted runs. Every run
  (including `--mock`) persists a checkpoint, so `--resume` is always
  discoverable without needing a live API key.

Ahead: LangSmith observability (Phase 8) and a FastAPI-ready service layer
(Phase 9) - both optional hardening beyond V1, per the spec's own phase plan.

## Requirements

- Python 3.11+ (developed against 3.12)
- Windows, Linux, or macOS

## 1. Create a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
```

**Linux / macOS:**

```bash
python3 -m venv .venv
```

## 2. Activate it

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**

```bash
source .venv/bin/activate
```

## 3. Install dependencies

```powershell
pip install -r requirements-dev.txt
pip install -e . --no-deps
```

(`requirements-dev.txt` includes everything in `requirements.txt` plus
pytest/mypy/ruff. The editable install lets `python -m docuresearch` and
`import docuresearch` resolve without manually managing `PYTHONPATH`.)

## 4. Create your `.env`

```powershell
Copy-Item .env.example .env
```

Then fill in whichever API keys you have (`OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`, etc.). **All keys are optional** — missing providers
are handled gracefully; `--mock` mode needs none of them at all.

## 5. Run tests

```powershell
pytest -v
```

Or, using the PowerShell dev helper:

```powershell
./scripts/dev.ps1 test
```

(Linux/macOS/`make` users: `make test`.)

Also runnable: `ruff check src tests` (lint) and `mypy src` (types).

## 6. Run the application

```powershell
python -m docuresearch --mock
```

## 7. Running a sample research job

Mock mode (no API keys needed, deterministic placeholder data):

```powershell
python -m docuresearch --mock --output output\sample_run.json
```

Real topic (requires `OPENAI_API_KEY` in `.env` for research planning/claim
extraction/verification, and `SEARCH_API_KEY` for web search - Wikipedia
needs no key. Any capability whose provider is missing just contributes
nothing and logs a warning, rather than failing the run):

```powershell
python -m docuresearch --topic "The Indian Telecom Revolution" --language hinglish --depth deep --length 20min
```

Full CLI flags:

```
--topic TEXT            Documentary topic (required unless --mock)
--language TEXT         Script language (default: english)
--depth {quick,standard,deep,investigative}
--length TEXT           Target script length, e.g. "10min", "20min"
--audience TEXT         Target audience
--template PATH         Path to a user-provided script template file
--tone TEXT             Optional tone/style instructions
--date-range TEXT       Optional date range filter
--geo TEXT              Optional geographic focus
--mock                  Run with mock research data, no external APIs
--research-only         Run research only (through the evidence matrix), persist, then stop
--script-only           With --resume: (re-)run only the story/script phase from a checkpoint
--resume RESEARCH_ID    Resume a persisted research run by id (see --list-runs)
--list-runs             List persisted research runs and exit
--output PATH           Write the full structured result as JSON
--verbose               DEBUG-level logging
```

Two-phase example (research once, generate the script separately/later):

```powershell
python -m docuresearch --topic "The Indian Telecom Revolution" --research-only
python -m docuresearch --resume RUN-f54a9aad --script-only
python -m docuresearch --list-runs
```

## Project layout

```
src/docuresearch/
    config/settings.py       Environment-driven settings (pydantic-settings)
    models/                  Pydantic models: state, sources, claims, research, script
    graph/                   LangGraph nodes, routers, and compiled workflow
    mock/sample_data.py      Deterministic mock dataset (5 sources / 10 claims) for --mock
    llm/factory.py           ModelFactory: get_chat_model()/get_fast_model() (OpenAI today)
    agents/
        research_agent.py      LLM research planner -> ResearchPlan (structured output)
        story_agent.py           Adaptive story architecture, grounded in real claim IDs
        hook_agent.py             Grounded hook generation, self-scored, claim-ID validated
        script_agent.py           Script writer: language/tone/style-aware, grounded, feedback-capable
    tools/                   Pluggable source adapters
        web_search.py          Tavily-backed web search (SearchProvider protocol; no-op if no key)
        wikipedia.py            Wikipedia search/page-extract/reference-links tool
        webpage.py               Robust page fetch+extract: SSRF guard, robots.txt, size cap
        source_ranker.py         Heuristic scoring (spec sec. 10) + non-destructive dedup (sec. 16)
        (news.py, academic.py, archive.py land in later phases)
    extraction/
        claim_extractor.py     LLM claim extraction from one document -> list[Claim]
        template_analyzer.py     Script template -> ScriptStyleSpec (style only, never facts)
    verification/
        claim_verifier.py      LLM claim-vs-source-excerpt verification -> status/confidence
        contradiction_detector.py  Cross-claim conflict detection (similarity pre-filter + LLM verdict)
        script_fact_checker.py    Independent script-wording-vs-claim check (exaggeration/gaps/inference)
        (confidence.py tuning lands later)
    prompting/
        research_prompts.py    Prompts for plan generation + claim extraction
        verification_prompts.py Prompts for claim/contradiction/script-wording checks
        writing_prompts.py       Prompts for story architecture, hooks, template analysis, script writing
    storage/
        cache.py                File-based ResearchCache (URL-hash-keyed, TTL-based)
        models.py                 SQLAlchemy ORM (ResearchRunORM) + RunStatus enum
        repository.py             ResearchRunRepository: checkpoint/load/list persisted runs
    services/                 (Phase 9+) service layer for a future FastAPI surface
    utils/                    logging, text, hashing, retry helpers
    cli.py, main.py, __main__.py
tests/                        pytest suite (no live API keys required; HTTP + LLM calls mocked)
data/{cache,raw,processed}/   research cache and document storage (gitignored contents)
output/                       generated JSON results
config/                       non-Python config assets
templates/                    user-provided documentary script templates
```

`services/` is still scaffolded as an empty package (`__init__.py` only) -
an intentional extension point for the future FastAPI layer, not missing
work. Everything else listed above is real and tested.

## Design notes

- **State** (`models/state.py`) is a `TypedDict`, per LangGraph convention.
  Nodes never mutate it; they return partial-update dicts that LangGraph
  merges in. `errors`/`warnings` use an additive reducer so future
  parallel nodes can append safely; everything else uses replace semantics.
- **Mock mode** (`mock/sample_data.py`) always tags its data with
  `[MOCK]` / publisher `"Mock Source"` so it can never be confused with
  real, fact-checked research output.
- **Quality gate**: `quality_control` computes an 8-dimension weighted
  score; `route_after_quality_control` sends the run back through
  `final_revision → fact_check_script → citation_audit → quality_control`
  when factual safety or citation completeness are too low, capped by
  `max_iterations` so the graph always terminates.
- **Webpage extraction safety** (`tools/webpage.py`): every fetch is
  scheme-restricted to http/https, DNS-resolved and checked against
  private/loopback/link-local/reserved IP ranges before any request is
  made (SSRF guard), robots.txt-checked, size-capped via streaming, and
  timeout-bounded. 401/403/404/429/timeouts are classified into
  `SourceAvailability` rather than retried-around or faked - if content
  can't be legitimately accessed, that's the result.
- **Mock vs. live fallback policy** (`graph/nodes.py`): `mock_mode=True`
  never makes a network/LLM call. `mock_mode=False` uses real tools/LLM
  calls; if a specific provider is unavailable, that capability
  contributes nothing (empty list + logged warning) rather than silently
  substituting the `[MOCK]` placeholder dataset - mixing fabricated facts
  into what's presented as real research would violate the project's
  core "never fabricate" rule. The one exception is research-plan
  *questions* (not facts): without `OPENAI_API_KEY`, `create_research_plan`
  falls back to a generic (real, non-fabricated) question template rather
  than producing no plan at all.
- **Structured LLM output**: every LLM call uses
  `model.with_structured_output(PydanticSchema)`, so responses are
  validated against a Pydantic model rather than parsed as free text
  (spec section 56). Agent/extraction/verification functions take an
  optional `model` parameter for dependency injection, which is how the
  test suite exercises them without any real API key (`tests/conftest.py`
  provides a `FakeChatModel`).
- **Non-destructive deduplication** (`tools/source_ranker.py`): near-duplicate
  or syndicated sources are never removed from the source list - a claim may
  already cite one by ID (spec section 7's pipeline runs `rank_sources` after
  `extract_claims`). Instead, only the best copy in a duplicate cluster keeps
  its corroboration credit; the rest are scored down and tagged
  `duplicate_content` - "20 sites copying one press release" doesn't read as
  20 independent sources (spec section 16).
- **Contradiction detection** (`verification/contradiction_detector.py`) is
  cross-*claim* (do two different claims disagree?), distinct from
  `claim_verifier.py`'s per-claim check against its own sources. A cheap
  text-similarity pre-filter keeps the LLM-call count bounded. Confirmed
  contradictions are written back onto both claims (`verification_status`,
  `contradicting_evidence`), and `verified_claims`/`unverified_claims` are
  recomputed afterward so a claim that turns out disputed never lingers in
  the "verified" list.
- **Adaptive research loop** (`graph/nodes.py: expand_weak_claims`, spec
  section 52): after verification, `route_after_verify_claims` checks
  whether any HIGH/CRITICAL-importance claim still lacks real support
  (unverified, or verified with confidence below threshold - disputed claims
  are excluded, since conflicting evidence is a different problem from too
  little of it). If so, it loops through a bounded pass of targeted queries
  ("`<claim> evidence`", "`<claim> official data`") → fetch → re-verify for
  just those claims, capped by `MAX_RESEARCH_ITERATIONS` so it always
  terminates. This is a separate counter/budget from the
  quality-control/script-revision loop.
- **Grounded creative generation** (`agents/story_agent.py`,
  `agents/hook_agent.py`, `agents/script_agent.py`, spec sections 25-27, 30):
  the LLM is free to be creative about *structure* (how many sections, which
  hook angles, how narration is phrased) but never about *facts* - all three
  prompts list only the already-verified claims by `[claim_id]` and instruct
  the model to reference only those IDs. The agent code then filters every
  returned ID against the real claim set and silently drops anything
  invented, rather than trusting the model's word for it. The script writer
  also never regenerates the hook - it receives the already-scored, already-
  grounded hook from `hook_agent.py` as fixed context and writes around it.
  If any of these LLM calls fails, returns nothing usable, or no
  `OPENAI_API_KEY` is configured, the corresponding node falls back to its
  original deterministic template - live mode degrades to something
  reasonable, never to an error or to fabricated content.
- **Independent script fact-checking** (`verification/script_fact_checker.py`,
  spec section 34): deliberately a *different* check than
  `claim_verifier.py`. That module checks a claim against its source
  excerpts once, before writing. This one checks the *script's wording*
  against the claims it cites, after writing - catching exaggeration,
  dropped caveats, or unsupported inferences the writer introduced even
  when citing a real, previously-verified claim. `citation_audit` adds a
  further, deterministic layer: it flags citations that don't resolve to a
  real source, and known contradictions the script never acknowledges
  anywhere (spec section 35).
- **A working revision loop** (`final_revision`, live mode): early testing
  surfaced the same failure mode twice in this project - a revision loop
  that re-checks an *unchanged* draft just reproduces the same failure and
  burns the whole iteration budget for nothing (this happened with the
  citation-completeness formula in Phase 1's mock pipeline too). So
  `final_revision` now actually regenerates the script, passing the
  previous fact-check/citation failures back into `generate_script` as
  explicit feedback ("fix these issues") - the same grounded-generation
  function Phase 6 uses everywhere else, just with revision notes attached.
- **CLI warnings**: every accumulated `state["warnings"]` entry (missing
  providers, LLM failures, empty results, etc.) is printed under a
  `--- WARNINGS ---` section in the CLI report, in addition to whatever a
  given node also logs via structlog - so degraded-but-completed runs are
  never silently indistinguishable from fully-succeeded ones.
