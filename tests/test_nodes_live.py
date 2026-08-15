"""Tests for the non-mock ("live") branches of graph nodes.

All network/LLM-touching dependencies are monkeypatched - these tests never
hit a real API, but they do exercise the real wiring/control-flow that would
run against OpenAI/Tavily/Wikipedia/webpages in production.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from docuresearch.config.settings import Settings
from docuresearch.graph import nodes
from docuresearch.models.claims import (
    Claim,
    ClaimImportance,
    Contradiction,
    EvidenceMatrixEntry,
    VerificationStatus,
)
from docuresearch.models.research import (
    Hook,
    HookType,
    ResearchDepth,
    ResearchPlan,
    StoryArchitecture,
    StorySection,
)
from docuresearch.models.script import (
    DraftScript,
    FactCheckFinding,
    FactCheckResult,
    FactCheckVerdict,
    ScriptSection,
    ScriptStyleSpec,
)
from docuresearch.models.sources import Document, Source, SourceAvailability, SourceType
from docuresearch.models.state import new_initial_state
from docuresearch.tools.web_search import SearchHit
from docuresearch.tools.webpage import ExtractionResult
from docuresearch.tools.wikipedia import WikipediaPage
from docuresearch.verification.claim_verifier import ClaimVerificationOutcome


def _settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


# --- create_research_plan ----------------------------------------------------


@pytest.mark.asyncio
async def test_create_research_plan_live_without_openai_key_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nodes, "get_settings", lambda: _settings(openai_api_key=None))
    state = new_initial_state(research_id="R", topic="Topic", mock_mode=False)

    result = await nodes.create_research_plan(state)

    plan = result["research_plan"]
    assert plan.topic == "Topic"
    assert "OPENAI_API_KEY" in (plan.notes or "")
    assert len(plan.key_questions) == 5


@pytest.mark.asyncio
async def test_create_research_plan_live_uses_llm_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nodes, "get_settings", lambda: _settings(openai_api_key="sk-test"))
    canned_plan = ResearchPlan(topic="Topic", depth=ResearchDepth.STANDARD, key_questions=[])

    async def fake_generate(*args: object, **kwargs: object) -> ResearchPlan:
        return canned_plan

    monkeypatch.setattr(nodes, "generate_research_plan", fake_generate)
    state = new_initial_state(research_id="R", topic="Topic", mock_mode=False)

    result = await nodes.create_research_plan(state)
    assert result["research_plan"] is canned_plan


@pytest.mark.asyncio
async def test_create_research_plan_live_llm_failure_falls_back_with_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nodes, "get_settings", lambda: _settings(openai_api_key="sk-test"))

    async def failing_generate(*args: object, **kwargs: object) -> ResearchPlan:
        raise RuntimeError("boom")

    monkeypatch.setattr(nodes, "generate_research_plan", failing_generate)
    state = new_initial_state(research_id="R", topic="Topic", mock_mode=False)

    result = await nodes.create_research_plan(state)
    assert "LLM research planning failed" in result["warnings"][0]
    assert result["research_plan"].topic == "Topic"


# --- parallel_source_discovery -----------------------------------------------


class _FakeSearchProvider:
    async def search(self, query: str, *, max_results: int = 10) -> list[SearchHit]:
        return [SearchHit(title="A Hit", url="https://example.com/a", snippet="s", score=0.7)]


class _FakeWikiTool:
    async def search(self, query: str, *, max_results: int = 5) -> list[str]:
        return ["Some Title"]

    async def get_page(self, title: str) -> WikipediaPage:
        return WikipediaPage(
            title=title,
            page_id=1,
            extract="Extract text",
            url="https://en.wikipedia.org/wiki/Some_Title",
            references=["https://example.com/ref1"],
        )


class _FakeWikiToolFactory:
    @staticmethod
    def from_settings(settings: Settings) -> _FakeWikiTool:
        return _FakeWikiTool()


@pytest.mark.asyncio
async def test_parallel_source_discovery_live_combines_search_and_wikipedia(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nodes, "get_settings", lambda: _settings(search_api_key="k"))
    monkeypatch.setattr(nodes, "get_search_provider", lambda settings: _FakeSearchProvider())
    monkeypatch.setattr(nodes, "WikipediaTool", _FakeWikiToolFactory)

    state = new_initial_state(research_id="R", topic="Topic", mock_mode=False)
    state["research_queries"] = ["Topic query"]

    result = await nodes.parallel_source_discovery(state)
    sources = result["sources"]
    urls = {str(s.url) for s in sources}

    assert "https://example.com/a" in urls
    assert any(s.source_type == SourceType.WIKIPEDIA for s in sources)
    assert "https://example.com/ref1" in urls


@pytest.mark.asyncio
async def test_parallel_source_discovery_live_warns_when_nothing_found(monkeypatch: pytest.MonkeyPatch) -> None:
    class _EmptySearchProvider:
        async def search(self, query: str, *, max_results: int = 10) -> list[SearchHit]:
            return []

    class _EmptyWikiTool:
        async def search(self, query: str, *, max_results: int = 5) -> list[str]:
            return []

        async def get_page(self, title: str) -> None:
            return None

    class _EmptyWikiFactory:
        @staticmethod
        def from_settings(settings: Settings) -> _EmptyWikiTool:
            return _EmptyWikiTool()

    monkeypatch.setattr(nodes, "get_settings", lambda: _settings())
    monkeypatch.setattr(nodes, "get_search_provider", lambda settings: _EmptySearchProvider())
    monkeypatch.setattr(nodes, "WikipediaTool", _EmptyWikiFactory)

    state = new_initial_state(research_id="R", topic="Topic", mock_mode=False)
    result = await nodes.parallel_source_discovery(state)
    assert result["sources"] == []
    assert result["warnings"]


# --- collect_sources ----------------------------------------------------------


class _FakeExtractor:
    async def extract(self, url: str) -> ExtractionResult:
        if "fail" in url:
            return ExtractionResult(availability=SourceAvailability.UNAVAILABLE, error="404")
        return ExtractionResult(availability=SourceAvailability.AVAILABLE, title="Fetched Title", text="Fetched body text.")


class _FakeExtractorFactory:
    @staticmethod
    def from_settings(settings: Settings) -> _FakeExtractor:
        return _FakeExtractor()


@pytest.mark.asyncio
async def test_collect_sources_live_fetches_missing_text(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(nodes, "get_settings", lambda: _settings(cache_dir=str(tmp_path / "cache")))
    monkeypatch.setattr(nodes, "WebpageExtractor", _FakeExtractorFactory)

    state = new_initial_state(research_id="R", topic="Topic", mock_mode=False)
    state["sources"] = [
        Source(id="S-1", url="https://example.com/ok", title="OK", source_type=SourceType.WEB_SEARCH),
        Source(id="S-2", url="https://example.com/fail", title="Fail", source_type=SourceType.WEB_SEARCH),
    ]

    result = await nodes.collect_sources(state)
    sources_by_id = {s.id: s for s in result["sources"]}

    assert sources_by_id["S-1"].text == "Fetched body text."
    assert sources_by_id["S-1"].availability == SourceAvailability.AVAILABLE
    assert sources_by_id["S-2"].text is None
    assert sources_by_id["S-2"].availability == SourceAvailability.UNAVAILABLE
    assert result["warnings"]


# --- extract_claims -----------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_claims_live_no_openai_key_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nodes, "get_settings", lambda: _settings(openai_api_key=None))
    state = new_initial_state(research_id="R", topic="Topic", mock_mode=False)

    result = await nodes.extract_claims(state)
    assert result["claims"] == []
    assert result["warnings"]


@pytest.mark.asyncio
async def test_extract_claims_live_aggregates_per_document(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nodes, "get_settings", lambda: _settings(openai_api_key="sk-test"))

    async def fake_extract(document_text: str, *, source_id: str, topic: str, source_label: str = "", model: object = None) -> list[Claim]:
        return [Claim(claim_id=f"C-{source_id}", text=f"Claim from {source_id}")]

    monkeypatch.setattr(nodes, "extract_claims_from_document", fake_extract)

    state = new_initial_state(research_id="R", topic="Topic", mock_mode=False)
    state["documents"] = [
        Document(id="D-1", source_id="S-1", clean_text="Some text about the topic."),
        Document(id="D-2", source_id="S-2", clean_text="More text about the topic."),
    ]

    result = await nodes.extract_claims(state)
    assert len(result["claims"]) == 2
    assert {c.text for c in result["claims"]} == {"Claim from S-1", "Claim from S-2"}


# --- verify_claims --------------------------------------------------------------


@pytest.mark.asyncio
async def test_verify_claims_live_updates_status_from_verifier(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nodes, "get_settings", lambda: _settings(openai_api_key="sk-test"))

    async def fake_verify(claim: Claim, excerpts: list[str], *, model: object = None) -> ClaimVerificationOutcome:
        return ClaimVerificationOutcome(
            verification_status=VerificationStatus.VERIFIED, confidence=0.95, supports=True, notes="ok"
        )

    monkeypatch.setattr(nodes, "verify_claim_against_sources", fake_verify)

    state = new_initial_state(research_id="R", topic="Topic", mock_mode=False)
    state["claims"] = [Claim(claim_id="C-1", text="A claim.", source_ids=["S-1"])]
    state["sources"] = [Source(id="S-1", title="Src", source_type=SourceType.WEB_SEARCH, text="Supporting text.")]

    result = await nodes.verify_claims(state)
    assert result["claims"][0].verification_status == VerificationStatus.VERIFIED
    assert result["claims"][0].confidence == 0.95
    assert len(result["verified_claims"]) == 1
    assert len(result["unverified_claims"]) == 0


@pytest.mark.asyncio
async def test_verify_claims_live_no_openai_key_leaves_claims_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nodes, "get_settings", lambda: _settings(openai_api_key=None))
    state = new_initial_state(research_id="R", topic="Topic", mock_mode=False)
    original = Claim(claim_id="C-1", text="A claim.")
    state["claims"] = [original]

    result = await nodes.verify_claims(state)
    assert result["claims"][0].verification_status == VerificationStatus.UNVERIFIED
    assert result["unverified_claims"] == [original]


# --- rank_sources (live) ---------------------------------------------------------


def test_rank_sources_live_dedups_and_scores() -> None:
    state = new_initial_state(research_id="R", topic="Topic", mock_mode=False)
    state["sources"] = [
        Source(id="S-1", title="T", source_type=SourceType.NEWS, domain="a.com", text="Same content here."),
        Source(id="S-2", title="T", source_type=SourceType.NEWS, domain="b.com", text="Same content here."),
    ]
    result = nodes.rank_sources(state)
    sources = result["sources"]
    assert len(sources) == 2
    assert {s.metadata.get("corroboration_count") for s in sources} == {2}


# --- detect_contradictions (live) -------------------------------------------------


@pytest.mark.asyncio
async def test_detect_contradictions_live_marks_conflicting_claims(monkeypatch: pytest.MonkeyPatch) -> None:
    from docuresearch.verification.contradiction_detector import ContradictionVerdict

    monkeypatch.setattr(nodes, "get_settings", lambda: _settings(openai_api_key="sk-test"))

    async def fake_check(a: Claim, b: Claim, *, model: object = None) -> ContradictionVerdict:
        return ContradictionVerdict(conflicts=True, description="Dates disagree.")

    monkeypatch.setattr(nodes, "check_pair_for_contradiction", fake_check)

    state = new_initial_state(research_id="R", topic="Topic", mock_mode=False)
    state["claims"] = [
        Claim(
            claim_id="C-1",
            text="Launched in 2016.",
            verification_status=VerificationStatus.VERIFIED,
            confidence=0.9,
            source_ids=["S-1"],
        ),
        Claim(
            claim_id="C-2",
            text="Launched in 2015.",
            verification_status=VerificationStatus.VERIFIED,
            confidence=0.9,
            source_ids=["S-2"],
        ),
    ]

    result = await nodes.detect_contradictions(state)
    assert len(result["contradictions"]) == 1

    updated = {c.claim_id: c for c in result["claims"]}
    assert updated["C-1"].verification_status == VerificationStatus.DISPUTED
    assert updated["C-2"].verification_status == VerificationStatus.DISPUTED
    assert "S-2" in updated["C-1"].contradicting_evidence
    assert "S-1" in updated["C-2"].contradicting_evidence
    assert result["verified_claims"] == []  # both reclassified out of "verified"


@pytest.mark.asyncio
async def test_detect_contradictions_live_no_openai_key_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nodes, "get_settings", lambda: _settings(openai_api_key=None))
    state = new_initial_state(research_id="R", topic="Topic", mock_mode=False)
    state["claims"] = [Claim(claim_id="C-1", text="Launched in 2016.")]

    result = await nodes.detect_contradictions(state)
    assert result["contradictions"] == []


# --- expand_weak_claims (adaptive research loop) ----------------------------------


class _FakeSearchProviderForExpand:
    async def search(self, query: str, *, max_results: int = 10) -> list[SearchHit]:
        return [SearchHit(title="New Evidence", url="https://example.com/new-evidence", snippet="s", score=0.8)]


class _FakeExtractorForExpand:
    async def extract(self, url: str) -> ExtractionResult:
        return ExtractionResult(
            availability=SourceAvailability.AVAILABLE, title="New Evidence", text="Confirms the claim clearly."
        )


class _FakeExtractorFactoryForExpand:
    @staticmethod
    def from_settings(settings: Settings) -> _FakeExtractorForExpand:
        return _FakeExtractorForExpand()


@pytest.mark.asyncio
async def test_expand_weak_claims_chases_evidence_and_reverifies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        nodes, "get_settings", lambda: _settings(openai_api_key="sk-test", cache_dir=str(tmp_path / "cache"))
    )
    monkeypatch.setattr(nodes, "get_search_provider", lambda settings: _FakeSearchProviderForExpand())
    monkeypatch.setattr(nodes, "WebpageExtractor", _FakeExtractorFactoryForExpand)

    async def fake_verify(claim: Claim, excerpts: list[str], *, model: object = None) -> ClaimVerificationOutcome:
        return ClaimVerificationOutcome(
            verification_status=VerificationStatus.VERIFIED, confidence=0.9, supports=True, notes="now confirmed"
        )

    monkeypatch.setattr(nodes, "verify_claim_against_sources", fake_verify)

    state = new_initial_state(research_id="R", topic="Topic", mock_mode=False)
    state["research_iteration_count"] = 0
    state["claims"] = [
        Claim(
            claim_id="C-1",
            text="An important under-evidenced claim.",
            importance=ClaimImportance.CRITICAL,
            verification_status=VerificationStatus.UNVERIFIED,
            confidence=0.0,
        )
    ]
    state["sources"] = []

    result = await nodes.expand_weak_claims(state)

    assert result["research_iteration_count"] == 1
    updated_claim = result["claims"][0]
    assert updated_claim.verification_status == VerificationStatus.VERIFIED
    assert updated_claim.confidence == 0.9
    assert len(result["sources"]) >= 1
    assert any(s.url and "new-evidence" in str(s.url) for s in result["sources"])
    assert result["verified_claims"] == [updated_claim]


@pytest.mark.asyncio
async def test_expand_weak_claims_skips_claims_that_are_not_weak(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nodes, "get_settings", lambda: _settings(openai_api_key="sk-test"))
    state = new_initial_state(research_id="R", topic="Topic", mock_mode=False)
    state["claims"] = [
        Claim(
            claim_id="C-1",
            text="Already well-supported claim.",
            importance=ClaimImportance.HIGH,
            verification_status=VerificationStatus.VERIFIED,
            confidence=0.9,
        )
    ]
    state["sources"] = []

    result = await nodes.expand_weak_claims(state)
    assert result["claims"] == state["claims"]
    assert result["sources"] == []


# --- analyze_script_template ------------------------------------------------------


@pytest.mark.asyncio
async def test_analyze_script_template_noop_without_template_path() -> None:
    state = new_initial_state(research_id="R", topic="Topic", mock_mode=False)
    result = await nodes.analyze_script_template(state)
    assert result == {}


@pytest.mark.asyncio
async def test_analyze_script_template_skipped_in_mock_mode() -> None:
    state = new_initial_state(research_id="R", topic="Topic", mock_mode=True, script_template_path="template.txt")
    result = await nodes.analyze_script_template(state)
    assert "script_template_spec" not in result
    assert result["warnings"]


@pytest.mark.asyncio
async def test_analyze_script_template_skipped_without_openai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nodes, "get_settings", lambda: _settings(openai_api_key=None))
    state = new_initial_state(research_id="R", topic="Topic", mock_mode=False, script_template_path="template.txt")
    result = await nodes.analyze_script_template(state)
    assert "script_template_spec" not in result
    assert result["warnings"]


@pytest.mark.asyncio
async def test_analyze_script_template_warns_on_missing_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nodes, "get_settings", lambda: _settings(openai_api_key="sk-test"))
    state = new_initial_state(
        research_id="R", topic="Topic", mock_mode=False, script_template_path="does_not_exist.txt"
    )
    result = await nodes.analyze_script_template(state)
    assert "script_template_spec" not in result
    assert result["warnings"]


@pytest.mark.asyncio
async def test_analyze_script_template_warns_on_empty_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(nodes, "get_settings", lambda: _settings(openai_api_key="sk-test"))
    template_path = tmp_path / "empty.txt"
    template_path.write_text("   ", encoding="utf-8")

    state = new_initial_state(
        research_id="R", topic="Topic", mock_mode=False, script_template_path=str(template_path)
    )
    result = await nodes.analyze_script_template(state)
    assert "script_template_spec" not in result
    assert result["warnings"]


@pytest.mark.asyncio
async def test_analyze_script_template_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(nodes, "get_settings", lambda: _settings(openai_api_key="sk-test"))
    template_path = tmp_path / "template.txt"
    template_path.write_text("HOOK: ...\nCHAPTER 1: ...", encoding="utf-8")

    canned_spec = ScriptStyleSpec(tone="urgent")

    async def fake_analyze(template_text: str, *, model: object = None) -> ScriptStyleSpec:
        return canned_spec

    monkeypatch.setattr(nodes, "analyze_template", fake_analyze)

    state = new_initial_state(
        research_id="R", topic="Topic", mock_mode=False, script_template_path=str(template_path)
    )
    result = await nodes.analyze_script_template(state)
    assert result["script_template_spec"] is canned_spec


@pytest.mark.asyncio
async def test_analyze_script_template_llm_failure_warns_without_crashing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(nodes, "get_settings", lambda: _settings(openai_api_key="sk-test"))
    template_path = tmp_path / "template.txt"
    template_path.write_text("HOOK: ...", encoding="utf-8")

    async def failing_analyze(template_text: str, *, model: object = None) -> ScriptStyleSpec:
        raise RuntimeError("boom")

    monkeypatch.setattr(nodes, "analyze_template", failing_analyze)

    state = new_initial_state(
        research_id="R", topic="Topic", mock_mode=False, script_template_path=str(template_path)
    )
    result = await nodes.analyze_script_template(state)
    assert "script_template_spec" not in result
    assert result["warnings"]


# --- build_story_architecture (live) -----------------------------------------------


@pytest.mark.asyncio
async def test_build_story_architecture_live_uses_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nodes, "get_settings", lambda: _settings(openai_api_key="sk-test"))
    canned = StoryArchitecture(central_question="Why?", sections=[StorySection(id="SEC-1", name="Origins", purpose="p")])

    async def fake_generate(topic: str, claims: list[Claim], contradictions: list[object], *, model: object = None) -> StoryArchitecture:
        return canned

    monkeypatch.setattr(nodes, "generate_story_architecture", fake_generate)

    state = new_initial_state(research_id="R", topic="Topic", mock_mode=False)
    result = await nodes.build_story_architecture(state)
    assert result["story_outline"] is canned


@pytest.mark.asyncio
async def test_build_story_architecture_live_falls_back_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nodes, "get_settings", lambda: _settings(openai_api_key="sk-test"))

    async def failing_generate(topic: str, claims: list[Claim], contradictions: list[object], *, model: object = None) -> StoryArchitecture:
        raise RuntimeError("boom")

    monkeypatch.setattr(nodes, "generate_story_architecture", failing_generate)

    state = new_initial_state(research_id="R", topic="Topic", mock_mode=False)
    result = await nodes.build_story_architecture(state)
    assert len(result["story_outline"].sections) == 6  # the fixed fallback template
    assert result["warnings"]


@pytest.mark.asyncio
async def test_build_story_architecture_live_falls_back_on_empty_sections(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nodes, "get_settings", lambda: _settings(openai_api_key="sk-test"))

    async def empty_generate(topic: str, claims: list[Claim], contradictions: list[object], *, model: object = None) -> StoryArchitecture:
        return StoryArchitecture(central_question="Why?", sections=[])

    monkeypatch.setattr(nodes, "generate_story_architecture", empty_generate)

    state = new_initial_state(research_id="R", topic="Topic", mock_mode=False)
    result = await nodes.build_story_architecture(state)
    assert len(result["story_outline"].sections) == 6


@pytest.mark.asyncio
async def test_build_story_architecture_live_no_openai_key_uses_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nodes, "get_settings", lambda: _settings(openai_api_key=None))
    state = new_initial_state(research_id="R", topic="Topic", mock_mode=False)
    result = await nodes.build_story_architecture(state)
    assert len(result["story_outline"].sections) == 6


# --- generate_hooks (live) ----------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_hooks_live_uses_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nodes, "get_settings", lambda: _settings(openai_api_key="sk-test"))
    canned = [Hook(id="H-1", hook_type=HookType.QUESTION, text="Why did this happen?")]

    async def fake_generate(topic: str, claims: list[Claim], contradictions: list[object], *, model: object = None) -> list[Hook]:
        return canned

    monkeypatch.setattr(nodes, "generate_hooks_llm", fake_generate)

    state = new_initial_state(research_id="R", topic="Topic", mock_mode=False)
    result = await nodes.generate_hooks(state)
    assert result["hook_options"] is canned


@pytest.mark.asyncio
async def test_generate_hooks_live_falls_back_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nodes, "get_settings", lambda: _settings(openai_api_key="sk-test"))

    async def failing_generate(topic: str, claims: list[Claim], contradictions: list[object], *, model: object = None) -> list[Hook]:
        raise RuntimeError("boom")

    monkeypatch.setattr(nodes, "generate_hooks_llm", failing_generate)

    state = new_initial_state(research_id="R", topic="Topic", mock_mode=False)
    result = await nodes.generate_hooks(state)
    assert len(result["hook_options"]) == 5  # the fixed fallback templates
    assert result["warnings"]


@pytest.mark.asyncio
async def test_generate_hooks_live_falls_back_on_empty_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nodes, "get_settings", lambda: _settings(openai_api_key="sk-test"))

    async def empty_generate(topic: str, claims: list[Claim], contradictions: list[object], *, model: object = None) -> list[Hook]:
        return []

    monkeypatch.setattr(nodes, "generate_hooks_llm", empty_generate)

    state = new_initial_state(research_id="R", topic="Topic", mock_mode=False)
    result = await nodes.generate_hooks(state)
    assert len(result["hook_options"]) == 5


@pytest.mark.asyncio
async def test_generate_hooks_live_no_openai_key_uses_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nodes, "get_settings", lambda: _settings(openai_api_key=None))
    state = new_initial_state(research_id="R", topic="Topic", mock_mode=False)
    result = await nodes.generate_hooks(state)
    assert len(result["hook_options"]) == 5


# --- generate_script (live) ----------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_script_live_uses_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nodes, "get_settings", lambda: _settings(openai_api_key="sk-test"))
    canned = DraftScript(version=1, hook_text="H", sections=[ScriptSection(id="SS-1", heading="A", narration="n")])

    async def fake_generate(topic: str, **kwargs: object) -> DraftScript:
        return canned

    monkeypatch.setattr(nodes, "generate_script_llm", fake_generate)

    state = new_initial_state(research_id="R", topic="Topic", mock_mode=False)
    result = await nodes.generate_script(state)
    assert result["draft_script"] is canned


@pytest.mark.asyncio
async def test_generate_script_live_falls_back_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nodes, "get_settings", lambda: _settings(openai_api_key="sk-test"))

    async def failing_generate(topic: str, **kwargs: object) -> DraftScript:
        raise RuntimeError("boom")

    monkeypatch.setattr(nodes, "generate_script_llm", failing_generate)
    state = new_initial_state(research_id="R", topic="Topic", mock_mode=False)
    result = await nodes.generate_script(state)
    assert result["draft_script"] is not None
    assert result["warnings"]


@pytest.mark.asyncio
async def test_generate_script_live_falls_back_on_empty_sections(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nodes, "get_settings", lambda: _settings(openai_api_key="sk-test"))

    async def empty_generate(topic: str, **kwargs: object) -> DraftScript:
        return DraftScript(version=1, hook_text="H", sections=[])

    monkeypatch.setattr(nodes, "generate_script_llm", empty_generate)
    state = new_initial_state(research_id="R", topic="Topic", mock_mode=False)
    result = await nodes.generate_script(state)
    assert result["warnings"]


@pytest.mark.asyncio
async def test_generate_script_live_no_openai_key_uses_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nodes, "get_settings", lambda: _settings(openai_api_key=None))
    state = new_initial_state(research_id="R", topic="Topic", mock_mode=False)
    result = await nodes.generate_script(state)
    assert result["draft_script"] is not None


# --- fact_check_script (live) ---------------------------------------------------------


@pytest.mark.asyncio
async def test_fact_check_script_live_uses_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nodes, "get_settings", lambda: _settings(openai_api_key="sk-test"))

    async def fake_check(narration: str, cited_claims: list[Claim], *, model: object = None) -> list[FactCheckFinding]:
        return [FactCheckFinding(claim_id="C-1", statement=narration, verdict=FactCheckVerdict.PASS)]

    monkeypatch.setattr(nodes, "fact_check_section", fake_check)

    state = new_initial_state(research_id="R", topic="Topic", mock_mode=False)
    state["draft_script"] = DraftScript(
        version=1,
        hook_text="H",
        sections=[ScriptSection(id="SS-1", heading="A", narration="n", cited_claim_ids=["C-1"])],
    )
    state["claims"] = [Claim(claim_id="C-1", text="x")]

    result = await nodes.fact_check_script(state)
    fact_check = result["fact_check_results"]
    assert fact_check.overall_verdict == FactCheckVerdict.PASS
    assert len(fact_check.findings) == 1


@pytest.mark.asyncio
async def test_fact_check_script_live_no_draft_uses_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nodes, "get_settings", lambda: _settings(openai_api_key="sk-test"))
    state = new_initial_state(research_id="R", topic="Topic", mock_mode=False)
    result = await nodes.fact_check_script(state)
    assert result["fact_check_results"] is not None


@pytest.mark.asyncio
async def test_fact_check_script_live_no_openai_key_uses_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nodes, "get_settings", lambda: _settings(openai_api_key=None))
    state = new_initial_state(research_id="R", topic="Topic", mock_mode=False)
    state["draft_script"] = DraftScript(version=1, hook_text="H", sections=[])
    result = await nodes.fact_check_script(state)
    assert result["fact_check_results"] is not None


# --- citation_audit (deterministic, runs the same in mock and live) ------------------


def test_citation_audit_warns_on_unresolvable_citation() -> None:
    state = new_initial_state(research_id="R", topic="Topic", mock_mode=False)
    state["draft_script"] = DraftScript(
        version=1, hook_text="H", sections=[ScriptSection(id="SS-1", heading="A", narration="n", cited_claim_ids=["C-1"])]
    )
    state["evidence"] = []

    result = nodes.citation_audit(state)
    assert result["citation_map"] == {}
    assert any("C-1" in w for w in result["warnings"])


def test_citation_audit_warns_on_unacknowledged_contradiction() -> None:
    state = new_initial_state(research_id="R", topic="Topic", mock_mode=False)
    state["draft_script"] = DraftScript(version=1, hook_text="H", sections=[])
    state["contradictions"] = [Contradiction(id="X-1", claim_ids=["C-1", "C-2"], description="conflict")]

    result = nodes.citation_audit(state)
    assert any("contradiction" in w.lower() for w in result["warnings"])


def test_citation_audit_resolves_valid_citation_without_warning() -> None:
    state = new_initial_state(research_id="R", topic="Topic", mock_mode=False)
    state["draft_script"] = DraftScript(
        version=1, hook_text="H", sections=[ScriptSection(id="SS-1", heading="A", narration="n", cited_claim_ids=["C-1"])]
    )
    state["evidence"] = [EvidenceMatrixEntry(claim_id="C-1", claim_text="x", supporting_source_ids=["S-1"])]
    state["sources"] = [Source(id="S-1", title="Src", source_type=SourceType.WEB_SEARCH)]

    result = nodes.citation_audit(state)
    assert result["citation_map"] == {"C-1": "S-1"}
    assert result["warnings"] == []


# --- final_revision (live regeneration) -----------------------------------------------


@pytest.mark.asyncio
async def test_final_revision_live_regenerates_script_with_feedback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nodes, "get_settings", lambda: _settings(openai_api_key="sk-test"))
    captured: dict[str, object] = {}

    async def fake_generate(topic: str, *, feedback: list[str] | None = None, version: int = 1, **kwargs: object) -> DraftScript:
        captured["feedback"] = feedback
        captured["version"] = version
        return DraftScript(version=version, hook_text="H", sections=[ScriptSection(id="SS-1", heading="A", narration="n")])

    monkeypatch.setattr(nodes, "generate_script_llm", fake_generate)

    state = new_initial_state(research_id="R", topic="Topic", mock_mode=False)
    state["iteration_count"] = 0
    state["fact_check_results"] = FactCheckResult(
        findings=[FactCheckFinding(claim_id="C-1", statement="s", verdict=FactCheckVerdict.FAIL, notes="bad")],
        overall_verdict=FactCheckVerdict.FAIL,
    )

    result = await nodes.final_revision(state)
    assert result["iteration_count"] == 1
    assert result["draft_script"].version == 2
    assert captured["feedback"]
    assert result["warnings"]


@pytest.mark.asyncio
async def test_final_revision_live_no_openai_key_does_not_regenerate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nodes, "get_settings", lambda: _settings(openai_api_key=None))
    state = new_initial_state(research_id="R", topic="Topic", mock_mode=False)
    result = await nodes.final_revision(state)
    assert "draft_script" not in result
    assert result["iteration_count"] == 1


@pytest.mark.asyncio
async def test_final_revision_live_llm_failure_keeps_iteration_and_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nodes, "get_settings", lambda: _settings(openai_api_key="sk-test"))

    async def failing_generate(topic: str, **kwargs: object) -> DraftScript:
        raise RuntimeError("boom")

    monkeypatch.setattr(nodes, "generate_script_llm", failing_generate)
    state = new_initial_state(research_id="R", topic="Topic", mock_mode=False)
    result = await nodes.final_revision(state)
    assert "draft_script" not in result
    assert result["iteration_count"] == 1
    assert result["warnings"]


@pytest.mark.asyncio
async def test_final_revision_mock_mode_unchanged() -> None:
    state = new_initial_state(research_id="R", topic="Topic", mock_mode=True)
    result = await nodes.final_revision(state)
    assert result["iteration_count"] == 1
    assert "draft_script" not in result
