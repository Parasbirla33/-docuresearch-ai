"""LangGraph node functions.

Every node takes the current `DocuResearchState` and returns a partial dict of
updates - never mutates state in place.

`mock_mode=True` always uses the clearly-labelled placeholder dataset (see
`docuresearch.mock.sample_data`) and never makes a network/LLM call - this is
the path exercised by the automated tests and by `--mock`.

`mock_mode=False` uses real tools (`docuresearch.tools.*`) and real LLM calls
(`docuresearch.agents.research_agent`, `docuresearch.extraction.claim_extractor`,
`docuresearch.verification.claim_verifier`). If a specific provider isn't
configured (no OPENAI_API_KEY / no SEARCH_API_KEY / ...), that capability
contributes nothing and a warning is logged - it never falls back to the fake
mock dataset, since mixing fabricated placeholder facts into what's presented
as real research would violate the project's core "never fabricate" rule.
The one exception is research-plan *questions* (not facts): without an LLM,
`create_research_plan` falls back to a generic, non-fabricated question
template rather than producing no plan at all.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from pydantic import HttpUrl, ValidationError

from docuresearch.agents.hook_agent import generate_hooks as generate_hooks_llm
from docuresearch.agents.research_agent import generate_research_plan
from docuresearch.agents.script_agent import generate_script as generate_script_llm
from docuresearch.agents.story_agent import generate_story_architecture
from docuresearch.config.settings import get_settings
from docuresearch.extraction.claim_extractor import extract_claims as extract_claims_from_document
from docuresearch.extraction.template_analyzer import analyze_script_template as analyze_template
from docuresearch.mock.sample_data import build_mock_claims, build_mock_sources
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
    ResearchQuestion,
    StoryArchitecture,
    StorySection,
    VisualSuggestion,
)
from docuresearch.models.script import (
    CitationEntry,
    DraftScript,
    FactCheckFinding,
    FactCheckResult,
    FactCheckVerdict,
    FinalOutput,
    QualityScore,
    ScriptSection,
)
from docuresearch.models.sources import Document, Source, SourceAvailability, SourceType
from docuresearch.models.state import DocuResearchState
from docuresearch.storage.cache import ResearchCache
from docuresearch.tools.source_ranker import rank_sources as rank_and_dedupe_sources
from docuresearch.tools.web_search import SearchHit, get_search_provider
from docuresearch.tools.webpage import WebpageExtractor
from docuresearch.tools.wikipedia import WikipediaPage, WikipediaTool
from docuresearch.utils.hashing import new_id, normalize_url
from docuresearch.utils.logging import get_logger
from docuresearch.utils.text import truncate
from docuresearch.verification.claim_verifier import verify_claim as verify_claim_against_sources
from docuresearch.verification.contradiction_detector import (
    check_pair_for_contradiction,
    find_candidate_pairs,
)
from docuresearch.verification.script_fact_checker import fact_check_section

logger = get_logger(__name__)

QUALITY_FACTUAL_SAFETY_THRESHOLD = 70.0
QUALITY_CITATION_THRESHOLD = 60.0
MAX_VERIFICATION_EXCERPT_CHARS = 2000
WEAK_CONFIDENCE_THRESHOLD = 0.5
MAX_WEAK_CLAIMS_PER_PASS = 5
MAX_NEW_SOURCES_PER_CLAIM = 3

_SEVERITY_ORDER = list(ClaimImportance)  # declaration order = LOW < MEDIUM < HIGH < CRITICAL


def _more_severe(a: ClaimImportance, b: ClaimImportance) -> ClaimImportance:
    return a if _SEVERITY_ORDER.index(a) >= _SEVERITY_ORDER.index(b) else b


def is_weak_important_claim(claim: Claim) -> bool:
    """A claim worth chasing more evidence for: important, but not yet well-supported.

    Disputed/false claims are excluded - those have evidence that conflicts,
    which is a different problem (contradiction detection) from having too
    little evidence.
    """
    if claim.importance not in (ClaimImportance.HIGH, ClaimImportance.CRITICAL):
        return False
    if claim.verification_status in (VerificationStatus.DISPUTED, VerificationStatus.FALSE_OR_UNSUPPORTED):
        return False
    if claim.verification_status == VerificationStatus.UNVERIFIED:
        return True
    return claim.confidence < WEAK_CONFIDENCE_THRESHOLD


def _split_verified(claims: list[Claim]) -> tuple[list[Claim], list[Claim]]:
    verified = [c for c in claims if c.verification_status == VerificationStatus.VERIFIED]
    unverified = [
        c
        for c in claims
        if c.verification_status in (VerificationStatus.UNVERIFIED, VerificationStatus.FALSE_OR_UNSUPPORTED)
    ]
    return verified, unverified


def _safe_http_url(url: str) -> HttpUrl | None:
    try:
        return HttpUrl(url)
    except ValidationError:
        return None


def _now() -> datetime:
    return datetime.now(UTC)


def intake_request(state: DocuResearchState) -> dict[str, Any]:
    """Validate/normalize user inputs. First node in the graph."""
    logger.info("intake_request", topic=state.get("topic"), mock=state.get("mock_mode"))
    warnings: list[str] = []
    if not state.get("topic"):
        warnings.append("No topic provided.")
    if not state.get("mock_mode") and not get_settings().has_openai:
        warnings.append(
            "No OPENAI_API_KEY configured - research planning falls back to a generic "
            "question template, and claim extraction/verification will be skipped."
        )
    return {"warnings": warnings}


async def analyze_script_template(state: DocuResearchState) -> dict[str, Any]:
    """Read and analyze the user's script template (style/structure only - never its facts).

    A no-op (leaves `script_template_spec` as None) when no template was
    given, in mock mode, or without an LLM provider - the script generator
    falls back to sensible defaults in that case, never to fabricated style.
    """
    template_path = state.get("script_template_path")
    if not template_path:
        return {}

    if state.get("mock_mode"):
        logger.warning("script_template_skipped", reason="mock_mode", path=template_path)
        return {"warnings": ["Script template provided but skipped: template analysis never runs in mock mode."]}

    settings = get_settings()
    if not settings.has_openai:
        logger.warning("script_template_skipped", reason="no_openai_key", path=template_path)
        return {"warnings": ["Script template provided but OPENAI_API_KEY is not set - template analysis skipped."]}

    try:
        template_text = Path(template_path).read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("script_template_read_failed", path=template_path, error=str(exc))
        return {"warnings": [f"Could not read script template at {template_path}: {exc}"]}

    if not template_text.strip():
        logger.warning("script_template_empty", path=template_path)
        return {"warnings": [f"Script template at {template_path} is empty - ignoring it."]}

    try:
        spec = await analyze_template(template_text)
    except Exception as exc:  # noqa: BLE001 - template analysis failing must not abort the run
        logger.warning("script_template_analysis_failed", error=str(exc))
        return {"warnings": [f"Script template analysis failed ({exc}); using default script style."]}

    return {"script_template_spec": spec}


def _fallback_research_plan(topic: str, depth: ResearchDepth, *, note: str) -> ResearchPlan:
    """Generic (non-LLM, non-fabricated) question template. Real questions, no invented facts."""
    questions = [
        ResearchQuestion(id=new_id("Q"), question=f"What is the historical context of {topic}?"),
        ResearchQuestion(id=new_id("Q"), question=f"What policies or decisions shaped {topic}?"),
        ResearchQuestion(
            id=new_id("Q"),
            question=f"Which people/organizations are central to {topic}?",
            requires_multiple_sources=True,
        ),
        ResearchQuestion(
            id=new_id("Q"),
            question=f"What measurable effects resulted from {topic}?",
            requires_primary_source=True,
        ),
        ResearchQuestion(id=new_id("Q"), question=f"What controversies exist around {topic}?"),
    ]
    return ResearchPlan(
        topic=topic,
        depth=depth,
        key_questions=questions,
        notes=note,
    )


async def create_research_plan(state: DocuResearchState) -> dict[str, Any]:
    """Generate a research plan: key questions, entities, controversies to chase."""
    topic = state.get("topic", "Untitled topic")
    depth = state.get("research_depth", ResearchDepth.STANDARD)

    if state.get("mock_mode"):
        plan = _fallback_research_plan(
            topic, depth, note="[MOCK] Placeholder plan - mock mode never calls a real LLM."
        )
        return {"research_plan": plan}

    settings = get_settings()
    if not settings.has_openai:
        plan = _fallback_research_plan(
            topic, depth, note="Generic question template - no OPENAI_API_KEY configured."
        )
        return {"research_plan": plan}

    try:
        plan = await generate_research_plan(
            topic,
            depth=depth,
            date_range=state.get("date_range"),
            geographic_focus=state.get("geographic_focus"),
        )
        return {"research_plan": plan}
    except Exception as exc:  # noqa: BLE001 - any LLM/SDK/network failure degrades gracefully, never crashes the run
        logger.warning("research_plan_llm_failed", error=str(exc))
        plan = _fallback_research_plan(
            topic, depth, note=f"Generic question template - LLM planning failed ({exc})."
        )
        return {
            "research_plan": plan,
            "warnings": [f"LLM research planning failed; used a generic question template. ({exc})"],
        }


def generate_search_queries(state: DocuResearchState) -> dict[str, Any]:
    """Turn research questions into concrete search queries, including opposing-view queries."""
    plan = state.get("research_plan")
    queries: list[str] = []
    if plan:
        for q in plan.key_questions:
            queries.append(q.question)
        topic = plan.topic
        queries += [f"{topic} criticism", f"{topic} controversy", f"{topic} official data"]
    return {"research_queries": queries}


async def parallel_source_discovery(state: DocuResearchState) -> dict[str, Any]:
    """Discover candidate sources across pluggable adapters (web search, Wikipedia, ...).

    Wikipedia's own external links become candidate sources for deeper
    verification (spec section 19), not treated as authoritative themselves.
    """
    topic = state.get("topic", "Untitled topic")

    if state.get("mock_mode"):
        mock_sources = build_mock_sources(topic)
        logger.info("parallel_source_discovery", mode="mock", discovered=len(mock_sources))
        return {"sources": mock_sources}

    settings = get_settings()
    queries = state.get("research_queries") or [topic]
    depth = state.get("research_depth", ResearchDepth.STANDARD)
    target = settings.depth_limits.target_for(depth, hard_cap=settings.max_research_sources)

    search_provider = get_search_provider(settings)
    wiki_tool = WikipediaTool.from_settings(settings)
    semaphore = asyncio.Semaphore(settings.max_concurrent_requests)

    async def _web_search(query: str) -> list[SearchHit]:
        async with semaphore:
            try:
                return await search_provider.search(query, max_results=5)
            except Exception as exc:  # noqa: BLE001 - one bad query must not abort the whole discovery fan-out
                logger.warning("web_search_query_failed", query=query, error=str(exc))
                return []

    async def _wiki_search(query: str) -> list[str]:
        async with semaphore:
            try:
                return await wiki_tool.search(query, max_results=2)
            except Exception as exc:  # noqa: BLE001 - one bad query must not abort the whole discovery fan-out
                logger.warning("wikipedia_search_query_failed", query=query, error=str(exc))
                return []

    search_batches, wiki_title_batches = await asyncio.gather(
        asyncio.gather(*(_web_search(q) for q in queries)),
        asyncio.gather(*(_wiki_search(q) for q in queries)),
    )
    wiki_titles = list(dict.fromkeys(t for titles in wiki_title_batches for t in titles))

    async def _wiki_page(title: str) -> WikipediaPage | None:
        async with semaphore:
            try:
                return await wiki_tool.get_page(title)
            except Exception as exc:  # noqa: BLE001 - one bad page must not abort the whole discovery fan-out
                logger.warning("wikipedia_get_page_failed", title=title, error=str(exc))
                return None

    wiki_pages = await asyncio.gather(*(_wiki_page(t) for t in wiki_titles))

    sources: list[Source] = []
    seen_urls: set[str] = set()

    def _add(url: str, **fields: Any) -> None:
        parsed_url = _safe_http_url(url)
        if parsed_url is None:
            return
        normalized = normalize_url(url)
        if normalized in seen_urls:
            return
        seen_urls.add(normalized)
        sources.append(Source(id=new_id("S"), url=parsed_url, domain=urlparse(url).netloc, **fields))

    for hits in search_batches:
        for hit in hits:
            _add(
                hit.url,
                title=hit.title or hit.url,
                source_type=SourceType.WEB_SEARCH,
                summary=hit.snippet or None,
                relevance_score=hit.score,
                authority_score=0.5,
                credibility_score=0.5,
            )

    for page in wiki_pages:
        if page is None:
            continue
        _add(
            page.url,
            title=page.title,
            source_type=SourceType.WIKIPEDIA,
            text=page.extract or None,
            relevance_score=0.6,
            authority_score=0.5,
            credibility_score=0.55,
        )
        for ref_url in page.references[:5]:
            _add(
                ref_url,
                title=ref_url,
                source_type=SourceType.UNKNOWN,
                relevance_score=0.3,
                authority_score=0.4,
                credibility_score=0.4,
            )

    sources = sources[:target]
    warnings = (
        ["No sources discovered - check SEARCH_API_KEY / WIKIPEDIA_API_ENABLED configuration."]
        if not sources
        else []
    )

    logger.info("parallel_source_discovery", mode="live", discovered=len(sources))
    return {"sources": sources, "warnings": warnings}


async def collect_sources(state: DocuResearchState) -> dict[str, Any]:
    """Fetch/cache full content for discovered sources. No-op in mock mode - already populated."""
    if state.get("mock_mode"):
        return {}

    sources = state.get("sources", [])
    to_fetch = [s for s in sources if s.url and not s.text]
    if not to_fetch:
        return {}

    settings = get_settings()
    cache = ResearchCache(Path(settings.cache_dir))
    extractor = WebpageExtractor.from_settings(settings)
    semaphore = asyncio.Semaphore(settings.max_concurrent_requests)

    async def _fetch_one(source: Source) -> Source:
        url = str(source.url)
        cached = cache.get(url)
        if cached:
            return source.model_copy(
                update={
                    "text": cached.content,
                    "title": cached.title or source.title,
                    "availability": SourceAvailability.AVAILABLE,
                }
            )

        async with semaphore:
            result = await extractor.extract(url)

        if result.availability != SourceAvailability.AVAILABLE or not result.text:
            logger.info("collect_source_unavailable", url=url, availability=result.availability.value)
            return source.model_copy(update={"availability": result.availability})

        cache.set(url, title=result.title, content=result.text)
        return source.model_copy(
            update={
                "text": result.text,
                "title": result.title or source.title,
                "availability": SourceAvailability.AVAILABLE,
            }
        )

    fetched = await asyncio.gather(*(_fetch_one(s) for s in to_fetch))
    fetched_by_id = {s.id: s for s in fetched}
    updated_sources = [fetched_by_id.get(s.id, s) for s in sources]

    unavailable = sum(1 for s in updated_sources if s.availability != SourceAvailability.AVAILABLE)
    warnings = (
        [f"{unavailable} of {len(to_fetch)} newly discovered sources were unavailable and contributed no text."]
        if unavailable
        else []
    )
    return {"sources": updated_sources, "warnings": warnings}


def extract_documents(state: DocuResearchState) -> dict[str, Any]:
    """Extract clean article text from collected sources into Document records."""
    method = "mock" if state.get("mock_mode") else "trafilatura"
    documents = [
        Document(
            id=new_id("D"),
            source_id=s.id,
            clean_text=s.text or "",
            word_count=len((s.text or "").split()),
            extracted_at=_now(),
            extraction_method=method,
        )
        for s in state.get("sources", [])
        if s.text
    ]
    return {"documents": documents}


async def extract_claims(state: DocuResearchState) -> dict[str, Any]:
    """Extract atomic, checkable claims from documents/sources."""
    topic = state.get("topic", "Untitled topic")

    if state.get("mock_mode"):
        sources = state.get("sources", [])
        claims = build_mock_claims(topic, sources)
        logger.info("extract_claims", mode="mock", count=len(claims))
        return {"claims": claims}

    settings = get_settings()
    if not settings.has_openai:
        logger.warning("extract_claims_no_llm_provider", detail="No OPENAI_API_KEY - no claims extracted.")
        return {"claims": [], "warnings": ["No OPENAI_API_KEY configured - claim extraction skipped."]}

    documents = [d for d in state.get("documents", []) if d.clean_text.strip()]
    sources_by_id = {s.id: s for s in state.get("sources", [])}
    semaphore = asyncio.Semaphore(settings.max_concurrent_requests)

    async def _extract_one(document: Document) -> list[Claim]:
        source = sources_by_id.get(document.source_id)
        label = source.title if source else document.source_id
        async with semaphore:
            try:
                return await extract_claims_from_document(
                    document.clean_text, source_id=document.source_id, topic=topic, source_label=label
                )
            except Exception as exc:  # noqa: BLE001 - one bad document must not abort the whole extraction fan-out
                logger.warning("claim_extraction_failed", source_id=document.source_id, error=str(exc))
                return []

    results = await asyncio.gather(*(_extract_one(d) for d in documents))
    claims = [c for batch in results for c in batch]

    logger.info("extract_claims", mode="live", count=len(claims))
    warnings = ["No claims could be extracted from the collected sources."] if not claims else []
    return {"claims": claims, "warnings": warnings}


def rank_sources(state: DocuResearchState) -> dict[str, Any]:
    """Score/rank sources by authority, relevance, credibility (spec section 10)."""
    sources = state.get("sources", [])
    if state.get("mock_mode"):
        return {"sources": sorted(sources, key=lambda s: s.confidence_score, reverse=True)}
    return {"sources": rank_and_dedupe_sources(sources, state.get("topic", ""))}


async def verify_claims(state: DocuResearchState) -> dict[str, Any]:
    """Verify each claim against its cited sources (live mode), then split verified/unverified."""
    claims = state.get("claims", [])

    if not state.get("mock_mode") and claims:
        settings = get_settings()
        if settings.has_openai:
            sources_by_id = {s.id: s for s in state.get("sources", [])}
            semaphore = asyncio.Semaphore(settings.max_concurrent_requests)

            async def _verify_one(claim: Claim) -> Claim:
                cited_texts = [
                    sources_by_id[sid].text for sid in claim.source_ids if sid in sources_by_id
                ]
                excerpts = [truncate(text, MAX_VERIFICATION_EXCERPT_CHARS) for text in cited_texts if text]
                async with semaphore:
                    try:
                        outcome = await verify_claim_against_sources(claim, excerpts)
                    except Exception as exc:  # noqa: BLE001 - one bad claim must not abort the whole verification fan-out
                        logger.warning("claim_verification_failed", claim_id=claim.claim_id, error=str(exc))
                        return claim
                return claim.model_copy(
                    update={
                        "verification_status": outcome.verification_status,
                        "confidence": outcome.confidence,
                        "notes": outcome.notes,
                    }
                )

            claims = list(await asyncio.gather(*(_verify_one(c) for c in claims)))
        else:
            logger.warning("verify_claims_no_llm_provider", detail="No OPENAI_API_KEY - claims left unverified.")

    verified, unverified = _split_verified(claims)
    return {"claims": claims, "verified_claims": verified, "unverified_claims": unverified}


async def expand_weak_claims(state: DocuResearchState) -> dict[str, Any]:
    """Adaptive research loop (spec section 52).

    Chases additional targeted evidence for important claims that don't yet
    have enough of it, then re-verifies just those claims. Bounded by
    MAX_WEAK_CLAIMS_PER_PASS/MAX_NEW_SOURCES_PER_CLAIM per pass and by
    MAX_RESEARCH_ITERATIONS overall (enforced by `route_after_verify_claims`).
    """
    settings = get_settings()
    claims = state.get("claims", [])
    existing_sources = list(state.get("sources", []))
    existing_by_id = {s.id: s for s in existing_sources}
    seen_urls = {normalize_url(str(s.url)) for s in existing_sources if s.url}

    weak = sorted((c for c in claims if is_weak_important_claim(c)), key=lambda c: c.confidence)[
        :MAX_WEAK_CLAIMS_PER_PASS
    ]

    search_provider = get_search_provider(settings)
    extractor = WebpageExtractor.from_settings(settings)
    cache = ResearchCache(Path(settings.cache_dir))
    semaphore = asyncio.Semaphore(settings.max_concurrent_requests)

    async def _search_query(query: str) -> list[SearchHit]:
        async with semaphore:
            try:
                return await search_provider.search(query, max_results=3)
            except Exception as exc:  # noqa: BLE001 - one bad query must not abort the whole pass
                logger.warning("expand_weak_claims_search_failed", query=query, error=str(exc))
                return []

    async def _fetch_text(source: Source) -> Source:
        url = str(source.url)
        cached = cache.get(url)
        if cached:
            return source.model_copy(
                update={"text": cached.content, "title": cached.title or source.title, "availability": SourceAvailability.AVAILABLE}
            )
        async with semaphore:
            result = await extractor.extract(url)
        if result.availability != SourceAvailability.AVAILABLE or not result.text:
            return source.model_copy(update={"availability": result.availability})
        cache.set(url, title=result.title, content=result.text)
        return source.model_copy(
            update={"text": result.text, "title": result.title or source.title, "availability": SourceAvailability.AVAILABLE}
        )

    async def _gather_new_sources(claim: Claim) -> list[Source]:
        queries = [f"{claim.text} evidence", f"{claim.text} official data"]
        batches = await asyncio.gather(*(_search_query(q) for q in queries))

        candidates: list[Source] = []
        for hit in (hit for batch in batches for hit in batch):
            if len(candidates) >= MAX_NEW_SOURCES_PER_CLAIM:
                break
            normalized = normalize_url(hit.url)
            if normalized in seen_urls:
                continue
            parsed_url = _safe_http_url(hit.url)
            if parsed_url is None:
                continue
            seen_urls.add(normalized)
            candidates.append(
                Source(
                    id=new_id("S"),
                    url=parsed_url,
                    title=hit.title or hit.url,
                    source_type=SourceType.WEB_SEARCH,
                    domain=urlparse(hit.url).netloc,
                    summary=hit.snippet or None,
                    relevance_score=hit.score,
                    authority_score=0.5,
                    credibility_score=0.5,
                )
            )
        return list(await asyncio.gather(*(_fetch_text(c) for c in candidates)))

    all_new_sources: list[Source] = []
    updated_claims_by_id: dict[str, Claim] = {}

    for claim in weak:
        fetched = await _gather_new_sources(claim)
        all_new_sources.extend(fetched)
        usable = [s for s in fetched if s.text]

        cited_texts = [existing_by_id[sid].text for sid in claim.source_ids if sid in existing_by_id]
        excerpts = [truncate(t, MAX_VERIFICATION_EXCERPT_CHARS) for t in cited_texts if t] + [
            truncate(s.text, MAX_VERIFICATION_EXCERPT_CHARS) for s in usable if s.text
        ]

        try:
            outcome = await verify_claim_against_sources(claim, excerpts)
        except Exception as exc:  # noqa: BLE001 - one bad claim must not abort the whole pass
            logger.warning("expand_weak_claims_verify_failed", claim_id=claim.claim_id, error=str(exc))
            continue

        updated_claims_by_id[claim.claim_id] = claim.model_copy(
            update={
                "verification_status": outcome.verification_status,
                "confidence": outcome.confidence,
                "notes": outcome.notes,
                "source_ids": list(dict.fromkeys([*claim.source_ids, *(s.id for s in usable)])),
            }
        )

    updated_claims = [updated_claims_by_id.get(c.claim_id, c) for c in claims]
    updated_sources = existing_sources + all_new_sources
    verified, unverified = _split_verified(updated_claims)
    iteration = state.get("research_iteration_count", 0) + 1

    logger.info(
        "expand_weak_claims",
        iteration=iteration,
        weak_claim_count=len(weak),
        new_sources=len(all_new_sources),
    )

    return {
        "claims": updated_claims,
        "sources": updated_sources,
        "verified_claims": verified,
        "unverified_claims": unverified,
        "research_iteration_count": iteration,
        "warnings": (
            [f"Research loop pass {iteration}: chased {len(weak)} under-evidenced important claim(s)."]
            if weak
            else []
        ),
    }


async def detect_contradictions(state: DocuResearchState) -> dict[str, Any]:
    """Find claims that conflict with each other, mock path or real cross-claim detection."""
    claims = state.get("claims", [])

    if state.get("mock_mode"):
        disputed = [c for c in claims if c.verification_status == VerificationStatus.DISPUTED]
        contradictions: list[Contradiction] = []
        for i in range(0, len(disputed) - 1, 2):
            a, b = disputed[i], disputed[i + 1]
            contradictions.append(
                Contradiction(
                    id=new_id("X"),
                    claim_ids=[a.claim_id, b.claim_id],
                    description=f"Conflicting claims: '{a.text}' vs '{b.text}'",
                    side_a_source_ids=a.supporting_evidence,
                    side_b_source_ids=b.supporting_evidence,
                    severity=a.importance,
                )
            )
        if len(disputed) % 2 == 1:
            lone = disputed[-1]
            contradictions.append(
                Contradiction(
                    id=new_id("X"),
                    claim_ids=[lone.claim_id],
                    description=f"Disputed statistic without a clean opposing claim: '{lone.text}'",
                    side_a_source_ids=lone.supporting_evidence,
                    side_b_source_ids=lone.contradicting_evidence,
                    severity=lone.importance,
                )
            )
        return {"contradictions": contradictions}

    settings = get_settings()
    if not settings.has_openai or not claims:
        return {"contradictions": []}

    pairs = find_candidate_pairs(claims)
    if not pairs:
        return {"contradictions": []}

    semaphore = asyncio.Semaphore(settings.max_concurrent_requests)

    async def _check(pair: tuple[Claim, Claim]) -> tuple[Claim, Claim, bool, str | None]:
        a, b = pair
        async with semaphore:
            try:
                verdict = await check_pair_for_contradiction(a, b)
            except Exception as exc:  # noqa: BLE001 - one bad pair must not abort the whole scan
                logger.warning("contradiction_check_failed", claim_a=a.claim_id, claim_b=b.claim_id, error=str(exc))
                return a, b, False, None
        return a, b, verdict.conflicts, verdict.description

    results = await asyncio.gather(*(_check(pair) for pair in pairs))

    contradictions = []
    conflicting_claim_ids: dict[str, set[str]] = {}
    for a, b, conflicts, description in results:
        if not conflicts:
            continue
        contradictions.append(
            Contradiction(
                id=new_id("X"),
                claim_ids=[a.claim_id, b.claim_id],
                description=description or f"Conflicting claims: '{a.text}' vs '{b.text}'",
                side_a_source_ids=a.source_ids,
                side_b_source_ids=b.source_ids,
                severity=_more_severe(a.importance, b.importance),
            )
        )
        conflicting_claim_ids.setdefault(a.claim_id, set()).update(b.source_ids)
        conflicting_claim_ids.setdefault(b.claim_id, set()).update(a.source_ids)

    if conflicting_claim_ids:
        updated_claims = []
        for claim in claims:
            opposing_sources = conflicting_claim_ids.get(claim.claim_id)
            if not opposing_sources:
                updated_claims.append(claim)
                continue
            new_status = (
                claim.verification_status
                if claim.verification_status == VerificationStatus.FALSE_OR_UNSUPPORTED
                else VerificationStatus.DISPUTED
            )
            updated_claims.append(
                claim.model_copy(
                    update={
                        "verification_status": new_status,
                        "confidence": min(claim.confidence, 0.5),
                        "contradicting_evidence": list(
                            dict.fromkeys([*claim.contradicting_evidence, *opposing_sources])
                        ),
                    }
                )
            )
        claims = updated_claims

    verified, unverified = _split_verified(claims)
    logger.info("detect_contradictions", mode="live", count=len(contradictions))
    return {
        "claims": claims,
        "verified_claims": verified,
        "unverified_claims": unverified,
        "contradictions": contradictions,
    }


def build_evidence_matrix(state: DocuResearchState) -> dict[str, Any]:
    """Build the claim -> supporting/contradicting sources -> confidence -> status matrix."""
    evidence = [
        EvidenceMatrixEntry(
            claim_id=c.claim_id,
            claim_text=c.text,
            supporting_source_ids=c.supporting_evidence or c.source_ids,
            contradicting_source_ids=c.contradicting_evidence,
            confidence=c.confidence,
            status=c.verification_status,
        )
        for c in state.get("claims", [])
    ]
    return {"evidence": evidence}


def _fallback_story_architecture(topic: str, verified: list[Claim]) -> StoryArchitecture:
    """Fixed 6-beat template. Used for mock mode and as a live-mode fallback on LLM failure."""
    section_defs = [
        ("Context", "Establish the world before the central event."),
        ("Inciting Event", "Introduce the change that set everything in motion."),
        ("Evidence", "Present the strongest verified claims."),
        ("Conflict", "Present disputed/contradictory claims fairly."),
        ("Consequences", "Show measurable effects."),
        ("Current Status", "Where things stand today."),
    ]
    sections = [
        StorySection(
            id=new_id("SEC"),
            name=name,
            purpose=purpose,
            key_claim_ids=[verified[i % len(verified)].claim_id] if verified else [],
            order=i,
        )
        for i, (name, purpose) in enumerate(section_defs)
    ]
    return StoryArchitecture(
        central_question=f"What really happened with {topic}, and why does it matter?",
        sections=sections,
        theme="Transformation grounded in verified evidence.",
    )


async def build_story_architecture(state: DocuResearchState) -> dict[str, Any]:
    """Convert verified research into an adaptive documentary narrative structure."""
    topic = state.get("topic", "Untitled topic")
    verified = state.get("verified_claims", [])

    if state.get("mock_mode"):
        return {"story_outline": _fallback_story_architecture(topic, verified)}

    settings = get_settings()
    if not settings.has_openai:
        return {"story_outline": _fallback_story_architecture(topic, verified)}

    try:
        architecture = await generate_story_architecture(topic, verified, state.get("contradictions", []))
    except Exception as exc:  # noqa: BLE001 - LLM/SDK failure must not abort the run
        logger.warning("story_architecture_llm_failed", error=str(exc))
        return {
            "story_outline": _fallback_story_architecture(topic, verified),
            "warnings": [f"LLM story architecture failed; used the default structure. ({exc})"],
        }

    if not architecture.sections:
        logger.warning("story_architecture_empty", detail="LLM returned no sections; using default structure.")
        return {"story_outline": _fallback_story_architecture(topic, verified)}

    return {"story_outline": architecture}


def _fallback_hooks(topic: str, contradictions: list[Contradiction]) -> list[Hook]:
    """Fixed hook templates. Used for mock mode and as a live-mode fallback on LLM failure."""
    templates = [
        (HookType.QUESTION, f"What actually caused {topic} - and who benefited most?"),
        (HookType.CONTRADICTION, f"Officials and industry insiders tell two different stories about {topic}."),
        (HookType.HISTORICAL, f"Before {topic}, almost nobody saw this coming."),
        (HookType.LESSER_KNOWN, f"Here's what most accounts of {topic} leave out."),
        (HookType.MONEY_POWER, f"Behind {topic} was a fight over money and control."),
    ]
    hooks: list[Hook] = []
    for hook_type, text in templates:
        clickbait_risk = 0.5 if hook_type == HookType.MONEY_POWER else 0.2
        hooks.append(
            Hook(
                id=new_id("H"),
                hook_type=hook_type,
                text=text,
                supporting_claim_ids=[c.claim_ids[0] for c in contradictions[:1]],
                curiosity_score=0.8,
                specificity_score=0.7,
                emotional_tension_score=0.65,
                information_gap_score=0.75,
                relevance_score=0.9,
                truthfulness_score=0.95,
                clickbait_risk=clickbait_risk,
            )
        )
    return hooks


async def generate_hooks(state: DocuResearchState) -> dict[str, Any]:
    """Generate multiple candidate hooks, each factually defensible."""
    topic = state.get("topic", "Untitled topic")
    contradictions = state.get("contradictions", [])

    if state.get("mock_mode"):
        return {"hook_options": _fallback_hooks(topic, contradictions)}

    settings = get_settings()
    if not settings.has_openai:
        return {"hook_options": _fallback_hooks(topic, contradictions)}

    verified = state.get("verified_claims", [])
    try:
        hooks = await generate_hooks_llm(topic, verified, contradictions)
    except Exception as exc:  # noqa: BLE001 - LLM/SDK failure must not abort the run
        logger.warning("hook_generation_llm_failed", error=str(exc))
        return {
            "hook_options": _fallback_hooks(topic, contradictions),
            "warnings": [f"LLM hook generation failed; used default hooks. ({exc})"],
        }

    if not hooks:
        logger.warning("hook_generation_empty", detail="LLM returned no hooks; using default hooks.")
        return {"hook_options": _fallback_hooks(topic, contradictions)}

    return {"hook_options": hooks}


def select_best_hook(state: DocuResearchState) -> dict[str, Any]:
    """Pick the highest-scoring hook."""
    hooks = state.get("hook_options", [])
    if not hooks:
        return {"selected_hook": None}
    best = max(hooks, key=lambda h: h.overall_score)
    return {"selected_hook": best}


def _fallback_draft_script(state: DocuResearchState, hook_text: str) -> DraftScript:
    """Deterministic rule-based script. Used for mock mode and as a live-mode fallback."""
    topic = state.get("topic", "Untitled topic")
    architecture = state.get("story_outline")
    verified = state.get("verified_claims", [])

    sections: list[ScriptSection] = []
    if architecture:
        for sec in architecture.sections:
            cited = [c.claim_id for c in verified if c.claim_id in sec.key_claim_ids] or [
                c.claim_id for c in verified[:1]
            ]
            narration = (
                f"[{sec.name}] {sec.purpose} "
                f"{'Supported by verified research. ' if cited else ''}"
                f"(citations: {', '.join(cited) if cited else 'none available'})"
            )
            sections.append(
                ScriptSection(
                    id=new_id("SS"),
                    heading=sec.name,
                    narration=narration,
                    cited_claim_ids=cited,
                    order=sec.order,
                )
            )

    full_text = "\n\n".join([hook_text, *(s.narration for s in sections)])
    return DraftScript(
        version=state.get("iteration_count", 0) + 1,
        title_suggestions=[f"{topic}: The Untold Story", f"Inside {topic}", f"{topic} Uncovered"],
        hook_text=hook_text,
        introduction=f"This documentary examines {topic} using verified, source-backed evidence.",
        sections=sections,
        conclusion=f"What happened with {topic} continues to shape what comes next.",
        cta=None,
        full_text=full_text,
        generated_at=_now(),
        model_name="mock" if state.get("mock_mode") else "fallback",
        prompt_version="v1",
    )


async def generate_script(state: DocuResearchState) -> dict[str, Any]:
    """Generate the documentary script from story architecture + verified claims + hook."""
    topic = state.get("topic", "Untitled topic")
    hook = state.get("selected_hook")
    hook_text = hook.text if hook else f"This is the story of {topic}."

    if state.get("mock_mode"):
        return {"draft_script": _fallback_draft_script(state, hook_text)}

    settings = get_settings()
    if not settings.has_openai:
        return {"draft_script": _fallback_draft_script(state, hook_text)}

    try:
        draft = await generate_script_llm(
            topic,
            language=state.get("language", "english"),
            tone=state.get("tone"),
            script_length=state.get("script_length", "10min"),
            target_audience=state.get("target_audience", "general audience"),
            story_architecture=state.get("story_outline"),
            claims=state.get("verified_claims", []),
            hook_text=hook_text,
            style_spec=state.get("script_template_spec"),
            version=state.get("iteration_count", 0) + 1,
            model_name=settings.default_model,
        )
    except Exception as exc:  # noqa: BLE001 - LLM/SDK failure must not abort the run
        logger.warning("script_generation_llm_failed", error=str(exc))
        return {
            "draft_script": _fallback_draft_script(state, hook_text),
            "warnings": [f"LLM script generation failed; used a minimal fallback script. ({exc})"],
        }

    if not draft.sections:
        logger.warning("script_generation_empty")
        return {
            "draft_script": _fallback_draft_script(state, hook_text),
            "warnings": ["LLM produced an empty script; used a minimal fallback."],
        }

    return {"draft_script": draft}


def _fallback_fact_check(draft: DraftScript | None, evidence: list[EvidenceMatrixEntry]) -> FactCheckResult:
    """Mechanical fact check using the evidence matrix. Used for mock mode and live-mode fallback."""
    evidence_by_claim = {e.claim_id: e for e in evidence}
    findings: list[FactCheckFinding] = []
    if draft:
        for sec in draft.sections:
            if not sec.cited_claim_ids:
                findings.append(
                    FactCheckFinding(
                        statement=sec.narration,
                        verdict=FactCheckVerdict.NEEDS_REVISION,
                        unsupported_inference=True,
                        notes="Section has no cited claims.",
                    )
                )
                continue
            for claim_id in sec.cited_claim_ids:
                entry = evidence_by_claim.get(claim_id)
                if entry is None:
                    verdict = FactCheckVerdict.FAIL
                elif entry.status == VerificationStatus.VERIFIED:
                    verdict = FactCheckVerdict.PASS
                elif entry.status in (VerificationStatus.PARTIALLY_VERIFIED, VerificationStatus.DISPUTED):
                    verdict = FactCheckVerdict.NEEDS_REVISION
                else:
                    verdict = FactCheckVerdict.FAIL
                findings.append(
                    FactCheckFinding(
                        claim_id=claim_id,
                        statement=sec.narration,
                        verdict=verdict,
                        notes=None if entry else "No evidence entry found for cited claim.",
                    )
                )

    overall = FactCheckVerdict.PASS
    if any(f.verdict == FactCheckVerdict.FAIL for f in findings):
        overall = FactCheckVerdict.FAIL
    elif any(f.verdict == FactCheckVerdict.NEEDS_REVISION for f in findings):
        overall = FactCheckVerdict.NEEDS_REVISION
    return FactCheckResult(findings=findings, overall_verdict=overall)


async def fact_check_script(state: DocuResearchState) -> dict[str, Any]:
    """Independently re-check the script's wording against the claims it cites."""
    draft = state.get("draft_script")

    if state.get("mock_mode"):
        return {"fact_check_results": _fallback_fact_check(draft, state.get("evidence", []))}

    settings = get_settings()
    if not settings.has_openai or not draft:
        return {"fact_check_results": _fallback_fact_check(draft, state.get("evidence", []))}

    claims_by_id = {c.claim_id: c for c in state.get("claims", [])}
    semaphore = asyncio.Semaphore(settings.max_concurrent_requests)

    async def _check_section(sec: ScriptSection) -> list[FactCheckFinding]:
        cited_claims = [claims_by_id[cid] for cid in sec.cited_claim_ids if cid in claims_by_id]
        async with semaphore:
            try:
                return await fact_check_section(sec.narration, cited_claims)
            except Exception as exc:  # noqa: BLE001 - one bad section must not abort the whole check
                logger.warning("script_fact_check_failed", section=sec.heading, error=str(exc))
                return [
                    FactCheckFinding(
                        statement=sec.narration,
                        verdict=FactCheckVerdict.NEEDS_REVISION,
                        notes=f"Fact-check failed: {exc}",
                    )
                ]

    results = await asyncio.gather(*(_check_section(sec) for sec in draft.sections))
    findings = [f for batch in results for f in batch]

    overall = FactCheckVerdict.PASS
    if any(f.verdict == FactCheckVerdict.FAIL for f in findings):
        overall = FactCheckVerdict.FAIL
    elif any(f.verdict == FactCheckVerdict.NEEDS_REVISION for f in findings):
        overall = FactCheckVerdict.NEEDS_REVISION

    logger.info("fact_check_script", mode="live", overall=overall.value, findings=len(findings))
    return {"fact_check_results": FactCheckResult(findings=findings, overall_verdict=overall)}


def citation_audit(state: DocuResearchState) -> dict[str, Any]:
    """Build the claim -> source citation map and audit it against the evidence matrix.

    Deterministic/structural - flags citations that don't resolve to a real
    source and known contradictions the script never acknowledges anywhere
    (spec section 35). Runs the same way in mock and live mode.
    """
    evidence_by_claim = {e.claim_id: e for e in state.get("evidence", [])}
    sources_by_id = {s.id: s for s in state.get("sources", [])}
    citation_map: dict[str, str] = {}
    warnings: list[str] = []

    draft = state.get("draft_script")
    cited_ids: set[str] = set()
    if draft:
        for sec in draft.sections:
            for claim_id in sec.cited_claim_ids:
                cited_ids.add(claim_id)
                entry = evidence_by_claim.get(claim_id)
                if not entry or not entry.supporting_source_ids:
                    warnings.append(f"Claim {claim_id} is cited in the script but has no resolvable source.")
                    continue
                source = sources_by_id.get(entry.supporting_source_ids[0])
                if source:
                    citation_map[claim_id] = source.id
                else:
                    warnings.append(
                        f"Claim {claim_id} cites source {entry.supporting_source_ids[0]}, "
                        "which is not in the collected source list."
                    )

    unacknowledged = [c for c in state.get("contradictions", []) if not (set(c.claim_ids) & cited_ids)]
    if unacknowledged:
        warnings.append(f"{len(unacknowledged)} known contradiction(s) are not reflected anywhere in the script.")

    return {"citation_map": citation_map, "warnings": warnings}


def quality_control(state: DocuResearchState) -> dict[str, Any]:
    """Compute the weighted quality score across all dimensions."""
    claims = state.get("claims", [])
    verified = state.get("verified_claims", [])
    sources = state.get("sources", [])
    fact_check = state.get("fact_check_results")
    draft = state.get("draft_script")
    citation_map = state.get("citation_map", {})

    research_depth = min(100.0, len(sources) * 20.0)
    source_quality = (
        round(sum(s.confidence_score for s in sources) / len(sources) * 100, 2) if sources else 0.0
    )
    claim_verification = (
        round(len(verified) / len(claims) * 100, 2) if claims else 0.0
    )
    narrative_quality = 80.0 if draft and draft.sections else 30.0
    selected_hook = state.get("selected_hook")
    hook_quality = round((selected_hook.overall_score if selected_hook else 0.0) * 100, 2)
    script_structure = 90.0 if draft and len(draft.sections) >= 3 else 50.0

    cited_slots = [cid for s in (draft.sections if draft else []) for cid in s.cited_claim_ids]
    matched_slots = sum(1 for cid in cited_slots if cid in citation_map)
    citation_completeness = round(matched_slots / len(cited_slots) * 100, 2) if cited_slots else 0.0

    if fact_check is None:
        factual_safety = 0.0
    else:
        fail_ratio = fact_check.fail_count / len(fact_check.findings) if fact_check.findings else 1.0
        factual_safety = round((1 - fail_ratio) * 100, 2)

    score = QualityScore(
        research_depth=research_depth,
        source_quality=source_quality,
        claim_verification=claim_verification,
        narrative_quality=narrative_quality,
        hook_quality=hook_quality,
        script_structure=script_structure,
        citation_completeness=citation_completeness,
        factual_safety=factual_safety,
    )
    logger.info("quality_control", overall=score.overall, factual_safety=factual_safety)
    return {"quality_score": score}


def _build_revision_feedback(state: DocuResearchState) -> list[str]:
    """Summarize what fact-checking/citation-auditing flagged, for the next draft to fix."""
    feedback: list[str] = []
    fact_check = state.get("fact_check_results")
    if fact_check:
        failing = [f for f in fact_check.findings if f.verdict != FactCheckVerdict.PASS]
        feedback.extend(f"Section citing {f.claim_id or 'an unlisted claim'}: {f.notes or f.verdict.value}" for f in failing[:8])

    quality = state.get("quality_score")
    if quality and quality.citation_completeness < QUALITY_CITATION_THRESHOLD:
        feedback.append(
            "Several cited claim IDs could not be resolved to a real source - only cite claim IDs "
            "that were actually provided in the claims list."
        )
    return feedback


async def final_revision(state: DocuResearchState) -> dict[str, Any]:
    """Revise the script when quality control fails. Increments the iteration counter.

    In live mode this actually regenerates the script with feedback on what
    failed - re-running fact_check_script on an unchanged draft would just
    reproduce the same failure and waste the whole iteration budget.
    """
    iteration = state.get("iteration_count", 0) + 1
    logger.info("final_revision", iteration=iteration)

    if state.get("mock_mode"):
        return {
            "iteration_count": iteration,
            "warnings": [f"Revision pass {iteration}: regenerating script for quality issues."],
        }

    settings = get_settings()
    if not settings.has_openai:
        return {
            "iteration_count": iteration,
            "warnings": [f"Revision pass {iteration}: no OPENAI_API_KEY - cannot regenerate the script."],
        }

    topic = state.get("topic", "Untitled topic")
    hook = state.get("selected_hook")
    hook_text = hook.text if hook else f"This is the story of {topic}."
    feedback = _build_revision_feedback(state)

    try:
        draft = await generate_script_llm(
            topic,
            language=state.get("language", "english"),
            tone=state.get("tone"),
            script_length=state.get("script_length", "10min"),
            target_audience=state.get("target_audience", "general audience"),
            story_architecture=state.get("story_outline"),
            claims=state.get("verified_claims", []),
            hook_text=hook_text,
            style_spec=state.get("script_template_spec"),
            feedback=feedback,
            version=iteration + 1,
            model_name=settings.default_model,
        )
    except Exception as exc:  # noqa: BLE001 - LLM/SDK failure must not abort the run
        logger.warning("final_revision_llm_failed", error=str(exc))
        return {
            "iteration_count": iteration,
            "warnings": [
                f"Revision pass {iteration} failed to regenerate the script ({exc}); keeping the previous draft."
            ],
        }

    return {
        "draft_script": draft,
        "iteration_count": iteration,
        "warnings": [f"Revision pass {iteration}: regenerated the script addressing {len(feedback)} flagged issue(s)."],
    }


def final_output(state: DocuResearchState) -> dict[str, Any]:
    """Assemble the final structured deliverable."""
    draft = state.get("draft_script")
    sources = state.get("sources", [])
    research_plan = state.get("research_plan")
    output = FinalOutput(
        research_id=state.get("research_id", "unknown"),
        topic=state.get("topic", "Untitled topic"),
        title_suggestions=draft.title_suggestions if draft else [],
        opening_hook=draft.hook_text if draft else "",
        script=draft,
        story_architecture=state.get("story_outline"),
        source_references=[
            {"id": s.id, "title": s.title, "url": str(s.url) if s.url else None} for s in sources
        ],
        claim_to_source_map=[
            CitationEntry(
                claim_id=claim_id,
                source_id=source_id,
                source_label=next((s.title for s in sources if s.id == source_id), source_id),
                confidence=next(
                    (e.confidence for e in state.get("evidence", []) if e.claim_id == claim_id), 0.0
                ),
            )
            for claim_id, source_id in state.get("citation_map", {}).items()
        ],
        unverified_claims=[c.text for c in state.get("unverified_claims", [])],
        contradictory_claims=state.get("contradictions", []),
        research_notes=[research_plan.notes] if research_plan and research_plan.notes else [],
        visual_suggestions=[
            VisualSuggestion(
                section_id=sec.id,
                section_name=sec.heading,
                broll_keywords=[state.get("topic", ""), sec.heading],
                archival_search_keywords=[f"{state.get('topic', '')} {sec.heading} archival footage"],
            )
            for sec in (draft.sections if draft else [])
        ],
        hook_options=state.get("hook_options", []),
        fact_check=state.get("fact_check_results"),
        quality_score=state.get("quality_score"),
        generated_at=_now(),
    )
    return {"final_output": output}
