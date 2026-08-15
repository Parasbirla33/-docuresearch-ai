"""Source scoring and deduplication (spec sections 10, 16).

Scoring never blindly trusts a single signal - domain/source-type reputation,
primary-source status, transparency (identified author/publisher), and
corroboration all contribute. Wikipedia's authority is explicitly capped
rather than ever treated as a final authority.

Deduplication is non-destructive: near-duplicate/syndicated sources are
scored down and tagged rather than removed, since claims may already cite
their IDs by the time ranking runs (spec section 7's pipeline order runs
`rank_sources` after `extract_claims`). This still satisfies "don't count
copied articles as independent corroboration" - only the best copy in a
duplicate cluster gets the corroboration bonus.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from docuresearch.models.sources import Source, SourceType
from docuresearch.utils.hashing import content_hash
from docuresearch.utils.logging import get_logger

logger = get_logger(__name__)

WIKIPEDIA_AUTHORITY_CAP = 0.55
DUPLICATE_SCORE_PENALTY = 0.3
TITLE_SIMILARITY_DEDUP_THRESHOLD = 0.9

# Common words carry no topical signal and would inflate overlap for almost
# any pair of texts (e.g. "the", "of", "how" match everything).
_STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "for", "to", "and", "or", "is", "are",
    "was", "were", "how", "what", "why", "which", "who", "did", "does", "do",
    "that", "this", "these", "those", "its", "as", "with", "by", "from", "at",
    "be", "been", "into", "about", "than", "not", "but",
}
_WORD_RE = re.compile(r"[a-z0-9]+")
# Minimum shared-prefix length to count two words as the same keyword (see
# _fuzzy_contains) - long enough to avoid short generic-word false positives
# (e.g. "art"/"article"), short enough to catch real variants like
# telecom/telecommunications or india/indian.
_MIN_PREFIX_MATCH = 4


def _keywords(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS and len(w) > 2}


def _fuzzy_contains(word: str, candidates: set[str]) -> bool:
    """True if `word` shares a >=_MIN_PREFIX_MATCH-char prefix with any candidate.

    Plain exact-token matching misses obviously-on-topic titles that just use
    a different word form than the topic phrase - "Telecommunications in
    India" shares zero exact tokens with "Indian Telecom Revolution" despite
    clearly being about it (telecom/telecommunications, indian/india).
    """
    for candidate in candidates:
        shorter, longer = (word, candidate) if len(word) <= len(candidate) else (candidate, word)
        if len(shorter) >= _MIN_PREFIX_MATCH and longer.startswith(shorter):
            return True
    return False


def _overlap_ratio(topic_words: set[str], candidate_words: set[str]) -> float:
    if not topic_words:
        return 0.0
    matches = sum(1 for w in topic_words if _fuzzy_contains(w, candidate_words))
    return matches / len(topic_words)


def score_relevance(source: Source, topic: str) -> float:
    """Deterministic topic-overlap heuristic, weighting title over summary/body.

    Discovery adapters (web search, Wikipedia, ...) otherwise assign every
    hit from a given source type the same flat relevance regardless of
    whether it's actually on-topic - a Wikipedia search for a multi-word
    question routinely returns tangential pages, and this is what keeps
    those from being scored/ranked as if they were as good as a real match.
    Not a substitute for claim extraction actually finding usable evidence -
    just stops irrelevant hits from looking equally trustworthy.
    """
    topic_words = _keywords(topic)
    if not topic_words:
        return 0.5

    title_overlap = _overlap_ratio(topic_words, _keywords(source.title))
    body_sample = source.summary or (source.text[:500] if source.text else "")
    body_overlap = _overlap_ratio(topic_words, _keywords(body_sample))

    return round(min(1.0, 0.7 * title_overlap + 0.3 * body_overlap), 4)

_HIGH_AUTHORITY_TLDS = (".gov", ".gov.in", ".edu", ".int")
_HIGH_AUTHORITY_TYPES = {
    SourceType.GOVERNMENT,
    SourceType.GOVERNMENT_REPORT,
    SourceType.REGULATORY_BODY,
    SourceType.COURT_DOCUMENT,
    SourceType.INTERNATIONAL_ORGANIZATION,
    SourceType.ACADEMIC_PAPER,
}
_MID_AUTHORITY_TYPES = {SourceType.NEWS, SourceType.RESEARCH_ORGANIZATION, SourceType.NGO_REPORT}
_LOW_AUTHORITY_TYPES = {SourceType.COMPANY_OFFICIAL, SourceType.BOOK_PUBLICATION, SourceType.PUBLIC_DATASET}


def _domain_authority(domain: str | None, source_type: SourceType) -> float:
    if source_type in _HIGH_AUTHORITY_TYPES:
        return 0.9
    if domain and any(domain.endswith(tld) for tld in _HIGH_AUTHORITY_TLDS):
        return 0.85
    if source_type in _MID_AUTHORITY_TYPES:
        return 0.6
    if source_type in _LOW_AUTHORITY_TYPES:
        return 0.4
    return 0.5


def score_source(
    source: Source,
    *,
    topic: str | None = None,
    corroboration_count: int = 1,
    is_duplicate: bool = False,
) -> Source:
    """Recompute authority/credibility (and, if `topic` is given, relevance) for one source.

    `topic` is optional and defaults to leaving `relevance_score` untouched -
    callers that only care about authority/dedup (or don't have a topic in
    scope) get the same behavior as before.
    """
    authority = _domain_authority(source.domain, source.source_type)

    if source.primary_source:
        authority = min(1.0, authority + 0.15)
    if source.source_type == SourceType.WIKIPEDIA:
        # Applied last, as a hard ceiling - no other signal may push Wikipedia
        # past this cap. Never treat it as a final authority (spec section 10).
        authority = min(authority, WIKIPEDIA_AUTHORITY_CAP)

    transparency = 0.1 if (source.author or source.publisher) else 0.0
    corroboration_bonus = 0.0 if is_duplicate else min(0.15, 0.05 * max(0, corroboration_count - 1))
    credibility = min(1.0, authority * 0.7 + transparency + corroboration_bonus + 0.1)

    metadata = {**source.metadata, "corroboration_count": corroboration_count}
    if is_duplicate:
        authority *= DUPLICATE_SCORE_PENALTY
        credibility *= DUPLICATE_SCORE_PENALTY
        metadata["duplicate_content"] = True

    updates: dict[str, object] = {
        "authority_score": round(authority, 4),
        "credibility_score": round(credibility, 4),
        "metadata": metadata,
    }
    if topic:
        updates["relevance_score"] = score_relevance(source, topic)

    return source.model_copy(update=updates)


def _normalize_title(title: str) -> str:
    return " ".join(title.lower().split())


def _is_duplicate_pair(a: Source, b: Source) -> bool:
    if a.text and b.text and content_hash(a.text) == content_hash(b.text):
        return True
    if a.domain == b.domain:
        return False  # same-domain, different content -> not a syndication duplicate
    similarity = SequenceMatcher(None, _normalize_title(a.title), _normalize_title(b.title)).ratio()
    return similarity >= TITLE_SIMILARITY_DEDUP_THRESHOLD


def deduplicate_and_score(sources: list[Source], topic: str | None = None) -> list[Source]:
    """Cluster near-duplicate/syndicated sources and rescore each source.

    Keeps every source (never drops one - a claim may already cite its ID);
    only the best-scoring member of each cluster gets full corroboration credit.
    `topic`, if given, also recomputes each source's `relevance_score`.
    """
    n = len(sources)
    cluster_of: dict[int, int] = {}
    clusters: list[list[int]] = []

    for i in range(n):
        if i in cluster_of:
            continue
        cluster = [i]
        cluster_of[i] = len(clusters)
        for j in range(i + 1, n):
            if j in cluster_of:
                continue
            if _is_duplicate_pair(sources[i], sources[j]):
                cluster.append(j)
                cluster_of[j] = len(clusters)
        clusters.append(cluster)

    result: list[Source] = list(sources)
    duplicate_clusters = sum(1 for c in clusters if len(c) > 1)
    for cluster in clusters:
        corroboration = len(cluster)
        best_idx = max(cluster, key=lambda idx: sources[idx].confidence_score)
        for idx in cluster:
            result[idx] = score_source(
                sources[idx],
                topic=topic,
                corroboration_count=corroboration,
                is_duplicate=(idx != best_idx and corroboration > 1),
            )

    logger.info("deduplicate_and_score", sources=n, duplicate_clusters=duplicate_clusters)
    return result


def rank_sources(sources: list[Source], topic: str | None = None) -> list[Source]:
    """Deduplicate, rescore, and sort sources by confidence (spec section 10).

    `topic`, if given, also recomputes each source's `relevance_score`
    against it (see `score_relevance`) before sorting.
    """
    scored = deduplicate_and_score(sources, topic)
    return sorted(scored, key=lambda s: s.confidence_score, reverse=True)
