"""Tests for source scoring and non-destructive deduplication (spec sections 10, 16)."""

from __future__ import annotations

from docuresearch.models.sources import Source, SourceType
from docuresearch.tools.source_ranker import (
    WIKIPEDIA_AUTHORITY_CAP,
    deduplicate_and_score,
    rank_sources,
    score_relevance,
    score_source,
)


def _source(id_: str, **overrides: object) -> Source:
    defaults: dict[str, object] = {
        "id": id_,
        "title": f"Title {id_}",
        "source_type": SourceType.NEWS,
        "domain": f"{id_.lower()}.example.com",
    }
    defaults.update(overrides)
    return Source(**defaults)  # type: ignore[arg-type]


def test_wikipedia_authority_is_capped() -> None:
    source = _source("S1", source_type=SourceType.WIKIPEDIA, primary_source=True)
    scored = score_source(source)
    assert scored.authority_score <= WIKIPEDIA_AUTHORITY_CAP


def test_government_report_gets_high_authority() -> None:
    source = _source("S1", source_type=SourceType.GOVERNMENT_REPORT)
    scored = score_source(source)
    assert scored.authority_score >= 0.85


def test_company_official_gets_low_authority() -> None:
    source = _source("S1", source_type=SourceType.COMPANY_OFFICIAL)
    scored = score_source(source)
    assert scored.authority_score <= 0.5


def test_primary_source_boosts_authority() -> None:
    base = _source("S1", source_type=SourceType.NEWS)
    boosted = _source("S2", source_type=SourceType.NEWS, primary_source=True)
    assert score_source(boosted).authority_score > score_source(base).authority_score


def test_transparency_signal_raises_credibility() -> None:
    anonymous = _source("S1", source_type=SourceType.NEWS)
    attributed = _source("S2", source_type=SourceType.NEWS, author="Jane Doe")
    assert score_source(attributed).credibility_score > score_source(anonymous).credibility_score


def test_deduplicate_merges_identical_content_across_domains() -> None:
    a = _source("S1", domain="siteone.com", text="Identical press release text.")
    b = _source("S2", domain="sitetwo.com", text="Identical press release text.")

    result = deduplicate_and_score([a, b])
    assert len(result) == 2  # never drops a source - claims may cite either id

    by_id = {s.id: s for s in result}
    corroboration_counts = {s.metadata.get("corroboration_count") for s in result}
    assert corroboration_counts == {2}

    duplicate_flags = [s.metadata.get("duplicate_content", False) for s in result]
    assert duplicate_flags.count(True) == 1  # exactly one copy is marked as the redundant duplicate
    # the marked duplicate must be scored strictly lower than the kept representative
    representative_id, duplicate_id = ("S1", "S2") if not by_id["S1"].metadata.get("duplicate_content") else ("S2", "S1")
    assert by_id[duplicate_id].authority_score < by_id[representative_id].authority_score


def test_deduplicate_merges_near_identical_titles_across_domains() -> None:
    a = _source("S1", domain="siteone.com", title="Company X Enters Market in 2016")
    b = _source("S2", domain="sitetwo.com", title="Company X Enters Market in 2016")

    result = deduplicate_and_score([a, b])
    assert len(result) == 2
    assert {s.metadata.get("corroboration_count") for s in result} == {2}


def test_deduplicate_does_not_merge_same_domain_different_content() -> None:
    a = _source("S1", domain="siteone.com", title="Article A", text="Text A")
    b = _source("S2", domain="siteone.com", title="Article B", text="Text B")

    result = deduplicate_and_score([a, b])
    assert all(s.metadata.get("corroboration_count") == 1 for s in result)
    assert all(not s.metadata.get("duplicate_content", False) for s in result)


def test_deduplicate_does_not_merge_unrelated_sources() -> None:
    a = _source("S1", domain="siteone.com", title="Completely unrelated headline")
    b = _source("S2", domain="sitetwo.com", title="Another distinct topic entirely")

    result = deduplicate_and_score([a, b])
    assert all(s.metadata.get("corroboration_count") == 1 for s in result)


def test_rank_sources_sorts_by_confidence_descending() -> None:
    low = _source("S1", source_type=SourceType.COMPANY_OFFICIAL)
    high = _source("S2", source_type=SourceType.GOVERNMENT_REPORT)

    ranked = rank_sources([low, high])
    assert [s.id for s in ranked] == ["S2", "S1"]


def test_score_relevance_scores_matching_title_highly() -> None:
    on_topic = _source("S1", title="The Indian Telecom Revolution: A History")
    score = score_relevance(on_topic, "The Indian Telecom Revolution")
    assert score > 0.5


def test_score_relevance_scores_unrelated_title_near_zero() -> None:
    off_topic = _source("S1", title="Monopoly (board game)")
    score = score_relevance(off_topic, "The Indian Telecom Revolution")
    assert score == 0.0


def test_score_relevance_partial_overlap_scores_between_the_extremes() -> None:
    partial = _source("S1", title="Indian National Congress")
    on_topic = _source("S2", title="The Indian Telecom Revolution: A History")
    off_topic = _source("S3", title="Monopoly (board game)")

    partial_score = score_relevance(partial, "The Indian Telecom Revolution")
    assert score_relevance(off_topic, "The Indian Telecom Revolution") < partial_score < score_relevance(
        on_topic, "The Indian Telecom Revolution"
    )


def test_score_relevance_falls_back_to_summary_when_title_is_uninformative() -> None:
    url_titled = _source(
        "S1",
        title="http://www.itu.int/ITU-D/ICTEYE/Reporting/DynamicReportWizard.aspx",
        summary="Statistics on the Indian telecom revolution and mobile penetration.",
    )
    score = score_relevance(url_titled, "The Indian Telecom Revolution")
    assert score > 0.0


def test_score_source_without_topic_leaves_relevance_untouched() -> None:
    source = _source("S1", relevance_score=0.42)
    scored = score_source(source)
    assert scored.relevance_score == 0.42


def test_score_source_with_topic_recomputes_relevance() -> None:
    off_topic = _source("S1", title="Monopoly (board game)", relevance_score=0.6)
    scored = score_source(off_topic, topic="The Indian Telecom Revolution")
    assert scored.relevance_score == 0.0


def test_rank_sources_with_topic_demotes_irrelevant_hits() -> None:
    on_topic = _source(
        "S1", source_type=SourceType.WIKIPEDIA, title="Telecommunications in India", relevance_score=0.6
    )
    off_topic = _source("S2", source_type=SourceType.WIKIPEDIA, title="Monopoly (board game)", relevance_score=0.6)

    ranked = rank_sources([off_topic, on_topic], topic="The Indian Telecom Revolution")
    assert [s.id for s in ranked] == ["S1", "S2"]
