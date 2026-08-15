"""DocuResearch AI command-line interface.

`--mock` and `--topic <real topic>` run the full pipeline end to end.
`--research-only` stops after the (expensive, LLM/search-backed) research
phase and persists a checkpoint; `--resume RESEARCH_ID [--script-only]`
reloads that checkpoint and (re-)runs just the story/script phase from it,
without redoing research. See `storage/repository.py` (Phase 7).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from docuresearch.config.settings import get_settings
from docuresearch.graph.workflow import run_research_phase, run_script_phase
from docuresearch.models.research import ResearchDepth
from docuresearch.models.state import DocuResearchState, new_initial_state
from docuresearch.storage.models import RunStatus
from docuresearch.storage.repository import ResearchRunRepository, RunSummary
from docuresearch.utils.hashing import new_id
from docuresearch.utils.logging import configure_logging, configure_stdio_encoding, get_logger

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="docuresearch",
        description="DocuResearch AI - documentary research & script generation agent",
    )
    parser.add_argument("--topic", type=str, default=None, help="Documentary topic")
    parser.add_argument("--language", type=str, default="english", help="Script language")
    parser.add_argument(
        "--depth",
        type=str,
        choices=[d.value for d in ResearchDepth],
        default=ResearchDepth.STANDARD.value,
        help="Research depth",
    )
    parser.add_argument("--length", type=str, default="10min", help="Target script length")
    parser.add_argument("--audience", type=str, default="general audience", help="Target audience")
    parser.add_argument("--template", type=str, default=None, help="Path to a user script template file")
    parser.add_argument("--tone", type=str, default=None, help="Optional tone/style instructions")
    parser.add_argument("--date-range", type=str, default=None, help="Optional date range filter")
    parser.add_argument("--geo", type=str, default=None, help="Optional geographic focus")

    parser.add_argument("--mock", action="store_true", help="Run with mock research data, no external APIs")
    parser.add_argument(
        "--research-only",
        action="store_true",
        help="Run research only (through the evidence matrix), persist, then stop",
    )
    parser.add_argument(
        "--script-only",
        action="store_true",
        help="With --resume: (re-)run only the story/script phase from a persisted checkpoint",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="RESEARCH_ID",
        help="Resume a persisted research run by id (see --list-runs)",
    )
    parser.add_argument(
        "--list-runs", action="store_true", help="List persisted research runs and exit"
    )
    parser.add_argument("--output", type=str, default=None, help="Path to write the JSON result")
    parser.add_argument("--verbose", action="store_true", help="Verbose (DEBUG) logging")
    return parser


def _print_report(result: DocuResearchState, *, research_only: bool = False) -> None:
    final = result.get("final_output")
    quality = result.get("quality_score")
    fact_check = result.get("fact_check_results")

    print("\n" + "=" * 78)
    print("DOCUMENTARY RESEARCH REPORT")
    print("=" * 78)
    print(f"Research ID: {result.get('research_id')}")
    print(f"Topic: {result.get('topic')}")
    print(f"Research depth: {result.get('research_depth')}")
    print(f"Sources analyzed: {len(result.get('sources', []))}")
    print(f"Claims extracted: {len(result.get('claims', []))}")
    print(f"Verified claims: {len(result.get('verified_claims', []))}")
    print(f"Unverified claims: {len(result.get('unverified_claims', []))}")
    print(f"Contradictions detected: {len(result.get('contradictions', []))}")

    warnings = result.get("warnings", [])
    if warnings:
        print("\n--- WARNINGS ---")
        for w in warnings:
            print(f"  ! {w}")

    plan = result.get("research_plan")
    if plan:
        print("\n--- RESEARCH PLAN ---")
        for q in plan.key_questions:
            print(f"  - {q.question}")

    print("\n--- SOURCES ---")
    for s in result.get("sources", []):
        print(f"  [{s.id}] ({s.source_type.value}, score={s.confidence_score}) {s.title}")

    if not research_only:
        print("\n--- HOOK OPTIONS ---")
        selected_hook = result.get("selected_hook")
        for h in result.get("hook_options", []):
            marker = "* " if selected_hook and h.id == selected_hook.id else "  "
            print(f"{marker}[{h.hook_type.value}] {h.text} (score={h.overall_score})")

    draft = result.get("draft_script")
    if draft:
        print("\n--- FINAL SCRIPT ---")
        print(f"Titles: {', '.join(draft.title_suggestions)}")
        print(f"\nHOOK: {draft.hook_text}")
        print(f"\nINTRO: {draft.introduction}")
        for sec in draft.sections:
            print(f"\n[{sec.heading}]\n{sec.narration}")
        print(f"\nCONCLUSION: {draft.conclusion}")

    if fact_check:
        print("\n--- FACT CHECK REPORT ---")
        print(f"Overall verdict: {fact_check.overall_verdict.value}")
        for f in fact_check.findings:
            print(f"  - [{f.verdict.value}] claim={f.claim_id} {f.notes or ''}")

    if not research_only:
        print("\n--- SOURCE / CITATION MAP ---")
        for claim_id, source_id in result.get("citation_map", {}).items():
            print(f"  {claim_id} -> {source_id}")

    if final:
        print("\n--- VISUAL SUGGESTIONS ---")
        for v in final.visual_suggestions:
            print(f"  [{v.section_name}] B-roll: {', '.join(v.broll_keywords)}")
            print(f"    Archival keywords: {', '.join(v.archival_search_keywords)}")

    if quality:
        print("\n--- QUALITY SCORE ---")
        print(f"  research_depth:         {quality.research_depth}")
        print(f"  source_quality:         {quality.source_quality}")
        print(f"  claim_verification:     {quality.claim_verification}")
        print(f"  narrative_quality:      {quality.narrative_quality}")
        print(f"  hook_quality:           {quality.hook_quality}")
        print(f"  script_structure:       {quality.script_structure}")
        print(f"  citation_completeness:  {quality.citation_completeness}")
        print(f"  factual_safety:         {quality.factual_safety}")
        print(f"  OVERALL:                {quality.overall}")

    if research_only:
        print(f"\nResume with: --resume {result.get('research_id')} --script-only")

    print("\n" + "=" * 78 + "\n")


def _result_to_json(result: DocuResearchState) -> str:
    def default(obj: object) -> object:
        if hasattr(obj, "model_dump"):
            return obj.model_dump(mode="json")
        return str(obj)

    serializable = {k: v for k, v in result.items() if k != "errors" or v}
    return json.dumps(serializable, default=default, indent=2)


def _print_runs(runs: list[RunSummary]) -> None:
    print("\n" + "=" * 78)
    print("PERSISTED RESEARCH RUNS")
    print("=" * 78)
    if not runs:
        print("(none)")
    for run in runs:
        mock_tag = " [MOCK]" if run.mock_mode else ""
        print(f"  [{run.status}]{mock_tag} {run.research_id}  {run.updated_at.isoformat()}  {run.topic}")
    print("\n" + "=" * 78 + "\n")


def main(argv: list[str] | None = None) -> int:
    configure_stdio_encoding()

    parser = build_parser()
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(level="DEBUG" if args.verbose else settings.log_level)

    repository = ResearchRunRepository()

    if args.list_runs:
        _print_runs(repository.list_runs())
        return 0

    if args.research_only and args.resume:
        parser.error("--research-only starts a fresh run; use --resume on its own (with --script-only) instead")

    if not args.mock and not args.topic and not args.resume:
        parser.error("--topic is required unless --mock or --resume is used")

    if args.resume:
        state = repository.load(args.resume)
        if state is None:
            print(f"No persisted run found for research id '{args.resume}'.", file=sys.stderr)
            return 1

        status = repository.get_status(args.resume)
        if args.script_only or status == RunStatus.RESEARCH_COMPLETE.value:
            result = asyncio.run(run_script_phase(state))
            repository.save_checkpoint(result, status=RunStatus.COMPLETE)
        else:
            result = state

        if result.get("errors"):
            for err in result["errors"]:
                logger.error("pipeline_error", error=err)

        _print_report(result)

        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(_result_to_json(result), encoding="utf-8")
            print(f"Wrote JSON result to {output_path}")

        return 0

    topic = args.topic or "The Rise of a Modern Industry (mock topic)"

    initial_state = new_initial_state(
        research_id=new_id("RUN"),
        topic=topic,
        language=args.language,
        target_audience=args.audience,
        script_length=args.length,
        script_template_path=args.template,
        tone=args.tone,
        date_range=args.date_range,
        geographic_focus=args.geo,
        research_depth=ResearchDepth(args.depth),
        mock_mode=args.mock,
    )

    research_state = asyncio.run(run_research_phase(initial_state))
    repository.save_checkpoint(research_state, status=RunStatus.RESEARCH_COMPLETE)

    if args.research_only:
        if research_state.get("errors"):
            for err in research_state["errors"]:
                logger.error("pipeline_error", error=err)
        _print_report(research_state, research_only=True)
        return 0

    result = asyncio.run(run_script_phase(research_state))
    repository.save_checkpoint(result, status=RunStatus.COMPLETE)

    if result.get("errors"):
        for err in result["errors"]:
            logger.error("pipeline_error", error=err)

    _print_report(result)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(_result_to_json(result), encoding="utf-8")
        print(f"Wrote JSON result to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
