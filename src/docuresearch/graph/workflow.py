"""Builds and compiles the DocuResearch AI LangGraph StateGraph.

See section 7 of the product spec for the canonical node sequence, and the
module docstring in graph/nodes.py for what each node currently does in
Phase 1 (mock pipeline).
"""

from __future__ import annotations

from collections.abc import Hashable
from functools import lru_cache

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from docuresearch.graph import nodes
from docuresearch.graph.routers import route_after_quality_control, route_after_verify_claims
from docuresearch.models.state import DocuResearchState


def _add_research_nodes(graph: StateGraph) -> None:
    """Nodes/edges for intake through the evidence matrix (spec sec. 7, first half).

    Self-contained: the adaptive research loop (`expand_weak_claims`) never
    loops back past `build_evidence_matrix`, so this half can be compiled and
    run as its own graph (Phase 7's `--research-only` / resumable-run split).
    """
    graph.add_node("intake_request", nodes.intake_request)
    graph.add_node("analyze_script_template", nodes.analyze_script_template)
    graph.add_node("create_research_plan", nodes.create_research_plan)
    graph.add_node("generate_search_queries", nodes.generate_search_queries)
    graph.add_node("parallel_source_discovery", nodes.parallel_source_discovery)
    graph.add_node("collect_sources", nodes.collect_sources)
    graph.add_node("extract_documents", nodes.extract_documents)
    graph.add_node("extract_claims", nodes.extract_claims)
    graph.add_node("rank_sources", nodes.rank_sources)
    graph.add_node("verify_claims", nodes.verify_claims)
    graph.add_node("expand_weak_claims", nodes.expand_weak_claims)
    graph.add_node("detect_contradictions", nodes.detect_contradictions)
    graph.add_node("build_evidence_matrix", nodes.build_evidence_matrix)

    graph.add_edge(START, "intake_request")
    graph.add_edge("intake_request", "analyze_script_template")
    graph.add_edge("analyze_script_template", "create_research_plan")
    graph.add_edge("create_research_plan", "generate_search_queries")
    graph.add_edge("generate_search_queries", "parallel_source_discovery")
    graph.add_edge("parallel_source_discovery", "collect_sources")
    graph.add_edge("collect_sources", "extract_documents")
    graph.add_edge("extract_documents", "extract_claims")
    graph.add_edge("extract_claims", "rank_sources")
    graph.add_edge("rank_sources", "verify_claims")

    research_loop_routes: dict[Hashable, str] = {
        "expand_weak_claims": "expand_weak_claims",
        "detect_contradictions": "detect_contradictions",
    }
    graph.add_conditional_edges("verify_claims", route_after_verify_claims, research_loop_routes)
    graph.add_conditional_edges("expand_weak_claims", route_after_verify_claims, research_loop_routes)

    graph.add_edge("detect_contradictions", "build_evidence_matrix")


def _add_script_nodes(graph: StateGraph) -> None:
    """Nodes/edges for story architecture through final output (spec sec. 7, second half)."""
    graph.add_node("build_story_architecture", nodes.build_story_architecture)
    graph.add_node("generate_hooks", nodes.generate_hooks)
    graph.add_node("select_best_hook", nodes.select_best_hook)
    graph.add_node("generate_script", nodes.generate_script)
    graph.add_node("fact_check_script", nodes.fact_check_script)
    graph.add_node("citation_audit", nodes.citation_audit)
    graph.add_node("quality_control", nodes.quality_control)
    graph.add_node("final_revision", nodes.final_revision)
    graph.add_node("final_output", nodes.final_output)

    graph.add_edge("build_story_architecture", "generate_hooks")
    graph.add_edge("generate_hooks", "select_best_hook")
    graph.add_edge("select_best_hook", "generate_script")
    graph.add_edge("generate_script", "fact_check_script")
    graph.add_edge("fact_check_script", "citation_audit")
    graph.add_edge("citation_audit", "quality_control")

    graph.add_conditional_edges(
        "quality_control",
        route_after_quality_control,
        {"final_revision": "final_revision", "final_output": "final_output"},
    )
    graph.add_edge("final_revision", "fact_check_script")
    graph.add_edge("final_output", END)


def build_graph() -> CompiledStateGraph:
    """Assemble every node and edge for the full pipeline, then compile the graph."""
    graph = StateGraph(DocuResearchState)
    _add_research_nodes(graph)
    _add_script_nodes(graph)
    graph.add_edge("build_evidence_matrix", "build_story_architecture")
    return graph.compile()


def build_research_graph() -> CompiledStateGraph:
    """Intake through the evidence matrix only - Phase 7's `--research-only` graph."""
    graph = StateGraph(DocuResearchState)
    _add_research_nodes(graph)
    graph.add_edge("build_evidence_matrix", END)
    return graph.compile()


def build_script_graph() -> CompiledStateGraph:
    """Story architecture through final output only - Phase 7's `--script-only` graph.

    Takes an already-research-complete state as its initial input.
    """
    graph = StateGraph(DocuResearchState)
    _add_script_nodes(graph)
    graph.add_edge(START, "build_story_architecture")
    return graph.compile()


@lru_cache
def get_compiled_graph() -> CompiledStateGraph:
    """Cached compiled full-pipeline graph - compile once per process."""
    return build_graph()


@lru_cache
def get_research_graph() -> CompiledStateGraph:
    """Cached compiled research-phase graph - compile once per process."""
    return build_research_graph()


@lru_cache
def get_script_graph() -> CompiledStateGraph:
    """Cached compiled script-phase graph - compile once per process."""
    return build_script_graph()


async def run_research(initial_state: DocuResearchState) -> DocuResearchState:
    """Run the full compiled graph to completion and return the final state."""
    graph = get_compiled_graph()
    result = await graph.ainvoke(initial_state)
    return result  # type: ignore[return-value]


async def run_research_phase(initial_state: DocuResearchState) -> DocuResearchState:
    """Run intake through the evidence matrix only, and return that state."""
    graph = get_research_graph()
    result = await graph.ainvoke(initial_state)
    return result  # type: ignore[return-value]


async def run_script_phase(state: DocuResearchState) -> DocuResearchState:
    """Run story architecture through final output, given an already-researched state."""
    graph = get_script_graph()
    result = await graph.ainvoke(state)
    return result  # type: ignore[return-value]
