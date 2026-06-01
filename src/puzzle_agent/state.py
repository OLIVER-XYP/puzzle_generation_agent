"""LangGraph shared state."""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Set, TypedDict


class GraphState(TypedDict, total=False):
    # --- Studio / natural language input ---
    user_query: str                      # e.g. "给规则4生成5道题" — NL input
    rules: List[str]                     # e.g. ["1","5","10"] — override config
    count: int                           # e.g. 10 — override config

    config: Dict[str, Any]

    # job queue produced by the dispatcher
    jobs: List[Dict[str, Any]]          # [{rule_id, count, target}]
    job_cursor: int
    current_rule_id: Optional[str]
    target_params: Dict[str, Any]       # calibrated structural target for current rule

    # difficulty sampling -> synthesizer
    difficulty_plan: List[Any]          # per-slot difficulty values (replicates orig dist)
    current_slot: Any                   # the planned difficulty value for the current slot
    sampled_params: Dict[str, Any]
    rng_state: int                      # incrementing seed offset for determinism

    # current candidate flowing through synth -> solve -> verify
    candidate: Optional[Dict[str, Any]]   # StructuredProblem.to_dict()
    solve_result: Optional[Dict[str, Any]]
    verdict: Optional[Dict[str, Any]]     # {accepted: bool, reasons: [...]}

    # per-rule progress
    accepted_count: int                 # accepted for current rule
    item_retries: int                   # retries for the current candidate slot

    # dedup stores
    eval_hashes: List[str]              # from the original eval set (serialisable)
    accepted_hashes: List[str]         # synthesized-so-far

    # natural language processing
    _nl_result: Any                     # RewriteResult from query rewriter (internal)

    # outputs / logs
    accepted_records: List[Dict[str, Any]]
    rejected_log: List[Dict[str, Any]]
    stats: Dict[str, Any]
    output_path: str                    # set by save_output node
