"""LangGraph V2 with checkpointing, interrupt, and subgraph splitting (P1).

Improvements over V1:
  1. SqliteSaver checkpoint — resume interrupted runs
  2. interrupt_before — pause before data_preprocessor for manual review
  3. Subgraph per rule category — word/math/spatial run independently
"""
from __future__ import annotations
import json
import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
try:
    from langgraph.checkpoint.sqlite import SqliteSaver
    HAS_SQLITE_CHECKPOINT = True
except ImportError:
    HAS_SQLITE_CHECKPOINT = False

from .state import GraphState


# ======================================================================
# Checkpoint factory
# ======================================================================

def create_checkpointer(storage_path: str = "data/checkpoints.db") -> Any:
    """Create checkpoint store. Uses SQLite if available, else in-memory."""
    if HAS_SQLITE_CHECKPOINT:
        os.makedirs(os.path.dirname(storage_path), exist_ok=True)
        return SqliteSaver.from_conn_string(storage_path)
    else:
        print("[graph_v2] SqliteSaver not available, using MemorySaver")
        return MemorySaver()


# ======================================================================
# Subgraph definitions — per rule category
# ======================================================================

RULE_CATEGORIES = {
    "word": ["1","2","3","4","5","6","7","8","24"],
    "math": ["9","10","11","12","13","14","15","16","17","25"],
    "spatial": ["18","19","20","21","22","23"],
}


def _build_subgraph(category: str, rule_ids: List[str],
                    make_nodes_fn) -> StateGraph:
    """Build a subgraph for a specific rule category."""
    n = make_nodes_fn(category, rule_ids)

    g = StateGraph(GraphState)
    g.add_node("dispatcher", n["dispatcher"])
    g.add_node("synthesizer", n["synthesizer"])
    g.add_node("crosscheck", n["crosscheck"])
    g.add_node("verification", n["verification"])
    g.add_node("data_preprocessor", n["data_preprocessor"])

    g.add_edge(START, "dispatcher")
    g.add_conditional_edges("dispatcher", n["route_dispatch"],
                            {"synthesizer": "synthesizer", END: END})
    g.add_edge("synthesizer", "crosscheck")
    g.add_edge("crosscheck", "verification")
    g.add_conditional_edges("verification", n["route_verify"],
                            {"data_preprocessor": "data_preprocessor",
                             "synthesizer": "synthesizer",
                             "dispatcher": "dispatcher"})
    g.add_conditional_edges("data_preprocessor", n["route_after_accept"],
                            {"dispatcher": "dispatcher",
                             "synthesizer": "synthesizer"})
    return g


# ======================================================================
# Orchestrator — runs subgraphs and merges results
# ======================================================================

class OrchestratorGraph:
    """Top-level orchestrator that routes to per-category subgraphs."""

    def __init__(self, cfg: Dict[str, Any], ctx):
        self.cfg = cfg
        self.ctx = ctx
        self.checkpointer = create_checkpointer()
        self.subgraphs: Dict[str, Any] = {}

    def build(self) -> StateGraph:
        """Build the orchestrator graph with subgraph nodes."""
        g = StateGraph(GraphState)

        # Determine which subgraphs are needed based on requested rules
        requested = set(self.cfg.get("run", {}).get("rules", []))
        if not requested:
            requested = set(RULE_CATEGORIES["word"] + RULE_CATEGORIES["math"] + RULE_CATEGORIES["spatial"])

        for cat, rids in RULE_CATEGORIES.items():
            active = [r for r in rids if r in requested]
            if active:
                sub = _build_subgraph(cat, active, self._make_nodes_for_category)
                compiled = sub.compile(checkpointer=self.checkpointer,
                                       interrupt_before=["data_preprocessor"])
                self.subgraphs[cat] = compiled
                # Add subgraph node to orchestrator
                def make_node(cat=cat, comp=compiled):
                    def _run(state):
                        result = comp.invoke(state, config={"recursion_limit": 10000})
                        # Merge results
                        if "accepted_records" in result:
                            existing = state.get("accepted_records", [])
                            state["accepted_records"] = existing + result["accepted_records"]
                        return state
                    return _run
                g.add_node(f"sub_{cat}", make_node())

        # Routing: visit each active subgraph in sequence
        active_cats = list(self.subgraphs.keys())
        if not active_cats:
            g.add_node("dummy", lambda s: s)
            g.add_edge(START, "dummy")
            g.add_edge("dummy", END)
        else:
            # Sequential routing through subgraphs
            g.add_edge(START, f"sub_{active_cats[0]}")
            for i in range(len(active_cats) - 1):
                g.add_edge(f"sub_{active_cats[i]}", f"sub_{active_cats[i+1]}")
            g.add_edge(f"sub_{active_cats[-1]}", END)

        return g.compile(checkpointer=self.checkpointer)

    def _make_nodes_for_category(self, category: str, rule_ids: List[str]):
        """Create LLM generation nodes for a category subgraph."""
        from .llm_gen import LlmGen
        llm = LlmGen(self.cfg)

        def dispatcher(state):
            if "jobs" not in state:
                jobs = [{"rule_id": rid, "count": self.cfg["run"]["count_per_rule"]}
                        for rid in rule_ids if rid in self.ctx.rules]
                state.update(jobs=jobs, job_cursor=0, accepted_hashes=[],
                            accepted_records=state.get("accepted_records", []),
                            stats={})
            else:
                state["job_cursor"] = state["job_cursor"] + 1
            return state

        def route_dispatch(state):
            return "synthesizer" if state["job_cursor"] < len(state["jobs"]) else END

        def synthesizer(state):
            jobs = state["jobs"]
            job = jobs[state["job_cursor"]]
            rid = job["rule_id"]
            rule = self.ctx.rules.get(rid)
            if not rule:
                return state
            result = llm.generate(
                rule.rule_content, rule.title, rule.examples, rule.target, rule_id=rid,
            )
            state["candidate"] = result if result else None
            return state

        def crosscheck(state):
            cand = state.get("candidate")
            if not cand:
                state["solve_result"] = {"solvable": False}
                return state
            from .validators import validate_answer
            rid = state["jobs"][state["job_cursor"]]["rule_id"]
            ok, errs = validate_answer(rid, cand.get("answer",""), cand.get("question",""))
            state["solve_result"] = {"solvable": ok, "errors": errs}
            return state

        def verification(state):
            cand = state.get("candidate")
            sr = state.get("solve_result", {})
            ok = cand is not None and sr.get("solvable", False)
            ans = cand.get("answer","") if cand else ""
            fmt_ok = ans.strip().startswith("[[")
            state["verdict"] = {"accepted": ok and fmt_ok}
            return state

        def route_verify(state):
            return "data_preprocessor" if state["verdict"]["accepted"] else "synthesizer"

        def data_preprocessor(state):
            cand = state["candidate"]
            job = state["jobs"][state["job_cursor"]]
            rid = job["rule_id"]
            rule = self.ctx.rules.get(rid)
            if cand:
                rec = {
                    "idx": f"syn-{rid}-{len(state['accepted_records'])}",
                    "rule_id": rid,
                    "title": rule.title if rule else "",
                    "question": cand.get("question",""),
                    "answer": cand.get("answer",""),
                    "metadata": {},
                }
                state["accepted_records"] = state.get("accepted_records", []) + [rec]
            return state

        def route_after_accept(state):
            return "dispatcher"

        return dict(
            dispatcher=dispatcher, route_dispatch=route_dispatch,
            synthesizer=synthesizer, crosscheck=crosscheck,
            verification=verification, route_verify=route_verify,
            data_preprocessor=data_preprocessor,
            route_after_accept=route_after_accept,
        )


def build_orchestrator(cfg: Dict[str, Any], ctx) -> Any:
    """Build the V2 orchestrator graph with checkpointing and subgraphs."""
    orch = OrchestratorGraph(cfg, ctx)
    return orch.build()
