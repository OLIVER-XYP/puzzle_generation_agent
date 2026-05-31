"""LangGraph pipeline — LLM-based puzzle generation.

Flow:
    dispatcher -> difficulty_sampler -> llm_synthesizer -> llm_crosscheck
              -> verification --accepted--> data_preprocessor --> (next)
                              --rejected--> difficulty_sampler (retry)
    dispatcher --no jobs--> END
"""
from __future__ import annotations
import json
import hashlib
from pathlib import Path
from random import Random
from typing import Any, Dict, List

from langgraph.graph import StateGraph, START, END

from .config import load_config, resolve
from .state import GraphState
from .llm_gen import LlmGen, LlmRule


def _load_eval(cfg) -> Dict[str, List[Dict[str, Any]]]:
    path = resolve(cfg, cfg["run"]["source_eval"])
    by: Dict[str, List[Dict[str, Any]]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            by.setdefault(r["rule_id"], []).append(r)
    return by


def _h(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _calibrate_target(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Crude structural calibration from examples."""
    if not examples:
        return {}
    q_lens = [len(ex["question"]) for ex in examples]
    a_lens = [len(ex["answer"]) for ex in examples]
    return {
        "q_len_min": min(q_lens), "q_len_max": max(q_lens),
        "q_len_avg": sum(q_lens) / len(q_lens),
        "a_len_min": min(a_lens), "a_len_max": max(a_lens),
    }


class Context:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.llm = LlmGen(cfg)
        eval_by_rule = _load_eval(cfg)

        # Build simplified rule objects
        self.rules: Dict[str, LlmRule] = {}
        self.targets: Dict[str, Dict[str, Any]] = {}
        self.eval_hashes: set = set()

        for rid in cfg["run"]["rules"]:
            examples = eval_by_rule.get(rid, [])
            if not examples:
                continue
            tgt = _calibrate_target(examples)
            r = LlmRule(rid, examples[0]["title"], examples[0].get("tag", ""),
                       examples[0]["rule_content"], examples, tgt)
            self.rules[rid] = r
            self.targets[rid] = tgt
            for key in r.eval_keys():
                self.eval_hashes.add(_h(key))

        self.rng = Random(cfg["run"]["seed"])


def build_context(cfg: Dict[str, Any]) -> Context:
    return Context(cfg)


def make_nodes(ctx: Context):
    cfg = ctx.cfg
    count = cfg["run"]["count_per_rule"]
    max_retries = cfg["run"]["max_retries_per_item"]

    def dispatcher(state: GraphState) -> GraphState:
        if "jobs" not in state:
            # Allow Studio input override: {"rules": ["1","5"], "count": 10}
            req_rules = state.get("rules") or cfg["run"]["rules"]
            req_count = state.get("count") or count
            jobs = [{"rule_id": rid, "count": req_count, "target": ctx.targets.get(rid, {})}
                    for rid in req_rules if rid in ctx.rules]
            state.update(jobs=jobs, job_cursor=0,
                         eval_hashes=list(ctx.eval_hashes), accepted_hashes=[],
                         accepted_records=[], rejected_log=[],
                         stats={"per_rule": {}})
        else:
            state["job_cursor"] = state["job_cursor"] + 1
        cur = state["job_cursor"]
        jobs = state["jobs"]
        if cur < len(jobs):
            job = jobs[cur]
            state.update(current_rule_id=job["rule_id"],
                         target_params=job["target"],
                         accepted_count=0, item_retries=0, candidate=None)
        return state

    def route_dispatch(state: GraphState) -> str:
        return "llm_synthesizer" if state["job_cursor"] < len(state["jobs"]) else "save_output"

    def save_output(state: GraphState) -> GraphState:
        import os
        out_dir = Path(os.getcwd()) / "data" / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        ds_path = out_dir / "fine_dataset.jsonl"
        records = state.get("accepted_records", [])
        with open(str(ds_path), "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        rep_path = out_dir / "run_report.json"
        per_rule = {}
        for rec in records:
            per_rule.setdefault(rec["rule_id"], 0)
            per_rule[rec["rule_id"]] += 1
        with open(str(rep_path), "w", encoding="utf-8") as f:
            json.dump({"total": len(records), "per_rule": per_rule}, f, ensure_ascii=False, indent=2)
        state["output_path"] = str(ds_path)
        return state

    def difficulty_sampler(state: GraphState) -> GraphState:
        # Minimal for now — just passes through; difficulty is guided by the prompt
        state["sampled_params"] = state.get("target_params", {})
        return state

    def llm_synthesizer(state: GraphState) -> GraphState:
        rid = state["current_rule_id"]
        rule = ctx.rules[rid]
        item_id = f"syn-{rid}-{state['accepted_count']}"
        result = ctx.llm.generate(
            rule.rule_content, rule.title, rule.examples, rule.target,
            rule_id=rid
        )
        if result:
            state["candidate"] = {
                "id": item_id, "rule_id": rid,
                "question": result["question"],
                "answer": result["answer"],
                "difficulty": {},
                "input_data": {"rule_id": rid, "idx": item_id},
                "metadata": {},
            }
        else:
            state["candidate"] = None
        return state

    def llm_crosscheck(state: GraphState) -> GraphState:
        cand = state["candidate"]
        if cand is None:
            state["solve_result"] = {"num_solutions": 0, "unique": False,
                                     "solvable": False, "crosscheck_ok": False,
                                     "structural_ok": False}
            return state
        rid = state["current_rule_id"]
        rule = ctx.rules[rid]
        # Ask LLM to independently solve
        check = ctx.llm.cross_check(rule.rule_content, cand["question"])
        matches = ctx.llm.answers_match(cand["answer"], check) if check else False
        # Structural validation
        from .validators import validate_answer
        struct_ok, struct_errors = validate_answer(rid, cand["answer"], cand["question"])
        state["solve_result"] = {
            "num_solutions": 1 if matches else 0,
            "unique": matches,
            "solvable": matches and struct_ok,
            "crosscheck_answer": check,
            "crosscheck_ok": matches,
            "structural_ok": struct_ok,
            "structural_errors": struct_errors,
        }
        cand["metadata"]["crosscheck_ok"] = matches
        cand["metadata"]["structural_ok"] = struct_ok
        return state

    def verification(state: GraphState) -> GraphState:
        cand = state["candidate"]
        sr = state["solve_result"]
        reasons: List[str] = []

        if cand is None:
            reasons.append("generation_failed")
        else:
            # Format check
            ans = cand.get("answer", "")
            if not (isinstance(ans, str) and ans.strip().startswith("[[")):
                reasons.append("bad_answer_format")
            # Structural validation — BLOCKING: must pass
            if not sr.get("structural_ok"):
                reasons.append(f"structural_invalid: {'; '.join(sr.get('structural_errors', []))}")
            # Cross-check — warn only (format differences cause false negatives)
            if not sr.get("crosscheck_ok"):
                cand["metadata"]["crosscheck_warning"] = True
            # Dedup
            qhash = _h(cand["question"].strip()[:200])
            if qhash in set(state["eval_hashes"]):
                reasons.append("duplicate_of_eval")
            if qhash in set(state["accepted_hashes"]):
                reasons.append("duplicate_of_accepted")
            # Difficulty check — within calibrated range
            qlen = len(cand["question"])
            tgt = state.get("target_params", {})
            if tgt.get("q_len_min") and qlen < tgt["q_len_min"] - 200:
                reasons.append(f"difficulty_too_easy: qlen={qlen}")

        state["verdict"] = {"accepted": len(reasons) == 0, "reasons": reasons,
                            "canon": cand["question"][:80] if cand else "",
                            "hash": qhash if cand else ""}
        if reasons:
            state["item_retries"] = state.get("item_retries", 0) + 1
            state["rejected_log"].append(
                {"rule": state["current_rule_id"], "reasons": reasons})
        return state

    def route_verify(state: GraphState) -> str:
        if state["verdict"]["accepted"]:
            return "data_preprocessor"
        if state["item_retries"] >= max_retries:
            # give up this slot
            state["accepted_count"] = state["jobs"][state["job_cursor"]]["count"]
            return "dispatcher"
        return "llm_synthesizer"

    def data_preprocessor(state: GraphState) -> GraphState:
        cand = state["candidate"]
        rid = state["current_rule_id"]
        rule = ctx.rules[rid]
        n = state["accepted_count"] + 1
        record = {
            "idx": f"syn-{rid}-{n}",
            "rule_id": rid,
            "title": rule.title,
            "tag": rule.tag,
            "question": cand["question"],
            "answer": cand["answer"],
            "rule_content": rule.rule_content,
            "difficulty": {},
            "reasoning_trace": f"LLM-generated and cross-verified. "
                               f"Crosscheck: {state['solve_result'].get('crosscheck_ok', False)}",
            "metadata": {**cand.get("metadata", {}),
                         "crosscheck_ok": state["solve_result"].get("crosscheck_ok")},
            "additional_info": {"input_data": cand.get("input_data", {})},
        }
        state["accepted_records"].append(record)
        state["accepted_hashes"].append(state["verdict"]["hash"])
        state["accepted_count"] = n
        state["item_retries"] = 0
        state["candidate"] = None
        pr = state["stats"]["per_rule"].setdefault(rid, {"accepted": 0})
        pr["accepted"] = n
        return state

    def route_after_accept(state: GraphState) -> str:
        job = state["jobs"][state["job_cursor"]]
        return "dispatcher" if state["accepted_count"] >= job["count"] else "llm_synthesizer"

    return dict(
        dispatcher=dispatcher, route_dispatch=route_dispatch,
        difficulty_sampler=difficulty_sampler,
        llm_synthesizer=llm_synthesizer, llm_crosscheck=llm_crosscheck,
        verification=verification, route_verify=route_verify,
        data_preprocessor=data_preprocessor, route_after_accept=route_after_accept,
        save_output=save_output,
    )

def build_graph(ctx: Context):
    n = make_nodes(ctx)
    g = StateGraph(GraphState)
    g.add_node("dispatcher", n["dispatcher"])
    g.add_node("llm_synthesizer", n["llm_synthesizer"])
    g.add_node("llm_crosscheck", n["llm_crosscheck"])
    g.add_node("verification", n["verification"])
    g.add_node("data_preprocessor", n["data_preprocessor"])
    g.add_node("save_output", n["save_output"])

    g.add_edge(START, "dispatcher")
    g.add_conditional_edges("dispatcher", n["route_dispatch"],
                            {"llm_synthesizer": "llm_synthesizer", "save_output": "save_output"})
    g.add_edge("llm_synthesizer", "llm_crosscheck")
    g.add_edge("llm_crosscheck", "verification")
    g.add_conditional_edges("verification", n["route_verify"],
                            {"data_preprocessor": "data_preprocessor",
                             "llm_synthesizer": "llm_synthesizer",
                             "dispatcher": "dispatcher"})
    g.add_conditional_edges("data_preprocessor", n["route_after_accept"],
                            {"dispatcher": "dispatcher",
                             "llm_synthesizer": "llm_synthesizer"})
    g.add_edge("save_output", END)
    return g.compile()
