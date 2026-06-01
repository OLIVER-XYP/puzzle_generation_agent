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
from .llm_gen import LlmRule
from .agents import MultiAgentPipeline
from .memory import create_memory


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
        self.pipeline = MultiAgentPipeline(cfg)
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

        # Tool integration: enable LLM function calling for generation/solving/review
        self.use_tools = cfg.get("generator", {}).get("use_tools", True)

        # Memory: two-tier persistence (STM + LTM)
        mem_cfg = cfg.get("memory", {})
        self.memory = create_memory(
            use_redis=(mem_cfg.get("stm", "dict") == "redis"),
            use_postgres=(mem_cfg.get("ltm", "sqlite") == "postgres"),
            redis_url=mem_cfg.get("redis_url", ""),
            db_url=mem_cfg.get("database_url", ""),
            db_path=mem_cfg.get("db_path", "data/memory.db"),
        )


def build_context(cfg: Dict[str, Any]) -> Context:
    return Context(cfg)


def _render_nl_command(ctx: "Context", inspect_intents, other_intents) -> str:
    """Render a text answer for non-generation NL commands (list/inspect/help/stats).

    Used by the rewrite_query node so the Studio frontend shows a useful reply
    instead of silently generating puzzles.
    """
    lines: List[str] = []
    actions = {i.action for i in other_intents}

    if inspect_intents:
        for intent in inspect_intents:
            rid = intent.params.get("rule_id", "")
            rule = ctx.rules.get(rid)
            if rule:
                ex_q = rule.examples[0]["question"][:200] if rule.examples else ""
                lines.append(f"规则 {rid}：{rule.title} ({rule.tag})")
                lines.append(f"  说明：{rule.rule_content[:300]}")
                lines.append(f"  示例：{ex_q}")
            else:
                lines.append(f"规则 {rid} 不存在。可用规则：{sorted(ctx.rules.keys(), key=int)}")

    if "LIST_RULES" in actions:
        lines.append(f"共 {len(ctx.rules)} 种规则：")
        for rid in sorted(ctx.rules.keys(), key=int):
            r = ctx.rules[rid]
            lines.append(f"  • 规则 {rid}: {r.title} ({r.tag})")

    if "STATS" in actions:
        try:
            stats = ctx.memory.get_session_stats()
            lines.append(f"统计：{stats}")
        except Exception:
            lines.append("暂无统计数据。")

    if "HELP" in actions or not lines:
        lines.append("我可以：列出所有规则 / 查看规则N / 给规则N生成M道题 / 出几道数学题")

    return "\n".join(lines)


def make_nodes(ctx: Context):
    cfg = ctx.cfg
    count = cfg["run"]["count_per_rule"]
    max_retries = cfg["run"]["max_retries_per_item"]

    # ---- Natural Language → Structured Intent ----
    def rewrite_query(state: GraphState) -> GraphState:
        """Process natural language input via QueryRewriter.

        Only runs when user_query is provided and jobs haven't been set yet
        (e.g., from LangGraph Studio input panel or API call).
        Structured input (rules + count) bypasses this node.
        """
        user_query = state.get("user_query", "")
        if not user_query or "jobs" in state:
            return state  # already initialized or no NL input

        from .rewriter import QueryRewriter
        from .agents import LlmClient

        # Build lightweight rewriter with rule index
        rule_index = {}
        for rid, rule in ctx.rules.items():
            rule_index[rid] = {
                "title": rule.title,
                "tag": rule.tag,
                "rule_content": rule.rule_content,
                "examples": rule.examples,
                "target": rule.target,
            }

        client = LlmClient(cfg)
        rewriter = QueryRewriter(rule_index, client if client.enabled else None)
        result = rewriter.rewrite(user_query)

        # Extract generation intents
        gen_intents = [i for i in result.intents if i.action == "GENERATE"]
        inspect_intents = [i for i in result.intents if i.action == "INSPECT_RULE"]
        other_intents = [i for i in result.intents if i.action not in ("GENERATE", "INSPECT_RULE")]

        # Handle non-generation commands (list, stats, inspect, help, etc.)
        if not gen_intents and (other_intents or inspect_intents):
            # Non-generation request: render an answer, set count=0 → no generation
            nl_response = _render_nl_command(ctx, inspect_intents, other_intents)
            state.update(
                user_query=user_query,
                rules=[],          # no generation targets
                count=0,
                _nl_result=result,
                nl_response=nl_response,
            )
            return state

        # Generation request: extract rules and count
        req_rules = []
        req_count = count
        for intent in gen_intents:
            rid = intent.params.get("rule_id", "")
            c = intent.params.get("count", count)
            if rid and rid in ctx.rules:
                if rid not in req_rules:
                    req_rules.append(rid)
                req_count = c  # use the last count specified

        if not req_rules:
            # Fallback: if no specific rules extracted, use config defaults
            req_rules = cfg["run"]["rules"]

        # Set rules/count as hints; dispatcher will read them and build jobs
        state.update(
            user_query=user_query,
            rules=req_rules,
            count=req_count,
            _nl_result=result,  # preserved for stats/output
        )
        return state

    # ---- Job Dispatcher ----
    def dispatcher(state: GraphState) -> GraphState:
        if "jobs" not in state:
            # --- session start ---
            ctx.memory.start_session()
            # Allow Studio input override: {"rules": ["1","5"], "count": 10}
            # NOTE: use explicit None checks — count=0 (a non-generation NL command)
            # must NOT fall back to the config default.
            req_rules = state.get("rules")
            if req_rules is None:
                req_rules = cfg["run"]["rules"]
            req_count = state.get("count")
            if req_count is None:
                req_count = count
            # count==0 or empty rules → no generation jobs (e.g. list/inspect command)
            jobs = ([{"rule_id": rid, "count": req_count, "target": ctx.targets.get(rid, {})}
                     for rid in req_rules if rid in ctx.rules]
                    if req_count > 0 else [])
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
        return "llm_synthesizer" if state["job_cursor"] < len(state["jobs"]) else "summarize"

    def summarize(state: GraphState) -> GraphState:
        """Aggregate the whole run into a RunSummary before saving.

        Pulls production stats (state), quality stats (tracer), tool usage
        (pipeline), and memory session stats into state["summary"].
        """
        from .summary import build_run_summary
        with_narrative = cfg.get("run", {}).get("summary_narrative", False)
        run_summary = build_run_summary(state, ctx, with_narrative=with_narrative)
        state["summary"] = run_summary.to_dict()
        return state

    def save_output(state: GraphState) -> GraphState:
        import os
        out_dir = Path(os.getcwd()) / "data" / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        ds_path = out_dir / "fine_dataset.jsonl"
        records = state.get("accepted_records", [])
        state["output_path"] = str(ds_path)

        # Guard: a non-generation command (list/inspect/help) produces no records.
        # Do NOT overwrite the existing dataset with an empty file in that case.
        if not records:
            ctx.memory.end_session(notes="no records (non-generation command)")
            return state

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

        # --- persist the run summary alongside the dataset ---
        summary = state.get("summary")
        if summary:
            sum_path = out_dir / "run_summary.json"
            with open(str(sum_path), "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

        # --- persistence: end session ---
        ctx.memory.update_session_count(len(records))
        ctx.memory.end_session(notes=f"wrote {len(records)} puzzles to {ds_path}")
        return state

    def difficulty_sampler(state: GraphState) -> GraphState:
        # Minimal for now — just passes through; difficulty is guided by the prompt
        state["sampled_params"] = state.get("target_params", {})
        return state

    def llm_synthesizer(state: GraphState) -> GraphState:
        rid = state["current_rule_id"]
        rule = ctx.rules[rid]
        item_id = f"syn-{rid}-{state['accepted_count']}"
        output = ctx.pipeline.run(
            rule.rule_content, rule.title, rid, rule.examples, rule.target,
            require_reviewer_pass=False,
            use_tools=ctx.use_tools,
        )
        if output.question and output.answer:
            state["candidate"] = {
                "id": item_id, "rule_id": rid,
                "question": output.question,
                "answer": output.answer,
                "plan": output.plan,
                "difficulty": {},
                "input_data": {"rule_id": rid, "idx": item_id},
                "metadata": {
                    "generator_ok": output.generator_ok,
                    "solver_ok": output.solver_ok,
                    "reviewer_score": output.reviewer_score,
                    "reviewer_issues": output.reviewer_issues,
                },
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
        # Independent Solver (temp=0, blind to Generator output)
        sol_result = ctx.pipeline.solver.solve(
            rule.rule_content, cand["question"], rule_id=rid,
        )
        sol_answer = sol_result.parsed.get("answer", "") if sol_result.parsed else ""
        # Reviewer compares Generator vs Solver answers
        rev_result = ctx.pipeline.reviewer.review(
            rule.rule_content, cand["question"], cand["answer"], sol_answer,
            rule_id=rid,
        )
        rev_data = rev_result.parsed or {}
        answers_match = rev_data.get("answers_match", False) or rev_data.get("verdict") == "PASS"
        # Deterministic structural validation
        from .validators import validate_answer
        struct_ok, struct_errors = validate_answer(rid, cand["answer"], cand["question"])
        state["solve_result"] = {
            "num_solutions": 1 if answers_match else 0,
            "unique": answers_match,
            "solvable": answers_match and struct_ok,
            "crosscheck_answer": sol_answer,
            "crosscheck_ok": answers_match,
            "structural_ok": struct_ok,
            "structural_errors": struct_errors,
            "reviewer_verdict": rev_data.get("verdict", "FAIL"),
            "reviewer_score": rev_data.get("score", 0),
        }
        cand["metadata"]["crosscheck_ok"] = answers_match
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

        # --- persistence: log every verification attempt ---
        ctx.memory.log_generation(
            rule_id=state["current_rule_id"],
            agent_role="verification",
            stage="verify",
            passed=len(reasons) == 0,
            errors=list(reasons),
        )
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

        # --- persistence: save accepted puzzle ---
        ctx.memory.save_puzzle(record)
        return state

    def route_after_accept(state: GraphState) -> str:
        job = state["jobs"][state["job_cursor"]]
        return "dispatcher" if state["accepted_count"] >= job["count"] else "llm_synthesizer"

    return dict(
        rewrite_query=rewrite_query,
        dispatcher=dispatcher, route_dispatch=route_dispatch,
        difficulty_sampler=difficulty_sampler,
        llm_synthesizer=llm_synthesizer, llm_crosscheck=llm_crosscheck,
        verification=verification, route_verify=route_verify,
        data_preprocessor=data_preprocessor, route_after_accept=route_after_accept,
        summarize=summarize, save_output=save_output,
    )

def build_graph(ctx: Context):
    n = make_nodes(ctx)
    g = StateGraph(GraphState)
    g.add_node("rewrite_query", n["rewrite_query"])
    g.add_node("dispatcher", n["dispatcher"])
    g.add_node("llm_synthesizer", n["llm_synthesizer"])
    g.add_node("llm_crosscheck", n["llm_crosscheck"])
    g.add_node("verification", n["verification"])
    g.add_node("data_preprocessor", n["data_preprocessor"])
    g.add_node("summarize", n["summarize"])
    g.add_node("save_output", n["save_output"])

    # START → rewrite_query (NL→Intent) → dispatcher → ...
    g.add_edge(START, "rewrite_query")
    g.add_edge("rewrite_query", "dispatcher")
    g.add_conditional_edges("dispatcher", n["route_dispatch"],
                            {"llm_synthesizer": "llm_synthesizer", "summarize": "summarize"})
    g.add_edge("llm_synthesizer", "llm_crosscheck")
    g.add_edge("llm_crosscheck", "verification")
    g.add_conditional_edges("verification", n["route_verify"],
                            {"data_preprocessor": "data_preprocessor",
                             "llm_synthesizer": "llm_synthesizer",
                             "dispatcher": "dispatcher"})
    g.add_conditional_edges("data_preprocessor", n["route_after_accept"],
                            {"dispatcher": "dispatcher",
                             "llm_synthesizer": "llm_synthesizer"})
    # summarize → save_output → END
    g.add_edge("summarize", "save_output")
    g.add_edge("save_output", END)
    return g.compile()
