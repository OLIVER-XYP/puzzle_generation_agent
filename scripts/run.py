"""CLI entry point: run the LLM-based puzzle-generation pipeline.

Single-rule mode:  python scripts/run.py --rules 4 --count 5
Multi-rule mode:   python scripts/run.py --rules 4,10,25 --count 2
                   (rules processed in parallel via ThreadPoolExecutor)
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from puzzle_agent.config import load_config, resolve          # noqa: E402
from puzzle_agent.graph import build_context, build_graph     # noqa: E402


def _load_prior_data(out_dir: Path) -> dict:
    """Read existing output data for supplement mode awareness.

    Returns: {"hashes": set, "by_rule": {rule_id: {"count": N, "topics": [...]}}}
    """
    result = {"hashes": set(), "by_rule": {}}
    ds_path = out_dir / "fine_dataset.jsonl"
    if not ds_path.exists():
        return result
    try:
        with open(ds_path, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                rid = r.get("rule_id", "")
                q = r.get("question", "")
                qhash = hashlib.sha1(q.strip()[:200].encode()).hexdigest() if q else ""
                if qhash:
                    result["hashes"].add(qhash)
                if rid:
                    entry = result["by_rule"].setdefault(rid, {"count": 0, "topics": []})
                    entry["count"] += 1
                    # Extract first 3 words as topic hint
                    topic = " ".join(q.strip().split()[:3]) if q else ""
                    if topic and topic not in entry["topics"]:
                        entry["topics"].append(topic[:60])
    except Exception:
        pass
    return result


def _run_rules(cfg, ctx, rules: list, count: int) -> dict:
    """Run the graph for a specific set of rules. Returns accepted_records."""
    run_cfg = copy.deepcopy(cfg)
    run_cfg["run"]["rules"] = list(rules)
    run_cfg["run"]["count_per_rule"] = count
    run_cfg["run"]["seed"] = cfg["run"]["seed"] + hash(str(rules)) % 10000

    run_ctx = build_context(run_cfg)
    app = build_graph(run_ctx)

    final = app.invoke({}, config={"recursion_limit": run_cfg["run"]["recursion_limit"]})
    records = final.get("accepted_records", [])
    run_ctx.memory.close()
    return {"records": records, "rules": list(rules)}


def main():
    ap = argparse.ArgumentParser(description="Synthesize & validate puzzles via LLM + LangGraph")
    ap.add_argument("--config", default=None)
    ap.add_argument("--rules", default=None, help="comma-separated rule ids")
    ap.add_argument("--count", type=int, default=None, help="puzzles per rule")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--parallel", action="store_true",
                    help="force parallel mode (default: auto-detect when >1 rules)")
    ap.add_argument("--workers", type=int, default=4, help="max parallel workers")
    ap.add_argument("--tools", dest="tools", action="store_true", default=None,
                    help="enable LLM function calling (default: on)")
    ap.add_argument("--no-tools", dest="tools", action="store_false",
                    help="disable LLM function calling (plain text generation)")
    ap.add_argument("--summary-narrative", action="store_true",
                    help="add a one-paragraph LLM narrative to the run summary")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.rules:
        cfg["run"]["rules"] = [r.strip() for r in args.rules.split(",")]
    if args.count:
        cfg["run"]["count_per_rule"] = args.count
    if args.seed is not None:
        cfg["run"]["seed"] = args.seed
    if args.temperature is not None:
        cfg["generator"]["temperature"] = args.temperature
    if args.tools is not None:
        cfg["generator"]["use_tools"] = args.tools
    if args.summary_narrative:
        cfg.setdefault("run", {})["summary_narrative"] = True

    rules = cfg["run"]["rules"]
    count = cfg["run"]["count_per_rule"]
    multi_rule = len(rules) > 1 or args.parallel

    t_start = time.time()

    if multi_rule:
        # --- Parallel mode: one graph invocation per rule ---
        print(f"[run] PARALLEL mode: {len(rules)} rules x {count} puzzles "
              f"(workers={args.workers})")
        all_records = []
        futures = {}
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for rid in rules:
                f = executor.submit(_run_rules, cfg, None, [rid], count)
                futures[f] = rid

            for future in as_completed(futures):
                rid = futures[future]
                try:
                    result = future.result()
                    n = len(result["records"])
                    all_records.extend(result["records"])
                    print(f"[run] R{rid}: {n} accepted")
                except Exception as e:
                    print(f"[run] R{rid}: FAILED - {e}")

        records = all_records
        ctx = None
        graph_summary = None     # built fresh below from aggregated records
    else:
        # --- Sequential mode: single graph with all rules ---
        ctx = build_context(cfg)
        app = build_graph(ctx)

        # Check for prior data (supplement mode)
        out_dir = resolve(cfg, cfg["run"]["out_dir"])
        prior = _load_prior_data(out_dir)
        prior_rules = {rid: info for rid, info in prior["by_rule"].items()
                      if rid in rules}
        graph_input = {}
        if prior_rules:
            total_prior = sum(v["count"] for v in prior_rules.values())
            print(f"[run] Supplement mode: found {total_prior} prior records "
                  f"for {len(prior_rules)} requested rules")
            # Inject prior hashes for dedup
            dedup_hashes = list(prior["hashes"]) + list(ctx.eval_hashes)
            graph_input["eval_hashes"] = dedup_hashes
        else:
            print(f"[run] First mode: no prior data for requested rules")

        print(f"[run] rules={rules} count={count} "
              f"gen={'on' if ctx.pipeline.client.enabled else 'off'}")

        final = app.invoke(graph_input,
                          config={"recursion_limit": cfg["run"]["recursion_limit"]})
        records = final.get("accepted_records", [])
        graph_summary = final.get("summary")

    # --- Output ---
    out_dir = resolve(cfg, cfg["run"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    ds_path = out_dir / "fine_dataset.jsonl"
    with open(ds_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    rep_path = out_dir / "run_report.json"
    per_rule = {}
    for rid in rules:
        n = sum(1 for r in records if r["rule_id"] == rid)
        per_rule[rid] = {"accepted": n}
    report = {"total_accepted": len(records), "per_rule": per_rule,
              "elapsed_s": round(time.time() - t_start, 1)}
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[run] wrote {len(records)} samples -> {ds_path} "
          f"({report['elapsed_s']}s)")
    print(f"[run] per-rule: {json.dumps(per_rule, ensure_ascii=False)}")

    # --- memory ---
    if ctx is not None:
        stats = ctx.memory.get_session_stats()
        if "error" not in stats:
            print(f"[memory] session={stats['session_id']} "
                  f"logs={stats['total_logs']} accepted={stats['total_passed']} "
                  f"pass_rate={stats['pass_rate']}")
            if stats.get("by_rule"):
                top_rules = list(stats["by_rule"].items())[:8]
                print(f"[memory] by_rule: " +
                      ", ".join(f"R{r}={v['passed']}/{v['total']}" for r, v in top_rules))
        ctx.memory.close()

    # --- tracer ---
    from puzzle_agent.tracer import get_tracer
    tracer = get_tracer()
    t_summary = tracer.summary()
    print(f"[trace] {t_summary['total_calls']} LLM calls, "
          f"failed={t_summary['failed']}, pass_rate={t_summary['pass_rate']}")
    if t_summary.get("failed", 0) > 0:
        decisions = tracer.generate_sft_report()
        for rid, d in sorted(decisions.items()):
            if d.priority in ("HIGH", "MEDIUM"):
                top = ", ".join(f"{e[0]}({e[1]})" for e in d.top_errors)
                print(f"[trace] SFT candidate: R{rid} fail_rate={d.failure_rate:.0%} "
                      f"priority={d.priority} samples_needed={d.samples_needed} "
                      f"errors=[{top}]")

    # --- tool stats ---
    if cfg.get("generator", {}).get("use_tools"):
        # Collect tool stats from the pipeline (graph mode)
        if ctx is not None:
            tool_stats = ctx.pipeline.get_tool_stats()
        else:
            # Parallel mode: build a fresh context just to read config
            tool_stats = {}
        print(f"[tools] enabled (function calling)")
        if tool_stats:
            print(f"[tools] generator: {tool_stats.get('generator', {})}")
            print(f"[tools] solver: {tool_stats.get('solver', {})}")
            print(f"[tools] reviewer: {tool_stats.get('reviewer', {})}")
            print(f"[tools] total tool-call rounds: {tool_stats.get('total_rounds', 0)}")

    # --- consolidated run summary (from the summarize graph node) ---
    from puzzle_agent.summary import format_summary_dict, build_run_summary
    if graph_summary is None:
        # Parallel mode: the per-rule sub-graphs each summarized separately;
        # build one combined summary from the aggregated records.
        combined_state = {"accepted_records": records, "rejected_log": [],
                          "rules": rules}
        graph_summary = build_run_summary(combined_state, ctx=None).to_dict()
    print()
    print(format_summary_dict(graph_summary))


if __name__ == "__main__":
    main()
