"""Agent benchmark runner and scoring for PuzzleAgent."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import load_config, resolve
from .graph import build_context, build_graph


DEFAULT_BENCHMARK_RULES = ["4", "10", "15", "25"]
DEFAULT_BENCHMARK_SEED = 20260608


@dataclass
class BenchmarkSpec:
    rules: List[str] = field(default_factory=lambda: list(DEFAULT_BENCHMARK_RULES))
    count_per_rule: int = 3
    seed: int = DEFAULT_BENCHMARK_SEED
    out_dir: str = "data/evolution"
    prompt_version: str = "builtin"


def build_spec(cfg: Dict[str, Any], rules: Optional[List[str]] = None,
               count: Optional[int] = None, seed: Optional[int] = None,
               prompt_version: Optional[str] = None,
               out_dir: Optional[str] = None) -> BenchmarkSpec:
    bench = cfg.get("benchmark", {}) or {}
    evolution = cfg.get("evolution", {}) or {}
    return BenchmarkSpec(
        rules=list(rules or bench.get("rules") or DEFAULT_BENCHMARK_RULES),
        count_per_rule=int(count or bench.get("count_per_rule") or 3),
        seed=int(seed if seed is not None else bench.get("seed", DEFAULT_BENCHMARK_SEED)),
        out_dir=str(out_dir or bench.get("out_dir") or "data/evolution"),
        prompt_version=str(prompt_version or evolution.get("active_prompt_version") or "builtin"),
    )


def score_report(report: Dict[str, Any]) -> Dict[str, Any]:
    totals = report.get("totals", {})
    requested = max(int(totals.get("requested", 0) or 0), 1)
    accepted = int(totals.get("accepted", 0) or 0)
    attempts = max(int(totals.get("attempts", accepted) or accepted or 1), 1)
    structural_passed = int(totals.get("structural_passed", accepted) or 0)
    crosscheck_passed = int(totals.get("crosscheck_passed", 0) or 0)
    duplicates = int(totals.get("duplicates", 0) or 0)
    elapsed_s = float(totals.get("elapsed_s", 0.0) or 0.0)

    accept_rate = accepted / requested
    structural_pass_rate = structural_passed / attempts
    crosscheck_pass_rate = crosscheck_passed / max(accepted, 1)
    non_duplicate_rate = max(0.0, 1.0 - duplicates / attempts)
    latency_score = _latency_score(elapsed_s, requested)
    overall = (
        0.45 * accept_rate
        + 0.25 * structural_pass_rate
        + 0.15 * crosscheck_pass_rate
        + 0.10 * non_duplicate_rate
        + 0.05 * latency_score
    )
    return {
        "overall_score": round(overall, 6),
        "accept_rate": round(accept_rate, 6),
        "structural_pass_rate": round(structural_pass_rate, 6),
        "crosscheck_pass_rate": round(crosscheck_pass_rate, 6),
        "non_duplicate_rate": round(non_duplicate_rate, 6),
        "latency_score": round(latency_score, 6),
    }


def run_benchmark(config_path: str | None = None, spec: BenchmarkSpec | None = None) -> Dict[str, Any]:
    cfg = load_config(config_path)
    spec = spec or build_spec(cfg)
    cfg["run"]["rules"] = list(spec.rules)
    cfg["run"]["count_per_rule"] = spec.count_per_rule
    cfg["run"]["seed"] = spec.seed
    cfg.setdefault("evolution", {})["active_prompt_version"] = spec.prompt_version
    cfg.setdefault("evolution", {})["_explicit_active_prompt_version"] = True

    run_id = f"bench-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{spec.prompt_version}"
    out_base = resolve(cfg, spec.out_dir) / run_id
    out_base.mkdir(parents=True, exist_ok=True)

    ctx = build_context(cfg)
    app = build_graph(ctx)
    t0 = time.time()
    final = app.invoke({}, config={"recursion_limit": cfg["run"]["recursion_limit"]})
    elapsed_s = round(time.time() - t0, 3)

    records = final.get("accepted_records", [])
    rejected = final.get("rejected_log", [])
    summary = final.get("summary", {})
    per_rule = summarize_per_rule(spec.rules, records, rejected, spec.count_per_rule)

    totals = summarize_totals(spec.rules, records, rejected, spec.count_per_rule, elapsed_s)
    report = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "prompt_version": spec.prompt_version,
        "spec": {
            "rules": spec.rules,
            "count_per_rule": spec.count_per_rule,
            "seed": spec.seed,
        },
        "totals": totals,
        "per_rule": per_rule,
        "top_rejection_reasons": dict(Counter(
            reason.split(":")[0].strip()
            for item in rejected
            for reason in item.get("reasons", [])
        ).most_common(10)),
        "trace": _trace_summary(),
        "tool_rounds": _tool_stats(ctx),
        "run_summary": summary,
        "evolution_recommendations": build_evolution_recommendations(rejected),
    }
    report["scores"] = score_report(report)

    (out_base / "benchmark_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if records:
        with open(out_base / "accepted_records.jsonl", "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    ctx.memory.close()
    return report


def summarize_per_rule(rules: List[str], records: List[Dict[str, Any]],
                       rejected: List[Dict[str, Any]], count_per_rule: int) -> Dict[str, Dict[str, Any]]:
    per: Dict[str, Dict[str, Any]] = {}
    rejected_by_rule: Dict[str, List[Dict[str, Any]]] = {}
    for item in rejected:
        rejected_by_rule.setdefault(str(item.get("rule", "")), []).append(item)
    for rid in rules:
        recs = [r for r in records if str(r.get("rule_id")) == str(rid)]
        rejs = rejected_by_rule.get(str(rid), [])
        structural = sum(1 for r in recs if r.get("metadata", {}).get("structural_ok", True))
        crosscheck = sum(1 for r in recs if r.get("metadata", {}).get("crosscheck_ok", False))
        reviewer_scores = [
            float(r.get("metadata", {}).get("reviewer_score", 0) or 0)
            for r in recs
        ]
        per[str(rid)] = {
            "requested": count_per_rule,
            "accepted": len(recs),
            "rejected": len(rejs),
            "accept_rate": round(len(recs) / max(count_per_rule, 1), 6),
            "structural_passed": structural,
            "crosscheck_passed": crosscheck,
            "avg_reviewer_score": round(sum(reviewer_scores) / max(len(reviewer_scores), 1), 3),
            "top_rejection_reasons": dict(Counter(
                reason.split(":")[0].strip()
                for item in rejs
                for reason in item.get("reasons", [])
            ).most_common(5)),
        }
    return per


def summarize_totals(rules: List[str], records: List[Dict[str, Any]],
                     rejected: List[Dict[str, Any]], count_per_rule: int,
                     elapsed_s: float) -> Dict[str, Any]:
    attempts = len(records) + len(rejected)
    structural = sum(1 for r in records if r.get("metadata", {}).get("structural_ok", True))
    crosscheck = sum(1 for r in records if r.get("metadata", {}).get("crosscheck_ok", False))
    duplicates = sum(
        1 for item in rejected
        for reason in item.get("reasons", [])
        if reason.startswith("duplicate")
    )
    return {
        "requested": len(rules) * count_per_rule,
        "accepted": len(records),
        "rejected": len(rejected),
        "attempts": attempts,
        "structural_passed": structural,
        "crosscheck_passed": crosscheck,
        "duplicates": duplicates,
        "elapsed_s": elapsed_s,
    }


def build_evolution_recommendations(rejected: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    counts = Counter()
    for item in rejected:
        for reason in item.get("reasons", []):
            counts[reason.split(":")[0].strip()] += 1
    return [
        {"failure_type": key, "count": count, "action": recommend_action(key)}
        for key, count in counts.most_common()
    ]


def recommend_action(failure_type: str) -> str:
    if failure_type == "bad_answer_format":
        return "strengthen JSON-only and [[...]] answer-wrapper constraints"
    if failure_type == "structural_invalid":
        return "add rule-specific constraint checklist and tool verification reminder"
    if failure_type.startswith("duplicate"):
        return "increase diversity and anti-copy guidance"
    if "crosscheck" in failure_type:
        return "tighten solver/reviewer agreement criteria"
    return "inspect traces and add targeted prompt guardrail"


def _latency_score(elapsed_s: float, requested: int) -> float:
    if elapsed_s <= 0:
        return 1.0
    seconds_per_item = elapsed_s / max(requested, 1)
    return max(0.0, min(1.0, 1.0 - seconds_per_item / 120.0))


def _trace_summary() -> Dict[str, Any]:
    try:
        from .tracer import get_tracer
        return get_tracer().summary()
    except Exception:
        return {}


def _tool_stats(ctx: Any) -> Dict[str, Any]:
    try:
        return ctx.pipeline.get_tool_stats()
    except Exception:
        return {}
