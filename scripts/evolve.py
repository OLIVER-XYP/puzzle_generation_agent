"""CLI: propose and optionally gate/prompt-promote self-evolution candidates."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from puzzle_agent.benchmark import build_spec, run_benchmark  # noqa: E402
from puzzle_agent.config import load_config                   # noqa: E402
from puzzle_agent.evolution import (                          # noqa: E402
    EvolutionGate,
    gate_candidate,
    load_report,
    promote_version,
    propose_prompt_versions,
    save_candidates,
)


def main():
    ap = argparse.ArgumentParser(description="Run prompt-level self-evolution")
    ap.add_argument("--config", default=None)
    ap.add_argument("--benchmark", required=True, help="parent benchmark_report.json")
    ap.add_argument("--run-gate", action="store_true",
                    help="benchmark each candidate and evaluate promotion gate")
    ap.add_argument("--promote", action="store_true",
                    help="promote best passing candidate; requires --run-gate")
    ap.add_argument("--max-candidates", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    parent_report = load_report(args.benchmark)
    candidates = propose_prompt_versions(
        parent_report, cfg, max_candidates=args.max_candidates)
    registry_path = save_candidates(cfg, candidates)

    result = {
        "registry_path": str(registry_path),
        "candidates": [
            {
                "version_id": c["version_id"],
                "parent": c["parent"],
                "failure_types": c["failure_types"],
            }
            for c in candidates
        ],
        "gate_results": [],
        "promoted": "",
    }

    if args.run_gate:
        gate_cfg = cfg.get("evolution", {}) or {}
        gate = EvolutionGate(
            promote_min_score_delta=float(gate_cfg.get("promote_min_score_delta", 0.05)),
            max_rule_regression=float(gate_cfg.get("max_rule_regression", 0.10)),
        )
        best = None
        for candidate in candidates:
            spec_data = parent_report.get("spec", {})
            spec = build_spec(
                cfg,
                rules=spec_data.get("rules"),
                count=spec_data.get("count_per_rule"),
                seed=spec_data.get("seed"),
                prompt_version=candidate["version_id"],
            )
            candidate_report = run_benchmark(args.config, spec)
            gate_result = gate_candidate(parent_report, candidate_report, gate)
            gate_item = {
                "version_id": candidate["version_id"],
                "benchmark_run_id": candidate_report["run_id"],
                **gate_result,
            }
            result["gate_results"].append(gate_item)
            if gate_result["passed"]:
                if best is None or gate_result["candidate_score"] > best["candidate_score"]:
                    best = gate_item
        if args.promote and best:
            promote_version(cfg, best["version_id"])
            result["promoted"] = best["version_id"]

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
