"""CLI entry point: run the LLM-based puzzle-generation pipeline."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from puzzle_agent.config import load_config, resolve          # noqa: E402
from puzzle_agent.graph import build_context, build_graph     # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Synthesize & validate puzzles via LLM + LangGraph")
    ap.add_argument("--config", default=None)
    ap.add_argument("--rules", default=None, help="comma-separated rule ids")
    ap.add_argument("--count", type=int, default=None, help="puzzles per rule")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=None)
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

    ctx = build_context(cfg)
    app = build_graph(ctx)

    print(f"[run] rules={cfg['run']['rules']} count={cfg['run']['count_per_rule']} "
          f"gen={'on' if ctx.llm.enabled else 'off'}")

    final = app.invoke({}, config={"recursion_limit": cfg["run"]["recursion_limit"]})

    out_dir = resolve(cfg, cfg["run"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    records = final.get("accepted_records", [])

    ds_path = out_dir / "fine_dataset.jsonl"
    with open(ds_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    rep_path = out_dir / "run_report.json"
    per_rule = {}
    for rid in cfg["run"]["rules"]:
        n = sum(1 for r in records if r["rule_id"] == rid)
        per_rule[rid] = {"accepted": n}
    report = {"total_accepted": len(records), "per_rule": per_rule}
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[run] wrote {len(records)} samples -> {ds_path}")
    print(f"[run] per-rule: {json.dumps(per_rule, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
