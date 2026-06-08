"""CLI: run PuzzleAgent benchmark."""
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


def main():
    ap = argparse.ArgumentParser(description="Run self-evolution benchmark")
    ap.add_argument("--config", default=None)
    ap.add_argument("--rules", default=None, help="comma-separated rule ids")
    ap.add_argument("--count", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--prompt-version", default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    rules = [r.strip() for r in args.rules.split(",")] if args.rules else None
    spec = build_spec(
        cfg,
        rules=rules,
        count=args.count,
        seed=args.seed,
        prompt_version=args.prompt_version,
        out_dir=args.out_dir,
    )
    report = run_benchmark(args.config, spec)
    print(json.dumps({
        "run_id": report["run_id"],
        "prompt_version": report["prompt_version"],
        "scores": report["scores"],
        "top_rejection_reasons": report["top_rejection_reasons"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
