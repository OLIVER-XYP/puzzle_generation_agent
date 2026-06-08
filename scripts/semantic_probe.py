"""CLI: run deterministic fuzzy-semantic probes for PuzzleAgent."""
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

from puzzle_agent.config import load_config, resolve          # noqa: E402
from puzzle_agent.graph import build_context                  # noqa: E402
from puzzle_agent.semantic_probe import (                     # noqa: E402
    SemanticProbeRunner,
    write_probe_markdown,
    write_probe_report,
)


def main():
    ap = argparse.ArgumentParser(description="Run fuzzy semantic probe benchmark")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="data/evolution/semantic_probe_report.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ctx = build_context(cfg)
    try:
        rule_index = {
            rid: {
                "title": r.title,
                "tag": r.tag,
                "rule_content": r.rule_content,
                "examples": r.examples,
                "target": r.target,
            }
            for rid, r in ctx.rules.items()
        }
        runner = SemanticProbeRunner(rule_index, user_memory=ctx.user_memory)
        report = runner.run()
        out_path = resolve(cfg, args.out)
        write_probe_report(report, out_path)
        md_path = out_path.with_suffix(".md")
        write_probe_markdown(report, md_path)
        print(json.dumps({
            "total": report["total"],
            "passed": report["passed"],
            "pass_rate": report["pass_rate"],
            "out": str(out_path),
            "markdown": str(md_path),
            "failures": [r for r in report["results"] if not r["passed"]],
        }, ensure_ascii=False, indent=2))
        if report["passed"] != report["total"]:
            raise SystemExit(1)
    finally:
        ctx.memory.close()


if __name__ == "__main__":
    main()
