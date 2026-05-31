"""Independent post-hoc verification of the generated Fine Dataset.
Checks: (1) no eval duplication, (2) format valid, (3) cross-check if available."""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from puzzle_agent.config import load_config, resolve
from puzzle_agent.llm_gen import LlmGen


def main():
    cfg = load_config()
    ds = resolve(cfg, cfg["run"]["out_dir"]) / "fine_dataset.jsonl"
    if not ds.exists():
        print(f"No dataset at {ds}")
        return

    rows = [json.loads(l) for l in open(ds, encoding="utf-8")]
    print(f"Records: {len(rows)}")

    # Load eval hashes
    eval_by = {}
    src = resolve(cfg, cfg["run"]["source_eval"])
    with open(src, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            eval_by.setdefault(r["rule_id"], []).append(r)

    problems = []
    per_rule = {}
    syn_hashes = set()

    for r in rows:
        rid = r["rule_id"]
        st = per_rule.setdefault(rid, {"n": 0, "fmt": 0, "evaldup": 0, "intradup": 0, "xcheck": 0})
        st["n"] += 1

        # Format check
        ans = r.get("answer", "")
        if ans.strip().startswith("[["):
            st["fmt"] += 1
        else:
            problems.append((r["idx"], "bad_format"))

        # Eval dedup
        qkey = r["question"].strip()[:200]
        eval_keys = {}
        for ex in eval_by.get(rid, []):
            eval_keys[ex["question"].strip()[:200]] = True
        if qkey in eval_keys:
            problems.append((r["idx"], "DUP_OF_EVAL"))
        else:
            st["evaldup"] += 1

        # Intra-set dedup
        if qkey in syn_hashes:
            problems.append((r["idx"], "DUP_INTRA"))
        else:
            st["intradup"] += 1
        syn_hashes.add(qkey)

        # Cross-check status
        if r.get("metadata", {}).get("crosscheck_ok"):
            st["xcheck"] += 1

    print(f"{'rule':>5} {'n':>3} {'fmt_ok':>6} {'no_evaldup':>10} {'no_intradup':>11} {'xcheck_ok':>9}")
    for rid in sorted(per_rule.keys(), key=int):
        s = per_rule[rid]
        print(f"{rid:>5} {s['n']:>3} {s['fmt']:>6} {s['evaldup']:>10} {s['intradup']:>11} {s['xcheck']:>9}")

    if problems:
        print(f"\nPROBLEMS ({len(problems)}):")
        for idx, msg in problems[:30]:
            print(f"  {idx}: {msg}")
    else:
        print("\nNO PROBLEMS.")


if __name__ == "__main__":
    main()
