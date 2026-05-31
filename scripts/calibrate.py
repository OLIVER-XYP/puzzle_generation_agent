"""Print the calibrated structural difficulty targets for each configured rule."""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from puzzle_agent.config import load_config                    # noqa: E402
from puzzle_agent.graph import build_context                   # noqa: E402


def main():
    cfg = load_config()
    ctx = build_context(cfg)
    print(json.dumps({rid: ctx.targets[rid] for rid in cfg["run"]["rules"]},
                     ensure_ascii=False, indent=2))
    print(f"eval canonical hashes loaded: {len(ctx.eval_hashes)}")


if __name__ == "__main__":
    main()
