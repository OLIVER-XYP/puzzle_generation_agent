"""LangGraph Studio entry point.

Expose the compiled puzzle-generation graph so LangGraph Studio can
visualize and step through it interactively.

Usage:
    langgraph dev   # opens Studio at http://localhost:8123

In the Studio input panel, you can type either:

  Structured:
    {"rules": ["4","10"], "count": 5}

  Natural language:
    {"user_query": "给规则4生成5道题"}
    {"user_query": "出几道简单数学题"}
    {"user_query": "列出所有规则，然后给规则15出3道"}
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from puzzle_agent.config import load_config        # noqa: E402
from puzzle_agent.graph import build_context, build_graph  # noqa: E402

# Build once at module level for the Studio server
_cfg = load_config()
_ctx = build_context(_cfg)
_compiled = build_graph(_ctx)


def get_graph():
    """Return the compiled LangGraph app for Studio visualization.

    The graph accepts:
      - user_query: str   (natural language, e.g. "给规则4生成5道题")
      - rules: list[str]  (structured override, e.g. ["4","10"])
      - count: int        (puzzles per rule)
      - seed: int         (random seed)
    """
    return _compiled
