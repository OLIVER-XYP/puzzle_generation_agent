"""LangGraph Studio entry point.

Expose the compiled puzzle-generation graph so LangGraph Studio can
visualize and step through it interactively.
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
    """Return the compiled LangGraph app for Studio visualization."""
    return _compiled
