import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from puzzle_agent.config import load_config
from puzzle_agent.graph import build_context
from puzzle_agent.semantic_probe import SemanticProbeRunner, DEFAULT_PROBES


def test_default_fuzzy_semantic_probes_pass(tmp_path):
    cfg = load_config()
    cfg["memory"]["db_path"] = str(tmp_path / "memory.db")
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
        report = SemanticProbeRunner(rule_index, ctx.user_memory).run(DEFAULT_PROBES)
        assert report["passed"] == report["total"]
        assert report["pass_rate"] == 1.0
    finally:
        ctx.memory.close()
