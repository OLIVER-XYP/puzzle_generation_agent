"""Fuzzy semantic probe benchmark for the conversational agent.

This is intentionally deterministic and offline: it evaluates the rule-based
QueryRewriter and behavior-memory layer without depending on an LLM call.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .rewriter import QueryRewriter, ParsedIntent
from .user_memory import UserMemoryManager


@dataclass
class SemanticProbe:
    probe_id: str
    query: str
    expected_actions: List[str]
    expected_rules: List[str] = field(default_factory=list)
    expected_bucket: str = ""
    session_summary: str = ""
    setup_memory: Dict[str, Any] = field(default_factory=dict)
    min_recommendations: int = 0
    category: str = "general"


@dataclass
class ProbeResult:
    probe_id: str
    passed: bool
    errors: List[str]
    actual: Dict[str, Any]


DEFAULT_PROBES: List[SemanticProbe] = [
    SemanticProbe(
        probe_id="followup_same_rule",
        query="再来两道刚才那种，难一点",
        session_summary="Rule 10 (24 Points): 1/1 passed (first mode, 1 records)",
        expected_actions=["GENERATE"],
        expected_rules=["10"],
        expected_bucket="long",
        category="context_ellipsis",
    ),
    SemanticProbe(
        probe_id="domain_fuzzy_math_hard",
        query="来点烧脑的数字网格题",
        expected_actions=["GENERATE", "GENERATE", "GENERATE"],
        expected_rules=["11", "12", "13"],
        expected_bucket="long",
        category="domain_fuzzy",
    ),
    SemanticProbe(
        probe_id="similar_sudoku_not_same",
        query="类似数独但不要数独本体，给我三个",
        expected_actions=["GENERATE", "GENERATE", "GENERATE"],
        expected_rules=["16", "17", "25"],
        category="analogy_negation",
    ),
    SemanticProbe(
        probe_id="recommend_next",
        query="根据我之前的偏好推荐下一步",
        expected_actions=["RECOMMEND"],
        setup_memory={
            "preferred_rules": {"10": 3},
            "preferred_tags": {"math": 4},
            "difficulty_preference": "hard",
        },
        min_recommendations=3,
        category="active_recommendation",
    ),
    SemanticProbe(
        probe_id="multi_rule_each",
        query="规则10和25都来一道",
        expected_actions=["GENERATE", "GENERATE"],
        expected_rules=["10", "25"],
        category="compound_distribution",
    ),
    SemanticProbe(
        probe_id="word_not_math",
        query="不要数学，来点文字类的题",
        expected_actions=["GENERATE", "GENERATE", "GENERATE"],
        expected_rules=["1", "2", "3"],
        category="negation_domain",
    ),
    SemanticProbe(
        probe_id="followup_all_recent_rules",
        query="都再补一道",
        session_summary=(
            "Rule 10 (24 Points): 1/1 passed (first mode, 1 records)\n"
            "Rule 25 (Skyscrapers): 1/1 passed (first mode, 1 records)"
        ),
        expected_actions=["GENERATE", "GENERATE"],
        expected_rules=["10", "25"],
        category="context_ellipsis",
    ),
    SemanticProbe(
        probe_id="analog_24_not_24",
        query="像24点那种计算感，但换个形式，来两道",
        expected_actions=["GENERATE", "GENERATE", "GENERATE"],
        expected_rules=["9", "11", "12"],
        category="analogy",
    ),
    SemanticProbe(
        probe_id="negative_spatial_math",
        query="不要空间类，想练计算推理",
        expected_actions=["GENERATE", "GENERATE", "GENERATE"],
        expected_rules=["7", "15", "17"],
        category="negation_domain",
    ),
    SemanticProbe(
        probe_id="english_fuzzy_grid",
        query="give me 2 hard grid logic puzzles, not sudoku",
        expected_actions=["GENERATE", "GENERATE", "GENERATE"],
        expected_rules=["11", "12", "13"],
        expected_bucket="long",
        category="cross_lingual_fuzzy",
    ),
]


class SemanticProbeRunner:
    def __init__(self, rule_index: Dict[str, Dict[str, Any]], user_memory: Optional[UserMemoryManager] = None):
        self.rule_index = rule_index
        self.user_memory = user_memory
        self.rewriter = QueryRewriter(rule_index, llm_client=None)

    def run_probe(self, probe: SemanticProbe) -> ProbeResult:
        if self.user_memory and probe.setup_memory:
            profile = self.user_memory.get_profile()
            for key, value in probe.setup_memory.items():
                setattr(profile, key, value)
            self.user_memory.save_profile(profile)

        rewrite = self.rewriter.rewrite(probe.query, probe.session_summary)
        actual = {
            "actions": [i.action for i in rewrite.intents],
            "rules": [str(i.params.get("rule_id", "")) for i in rewrite.intents if i.params.get("rule_id")],
            "buckets": [i.params.get("bucket", "") for i in rewrite.intents if i.params.get("bucket")],
            "clarification_needed": rewrite.clarification_needed,
        }
        if self.user_memory and any(a == "RECOMMEND" for a in actual["actions"]):
            actual["recommendations"] = self.user_memory.recommend_rules(limit=probe.min_recommendations or 3)

        errors: List[str] = []
        if actual["actions"] != probe.expected_actions:
            errors.append(f"actions {actual['actions']} != {probe.expected_actions}")
        if probe.expected_rules and actual["rules"][:len(probe.expected_rules)] != probe.expected_rules:
            errors.append(f"rules {actual['rules']} do not start with {probe.expected_rules}")
        if probe.expected_bucket and probe.expected_bucket not in actual["buckets"]:
            errors.append(f"bucket {probe.expected_bucket} not in {actual['buckets']}")
        if probe.min_recommendations:
            if len(actual.get("recommendations", [])) < probe.min_recommendations:
                errors.append("not enough recommendations")
        return ProbeResult(probe.probe_id, not errors, errors, actual)

    def run(self, probes: Optional[List[SemanticProbe]] = None) -> Dict[str, Any]:
        probes = probes or DEFAULT_PROBES
        results = [self.run_probe(p) for p in probes]
        passed = sum(1 for r in results if r.passed)
        return {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "total": len(results),
            "passed": passed,
            "pass_rate": round(passed / max(len(results), 1), 6),
            "results": [
                {
                    "probe_id": r.probe_id,
                    "passed": r.passed,
                    "errors": r.errors,
                    "actual": r.actual,
                }
                for r in results
            ],
        }


def write_probe_report(report: Dict[str, Any], out_path: str | Path) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def format_probe_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# Fuzzy Semantic Probe Report",
        "",
        "This offline benchmark evaluates whether PuzzleAgent can resolve vague, "
        "context-dependent user language into executable agent intents without "
        "calling an LLM.",
        "",
        "## Summary",
        "",
        f"- Total probes: {report.get('total', 0)}",
        f"- Passed: {report.get('passed', 0)}",
        f"- Pass rate: {report.get('pass_rate', 0):.0%}",
        f"- Generated at: {report.get('created_at', '')}",
        f"- Reproduction: `python scripts/semantic_probe.py --out data/evolution/semantic_probe_report.json`",
        "",
        "## Probe Categories",
        "",
        "- Context ellipsis: follow-up requests like \"two more like the previous one\".",
        "- Domain fuzziness: broad category requests like \"hard number-grid puzzles\".",
        "- Analogy and negation: requests like \"similar to Sudoku, but not Sudoku itself\".",
        "- Active recommendation: next-best-rule selection from user memory.",
        "- Compound distribution: multi-rule requests using all/each semantics.",
        "- Negative domain constraints: requests such as \"not math\".",
        "- Cross-lingual fuzziness: mixed English requests such as hard grid logic.",
        "",
        "## Results",
        "",
        "| Probe | Passed | Actions | Rules | Errors |",
        "|---|---:|---|---|---|",
    ]
    for item in report.get("results", []):
        actual = item.get("actual", {})
        actions = ", ".join(actual.get("actions", []))
        rules = ", ".join(actual.get("rules", []))
        errors = "; ".join(item.get("errors", []))
        lines.append(
            f"| {item.get('probe_id')} | {'yes' if item.get('passed') else 'no'} "
            f"| {actions} | {rules} | {errors} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "A passing run means the agent can deterministically map representative "
        "ambiguous requests to stable actions before invoking generation. This is "
        "useful for paper/project explanation because it separates semantic "
        "routing quality from downstream LLM puzzle-generation quality.",
        "",
        "## Paper-Style Claim",
        "",
        "The probe suite operationalizes semantic robustness as exact-match intent "
        "routing under vague language. The measured pass rate is not a puzzle "
        "quality score; it is an upstream controller score showing whether the "
        "agent can choose the correct rule family, count, difficulty bucket, and "
        "recommendation action before generation.",
    ])
    return "\n".join(lines) + "\n"


def write_probe_markdown(report: Dict[str, Any], out_path: str | Path) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_probe_markdown(report), encoding="utf-8")
    return path
