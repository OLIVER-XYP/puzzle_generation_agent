"""Prompt-level self-evolution for PuzzleAgent.

The evolution loop never edits source code. It proposes prompt versions,
evaluates them with the benchmark, and promotes only versions that pass a
strict regression gate.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import copy
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .prompt_provider import PromptBundle, PromptProvider


FAILURE_PATCHES: Dict[str, Dict[str, str]] = {
    "bad_answer_format": {
        "target": "generator",
        "text": (
            "\n## Evolution Guardrail: Answer Format\n"
            "- The final response must be parseable JSON only after any <planning> block.\n"
            "- The JSON answer field must start with [[ and end with ]].\n"
            "- Before finalizing, explicitly check the wrapper in the TODO list.\n"
        ),
    },
    "structural_invalid": {
        "target": "generator",
        "text": (
            "\n## Evolution Guardrail: Structural Validity\n"
            "- Translate every rule constraint into a checklist item before designing the puzzle.\n"
            "- For grid/number rules, verify row length, uniqueness, symbols, and clue consistency.\n"
            "- If a deterministic tool is available, call it once to verify the final answer.\n"
        ),
    },
    "duplicate_of_eval": {
        "target": "generator",
        "text": (
            "\n## Evolution Guardrail: Diversity\n"
            "- Do not reuse names, numbers, grids, or story templates from examples.\n"
            "- Change at least two independent surface features from every few-shot example.\n"
        ),
    },
    "duplicate_of_accepted": {
        "target": "generator",
        "text": (
            "\n## Evolution Guardrail: Batch Diversity\n"
            "- Avoid repeating patterns from earlier accepted outputs in this run.\n"
            "- Prefer a different topic, grid layout, clue style, or number set for each item.\n"
        ),
    },
    "crosscheck": {
        "target": "reviewer",
        "text": (
            "\n## Evolution Guardrail: Cross-check Strictness\n"
            "- Treat answer mismatch as FAIL unless equivalence is explicitly justified.\n"
            "- Prefer deterministic validation results over natural-language plausibility.\n"
        ),
    },
}


@dataclass
class EvolutionGate:
    promote_min_score_delta: float = 0.05
    max_rule_regression: float = 0.10


def load_report(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def propose_prompt_versions(report: Dict[str, Any], cfg: Dict[str, Any],
                            max_candidates: Optional[int] = None) -> List[Dict[str, Any]]:
    evolution_cfg = cfg.get("evolution", {}) or {}
    max_candidates = int(max_candidates or evolution_cfg.get("max_candidates") or 3)
    provider = PromptProvider(cfg)
    parent_version = str(report.get("prompt_version") or provider.active_version())
    parent = provider.load(parent_version)
    failures = _rank_failures(report)
    if not failures:
        failures = ["structural_invalid"]

    candidates: List[Dict[str, Any]] = []
    for i, failure in enumerate(failures[:max_candidates], 1):
        bundle = apply_failure_patch(parent, [failure])
        version_id = _candidate_id(parent.version_id, failure, i)
        candidates.append({
            "version_id": version_id,
            "label": f"evolve {failure}",
            "parent": parent.version_id,
            "failure_types": [failure],
            "prompts": bundle.to_dict(),
            "notes": f"Generated from benchmark {report.get('run_id', '')}: {failure}",
            "created": datetime.now().isoformat(),
        })

    if len(failures) > 1 and len(candidates) < max_candidates:
        bundle = apply_failure_patch(parent, failures[:2])
        candidates.append({
            "version_id": _candidate_id(parent.version_id, "combined", len(candidates) + 1),
            "label": "evolve combined top failures",
            "parent": parent.version_id,
            "failure_types": failures[:2],
            "prompts": bundle.to_dict(),
            "notes": f"Generated from benchmark {report.get('run_id', '')}: combined",
            "created": datetime.now().isoformat(),
        })
    return candidates[:max_candidates]


def apply_failure_patch(parent: PromptBundle, failures: List[str]) -> PromptBundle:
    prompts = parent.to_dict()
    for failure in failures:
        key = _normalize_failure(failure)
        patch = FAILURE_PATCHES.get(key)
        if not patch:
            patch = {
                "target": "generator",
                "text": f"\n## Evolution Guardrail: {key}\n- Avoid this failure mode: {failure}.\n",
            }
        target = patch["target"]
        prompts[target] = prompts.get(target, "") + patch["text"]
    return PromptBundle(
        version_id=parent.version_id,
        generator=prompts["generator"],
        solver=prompts["solver"],
        reviewer=prompts["reviewer"],
    )


def save_candidates(cfg: Dict[str, Any], candidates: List[Dict[str, Any]]) -> Path:
    root = Path(cfg.get("_root", "."))
    path = root / "data" / "prompt_registry.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"versions": []}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            data = {"versions": []}
    existing = {v.get("version_id"): v for v in data.get("versions", [])}
    for candidate in candidates:
        existing[candidate["version_id"]] = candidate
    data["versions"] = list(existing.values())
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def gate_candidate(parent_report: Dict[str, Any], candidate_report: Dict[str, Any],
                   gate: EvolutionGate) -> Dict[str, Any]:
    parent_scores = parent_report.get("scores", {})
    candidate_scores = candidate_report.get("scores", {})
    parent_overall = float(parent_scores.get("overall_score", 0.0) or 0.0)
    candidate_overall = float(candidate_scores.get("overall_score", 0.0) or 0.0)
    delta = candidate_overall - parent_overall

    reasons: List[str] = []
    if delta < gate.promote_min_score_delta:
        reasons.append(
            f"overall score delta {delta:.3f} < required {gate.promote_min_score_delta:.3f}")
    if candidate_scores.get("structural_pass_rate", 0) < parent_scores.get("structural_pass_rate", 0):
        reasons.append("structural pass rate regressed")
    if candidate_scores.get("non_duplicate_rate", 0) < parent_scores.get("non_duplicate_rate", 0):
        reasons.append("duplicate rate increased")

    parent_per = parent_report.get("per_rule", {})
    candidate_per = candidate_report.get("per_rule", {})
    for rid, parent_rule in parent_per.items():
        p = float(parent_rule.get("accept_rate", 0.0) or 0.0)
        c = float(candidate_per.get(rid, {}).get("accept_rate", 0.0) or 0.0)
        if p - c > gate.max_rule_regression:
            reasons.append(f"R{rid} accept_rate regressed by {p - c:.3f}")

    return {
        "passed": not reasons,
        "reasons": reasons,
        "overall_delta": round(delta, 6),
        "parent_score": parent_overall,
        "candidate_score": candidate_overall,
    }


def promote_version(cfg: Dict[str, Any], version_id: str, memory: Optional[Any] = None) -> None:
    cfg.setdefault("evolution", {})["active_prompt_version"] = version_id
    cfg.setdefault("evolution", {})["_explicit_active_prompt_version"] = True
    root = Path(cfg.get("_root", "."))
    active_path = root / "data" / "evolution" / "active_prompt_version.txt"
    active_path.parent.mkdir(parents=True, exist_ok=True)
    active_path.write_text(version_id, encoding="utf-8")
    if memory is not None:
        memory.set_fact("evolution:active_prompt_version", version_id)


def _rank_failures(report: Dict[str, Any]) -> List[str]:
    reasons = report.get("top_rejection_reasons") or {}
    recs = report.get("evolution_recommendations") or []
    ordered = [r.get("failure_type") for r in recs if r.get("failure_type")]
    for key in sorted(reasons, key=lambda k: reasons[k], reverse=True):
        if key not in ordered:
            ordered.append(key)
    return ordered


def _normalize_failure(reason: str) -> str:
    head = reason.split(":")[0].strip()
    if head in FAILURE_PATCHES:
        return head
    if head.startswith("duplicate"):
        return "duplicate_of_eval"
    if "crosscheck" in head:
        return "crosscheck"
    return head


def _candidate_id(parent: str, failure: str, idx: int) -> str:
    clean = re.sub(r"[^a-zA-Z0-9_-]+", "-", failure).strip("-").lower()[:32]
    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"{parent}-evo-{idx}-{clean}-{stamp}"
