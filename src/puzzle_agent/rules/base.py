"""Rule plugin interface.

Each puzzle type implements this ABC. The LangGraph nodes are rule-agnostic and
dispatch to the registered plugin via the registry.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from random import Random
from typing import Any, Dict, List

from ..schema import StructuredProblem, SolveResult


class Rule(ABC):
    rule_id: str = ""
    title: str = ""
    tag: str = ""
    rule_content: str = ""

    # ----- difficulty (structural) -----
    @abstractmethod
    def calibrate(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Measure the structural difficulty target from the original 10 items."""

    @abstractmethod
    def sample_params(self, target: Dict[str, Any], rng: Random) -> Dict[str, Any]:
        """Pick structural params for the next candidate, matching the target."""

    # ----- difficulty plan (per-slot, replicates the original distribution) -----
    def feature_key(self) -> str:
        """Name of the single structural difficulty axis used for plan matching."""
        return ""

    def plan_values(self, target: Dict[str, Any]) -> List[Any]:
        """The per-item difficulty values observed in the original 10 items.
        Empty list => this rule has no discrete per-slot axis (range band only)."""
        return []

    def build_plan(self, target: Dict[str, Any], count: int, rng: Random) -> List[Any]:
        """Replicate the original distribution to length `count` and shuffle, so the
        accepted set reproduces the original mean/spread of difficulty."""
        vals = self.plan_values(target)
        if not vals:
            return [None] * count
        reps = (count // len(vals)) + 1
        plan = (list(vals) * reps)[:count]
        rng.shuffle(plan)
        return plan

    def params_for(self, target: Dict[str, Any], value: Any, rng: Random) -> Dict[str, Any]:
        """Build generation params that should hit the planned difficulty `value`.
        Defaults to the free sampler when there is no per-slot axis."""
        return self.sample_params(target, rng)

    def feature_value(self, input_data: Dict[str, Any]) -> Any:
        return self.difficulty_features(input_data).get(self.feature_key())

    def matches_slot(self, input_data: Dict[str, Any], slot_value: Any) -> bool:
        """Exact match of the candidate's difficulty axis to the planned slot."""
        if slot_value is None:
            return True
        return self.feature_value(input_data) == slot_value

    # ----- synthesis -----
    @abstractmethod
    def generate(self, params: Dict[str, Any], rng: Random, item_id: str) -> StructuredProblem:
        """Construct a Structured Problem (input_data + difficulty + metadata)."""

    # ----- deterministic solving / uniqueness -----
    @abstractmethod
    def solve(self, input_data: Dict[str, Any]) -> SolveResult:
        """Deterministically solve and count solutions (capped) for uniqueness."""

    # ----- uniqueness policy -----
    def acceptance_policy(self) -> str:
        """'unique' => require exactly one solution;
        'solvable' => require at least one (answer is one valid solution / full set)."""
        return "unique"

    # ----- rendering -----
    @abstractmethod
    def render_question(self, input_data: Dict[str, Any]) -> str:
        """Natural-language question text, matching the original format."""

    @abstractmethod
    def render_answer(self, ground_truth: Any, input_data: Dict[str, Any]) -> str:
        """Final answer string wrapped in [[...]]."""

    # ----- dedup -----
    @abstractmethod
    def canonical(self, input_data: Dict[str, Any]) -> str:
        """Canonical key for duplicate detection."""

    def eval_keys(self, examples: List[Dict[str, Any]]) -> List[str]:
        """Canonical keys for the original eval items (same space as `canonical`),
        used to guarantee synthesized puzzles never duplicate the eval set."""
        return []

    # ----- difficulty features for verification band check -----
    @abstractmethod
    def difficulty_features(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Structural metrics compared against the calibrated target."""

    def within_band(self, feats: Dict[str, Any], target: Dict[str, Any], tol: Dict[str, Any]) -> bool:
        """Default: every target key present in feats must match exactly; override per rule."""
        for k, v in target.items():
            if k.startswith("_"):
                continue
            if k in feats and feats[k] != v:
                return False
        return True
