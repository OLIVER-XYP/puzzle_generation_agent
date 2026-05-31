"""Structured problem schema and final export record."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class StructuredProblem:
    """The 'Structured Problem' produced by the Rule Synthesizer.

    Mirrors the fields described in the design brief:
    id, rule_id, difficulty, input_data, ground_truth, reasoning_trace,
    metadata, additional_info.
    """
    id: str
    rule_id: str
    difficulty: Dict[str, Any]                 # structural difficulty features
    input_data: Dict[str, Any]                 # rule-specific puzzle definition
    ground_truth: Any = None                   # filled by the deterministic solver
    reasoning_trace: str = ""                  # template or teacher-generated
    metadata: Dict[str, Any] = field(default_factory=dict)
    additional_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SolveResult:
    """Output of a deterministic solver."""
    ground_truth: Any                          # canonical answer object
    num_solutions: int                         # capped (e.g. at 2) for uniqueness check
    unique: bool                               # num_solutions == 1
    solvable: bool                             # num_solutions >= 1
    steps: List[str] = field(default_factory=list)   # deterministic reasoning steps
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FinalRecord:
    """Final accepted sample, schema-compatible with the original puzzle.jsonl
    plus synthesis provenance fields."""
    idx: str
    rule_id: str
    title: str
    tag: str
    question: str
    answer: str
    rule_content: str
    difficulty: Dict[str, Any]
    reasoning_trace: str
    metadata: Dict[str, Any]
    additional_info: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
