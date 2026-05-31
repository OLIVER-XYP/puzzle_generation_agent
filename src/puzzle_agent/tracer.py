"""Trace + Diagnosis system (P0).

Every LLM call is recorded with prompt, response, errors, latency.
Auto-diagnoses root cause and generates SFT recommendations.
"""
from __future__ import annotations
import json
import hashlib
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class CallTrace:
    """Single LLM call trace."""
    trace_id: str
    agent_role: str           # "generator" | "solver" | "reviewer" | "rewriter"
    rule_id: str
    attempt: int
    stage: str                # "generation" | "parsing" | "format_validation" | "structural_validation" | "crosscheck"
    prompt_system: str
    prompt_user: str
    response_raw: str
    parsed: Optional[Dict[str, Any]] = None
    errors: List[str] = field(default_factory=list)
    error_category: str = ""  # "parse" | "structure" | "crosscheck" | "duplicate" | "format" | "none"
    latency_ms: int = 0
    tokens_approx: int = 0
    model: str = ""
    temperature: float = 0.0

    @property
    def trace_hash(self) -> str:
        return hashlib.md5((self.prompt_user + self.response_raw).encode()).hexdigest()[:8]

    @property
    def succeeded(self) -> bool:
        return len(self.errors) == 0


@dataclass
class Diagnosis:
    """Auto-generated diagnosis for a failed trace."""
    trace_id: str
    stage: str
    root_cause: str
    category: str            # "prompt_issue" | "model_capability" | "parser_bug" | "data_issue"
    suggestion: str
    sft_recommended: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "stage": self.stage,
            "root_cause": self.root_cause,
            "category": self.category,
            "suggestion": self.suggestion,
            "sft_recommended": self.sft_recommended,
        }


@dataclass
class SFTDecision:
    """Per-rule SFT decision based on accumulated traces."""
    rule_id: str
    rule_title: str
    total_calls: int
    failure_rate: float
    top_errors: List[Tuple[str, int]]
    priority: str             # "HIGH" | "MEDIUM" | "LOW"
    samples_needed: int
    recommendation: str


class Tracer:
    """Records all LLM calls and generates diagnoses."""

    def __init__(self):
        self.traces: List[CallTrace] = []
        self._session_start = datetime.now()

    def record(self, **kwargs) -> CallTrace:
        trace = CallTrace(
            trace_id=f"tr-{len(self.traces)+1:04d}",
            **kwargs,
        )
        self.traces.append(trace)
        return trace

    def recent_failures(self, rule_id: str = None, agent_role: str = None, limit: int = 20) -> List[CallTrace]:
        failed = [t for t in self.traces if not t.succeeded]
        if rule_id:
            failed = [t for t in failed if t.rule_id == rule_id]
        if agent_role:
            failed = [t for t in failed if t.agent_role == agent_role]
        return failed[-limit:]

    def diagnose(self, trace: CallTrace) -> Diagnosis:
        """Automatically diagnose a failed trace."""
        if not trace.errors:
            return Diagnosis(
                trace_id=trace.trace_id,
                stage=trace.stage,
                root_cause="Unknown (no errors recorded)",
                category="data_issue",
                suggestion="Check if errors were correctly captured.",
                sft_recommended=False,
            )

        first = trace.errors[0]
        if "parse" in first.lower() or "json" in first.lower():
            return Diagnosis(
                trace_id=trace.trace_id,
                stage="parsing",
                root_cause="LLM output format not parseable as JSON",
                category="prompt_issue",
                suggestion="Strengthen JSON schema constraints in system prompt. Add explicit format example in user prompt.",
                sft_recommended=False,
            )
        elif "structural" in first.lower() or "invalid" in first.lower():
            return Diagnosis(
                trace_id=trace.trace_id,
                stage="structural_validation",
                root_cause=f"Generated answer violates rule constraints: {first}",
                category="model_capability",
                suggestion=f"Consider SFT for this rule type. Collect {self._estimate_samples_needed(trace.rule_id)} failure pairs.",
                sft_recommended=True,
            )
        elif "crosscheck" in first.lower() or "mismatch" in first.lower():
            return Diagnosis(
                trace_id=trace.trace_id,
                stage="crosscheck",
                root_cause="Generator and Solver disagree on answer",
                category="model_capability",
                suggestion="Increase Solver temperature to 0. Consider using reasoner model for this rule.",
                sft_recommended=True,
            )
        elif "duplicate" in first.lower():
            return Diagnosis(
                trace_id=trace.trace_id,
                stage="dedup",
                root_cause="Generated question too similar to existing ones",
                category="prompt_issue",
                suggestion="Increase diversity by adding more diverse examples and raising temperature.",
                sft_recommended=False,
            )
        elif "format" in first.lower() or "[[ " in first.lower():
            return Diagnosis(
                trace_id=trace.trace_id,
                stage="format_validation",
                root_cause="Answer not wrapped in [[...]]",
                category="prompt_issue",
                suggestion="Add explicit format reminder in the final line of the prompt.",
                sft_recommended=False,
            )
        else:
            return Diagnosis(
                trace_id=trace.trace_id,
                stage=trace.stage,
                root_cause=f"Unrecognized error: {first}",
                category="data_issue",
                suggestion="Review trace manually.",
                sft_recommended=False,
            )

    def _estimate_samples_needed(self, rule_id: str) -> int:
        failures = [t for t in self.traces if t.rule_id == rule_id and not t.succeeded]
        return min(len(failures) * 2, 100)

    def generate_sft_report(self) -> Dict[str, SFTDecision]:
        """Analyze all traces and generate per-rule SFT recommendations."""
        by_rule = defaultdict(list)
        for t in self.traces:
            by_rule[t.rule_id].append(t)

        decisions = {}
        for rule_id, traces in sorted(by_rule.items()):
            failed = [t for t in traces if not t.succeeded]
            rate = len(failed) / len(traces) if traces else 0
            error_counts = Counter()
            for t in failed:
                for e in t.errors:
                    error_counts[e.split(":")[0].strip()] += 1

            if rate > 0.30:
                priority = "HIGH"
            elif rate > 0.10:
                priority = "MEDIUM"
            else:
                priority = "LOW"

            decisions[rule_id] = SFTDecision(
                rule_id=rule_id,
                rule_title=f"Rule {rule_id}",
                total_calls=len(traces),
                failure_rate=rate,
                top_errors=error_counts.most_common(3),
                priority=priority,
                samples_needed=len(failed) * 2 if failed else 0,
                recommendation=(
                    f"Collect {len(failed)*2} (question, correct_answer) pairs and fine-tune"
                    if failed else "No action needed"
                ),
            )

        return {k: v for k, v in sorted(decisions.items(),
                key=lambda x: x[1].priority_order(), reverse=True)}

    def summary(self) -> Dict[str, Any]:
        """Quick summary for dashboard."""
        total = len(self.traces)
        failed = sum(1 for t in self.traces if not t.succeeded)
        by_role = Counter(t.agent_role for t in self.traces)
        return {
            "total_calls": total,
            "failed": failed,
            "pass_rate": f"{(total-failed)/max(total,1)*100:.1f}%",
            "by_role": dict(by_role),
            "session_minutes": (datetime.now() - self._session_start).total_seconds() / 60,
        }


# Global singleton
_tracer = Tracer()


def get_tracer() -> Tracer:
    return _tracer


# ---- Trace helpers ----

def trace_call(agent_role: str, rule_id: str, attempt: int, stage: str,
               prompt_system: str, prompt_user: str, response_raw: str,
               parsed: Optional[Dict] = None, errors: List[str] = None,
               model: str = "", temp: float = 0.0,
               start_time: float = 0) -> CallTrace:
    latency = int((time.time() - start_time) * 1000) if start_time else 0
    errs = errors or []
    cat = "none"
    if errs:
        e0 = errs[0].lower()
        if any(w in e0 for w in ["parse", "json"]):
            cat = "parse"
        elif any(w in e0 for w in ["structural", "invalid", "row", "column", "duplicate"]):
            cat = "structure"
        elif "crosscheck" in e0:
            cat = "crosscheck"
        elif "duplicate" in e0:
            cat = "duplicate"
        elif "format" in e0:
            cat = "format"

    return _tracer.record(
        agent_role=agent_role, rule_id=rule_id, attempt=attempt, stage=stage,
        prompt_system=prompt_system, prompt_user=prompt_user,
        response_raw=response_raw, parsed=parsed, errors=errs,
        error_category=cat, latency_ms=latency,
        tokens_approx=len(prompt_system)//4 + len(prompt_user)//4 + len(response_raw)//4,
        model=model, temperature=temp,
    )
