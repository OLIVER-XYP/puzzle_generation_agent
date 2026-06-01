"""Run summary aggregation (P1).

After a generation run completes, the `summarize` graph node aggregates
everything into a single RunSummary:

  - production: total accepted, per-rule counts, rejection reasons
  - quality:    tracer pass-rate, SFT candidates
  - tooling:    function-calling rounds per agent
  - memory:     persisted session stats

Interfaces:
  RunSummary           — dataclass holding the aggregated view
  build_run_summary()  — construct one from graph state + context
  format_summary_text()— human-readable report (also RunSummary.to_text())
  llm_narrative()      — optional one-paragraph NL summary via the LLM
"""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class RunSummary:
    """Aggregated view of a single generation run."""
    # production
    total_accepted: int = 0
    total_rejected: int = 0
    per_rule: Dict[str, int] = field(default_factory=dict)
    rejection_reasons: Dict[str, int] = field(default_factory=dict)
    requested_rules: List[str] = field(default_factory=list)
    # quality (tracer)
    llm_calls: int = 0
    llm_failed: int = 0
    pass_rate: str = "0.0%"
    by_role: Dict[str, int] = field(default_factory=dict)
    sft_candidates: List[Dict[str, Any]] = field(default_factory=list)
    # tooling
    tools_enabled: bool = False
    tool_rounds: Dict[str, int] = field(default_factory=dict)
    total_tool_rounds: int = 0
    # memory
    session_id: str = ""
    memory_pass_rate: float = 0.0
    # meta
    nl_summary: str = ""
    nl_response: str = ""    # rendered answer for non-generation commands (list/inspect/help)
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "production": {
                "total_accepted": self.total_accepted,
                "total_rejected": self.total_rejected,
                "per_rule": self.per_rule,
                "rejection_reasons": self.rejection_reasons,
                "requested_rules": self.requested_rules,
            },
            "quality": {
                "llm_calls": self.llm_calls,
                "llm_failed": self.llm_failed,
                "pass_rate": self.pass_rate,
                "by_role": self.by_role,
                "sft_candidates": self.sft_candidates,
            },
            "tooling": {
                "enabled": self.tools_enabled,
                "rounds": self.tool_rounds,
                "total_rounds": self.total_tool_rounds,
            },
            "memory": {
                "session_id": self.session_id,
                "pass_rate": self.memory_pass_rate,
            },
            "nl_summary": self.nl_summary,
            "nl_response": self.nl_response,
            "generated_at": self.generated_at,
        }

    def to_text(self) -> str:
        return format_summary_text(self)


def build_run_summary(state: Dict[str, Any], ctx: Any = None,
                      with_narrative: bool = False) -> RunSummary:
    """Construct a RunSummary from final graph state + optional Context.

    Args:
        state: the final GraphState (accepted_records, rejected_log, ...)
        ctx:   the graph Context (gives access to pipeline tool stats, memory)
        with_narrative: if True and an LLM is available, add a one-paragraph NL summary
    """
    s = RunSummary()
    s.nl_response = state.get("nl_response", "")

    # --- production ---
    records = state.get("accepted_records", [])
    rejected = state.get("rejected_log", [])
    s.total_accepted = len(records)
    s.total_rejected = len(rejected)

    per_rule = Counter()
    for rec in records:
        per_rule[rec.get("rule_id", "?")] += 1
    s.per_rule = dict(per_rule)

    reason_counts = Counter()
    for r in rejected:
        for reason in r.get("reasons", []):
            # collapse parametrized reasons to their head ("structural_invalid: ..." -> "structural_invalid")
            reason_counts[reason.split(":")[0].strip()] += 1
    s.rejection_reasons = dict(reason_counts.most_common(10))

    jobs = state.get("jobs", [])
    s.requested_rules = [j.get("rule_id") for j in jobs] if jobs else state.get("rules", [])

    # --- quality (tracer) ---
    try:
        from .tracer import get_tracer
        tracer = get_tracer()
        t = tracer.summary()
        s.llm_calls = t.get("total_calls", 0)
        s.llm_failed = t.get("failed", 0)
        s.pass_rate = t.get("pass_rate", "0.0%")
        s.by_role = t.get("by_role", {})
        if s.llm_failed > 0:
            decisions = tracer.generate_sft_report()
            for rid, d in decisions.items():
                if d.priority in ("HIGH", "MEDIUM"):
                    s.sft_candidates.append({
                        "rule_id": rid,
                        "priority": d.priority,
                        "failure_rate": round(d.failure_rate, 3),
                        "samples_needed": d.samples_needed,
                        "top_errors": [e[0] for e in d.top_errors[:2]],
                    })
    except Exception:
        pass

    # --- tooling ---
    if ctx is not None and hasattr(ctx, "use_tools"):
        s.tools_enabled = ctx.use_tools
        if hasattr(ctx, "pipeline") and hasattr(ctx.pipeline, "get_tool_stats"):
            ts = ctx.pipeline.get_tool_stats()
            s.tool_rounds = {
                "generator": ts.get("generator", {}).get("tool_rounds", 0),
                "solver": ts.get("solver", {}).get("tool_rounds", 0),
                "reviewer": ts.get("reviewer", {}).get("tool_rounds", 0),
            }
            s.total_tool_rounds = ts.get("total_rounds", 0)

    # --- memory ---
    if ctx is not None and hasattr(ctx, "memory"):
        try:
            stats = ctx.memory.get_session_stats()
            if "error" not in stats:
                s.session_id = stats.get("session_id", "")
                s.memory_pass_rate = stats.get("pass_rate", 0.0)
        except Exception:
            pass

    # --- optional NL narrative ---
    if with_narrative and ctx is not None and hasattr(ctx, "pipeline"):
        try:
            s.nl_summary = llm_narrative(s, ctx.pipeline.client)
        except Exception:
            s.nl_summary = ""

    return s


def format_summary_text(s: RunSummary) -> str:
    """Render a RunSummary as a human-readable report."""
    lines = []
    lines.append("=" * 56)
    lines.append("  RUN SUMMARY")
    lines.append("=" * 56)
    if s.nl_response:
        lines.append("  Command response:")
        for ln in s.nl_response.splitlines():
            lines.append(f"    {ln}")
        lines.append("-" * 56)
    lines.append(f"  Accepted : {s.total_accepted}   Rejected: {s.total_rejected}")
    if s.per_rule:
        per = ", ".join(f"R{k}={v}" for k, v in sorted(s.per_rule.items()))
        lines.append(f"  Per-rule : {per}")
    if s.rejection_reasons:
        rej = ", ".join(f"{k}({v})" for k, v in s.rejection_reasons.items())
        lines.append(f"  Rejected reasons: {rej}")
    lines.append("-" * 56)
    lines.append(f"  LLM calls: {s.llm_calls}  failed={s.llm_failed}  pass_rate={s.pass_rate}")
    if s.by_role:
        roles = ", ".join(f"{k}={v}" for k, v in s.by_role.items())
        lines.append(f"  By role  : {roles}")
    if s.sft_candidates:
        lines.append("  SFT candidates:")
        for c in s.sft_candidates:
            errs = ", ".join(c["top_errors"])
            lines.append(f"    - R{c['rule_id']} {c['priority']} "
                         f"fail={c['failure_rate']:.0%} need={c['samples_needed']} [{errs}]")
    lines.append("-" * 56)
    lines.append(f"  Tools    : {'ON' if s.tools_enabled else 'OFF'}  "
                 f"total_rounds={s.total_tool_rounds}")
    if s.tool_rounds:
        tr = ", ".join(f"{k}={v}" for k, v in s.tool_rounds.items())
        lines.append(f"  Tool use : {tr}")
    if s.session_id:
        lines.append("-" * 56)
        lines.append(f"  Memory   : session={s.session_id} pass_rate={s.memory_pass_rate}")
    if s.nl_summary:
        lines.append("-" * 56)
        lines.append("  Narrative:")
        lines.append(f"    {s.nl_summary}")
    lines.append("=" * 56)
    return "\n".join(lines)


def format_summary_dict(d: Dict[str, Any]) -> str:
    """Render the dict produced by RunSummary.to_dict() (e.g. from graph state).

    Lets callers print a summary without reconstructing the RunSummary object.
    """
    s = RunSummary()
    prod = d.get("production", {})
    qual = d.get("quality", {})
    tool = d.get("tooling", {})
    mem = d.get("memory", {})
    s.total_accepted = prod.get("total_accepted", 0)
    s.total_rejected = prod.get("total_rejected", 0)
    s.per_rule = prod.get("per_rule", {})
    s.rejection_reasons = prod.get("rejection_reasons", {})
    s.requested_rules = prod.get("requested_rules", [])
    s.llm_calls = qual.get("llm_calls", 0)
    s.llm_failed = qual.get("llm_failed", 0)
    s.pass_rate = qual.get("pass_rate", "0.0%")
    s.by_role = qual.get("by_role", {})
    s.sft_candidates = qual.get("sft_candidates", [])
    s.tools_enabled = tool.get("enabled", False)
    s.tool_rounds = tool.get("rounds", {})
    s.total_tool_rounds = tool.get("total_rounds", 0)
    s.session_id = mem.get("session_id", "")
    s.memory_pass_rate = mem.get("pass_rate", 0.0)
    s.nl_summary = d.get("nl_summary", "")
    s.nl_response = d.get("nl_response", "")
    return format_summary_text(s)


def llm_narrative(s: RunSummary, client: Any) -> str:
    """Generate a one-paragraph Chinese narrative summary via the LLM.

    Returns "" if the client is disabled or the call fails.
    """
    if client is None or not getattr(client, "enabled", False):
        return ""
    facts = (
        f"Accepted={s.total_accepted}, Rejected={s.total_rejected}, "
        f"per_rule={s.per_rule}, pass_rate={s.pass_rate}, "
        f"tool_rounds={s.total_tool_rounds}, "
        f"sft_candidates={[c['rule_id'] for c in s.sft_candidates]}"
    )
    system = "你是数据生成流水线的报告助手。用一段简洁的中文总结本次运行，不超过120字。"
    user = f"本次运行统计：{facts}\n请总结产出、质量与需要关注的规则。"
    try:
        raw, _ = client.chat(system, user, temperature=0.3, max_tokens=256)
        return (raw or "").strip()
    except Exception:
        return ""
