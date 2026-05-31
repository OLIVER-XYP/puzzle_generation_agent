"""Teacher model client (DeepSeek, OpenAI-compatible).

Used to produce reasoning traces (and optionally an independent answer for
cross-checking). The deterministic solver remains authoritative for ground truth.
The teacher is optional: if disabled or no API key, a deterministic template
trace is used so the pipeline runs fully offline.
"""
from __future__ import annotations
from typing import Any, Dict, Optional


class Teacher:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg.get("teacher", {})
        self.enabled = bool(self.cfg.get("enabled")) and bool(self.cfg.get("api_key"))
        self._client = None
        if self.enabled:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.cfg["api_key"],
                    base_url=self.cfg.get("base_url", "https://api.deepseek.com"),
                )
            except Exception as e:  # pragma: no cover
                self.enabled = False
                self._err = str(e)

    def _chat(self, system: str, user: str) -> Optional[str]:
        if not self.enabled:
            return None
        try:
            resp = self._client.chat.completions.create(
                model=self.cfg.get("model", "deepseek-v4-pro"),
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=float(self.cfg.get("temperature", 0.2)),
                max_tokens=int(self.cfg.get("max_tokens", 1024)),
            )
            return resp.choices[0].message.content
        except Exception:  # pragma: no cover - network/runtime issues
            return None

    def reasoning_trace(self, rule_content: str, question: str, answer: str,
                        fallback_steps: list[str]) -> str:
        """Produce a step-by-step reasoning trace for the (verified) answer."""
        if self.enabled:
            sys = ("You are a meticulous puzzle solver. Given the rule, the puzzle, "
                   "and the verified correct answer, write a concise step-by-step "
                   "reasoning trace that derives that answer. Do not change the answer.")
            usr = (f"### Rule\n{rule_content}\n\n### Puzzle\n{question}\n\n"
                   f"### Verified answer\n{answer}\n\n### Task\nWrite the reasoning trace.")
            out = self._chat(sys, usr)
            if out:
                return out.strip()
        # deterministic fallback
        return "\n".join(fallback_steps) if fallback_steps else \
            "Apply the rule constraints and solve deterministically; verified answer: " + answer

    def independent_answer(self, rule_content: str, question: str) -> Optional[str]:
        """Optional cross-check: ask the teacher to solve from scratch."""
        if not self.enabled:
            return None
        sys = ("You are a puzzle solver. Solve strictly per the rule and output ONLY "
               "the final answer wrapped in double square brackets [[...]].")
        usr = f"### Rule\n{rule_content}\n\n### Puzzle\n{question}"
        return self._chat(sys, usr)
