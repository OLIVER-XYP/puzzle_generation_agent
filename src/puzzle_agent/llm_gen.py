"""Production-grade LLM puzzle generator.

Self-correction loop: generate → validate → if errors, retry with feedback.
Layered prompt with token budget and dynamic example selection.
"""
from __future__ import annotations
import json
import os
import re
import time
import hashlib
from typing import Any, Dict, List, Optional, Tuple

from .prompt_builder import (
    GENERATOR_SYSTEM,
    build_rule_prompt,
    ExampleSelector,
    format_examples,
    build_task_prompt,
    PromptBudget,
    estimate_tokens,
)


class LlmGen:
    """Production puzzle generator with self-correction."""

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg.get("generator", {})
        self.max_attempts = self.cfg.get("max_generation_attempts", 3)
        self.max_tokens = self.cfg.get("max_output_tokens", 4096)

        api_key = self.cfg.get("api_key", "")
        if not api_key:
            api_key = os.environ.get("DEEPSEEK_API_KEY", "")
            self.cfg["api_key"] = api_key

        self._client = None
        self._reasoner_client = None
        self.enabled = False

        if api_key:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=api_key,
                    base_url=self.cfg.get("base_url", "https://api.deepseek.com"),
                )
                # Also try reasoner client for CoT tasks
                self._reasoner_client = OpenAI(
                    api_key=api_key,
                    base_url=self.cfg.get("base_url", "https://api.deepseek.com"),
                )
                self.enabled = True
            except Exception:
                self.enabled = False

        # Per-rule example selectors (lazy init)
        self._selectors: Dict[str, ExampleSelector] = {}
        # Stats
        self.stats: Dict[str, Any] = dict(calls=0, tokens_in=0, tokens_out=0,
                                           retries=0, failures=0)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def generate(
        self,
        rule_content: str,
        rule_title: str,
        examples: List[Dict[str, Any]],
        target: Dict[str, Any],
        *,
        rule_id: str = "",
        prior_failures: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Generate one new puzzle with self-correction loop.

        Errors from structural validation are fed back into the prompt so the
        LLM can fix them on the next attempt.
        """
        # Init selector
        key = rule_title
        if key not in self._selectors:
            self._selectors[key] = ExampleSelector(examples)
        selector = self._selectors[key]

        # Determine difficulty bucket
        avg_len = target.get("q_len_avg", 400) if target else 400
        if avg_len < 300:
            bucket = "short"
        elif avg_len < 600:
            bucket = "medium"
        else:
            bucket = "long"

        errors: List[str] = list(prior_failures or [])

        for attempt in range(self.max_attempts):
            # Build prompt (includes previous errors for self-correction)
            selected = selector.select(n=3, target_bucket=bucket)
            prompt = self._build_prompt(
                rule_content, rule_title, selected, target, errors, attempt, rule_id
            )

            # Call LLM
            raw = self._chat(GENERATOR_SYSTEM, prompt,
                             max_tokens=self.max_tokens,
                             temperature=0.7 + attempt * 0.1)
            self.stats["calls"] += 1

            if not raw:
                self.stats["failures"] += 1
                continue

            # Parse
            parsed = self._parse(raw)
            if parsed is None:
                errors = ["Failed to parse valid JSON. Ensure output has valid JSON with 'question' and 'answer' fields. Escape double quotes inside strings."]
                self.stats["retries"] += 1
                continue

            # Validate format
            fmt_errors = self._validate_format(parsed)
            if fmt_errors:
                errors = fmt_errors
                self.stats["retries"] += 1
                continue

            # Structural validation — feed back to self-correction
            from .validators import validate_answer
            struct_ok, struct_errors = validate_answer(
                rule_id, parsed["answer"], parsed["question"]
            )
            if not struct_ok:
                errors = struct_errors
                self.stats["retries"] += 1
                continue

            # Success
            return parsed

        self.stats["failures"] += 1
        return None

    def cross_check(self, rule_content: str, question: str) -> Optional[str]:
        """Independent LLM solve using reasoner model for accuracy."""
        if not self.enabled:
            return None
        prompt = build_rule_prompt(rule_content, "") + "\n\n" + \
                 f"## Puzzle\n{question}\n\n## Task\nSolve this puzzle step by step. Output ONLY the final answer wrapped in [[...]]."
        raw = self._chat(
            "You are a meticulous puzzle solver. Solve EXACTLY according to the rule. "
            "Output only the answer in [[...]].",
            prompt,
            max_tokens=2048,
            temperature=0.0,
        )
        return raw.strip() if raw else None

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _build_prompt(self, rule_content, rule_title, examples, target, errors, attempt, rule_id=""):
        parts = []

        # L1 — Rule
        rule_section = build_rule_prompt(rule_content, rule_title)
        parts.append(rule_section)

        # L1.5 — Rule-specific format hints (for complex grid puzzles)
        from .rule_hints import get_hint
        hint = get_hint(rule_id)
        if hint:
            parts.append(hint)

        # L2 — Examples
        budget = PromptBudget()
        budget.add(GENERATOR_SYSTEM)
        budget.add(rule_section)
        budget.add(hint)
        trimmed = budget.truncate_examples(examples, max_tokens=4000)
        parts.append("## Examples\n" + format_examples(trimmed))

        # L3 — Task (with self-correction errors)
        task = build_task_prompt(target, errors)
        parts.append(task)

        # Assemble
        full = "\n\n".join(parts)
        self.stats["tokens_in"] += estimate_tokens(GENERATOR_SYSTEM) + estimate_tokens(full)
        return full

    def _chat(self, system: str, user: str, max_tokens: int = 4096,
              temperature: float = 0.7, use_reasoner: bool = False) -> Optional[str]:
        if not self.enabled:
            return None
        client = self._client
        model = self.cfg.get("model", "deepseek-v4-pro")
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = resp.choices[0].message.content
            if content:
                self.stats["tokens_out"] += estimate_tokens(content)
            return content
        except Exception:
            return None

    def _parse(self, raw: str) -> Optional[Dict[str, Any]]:
        """Parse LLM output with multiple fallback strategies."""
        raw = raw.strip()

        # Strategy 1: ```json ... ``` block
        m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(1))
                if "question" in data and "answer" in data:
                    return data
            except json.JSONDecodeError:
                pass

        # Strategy 2: Raw JSON
        try:
            data = json.loads(raw)
            if "question" in data and "answer" in data:
                return data
        except json.JSONDecodeError:
            pass

        # Strategy 3: Extract question/answer field by regex
        qm = re.search(r'"question"\s*:\s*"(.*?)"(?:\s*[,}])', raw, re.DOTALL)
        am = re.search(r'"answer"\s*:\s*"(.*?)"(?:\s*[,}])', raw, re.DOTALL)
        if qm and am:
            q = self._unescape(qm.group(1))
            a = self._unescape(am.group(1))
            return {"question": q, "answer": a}

        # Strategy 4: Look for [[...]] pattern as answer, rest as question
        ans_match = re.search(r"\[\[(.+?)\]\]", raw, re.DOTALL)
        if ans_match:
            answer = f"[[{ans_match.group(1)}]]"
            # Question is everything before the answer
            question = raw[:ans_match.start()].strip()
            if len(question) > 20:
                return {"question": question, "answer": answer}

        return None

    @staticmethod
    def _unescape(s: str) -> str:
        """Handle JSON-escaped strings."""
        s = s.replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t')
        return s

    def _validate_format(self, parsed: Dict[str, Any]) -> List[str]:
        errors = []

        q = parsed.get("question", "")
        a = parsed.get("answer", "")

        if len(q) < 20:
            errors.append("Question too short (< 20 chars). Provide a complete puzzle statement.")

        if not a.strip().startswith("[["):
            errors.append("Answer must be wrapped in [[ and ]]. Current: " + a[:50])

        if len(a) < 6:
            errors.append("Answer too short. Provide the complete solution.")

        return errors

    @staticmethod
    def answers_match(a: str, b: str) -> bool:
        """Compare two answers, normalizing formatting."""
        if not a or not b:
            return False

        def norm(s):
            s = s.strip()
            if s.startswith("[["):
                s = s[2:]
            if s.endswith("]]"):
                s = s[:-2]
            s = s.replace(",", " ").replace("\n", " ")
            s = re.sub(r"\s+", " ", s).strip().lower()
            return s

        return norm(a) == norm(b)


# ------------------------------------------------------------------ #
# Minimal rule wrapper
# ------------------------------------------------------------------ #

class LlmRule:
    def __init__(self, rule_id: str, title: str, tag: str, rule_content: str,
                 examples: List[Dict[str, Any]], target: Dict[str, Any]):
        self.rule_id = rule_id
        self.title = title
        self.tag = tag
        self.rule_content = rule_content
        self.examples = examples
        self.target = target
        self._eval_keys = [
            f"LLM:{rule_id}:" + hashlib.sha1(
                ex["question"].strip()[:100].encode()
            ).hexdigest()[:8]
            for ex in examples
        ]

    def eval_keys(self) -> List[str]:
        return self._eval_keys
