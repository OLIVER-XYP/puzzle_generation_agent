"""Layered prompt construction with TODO-list planning (P0).

Architecture:
  L0  System (cached)     — role + output format + planning protocol
  L1  Rule (cached)       — rule_content + structured constraints
  L2  Examples (dynamic)  — smart few-shot selection (difficulty × diversity)
  L3  Task (per-request)  — target difficulty, dedup hints, self-correction errors
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import json


# ======================================================================
# L0 — Role-specific system prompts
# ======================================================================

GENERATOR_SYSTEM = """You are an EXPERT puzzle designer. Your ONLY job is to create NEW, valid puzzles.

## Protocol (MUST follow)

### Phase 1: TODO List
Before generating ANY output, write a TODO list inside <planning> tags:

<planning>
- [ ] Check constraint 1: [list each constraint from the rule]
- [ ] Check constraint 2: [...]
- [ ] Design puzzle structure: [grid size / word list / numbers / ...]
- [ ] Verify uniqueness: [how you'll ensure solvable + correct]
- [ ] Verify format matches examples
- [ ] Check against examples: [confirm your puzzle is different from ALL examples]
</planning>

Each [ ] item MUST be checked before you output the final JSON.

### Phase 2: Execute
After the <planning> block, output valid JSON:
```json
{
  "planning": "your thinking...",
  "question": "...",
  "answer": "[[...]]"
}
```

## Critical Rules
- Output VALID JSON. Escape quotes inside strings with \\\".
- Answer MUST be wrapped in [[ and ]].
- NEVER copy an example verbatim.
- If any constraint fails in Phase 1, STOP and restart.
- The <planning> block is NOT optional. Output it every time.
"""

SOLVER_SYSTEM = """You are a Meticulous puzzle SOLVER. Your ONLY job is to solve puzzles correctly.

## Protocol
1. Read the rule and puzzle carefully
2. Solve step by step, showing your reasoning in <solving> tags
3. Output ONLY the final answer wrapped in [[...]]

## Rules
- Follow the rule EXACTLY. Do not assume anything not stated.
- Double-check your solution against ALL constraints before outputting.
- Temperature is set to 0 for deterministic results.
"""

REVIEWER_SYSTEM = """You are a puzzle quality REVIEWER. Your job is to compare two answers.

## Protocol
1. Receive the puzzle, the Generator's answer, and the Solver's answer
2. Check if they are equivalent (different formatting may still be correct)
3. Check if the answer satisfies all rule constraints
4. Output a structured verdict:

```json
{
  "verdict": "PASS" | "FAIL",
  "generator_correct": true/false,
  "solver_correct": true/false,
  "answers_match": true/false,
  "issues": ["... any problems found ..."],
  "score": 1-10
}
```

A score of 1-3 means the puzzle is likely flawed.
A score of 8-10 means the puzzle is excellent and likely valid.
"""

REWRITER_SYSTEM = """You are a query understanding specialist. Parse user requests into structured intents.

## Protocol
Given a user message, output JSON with the parsed intent(s):

```json
{
  "intents": [
    {"action": "LIST_RULES" | "INSPECT_RULE" | "GENERATE" | "VALIDATE" | "EXPORT" | "STATS" | "HELP",
     "params": {...}}
  ],
  "missing_fields": ["... things the user didn't specify ..."],
  "clarify_question": "Return a friendly question to ask if information is missing, else null",
  "confidence": 0.0-1.0
}
```

## Multi-intent examples
"看看规则10和25，各出3道题" →
  intents: [INSPECT(10), INSPECT(25), GENERATE(10,3), GENERATE(25,3)]

"出几道简单的数学题" →
  intents: [GENERATE({domain: math}, 3, easy)]
  missing: ["具体规则ID"]
  clarify: "数学类包含规则9-17和25，你想出哪种类型的？或者全部都要？"

If the user is vague, set clarify_question, don't guess.
"""


# ======================================================================
# L1 — Rule prompt
# ======================================================================

def build_rule_prompt(rule_content: str, rule_title: str) -> str:
    return f"""## Puzzle Rule: {rule_title}

{rule_content}

## Format Notes
- Match the EXACT format of the examples below (question layout, answer format)
- If the rule involves a grid, preserve the exact grid dimensions and notation
"""


# ======================================================================
# L2 — Example selection (enhanced: difficulty × topic diversity)
# ======================================================================

class ExampleSelector:
    """Select diverse, difficulty-matched examples for few-shot prompting."""

    def __init__(self, examples: List[Dict[str, Any]]):
        self.examples = examples
        self._buckets: Dict[str, List[int]] = dict(short=[], medium=[], long=[])
        self._topic_groups: Dict[str, List[int]] = {}
        self._build_index()

    def _build_index(self):
        for i, ex in enumerate(self.examples):
            qlen = len(ex["question"])
            if qlen < 300:
                self._buckets["short"].append(i)
            elif qlen < 600:
                self._buckets["medium"].append(i)
            else:
                self._buckets["long"].append(i)

            # Group by first significant word in question (rough topic clustering)
            q = ex["question"].strip().lower()
            first_word = q.split()[0] if q.split() else "other"
            if first_word.isdigit():
                first_word = "numeric"
            elif len(first_word) < 3:
                first_word = q.split()[1] if len(q.split()) > 1 else "other"
            self._topic_groups.setdefault(first_word[:10], []).append(i)

    def select(self, n: int = 3, target_bucket: str = "medium",
               exclude_fingerprints: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Select diverse examples: same-diff + cross-diff + different-topic."""
        exclude_set = set(exclude_fingerprints or [])
        selected: List[int] = []

        def _pick(pool: List[int], prefer_fresh: bool = True) -> Optional[int]:
            candidates = [i for i in pool if self._fp(i) not in exclude_set
                         and i not in selected]
            if not candidates:
                return None
            if prefer_fresh and len(candidates) > 1:
                # deterministic-ish selection based on hash
                idx = candidates[hash(str(selected)) % len(candidates)]
                return idx
            return candidates[0]

        # 1. Same difficulty bucket
        same = _pick(self._buckets.get(target_bucket, []))
        if same is not None:
            selected.append(same)

        # 2. Cross-difficulty (diversity)
        for b in ["medium", "short", "long"]:
            if b != target_bucket and len(selected) < 2:
                cross = _pick(self._buckets.get(b, []))
                if cross is not None:
                    selected.append(cross)

        # 3. Different topic from already selected
        selected_topics = set()
        for s in selected:
            selected_topics.add(self._topic_of(s))
        remaining_slots = n - len(selected)
        topic_candidates = []
        for topic, idxs in self._topic_groups.items():
            if topic not in selected_topics:
                topic_candidates.extend(idxs)
        for _ in range(remaining_slots):
            extra = _pick(topic_candidates, prefer_fresh=True)
            if extra is not None:
                selected.append(extra)
                topic_candidates = [i for i in topic_candidates if i != extra]

        # Fill remaining
        all_ids = [i for i in range(len(self.examples))
                   if self._fp(i) not in exclude_set and i not in selected]
        while len(selected) < n and all_ids:
            idx = all_ids.pop(0)
            selected.append(idx)

        return [self.examples[i] for i in selected[:n]]

    def _fp(self, idx: int) -> str:
        return "ex:" + hashlib.md5(
            self.examples[idx]["question"][:100].encode()).hexdigest()[:8]

    def _topic_of(self, idx: int) -> str:
        q = self.examples[idx]["question"].strip().lower().split()
        return q[0] if q else "other"


def format_examples(examples: List[Dict[str, Any]]) -> str:
    parts = []
    for i, ex in enumerate(examples):
        parts.append(
            f"### Example {i+1}\n"
            f"Question: {ex['question']}\n"
            f"Answer: {ex['answer']}"
        )
    return "\n\n".join(parts)


# ======================================================================
# L3 — Task prompt
# ======================================================================

def build_task_prompt(
    target: Dict[str, Any],
    prior_errors: Optional[List[str]] = None,
    existing_questions: Optional[List[str]] = None,
) -> str:
    parts = ["## Task\n"]

    if target:
        parts.append(f"Target difficulty: question length {target.get('q_len_min','?')}-"
                     f"{target.get('q_len_max','?')} chars, average ~{target.get('q_len_avg','?')}.")

    if prior_errors:
        parts.append("\n### ⚠️ Previous Attempt Failed — FIX THESE ERRORS:")
        for err in prior_errors:
            parts.append(f"  - {err}")
        parts.append("Your next attempt MUST address ALL of the above errors.")

    if existing_questions:
        parts.append(f"\nAvoid topics similar to: {', '.join(existing_questions[:5])}")

    parts.append("\nRemember: output the <planning> block first, then the JSON.")
    return "\n".join(parts)


# ======================================================================
# Token management
# ======================================================================

def estimate_tokens(text: str) -> int:
    return len(text) // 4


class PromptBudget:
    def __init__(self, max_input_tokens: int = 16000):
        self.max_input = max_input_tokens
        self.used = 0

    def can_fit(self, text: str) -> bool:
        return self.used + estimate_tokens(text) < self.max_input

    def add(self, text: str):
        self.used += estimate_tokens(text)

    def truncate_examples(self, examples: List[Dict[str, Any]],
                          max_tokens: int = 3000) -> List[Dict[str, Any]]:
        kept = list(examples)
        while kept and estimate_tokens(format_examples(kept)) > max_tokens:
            kept = kept[:-1]
        return kept
