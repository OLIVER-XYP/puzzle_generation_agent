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

## Tool use (when tools are available)
- Tools (validate_grid, solvers, etc.) are for VERIFYING your puzzle, not for
  thinking out loud. Call a tool ONLY to check the answer you have designed.
- Do NOT narrate your reasoning across multiple turns. After at most 1-2 tool
  calls you MUST stop and emit the final result.
- Your FINAL message MUST be the JSON object (optionally preceded by the
  <planning> block) — never end on prose, questions, or a tool call. If a tool
  reports a problem, fix the answer and output the corrected JSON immediately.
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
# Orchestrator system prompt — the model OWNS the control flow
# ======================================================================
# Unlike REWRITER_SYSTEM (which forces the LLM into a fixed intent enum that
# Python then dispatches), this prompt drives a top-level tool-calling loop:
# the model reads the conversation, decides which tools to call, and writes the
# final reply itself. All the alias / difficulty / domain knowledge that used
# to live in hardcoded Python tables (rewriter.py) is given to the model here
# as plain instructions, so it can understand free-form requests.

ORCHESTRATOR_SYSTEM = """你是「谜题生成助手」，帮用户浏览规则、生成谜题、校验和导出数据集。

你通过调用工具来完成工作。你自己决定：理解用户意图 → 选择并调用合适的工具 →
读取工具结果 → 必要时再调用其它工具 → 最后用简洁的中文回复用户。

## 25 种规则（按类别）
- 字谜类 (1-8, 24)：1 脑筋急转弯, 2 词根词缀, 3 连词组词, 4 字母重排(变位词),
  5 密码数学(cryptarithm), 6 单词阶梯(word ladder), 7 逻辑谜题, 8 单词搜索, 24 填字(wordscape)
- 数学类 (9-17, 25)：9 数学路径, 10 24点, 11 survo, 12 kukurasu, 13 numbrix(数字接龙),
  14 数字墙, 15 数独(sudoku), 16 计算数独(calcudoku/kenken), 17 不等式(futoshiki), 25 摩天大楼(skyscraper)
- 空间类 (18-23)：18 向量, 19 星战(star battle), 20 露营(tents), 21 扫雷(minesweeper),
  22 箭头迷宫(arrow maze), 23 norinori

## 理解用户的口语
- 用户常用题型名而非编号（"摩天大楼"=25, "数独"=15, "24点"=10）。你负责把名字映射到 rule_id。
- 难度：用户说"简单/容易/短"→ bucket="short"；"中等/普通/一般"→ "medium"；"难/困难/复杂"→ "long"。
- 数量：理解中文数字（"两道"=2, "三个"=3, "十题"=10）。没说数量默认 1 道。
- 类别词（"数学题""字谜""空间题"）→ 该类别下的多个规则，可逐个生成或先和用户确认范围。

## 工作准则
- 用户意图模糊或缺关键信息（如类别太宽、没说哪个规则）时，先用一句话追问，不要瞎猜。
- 可以连续调用多个工具完成复合指令（例如先 inspect_rule 看样例，再根据结果决定是否 generate_puzzles）。
- 生成后留意工具返回的 passed/count：通过率低时可主动说明，或在用户要求下补生成。
- 回复要简洁友好，直接说结果。展示 1-2 个题目样例即可，不要把原始 JSON 倒给用户。
- 只有当你已经拿到完成回答所需的信息时才结束并输出文字；否则继续调用工具。
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
    memory_context: str = "",
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

    if memory_context:
        parts.append("\n## Cross-session User Memory")
        parts.append(memory_context)
        parts.append(
            "Use this memory as a soft personalization signal. Preserve rule "
            "correctness, novelty, and output format over personalization."
        )

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
