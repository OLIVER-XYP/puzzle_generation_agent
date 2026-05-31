"""SessionManager (P1) — First-generation vs supplement routing.

Tracks generation rounds so the agent knows whether it's starting fresh
or adding to an existing batch. Different prompt strategies for each mode:
  - First: encourages breadth, diversity across the rule space
  - Supplement: injects prior topics/difficulty/hashes for dedup + differentiation
"""
from __future__ import annotations
import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set


@dataclass
class GenerationBatch:
    """One generation round for a rule."""
    batch_id: str
    rule_id: str
    rule_title: str
    mode: str              # "first" | "supplement"
    count: int
    passed: int
    records: List[Dict[str, Any]] = field(default_factory=list)
    created_at: str = ""

    @property
    def question_hashes(self) -> Set[str]:
        return {self._hash_q(r["question"]) for r in self.records if r.get("question")}

    @property
    def topics(self) -> List[str]:
        """Extract key topic words from generated questions."""
        topics = []
        for r in self.records:
            q = r.get("question", "")
            words = q.strip().lower().split()[:5]
            topics.append(" ".join(words))
        return topics

    @property
    def difficulty_distribution(self) -> Dict[str, int]:
        dist: Dict[str, int] = {}
        for r in self.records:
            qlen = len(r.get("question", ""))
            bucket = "short" if qlen < 300 else ("long" if qlen > 600 else "medium")
            dist[bucket] = dist.get(bucket, 0) + 1
        return dist

    @staticmethod
    def _hash_q(q: str) -> str:
        return hashlib.sha1(q.strip()[:200].encode()).hexdigest()


class SessionManager:
    """Tracks all generation rounds for multi-turn conversations."""

    def __init__(self):
        self.batches: List[GenerationBatch] = []
        self.current_rule_id: Optional[str] = None

    def start_batch(self, rule_id: str, rule_title: str, count: int) -> GenerationBatch:
        mode = "first"
        prior = self._prior_for(rule_id)
        if prior:
            mode = "supplement"

        batch = GenerationBatch(
            batch_id=str(uuid.uuid4())[:8],
            rule_id=rule_id,
            rule_title=rule_title,
            mode=mode,
            count=count,
            passed=0,
            created_at=datetime.now().isoformat(),
        )
        self.current_rule_id = rule_id
        self.batches.append(batch)
        return batch

    def add_result(self, batch_id: str, record: Dict[str, Any]):
        for b in self.batches:
            if b.batch_id == batch_id:
                b.records.append(record)
                if record.get("generator_ok"):
                    b.passed += 1
                return

    def _prior_for(self, rule_id: str) -> Optional[GenerationBatch]:
        """Find the most recent COMPLETED batch for this rule (excludes current)."""
        for b in reversed(self.batches[:-1]):  # exclude current batch (last element)
            if b.rule_id == rule_id and b.passed > 0:
                return b
        return None

    def _prior_for(self, rule_id: str, exclude_batch_id: str = "") -> Optional[GenerationBatch]:
        """Find the most recent completed batch, excluding the given ID."""
        for b in reversed(self.batches):
            if b.batch_id == exclude_batch_id:
                continue
            if b.rule_id == rule_id and b.passed > 0:
                return b
        return None

    def build_context(self, rule_id: str, current_batch_id: str = "") -> Dict[str, Any]:
        """Build a context dict to inject into the generation prompt."""
        prior = self._prior_for(rule_id, exclude_batch_id=current_batch_id)
        if not prior:
            return {"mode": "first"}

        return {
            "mode": "supplement",
            "prior_count": len(prior.records),
            "prior_topics": prior.topics[:5],
            "prior_difficulty": prior.difficulty_distribution,
            "dedup_hashes": prior.question_hashes,
        }

    def describe_context(self) -> str:
        """Human-readable context string for the LLM."""
        if not self.batches:
            return "No generation history."

        lines = []
        for b in self.batches[-3:]:
            lines.append(
                f"Rule {b.rule_id} ({b.rule_title}): {b.passed}/{b.count} passed "
                f"({b.mode} mode, {len(b.records)} records)"
            )
        return "\n".join(lines)

    def all_hashes(self) -> Set[str]:
        """All question hashes across all batches (for global dedup)."""
        hashes: Set[str] = set()
        for b in self.batches:
            hashes |= b.question_hashes
        return hashes

    def is_empty(self) -> bool:
        return len(self.batches) == 0

    def total_generated(self) -> int:
        return sum(b.passed for b in self.batches)


# ======================================================================
# Prompt templates for each mode
# ======================================================================

FIRST_GENERATION_HINT = """
## Generation Mode: FIRST BATCH
This is the first batch of puzzles for this rule. Focus on:
1. **Coverage** — produce diverse examples spanning the rule's possibility space
2. **Difficulty range** — include easy, medium, and hard variations
3. **Novelty** — avoid patterns that appear in the training examples
"""

SUPPLEMENT_HINT = """
## Generation Mode: SUPPLEMENT
Additional puzzles are being added to an existing batch. Focus on:
1. **Differentiation** — avoid topics already covered: {prior_topics}
2. **Gap filling** — the existing difficulty distribution is {prior_difficulty}.
   Generate in difficulty buckets that are underrepresented.
3. **Strict dedup** — do NOT produce questions matching any of the {prior_count}
   existing records.
"""


def get_mode_hint(context: Dict[str, Any]) -> str:
    """Return the appropriate prompt hint based on generation mode."""
    mode = context.get("mode", "first")
    if mode == "supplement":
        return SUPPLEMENT_HINT.format(
            prior_topics=", ".join(context.get("prior_topics", ["unknown"])[:5]),
            prior_difficulty=str(context.get("prior_difficulty", {})),
            prior_count=context.get("prior_count", 0),
        )
    return FIRST_GENERATION_HINT
