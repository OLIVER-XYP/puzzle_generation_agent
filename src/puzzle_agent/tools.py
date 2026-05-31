"""Concrete tool implementations (P2).

Tools are deterministic, fast functions that verify puzzle properties without
LLM inference. They reduce uncertainty by providing hard guarantees.

Tool registry allows the agent and LLM to call these tools during generation.
"""
from __future__ import annotations
import re
import hashlib
from collections import Counter
from fractions import Fraction
from itertools import permutations, product
from typing import Any, Callable, Dict, List, Optional, Tuple


# ======================================================================
# Tool: validate_grid
# ======================================================================

def validate_grid(answer: str, rule_type: str) -> Tuple[bool, List[str]]:
    """Validate that a grid answer satisfies rule-specific constraints.

    Args:
        answer: Raw answer string with [[...]] wrapper
        rule_type: "latin_square" | "sudoku" | "generic_grid"

    Returns:
        (valid: bool, errors: list of error messages)
    """
    grid, err = _parse_grid(answer)
    if err:
        return False, [err]

    errors = []

    if rule_type == "latin_square":
        if not _is_digit_grid(grid):
            return False, ["Grid contains non-numeric values"]
        nums = [[int(c) for c in row] for row in grid]
        n = len(nums)
        valid = set(range(1, n + 1))
        for i, row in enumerate(nums):
            row_set = set(row)
            if row_set != valid:
                errors.append(f"Row {i+1}: expected {valid}, got {row_set}")
        for j in range(n):
            col_set = set(nums[i][j] for i in range(n))
            if col_set != valid:
                errors.append(f"Col {j+1}: expected {valid}, got {col_set}")

    elif rule_type == "sudoku":
        if len(grid) != 9 or any(len(r) != 9 for r in grid):
            errors.append("Grid must be 9x9")
        else:
            if not _is_digit_grid(grid):
                return False, ["Sudoku grid must be all digits"]
            nums = [[int(c) for c in row] for row in grid]
            valid = set(range(1, 10))
            for i in range(9):
                if set(nums[i]) != valid:
                    errors.append(f"Row {i+1} invalid")
            for j in range(9):
                if set(nums[i][j] for i in range(9)) != valid:
                    errors.append(f"Col {j+1} invalid")
            for br in range(0, 9, 3):
                for bc in range(0, 9, 3):
                    box = set()
                    for r in range(3):
                        for c in range(3):
                            box.add(nums[br+r][bc+c])
                    if box != valid:
                        errors.append(f"Box ({br//3+1},{bc//3+1}) invalid")

    elif rule_type == "generic_grid":
        if not _is_rectangular(grid):
            errors.append("Grid rows have inconsistent lengths")

    return len(errors) == 0, errors


def _parse_grid(answer: str) -> Tuple[List[List[str]], str]:
    body = answer.strip()
    if body.startswith("[["):
        body = body[2:]
    if body.endswith("]]"):
        body = body[:-2]
    rows = [r.strip().split() for r in body.split(",") if r.strip()]
    if not rows:
        return [], "Cannot parse grid: no rows found"
    return rows, ""


def _is_digit_grid(grid: List[List[str]]) -> bool:
    return all(cell.lstrip("-").isdigit() for row in grid for cell in row)


def _is_rectangular(grid: List[List[str]]) -> bool:
    if not grid:
        return False
    n = len(grid[0])
    return all(len(row) == n for row in grid)


# ======================================================================
# Tool: check_duplicate
# ======================================================================

def check_duplicate(question: str, eval_set: List[Dict[str, Any]],
                   existing: List[Dict[str, Any]] = None) -> Tuple[bool, str]:
    """Check if a generated question duplicates an existing one.

    Uses fuzzy matching: compares first 200 chars after normalization.
    """
    q_key = _normalize(question[:200])
    eval_key = set()
    for ex in eval_set:
        eval_key.add(_normalize(ex["question"][:200]))

    if q_key in eval_key:
        return True, "Duplicate of eval set question"

    if existing:
        for ex in existing:
            if _normalize(ex.get("question", "")[:200]) == q_key:
                return True, "Duplicate of already-generated question"

    return False, ""


def _normalize(s: str) -> str:
    """Normalize text for duplicate detection."""
    s = s.lower().strip()
    s = re.sub(r'\s+', ' ', s)          # collapse whitespace
    s = re.sub(r'[^\w\s]', '', s)       # remove punctuation
    return s


# ======================================================================
# Tool: measure_difficulty
# ======================================================================

def measure_difficulty(question: str, rule_id: str = "") -> Dict[str, Any]:
    """Measure structural difficulty metrics for a question.

    Returns:
        {q_len, has_grid, grid_size, n_numbers, n_words, estimated_bucket}
    """
    q_len = len(question)
    tokens = question.split()

    has_grid = "X" in tokens and len([t for t in tokens if t == "X"]) > 3

    numbers = re.findall(r'\b\d+\b', question)
    n_numbers = len(numbers)

    words = [t for t in tokens if t.isalpha() and len(t) > 1]
    n_words = len(words)

    grid_size = 0
    if has_grid:
        lines = [l.strip() for l in question.split("\n") if l.strip()]
        grid_lines = [l for l in lines if all(c in "0123456789X\t " for c in l)]
        if grid_lines:
            grid_size = max(len(l.split()) for l in grid_lines)

    # bucket
    if q_len < 300:
        bucket = "short"
    elif q_len < 600:
        bucket = "medium"
    else:
        bucket = "long"

    return {
        "q_len": q_len, "has_grid": has_grid, "grid_size": grid_size,
        "n_numbers": n_numbers, "n_words": n_words, "bucket": bucket,
    }


# ======================================================================
# Tool: format_converter
# ======================================================================

def format_converter(raw: str, target_format: str) -> Tuple[str, List[str]]:
    """Convert between answer formats.

    Supported target formats:
      - "compact_grid": "a b, c d" → no spaces, pipe separator
      - "expanded_grid": "a b, c d" → aligned columns
      - "json_list": "a b, c d" → [["a","b"],["c","d"]]
      - "bare": strip [[...]] wrapper
    """
    errors = []
    body = raw.strip()
    wrapped = body.startswith("[[") and body.endswith("]]")
    if wrapped and target_format != "bare":
        body = body[2:-2]

    if target_format == "bare":
        return body, []

    # Parse as grid
    rows = [r.strip().split() for r in body.split(",") if r.strip()]

    if target_format == "compact_grid":
        result = "|".join("".join(str(c) for c in row) for row in rows)
    elif target_format == "expanded_grid":
        widths = [max(len(str(row[j])) for row in rows) for j in range(len(rows[0]))]
        result = "\n".join(
            " ".join(str(c).rjust(w) for c, w in zip(row, widths)) for row in rows
        )
    elif target_format == "json_list":
        import json
        result = json.dumps(rows, ensure_ascii=False)
    else:
        errors.append(f"Unknown format: {target_format}")
        result = raw

    return result, errors


# ======================================================================
# Tool: solve_bruteforce (lightweight deterministic solvers)
# ======================================================================

def solve_24points(numbers: List[int]) -> List[str]:
    """Enumerate all solutions for 24-point puzzle."""
    solutions: List[str] = []

    def _search(vals, exprs):
        if len(vals) == 1:
            if abs(vals[0] - 24) < 1e-9:
                solutions.append(_strip(exprs[0]))
            return
        n = len(vals)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                rest_v = [vals[k] for k in range(n) if k != i and k != j]
                rest_e = [exprs[k] for k in range(n) if k != i and k != j]
                a, b = vals[i], vals[j]
                ea, eb = exprs[i], exprs[j]
                ops = [(a+b, f"({ea}+{eb})"), (a-b, f"({ea}-{eb})"),
                       (a*b, f"({ea}*{eb})")]
                if b != 0:
                    ops.append((a/b, f"({ea}/{eb})"))
                for v, e in ops:
                    _search(rest_v + [v], rest_e + [e])

    vals = [Fraction(n) for n in numbers]
    exprs = [str(n) for n in numbers]
    _search(vals, exprs)
    return list(set(solutions))


def _strip(expr: str) -> str:
    if expr.startswith("(") and expr.endswith(")"):
        return expr[1:-1]
    return expr


def solve_cryptarithm(addends: List[str], result: str, cap: int = 5) -> List[Dict[str, int]]:
    """Brute-force solve a cryptarithm A+B=C. Returns up to `cap` solutions."""
    words = addends + [result]
    leading = {w[0] for w in words if len(w) > 1}
    maxlen = max(len(w) for w in words)
    assigned: Dict[str, int] = {}
    used = [False] * 10
    solutions: List[Dict[str, int]] = []

    def rec(i: int, carry: int):
        if len(solutions) >= cap:
            return
        if i == maxlen:
            if carry == 0:
                solutions.append(dict(assigned))
            return
        adds = [w[-1 - i] for w in addends if i < len(w)]
        res_letter = result[-1 - i] if i < len(result) else None
        unknown = sorted(set(l for l in adds if l not in assigned))

        def assign(idx: int):
            if len(solutions) >= cap:
                return
            if idx == len(unknown):
                s = sum(assigned[l] for l in adds) + carry
                d, c2 = s % 10, s // 10
                if res_letter is None:
                    if d == 0 and c2 == 0:
                        rec(i + 1, 0)
                    return
                if res_letter in assigned:
                    if assigned[res_letter] == d:
                        rec(i + 1, c2)
                else:
                    if not used[d] and (d != 0 or res_letter not in leading):
                        assigned[res_letter] = d
                        used[d] = True
                        rec(i + 1, c2)
                        del assigned[res_letter]
                        used[d] = False
                return

            l = unknown[idx]
            for dig in range(10):
                if used[dig]:
                    continue
                if dig == 0 and l in leading:
                    continue
                assigned[l] = dig
                used[dig] = True
                assign(idx + 1)
                del assigned[l]
                used[dig] = False

        assign(0)

    rec(0, 0)
    return solutions


# ======================================================================
# Tool Registry
# ======================================================================

class ToolRegistry:
    """Registry of all available tools with metadata."""

    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._descriptions: Dict[str, str] = {}
        self._register_defaults()

    def _register_defaults(self):
        self.register("validate_grid", validate_grid,
                      "Validate grid answer (latin_square/sudoku/generic_grid). Returns (ok, errors).")
        self.register("check_duplicate", check_duplicate,
                      "Check if question duplicates eval set. Returns (is_dup, reason).")
        self.register("measure_difficulty", measure_difficulty,
                      "Measure structural difficulty metrics. Returns {q_len, bucket, ...}.")
        self.register("format_converter", format_converter,
                      "Convert answer between formats (compact/expanded/json/bare).")
        self.register("solve_24points", solve_24points,
                      "Brute-force all 24-point solutions for given numbers.")
        self.register("solve_cryptarithm", solve_cryptarithm,
                      "Brute-force solve cryptarithm A+B=C. Returns list of assignments.")

    def register(self, name: str, fn: Callable, description: str):
        self._tools[name] = fn
        self._descriptions[name] = description

    def call(self, name: str, **kwargs) -> Any:
        if name not in self._tools:
            raise ValueError(f"Unknown tool: {name}. Available: {list(self._tools.keys())}")
        return self._tools[name](**kwargs)

    def list_tools(self) -> List[Dict[str, str]]:
        return [{"name": n, "desc": d} for n, d in self._descriptions.items()]

    def describe(self, name: str) -> str:
        return self._descriptions.get(name, "Unknown tool")


# Global singleton
_tool_registry = ToolRegistry()


def get_tools() -> ToolRegistry:
    return _tool_registry
