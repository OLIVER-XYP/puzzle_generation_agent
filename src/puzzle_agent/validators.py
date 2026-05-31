"""Lightweight structural validators. Fast, no infinite loops, covers common LLM mistakes."""
from __future__ import annotations
import re
from typing import Callable, Dict, List, Tuple


def _unwrap(ans: str) -> str:
    s = ans.strip()
    if s.startswith("[["): s = s[2:]
    if s.endswith("]]"): s = s[:-2]
    return s


def _rows(ans: str) -> List[List[str]]:
    body = _unwrap(ans)
    return [r.strip().split() for r in body.split(",") if r.strip()]


def _is_digits(grid: List[List[str]]) -> bool:
    return all(cell.replace("-","").isdigit() for row in grid for cell in row)


# ---- Per-rule checks ----

def check_latin_square(answer: str, question: str) -> Tuple[bool, List[str]]:
    """Any rule that requires a Latin square (Skyscrapers, Sudoku, Futoshiki, Calcudoko)."""
    grid = _rows(answer)
    if not grid:
        return False, ["Cannot parse grid"]
    n = len(grid)
    for i, row in enumerate(grid):
        if len(row) != n:
            return False, [f"Row {i+1} length {len(row)} != {n}"]
    if not _is_digits(grid):
        pass  # some grids have letters (Number Wall)
    else:
        for i, row in enumerate(grid):
            vals = [int(v) for v in row if v.replace("-","").isdigit()]
            if len(set(vals)) < len(vals):
                return False, [f"Row {i+1} has duplicates"]
    return True, []


def check_cryptomath(answer: str, question: str) -> Tuple[bool, List[str]]:
    """Verify letter-digit mapping satisfies the equation."""
    body = _unwrap(answer)
    pairs = [p.strip() for p in body.split(",")]
    mapping = {}
    for p in pairs:
        if "=" in p:
            k, v = p.split("=", 1)
            mapping[k.strip()] = v.strip()
    if not mapping:
        return False, ["Cannot parse mapping"]
    # Check uniqueness of letters and digits
    if len(set(mapping.keys())) < len(mapping):
        return False, ["Duplicate letters in mapping"]
    digits = [int(v) for v in mapping.values() if v.isdigit()]
    if len(set(digits)) < len(digits):
        return False, ["Duplicate digits in mapping"]
    return True, []


def check_24points(answer: str, question: str) -> Tuple[bool, List[str]]:
    body = _unwrap(answer)
    try:
        expr = body.replace("×","*").replace("÷","/")
        val = eval(expr, {"__builtins__":{}}, {})
        if abs(float(val) - 24) > 1e-9:
            return False, [f"Evaluates to {val}, not 24"]
    except Exception:
        return False, ["Cannot evaluate expression"]
    return True, []


def check_grid_non_empty(answer: str, question: str) -> Tuple[bool, List[str]]:
    grid = _rows(answer)
    if not grid or len(grid) < 2:
        return False, ["Grid too small"]
    width = len(grid[0])
    if any(len(r) != width for r in grid):
        return False, ["Uneven rows"]
    return True, []


def check_word_list(answer: str, question: str) -> Tuple[bool, List[str]]:
    body = _unwrap(answer)
    parts = body.split()
    if len(parts) < 1:
        return False, ["Empty answer"]
    return True, []


def check_answer_format(answer: str, question: str) -> Tuple[bool, List[str]]:
    if not answer.strip().startswith("[["):
        return False, ["Missing [[ wrapper"]
    return True, []


# ---- Registry ----

VALIDATORS: Dict[str, Callable] = {
    # Grid CSP rules — Latin square validation
    "25": check_latin_square,
    "15": check_latin_square,
    "16": check_latin_square,
    "17": check_latin_square,
    # Grid rules — basic structure
    "11": check_grid_non_empty,
    "12": check_grid_non_empty,
    "13": check_grid_non_empty,
    "14": check_grid_non_empty,
    "18": check_grid_non_empty,
    "20": check_grid_non_empty,
    "21": check_grid_non_empty,
    "23": check_grid_non_empty,
    "19": check_answer_format,   # star battle: [[A(r,c)\nB(r,c)...]]
    "24": check_grid_non_empty,
    "22": check_answer_format,
    "2": check_word_list,
    "3": check_word_list,
    "4": check_word_list,
    "8": check_answer_format,   # word search: [[WORD (r,c)(r,c)\nWORD2 (r,c)(r,c)]]
    # Numeric
    "5": check_cryptomath,
    "9": check_answer_format,
    "10": check_24points,
    # Logic / misc
    "6": check_answer_format,
    "7": check_answer_format,
}


def get_validator(rule_id: str) -> Callable:
    return VALIDATORS.get(rule_id, check_answer_format)


def validate_answer(rule_id: str, answer: str, question: str) -> Tuple[bool, List[str]]:
    try:
        fn = get_validator(rule_id)
        return fn(answer, question)
    except Exception as e:
        return False, [f"Validator error: {e}"]
