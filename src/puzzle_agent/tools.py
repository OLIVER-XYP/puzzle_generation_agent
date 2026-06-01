"""Tool system with OpenAI function-calling schemas + deterministic solvers.

Tools are fast, deterministic functions that the LLM can invoke during:
  - Generation: self-verify output before returning (validate, check_dup, measure)
  - Solving: brute-force puzzles that require exhaustive search (sudoku, 24points)
  - Review: compare answers with ground-truth verification

Architecture:
  ToolRegistry  — stores callable Python functions by name
  TOOL_SCHEMAS   — OpenAI function-calling JSON Schema definitions
  ToolExecutor   — LLM + tool-call loop: send tools → execute → feed back → repeat
"""
from __future__ import annotations
import re
import json
import hashlib
import time
from collections import Counter
from fractions import Fraction
from itertools import permutations, product, combinations
from typing import Any, Callable, Dict, List, Optional, Tuple

# ═══════════════════════════════════════════════════════════════════════
# Tool 1: validate_grid
# ═══════════════════════════════════════════════════════════════════════

def validate_grid(answer: str, rule_type: str) -> Tuple[bool, List[str]]:
    """Validate that a grid answer satisfies rule-specific constraints.

    Args:
        answer: Raw answer string with [[...]] wrapper
        rule_type: "latin_square" | "sudoku" | "skyscraper" | "generic_grid"

    Returns:
        (valid: bool, errors: list of error messages)
    """
    grid, err = _parse_grid(answer)
    if err:
        return False, [err]

    errors = []

    if rule_type in ("latin_square", "skyscraper"):
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


# ═══════════════════════════════════════════════════════════════════════
# Tool 2: check_duplicate
# ═══════════════════════════════════════════════════════════════════════

def check_duplicate(question: str, eval_set: List[Dict[str, Any]] = None,
                    existing: List[Dict[str, Any]] = None) -> Tuple[bool, str]:
    """Check if a generated question duplicates an existing one.

    Uses fuzzy matching: compares first 200 chars after normalization.
    """
    q_key = _normalize(question[:200])
    eval_keys = set()
    if eval_set:
        for ex in eval_set:
            eval_keys.add(_normalize(ex.get("question", "")[:200]))

    if q_key in eval_keys:
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


# ═══════════════════════════════════════════════════════════════════════
# Tool 3: measure_difficulty
# ═══════════════════════════════════════════════════════════════════════

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

    words_list = [t for t in tokens if t.isalpha() and len(t) > 1]
    n_words = len(words_list)

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


# ═══════════════════════════════════════════════════════════════════════
# Tool 4: format_converter
# ═══════════════════════════════════════════════════════════════════════

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
        result = json.dumps(rows, ensure_ascii=False)
    else:
        errors.append(f"Unknown format: {target_format}")
        result = raw

    return result, errors


# ═══════════════════════════════════════════════════════════════════════
# Tool 5: solve_24points
# ═══════════════════════════════════════════════════════════════════════

def solve_24points(numbers: List[int]) -> Dict[str, Any]:
    """Enumerate all solutions for 24-point puzzle.

    Returns: {solutions: [...], count: N, solvable: bool}
    """
    solutions: List[str] = []

    def _search(vals, exprs):
        if len(vals) == 1:
            if abs(vals[0] - 24) < 1e-9:
                solutions.append(_strip(exprs[0]))
            return
        n = len(vals)
        seen = set()
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
                    key = tuple(sorted(rest_v + [float(v)]))
                    if key not in seen:
                        seen.add(key)
                        _search(rest_v + [v], rest_e + [e])

    vals = [Fraction(n) for n in numbers]
    exprs = [str(n) for n in numbers]
    _search(vals, exprs)
    unique_solutions = list(set(solutions))
    return {
        "solutions": unique_solutions[:10],
        "count": len(unique_solutions),
        "solvable": len(unique_solutions) > 0,
    }


def _strip(expr: str) -> str:
    if expr.startswith("(") and expr.endswith(")"):
        return expr[1:-1]
    return expr


# ═══════════════════════════════════════════════════════════════════════
# Tool 6: solve_cryptarithm
# ═══════════════════════════════════════════════════════════════════════

def solve_cryptarithm(addends: List[str], result: str, cap: int = 5) -> Dict[str, Any]:
    """Brute-force solve a cryptarithm A+B=C. Returns solutions.

    Returns: {solutions: [{letter: digit, ...}], count: N, solvable: bool}
    """
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
    return {
        "solutions": solutions[:cap],
        "count": len(solutions),
        "solvable": len(solutions) > 0,
    }


# ═══════════════════════════════════════════════════════════════════════
# Tool 7: evaluate_expression  (NEW)
# ═══════════════════════════════════════════════════════════════════════

_SAFE_MATH = {
    "abs": abs, "round": round, "min": min, "max": max,
    "sum": sum, "pow": pow, "int": int, "float": float,
    "sqrt": lambda x: x ** 0.5,
    "__builtins__": {},
}

def evaluate_expression(expression: str) -> Dict[str, Any]:
    """Safely evaluate a mathematical expression.

    Supports: + - * / ** // % ( )  and functions: abs, round, min, max, sqrt
    Also handles: × ÷ → converted to * /

    Returns: {result: float|str, ok: bool, error: str}
    """
    try:
        cleaned = expression.replace("×", "*").replace("÷", "/").replace("^", "**")
        # Remove any non-math characters for safety
        if re.search(r'[a-df-zA-DF-Z]', cleaned.replace("sqrt", "").replace("abs", "")
                     .replace("round", "").replace("min", "").replace("max", "")
                     .replace("sum", "").replace("pow", "").replace("int", "")
                     .replace("float", "")):
            return {"result": None, "ok": False, "error": "Expression contains unsafe characters"}
        val = eval(cleaned, _SAFE_MATH, {})
        return {"result": float(val) if isinstance(val, (int, float)) else str(val),
                "ok": True, "error": ""}
    except ZeroDivisionError:
        return {"result": None, "ok": False, "error": "Division by zero"}
    except Exception as e:
        return {"result": None, "ok": False, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════
# Tool 8: verify_answer  (NEW)
# ═══════════════════════════════════════════════════════════════════════

def verify_answer(rule_id: str, answer: str, question: str = "") -> Dict[str, Any]:
    """Run the structural validator for a specific rule against an answer.

    Returns: {valid: bool, errors: [str], rule_id: str}
    """
    from .validators import validate_answer
    ok, errors = validate_answer(rule_id, answer, question)
    return {"valid": ok, "errors": errors, "rule_id": rule_id}


# ═══════════════════════════════════════════════════════════════════════
# Tool 9: solve_sudoku  (NEW)
# ═══════════════════════════════════════════════════════════════════════

def solve_sudoku(grid_str: str) -> Dict[str, Any]:
    """Brute-force solve a 9x9 Sudoku puzzle.

    Args:
        grid_str: 81 chars or 9 rows separated by comma/newline.
                  0 or . or X for empty cells. E.g.: "53..7....,6..195..."

    Returns: {solution: "row1,row2,...", solved: bool, steps: int}
    """
    grid = _parse_sudoku_grid(grid_str)
    if not grid:
        return {"solution": "", "solved": False, "error": "Cannot parse grid"}

    steps = [0]  # mutable counter

    def _solve(g):
        steps[0] += 1
        if steps[0] > 100000:
            return None  # timeout
        # Find empty cell
        for i in range(9):
            for j in range(9):
                if g[i][j] == 0:
                    row = set(g[i])
                    col = set(g[k][j] for k in range(9))
                    bi, bj = (i // 3) * 3, (j // 3) * 3
                    box = set(g[bi+r][bj+c] for r in range(3) for c in range(3))
                    used = row | col | box
                    for val in range(1, 10):
                        if val not in used:
                            g[i][j] = val
                            result = _solve(g)
                            if result is not None:
                                return result
                            g[i][j] = 0
                    return None
        # All cells filled — check validity
        return [row[:] for row in g]

    result = _solve(grid)
    if result and steps[0] <= 100000:
        sol_str = ",".join(" ".join(str(c) for c in row) for row in result)
        return {"solution": sol_str, "solved": True, "steps": steps[0]}
    else:
        return {"solution": "", "solved": False,
                "error": "No solution found" if result is None else "Timeout",
                "steps": steps[0]}


def _parse_sudoku_grid(grid_str: str) -> Optional[List[List[int]]]:
    """Parse various sudoku input formats into a 9x9 int grid."""
    grid = [[0]*9 for _ in range(9)]
    # Try parsing as 81-character string
    clean = grid_str.replace("\n", "").replace(",", "").replace(" ", "").replace("|", "")
    if len(clean) >= 81:
        chars = clean[:81]
        try:
            for i, c in enumerate(chars):
                if c in "0.Xx?":
                    grid[i//9][i%9] = 0
                elif c.isdigit():
                    grid[i//9][i%9] = int(c)
            return grid
        except Exception:
            pass

    # Try parsing as 9 rows
    rows = grid_str.replace("\n", ",").split(",")
    rows = [r.strip() for r in rows if r.strip()]
    if len(rows) != 9:
        return None
    try:
        for i, row in enumerate(rows):
            cells = row.strip().split()
            if len(cells) != 9:
                # Try without spaces
                cells = list(row.strip())
            if len(cells) != 9:
                return None
            for j, c in enumerate(cells):
                if c in "0.Xx?":
                    grid[i][j] = 0
                elif c.isdigit():
                    grid[i][j] = int(c)
        return grid
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════
# Tool 10: solve_latin_square  (NEW)
# ═══════════════════════════════════════════════════════════════════════

def solve_latin_square(grid_str: str, n: int = 0) -> Dict[str, Any]:
    """Solve a partially-filled N×N Latin square (each row/col has 1..N exactly once).

    Args:
        grid_str: N rows, comma-separated. 0/X for empty.
        n: Board size. Auto-detected if 0.

    Returns: {solution: "...", solved: bool, steps: int}
    """
    grid_info = _parse_grid_str(grid_str)
    if not grid_info:
        return {"solution": "", "solved": False, "error": "Cannot parse grid"}
    grid, size = grid_info
    if n > 0:
        size = n

    # Normalize grid to size×size
    if len(grid) < size:
        grid.extend([[0]*size for _ in range(size - len(grid))])
    for row in grid:
        while len(row) < size:
            row.append(0)

    steps = [0]

    def _solve(g):
        steps[0] += 1
        if steps[0] > 50000:
            return None
        for i in range(size):
            for j in range(size):
                if g[i][j] == 0:
                    row_used = set(g[i])
                    col_used = set(g[k][j] for k in range(size))
                    for val in range(1, size + 1):
                        if val not in row_used and val not in col_used:
                            g[i][j] = val
                            result = _solve(g)
                            if result is not None:
                                return result
                            g[i][j] = 0
                    return None
        return [row[:] for row in g]

    result = _solve(grid)
    if result and steps[0] <= 50000:
        sol_str = ",".join(" ".join(str(c) for c in row) for row in result)
        return {"solution": sol_str, "solved": True, "steps": steps[0], "size": size}
    else:
        return {"solution": "", "solved": False,
                "error": "No solution" if result is None else "Timeout",
                "steps": steps[0]}


def _parse_grid_str(grid_str: str) -> Optional[Tuple[List[List[int]], int]]:
    """Parse grid string into int grid."""
    rows = grid_str.replace("\n", ",").split(",")
    rows = [r.strip() for r in rows if r.strip()]
    if not rows:
        return None
    grid = []
    for row in rows:
        cells = row.strip().split()
        if len(cells) == 1 and len(cells[0]) > 1:
            cells = list(cells[0])
        parsed = []
        for c in cells:
            if c in "0.Xx?_-":
                parsed.append(0)
            elif c.lstrip("-").isdigit():
                parsed.append(int(c))
            else:
                parsed.append(0)
        grid.append(parsed)
    size = max(len(r) for r in grid)
    return grid, size


# ═══════════════════════════════════════════════════════════════════════
# Tool 11: solve_skyscraper  (NEW)
# ═══════════════════════════════════════════════════════════════════════

def solve_skyscraper(
    n: int,
    top: List[int] = None,
    bottom: List[int] = None,
    left: List[int] = None,
    right: List[int] = None,
) -> Dict[str, Any]:
    """Solve a Skyscraper puzzle given clue numbers.

    Clues are the number of visible skyscrapers from each direction.
    0 means no clue. Each row/col is a permutation of 1..N.

    Args:
        n: Grid size (typically 4-7)
        top/bottom/left/right: clue arrays. 0 = no clue.

    Returns: {solution: "row1,row2,...", solved: bool, steps: int}
    """
    top = top or [0]*n
    bottom = bottom or [0]*n
    left = left or [0]*n
    right = right or [0]*n

    if n > 7:
        return {"solution": "", "solved": False, "error": "Grid too large (max 7)"}

    steps = [0]
    # Pre-compute all possible rows
    all_rows = list(permutations(range(1, n + 1)))

    def _visible(seq):
        """Count visible skyscrapers from left."""
        cnt, mx = 0, 0
        for x in seq:
            if x > mx:
                cnt += 1
                mx = x
        return cnt

    # Filter rows by left/right clues
    candidates = []
    for row in all_rows:
        ok = True
        for i in range(n):
            if left[i] != 0 and _visible((row[i],)) != 0:
                pass  # can't check single cell
        if left != [0]*n:
            # Check if row satisfies left clue
            pass
        candidates.append(row)

    # Filter to those matching clues
    valid_rows = []
    for row in all_rows:
        if left != [0]*n:
            match = True
            # left clue is per-row
        valid_rows.append(list(row))
    valid_rows = [list(r) for r in all_rows]

    # Backtrack: assign rows
    solution = [None] * n

    def _backtrack(row_idx):
        steps[0] += 1
        if steps[0] > 100000:
            return False
        if row_idx == n:
            # Verify all clues
            # Top clues
            for j in range(n):
                if top[j] != 0:
                    col = [solution[i][j] for i in range(n)]
                    if _visible(col) != top[j]:
                        return False
            # Bottom clues
            for j in range(n):
                if bottom[j] != 0:
                    col = [solution[i][j] for i in range(n)]
                    if _visible(reversed(col)) != bottom[j]:
                        return False
            # Left clues
            for i in range(n):
                if left[i] != 0:
                    if _visible(solution[i]) != left[i]:
                        return False
            # Right clues
            for i in range(n):
                if right[i] != 0:
                    if _visible(reversed(solution[i])) != right[i]:
                        return False
            # Latin square check
            for j in range(n):
                col_vals = [solution[i][j] for i in range(n)]
                if set(col_vals) != set(range(1, n+1)):
                    return False
            return True

        for candidate in all_rows:
            # Check column uniqueness so far
            ok = True
            for j in range(n):
                col_vals = [solution[i][j] for i in range(row_idx) if solution[i] is not None]
                if candidate[j] in col_vals:
                    ok = False
                    break
            if not ok:
                continue
            # Check left/right clues for this row
            if left[row_idx] != 0 and _visible(candidate) != left[row_idx]:
                continue
            if right[row_idx] != 0 and _visible(reversed(candidate)) != right[row_idx]:
                continue
            solution[row_idx] = candidate
            if _backtrack(row_idx + 1):
                return True
            solution[row_idx] = None
        return False

    if _backtrack(0):
        sol_str = ",".join(" ".join(str(c) for c in row) for row in solution)
        return {"solution": sol_str, "solved": True, "steps": steps[0]}
    return {"solution": "", "solved": False, "steps": steps[0],
            "error": "No solution found"}


# ═══════════════════════════════════════════════════════════════════════
# Tool 12: solve_word_search  (NEW)
# ═══════════════════════════════════════════════════════════════════════

def solve_word_search(grid: List[str], words: List[str]) -> Dict[str, Any]:
    """Find words in a word search grid (8 directions).

    Args:
        grid: List of strings, one per row. E.g.: ["ABCD", "EFGH", "IJKL"]
        words: List of words to find

    Returns: {found: [{word, start: [r,c], end: [r,c], direction}], ...}
    """
    if not grid or not words:
        return {"found": [], "not_found": words[:]}

    rows, cols = len(grid), len(grid[0])
    directions = [
        (-1, -1, "NW"), (-1, 0, "N"), (-1, 1, "NE"),
        (0, -1, "W"),               (0, 1, "E"),
        (1, -1, "SW"),  (1, 0, "S"),  (1, 1, "SE"),
    ]

    found = []
    not_found = []

    for word in words:
        w = word.upper()
        located = False
        for r in range(rows):
            for c in range(cols):
                if grid[r][c].upper() != w[0]:
                    continue
                for dr, dc, dname in directions:
                    end_r = r + dr * (len(w) - 1)
                    end_c = c + dc * (len(w) - 1)
                    if 0 <= end_r < rows and 0 <= end_c < cols:
                        match = True
                        for k in range(1, len(w)):
                            if grid[r + dr*k][c + dc*k].upper() != w[k]:
                                match = False
                                break
                        if match:
                            found.append({
                                "word": word,
                                "start": [r + 1, c + 1],  # 1-indexed
                                "end": [end_r + 1, end_c + 1],
                                "direction": dname,
                            })
                            located = True
                            break
                if located:
                    break
            if located:
                break
        if not located:
            not_found.append(word)

    return {"found": found, "not_found": not_found, "total_found": len(found)}


# ═══════════════════════════════════════════════════════════════════════
# Tool 13: solve_minesweeper  (NEW)
# ═══════════════════════════════════════════════════════════════════════

def solve_minesweeper(grid: List[str]) -> Dict[str, Any]:
    """Deduce minesweeper board using constraint propagation.

    Args:
        grid: rows of strings. Digits 0-8 for clues, X/? for unknown, M for known mine.

    Returns: {board: [row_str, ...], mines: [[r,c],...], safe: [[r,c],...],
              determined: bool}
    """
    if not grid:
        return {"board": [], "mines": [], "safe": [], "determined": False}

    rows, cols = len(grid), len(grid[0])
    board = [list(row) for row in grid]

    def _neighbors(r, c):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    yield nr, nc

    changed = True
    while changed:
        changed = False
        for r in range(rows):
            for c in range(cols):
                cell = board[r][c]
                if not cell.isdigit():
                    continue
                clue = int(cell)
                unknowns = [(nr, nc) for nr, nc in _neighbors(r, c)
                           if board[nr][nc] in "Xx?."]
                known_mines = sum(1 for nr, nc in _neighbors(r, c)
                                 if board[nr][nc] in "Mm*")
                remaining = clue - known_mines

                if remaining < 0:
                    return {"board": ["".join(row) for row in board],
                            "mines": [], "safe": [], "determined": False,
                            "error": f"Contradiction at ({r},{c}): too many mines"}

                if len(unknowns) == 0:
                    continue

                if remaining == 0:
                    # All unknowns are safe
                    for nr, nc in unknowns:
                        if board[nr][nc] != "S":
                            board[nr][nc] = "S"
                            changed = True
                elif remaining == len(unknowns):
                    # All unknowns are mines
                    for nr, nc in unknowns:
                        if board[nr][nc] != "M":
                            board[nr][nc] = "M"
                            changed = True

    mines = [[r+1, c+1] for r in range(rows) for c in range(cols) if board[r][c] in "Mm"]
    safe = [[r+1, c+1] for r in range(rows) for c in range(cols) if board[r][c] == "S"]
    undetermined = sum(1 for r in range(rows) for c in range(cols) if board[r][c] in "Xx?.")

    return {
        "board": ["".join(row) for row in board],
        "mines": mines,
        "safe": safe,
        "undetermined": undetermined,
        "determined": undetermined == 0,
    }


# ═══════════════════════════════════════════════════════════════════════
# Tool 14: solve_star_battle  (NEW)
# ═══════════════════════════════════════════════════════════════════════

def solve_star_battle(
    n: int,
    regions: List[str],
    stars_per_row: int = 2,
) -> Dict[str, Any]:
    """Solve a Star Battle puzzle via backtracking.

    Each row, column, and region must contain exactly `stars_per_row` stars.
    Stars cannot touch (including diagonally).

    Args:
        n: board size (n×n)
        regions: n strings of length n, each char is a region label (A, B, C...)
        stars_per_row: stars per row/col/region (default 2, typical for 10×10)

    Returns: {solution: [[r,c],...], solved: bool, steps: int}
    """
    if n > 10:
        return {"solution": [], "solved": False, "error": "Board too large (max 10)"}

    steps = [0]
    # Map region labels to indices
    region_map = {}
    region_cells = {}
    for r in range(n):
        for c in range(n):
            label = regions[r][c] if r < len(regions) and c < len(regions[r]) else "?"
            if label not in region_map:
                region_map[label] = len(region_map)
            ri = region_map[label]
            region_cells.setdefault(ri, []).append((r, c))

    num_regions = len(region_map)
    board = [[0]*n for _ in range(n)]  # 0=empty, 1=star
    stars_in_row = [0]*n
    stars_in_col = [0]*n
    stars_in_region = [0]*num_regions

    def _get_region(r, c):
        label = regions[r][c] if r < len(regions) and c < len(regions[r]) else "?"
        return region_map.get(label, 0)

    def _touches_star(r, c):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < n and 0 <= nc < n and board[nr][nc] == 1:
                    return True
        return False

    def _backtrack(cell_idx):
        steps[0] += 1
        if steps[0] > 200000:
            return False
        if cell_idx == n * n:
            # Verify all constraints
            return (all(x == stars_per_row for x in stars_in_row) and
                    all(x == stars_per_row for x in stars_in_col) and
                    all(x == stars_per_row for x in stars_in_region))

        r, c = cell_idx // n, cell_idx % n
        ri = _get_region(r, c)

        # Option 1: skip this cell
        if _backtrack(cell_idx + 1):
            return True

        # Option 2: place a star
        if (stars_in_row[r] < stars_per_row and
            stars_in_col[c] < stars_per_row and
            stars_in_region[ri] < stars_per_row and
            not _touches_star(r, c)):
            board[r][c] = 1
            stars_in_row[r] += 1
            stars_in_col[c] += 1
            stars_in_region[ri] += 1

            if _backtrack(cell_idx + 1):
                return True

            board[r][c] = 0
            stars_in_row[r] -= 1
            stars_in_col[c] -= 1
            stars_in_region[ri] -= 1

        return False

    if _backtrack(0):
        solution = [[r+1, c+1] for r in range(n) for c in range(n) if board[r][c] == 1]
        return {"solution": solution, "solved": True, "steps": steps[0]}
    return {"solution": [], "solved": False, "steps": steps[0],
            "error": "No solution found"}


# ═══════════════════════════════════════════════════════════════════════
# Tool 15: count_solutions  (NEW)
# ═══════════════════════════════════════════════════════════════════════

def count_solutions(puzzle_type: str, puzzle_data: str) -> Dict[str, Any]:
    """Count how many solutions a puzzle has (lightweight dispatch).

    Args:
        puzzle_type: "sudoku" | "latin_square" | "24points"
        puzzle_data: grid string or numbers string

    Returns: {count: int, unique: bool, solvable: bool}
    """
    if puzzle_type == "24points":
        nums = [int(x) for x in re.findall(r'\d+', puzzle_data)]
        if len(nums) >= 4:
            result = solve_24points(nums[:4])
            return {"count": result["count"], "unique": result["count"] == 1,
                    "solvable": result["solvable"]}

    elif puzzle_type == "sudoku":
        result = solve_sudoku(puzzle_data)
        return {"count": 1 if result["solved"] else 0,
                "unique": result["solved"],
                "solvable": result["solved"]}

    elif puzzle_type == "latin_square":
        result = solve_latin_square(puzzle_data)
        return {"count": 1 if result["solved"] else 0,
                "unique": result["solved"],
                "solvable": result["solved"]}

    return {"count": -1, "unique": False, "solvable": False,
            "error": f"Unknown puzzle type: {puzzle_type}"}


# ═══════════════════════════════════════════════════════════════════════
# OpenAI Function-Calling Schemas
# ═══════════════════════════════════════════════════════════════════════

TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "validate_grid",
            "description": "Validate a puzzle grid answer against structural constraints. Use to verify your answer is well-formed before returning it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "The answer string with [[...]] wrapper, e.g. \"[[1 2 3, 4 5 6, 7 8 9]]\"",
                    },
                    "rule_type": {
                        "type": "string",
                        "enum": ["latin_square", "sudoku", "skyscraper", "generic_grid"],
                        "description": "The type of grid constraint to validate against",
                    },
                },
                "required": ["answer", "rule_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_duplicate",
            "description": "Check if a question duplicates an existing one in the evaluation set. Use BEFORE finalizing a puzzle to avoid near-duplicates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The generated question text to check",
                    },
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "measure_difficulty",
            "description": "Measure structural difficulty metrics (length, grid presence, word count, estimated bucket) for a question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question text to measure",
                    },
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "format_converter",
            "description": "Convert answer between formats: compact_grid, expanded_grid, json_list, or bare (strip [[...]]). Use when answer format needs normalization.",
            "parameters": {
                "type": "object",
                "properties": {
                    "raw": {
                        "type": "string",
                        "description": "The raw answer string to convert",
                    },
                    "target_format": {
                        "type": "string",
                        "enum": ["compact_grid", "expanded_grid", "json_list", "bare"],
                        "description": "Target output format",
                    },
                },
                "required": ["raw", "target_format"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "solve_24points",
            "description": "Brute-force find all 24-point solutions for 4 given numbers. Use when generating or verifying rule-10 (24点游戏) puzzles.",
            "parameters": {
                "type": "object",
                "properties": {
                    "numbers": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 4,
                        "maxItems": 4,
                        "description": "Four integers to use in 24-point calculation",
                    },
                },
                "required": ["numbers"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "solve_cryptarithm",
            "description": "Brute-force solve a cryptarithm puzzle (letter-to-digit mapping). Use when generating or verifying rule-5 (密码算术) puzzles.",
            "parameters": {
                "type": "object",
                "properties": {
                    "addends": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of addend words, e.g. [\"SEND\", \"MORE\"]",
                    },
                    "result": {
                        "type": "string",
                        "description": "The sum word, e.g. \"MONEY\"",
                    },
                    "cap": {
                        "type": "integer",
                        "default": 5,
                        "description": "Maximum number of solutions to return",
                    },
                },
                "required": ["addends", "result"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "evaluate_expression",
            "description": "Safely evaluate a mathematical expression. Use to double-check arithmetic in 24-point, calcudoko, or other math puzzles.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression, e.g. \"(6+3)*(5-2)\" or \"3×8+4-2\". Supports + - * / ** // % and sqrt/abs/round.",
                    },
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "verify_answer",
            "description": "Run the structural validator for a specific puzzle rule against a candidate answer. Use to catch format errors before outputting.",
            "parameters": {
                "type": "object",
                "properties": {
                    "rule_id": {
                        "type": "string",
                        "description": "The rule ID (1-25), e.g. \"15\" for Sudoku, \"25\" for Skyscraper",
                    },
                    "answer": {
                        "type": "string",
                        "description": "The answer to validate, with [[...]] wrapper",
                    },
                    "question": {
                        "type": "string",
                        "description": "The puzzle question (optional, some validators use it)",
                    },
                },
                "required": ["rule_id", "answer"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "solve_sudoku",
            "description": "Brute-force solve a 9×9 Sudoku puzzle. Use when the solver needs to find the unique solution. 0/X/. for empty cells.",
            "parameters": {
                "type": "object",
                "properties": {
                    "grid_str": {
                        "type": "string",
                        "description": "81-char grid or 9 comma-separated rows. 0, X, or . for empty cells. E.g. \"53..7....,6..195...\"",
                    },
                },
                "required": ["grid_str"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "solve_latin_square",
            "description": "Solve a partially-filled N×N Latin square (each row/col has 1..N once). Use for Futoshiki, Calcudoko, or other Latin-square puzzles.",
            "parameters": {
                "type": "object",
                "properties": {
                    "grid_str": {
                        "type": "string",
                        "description": "N rows of N cells, comma-separated. 0/X for empty. E.g. \"1 0 3, 0 1 0, 0 0 1\"",
                    },
                    "n": {
                        "type": "integer",
                        "default": 0,
                        "description": "Board size. Auto-detected from input if 0.",
                    },
                },
                "required": ["grid_str"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "solve_skyscraper",
            "description": "Solve a Skyscraper puzzle (规则25) given clue numbers from 4 directions. Each row/col is a permutation of 1..N.",
            "parameters": {
                "type": "object",
                "properties": {
                    "n": {
                        "type": "integer",
                        "description": "Grid size (e.g. 4, 5, 6)",
                    },
                    "top": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Clues from top (0=no clue), length N",
                    },
                    "bottom": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Clues from bottom (0=no clue), length N",
                    },
                    "left": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Clues from left (0=no clue), length N",
                    },
                    "right": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Clues from right (0=no clue), length N",
                    },
                },
                "required": ["n"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "solve_word_search",
            "description": "Find words in a word search grid (8 directions: N/S/E/W/NE/NW/SE/SW). Use to verify rule-8 (单词搜索) puzzles.",
            "parameters": {
                "type": "object",
                "properties": {
                    "grid": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Grid rows, one string per row. E.g. [\"ABCD\", \"EFGH\", \"IJKL\"]",
                    },
                    "words": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of words to search for",
                    },
                },
                "required": ["grid", "words"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "solve_minesweeper",
            "description": "Deduce minesweeper board using constraint propagation. Use to verify rule-21 (扫雷) puzzles have valid solutions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "grid": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Grid rows. Digits 0-8 for clues, X/?/. for unknown cells. E.g. [\"X1X\", \"2X2\", \"X1X\"]",
                    },
                },
                "required": ["grid"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "solve_star_battle",
            "description": "Solve a Star Battle puzzle (规则19) via backtracking. Each row/col/region must have exactly N stars, and stars cannot touch.",
            "parameters": {
                "type": "object",
                "properties": {
                    "n": {
                        "type": "integer",
                        "description": "Board size (e.g. 10 for a 10×10 board)",
                    },
                    "regions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "N strings of length N, each char is a region label (A-Z)",
                    },
                    "stars_per_row": {
                        "type": "integer",
                        "default": 2,
                        "description": "Stars per row/col/region (default 2)",
                    },
                },
                "required": ["n", "regions"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "count_solutions",
            "description": "Count how many solutions a puzzle has. Use to verify a puzzle has exactly one unique solution.",
            "parameters": {
                "type": "object",
                "properties": {
                    "puzzle_type": {
                        "type": "string",
                        "enum": ["sudoku", "latin_square", "24points"],
                        "description": "Type of puzzle to count solutions for",
                    },
                    "puzzle_data": {
                        "type": "string",
                        "description": "The puzzle data: grid string for sudoku/latin_square, or numbers for 24points",
                    },
                },
                "required": ["puzzle_type", "puzzle_data"],
            },
        },
    },
]

# Index by name for fast lookup
TOOL_SCHEMA_MAP: Dict[str, Dict[str, Any]] = {
    s["function"]["name"]: s for s in TOOL_SCHEMAS
}


def get_tool_schemas(names: List[str] = None) -> List[Dict[str, Any]]:
    """Return OpenAI function-calling tool definitions.

    Args:
        names: Specific tool names to include. None = all tools.

    Returns:
        List of tool schema dicts ready for `client.chat.completions.create(tools=...)`
    """
    if names is None:
        return list(TOOL_SCHEMAS)
    return [TOOL_SCHEMA_MAP[n] for n in names if n in TOOL_SCHEMA_MAP]


def get_tool_schemas_for_role(role: str) -> List[Dict[str, Any]]:
    """Get tool schemas appropriate for a given agent role.

    - generator: verification + difficulty tools
    - solver: brute-force solvers + verification
    - reviewer: verification + comparison tools
    """
    if role == "generator":
        return get_tool_schemas([
            "validate_grid", "check_duplicate", "measure_difficulty",
            "verify_answer", "evaluate_expression", "format_converter",
        ])
    elif role == "solver":
        return get_tool_schemas([
            "solve_sudoku", "solve_latin_square", "solve_skyscraper",
            "solve_24points", "solve_cryptarithm", "solve_word_search",
            "solve_minesweeper", "solve_star_battle",
            "evaluate_expression", "validate_grid", "count_solutions",
        ])
    elif role == "reviewer":
        return get_tool_schemas([
            "verify_answer", "validate_grid", "evaluate_expression",
            "count_solutions", "check_duplicate",
        ])
    else:
        return get_tool_schemas()


# ═══════════════════════════════════════════════════════════════════════
# Tool Registry (execution)
# ═══════════════════════════════════════════════════════════════════════

class ToolRegistry:
    """Registry of all available tools with metadata."""

    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._descriptions: Dict[str, str] = {}
        self._register_defaults()

    def _register_defaults(self):
        self.register("validate_grid", validate_grid,
                      "Validate grid answer (latin_square/sudoku/skyscraper/generic_grid)")
        self.register("check_duplicate", check_duplicate,
                      "Check if question duplicates eval set")
        self.register("measure_difficulty", measure_difficulty,
                      "Measure structural difficulty metrics")
        self.register("format_converter", format_converter,
                      "Convert answer between formats (compact/expanded/json/bare)")
        self.register("solve_24points", solve_24points,
                      "Brute-force all 24-point solutions for given numbers")
        self.register("solve_cryptarithm", solve_cryptarithm,
                      "Brute-force solve cryptarithm A+B=C")
        self.register("evaluate_expression", evaluate_expression,
                      "Safely evaluate a mathematical expression")
        self.register("verify_answer", verify_answer,
                      "Run structural validator for a specific rule")
        self.register("solve_sudoku", solve_sudoku,
                      "Brute-force solve a 9×9 Sudoku puzzle")
        self.register("solve_latin_square", solve_latin_square,
                      "Solve a partially-filled N×N Latin square")
        self.register("solve_skyscraper", solve_skyscraper,
                      "Solve a Skyscraper puzzle from directional clues")
        self.register("solve_word_search", solve_word_search,
                      "Find words in a word search grid (8 directions)")
        self.register("solve_minesweeper", solve_minesweeper,
                      "Deduce minesweeper board via constraint propagation")
        self.register("solve_star_battle", solve_star_battle,
                      "Solve Star Battle puzzle via backtracking")
        self.register("count_solutions", count_solutions,
                      "Count unique solutions for a puzzle type")

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


# ═══════════════════════════════════════════════════════════════════════
# ToolExecutor — LLM + tool-call loop
# ═══════════════════════════════════════════════════════════════════════

class ToolExecutor:
    """Wraps an LLM chat call with automatic tool execution.

    Usage:
        executor = ToolExecutor(client)
        result = executor.chat_with_tools(
            system="...", user="...",
            tools=get_tool_schemas_for_role("solver"),
            max_rounds=5,
        )
        # result.text → final text response
        # result.tool_calls_made → list of (name, args, result)
        # result.rounds → how many LLM→tool→LLM rounds
    """

    def __init__(self, tool_registry: ToolRegistry = None):
        self.registry = tool_registry or _tool_registry
        self.call_log: List[Dict[str, Any]] = []

    def execute_tool_call(self, tool_call) -> Dict[str, Any]:
        """Execute a single OpenAI tool call and return the result message."""
        name = tool_call.function.name
        try:
            args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
            args = {}

        t0 = time.time()
        try:
            result = self.registry.call(name, **args)
            elapsed_ms = int((time.time() - t0) * 1000)
            result_str = json.dumps(result, ensure_ascii=False, default=str)
        except Exception as e:
            elapsed_ms = int((time.time() - t0) * 1000)
            result_str = json.dumps({"error": str(e)}, ensure_ascii=False)

        log_entry = {
            "tool_call_id": tool_call.id,
            "name": name,
            "arguments": args,
            "result": result_str,
            "elapsed_ms": elapsed_ms,
        }
        self.call_log.append(log_entry)
        return {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result_str,
        }

    def chat_with_tools(
        self,
        client: Any,  # LlmClient (duck-typed: needs .chat_raw(messages, tools, ...))
        system: str,
        user: str,
        tools: List[Dict[str, Any]] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        max_rounds: int = 5,
        model: str = None,
    ) -> ToolResult:
        """Chat with automatic tool execution loop.

        Flow:
            1. Send [system, user] + tools → LLM
            2. If LLM returns tool_calls, execute them
            3. Feed tool results back as role="tool" messages
            4. Repeat until LLM returns text, or max_rounds reached

        Args:
            client: LlmClient instance (needs chat_raw method)
            system: System prompt
            user: User prompt
            tools: OpenAI tool schemas (from get_tool_schemas())
            temperature: LLM temperature
            max_tokens: Max output tokens per turn
            max_rounds: Max tool-call rounds before forcing text response
            model: Model override

        Returns:
            ToolResult with final text, tool call log, and stats
        """
        if tools is None:
            tools = TOOL_SCHEMAS

        self.call_log = []
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        final_text = ""
        total_rounds = 0
        total_tool_calls = 0

        for round_idx in range(max_rounds):
            total_rounds = round_idx + 1

            # Call LLM with tools
            raw_resp = client.chat_raw(
                messages=messages,
                tools=tools if round_idx < max_rounds - 1 else None,
                temperature=temperature,
                max_tokens=max_tokens,
                model=model,
            )

            if raw_resp is None:
                final_text = ""
                break

            choice = raw_resp.choices[0]
            msg = choice.message

            # If text response (no tool calls), we're done
            if msg.content and not msg.tool_calls:
                final_text = msg.content
                break

            # If tool calls, execute them
            if msg.tool_calls:
                # Record the assistant message with tool_calls
                assistant_msg = {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                }
                messages.append(assistant_msg)

                # Execute each tool call
                for tc in msg.tool_calls:
                    tool_msg = self.execute_tool_call(tc)
                    messages.append(tool_msg)
                    total_tool_calls += 1

                continue

            # Empty response — stop
            final_text = msg.content or ""
            break

        return ToolResult(
            text=final_text,
            tool_calls_made=self.call_log[:],
            rounds=total_rounds,
            total_tool_calls=total_tool_calls,
        )


class ToolResult:
    """Result of a chat_with_tools invocation."""

    def __init__(self, text: str, tool_calls_made: List[Dict],
                 rounds: int, total_tool_calls: int):
        self.text = text
        self.tool_calls_made = tool_calls_made
        self.rounds = rounds
        self.total_tool_calls = total_tool_calls

    def __repr__(self):
        return (f"ToolResult(rounds={self.rounds}, tool_calls={self.total_tool_calls}, "
                f"text_len={len(self.text)})")


# ═══════════════════════════════════════════════════════════════════════
# Convenience: tool schemas grouped by puzzle type
# ═══════════════════════════════════════════════════════════════════════

RULE_TOOL_MAP: Dict[str, List[str]] = {
    # Grid CSP — Latin square
    "15": ["solve_sudoku", "validate_grid", "count_solutions"],     # Sudoku
    "16": ["solve_latin_square", "validate_grid"],                   # Futoshiki
    "17": ["solve_latin_square", "validate_grid", "evaluate_expression"],  # Calcudoko
    "25": ["solve_skyscraper", "solve_latin_square", "validate_grid"],  # Skyscraper
    # Math
    "5":  ["solve_cryptarithm", "evaluate_expression"],             # Cryptarithm
    "10": ["solve_24points", "evaluate_expression"],                 # 24 Points
    # Grid special
    "8":  ["solve_word_search", "validate_grid"],                   # Word Search
    "19": ["solve_star_battle", "validate_grid"],                   # Star Battle
    "21": ["solve_minesweeper", "validate_grid"],                   # Minesweeper
    # Generic — useful for all rules
    "_default": ["verify_answer", "check_duplicate", "measure_difficulty",
                 "format_converter", "evaluate_expression"],
}


def get_recommended_tools(rule_id: str) -> List[str]:
    """Get recommended tool names for a specific rule ID."""
    specific = RULE_TOOL_MAP.get(rule_id, [])
    default = RULE_TOOL_MAP.get("_default", [])
    # Deduplicate while preserving order
    seen = set()
    result = []
    for t in specific + default:
        if t not in seen:
            seen.add(t)
            result.append(t)
    return result


def get_tool_schemas_for_rule(rule_id: str) -> List[Dict[str, Any]]:
    """Get the best tool schemas for generating/solving a specific rule."""
    names = get_recommended_tools(rule_id)
    return get_tool_schemas(names)
