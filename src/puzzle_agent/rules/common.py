"""Shared helpers for rule implementations (grid parsing/rendering, Latin squares)."""
from __future__ import annotations
import re
from random import Random
from typing import Any, List


def parse_answer_grid(answer: str) -> List[List[str]]:
    """'[[a b,c d]]' -> [['a','b'],['c','d']]."""
    m = re.findall(r"\[\[(.*?)\]\]", answer, re.S)
    if not m:
        return []
    body = m[0]
    return [row.strip().split() for row in body.split(",") if row.strip() != ""]


def grid_rows_cols(answer: str):
    g = parse_answer_grid(answer)
    if not g:
        return 0, 0
    return len(g), len(g[0])


def render_matrix(grid, sep=" ", rowsep=","):
    return rowsep.join(sep.join(str(c) for c in row) for row in grid)


def wrap(body: str) -> str:
    return f"[[{body}]]"


def random_latin_square(n: int, rng: Random) -> List[List[int]]:
    """A uniformly-ish random n x n Latin square (values 1..n) via row retry."""
    base = list(range(1, n + 1))
    while True:
        rows: List[List[int]] = []
        cols = [set() for _ in range(n)]
        ok = True
        for _ in range(n):
            perm = base[:]
            rng.shuffle(perm)
            if any(perm[c] in cols[c] for c in range(n)):
                ok = False
                break
            rows.append(perm)
            for c in range(n):
                cols[c].add(perm[c])
        if ok:
            return rows


def neighbors4(r, c, R, C):
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < R and 0 <= nc < C:
            yield nr, nc
