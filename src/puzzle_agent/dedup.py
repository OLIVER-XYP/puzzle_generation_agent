"""Canonicalization helpers for duplicate detection against the eval set."""
from __future__ import annotations
import hashlib
from typing import List


def _h(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def hash_canon(canon: str) -> str:
    return _h(canon)


def grid_symmetries(grid: List[List[str]]) -> List[List[List[str]]]:
    """All 8 dihedral symmetries of a square grid (rotations + reflections)."""
    def rot(g):
        return [list(row) for row in zip(*g[::-1])]
    def refl(g):
        return [row[::-1] for row in g]
    out = []
    g = [list(r) for r in grid]
    for _ in range(4):
        out.append([list(r) for r in g])
        out.append(refl(g))
        g = rot(g)
    return out


def canon_grid(grid: List[List[str]], use_symmetry: bool = True) -> str:
    """Canonical string for a grid, optionally minimal over the 8 symmetries."""
    def ser(g):
        return ",".join(" ".join(str(c) for c in row) for row in g)
    if not use_symmetry:
        return ser(grid)
    return min(ser(g) for g in grid_symmetries(grid))
