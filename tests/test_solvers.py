"""Solver/uniqueness smoke tests (run: python tests/test_solvers.py)."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from puzzle_agent.rules.r05_cryptomath import solve_cryptarithm, CryptoMath
from puzzle_agent.rules.r10_24points import enumerate_solutions
from puzzle_agent.rules.r25_skyscrapers import Skyscrapers, _visible
from puzzle_agent.config import load_config
from puzzle_agent.rules.r04_anagram import Anagram


def test_cryptarithm_send_more_money():
    sols = solve_cryptarithm(["SEND", "MORE"], "MONEY", cap=5)
    assert len(sols) == 1, f"SEND+MORE=MONEY must be unique, got {len(sols)}"
    s = sols[0]
    val = lambda w: int("".join(str(s[c]) for c in w))
    assert val("SEND") + val("MORE") == val("MONEY")
    print("OK cryptarithm SEND+MORE=MONEY unique =", s)


def test_24_points():
    sols = enumerate_solutions([9, 5, 2, 2], 24)
    assert sols, "9 5 2 2 should reach 24"
    # verify one expression numerically
    import re
    e = next(iter(sols))
    assert abs(eval(e.replace("/", "/")) - 24) < 1e-9
    print(f"OK 24 points 9 5 2 2 -> {len(sols)} solutions, e.g. {e}")


def test_skyscrapers_roundtrip_unique():
    sky = Skyscrapers()
    from random import Random
    rng = Random(1)
    uniq = 0
    for _ in range(20):
        sp = sky.generate({"n": 4}, rng, "t")
        sr = sky.solve(sp.input_data)
        assert sr.solvable, "the planted solution makes every puzzle solvable"
        # when the puzzle is uniquely solvable, the solver must recover the planted grid;
        # non-unique puzzles legitimately return a different first solution (and are rejected later)
        if sr.unique:
            assert sr.ground_truth == sp.input_data["solution"], "unique puzzle must recover planted solution"
        uniq += int(sr.unique)
    print(f"OK skyscrapers: {uniq}/20 generated 4x4 puzzles are uniquely solvable")
    assert uniq >= 10, "expected a healthy fraction of unique 4x4 puzzles"


def test_visible():
    assert _visible([1, 2, 3, 4]) == 4
    assert _visible([4, 3, 2, 1]) == 1
    assert _visible([2, 1, 4, 3]) == 2
    print("OK visibility counting")


def test_anagram():
    cfg = load_config()
    ana = Anagram(cfg)
    srcs = ana._anagram_sources(6)
    assert srcs, "need some 6-letter anagram sources"
    sr = ana.solve({"source": srcs[0], "mode": "all", "k": 6, "length": 6})
    assert sr.solvable and srcs[0] not in sr.ground_truth
    print(f"OK anagram: '{srcs[0]}' -> {sr.ground_truth}")


if __name__ == "__main__":
    test_visible()
    test_cryptarithm_send_more_money()
    test_24_points()
    test_skyscrapers_roundtrip_unique()
    test_anagram()
    print("\nALL TESTS PASSED")
