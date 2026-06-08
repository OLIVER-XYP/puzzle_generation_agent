import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from puzzle_agent.benchmark import score_report
from puzzle_agent.config import load_config
from puzzle_agent.evolution import (
    EvolutionGate,
    gate_candidate,
    propose_prompt_versions,
)
from puzzle_agent.graph import build_context, build_graph
from puzzle_agent.memory import create_memory
from puzzle_agent.user_memory import (
    UserMemoryManager,
    UserProfile,
    format_user_context,
    set_user_profile,
    get_user_profile,
)


def test_user_profile_roundtrip(tmp_path):
    memory = create_memory(db_path=str(tmp_path / "memory.db"))
    profile = UserProfile(
        user_id="u1",
        preferred_rules={"10": 5},
        preferred_tags={"math": 3},
        difficulty_preference="hard",
    )
    set_user_profile(memory, profile)

    loaded = get_user_profile(memory, "u1")
    assert loaded.preferred_rules["10"] == 5
    assert loaded.preferred_tags["math"] == 3
    assert loaded.difficulty_preference == "hard"
    memory.close()


def test_feedback_parsing_updates_difficulty_and_tags(tmp_path):
    memory = create_memory(db_path=str(tmp_path / "memory.db"))
    manager = UserMemoryManager(memory, {"enabled": True, "default_user_id": "u1"})
    profile = manager.ingest_text("我喜欢难一点的数学题，给我2道24点")

    assert profile.difficulty_preference == "hard"
    assert profile.preferred_tags["math"] > 0
    assert profile.preferred_rules["10"] > 0
    memory.close()


def test_memory_context_truncates_high_signal_content():
    profile = UserProfile(
        user_id="u1",
        preferred_rules={str(i): i for i in range(1, 26)},
        preferred_tags={"math": 10, "word": 8},
        difficulty_preference="hard",
    )
    text = format_user_context(profile, max_chars=120)
    assert len(text) <= 120
    assert "difficulty_preference=hard" in text


def test_benchmark_scoring_formula():
    report = {
        "totals": {
            "requested": 10,
            "accepted": 8,
            "attempts": 12,
            "structural_passed": 8,
            "crosscheck_passed": 6,
            "duplicates": 1,
            "elapsed_s": 60,
        }
    }
    scores = score_report(report)
    assert scores["accept_rate"] == 0.8
    assert scores["crosscheck_pass_rate"] == 0.75
    assert 0 < scores["overall_score"] <= 1


def test_evolution_candidates_from_failure_types(tmp_path):
    cfg = load_config()
    cfg["_root"] = str(tmp_path)
    report = {
        "run_id": "bench-test",
        "prompt_version": "builtin",
        "top_rejection_reasons": {"bad_answer_format": 3, "structural_invalid": 2},
    }
    candidates = propose_prompt_versions(report, cfg, max_candidates=2)
    assert len(candidates) == 2
    assert "Answer Format" in candidates[0]["prompts"]["generator"]
    assert candidates[0]["parent"] == "builtin"


def test_promotion_gate_rejects_regressions():
    parent = {
        "scores": {
            "overall_score": 0.70,
            "structural_pass_rate": 0.80,
            "non_duplicate_rate": 0.90,
        },
        "per_rule": {"10": {"accept_rate": 0.80}},
    }
    candidate = {
        "scores": {
            "overall_score": 0.76,
            "structural_pass_rate": 0.70,
            "non_duplicate_rate": 0.90,
        },
        "per_rule": {"10": {"accept_rate": 0.80}},
    }
    result = gate_candidate(parent, candidate, EvolutionGate())
    assert not result["passed"]
    assert "structural pass rate regressed" in result["reasons"]


def test_graph_memory_smoke_with_mock_pipeline(tmp_path):
    cfg = load_config()
    cfg = copy.deepcopy(cfg)
    cfg["memory"]["db_path"] = str(tmp_path / "memory.db")
    cfg["run"]["rules"] = ["10"]
    cfg["run"]["count_per_rule"] = 1
    cfg["run"]["max_retries_per_item"] = 1
    cfg["run"]["recursion_limit"] = 100
    ctx = build_context(cfg)
    ctx.targets["10"] = {}

    class Output:
        question = "Use 9 5 2 2 to make 24."
        answer = "[[(9-5)*2+16]]"
        plan = "mock"
        generator_ok = True
        solver_ok = True
        reviewer_score = 8
        reviewer_issues = []

    def fake_run(*args, **kwargs):
        assert "user_id=default" in kwargs.get("memory_context", "")
        return Output()

    ctx.pipeline.run = fake_run
    app = build_graph(ctx)
    final = app.invoke({"rules": ["10"], "count": 1}, config={"recursion_limit": 100})
    assert final["user_memory_context"]
    assert ctx.user_memory.get_profile("default").generation_stats["accepted_by_rule"]["10"] == 1
    ctx.memory.close()
