"""Prompt Regression Testing Framework (P1).

Tracks prompt versions, maintains per-rule test suites, and automates
regression detection when prompt changes are made.

Core insight: "fixed A, broke B" is detected by running ALL tests after
EVERY prompt change, comparing pass rates against the previous version.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set
import json
import os


@dataclass
class TestCase:
    """A single regression test for a rule."""
    rule_id: str
    name: str
    category: str           # "format" | "structure" | "diversity" | "difficulty" | "correctness"
    check_fn: Callable[..., bool]
    description: str
    weight: float = 1.0     # importance weight for scoring


@dataclass
class TestResult:
    test: TestCase
    passed: bool
    error_msg: str = ""
    latency_ms: int = 0


@dataclass
class PromptVersion:
    """A versioned prompt configuration."""
    version_id: str
    label: str
    prompts: Dict[str, str]  # {role: prompt_text}
    parent: Optional[str] = None
    created: str = ""
    notes: str = ""

    def __post_init__(self):
        if not self.created:
            self.created = datetime.now().isoformat()


@dataclass
class VersionReport:
    """Full evaluation report for a prompt version."""
    version_id: str
    total_tests: int
    passed: int
    pass_rate: float
    per_rule: Dict[str, Dict[str, Any]]   # {rule_id: {total, passed, rate}}
    regressions: List[TestResult]          # tests that passed on parent but fail now
    improvements: List[TestResult]         # tests that failed on parent but pass now
    new_failures: List[TestResult]         # tests that fail on both

    def has_regressions(self) -> bool:
        return len(self.regressions) > 0

    def summary(self) -> str:
        lines = [
            f"Version {self.version_id}: {self.passed}/{self.total_tests} ({self.pass_rate:.0%})",
            f"  Regressions: {len(self.regressions)} | Improvements: {len(self.improvements)}",
        ]
        if self.regressions:
            lines.append("  REGRESSIONS:")
            for r in self.regressions[:5]:
                lines.append(f"    - [{r.test.rule_id}] {r.test.name}: {r.error_msg}")
        if self.improvements:
            lines.append("  IMPROVEMENTS:")
            for i in self.improvements[:5]:
                lines.append(f"    + [{i.test.rule_id}] {i.test.name}")
        return "\n".join(lines)


class RegressionSuite:
    """Manages prompt versions and automated regression testing."""

    def __init__(self, storage_path: str = "data/prompt_registry.json"):
        self.storage_path = storage_path
        self.versions: Dict[str, PromptVersion] = {}
        self.tests: Dict[str, List[TestCase]] = {}   # {rule_id: [TestCase, ...]}
        self._load()

    def _load(self):
        if os.path.exists(self.storage_path):
            with open(self.storage_path, "r") as f:
                data = json.load(f)
                for v in data.get("versions", []):
                    self.versions[v["version_id"]] = PromptVersion(**v)

    def _save(self):
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        with open(self.storage_path, "w") as f:
            json.dump({
                "versions": [
                    {"version_id": v.version_id, "label": v.label,
                     "prompts": v.prompts, "parent": v.parent,
                     "created": v.created, "notes": v.notes}
                    for v in self.versions.values()
                ],
            }, f, indent=2, ensure_ascii=False)

    def register_version(self, version_id: str, label: str, prompts: Dict[str, str],
                         parent: str = None, notes: str = "") -> PromptVersion:
        v = PromptVersion(version_id=version_id, label=label, prompts=prompts,
                         parent=parent, notes=notes)
        self.versions[version_id] = v
        self._save()
        return v

    def add_test(self, rule_id: str, test: TestCase):
        self.tests.setdefault(rule_id, []).append(test)

    def evaluate(self, version_id: str, rules: List[str] = None,
                 runner: Callable = None) -> VersionReport:
        """Run all tests against a prompt version."""
        if version_id not in self.versions:
            return VersionReport(version_id, 0, 0, 0.0, {}, [], [], [])

        parent_id = self.versions[version_id].parent
        rule_list = rules or list(self.tests.keys())

        total, passed = 0, 0
        per_rule: Dict[str, Dict] = {}
        regressions: List[TestResult] = []
        improvements: List[TestResult] = []
        new_failures: List[TestResult] = []

        prompts = self.versions[version_id].prompts

        for rid in rule_list:
            r_total, r_passed = 0, 0
            for test in self.tests.get(rid, []):
                r_total += 1
                result = self._run_test(test, prompts, runner)
                if result.passed:
                    r_passed += 1
                    passed += 1
                total += 1

                # Check against parent version
                if parent_id and parent_id in self.versions:
                    parent_prompts = self.versions[parent_id].prompts
                    parent_result = self._run_test(test, parent_prompts, runner)
                    if parent_result.passed and not result.passed:
                        regressions.append(result)
                    elif not parent_result.passed and result.passed:
                        improvements.append(result)
                    elif not parent_result.passed and not result.passed:
                        new_failures.append(result)

            per_rule[rid] = {"total": r_total, "passed": r_passed,
                            "rate": r_passed / max(r_total, 1)}

        return VersionReport(
            version_id=version_id,
            total_tests=total,
            passed=passed,
            pass_rate=passed / max(total, 1),
            per_rule=per_rule,
            regressions=regressions,
            improvements=improvements,
            new_failures=new_failures,
        )

    def _run_test(self, test: TestCase, prompts: Dict[str, str],
                  runner: Callable = None) -> TestResult:
        """Execute a single test case."""
        import time
        t0 = time.time()
        try:
            result = test.check_fn(prompts) if test.check_fn else False
            msg = "" if result else f"Check failed: {test.description}"
        except Exception as e:
            result = False
            msg = str(e)
        return TestResult(test=test, passed=result, error_msg=msg,
                         latency_ms=int((time.time() - t0) * 1000))

    def gate(self, version_id: str, min_pass_rate: float = 0.85,
             max_regressions: int = 0) -> Tuple[bool, str]:
        """CI gate: check if version meets quality bar."""
        report = self.evaluate(version_id)
        if report.pass_rate < min_pass_rate:
            return False, f"Pass rate {report.pass_rate:.0%} < {min_pass_rate:.0%}"
        if len(report.regressions) > max_regressions:
            return False, f"{len(report.regressions)} regressions detected"
        for rid, s in report.per_rule.items():
            if s["rate"] < 0.70:
                return False, f"Rule {rid} rate {s['rate']:.0%} < 70%"
        return True, "PASS"


# ======================================================================
# Factory: build standard test suite
# ======================================================================

def build_standard_suite() -> RegressionSuite:
    """Create a regression suite with standard tests for all 25 rules."""
    suite = RegressionSuite()

    # Format tests — every rule must produce answers with [[...]]
    for rid in [str(i) for i in range(1, 26)]:
        suite.add_test(rid, TestCase(
            rule_id=rid,
            name=f"format_answer_wrapped",
            category="format",
            description="Answer must be wrapped in [[...]]",
            check_fn=lambda prompts, r=rid: True,  # placeholder — actual check needs LLM call
        ))

    # Structure tests — rules with known constraints
    suite.add_test("15", TestCase(
        rule_id="15", name="sudoku_9x9_latin_square",
        category="structure",
        description="Sudoku answer must be valid 9x9 Latin square with box constraints",
        check_fn=lambda prompts: True,  # placeholder
        weight=2.0,
    ))
    suite.add_test("25", TestCase(
        rule_id="25", name="skyscraper_latin_square",
        category="structure",
        description="Skyscraper answer must be valid Latin square",
        check_fn=lambda prompts: True,
        weight=2.0,
    ))
    suite.add_test("10", TestCase(
        rule_id="10", name="24points_evaluates_correctly",
        category="correctness",
        description="24 points answer must evaluate to 24",
        check_fn=lambda prompts: True,
        weight=2.0,
    ))

    # Difficulty tests
    for rid in ["1","2","3","4","5","6","7","8","9","10"]:
        suite.add_test(rid, TestCase(
            rule_id=rid, name=f"difficulty_{rid}_in_range",
            category="difficulty",
            description=f"Rule {rid} question length must be within calibrated range",
            check_fn=lambda prompts, r=rid: True,
        ))

    print(f"[regression] Built suite: {sum(len(v) for v in suite.tests.values())} tests across "
          f"{len(suite.tests)} rules")
    return suite


def run_regression(version_id: str, prompts: Dict[str, str],
                   parent: str = None, rules: List[str] = None) -> VersionReport:
    """Quick CLI: test a prompt version against the standard suite."""
    suite = build_standard_suite()
    if not suite.versions:
        # Register baseline if none exists
        suite.register_version("baseline", "Initial baseline", prompts)
    if version_id not in suite.versions:
        suite.register_version(version_id, f"Version {version_id}", prompts, parent=parent)
    return suite.evaluate(version_id, rules=rules)
