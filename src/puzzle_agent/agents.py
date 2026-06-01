"""Multi-Agent architecture (P0).

Generator → creates puzzles with TODO-list planning + tool-assisted verification
Solver   → independently solves them (temperature=0, different prompt) + brute-force tools
Reviewer → compares answers, scores quality, decides PASS/FAIL + tool-assisted checks

All three are "blind" to each other — Generator doesn't see Solver's output,
Solver doesn't see Generator's reasoning. Reviewer is the impartial judge.

Each agent can call deterministic tools (validators, brute-force solvers) via
OpenAI function-calling to reduce errors and improve accuracy.
"""
from __future__ import annotations
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .prompt_builder import (
    GENERATOR_SYSTEM, SOLVER_SYSTEM, REVIEWER_SYSTEM,
    build_rule_prompt, ExampleSelector, format_examples,
    build_task_prompt, estimate_tokens, PromptBudget,
)
from .tracer import trace_call, get_tracer
from .tools import (
    ToolExecutor, ToolResult,
    get_tool_schemas, get_tool_schemas_for_role, get_tool_schemas_for_rule,
)


@dataclass
class AgentResult:
    agent_role: str
    raw_response: str
    parsed: Optional[Dict[str, Any]]
    errors: List[str]
    latency_ms: int


@dataclass
class GenerationOutput:
    question: str
    answer: str
    plan: str = ""
    generator_ok: bool = False
    solver_answer: str = ""
    solver_ok: bool = False
    reviewer_verdict: str = ""
    reviewer_score: int = 0
    reviewer_issues: List[str] = field(default_factory=list)
    latency_total_ms: int = 0


class LlmClient:
    """Thin wrapper around DeepSeek API with tool-calling support."""

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.model = cfg.get("generator", {}).get("model", "deepseek-chat")
        self.base_url = cfg.get("generator", {}).get("base_url", "https://api.deepseek.com")
        api_key = cfg.get("generator", {}).get("api_key", "")
        if not api_key:
            api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        self.api_key = api_key
        self.enabled = bool(api_key)
        self._client = None
        self.tool_executor = ToolExecutor()
        if self.enabled:
            from openai import OpenAI
            self._client = OpenAI(api_key=api_key, base_url=self.base_url)

    def chat(self, system: str, user: str, temperature: float = 0.7,
             max_tokens: int = 4096, model: str = None) -> Tuple[str, float]:
        """Call LLM. Returns (text, latency_seconds)."""
        if not self.enabled:
            return "", 0.0
        t0 = time.time()
        resp = self._client.chat.completions.create(
            model=model or self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        elapsed = time.time() - t0
        return (resp.choices[0].message.content or ""), elapsed

    def chat_raw(self, messages: List[Dict[str, Any]], tools: List[Dict] = None,
                 temperature: float = 0.7, max_tokens: int = 4096,
                 model: str = None):
        """Low-level LLM call with full messages list + optional tools.

        Returns the raw OpenAI response object, or None on failure.
        """
        if not self.enabled:
            return None
        kwargs = dict(
            model=model or self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        try:
            return self._client.chat.completions.create(**kwargs)
        except Exception:
            return None

    def chat_with_tools(self, system: str, user: str,
                        tools: List[Dict] = None,
                        temperature: float = 0.7, max_tokens: int = 4096,
                        max_rounds: int = 5, model: str = None) -> ToolResult:
        """Chat with automatic tool execution loop.

        The LLM can call tools (validators, solvers) and receive results
        before producing the final text response.

        Args:
            system: System prompt
            user: User prompt
            tools: OpenAI tool schemas. Default: generator-appropriate tools.
            temperature: LLM temperature
            max_tokens: Max output tokens per round
            max_rounds: Max tool-call rounds
            model: Model override

        Returns:
            ToolResult with .text (final response), .tool_calls_made, .rounds
        """
        if tools is None:
            tools = get_tool_schemas_for_role("generator")
        return self.tool_executor.chat_with_tools(
            client=self, system=system, user=user,
            tools=tools, temperature=temperature,
            max_tokens=max_tokens, max_rounds=max_rounds, model=model,
        )


# ======================================================================
# Generator Agent
# ======================================================================

class GeneratorAgent:
    """Creates puzzles with planning-guided generation + optional tool-assisted verification."""

    def __init__(self, client: LlmClient, max_attempts: int = 3):
        self.client = client
        self.max_attempts = max_attempts
        self.tool_stats: Dict[str, int] = {"calls": 0, "tool_rounds": 0}

    def generate(
        self, rule_content: str, rule_title: str, rule_id: str,
        examples: List[Dict], target: Dict,
        prior_errors: List[str] = None,
        use_tools: bool = False,
    ) -> AgentResult:
        selector = ExampleSelector(examples)
        avg_len = target.get("q_len_avg", 400) if target else 400
        bucket = "medium" if avg_len < 600 else ("short" if avg_len < 300 else "long")

        errors = list(prior_errors or [])

        for attempt in range(self.max_attempts):
            # Build prompt
            selected = selector.select(n=3, target_bucket=bucket)
            rule_block = build_rule_prompt(rule_content, rule_title)
            budget = PromptBudget()
            budget.add(GENERATOR_SYSTEM)
            budget.add(rule_block)
            trimmed = budget.truncate_examples(selected, max_tokens=3000)
            ex_block = "## Examples\n" + format_examples(trimmed)
            task_block = build_task_prompt(target, errors)

            full_prompt = "\n\n".join([rule_block, ex_block, task_block])

            # Call LLM (with or without tools)
            t0 = time.time()
            if use_tools and self.client.enabled:
                tools = get_tool_schemas_for_role("generator")
                tool_result = self.client.chat_with_tools(
                    GENERATOR_SYSTEM, full_prompt,
                    tools=tools,
                    temperature=0.7 + attempt * 0.1,
                    max_tokens=4096,
                    max_rounds=4,
                )
                raw = tool_result.text
                self.tool_stats["calls"] += 1
                self.tool_stats["tool_rounds"] += tool_result.total_tool_calls
            else:
                raw, _ = self.client.chat(
                    GENERATOR_SYSTEM, full_prompt,
                    temperature=0.7 + attempt * 0.1, max_tokens=4096,
                )
            latency = int((time.time() - t0) * 1000)

            # Parse
            parsed, parse_errs = self._parse(raw)
            if parse_errs:
                errors = parse_errs
                trace_call("generator", rule_id, attempt, "parsing",
                          GENERATOR_SYSTEM, full_prompt, raw,
                          errors=errors, model=self.client.model, temp=0.7)
                continue

            # Validate format
            fmt_errs = self._validate(parsed)
            if fmt_errs:
                errors = fmt_errs
                trace_call("generator", rule_id, attempt, "format_validation",
                          GENERATOR_SYSTEM, full_prompt, raw, parsed=parsed,
                          errors=errors, model=self.client.model, temp=0.7)
                continue

            # Validate structure
            from .validators import validate_answer
            struct_ok, struct_errs = validate_answer(rule_id, parsed["answer"], parsed["question"])
            if not struct_ok:
                errors = struct_errs
                trace_call("generator", rule_id, attempt, "structural_validation",
                          GENERATOR_SYSTEM, full_prompt, raw, parsed=parsed,
                          errors=errors, model=self.client.model, temp=0.7)
                continue

            trace_call("generator", rule_id, attempt, "success",
                      GENERATOR_SYSTEM, full_prompt, raw, parsed=parsed,
                      errors=[], model=self.client.model, temp=0.7,
                      start_time=t0)
            return AgentResult("generator", raw, parsed, [], latency)

        # All attempts failed
        return AgentResult("generator", "", None, errors, latency)

    def _parse(self, raw: str) -> Tuple[Optional[Dict], List[str]]:
        """Multi-strategy JSON parser."""
        raw = raw.strip()
        # Strip out <planning> block
        plan_match = re.search(r"<planning>(.*?)</planning>", raw, re.DOTALL)
        plan_text = plan_match.group(1) if plan_match else ""
        clean = raw.replace(plan_match.group(0), "") if plan_match else raw

        strategies = [
            # Strategy 1: ```json ... ```
            lambda s: re.search(r"```(?:json)?\s*\n?(.*?)\n?```", s, re.DOTALL),
            # Strategy 2: Raw JSON parse
            lambda s: s,
        ]
        for strat in strategies:
            m = strat(clean)
            if isinstance(m, str):
                try:
                    data = json.loads(m)
                    if "question" in data and "answer" in data:
                        data["planning"] = data.get("planning", plan_text.strip())
                        return data, []
                except json.JSONDecodeError:
                    pass
            elif m:
                try:
                    data = json.loads(m.group(1))
                    if "question" in data and "answer" in data:
                        data["planning"] = data.get("planning", plan_text.strip())
                        return data, []
                except json.JSONDecodeError:
                    pass

        # Last resort: regex extraction
        qm = re.search(r'"question"\s*:\s*"(.*?)"(?:\s*[,}])', raw, re.DOTALL)
        am = re.search(r'"answer"\s*:\s*"(.*?)"(?:\s*[,}])', raw, re.DOTALL)
        if qm and am:
            return {"question": _unescape(qm.group(1)),
                    "answer": _unescape(am.group(1)),
                    "planning": plan_text.strip()}, []

        return None, ["Failed to parse JSON. Ensure valid JSON with 'question' and 'answer' fields."]

    def _validate(self, parsed: Dict) -> List[str]:
        errors = []
        q = parsed.get("question", "")
        a = parsed.get("answer", "")
        if len(q) < 20:
            errors.append("Question too short (<20 chars)")
        if not a.strip().startswith("[["):
            errors.append("Answer must be wrapped in [[...]]")
        if len(a) < 6:
            errors.append("Answer too short")
        return errors


# ======================================================================
# Solver Agent
# ======================================================================

class SolverAgent:
    """Independent solver — different system prompt, temperature=0.

    With use_tools=True, gains access to brute-force solvers (sudoku,
    24points, cryptarithm, word search, minesweeper, etc.) and can
    call them to verify or compute answers deterministically.
    """

    def __init__(self, client: LlmClient):
        self.client = client
        self.tool_stats: Dict[str, int] = {"calls": 0, "tool_rounds": 0}

    def solve(self, rule_content: str, question: str, rule_id: str = "",
              use_tools: bool = False) -> AgentResult:
        prompt = build_rule_prompt(rule_content, "") + f"\n\n## Puzzle\n{question}"
        t0 = time.time()

        if use_tools and self.client.enabled:
            tools = get_tool_schemas_for_role("solver")
            # For specific rules, add rule-specific tools
            rule_tools = get_tool_schemas_for_rule(rule_id) if rule_id else []
            # Merge without duplicating
            seen_names = {t["function"]["name"] for t in tools}
            for rt in rule_tools:
                if rt["function"]["name"] not in seen_names:
                    tools.append(rt)
                    seen_names.add(rt["function"]["name"])

            tool_result = self.client.chat_with_tools(
                SOLVER_SYSTEM, prompt,
                tools=tools,
                temperature=0.0, max_tokens=2048, max_rounds=6,
            )
            raw = tool_result.text
            self.tool_stats["calls"] += 1
            self.tool_stats["tool_rounds"] += tool_result.total_tool_calls
        else:
            raw, _ = self.client.chat(SOLVER_SYSTEM, prompt, temperature=0.0, max_tokens=2048)

        latency_ms = int((time.time() - t0) * 1000)

        # Extract [[...]] from response
        ans = ""
        m = re.search(r"\[\[(.+?)\]\]", raw, re.DOTALL)
        if m:
            ans = f"[[{m.group(1)}]]"
        else:
            ans = raw.strip()

        trace_call("solver", rule_id, 0, "solving",
                   SOLVER_SYSTEM, prompt, raw,
                   parsed={"answer": ans},
                   model=self.client.model, temp=0.0,
                   start_time=t0)
        return AgentResult("solver", raw, {"answer": ans}, [], latency_ms)


# ======================================================================
# Reviewer Agent
# ======================================================================

class ReviewerAgent:
    """Compares Generator and Solver answers, scores quality.

    With use_tools=True, can call verify_answer, evaluate_expression,
    and validators to get deterministic verdicts on answer correctness.
    """

    def __init__(self, client: LlmClient):
        self.client = client
        self.tool_stats: Dict[str, int] = {"calls": 0, "tool_rounds": 0}

    def review(self, rule_content: str, question: str,
               gen_answer: str, sol_answer: str, rule_id: str = "",
               use_tools: bool = False) -> AgentResult:
        prompt = (
            f"## Puzzle Rule\n{rule_content}\n\n"
            f"## Puzzle Question\n{question}\n\n"
            f"## Generator Answer (to evaluate)\n{gen_answer}\n\n"
            f"## Independent Solver Answer (ground truth reference)\n{sol_answer}\n\n"
            f"## Task\nCompare the two answers. Are they equivalent? "
            f"Does the Generator's answer satisfy all constraints? Output JSON."
        )
        t0 = time.time()

        if use_tools and self.client.enabled:
            tools = get_tool_schemas_for_role("reviewer")
            tool_result = self.client.chat_with_tools(
                REVIEWER_SYSTEM, prompt,
                tools=tools,
                temperature=0.0, max_tokens=1024, max_rounds=3,
            )
            raw = tool_result.text
            self.tool_stats["calls"] += 1
            self.tool_stats["tool_rounds"] += tool_result.total_tool_calls
        else:
            raw, _ = self.client.chat(REVIEWER_SYSTEM, prompt, temperature=0.0, max_tokens=1024)

        latency_ms = int((time.time() - t0) * 1000)

        parsed = None
        try:
            m = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
            parsed = json.loads(m.group(1) if m else raw)
        except (json.JSONDecodeError, AttributeError):
            parsed = {"verdict": "UNCLEAR", "score": 5, "issues": ["Could not parse reviewer output"]}

        trace_call("reviewer", rule_id, 0, "reviewing",
                   REVIEWER_SYSTEM, prompt, raw,
                   parsed=parsed,
                   model=self.client.model, temp=0.0,
                   start_time=t0)
        return AgentResult("reviewer", raw, parsed, [], latency_ms)


# ======================================================================
# Pipeline Orchestrator
# ======================================================================

class MultiAgentPipeline:
    """Orchestrate Generator → Solver → Reviewer with mutual cross-checking.

    Set use_tools=True to give each agent access to deterministic tools
    (brute-force solvers, validators, difficulty checkers) via function calling.
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.client = LlmClient(cfg)
        self.generator = GeneratorAgent(self.client, max_attempts=cfg.get("generator", {}).get("max_generation_attempts", 3))
        self.solver = SolverAgent(self.client)
        self.reviewer = ReviewerAgent(self.client)
        self.cfg = cfg
        self.use_tools = cfg.get("generator", {}).get("use_tools", True)

    def run(self, rule_content: str, rule_title: str, rule_id: str,
            examples: List[Dict], target: Dict,
            require_reviewer_pass: bool = True,
            use_tools: bool = None) -> GenerationOutput:
        t0_total = time.time()

        # Determine whether to use tools (pipeline-level or per-call override)
        _use_tools = use_tools if use_tools is not None else self.use_tools

        # Stage 1: Generate (optionally with tool-assisted self-verification)
        gen_result = self.generator.generate(
            rule_content, rule_title, rule_id, examples, target,
            use_tools=_use_tools,
        )
        if not gen_result.parsed:
            return GenerationOutput(
                question="", answer="",
                generator_ok=False,
                latency_total_ms=int((time.time() - t0_total) * 1000),
            )

        question = gen_result.parsed["question"]
        gen_answer = gen_result.parsed["answer"]

        # Stage 2: Independent solve (optionally with brute-force tools)
        sol_result = self.solver.solve(rule_content, question, rule_id=rule_id,
                                       use_tools=_use_tools)
        sol_answer = sol_result.parsed.get("answer", "") if sol_result.parsed else ""

        # Stage 3: Review (optionally with verification tools)
        rev_result = self.reviewer.review(rule_content, question, gen_answer, sol_answer,
                                          rule_id=rule_id, use_tools=_use_tools)
        rev_verdict = "FAIL"
        rev_score = 0
        rev_issues = []
        if rev_result.parsed:
            rev_verdict = rev_result.parsed.get("verdict", "FAIL")
            rev_score = rev_result.parsed.get("score", 0)
            rev_issues = rev_result.parsed.get("issues", [])

        if require_reviewer_pass and rev_verdict != "PASS":
            return GenerationOutput(
                question=question, answer=gen_answer,
                plan=gen_result.parsed.get("planning", ""),
                generator_ok=True, solver_answer=sol_answer, solver_ok=True,
                reviewer_verdict=rev_verdict, reviewer_score=rev_score,
                reviewer_issues=rev_issues,
                latency_total_ms=int((time.time() - t0_total) * 1000),
            )

        return GenerationOutput(
            question=question, answer=gen_answer,
            plan=gen_result.parsed.get("planning", ""),
            generator_ok=True, solver_answer=sol_answer, solver_ok=True,
            reviewer_verdict="PASS", reviewer_score=rev_score,
            reviewer_issues=rev_issues,
            latency_total_ms=int((time.time() - t0_total) * 1000),
        )

    def get_tool_stats(self) -> Dict[str, Any]:
        """Aggregate tool usage statistics across all agents."""
        return {
            "generator": dict(self.generator.tool_stats),
            "solver": dict(self.solver.tool_stats),
            "reviewer": dict(self.reviewer.tool_stats),
            "total_rounds": (self.generator.tool_stats.get("tool_rounds", 0) +
                           self.solver.tool_stats.get("tool_rounds", 0) +
                           self.reviewer.tool_stats.get("tool_rounds", 0)),
        }


# ======================================================================
# Helpers
# ======================================================================

def _unescape(s: str) -> str:
    s = s.replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t')
    return s


# ======================================================================
# Factory
# ======================================================================

def create_pipeline(config_path: str = None) -> MultiAgentPipeline:
    from .config import load_config
    cfg = load_config(config_path)
    return MultiAgentPipeline(cfg)
