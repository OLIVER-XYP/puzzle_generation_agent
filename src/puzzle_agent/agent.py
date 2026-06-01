"""PuzzleAgent — conversational agent with multi-agent pipeline (P0).

Uses:
  - QueryRewriter (LLM intent parsing + clarification)
  - MultiAgentPipeline (Generator→Solver→Reviewer)
  - Tracer (per-call trace + auto-diagnosis + SFT recommendations)
  - Enhanced prompts (TODO list planning)

Supports:
  - Single and multi-intent queries
  - Clarification questions when intent is ambiguous
  - Fast path (regex) + slow path (LLM) intent routing
"""
from __future__ import annotations
import json
import os
import hashlib
from typing import Any, Dict, List, Optional, Tuple

from .rewriter import QueryRewriter, ParsedIntent, RewriteResult, create_rewriter
from .agents import MultiAgentPipeline, create_pipeline, LlmClient
from .tracer import get_tracer
from .tools import get_tools
from .scheduler import create_executor, ParallelExecutor


class PuzzleAgent:
    """Conversational puzzle-generation agent."""

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.pipeline = create_pipeline()
        self.client = LlmClient(cfg)
        self.tracer = get_tracer()

        # Load rules metadata
        from .config import resolve
        self._rules: Dict[str, List[Dict]] = {}
        eval_path = resolve(cfg, cfg["run"]["source_eval"])
        with open(eval_path, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                self._rules.setdefault(r["rule_id"], []).append(r)

        # Build rule index
        self.rule_index: Dict[str, Dict] = {}
        for rid, examples in self._rules.items():
            self.rule_index[rid] = {
                "title": examples[0]["title"],
                "tag": examples[0].get("tag", ""),
                "rule_content": examples[0]["rule_content"],
                "examples": examples,
                "target": self._calc_target(examples),
            }

        # Query rewriter (with LLM for complex queries)
        self.rewriter = create_rewriter(self.rule_index, cfg)

        # Session manager (tracks first vs supplement mode)
        from .session import SessionManager
        self.session = SessionManager()

        # Tools and scheduler
        self.tools = get_tools()
        self.executor = create_executor(max_workers=5, max_rpm=30)

        self.messages: List[Dict] = []          # conversation history

    def _calc_target(self, examples):
        return {
            "q_len_avg": sum(len(e["question"]) for e in examples) / len(examples),
            "q_len_min": min(len(e["question"]) for e in examples),
            "q_len_max": max(len(e["question"]) for e in examples),
        }

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def chat(self, user_message: str) -> str:
        """Process one user message. Returns Agent response."""
        # Step 1: Rewrite query → structured intents
        session_summary = self._session_summary()
        rewritten = self.rewriter.rewrite(user_message, session_summary)

        # Step 2: If clarification needed, ask user
        if rewritten.clarification_needed and rewritten.clarify_question:
            return f"🤔 {rewritten.clarify_question}"

        # Step 3: Execute intents
        # If multiple GENERATE intents, use parallel executor
        gen_intents = [i for i in rewritten.intents if i.action == "GENERATE"]
        other_intents = [i for i in rewritten.intents if i.action != "GENERATE"]

        results = []
        # Execute non-generate intents first (e.g., INSPECT)
        for intent in other_intents:
            results.append(self._execute(intent))

        # Parallel generation for multiple rules
        if len(gen_intents) > 1:
            multi_result = self._execute_parallel(gen_intents)
            results.insert(0, multi_result)
        elif len(gen_intents) == 1:
            results.insert(0, self._execute(gen_intents[0]))

        # Step 4: Polish with LLM
        if len(results) == 1:
            return self._polish(rewritten.intents[0], results[0])
        else:
            # Multi-intent: summarize all
            summary = "\n".join(
                f"  • {self._summarize(i, r)}" for i, r in zip(rewritten.intents, results)
            )
            return f"执行了 {len(results)} 个操作:\n{summary}"

    # ------------------------------------------------------------------ #
    # Intent execution
    # ------------------------------------------------------------------ #

    def _execute(self, intent: ParsedIntent) -> Dict[str, Any]:
        action = intent.action
        params = intent.params

        if action == "LIST_RULES":
            return self._do_list_rules()
        elif action == "INSPECT_RULE":
            return self._do_inspect(params.get("rule_id", ""))
        elif action == "GENERATE":
            return self._do_generate(
                params["rule_id"], params.get("count", 1), params.get("bucket", "medium"))
        elif action == "VALIDATE":
            return self._do_validate()
        elif action == "EXPORT":
            return self._do_export(params.get("path", "data/out/fine_dataset.jsonl"))
        elif action == "STATS":
            return self._do_stats()
        elif action == "HELP":
            return self._do_help()
        return {"error": f"Unknown action: {action}"}

    def _do_list_rules(self) -> Dict:
        lines = []
        for rid in sorted(self.rule_index.keys(), key=int):
            r = self.rule_index[rid]
            tag = r["tag"]
            lines.append(f"• Rule {rid}: **{r['title']}** ({tag}) — {len(r['examples'])} 例")
        return {"action": "list_rules", "rules": lines, "total": len(lines)}

    def _do_inspect(self, rid: str) -> Dict:
        if rid not in self.rule_index:
            return {"error": f"Rule {rid} 不存在。可用: {sorted(self.rule_index.keys())}"}
        r = self.rule_index[rid]
        return {
            "action": "inspect",
            "rule_id": rid,
            "title": r["title"],
            "tag": r["tag"],
            "rule_content": r["rule_content"],
            "n_examples": len(r["examples"]),
            "difficulty": r["target"],
            "sample": {
                "question": r["examples"][0]["question"][:300],
                "answer": r["examples"][0]["answer"][:100],
            },
        }

    def _do_generate(self, rid: str, count: int, bucket: str) -> Dict:
        if rid not in self.rule_index:
            return {"error": f"Rule {rid} 不存在"}

        r = self.rule_index[rid]

        # Start a batch (detects first vs supplement mode)
        batch = self.session.start_batch(rid, r["title"], count)
        context = self.session.build_context(rid, current_batch_id=batch.batch_id)
        is_supplement = context["mode"] == "supplement"

        results = []
        for i in range(count):
            output = self.pipeline.run(
                r["rule_content"], r["title"], rid, r["examples"], r["target"],
                require_reviewer_pass=False,
                use_tools=self.pipeline.use_tools,  # respect pipeline-level setting
            )
            rec = {
                "index": i + 1,
                "question": output.question[:200] if output.question else "",
                "answer": output.answer[:150] if output.answer else "",
                "generator_ok": output.generator_ok,
                "reviewer_score": output.reviewer_score,
                "latency_s": round(output.latency_total_ms / 1000, 1),
            }
            self.session.add_result(batch.batch_id, rec)
            results.append(rec)

        passed = sum(1 for r in results if r["generator_ok"])
        return {
            "action": "generate", "rule_id": rid, "title": r["title"],
            "mode": "supplement" if is_supplement else "first",
            "count": count, "passed": passed, "results": results,
        }

    def _execute_parallel(self, intents: List[ParsedIntent]) -> Dict[str, Any]:
        """Execute multiple GENERATE intents in parallel using ThreadPoolExecutor."""
        jobs = []
        for intent in intents:
            rid = intent.params["rule_id"]
            count = intent.params.get("count", 1)
            r = self.rule_index.get(rid, {})
            jobs.append({"rule_id": rid, "title": r.get("title", f"Rule {rid}"), "count": count})

        def _gen_fn(rid, count):
            r = self.rule_index.get(rid)
            if not r:
                return [{"error": f"Rule {rid} not found"}]
            batch = self.session.start_batch(rid, r["title"], count)
            results = []
            for i in range(count):
                output = self.pipeline.run(
                    r["rule_content"], r["title"], rid, r["examples"], r["target"],
                    require_reviewer_pass=False,
                    use_tools=self.pipeline.use_tools)
                rec = {"index": i+1, "question": output.question[:200] if output.question else "",
                       "answer": output.answer[:150] if output.answer else "",
                       "generator_ok": output.generator_ok,
                       "latency_s": round(output.latency_total_ms/1000, 1)}
                self.session.add_result(batch.batch_id, rec)
                results.append(rec)
            return results

        per_rule = self.executor.generate_multi_rule(_gen_fn, jobs, progress=True)
        total_ok = sum(v["passed"] for v in per_rule.values())
        total_cnt = sum(len(v["results"]) for v in per_rule.values())
        return {"action": "generate_multi", "total_ok": total_ok, "total_count": total_cnt,
                "per_rule": {rid: {"passed": v["passed"], "failed": v["failed"]}
                            for rid, v in per_rule.items()}}

    def _do_validate(self) -> Dict:
        if self.session.is_empty():
            return {"error": "还没有生成过题目"}

        total = self.session.total_generated()
        issues = []
        for b in self.session.batches:
            for rec in b.records:
                if not rec.get("generator_ok"):
                    issues.append(f"Rule {b.rule_id} #{rec['index']}: 生成失败")
                if rec.get("reviewer_score", 0) < 5:
                    issues.append(f"Rule {b.rule_id} #{rec['index']}: Reviewer评分低({rec['reviewer_score']})")

        return {
            "action": "validate",
            "total": total,
            "passed": passed,
            "pass_rate": f"{passed/max(total,1)*100:.0f}%",
            "issues": issues[:10],
        }

    def _do_export(self, path: str) -> Dict:
        records = []
        for batch in self.session.batches:
            for rec in batch.records:
                if rec.get("generator_ok") and rec.get("question"):
                    records.append({
                        "idx": f"syn-{batch.rule_id}-{rec['index']}",
                        "rule_id": batch.rule_id,
                        "title": batch.rule_title,
                        "question": rec["question"],
                        "answer": rec["answer"],
                        "metadata": {"reviewer_score": rec.get("reviewer_score", 0)},
                    })

        if not records:
            return {"error": "没有可导出的数据"}

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        return {"action": "export", "exported": len(records), "path": os.path.abspath(path)}

    def _do_stats(self) -> Dict:
        trace = self.tracer.summary()
        return {
            "action": "stats",
            "batches": len(self.session.batches),
            "generated": self.session.total_generated(),
            "trace": trace,
            "trace_per_agent": trace.get("by_role", {}),
        }

    def _do_help(self) -> Dict:
        return {"action": "help", "text": _HELP_TEXT}

    # ------------------------------------------------------------------ #
    # Response polishing
    # ------------------------------------------------------------------ #

    def _polish(self, intent: ParsedIntent, result: Dict) -> str:
        """Use LLM to convert structured result → natural Chinese response."""
        if result.get("error"):
            return f"❌ {result['error']}"

        prompts = {
            "LIST_RULES": "Summarize the rule list into a friendly Chinese message. Mention the 3 categories (字谜/数学/空间). Keep under 200 chars.",
            "INSPECT_RULE": "Describe the rule briefly in Chinese. Include: title, what it's about, difficulty range, and show the example question. Keep under 300 chars.",
            "GENERATE": "Summarize generation results in Chinese. Show count, pass rate, and 1-2 example question+answer pairs. Keep under 300 chars.",
            "VALIDATE": "Report validation results in Chinese. State pass rate and any issues found.",
            "EXPORT": "Confirm export in Chinese. State file path and record count.",
            "STATS": "Report stats in Chinese. Include total generated, per-agent trace.",
            "HELP": "Output the help text verbatim.",
        }

        prompt = prompts.get(intent.action, "Summarize in Chinese under 200 chars.")
        msgs = [
            {"role": "system", "content": "你是友好的谜题助手。直接回复用户，不要多余的话。"},
            {"role": "user", "content": f"{prompt}\n\n数据: {json.dumps(result, ensure_ascii=False)[:1500]}"},
        ]
        raw, _ = self.client.chat(msgs[0]["content"], msgs[1]["content"],
                                   temperature=0.3, max_tokens=512)
        return raw or self._summarize(intent, result)

    def _summarize(self, intent: ParsedIntent, result: Dict) -> str:
        """Fallback: plain text summary without LLM."""
        a = intent.action
        if a == "LIST_RULES":
            return f"共 {result.get('total', '?')} 种规则。\n" + "\n".join(result.get("rules", [])[:30])
        elif a == "GENERATE":
            return (f"{result['title']}: 生成 {result['count']} 道，通过 {result['passed']}/{result['count']}。"
                    f" Reviewer平均分: {sum(r['reviewer_score'] for r in result['results'])/max(len(result['results']),1):.0f}/10")
        elif a == "VALIDATE":
            return f"验证: {result['total']} 条，通过率 {result['pass_rate']}"
        elif a == "EXPORT":
            return f"已导出 {result['exported']} 条 → {result['path']}"
        elif a == "STATS":
            t = result.get("trace", {})
            return (f"生成 {result['generated']} 题 | "
                    f"API调用 {t.get('total_calls',0)} | 成功率 {t.get('pass_rate','?')}")
        elif a == "HELP":
            return result.get("text", _HELP_TEXT)
        return json.dumps(result, ensure_ascii=False, indent=2)[:500]

    def _session_summary(self) -> str:
        return self.session.describe_context()


_HELP_TEXT = """🤖 **PuzzleAgent** (Multi-Agent v2)

你可以这样和我对话：
• "列出所有规则" — 查看 25 种谜题类型
• "查看规则10" — 看24点游戏的规则和样例
• "给规则25生成5道中等难度题" — 摩天大楼题
• "出几道简单的数学题" — 自动扩展数学类规则
• "验证一下质量" — 检查已生成的数据
• "导出到 data/out.jsonl" — 保存结果"""


# ======================================================================
# Factory
# ======================================================================

def create_agent(config_path: str = None) -> PuzzleAgent:
    from .config import load_config
    cfg = load_config(config_path)
    return PuzzleAgent(cfg)
