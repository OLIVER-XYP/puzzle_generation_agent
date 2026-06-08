"""LLM-based query rewriting (P0).

Replaces regex-based intent matching with LLM understanding.
Supports: multi-intent, domain expansion, clarification questions,
parallelized intent extraction.
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .prompt_builder import REWRITER_SYSTEM


# ======================================================================
# Domain → Rule mapping
# ======================================================================

DOMAIN_MAP = {
    "数学": ["9","10","11","12","13","14","15","16","17","25"],
    "字谜": ["1","2","3","4","5","6","7","8","24"],
    "空间": ["18","19","20","21","22","23"],
    "全部": ["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25"],
    "math": ["9","10","11","12","13","14","15","16","17","25"],
    "word": ["1","2","3","4","5","6","7","8","24"],
    "spatial": ["18","19","20","21","22","23"],
    "all": ["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25"],
}

DIFFICULTY_ALIASES = {
    "简单": "short", "容易": "short", "短": "short", "easy": "short",
    "中等": "medium", "普通": "medium", "一般": "medium", "mid": "medium",
    "困难": "long", "难": "long", "难一点": "long", "难一些": "long",
    "难些": "long", "复杂": "long", "烧脑": "long", "hard": "long",
}

# Rule-name aliases (Chinese + English) → rule_id.
# Lets users say "摩天大楼"/"skyscraper" instead of "规则25".
RULE_NAME_ALIASES = {
    # 1-8, 24 (word)
    "脑筋急转弯": "1", "word brain": "1", "brain teaser": "1",
    "词根": "2", "词缀": "2", "affix": "2", "root": "2",
    "连词": "3", "组词": "3", "connect word": "3",
    "字母重排": "4", "变位词": "4", "anagram": "4",
    "密码数学": "5", "加密数学": "5", "crypto": "5", "cryptarithm": "5",
    "单词阶梯": "6", "词梯": "6", "word ladder": "6",
    "逻辑": "7", "逻辑谜题": "7", "logic": "7",
    "单词搜索": "8", "找词": "8", "word search": "8",
    "填字": "24", "wordscape": "24",
    # 9-17, 25 (math)
    "数学路径": "9", "math path": "9",
    "24点": "10", "二十四点": "10", "24 point": "10", "24points": "10",
    "survo": "11",
    "kukurasu": "12", "库库拉苏": "12",
    "numbrix": "13", "数字接龙": "13",
    "数字墙": "14", "number wall": "14",
    "数独": "15", "sudoku": "15", "sudoko": "15",
    "计算数独": "16", "calcudoku": "16", "calcudoko": "16", "kenken": "16",
    "不等式": "17", "futoshiki": "17",
    "摩天大楼": "25", "摩天楼": "25", "skyscraper": "25", "skyscrapers": "25",
    # 18-23 (spatial)
    "向量": "18", "矢量": "18", "vector": "18",
    "星战": "19", "星空": "19", "star battle": "19",
    "露营": "20", "帐篷": "20", "campsite": "20", "tents": "20",
    "扫雷": "21", "minesweeper": "21",
    "箭头迷宫": "22", "arrow maze": "22",
    "norinori": "23", "诺里诺里": "23",
}

# Chinese numerals → int (for "两道", "三个", "十题" ...)
CN_NUMERALS = {
    "零": 0, "一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
}


def _parse_count(query: str) -> Optional[int]:
    """Extract a count from query: Arabic digits OR Chinese numerals."""
    m = re.search(r'(\d+)\s*(?:个|道|题|条|次|puzzle|each)', query)
    if m:
        return int(m.group(1))
    m2 = re.search(r'(?:each|per)\s+(\d+)', query)
    if m2:
        return int(m2.group(1))
    # English: "give me 2 ...", "generate 3 ...", "make 5 ..."
    m_en = re.search(r'(?:give\s+me|give|generate|make|create|want|add|need)\s+(\d+)', query)
    if m_en:
        return int(m_en.group(1))
    # Chinese numeral immediately before a counter word
    m3 = re.search(r'([零一两二三四五六七八九十]+)\s*(?:个|道|题|条|次)', query)
    if m3:
        return _cn_to_int(m3.group(1))
    return None


def _cn_to_int(s: str) -> int:
    """Convert simple Chinese numeral string to int (supports 1-99)."""
    if s in CN_NUMERALS:
        return CN_NUMERALS[s]
    if "十" in s:
        parts = s.split("十")
        tens = CN_NUMERALS.get(parts[0], 1) if parts[0] else 1
        ones = CN_NUMERALS.get(parts[1], 0) if len(parts) > 1 and parts[1] else 0
        return tens * 10 + ones
    return CN_NUMERALS.get(s, 1)


@dataclass
class ParsedIntent:
    action: str                      # LIST_RULES | INSPECT_RULE | GENERATE | VALIDATE | EXPORT | STATS | HELP | RECOMMEND
    params: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0


@dataclass
class RewriteResult:
    original_query: str
    intents: List[ParsedIntent]
    missing_fields: List[str] = field(default_factory=list)
    clarification_needed: bool = False
    clarify_question: str = ""
    raw_llm_output: str = ""


class QueryRewriter:
    """Intent parser: regex fast-path for simple queries, LLM for complex ones."""

    def __init__(self, rules: Dict[str, Any], llm_client=None):
        self.rules = rules
        self.llm = llm_client

    def rewrite(self, query: str, session_summary: str = "") -> RewriteResult:
        """Parse user query into structured intent(s)."""
        query_lower = query.lower().strip()

        # ---- Fast path: simple queries handled by regex ----
        fast = self._fast_path(query_lower, session_summary)

        # ---- Compound query detection: try to split and parse each part ----
        if self._is_compound(query_lower):
            compound = self._parse_compound(query_lower, session_summary)
            if compound and compound.intents:
                return compound

        # If fast path found intents, return them
        if fast and fast.intents:
            return fast

        # ---- Slow path: LLM-based parsing for complex queries ----
        llm_result = self._llm_path(query, session_summary)
        if fast and fast.intents:
            llm_result.intents = fast.intents + llm_result.intents
        return llm_result

    def _parse_compound(self, query: str, session_summary: str = "") -> Optional[RewriteResult]:
        """Split compound query on connectors and parse each part separately.

        E.g. "看规则25然后各出2道" → inspect(25) + generate(25,2)
        """
        import re
        # Split on common connectors
        parts = re.split(r'(?:然后|接着|并且|还有|，|,)\s*', query)
        if len(parts) <= 1:
            return None

        all_intents = []
        for part in parts:
            r = self._fast_path(part.strip(), session_summary)
            if r and r.intents:
                all_intents.extend(r.intents)

        if all_intents:
            return RewriteResult(query, all_intents)
        return None

    def _fast_path(self, query: str, session_summary: str = "") -> Optional[RewriteResult]:
        """Regex-based parsing for simple, unambiguous queries."""
        intents = []
        context_rules = self._context_rule_ids(session_summary)

        # LIST
        if any(w in query for w in ["list", "列出", "所有规则", "有哪些规则", "什么规则"]):
            intents.append(ParsedIntent("LIST_RULES"))
            return RewriteResult(query, intents)

        # HELP
        if any(w in query for w in ["帮助", "help", "怎么用", "能做什么", "功能"]):
            intents.append(ParsedIntent("HELP"))
            return RewriteResult(query, intents)

        # STATS
        if any(w in query for w in ["统计", "生成了多少", "进度"]):
            intents.append(ParsedIntent("STATS"))
            return RewriteResult(query, intents)

        # RECOMMEND / NEXT-BEST-ACTION
        if any(w in query for w in [
            "推荐", "下一步", "接下来", "适合", "该做什么", "recommend", "suggest", "next"
        ]):
            intents.append(ParsedIntent("RECOMMEND", {"limit": _parse_count(query) or 3}))
            return RewriteResult(query, intents)

        # VALIDATE
        if any(w in query for w in ["验证", "检查", "validate", "check data", "quality"]):
            intents.append(ParsedIntent("VALIDATE"))
            return RewriteResult(query, intents)

        # EXPORT
        if any(w in query for w in ["导出", "保存", "export"]):
            path = "data/out/fine_dataset.jsonl"
            m = re.search(r'(?:到|to|path[:=])\s*(\S+)', query)
            if m:
                path = m.group(1)
            intents.append(ParsedIntent("EXPORT", {"path": path}))
            return RewriteResult(query, intents)

        # GENERATE or INSPECT
        domain_rules = self._extract_domain_rules(query)
        rule_ids = self._extract_rule_ids(query)
        strong_domain_signal = any(w in query for w in ["网格", "宫格", "grid"])
        if rule_ids and domain_rules and (
            strong_domain_signal
            or (any(w in query for w in ["类似", "像", "like"]) and any(w in query for w in ["不要", "不是", "不一样", "换个", "not"]))
        ):
            rule_ids = []
        followup_generate = self._is_followup_generate(query)
        if not rule_ids and followup_generate and context_rules:
            if any(w in query for w in ["都", "各", "each", "all"]):
                rule_ids = context_rules[:3]
            else:
                rule_ids = [context_rules[0]]
        if rule_ids:
            count_explicit = _parse_count(query)
            count = count_explicit or 1

            bucket = "medium"
            for dk, dv in DIFFICULTY_ALIASES.items():
                if dk in query:
                    bucket = dv
                    break

            if any(w in query for w in ["查看", "看看", "看", "显示", "inspect", "详情", "show", "显示"]):
                intents.append(ParsedIntent("INSPECT_RULE", {"rule_id": rule_ids[0]}))
                return RewriteResult(query, intents)

            gen_verbs = ["生成", "出题", "出", "造题", "造", "做", "generate", "题目",
                         "问题", "编", "给我", "给", "想要", "要", "来", "give",
                         "create", "make", "add", "more", "another", "再", "加", "each",
                         "继续", "类似", "刚才", "上次", "那种", "同样"]
            # An explicit count ("十道", "3个") is itself a generation signal,
            # even without a verb (e.g. "十道困难的数独").
            if any(w in query for w in gen_verbs) or count_explicit is not None:
                # Multi-rule support: "each for rules 1,2,3" or "rules 10 and 25 each 2"
                is_multi = any(w in query for w in ["each", "各", "每", "分别", "都"])
                target_rids = rule_ids if is_multi else [rule_ids[0]]
                for rid in target_rids:
                    intents.append(ParsedIntent("GENERATE", {
                        "rule_id": rid,
                        "count": min(count, 20),
                        "bucket": bucket,
                    }))
                return RewriteResult(query, intents)

        # Domain-based and fuzzy category requests:
        # "出几道数学题" / "来点文字类" / "类似数独但不一样" / "烧脑推理"
        if domain_rules and self._is_generate_request(query):
            count = _parse_count(query) or 1
            bucket = "medium"
            for dk, dv in DIFFICULTY_ALIASES.items():
                if dk in query:
                    bucket = dv
                    break
            for rid in domain_rules[:3]:
                intents.append(ParsedIntent("GENERATE", {
                    "rule_id": rid, "count": count, "bucket": bucket,
                }))
            if intents:
                return RewriteResult(query, intents)

        return None

    def _llm_path(self, query: str, session_summary: str) -> RewriteResult:
        """LLM-based parsing for complex or ambiguous queries."""
        if not self.llm:
            return RewriteResult(query, [], clarification_needed=True,
                                clarify_question="抱歉，我不太理解你的意思。能换个说法吗？")

        context = f"可用规则: {len(self.rules)}个 (ID 1-25)。之前对话: {session_summary or '无'}"
        raw, _ = self.llm.chat(REWRITER_SYSTEM,
            f"{context}\n\n用户: {query}", temperature=0.0, max_tokens=1024)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Fallback to regex
            return RewriteResult(query, [], clarification_needed=True,
                                clarify_question="抱歉，我暂时没理解。你可以直接说「给规则10生成5道题」这样。")

        intents = []
        missing = data.get("missing_fields", [])
        clarify = data.get("clarify_question", "")

        for intent_data in data.get("intents", []):
            action = intent_data.get("action", "").upper().strip()
            if action in ("LIST", "LIST_RULES"):
                intents.append(ParsedIntent("LIST_RULES"))
            elif action in ("INSPECT", "INSPECT_RULE"):
                params = intent_data.get("params", {})
                intents.append(ParsedIntent("INSPECT_RULE", {"rule_id": str(params.get("rule_id", ""))}))
            elif action in ("GENERATE", "GENERATE_PUZZLES"):
                params = intent_data.get("params", {})
                rid = str(params.get("rule_id", ""))
                if rid:
                    intents.append(ParsedIntent("GENERATE", {
                        "rule_id": rid,
                        "count": params.get("count", 5),
                        "bucket": params.get("difficulty", "medium"),
                    }))
                elif params.get("domain"):
                    rids = DOMAIN_MAP.get(params["domain"], [])
                    for r in rids[:3]:
                        intents.append(ParsedIntent("GENERATE", {
                            "rule_id": r, "count": 1, "bucket": "medium",
                        }))
            elif action in ("VALIDATE",):
                intents.append(ParsedIntent("VALIDATE"))
            elif action in ("EXPORT",):
                intents.append(ParsedIntent("EXPORT", intent_data.get("params", {})))
            elif action in ("STATS",):
                intents.append(ParsedIntent("STATS"))
            elif action in ("HELP",):
                intents.append(ParsedIntent("HELP"))

        return RewriteResult(
            original_query=query,
            intents=intents or [ParsedIntent("HELP")],
            missing_fields=missing,
            clarification_needed=len(missing) > 0,
            clarify_question=clarify,
            raw_llm_output=raw,
        )

    def _extract_rule_ids(self, query: str) -> List[str]:
        """Extract rule IDs from numeric markers AND named aliases.

        1. Numbers after 'rule'/'规则'.
        2. Rule names ("摩天大楼", "数独", "sudoku" ...) via RULE_NAME_ALIASES.
        3. Fallback: bare 1-25 numbers that are not counts.
        """
        q = query.lower()
        found = []

        # 1. Named rule aliases (longest first to avoid partial-match collisions)
        for alias in sorted(RULE_NAME_ALIASES, key=len, reverse=True):
            if alias in q:
                rid = RULE_NAME_ALIASES[alias]
                if rid not in found:
                    found.append(rid)

        # 2. Clauses: 'rule X' or 'rule X and Y' or 'rules X,Y,Z'
        for m in re.finditer(
            r'(?:rules?|规则)\s*((?:1[0-9]|2[0-5]|[1-9])(?:[\s,，、]*(?:and|和|与|及)?[\s,，、]*(?:1[0-9]|2[0-5]|[1-9]))*)',
            q
        ):
            clause = m.group(1)
            nums = re.findall(r'(1[0-9]|2[0-5]|[1-9])', clause)
            for n in nums:
                if n not in found:
                    found.append(n)

        # 3. Fallback: bare 1-25 numbers that are clearly NOT count words
        if not found:
            all_nums = re.findall(r'\b(1[0-9]|2[0-5]|[1-9])\b', q)
            count_words = r'puzzles?|each|个|道|题|条|more|additional'
            for n in sorted(set(all_nums), key=int):
                if re.search(rf'\b{n}\s*(?:{count_words})', q):
                    continue  # skip counts
                found.append(n)

        return sorted(set(found), key=lambda x: (int(x), x))

    def _context_rule_ids(self, session_summary: str) -> List[str]:
        """Extract recent rule ids from SessionManager.describe_context output."""
        if not session_summary:
            return []
        found = []
        for m in re.finditer(r"Rule\s+([0-9]{1,2})", session_summary, flags=re.IGNORECASE):
            rid = m.group(1)
            if 1 <= int(rid) <= 25 and rid not in found:
                found.append(rid)
        return found

    def _is_followup_generate(self, query: str) -> bool:
        return any(w in query for w in [
            "再来", "再出", "继续", "补", "加", "更多", "more", "another",
            "类似", "像刚才", "刚才那种", "上次那种", "同样", "same"
        ])

    def _is_generate_request(self, query: str) -> bool:
        if any(w in query for w in [
            "生成", "出题", "出", "造", "题目", "来点", "弄点", "给我", "来",
            "想做", "想要", "练", "挑战", "generate", "make", "create", "give"
        ]):
            return True
        return _parse_count(query) is not None

    def _extract_domain_rules(self, query: str) -> List[str]:
        rules: List[str] = []
        negative_math = any(w in query for w in ["不要数学", "非数学", "不想数学", "not math"])
        negative_word = any(w in query for w in ["不要字谜", "不要文字", "非文字", "not word"])
        negative_spatial = any(w in query for w in ["不要空间", "非空间", "not spatial"])

        if any(w in query for w in ["网格", "宫格", "grid"]):
            rules.extend(["11", "12", "13", "15", "16", "17", "25"])
        if any(w in query for w in ["推理", "逻辑", "烧脑", "logic"]):
            rules.extend(["7", "15", "17", "25"])
        if any(w in query for w in ["类似数独", "像数独", "like sudoku"]):
            rules.extend(["16", "17", "25"])
        if any(w in query for w in ["类似24", "像24", "like 24"]):
            rules.extend(["9", "11", "12"])
        for domain, rids in DOMAIN_MAP.items():
            if domain in query:
                rules.extend(rids)

        if any(w in query for w in ["文字", "单词", "词语", "语言"]):
            rules.extend(DOMAIN_MAP["word"])
        if any(w in query for w in ["计算", "数字", "算术"]):
            rules.extend(DOMAIN_MAP["math"])
        if "not sudoku" in query or "不要数独" in query:
            rules = [r for r in rules if r != "15"]
        if "not 24" in query or "不要24" in query or "换个形式" in query:
            rules = [r for r in rules if r != "10"]

        deduped = []
        for rid in rules:
            if negative_math and rid in DOMAIN_MAP["math"]:
                continue
            if negative_word and rid in DOMAIN_MAP["word"]:
                continue
            if negative_spatial and rid in DOMAIN_MAP["spatial"]:
                continue
            if rid not in deduped:
                deduped.append(rid)
        return deduped

    def _is_compound(self, query: str) -> bool:
        """Check if query likely has multiple intents."""
        compound_markers = ["然后", "接着", "并且", "还有", "以及", "另外", "同时", "之后"]
        return any(m in query for m in compound_markers)


# ======================================================================
# Factory
# ======================================================================

def create_rewriter(rules: Dict[str, Any], cfg: Dict[str, Any] = None) -> QueryRewriter:
    from .agents import LlmClient
    client = LlmClient(cfg or {})
    return QueryRewriter(rules, client if client.enabled else None)
