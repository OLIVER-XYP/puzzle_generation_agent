"""User behavior memory for cross-session personalization.

The implementation intentionally stores the profile in MemoryManager.facts so
the feature works with the existing SQLite/PostgreSQL abstraction without a
schema migration.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import re
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_USER_ID = "default"

RULE_CATEGORIES: Dict[str, str] = {
    **{str(i): "word" for i in [1, 2, 3, 4, 5, 6, 7, 8, 24]},
    **{str(i): "math" for i in [9, 10, 11, 12, 13, 14, 15, 16, 17, 25]},
    **{str(i): "spatial" for i in [18, 19, 20, 21, 22, 23]},
}

RULE_ALIASES: Dict[str, str] = {
    "anagram": "4",
    "变位词": "4",
    "字母重排": "4",
    "cryptarithm": "5",
    "密码数学": "5",
    "word ladder": "6",
    "单词阶梯": "6",
    "24点": "10",
    "24 points": "10",
    "sudoku": "15",
    "数独": "15",
    "calcudoku": "16",
    "kenken": "16",
    "futoshiki": "17",
    "不等式": "17",
    "skyscraper": "25",
    "skyscrapers": "25",
    "摩天大楼": "25",
}


@dataclass
class UserProfile:
    user_id: str = DEFAULT_USER_ID
    preferred_rules: Dict[str, int] = field(default_factory=dict)
    preferred_tags: Dict[str, int] = field(default_factory=dict)
    difficulty_preference: str = "auto"
    feedback_history: List[Dict[str, Any]] = field(default_factory=list)
    generation_stats: Dict[str, Any] = field(default_factory=lambda: {
        "accepted_by_rule": {},
        "rejected_by_rule": {},
        "last_requested_rules": [],
    })
    updated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "preferred_rules": dict(self.preferred_rules),
            "preferred_tags": dict(self.preferred_tags),
            "difficulty_preference": self.difficulty_preference,
            "feedback_history": list(self.feedback_history),
            "generation_stats": dict(self.generation_stats),
            "updated_at": self.updated_at or _now(),
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]], user_id: str = DEFAULT_USER_ID) -> "UserProfile":
        if not data:
            return cls(user_id=user_id, updated_at=_now())
        stats = data.get("generation_stats") or {}
        stats.setdefault("accepted_by_rule", {})
        stats.setdefault("rejected_by_rule", {})
        stats.setdefault("last_requested_rules", [])
        return cls(
            user_id=str(data.get("user_id") or user_id),
            preferred_rules=_int_map(data.get("preferred_rules") or {}),
            preferred_tags=_int_map(data.get("preferred_tags") or {}),
            difficulty_preference=data.get("difficulty_preference") or "auto",
            feedback_history=list(data.get("feedback_history") or []),
            generation_stats=stats,
            updated_at=data.get("updated_at") or _now(),
        )


class UserMemoryManager:
    """Small behavior-memory layer backed by MemoryManager.facts."""

    def __init__(self, memory: Any, cfg: Optional[Dict[str, Any]] = None,
                 rule_index: Optional[Dict[str, Dict[str, Any]]] = None):
        self.memory = memory
        self.cfg = cfg or {}
        self.enabled = bool(self.cfg.get("enabled", True))
        self.default_user_id = str(self.cfg.get("default_user_id") or DEFAULT_USER_ID)
        self.max_context_chars = int(self.cfg.get("max_context_chars") or 1200)
        self.rule_index = rule_index or {}

    def fact_key(self, user_id: Optional[str] = None) -> str:
        return f"user_profile:{user_id or self.default_user_id}"

    def get_profile(self, user_id: Optional[str] = None) -> UserProfile:
        uid = user_id or self.default_user_id
        if not self.enabled:
            return UserProfile(user_id=uid, updated_at=_now())
        return UserProfile.from_dict(self.memory.get_fact(self.fact_key(uid)), uid)

    def save_profile(self, profile: UserProfile) -> UserProfile:
        profile.updated_at = _now()
        if self.enabled:
            self.memory.set_fact(self.fact_key(profile.user_id), profile.to_dict())
        return profile

    def ingest_text(self, text: str, user_id: Optional[str] = None) -> UserProfile:
        profile = self.get_profile(user_id)
        update_profile_from_text(profile, text, self.rule_index)
        return self.save_profile(profile)

    def record_feedback(self, feedback: str, user_id: Optional[str] = None,
                        rule_id: str = "", sentiment: str = "") -> UserProfile:
        profile = self.get_profile(user_id)
        if rule_id:
            _bump(profile.preferred_rules, str(rule_id), 2 if sentiment != "negative" else -1)
            tag = RULE_CATEGORIES.get(str(rule_id))
            if tag:
                _bump(profile.preferred_tags, tag, 1 if sentiment != "negative" else -1)
        update_profile_from_text(profile, feedback, self.rule_index)
        profile.feedback_history.append({
            "text": feedback,
            "rule_id": str(rule_id or ""),
            "sentiment": sentiment or infer_sentiment(feedback),
            "created_at": _now(),
        })
        profile.feedback_history = profile.feedback_history[-50:]
        return self.save_profile(profile)

    def record_requested_rules(self, user_id: Optional[str], rules: List[str]) -> UserProfile:
        profile = self.get_profile(user_id)
        clean = [str(r) for r in rules if r]
        profile.generation_stats["last_requested_rules"] = clean[-10:]
        for rid in clean:
            _bump(profile.preferred_rules, rid, 1)
            tag = RULE_CATEGORIES.get(rid)
            if tag:
                _bump(profile.preferred_tags, tag, 1)
        return self.save_profile(profile)

    def record_acceptance(self, user_id: Optional[str], rule_id: str, tag: str = "") -> UserProfile:
        profile = self.get_profile(user_id)
        stats = profile.generation_stats.setdefault("accepted_by_rule", {})
        _bump(stats, str(rule_id), 1)
        _bump(profile.preferred_rules, str(rule_id), 1)
        tag_key = normalize_tag(tag) or RULE_CATEGORIES.get(str(rule_id))
        if tag_key:
            _bump(profile.preferred_tags, tag_key, 1)
        return self.save_profile(profile)

    def record_rejection(self, user_id: Optional[str], rule_id: str,
                         reasons: List[str]) -> UserProfile:
        profile = self.get_profile(user_id)
        stats = profile.generation_stats.setdefault("rejected_by_rule", {})
        _bump(stats, str(rule_id), 1)
        profile.feedback_history.append({
            "type": "generation_rejection",
            "rule_id": str(rule_id),
            "reasons": list(reasons),
            "created_at": _now(),
        })
        profile.feedback_history = profile.feedback_history[-50:]
        return self.save_profile(profile)

    def context_text(self, user_id: Optional[str] = None) -> str:
        return format_user_context(self.get_profile(user_id), self.max_context_chars)

    def recommend_rules(self, user_id: Optional[str] = None, limit: int = 3) -> List[Dict[str, Any]]:
        profile = self.get_profile(user_id)
        return recommend_rules(profile, self.rule_index, limit=limit)


def get_user_profile(memory: Any, user_id: str = DEFAULT_USER_ID) -> UserProfile:
    return UserMemoryManager(memory, {"enabled": True, "default_user_id": user_id}).get_profile(user_id)


def set_user_profile(memory: Any, profile: UserProfile) -> UserProfile:
    return UserMemoryManager(memory, {"enabled": True, "default_user_id": profile.user_id}).save_profile(profile)


def update_profile_from_text(profile: UserProfile, text: str,
                             rule_index: Optional[Dict[str, Dict[str, Any]]] = None) -> UserProfile:
    lowered = text.lower()
    if any(w in text for w in ["难一点", "困难", "复杂", "hard"]) or "hard" in lowered:
        profile.difficulty_preference = "hard"
    elif any(w in text for w in ["简单", "容易", "短一点", "easy"]) or "easy" in lowered:
        profile.difficulty_preference = "easy"
    elif any(w in text for w in ["中等", "普通", "medium"]) or "medium" in lowered:
        profile.difficulty_preference = "medium"

    tag_hits = []
    if any(w in text for w in ["数学", "计算", "24点"]) or "math" in lowered:
        tag_hits.append("math")
    if any(w in text for w in ["字谜", "单词", "词", "文字"]) or "word" in lowered:
        tag_hits.append("word")
    if any(w in text for w in ["空间", "网格", "迷宫"]) or "spatial" in lowered:
        tag_hits.append("spatial")
    if any(w in text for w in ["逻辑", "推理", "logic"]) or "logic" in lowered:
        tag_hits.append("logic")
    for tag in tag_hits:
        _bump(profile.preferred_tags, tag, 2)

    for rid in extract_rule_ids(text):
        _bump(profile.preferred_rules, rid, 3)
        tag = RULE_CATEGORIES.get(rid)
        if tag:
            _bump(profile.preferred_tags, tag, 1)

    for alias, rid in RULE_ALIASES.items():
        if alias.lower() in lowered or alias in text:
            _bump(profile.preferred_rules, rid, 3)
            tag = RULE_CATEGORIES.get(rid)
            if tag:
                _bump(profile.preferred_tags, tag, 1)

    # If the project has richer tags, let the free text reinforce them too.
    for rid, meta in (rule_index or {}).items():
        title = str(meta.get("title", ""))
        tag = normalize_tag(str(meta.get("tag", "")))
        if title and title in text:
            _bump(profile.preferred_rules, str(rid), 2)
        if tag and tag in lowered:
            _bump(profile.preferred_tags, tag, 1)

    profile.updated_at = _now()
    return profile


def extract_rule_ids(text: str) -> List[str]:
    ids = []
    for match in re.finditer(r"(?:rule|规则)\s*([0-9]{1,2})", text, flags=re.IGNORECASE):
        rid = match.group(1)
        if 1 <= int(rid) <= 25:
            ids.append(rid)
    return ids


def infer_sentiment(text: str) -> str:
    if any(w in text for w in ["不喜欢", "不要", "太难", "太简单", "bad", "worse"]):
        return "negative"
    if any(w in text for w in ["喜欢", "不错", "更想", "prefer", "good", "great"]):
        return "positive"
    return "neutral"


def normalize_tag(tag: str) -> str:
    t = tag.strip().lower()
    if not t:
        return ""
    if any(w in t for w in ["math", "number", "sudoku", "logic"]):
        return "math" if "logic" not in t else "logic"
    if any(w in t for w in ["word", "anagram", "letter"]):
        return "word"
    if any(w in t for w in ["spatial", "grid", "maze"]):
        return "spatial"
    return t.split()[0][:24]


def format_user_context(profile: UserProfile, max_chars: int = 1200) -> str:
    parts = [f"user_id={profile.user_id}"]
    if profile.difficulty_preference != "auto":
        parts.append(f"difficulty_preference={profile.difficulty_preference}")
    top_rules = _top(profile.preferred_rules)
    if top_rules:
        parts.append("preferred_rules=" + ", ".join(f"R{k}:{v}" for k, v in top_rules))
    top_tags = _top(profile.preferred_tags)
    if top_tags:
        parts.append("preferred_tags=" + ", ".join(f"{k}:{v}" for k, v in top_tags))
    stats = profile.generation_stats or {}
    last = stats.get("last_requested_rules") or []
    if last:
        parts.append("last_requested_rules=" + ",".join(last[-5:]))
    accepted = _top(_int_map(stats.get("accepted_by_rule") or {}), 3)
    if accepted:
        parts.append("accepted_by_rule=" + ", ".join(f"R{k}:{v}" for k, v in accepted))
    rejected = _top(_int_map(stats.get("rejected_by_rule") or {}), 3)
    if rejected:
        parts.append("rejected_by_rule=" + ", ".join(f"R{k}:{v}" for k, v in rejected))
    if len(parts) == 1:
        parts.append("No durable user preferences yet.")
    text = "\n".join(parts)
    if len(text) > max_chars:
        text = text[:max_chars - 3] + "..."
    return text


def recommend_rules(profile: UserProfile, rule_index: Dict[str, Dict[str, Any]],
                    limit: int = 3) -> List[Dict[str, Any]]:
    scores: Dict[str, float] = {}
    for rid, value in profile.preferred_rules.items():
        scores[str(rid)] = scores.get(str(rid), 0.0) + float(value) * 3.0
    for rid, tag in RULE_CATEGORIES.items():
        scores[rid] = scores.get(rid, 0.0) + float(profile.preferred_tags.get(tag, 0))

    accepted = _int_map((profile.generation_stats or {}).get("accepted_by_rule") or {})
    rejected = _int_map((profile.generation_stats or {}).get("rejected_by_rule") or {})
    for rid, value in accepted.items():
        scores[rid] = scores.get(rid, 0.0) + value * 1.5
    for rid, value in rejected.items():
        scores[rid] = scores.get(rid, 0.0) - value * 0.75

    if not scores:
        for rid in ["10", "25", "15"]:
            scores[rid] = 1.0

    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], int(kv[0]) if kv[0].isdigit() else 999))
    result = []
    for rid, score in ranked[:limit]:
        meta = rule_index.get(rid, {})
        result.append({
            "rule_id": rid,
            "title": meta.get("title", f"Rule {rid}"),
            "tag": meta.get("tag", RULE_CATEGORIES.get(rid, "")),
            "score": round(score, 3),
            "reason": _recommend_reason(profile, rid),
        })
    return result


def _recommend_reason(profile: UserProfile, rid: str) -> str:
    if rid in profile.preferred_rules:
        return "matches frequently requested rule"
    tag = RULE_CATEGORIES.get(rid, "")
    if tag and profile.preferred_tags.get(tag):
        return f"matches preferred {tag} category"
    return "strong default benchmark rule"


def _top(data: Dict[str, int], limit: int = 5) -> List[Tuple[str, int]]:
    return sorted(_int_map(data).items(), key=lambda kv: (-kv[1], kv[0]))[:limit]


def _int_map(data: Dict[str, Any]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for k, v in data.items():
        try:
            out[str(k)] = int(v)
        except (TypeError, ValueError):
            out[str(k)] = 0
    return out


def _bump(data: Dict[str, Any], key: str, delta: int) -> None:
    data[key] = max(0, int(data.get(key, 0) or 0) + delta)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")
