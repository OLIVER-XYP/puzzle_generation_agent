"""Two-tier memory for PuzzleAgent.

Short-Term Memory (STM): Redis-compatible in-memory buffer.
  - Active conversation turns, tool cache, session state
  - Configurable TTL (default: session lifetime)
  - Falls back to Python dict if Redis unavailable

Long-Term Memory (LTM): PostgreSQL (production) / SQLite (dev).
  - Generated puzzles, trace logs, user preferences, prompt versions
  - Unified interface: same API regardless of backend
  - Connection pool for concurrent access
"""
from __future__ import annotations
import json
import os
import hashlib
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


# ======================================================================
# Abstract interfaces
# ======================================================================

class ShortTermStore(ABC):
    """Key-value store with TTL for short-term memory."""

    @abstractmethod
    def get(self, key: str) -> Optional[Any]: ...
    @abstractmethod
    def set(self, key: str, value: Any, ttl_seconds: int = 3600): ...
    @abstractmethod
    def delete(self, key: str): ...
    @abstractmethod
    def keys(self, pattern: str = "*") -> List[str]: ...


class LongTermStore(ABC):
    """Relational store for long-term persistence."""

    @abstractmethod
    def execute(self, sql: str, params: tuple = ()) -> Any: ...
    @abstractmethod
    def fetchone(self, sql: str, params: tuple = ()) -> Optional[Dict]: ...
    @abstractmethod
    def fetchall(self, sql: str, params: tuple = ()) -> List[Dict]: ...
    @abstractmethod
    def commit(self): ...
    @abstractmethod
    def close(self): ...


# ======================================================================
# STM Implementations
# ======================================================================

class DictStore(ShortTermStore):
    """In-memory dict with TTL (fallback when Redis unavailable)."""

    def __init__(self, max_size: int = 10000):
        self._data: OrderedDict = OrderedDict()
        self._expiry: Dict[str, float] = {}
        self._max_size = max_size

    def get(self, key: str) -> Optional[Any]:
        self._cleanup()
        if key in self._expiry and time.time() > self._expiry[key]:
            self.delete(key)
            return None
        return self._data.get(key)

    def set(self, key: str, value: Any, ttl_seconds: int = 3600):
        if len(self._data) >= self._max_size:
            oldest = next(iter(self._data))
            del self._data[oldest]
            self._expiry.pop(oldest, None)
        self._data[key] = value
        self._expiry[key] = time.time() + ttl_seconds
        # Move to end (most recently used)
        self._data.move_to_end(key)

    def delete(self, key: str):
        self._data.pop(key, None)
        self._expiry.pop(key, None)

    def keys(self, pattern: str = "*") -> List[str]:
        self._cleanup()
        if pattern == "*":
            return list(self._data.keys())
        import fnmatch
        return [k for k in self._data if fnmatch.fnmatch(k, pattern)]

    def _cleanup(self):
        now = time.time()
        expired = [k for k, t in self._expiry.items() if t < now]
        for k in expired:
            self._data.pop(k, None)
            self._expiry.pop(k, None)

    @property
    def size(self) -> int:
        return len(self._data)


class RedisStore(ShortTermStore):
    """Redis-backed STM (production)."""

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        try:
            import redis
            self._client = redis.from_url(redis_url)
            self._client.ping()
            self._available = True
        except Exception:
            self._available = False
            self._fallback = DictStore()

    @property
    def available(self) -> bool:
        return self._available

    def get(self, key: str) -> Optional[Any]:
        if not self._available:
            return self._fallback.get(key)
        val = self._client.get(key)
        return json.loads(val) if val else None

    def set(self, key: str, value: Any, ttl_seconds: int = 3600):
        if not self._available:
            self._fallback.set(key, value, ttl_seconds)
            return
        self._client.setex(key, ttl_seconds, json.dumps(value, ensure_ascii=False))

    def delete(self, key: str):
        if self._available:
            self._client.delete(key)
        self._fallback.delete(key)

    def keys(self, pattern: str = "*") -> List[str]:
        if self._available:
            return [k.decode() for k in self._client.keys(pattern)]
        return self._fallback.keys(pattern)


# ======================================================================
# LTM Implementations
# ======================================================================

class PostgresStore(LongTermStore):
    """PostgreSQL backend for production (connection pooling)."""

    def __init__(self, conn_string: str = None):
        conn_string = conn_string or os.environ.get(
            "DATABASE_URL",
            "postgresql://localhost:5432/puzzle_agent"
        )
        try:
            import psycopg2
            import psycopg2.pool
            self._pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=2, maxconn=10, dsn=conn_string
            )
            self._available = True
            self._init_schema_pg()
        except ImportError:
            print("[memory] psycopg2 not installed, falling back to SQLite")
            self._available = False
            self._fallback = None
        except Exception as e:
            print(f"[memory] PostgreSQL unavailable ({e}), falling back to SQLite")
            self._available = False
            self._fallback = None

    def _init_schema_pg(self):
        self.execute("""
            CREATE TABLE IF NOT EXISTS puzzles (
                id VARCHAR(64) PRIMARY KEY,
                rule_id VARCHAR(8) NOT NULL,
                title VARCHAR(256),
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                reviewer_score REAL DEFAULT 0,
                created_at TIMESTAMP DEFAULT NOW(),
                metadata JSONB DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_pg_puzzles_rule ON puzzles(rule_id);
            CREATE INDEX IF NOT EXISTS idx_pg_puzzles_created ON puzzles(created_at);

            CREATE TABLE IF NOT EXISTS sessions (
                id VARCHAR(64) PRIMARY KEY,
                started_at TIMESTAMP DEFAULT NOW(),
                ended_at TIMESTAMP,
                total_generated INTEGER DEFAULT 0,
                notes TEXT
            );

            CREATE TABLE IF NOT EXISTS generation_log (
                id SERIAL PRIMARY KEY,
                session_id VARCHAR(64),
                rule_id VARCHAR(8),
                agent_role VARCHAR(32),
                stage VARCHAR(64),
                passed BOOLEAN DEFAULT FALSE,
                errors JSONB DEFAULT '[]',
                latency_ms INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT NOW()
            );
            CREATE INDEX IF NOT EXISTS idx_pg_log_session ON generation_log(session_id);
            CREATE INDEX IF NOT EXISTS idx_pg_log_rule ON generation_log(rule_id);

            CREATE TABLE IF NOT EXISTS prompt_versions (
                version_id VARCHAR(64) PRIMARY KEY,
                label VARCHAR(256),
                prompts JSONB,
                parent VARCHAR(64),
                pass_rate REAL DEFAULT 0,
                created_at TIMESTAMP DEFAULT NOW()
            );

            CREATE TABLE IF NOT EXISTS facts (
                id SERIAL PRIMARY KEY,
                key VARCHAR(256) UNIQUE NOT NULL,
                value JSONB,
                updated_at TIMESTAMP DEFAULT NOW()
            );
        """)
        self.commit()

    def _conn(self):
        if not self._available:
            raise RuntimeError("PostgreSQL unavailable")
        return self._pool.getconn()

    def _return(self, conn):
        if self._available:
            self._pool.putconn(conn)

    def execute(self, sql: str, params: tuple = ()):
        conn = self._conn()
        try:
            cur = conn.cursor()
            sql_pg = sql.replace("?", "%s")
            cur.execute(sql_pg, params)
            return cur
        finally:
            self._return(conn)

    def fetchone(self, sql: str, params: tuple = ()) -> Optional[Dict]:
        conn = self._conn()
        try:
            cur = conn.cursor()
            cur.execute(sql.replace("?", "%s"), params)
            row = cur.fetchone()
            if row:
                cols = [d[0] for d in cur.description]
                return dict(zip(cols, row))
            return None
        finally:
            self._return(conn)

    def fetchall(self, sql: str, params: tuple = ()) -> List[Dict]:
        conn = self._conn()
        try:
            cur = conn.cursor()
            cur.execute(sql.replace("?", "%s"), params)
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r)) for r in rows]
        finally:
            self._return(conn)

    def commit(self):
        if self._available:
            conn = self._conn()
            try:
                conn.commit()
            finally:
                self._return(conn)

    def close(self):
        if self._available:
            self._pool.closeall()

    def get_fallback(self) -> LongTermStore:
        if self._fallback is None:
            self._fallback = SQLiteStore()
        return self._fallback


class SQLiteStore(LongTermStore):
    """SQLite backend for development."""

    def __init__(self, db_path: str = "data/memory.db"):
        import sqlite3
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS puzzles (
                id TEXT PRIMARY KEY, rule_id TEXT NOT NULL, title TEXT,
                question TEXT NOT NULL, answer TEXT NOT NULL,
                reviewer_score REAL DEFAULT 0,
                created_at TEXT DEFAULT (datetime('now')),
                metadata TEXT DEFAULT '{}'
            );
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                started_at TEXT DEFAULT (datetime('now')),
                ended_at TEXT, total_generated INTEGER DEFAULT 0, notes TEXT
            );
            CREATE TABLE IF NOT EXISTS generation_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT,
                rule_id TEXT, agent_role TEXT, stage TEXT,
                passed INTEGER DEFAULT 0, errors TEXT DEFAULT '[]',
                latency_ms INTEGER DEFAULT 0,
                created_at TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS prompt_versions (
                version_id TEXT PRIMARY KEY, label TEXT,
                prompts TEXT, parent TEXT, pass_rate REAL DEFAULT 0,
                created_at TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT UNIQUE, value TEXT,
                updated_at TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_sql_puzzles_rule ON puzzles(rule_id);
            CREATE INDEX IF NOT EXISTS idx_sql_log_session ON generation_log(session_id);
        """)
        self._conn.commit()

    def execute(self, sql: str, params: tuple = ()):
        return self._conn.execute(sql, params)

    def fetchone(self, sql: str, params: tuple = ()) -> Optional[Dict]:
        cur = self._conn.execute(sql, params)
        row = cur.fetchone()
        return dict(row) if row else None

    def fetchall(self, sql: str, params: tuple = ()) -> List[Dict]:
        cur = self._conn.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]

    def commit(self):
        self._conn.commit()

    def close(self):
        self._conn.close()


# ======================================================================
# Combined Memory Manager
# ======================================================================

class MemoryManager:
    """Two-tier memory: STM (Redis/Dict) + LTM (PostgreSQL/SQLite).

    Usage:
        # Production
        mm = MemoryManager(
            stm_backend="redis", redis_url="redis://...",
            ltm_backend="postgres", db_url="postgresql://..."
        )
        # Dev
        mm = MemoryManager(stm_backend="dict", ltm_backend="sqlite")
    """

    def __init__(
        self,
        stm_backend: str = "dict",       # "dict" | "redis"
        ltm_backend: str = "sqlite",      # "sqlite" | "postgres"
        redis_url: str = "",
        db_url: str = "",
        db_path: str = "data/memory.db",
    ):
        # STM
        if stm_backend == "redis":
            url = redis_url or os.environ.get("REDIS_URL", "redis://localhost:6379/0")
            self.stm_store = RedisStore(url)
        else:
            self.stm_store = DictStore()

        # LTM
        if ltm_backend == "postgres":
            url = db_url or os.environ.get("DATABASE_URL", "")
            self.ltm_store: LongTermStore = PostgresStore(url)
            if not getattr(self.ltm_store, '_available', False):
                print("[memory] Falling back to SQLite")
                self.ltm_store = self.ltm_store.get_fallback()
        else:
            self.ltm_store = SQLiteStore(db_path)

        # Conversation buffer
        self.conversation: List[Dict] = []
        self.current_session_id: Optional[str] = None

    # ---- Session ----

    def start_session(self) -> str:
        sid = datetime.now().strftime("sess-%Y%m%d-%H%M%S-%f")
        self.ltm_store.execute(
            "INSERT INTO sessions (id) VALUES (?)", (sid,)
        )
        self.ltm_store.commit()
        self.current_session_id = sid
        self.stm_store.set("current_session", sid, ttl_seconds=86400)
        self.conversation.clear()
        return sid

    def end_session(self, notes: str = ""):
        if self.current_session_id:
            self.ltm_store.execute(
                "UPDATE sessions SET ended_at=datetime('now'), notes=? WHERE id=?",
                (notes, self.current_session_id),
            )
            self.ltm_store.commit()

    # ---- STM operations ----

    def remember_recent(self, key: str, value: Any, ttl: int = 3600):
        self.stm_store.set(key, value, ttl)

    def recall_recent(self, key: str) -> Optional[Any]:
        return self.stm_store.get(key)

    def forget_recent(self, key: str):
        self.stm_store.delete(key)

    # ---- LTM operations ----

    def save_puzzle(self, puzzle: Dict[str, Any]) -> str:
        pid = puzzle.get("idx", "") or hashlib.md5(
            (puzzle.get("question","") + puzzle.get("answer","")).encode()
        ).hexdigest()[:12]
        self.ltm_store.execute(
            "INSERT OR REPLACE INTO puzzles (id, rule_id, title, question, answer, reviewer_score, metadata) "
            "VALUES (?,?,?,?,?,?,?)",
            (pid, puzzle.get("rule_id",""), puzzle.get("title",""),
             puzzle.get("question",""), puzzle.get("answer",""),
             puzzle.get("metadata", {}).get("reviewer_score", 0),
             json.dumps(puzzle.get("metadata", {}))),
        )
        self.ltm_store.commit()
        return pid

    def get_puzzles(self, rule_id: str = None, limit: int = 100) -> List[Dict]:
        if rule_id:
            return self.ltm_store.fetchall(
                "SELECT * FROM puzzles WHERE rule_id=? ORDER BY created_at DESC LIMIT ?",
                (rule_id, limit),
            )
        return self.ltm_store.fetchall(
            "SELECT * FROM puzzles ORDER BY created_at DESC LIMIT ?", (limit,),
        )

    def get_puzzle_count(self, rule_id: str = None) -> int:
        if rule_id:
            row = self.ltm_store.fetchone("SELECT COUNT(*) as c FROM puzzles WHERE rule_id=?", (rule_id,))
        else:
            row = self.ltm_store.fetchone("SELECT COUNT(*) as c FROM puzzles", ())
        return row["c"] if row else 0

    def log_generation(self, rule_id: str, agent_role: str, stage: str,
                       passed: bool, errors: List[str] = None, latency_ms: int = 0):
        self.ltm_store.execute(
            "INSERT INTO generation_log (session_id, rule_id, agent_role, stage, passed, errors, latency_ms) "
            "VALUES (?,?,?,?,?,?,?)",
            (self.current_session_id, rule_id, agent_role, stage,
             1 if passed else 0, json.dumps(errors or []), latency_ms),
        )
        self.ltm_store.commit()

    def get_rule_stats(self, rule_id: str) -> Dict[str, Any]:
        row = self.ltm_store.fetchone(
            "SELECT COUNT(*) as total, SUM(passed) as passed FROM generation_log WHERE rule_id=?",
            (rule_id,),
        )
        total = (row["total"] or 0) if row else 0
        passed = (row["passed"] or 0) if row else 0
        return {"total": total, "passed": passed, "rate": passed / max(total, 1)}

    def set_fact(self, key: str, value: Any):
        self.ltm_store.execute(
            "INSERT OR REPLACE INTO facts (key, value, updated_at) VALUES (?,?,datetime('now'))",
            (key, json.dumps(value, ensure_ascii=False)),
        )
        self.ltm_store.commit()

    def get_fact(self, key: str) -> Optional[Any]:
        row = self.ltm_store.fetchone("SELECT value FROM facts WHERE key=?", (key,))
        return json.loads(row["value"]) if row else None

    def save_puzzles_batch(self, records: List[Dict[str, Any]]) -> int:
        """Batch insert puzzles. Returns count."""
        for puzzle in records:
            self.save_puzzle(puzzle)
        return len(records)

    def log_generation_batch(self, entries: List[Dict[str, Any]]):
        """Batch insert generation log entries."""
        for e in entries:
            self.log_generation(
                e["rule_id"], e.get("agent_role", "verification"),
                e.get("stage", "verify"), e.get("passed", False),
                e.get("errors", []), e.get("latency_ms", 0),
            )

    def update_session_count(self, total: int):
        """Update total_generated for the current session."""
        if self.current_session_id:
            self.ltm_store.execute(
                "UPDATE sessions SET total_generated=? WHERE id=?",
                (total, self.current_session_id),
            )
            self.ltm_store.commit()

    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics for the current session."""
        if not self.current_session_id:
            return {"error": "no active session"}
        sid = self.current_session_id
        row = self.ltm_store.fetchone(
            "SELECT COUNT(*) as c FROM generation_log WHERE session_id=?", (sid,)
        )
        total_logs = row["c"] if row else 0
        passed_row = self.ltm_store.fetchone(
            "SELECT COUNT(*) as c FROM generation_log WHERE session_id=? AND passed=1", (sid,)
        )
        total_passed = passed_row["c"] if passed_row else 0
        by_rule_rows = self.ltm_store.fetchall(
            "SELECT rule_id, COUNT(*) as c, SUM(passed) as p "
            "FROM generation_log WHERE session_id=? GROUP BY rule_id ORDER BY c DESC",
            (sid,)
        )
        return {
            "session_id": sid,
            "total_logs": total_logs,
            "total_passed": total_passed,
            "pass_rate": round(total_passed / max(total_logs, 1), 3),
            "by_rule": {r["rule_id"]: {"total": r["c"], "passed": r["p"] or 0}
                       for r in by_rule_rows},
        }

    def get_user_context(self) -> str:
        puzzles = self.get_puzzles(limit=200)
        by_rule: Dict[str, int] = {}
        for p in puzzles:
            by_rule[p["rule_id"]] = by_rule.get(p["rule_id"], 0) + 1
        lines = []
        if by_rule:
            top = sorted(by_rule.items(), key=lambda x: -x[1])[:5]
            lines.append("Generated: " + ", ".join(f"R{r}={c}" for r, c in top))
        return "\n".join(lines) if lines else "No history."

    def close(self):
        self.ltm_store.close()


# ======================================================================
# Factory
# ======================================================================

def create_memory(
    use_redis: bool = False,
    use_postgres: bool = False,
    redis_url: str = "",
    db_url: str = "",
    db_path: str = "data/memory.db",
) -> MemoryManager:
    return MemoryManager(
        stm_backend="redis" if use_redis else "dict",
        ltm_backend="postgres" if use_postgres else "sqlite",
        redis_url=redis_url,
        db_url=db_url,
        db_path=db_path,
    )
