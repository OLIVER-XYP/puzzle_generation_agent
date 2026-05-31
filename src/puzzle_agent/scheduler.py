"""Async scheduler + rate limiter + parallel execution (P2).

Supports:
  - ThreadPoolExecutor for parallel rule generation
  - Rate limiter (requests per minute)
  - Exponential backoff retry
  - Progress tracking with tqdm (optional)
"""
from __future__ import annotations
import time
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


# ======================================================================
# Rate Limiter
# ======================================================================

@dataclass
class RateLimiter:
    """Token-bucket rate limiter for API calls."""
    max_rpm: int = 30                # max requests per minute
    max_retries: int = 3             # max retry attempts on rate limit
    backoff_base: float = 1.5        # exponential backoff multiplier

    _window: deque = field(default_factory=deque)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def acquire(self) -> float:
        """Wait until a request slot is available. Returns wait time in seconds."""
        with self._lock:
            now = time.time()
            # Remove timestamps older than 60s
            while self._window and self._window[0] < now - 60:
                self._window.popleft()

            if len(self._window) < self.max_rpm:
                self._window.append(now)
                return 0.0

            # Calculate wait time
            wait = self._window[0] + 60 - now + 0.1
            time.sleep(wait)
            self._window.append(time.time())
            return wait

    def call_with_retry(self, fn: Callable, *args, **kwargs) -> Any:
        """Call fn with rate limiting and exponential backoff retry."""
        last_error = None
        for attempt in range(self.max_retries + 1):
            wait = self.acquire()
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                if "rate" in error_str or "429" in error_str:
                    # Rate limited — backoff
                    delay = self.backoff_base ** attempt
                    time.sleep(delay)
                    continue
                elif "timeout" in error_str or "503" in error_str:
                    # Server error — backoff
                    delay = self.backoff_base ** (attempt + 1)
                    time.sleep(delay)
                    continue
                else:
                    raise
        raise last_error or RuntimeError("Max retries exceeded")


# ======================================================================
# Progress Tracker
# ======================================================================

@dataclass
class JobProgress:
    """Track progress of a generation job."""
    rule_id: str
    rule_title: str
    total: int
    completed: int = 0
    passed: int = 0
    failed: int = 0
    start_time: float = 0.0
    errors: List[str] = field(default_factory=list)

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time if self.start_time else 0

    @property
    def rate(self) -> float:
        return self.completed / max(self.elapsed, 0.001)

    def update(self, ok: bool, error: str = ""):
        self.completed += 1
        if ok:
            self.passed += 1
        else:
            self.failed += 1
            if error:
                self.errors.append(error)

    def summary(self) -> str:
        return (f"[{self.rule_title}] {self.completed}/{self.total} "
                f"({self.passed}P/{self.failed}F) in {self.elapsed:.1f}s")


# ======================================================================
# Parallel Executor
# ======================================================================

class ParallelExecutor:
    """Execute generation jobs in parallel across rules."""

    def __init__(self, max_workers: int = 5, rate_limiter: RateLimiter = None):
        self.max_workers = max_workers
        self.rate_limiter = rate_limiter or RateLimiter(max_rpm=30)

    def map(self, fn: Callable, items: List[Tuple], on_progress: Callable = None) -> List[Any]:
        """Execute fn(item) for each item in parallel with rate limiting.

        Args:
            fn: Function to call for each item. Signature: fn(item) -> result.
            items: List of items to process.
            on_progress: Optional callback(completed, total) for progress updates.

        Returns:
            List of results in the same order as items.
        """
        results: List[Any] = [None] * len(items)
        completed = 0

        def _rate_limited_call(idx, item):
            return idx, self.rate_limiter.call_with_retry(fn, *item)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(_rate_limited_call, i, item): i
                      for i, item in enumerate(items)}

            for future in as_completed(futures):
                try:
                    idx, result = future.result()
                    results[idx] = result
                    completed += 1
                    if on_progress:
                        on_progress(completed, len(items))
                except Exception as e:
                    idx = futures[future]
                    results[idx] = {"error": str(e)}
                    completed += 1

        return results

    def generate_multi_rule(
        self,
        generate_fn: Callable,
        jobs: List[Dict[str, Any]],
        progress: bool = True,
    ) -> Dict[str, Any]:
        """Parallel generation across multiple rules.

        Args:
            generate_fn: function(rule_id, count) -> list of results
            jobs: [{"rule_id": "...", "title": "...", "count": N}, ...]
            progress: whether to print progress

        Returns:
            {rule_id: {"results": [...], "passed": N, "failed": N}}
        """
        trackers = {}
        for j in jobs:
            rid = j["rule_id"]
            trackers[rid] = JobProgress(
                rule_id=rid, rule_title=j.get("title", ""), total=j["count"],
                start_time=time.time(),
            )

        items = []
        for j in jobs:
            rid = j["rule_id"]
            count = j["count"]
            for i in range(count):
                items.append((rid, i, generate_fn))

        total_items = len(items)

        def _gen_single(rid, idx, fn):
            try:
                result = fn(rid, 1)  # generate 1 at a time
                ok = bool(result and not result.get("error"))
                trackers[rid].update(ok, result.get("error", ""))
                return result
            except Exception as e:
                trackers[rid].update(False, str(e))
                return {"error": str(e)}

        if progress:
            print(f"[scheduler] Starting {total_items} items across {len(jobs)} rules "
                  f"(workers={self.max_workers})")

        t0 = time.time()
        results = self.map(_gen_single, items,
                          on_progress=lambda c, t: progress and
                          print(f"\r  {c}/{t} completed ({c/max(time.time()-t0,0.1):.1f}/s)", end=""))
        if progress:
            print()

        # Group results
        per_rule: Dict[str, Any] = {}
        idx = 0
        for j in jobs:
            rid = j["rule_id"]
            rule_results = results[idx:idx + j["count"]]
            per_rule[rid] = {
                "results": rule_results,
                "passed": sum(1 for r in rule_results if r and not r.get("error")),
                "failed": sum(1 for r in rule_results if not r or r.get("error")),
            }
            idx += j["count"]

        if progress:
            elapsed = time.time() - t0
            print(f"[scheduler] Done: {total_items} items in {elapsed:.1f}s "
                  f"({total_items/max(elapsed,0.1):.1f}/s)")

        return per_rule


# ======================================================================
# Factory
# ======================================================================

def create_executor(max_workers: int = 5, max_rpm: int = 30) -> ParallelExecutor:
    limiter = RateLimiter(max_rpm=max_rpm)
    return ParallelExecutor(max_workers=max_workers, rate_limiter=limiter)
