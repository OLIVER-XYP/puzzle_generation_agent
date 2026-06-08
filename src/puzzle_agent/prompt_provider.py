"""Versioned prompt loading for self-evolution.

Prompt versions live in data/prompt_registry.json. The provider falls back to
the built-in prompt constants, so old configs and tests keep working.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Optional

from .prompt_builder import GENERATOR_SYSTEM, SOLVER_SYSTEM, REVIEWER_SYSTEM


@dataclass(frozen=True)
class PromptBundle:
    version_id: str
    generator: str
    solver: str
    reviewer: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "generator": self.generator,
            "solver": self.solver,
            "reviewer": self.reviewer,
        }


BUILTIN_BUNDLE = PromptBundle(
    version_id="builtin",
    generator=GENERATOR_SYSTEM,
    solver=SOLVER_SYSTEM,
    reviewer=REVIEWER_SYSTEM,
)


class PromptProvider:
    """Load active prompt versions from config, memory facts, or registry."""

    def __init__(self, cfg: Dict[str, Any], memory: Optional[Any] = None):
        self.cfg = cfg
        self.memory = memory
        self.root = Path(cfg.get("_root", "."))
        self.registry_path = self.root / "data" / "prompt_registry.json"

    def active_version(self) -> str:
        evolution = self.cfg.get("evolution", {}) or {}
        version = evolution.get("active_prompt_version")
        if evolution.get("_explicit_active_prompt_version") and version:
            return str(version)
        if version and str(version) != "builtin":
            return str(version)
        if self.memory is not None:
            try:
                fact = self.memory.get_fact("evolution:active_prompt_version")
                if fact:
                    return str(fact)
            except Exception:
                pass
        active_file = self.root / "data" / "evolution" / "active_prompt_version.txt"
        if active_file.exists():
            try:
                value = active_file.read_text(encoding="utf-8").strip()
                if value:
                    return value
            except Exception:
                pass
        return "builtin"

    def load(self, version_id: Optional[str] = None) -> PromptBundle:
        vid = version_id or self.active_version()
        if vid == "builtin":
            return BUILTIN_BUNDLE

        record = self._find_record(vid)
        if not record:
            return BUILTIN_BUNDLE
        prompts = record.get("prompts") or {}
        return PromptBundle(
            version_id=record.get("version_id", vid),
            generator=prompts.get("generator") or BUILTIN_BUNDLE.generator,
            solver=prompts.get("solver") or BUILTIN_BUNDLE.solver,
            reviewer=prompts.get("reviewer") or BUILTIN_BUNDLE.reviewer,
        )

    def _find_record(self, version_id: str) -> Optional[Dict[str, Any]]:
        if not self.registry_path.exists():
            return None
        try:
            data = json.loads(self.registry_path.read_text(encoding="utf-8-sig"))
        except Exception:
            return None
        for record in data.get("versions", []):
            if record.get("version_id") == version_id:
                return record
        return None


def create_prompt_provider(cfg: Dict[str, Any], memory: Optional[Any] = None) -> PromptProvider:
    return PromptProvider(cfg, memory=memory)
