"""Persistent working memory for inverse-design planning and troubleshooting."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    """Base model that rejects unknown fields to keep memory records stable."""

    model_config = ConfigDict(extra="forbid")


class WorkingMemoryEntry(StrictModel):
    """Single working-memory record used by step2-step5 flows."""

    stage: Literal[
        "step2_doc_retrieval",
        "step3_mcp",
        "step3_config_generation",
        "step4_mcp",
        "step4_validation",
        "step4_rag",
        "step4_review",
        "step5_checkpoint",
        "step5_diagnosis",
        "step6_repair_trial",
        "recovery",
    ]
    key: str = ""
    failure_signature: str = ""
    scenario_fingerprint: str = ""
    summary: str
    evidence_urls: List[str] = Field(default_factory=list)
    issues: List[str] = Field(default_factory=list)
    proposed_fixes: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class InverseDesignWorkingMemory:
    """In-memory + JSONL-backed store for inverse-design traces and recall."""

    def __init__(self, storage_path: str | None = None, *, max_entries: int = 500) -> None:
        self._max_entries = max(10, int(max_entries))
        self._storage_path = Path(storage_path) if storage_path else _default_storage_path()
        self._entries: List[WorkingMemoryEntry] = []
        self._lock = Lock()
        self._loaded = False

    def record(
        self,
        *,
        stage: str,
        summary: str,
        key: str = "",
        failure_signature: str = "",
        scenario_fingerprint: str = "",
        evidence_urls: List[str] | None = None,
        issues: List[str] | None = None,
        proposed_fixes: List[str] | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> WorkingMemoryEntry:
        """Add a memory record and persist it to disk when possible."""

        self._ensure_loaded()
        entry = WorkingMemoryEntry(
            stage=stage,
            key=key.strip(),
            failure_signature=failure_signature.strip(),
            scenario_fingerprint=scenario_fingerprint.strip(),
            summary=summary.strip()[:500],
            evidence_urls=list(evidence_urls or []),
            issues=list(issues or []),
            proposed_fixes=list(proposed_fixes or []),
            metadata=dict(metadata or {}),
        )
        with self._lock:
            self._entries.append(entry)
            if len(self._entries) > self._max_entries:
                self._entries = self._entries[-self._max_entries :]
        self._append_to_disk(entry)
        return entry

    def recall(
        self,
        *,
        query: str,
        stage: str | None = None,
        failure_signature: str | None = None,
        scenario_fingerprint: str | None = None,
        limit: int = 3,
    ) -> List[WorkingMemoryEntry]:
        """Recall the most similar entries by keyword overlap + recency."""

        self._ensure_loaded()
        normalized_query = _normalize_text(query)
        tokens = set(normalized_query.split())

        scored: List[tuple[float, WorkingMemoryEntry]] = []
        for index, entry in enumerate(self._entries):
            if stage and entry.stage != stage:
                continue
            if failure_signature and entry.failure_signature != failure_signature:
                continue
            if scenario_fingerprint and entry.scenario_fingerprint != scenario_fingerprint:
                continue
            searchable = " ".join(
                [
                    entry.key,
                    entry.failure_signature,
                    entry.scenario_fingerprint,
                    entry.summary,
                    " ".join(entry.issues),
                    " ".join(entry.proposed_fixes),
                ]
            )
            searchable_tokens = set(_normalize_text(searchable).split())
            overlap = len(tokens.intersection(searchable_tokens))
            recency_bonus = (index + 1) / max(len(self._entries), 1)
            score = overlap * 10.0 + recency_bonus
            if score <= 0:
                continue
            scored.append((score, entry))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [entry for _, entry in scored[: max(1, int(limit))]]

    def recall_by_failure_signature(
        self,
        failure_signature: str,
        *,
        stage: str | None = None,
        limit: int = 3,
    ) -> List[WorkingMemoryEntry]:
        """Recall latest entries sharing the same failure signature."""

        signature = failure_signature.strip()
        if not signature:
            return []
        return self.recall(
            query=signature,
            stage=stage,
            failure_signature=signature,
            limit=limit,
        )

    def recall_by_scenario_fingerprint(
        self,
        scenario_fingerprint: str,
        *,
        stage: str | None = None,
        limit: int = 3,
        exact: bool = False,
    ) -> List[WorkingMemoryEntry]:
        """Recall entries by exact or similar scenario fingerprint."""

        fingerprint = scenario_fingerprint.strip()
        if not fingerprint:
            return []
        if exact:
            return self.recall(
                query=fingerprint,
                stage=stage,
                scenario_fingerprint=fingerprint,
                limit=limit,
            )

        normalized_tokens = set(_normalize_text(fingerprint).split())
        if not normalized_tokens:
            return []

        self._ensure_loaded()
        scored: List[tuple[float, WorkingMemoryEntry]] = []
        for index, entry in enumerate(self._entries):
            if stage and entry.stage != stage:
                continue
            if not entry.scenario_fingerprint:
                continue
            entry_tokens = set(_normalize_text(entry.scenario_fingerprint).split())
            overlap = len(normalized_tokens.intersection(entry_tokens))
            if overlap <= 0:
                continue
            recency_bonus = (index + 1) / max(len(self._entries), 1)
            score = overlap * 10.0 + recency_bonus
            scored.append((score, entry))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [entry for _, entry in scored[: max(1, int(limit))]]

    def list_entries(self, *, stage: str | None = None, limit: int = 100) -> List[WorkingMemoryEntry]:
        """Return the latest memory entries for debugging or reporting."""

        self._ensure_loaded()
        selected = [entry for entry in self._entries if not stage or entry.stage == stage]
        return selected[-max(1, int(limit)) :]

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            if self._storage_path.exists():
                try:
                    lines = self._storage_path.read_text(encoding="utf-8").splitlines()
                except Exception:
                    lines = []
                for line in lines[-self._max_entries :]:
                    raw = line.strip()
                    if not raw:
                        continue
                    try:
                        payload = json.loads(raw)
                        self._entries.append(WorkingMemoryEntry.model_validate(payload))
                    except Exception:
                        continue
            self._loaded = True

    def _append_to_disk(self, entry: WorkingMemoryEntry) -> None:
        try:
            self._storage_path.parent.mkdir(parents=True, exist_ok=True)
            with self._storage_path.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(entry.model_dump(), ensure_ascii=True))
                fp.write("\n")
        except Exception:
            return


def inverse_design_working_memory_schema() -> Dict[str, Any]:
    """Return schema for memory entries."""

    return WorkingMemoryEntry.model_json_schema()


def get_inverse_design_working_memory() -> InverseDesignWorkingMemory:
    """Get process-level shared working-memory instance."""

    global _WORKING_MEMORY_SINGLETON
    if _WORKING_MEMORY_SINGLETON is None:
        _WORKING_MEMORY_SINGLETON = InverseDesignWorkingMemory()
    return _WORKING_MEMORY_SINGLETON


def record_working_memory(
    *,
    stage: str,
    summary: str,
    key: str = "",
    failure_signature: str = "",
    scenario_fingerprint: str = "",
    evidence_urls: List[str] | None = None,
    issues: List[str] | None = None,
    proposed_fixes: List[str] | None = None,
    metadata: Dict[str, Any] | None = None,
) -> WorkingMemoryEntry:
    """Convenience wrapper to append a shared memory record."""

    return get_inverse_design_working_memory().record(
        stage=stage,
        summary=summary,
        key=key,
        failure_signature=failure_signature,
        scenario_fingerprint=scenario_fingerprint,
        evidence_urls=evidence_urls,
        issues=issues,
        proposed_fixes=proposed_fixes,
        metadata=metadata,
    )


def recall_working_memory(
    *,
    query: str,
    stage: str | None = None,
    failure_signature: str | None = None,
    scenario_fingerprint: str | None = None,
    limit: int = 3,
) -> List[WorkingMemoryEntry]:
    """Convenience wrapper to recall shared memory records."""

    return get_inverse_design_working_memory().recall(
        query=query,
        stage=stage,
        failure_signature=failure_signature,
        scenario_fingerprint=scenario_fingerprint,
        limit=limit,
    )


def recall_working_memory_by_failure_signature(
    *,
    failure_signature: str,
    stage: str | None = None,
    limit: int = 3,
) -> List[WorkingMemoryEntry]:
    """Convenience wrapper to recall memory entries by failure signature."""

    return get_inverse_design_working_memory().recall_by_failure_signature(
        failure_signature,
        stage=stage,
        limit=limit,
    )


def recall_working_memory_by_scenario_fingerprint(
    *,
    scenario_fingerprint: str,
    stage: str | None = None,
    limit: int = 3,
    exact: bool = False,
) -> List[WorkingMemoryEntry]:
    """Convenience wrapper to recall memory entries by Step6 scenario fingerprint."""

    return get_inverse_design_working_memory().recall_by_scenario_fingerprint(
        scenario_fingerprint,
        stage=stage,
        limit=limit,
        exact=exact,
    )


def _default_storage_path() -> Path:
    configured = os.getenv("INVERSE_DESIGN_WORKING_MEMORY_PATH", "").strip()
    if configured:
        return Path(configured)
    return Path("build") / "inverse_design_working_memory.jsonl"


def _normalize_text(text: str) -> str:
    lowered = text.lower()
    cleaned = re.sub(r"[^a-z0-9_\-\s]+", " ", lowered)
    return re.sub(r"\s+", " ", cleaned).strip()


_WORKING_MEMORY_SINGLETON: InverseDesignWorkingMemory | None = None
