"""Step4 RAG memory for semantic and hard-constraint guidance."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List

from pydantic import BaseModel, ConfigDict, Field

from PhotonicsAI.tools.inverse_design_config import InverseDesignConfigBundle
from PhotonicsAI.tools.inverse_design_contracts import (
    dedupe_contract_rules,
    objective_flux_key_contract,
    objective_semantic_contract_rules,
)
from PhotonicsAI.tools.inverse_design_doc_context import InverseDesignDocContext


class StrictModel(BaseModel):
    """Base model that rejects unknown fields to keep RAG packets stable."""

    model_config = ConfigDict(extra="forbid")


class Step4HardConstraint(StrictModel):
    """Hard constraint expected by Step4 deterministic validation."""

    code: str
    path: str
    expected: str
    reason: str


class Step5HardPhysicsGate(StrictModel):
    """Deterministic first-iteration physics gate used by Step5 execution.

    V16 NOTE: `input_flux_min`, `required_input_flux_keys`, `output_flux_min`,
    `max_input_normalized_coupling*` are LEGACY ratio-domain gates. For
    direction='-' mode sources they read REFLECTED power, not injection.
    New canonical gate is `min_absolute_ce_w` (Watts; ModeSource normalization).
    Legacy fields kept for v15-and-earlier bundle compatibility; for new
    bundles `min_absolute_ce_w` takes precedence when populated.
    """

    enabled: bool = True
    # V16 canonical gate (absolute, ModeSource 1 W reference)
    min_absolute_ce_w: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum |amps(target_port, exit_dir, target_mode_index)|² required "
            "at first iteration. Replaces the legacy ratio-domain "
            "`input_flux_min` gate. None means the gate is unset."
        ),
    )
    max_reflection_w: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Maximum reflection back into source mode at first iteration. "
            "Computed as |amps(src_port, opposite(src_dir), src_mode)|²."
        ),
    )
    # Legacy ratio-domain gates (DEPRECATED, retained for back-compat)
    input_flux_min: float = Field(default=1e-4, ge=0.0)
    output_flux_min: float = Field(default=1e-4, ge=0.0)
    energy_closure_tolerance: float = Field(default=0.4, ge=0.0, le=2.0)
    field_continuity_min_ratio: float = Field(default=0.05, ge=0.0, le=1.0)
    per_port_transmission_min: float = Field(
        default=0.01, ge=0.0, le=1.0,
        description=(
            "Minimum per-port transmission ratio (port_flux / input_flux). "
            "A wrongly-oriented monitor typically captures <1% of injected power; "
            "this threshold catches such anomalies."
        ),
    )
    output_imbalance_max_ratio: float = Field(
        default=100.0, ge=1.0,
        description=(
            "Maximum allowed ratio between the strongest and weakest output port flux. "
            "For nominally symmetric devices (e.g. 1x2 MMI) extreme imbalance indicates "
            "a monitor orientation mismatch."
        ),
    )
    max_input_normalized_coupling: float = Field(
        default=1.2,
        ge=0.1,
        le=10.0,
        description=(
            "Upper bound for per-case coupling_ratio_to_input. Values far above 1 "
            "usually indicate monitor normalization mismatch or geometry/monitor misalignment."
        ),
    )
    max_input_normalized_coupling_reverse: float = Field(
        default=2.5,
        ge=0.1,
        le=20.0,
        description=(
            "Relaxed upper bound for coupling_ratio_to_input when the case source "
            "propagates in reverse direction (source_direction='-'). Reverse-launch "
            "normalization can be noisier than forward-launch measurements."
        ),
    )
    require_field_artifact: bool = True
    require_mode_expansion_for_demux: bool = True
    required_input_flux_keys: List[str] = Field(
        default_factory=lambda: ["flux_port_o1", "flux_port_i1", "flux_input"]
    )
    required_output_flux_keys: List[str] = Field(
        default_factory=lambda: ["flux_port_o2", "flux_port_o3"]
    )


class Step4ConstraintPacket(StrictModel):
    """RAG packet used by Step4 validator and reviewer prompts."""

    component_type: str = ""
    objective_metric: str = ""
    semantic_rules: List[str] = Field(default_factory=list)
    hard_constraints: List[Step4HardConstraint] = Field(default_factory=list)
    required_monitor_types: List[str] = Field(default_factory=list)
    required_boundary: str = "pml"
    min_steps_per_wvl: int = 10
    hard_physics_gate: Step5HardPhysicsGate = Field(default_factory=Step5HardPhysicsGate)
    evidence_urls: List[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class Step4RAGMemoryEntry(StrictModel):
    """Persisted RAG memory entry for query-based recall."""

    query: str
    packet: Step4ConstraintPacket
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class InverseDesignStep4RAGMemory:
    """JSONL-backed RAG memory for Step4 semantic and hard-constraint packets."""

    def __init__(self, storage_path: str | None = None, *, max_entries: int = 400) -> None:
        self._max_entries = max(10, int(max_entries))
        self._storage_path = Path(storage_path) if storage_path else _default_storage_path()
        self._entries: List[Step4RAGMemoryEntry] = []
        self._lock = Lock()
        self._loaded = False

    def record(self, *, query: str, packet: Step4ConstraintPacket) -> Step4RAGMemoryEntry:
        """Persist one Step4 RAG packet."""

        self._ensure_loaded()
        entry = Step4RAGMemoryEntry(query=query.strip(), packet=packet)
        with self._lock:
            self._entries.append(entry)
            if len(self._entries) > self._max_entries:
                self._entries = self._entries[-self._max_entries :]
        self._append_to_disk(entry)
        return entry

    def recall(self, *, query: str, limit: int = 3) -> List[Step4ConstraintPacket]:
        """Recall the most relevant Step4 packets by keyword overlap + recency."""

        self._ensure_loaded()
        normalized_query = _normalize_text(query)
        tokens = set(normalized_query.split())

        scored: List[tuple[float, Step4RAGMemoryEntry]] = []
        for index, entry in enumerate(self._entries):
            searchable = " ".join(
                [
                    entry.query,
                    entry.packet.component_type,
                    entry.packet.objective_metric,
                    " ".join(entry.packet.semantic_rules),
                    " ".join(item.code for item in entry.packet.hard_constraints),
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
        return [entry.packet for _, entry in scored[: max(1, int(limit))]]

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
                        self._entries.append(Step4RAGMemoryEntry.model_validate(payload))
                    except Exception:
                        continue
            self._loaded = True

    def _append_to_disk(self, entry: Step4RAGMemoryEntry) -> None:
        try:
            self._storage_path.parent.mkdir(parents=True, exist_ok=True)
            with self._storage_path.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(entry.model_dump(), ensure_ascii=True))
                fp.write("\n")
        except Exception:
            return


def build_step4_constraint_packet(
    *,
    doc_context: InverseDesignDocContext | Dict[str, Any] | None = None,
    config_bundle: InverseDesignConfigBundle | Dict[str, Any] | None = None,
) -> Step4ConstraintPacket:
    """Build a Step4 RAG packet from Step2 evidence and Step3 config context."""

    typed_doc: InverseDesignDocContext | None = None
    if isinstance(doc_context, InverseDesignDocContext):
        typed_doc = doc_context
    elif isinstance(doc_context, dict):
        typed_doc = InverseDesignDocContext.model_validate(doc_context)

    typed_bundle: InverseDesignConfigBundle | None = None
    if isinstance(config_bundle, InverseDesignConfigBundle):
        typed_bundle = config_bundle
    elif isinstance(config_bundle, dict):
        typed_bundle = InverseDesignConfigBundle.model_validate(config_bundle)

    component_type = ""
    objective_metric = ""
    if typed_doc is not None:
        component_type = typed_doc.requirement.component_type or ""
        objective_metric = typed_doc.requirement.objective.metric or ""
    if typed_bundle is not None:
        component_type = typed_bundle.simulation_config.component_type
        objective_metric = typed_bundle.optimization_config.objective.metric

    semantic_rules: List[str] = []
    evidence_urls: List[str] = []
    required_monitor_types: List[str] = ["flux", "mode"]
    required_boundary = "pml"
    min_steps_per_wvl = 10
    required_input_flux_keys, required_output_flux_keys = objective_flux_key_contract("")

    if typed_doc is not None:
        for ref in typed_doc.references:
            evidence_urls.append(ref.url)
            for rule in ref.rules:
                semantic_rules.append(rule.rule)
        if typed_doc.guidance.require_pml:
            required_boundary = "pml"
        else:
            required_boundary = "periodic"

        monitors_from_guidance = [
            item.strip().lower()
            for item in typed_doc.guidance.recommended_monitors
            if item.strip()
        ]
        if monitors_from_guidance:
            required_monitor_types = sorted(set(monitors_from_guidance + ["flux", "mode"]))

        mesh_advice = typed_doc.guidance.mesh_advice.strip().lower()
        if mesh_advice == "mesh_override":
            min_steps_per_wvl = 24
        elif mesh_advice == "fine_mesh":
            min_steps_per_wvl = 26
        elif mesh_advice == "auto_grid":
            min_steps_per_wvl = 20

    if typed_bundle is not None:
        for ref in typed_bundle.simulation_config.doc_references:
            evidence_urls.append(ref.url)
            semantic_rules.extend(ref.rules)
        min_steps_per_wvl = max(min_steps_per_wvl, typed_bundle.simulation_config.domain.min_steps_per_wvl)

    objective_metric_lower = objective_metric.strip().lower()
    required_input_flux_keys, required_output_flux_keys = objective_flux_key_contract(objective_metric_lower)
    semantic_rules = dedupe_contract_rules(
        objective_semantic_contract_rules(objective_metric_lower) + semantic_rules,
        max_items=20,
    )
    evidence_urls = _dedupe_list(evidence_urls, max_items=12)

    hard_physics_gate = Step5HardPhysicsGate(
        required_input_flux_keys=required_input_flux_keys,
        required_output_flux_keys=required_output_flux_keys,
    )

    hard_constraints = [
        Step4HardConstraint(
            code="rag_boundary_requirement",
            path="simulation_config.domain.boundary",
            expected=required_boundary,
            reason="Step4 RAG requires boundary settings to match retrieved guidance.",
        ),
        Step4HardConstraint(
            code="rag_monitor_coverage",
            path="simulation_config.monitors",
            expected=",".join(required_monitor_types),
            reason="Step4 RAG requires monitor coverage for objective observability.",
        ),
        Step4HardConstraint(
            code="rag_mesh_floor",
            path="simulation_config.domain.min_steps_per_wvl",
            expected=str(min_steps_per_wvl),
            reason="Step4 RAG enforces minimum mesh density from semantic guidance.",
        ),
        Step4HardConstraint(
            code="rag_input_injection_floor",
            path="step5.first_iteration.input_flux",
            expected=f">={hard_physics_gate.input_flux_min}",
            reason="Block runs where input-side injection is below the minimum measurable threshold.",
        ),
        Step4HardConstraint(
            code="rag_output_observability",
            path="step5.first_iteration.output_flux",
            expected=(
                "any("
                + ",".join(hard_physics_gate.required_output_flux_keys)
                + f")>={hard_physics_gate.output_flux_min}"
            ),
            reason="Through/drop observability must exist before optimization is allowed to continue.",
        ),
        Step4HardConstraint(
            code="rag_energy_closure",
            path="step5.first_iteration.energy_closure_error",
            expected=f"<={hard_physics_gate.energy_closure_tolerance}",
            reason="Use coarse power closure to intercept non-physical boundary or source setups.",
        ),
        Step4HardConstraint(
            code="rag_field_continuity",
            path="step5.first_iteration.field_continuity_ratio",
            expected=f">={hard_physics_gate.field_continuity_min_ratio}",
            reason="Interface-region field continuity below threshold indicates geometry or boundary mistakes.",
        ),
        Step4HardConstraint(
            code="rag_monitor_coverage_integrity",
            path="step5.first_iteration.monitor_coverage",
            expected="required_monitor_types_present",
            reason="Missing key monitor coverage blocks execution before objective judging.",
        ),
    ]
    if objective_metric_lower in {"demux_routing", "mode_demux", "wdm_routing", "mux_routing"}:
        hard_constraints.append(
            Step4HardConstraint(
                code="rag_mode_expansion_observability",
                path="step5.first_iteration.mode_expansion_artifact",
                expected="present",
                reason="Mode-demux objectives must emit mode-expansion evidence for TE0/TE1 routing verification.",
            )
        )

    return Step4ConstraintPacket(
        component_type=component_type,
        objective_metric=objective_metric,
        semantic_rules=semantic_rules,
        hard_constraints=hard_constraints,
        required_monitor_types=required_monitor_types,
        required_boundary=required_boundary,
        min_steps_per_wvl=min_steps_per_wvl,
        hard_physics_gate=hard_physics_gate,
        evidence_urls=evidence_urls,
    )


def get_inverse_design_step4_rag_memory() -> InverseDesignStep4RAGMemory:
    """Get process-level shared Step4 RAG memory instance."""

    global _STEP4_RAG_MEMORY_SINGLETON
    if _STEP4_RAG_MEMORY_SINGLETON is None:
        _STEP4_RAG_MEMORY_SINGLETON = InverseDesignStep4RAGMemory()
    return _STEP4_RAG_MEMORY_SINGLETON


def inverse_design_step4_rag_schema() -> Dict[str, Any]:
    """Return schema for Step4 RAG packets."""

    return Step4ConstraintPacket.model_json_schema()


def _default_storage_path() -> Path:
    configured = os.getenv("INVERSE_DESIGN_STEP4_RAG_PATH", "").strip()
    if configured:
        return Path(configured)
    return Path("build") / "inverse_design_step4_rag_memory.jsonl"


def _dedupe_list(items: List[str], *, max_items: int) -> List[str]:
    output: List[str] = []
    seen: set[str] = set()
    for item in items:
        normalized = item.strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        output.append(normalized)
        seen.add(key)
        if len(output) >= max_items:
            break
    return output


def _normalize_text(text: str) -> str:
    lowered = text.lower()
    cleaned = re.sub(r"[^a-z0-9_\-\s]+", " ", lowered)
    return re.sub(r"\s+", " ", cleaned).strip()


_STEP4_RAG_MEMORY_SINGLETON: InverseDesignStep4RAGMemory | None = None
