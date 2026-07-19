"""Shared inverse-design data contracts used by agent and tool layers."""

from __future__ import annotations

from typing import Any, Dict, List, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    """Base model that rejects unknown fields to keep payloads stable."""

    model_config = ConfigDict(extra="forbid")


class DiagnosisRepairAction(StrictModel):
    """Suggested repair action from failure diagnosis."""

    action: Literal["set_value", "add_item", "remove_item", "regenerate"] = "regenerate"
    path: str = "root"
    value: Any = None
    reason: str


class RollbackCandidate(StrictModel):
    """Candidate rollback target with confidence and evidence links."""

    step: Literal["step2", "step3", "step4", "step5"]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = ""
    evidence_refs: List[str] = Field(default_factory=list)
    direct_patch_actions: List[DiagnosisRepairAction] = Field(default_factory=list)


class ScenarioFingerprint(StrictModel):
    """Compact scenario signature used for Step6 situation-memory recall."""

    component_type: str = ""
    objective_metric: str = ""
    objective_goal: str = ""
    wavelength_band: str = ""
    domain_ratio: str = ""
    monitor_topology_signature: str = ""
    boundary_type: str = ""


class CheckpointReport(StrictModel):
    """Structured Step6 checkpoint health report emitted every N iterations."""

    checkpoint_iteration: int = 0
    window_size: int = 0
    status: Literal["pass", "fail"] = "pass"
    error_family: str = ""
    error_subtype: str = ""
    reasons: List[str] = Field(default_factory=list)
    objective_values: List[float] = Field(default_factory=list)
    objective_delta: float = 0.0
    oscillation_ratio: float = 0.0
    parameter_update_norm: float = 0.0
    manufacturability_score: float = 1.0
    observability_score: float = 1.0
    metrics: dict[str, Any] = Field(default_factory=dict)


class OptimizerAttributionReport(StrictModel):
    """Optional Step6 hyperparameter-attribution report."""

    status: Literal["ok", "insufficient_samples", "unavailable", "failed"] = "unavailable"
    method: str = ""
    sample_count: int = 0
    feature_names: List[str] = Field(default_factory=list)
    importance_scores: Dict[str, float] = Field(default_factory=dict)
    most_important_param: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    warnings: List[str] = Field(default_factory=list)
    recommended_focus: List[str] = Field(default_factory=list)
    csv_path: str = ""


class RepairCandidate(StrictModel):
    """Machine-readable Step6 repair candidate for short-run evaluation."""

    candidate_id: str
    target_step: Literal["step2", "step3", "step4"]
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    patch_actions: List[DiagnosisRepairAction] = Field(default_factory=list)
    expected_effect: str = ""
    risk: str = "medium"
    evidence_refs: List[str] = Field(default_factory=list)
    rationale: str = ""
    analysis_metadata: Dict[str, Any] = Field(default_factory=dict)


class RepairTrialResult(StrictModel):
    """Short-run repair-trial result used to rank Step6 repair candidates."""

    candidate_id: str
    target_step: Literal["step2", "step3", "step4"]
    short_run_iterations: int = Field(default=0, ge=0)
    short_run_score: float = 0.0
    passed: bool = False
    outcome: Literal["success", "retry", "fallback", "failed"] = "failed"
    reason: str = ""
    patch_actions: List[DiagnosisRepairAction] = Field(default_factory=list)
    checkpoint_status: str = ""


class FailureDiagnosis(StrictModel):
    """Structured troubleshooting result for Step5 failure loops."""

    # Legacy fields kept for compatibility.
    category: Literal[
        "geometry_conflict",
        "monitor_invalid",
        "convergence_issue",
        "resource_limit",
        "unknown",
    ] = "unknown"
    summary: str
    recalled_memories: List[str] = Field(default_factory=list)
    evidence_urls: List[str] = Field(default_factory=list)
    evidence_snippets: List[str] = Field(default_factory=list)
    recommended_repairs: List[str] = Field(default_factory=list)
    repair_actions: List[DiagnosisRepairAction] = Field(default_factory=list)
    suggested_queries: List[str] = Field(default_factory=list)
    # Recovery-contract fields (implementation.md section 11).
    failure_signature: str = ""
    failed_step: str = "step5"
    failure_stage: str = "step5"
    root_cause_stage: Literal["step2", "step3", "step4", "step5", "unknown"] = "unknown"
    error_family: str = "unknown"
    error_subtype: str = "unknown"
    rollback_candidates: List[RollbackCandidate] = Field(default_factory=list)
    checkpoint_report: CheckpointReport | None = None
    checkpoint_reports: List[CheckpointReport] = Field(default_factory=list)
    scenario_fingerprint: ScenarioFingerprint | None = None
    scenario_memories: List[str] = Field(default_factory=list)
    repair_candidates: List[RepairCandidate] = Field(default_factory=list)
    attribution_report: OptimizerAttributionReport | None = None
    selected_rollback_step: str = ""
    selected_repair_candidate_id: str = ""
    replan_actions: List[DiagnosisRepairAction] = Field(default_factory=list)
    resume_from_step: str = ""
    recovery_attempt: int = 0
    recovery_outcome: Literal["pending", "success", "retry", "escalate", "fallback"] = "pending"
    requires_doc_refresh: bool = False
    confidence: float = 0.0
