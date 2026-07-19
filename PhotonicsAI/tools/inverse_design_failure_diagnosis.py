"""Failure diagnosis helpers for inverse-design troubleshooting loops."""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any, Callable, Dict, List, Literal

from pydantic import Field

from PhotonicsAI.tools.inverse_design_types import (
    CheckpointReport,
    DiagnosisRepairAction,
    FailureDiagnosis,
    OptimizerAttributionReport,
    RepairCandidate,
    RollbackCandidate,
    ScenarioFingerprint,
    StrictModel,
)
from PhotonicsAI.tools.inverse_design_optimizer_attribution import (
    analyze_optimizer_attribution,
    build_attribution_guided_candidates,
    collect_optimizer_attribution_samples,
)
from PhotonicsAI.tools.inverse_design_working_memory import (
    InverseDesignWorkingMemory,
    WorkingMemoryEntry,
    get_inverse_design_working_memory,
)
from PhotonicsAI.tools.tidy3d_tools import fetch_tidy3d_doc, search_tidy3d_docs

SearchFn = Callable[..., Dict[str, Any]]
FetchFn = Callable[..., Dict[str, Any]]
LLMCallFn = Callable[[str, str, str], str]
_ROLLBACK_STEPS = {"step2", "step3", "step4", "step5"}


class LLMDiagnosisResult(StrictModel):
    """Optional LLM-enhanced diagnosis payload."""

    category: str = ""
    summary: str = ""
    recommended_repairs: List[str] = Field(default_factory=list)
    repair_actions: List[DiagnosisRepairAction] = Field(default_factory=list)
    suggested_queries: List[str] = Field(default_factory=list)
    error_family: str = ""
    error_subtype: str = ""
    root_cause_stage: str = ""
    requires_doc_refresh: bool | None = None
    rollback_candidates: List[RollbackCandidate] = Field(default_factory=list)
    repair_candidates: List[RepairCandidate] = Field(default_factory=list)


def diagnose_inverse_design_failure(
    failure_message: str,
    *,
    component_type: str = "",
    objective_metric: str = "",
    objective_goal: str = "",
    recent_issues: List[str] | None = None,
    recent_iterations: List[Dict[str, Any]] | None = None,
    checkpoint_report: CheckpointReport | Dict[str, Any] | None = None,
    checkpoint_reports: List[CheckpointReport | Dict[str, Any]] | None = None,
    scenario_fingerprint: ScenarioFingerprint | Dict[str, Any] | None = None,
    failed_step: str = "step5",
    recovery_attempt: int = 0,
    confidence_threshold: float = 0.7,
    search_fn: SearchFn = search_tidy3d_docs,
    fetch_fn: FetchFn = fetch_tidy3d_doc,
    use_llm_advisor: bool | None = None,
    llm_call_fn: LLMCallFn | None = None,
    llm_model: str = "gpt-5.4",
    memory_store: InverseDesignWorkingMemory | None = None,
    enable_optimizer_attribution: bool = False,
    optimizer_attribution_min_samples: int = 12,
) -> FailureDiagnosis:
    """Diagnose failures by recalling memory, fetching docs, and proposing repairs."""

    message = str(failure_message or "").strip() or "Unknown inverse-design failure"
    issues = [str(item) for item in (recent_issues or [])]
    typed_checkpoint = _coerce_checkpoint_report(checkpoint_report)
    typed_checkpoint_reports = [
        _coerce_checkpoint_report(item)
        for item in (checkpoint_reports or [])
        if _coerce_checkpoint_report(item) is not None
    ]
    if typed_checkpoint is not None and not typed_checkpoint_reports:
        typed_checkpoint_reports = [typed_checkpoint]
    typed_scenario = _coerce_scenario_fingerprint(scenario_fingerprint)
    category = _heuristic_category(message=message, issues=issues, checkpoint_report=typed_checkpoint)
    signature = _build_failure_signature(
        component_type=component_type,
        objective_metric=objective_metric,
        category=category,
        message=message,
        issues=issues,
    )

    memory = memory_store or get_inverse_design_working_memory()
    recall_query = " ".join(
        part for part in [component_type, objective_metric, message, " ".join(issues)] if part
    )
    recalled_entries = memory.recall(query=recall_query, limit=3)
    signature_entries = memory.recall_by_failure_signature(signature, limit=5)
    scenario_entries = (
        memory.recall_by_scenario_fingerprint(
            _scenario_fingerprint_key(typed_scenario),
            limit=12,
        )
        if typed_scenario is not None
        else []
    )
    recalled_summaries = [_summarize_memory_entry(entry) for entry in recalled_entries]
    scenario_summaries = [_summarize_memory_entry(entry) for entry in scenario_entries]

    queries = _build_diagnosis_queries(
        component_type=component_type,
        objective_metric=objective_metric,
        failure_message=message,
        recent_issues=issues,
        checkpoint_report=typed_checkpoint,
    )
    evidence_urls, evidence_snippets = _fetch_evidence(
        queries,
        search_fn=search_fn,
        fetch_fn=fetch_fn,
    )
    error_family, error_subtype = _infer_error_family(
        message=message,
        issues=issues,
        checkpoint_report=typed_checkpoint,
    )
    root_cause_stage = _infer_root_cause_stage(error_family=error_family, category=category)
    requires_doc_refresh = error_family in {"documentation_gap", "backend_capability"}
    attribution_report = _build_optimizer_attribution_report(
        enabled=enable_optimizer_attribution,
        error_family=error_family,
        checkpoint_report=typed_checkpoint,
        checkpoint_reports=typed_checkpoint_reports,
        recent_iterations=list(recent_iterations or []),
        scenario_fingerprint=typed_scenario,
        memory=memory,
        min_samples=optimizer_attribution_min_samples,
    )

    recommended_repairs = _heuristic_repairs(
        category=category,
        issues=issues,
        checkpoint_report=typed_checkpoint,
    )
    repair_actions = _heuristic_repair_actions(
        category=category,
        error_family=error_family,
        error_subtype=error_subtype,
    )
    repair_candidates = _build_repair_candidates(
        category=category,
        error_family=error_family,
        error_subtype=error_subtype,
        checkpoint_report=typed_checkpoint,
        evidence_urls=evidence_urls,
        suggested_queries=queries,
        scenario_entries=scenario_entries,
        objective_metric=objective_metric,
        objective_goal=objective_goal,
        attribution_report=attribution_report,
    )
    rollback_candidates = _score_rollback_candidates(
        category=category,
        error_family=error_family,
        root_cause_stage=root_cause_stage,
        evidence_urls=evidence_urls,
        recalled_entries=signature_entries or recalled_entries,
        repair_actions=repair_actions or [
            action
            for candidate in repair_candidates
            for action in candidate.patch_actions
        ],
    )
    selected_step, selected_confidence = _select_primary_candidate(rollback_candidates)
    if selected_confidence < confidence_threshold:
        selected_step = ""
    selected_repair_candidate_id = _select_repair_candidate_id(repair_candidates, selected_step)
    replan_actions = _select_replan_actions(
        repair_candidates=repair_candidates,
        repair_actions=repair_actions,
        selected_repair_candidate_id=selected_repair_candidate_id,
    )
    summary = (
        f"Diagnosed {category} ({error_family}/{error_subtype}) from step `{failed_step}`. "
        f"Evidence docs={len(evidence_urls)}, recalled notes={len(recalled_summaries)}, "
        f"scenario recalls={len(scenario_summaries)}, rollback candidates={len(rollback_candidates)}, "
        f"repair candidates={len(repair_candidates)}."
    )

    diagnosis = FailureDiagnosis(
        category=category,
        summary=summary,
        recalled_memories=recalled_summaries,
        evidence_urls=evidence_urls,
        evidence_snippets=evidence_snippets,
        recommended_repairs=recommended_repairs,
        repair_actions=repair_actions,
        suggested_queries=queries,
        failure_signature=signature,
        failed_step=failed_step,
        failure_stage=failed_step,
        root_cause_stage=root_cause_stage,
        error_family=error_family,
        error_subtype=error_subtype,
        rollback_candidates=rollback_candidates,
        checkpoint_report=typed_checkpoint,
        checkpoint_reports=typed_checkpoint_reports,
        scenario_fingerprint=typed_scenario,
        scenario_memories=scenario_summaries,
        repair_candidates=repair_candidates,
        attribution_report=attribution_report,
        selected_rollback_step=selected_step,
        selected_repair_candidate_id=selected_repair_candidate_id,
        replan_actions=list(replan_actions),
        resume_from_step=selected_step,
        recovery_attempt=max(0, int(recovery_attempt)),
        recovery_outcome="pending",
        requires_doc_refresh=requires_doc_refresh,
        confidence=selected_confidence,
    )

    if _advisor_enabled(use_llm_advisor):
        diagnosis = _apply_llm_advisor(
            diagnosis,
            failure_message=message,
            issues=issues,
            llm_call_fn=llm_call_fn,
            llm_model=llm_model,
        )

    memory.record(
        stage="step5_diagnosis",
        key=component_type or "unknown_component",
        failure_signature=diagnosis.failure_signature,
        scenario_fingerprint=_scenario_fingerprint_key(typed_scenario),
        summary=diagnosis.summary,
        evidence_urls=diagnosis.evidence_urls,
        issues=issues,
        proposed_fixes=diagnosis.recommended_repairs,
        metadata={
            "category": diagnosis.category,
            "error_family": diagnosis.error_family,
            "error_subtype": diagnosis.error_subtype,
            "root_cause_stage": diagnosis.root_cause_stage,
            "objective_metric": objective_metric,
            "suggested_queries": diagnosis.suggested_queries,
            "rollback_candidates": [
                candidate.model_dump() for candidate in diagnosis.rollback_candidates
            ],
            "repair_candidates": [
                candidate.model_dump() for candidate in diagnosis.repair_candidates
            ],
            "selected_rollback_step": diagnosis.selected_rollback_step,
            "selected_repair_candidate_id": diagnosis.selected_repair_candidate_id,
            "recovery_attempt": diagnosis.recovery_attempt,
            "attribution_report": (
                diagnosis.attribution_report.model_dump()
                if diagnosis.attribution_report is not None
                else None
            ),
            "checkpoint_report": (
                diagnosis.checkpoint_report.model_dump()
                if diagnosis.checkpoint_report is not None
                else None
            ),
            "recent_iterations": list(recent_iterations or [])[-5:],
        },
    )
    return diagnosis


def _build_optimizer_attribution_report(
    *,
    enabled: bool,
    error_family: str,
    checkpoint_report: CheckpointReport | None,
    checkpoint_reports: List[CheckpointReport],
    recent_iterations: List[Dict[str, Any]],
    scenario_fingerprint: ScenarioFingerprint | None,
    memory: InverseDesignWorkingMemory,
    min_samples: int,
) -> OptimizerAttributionReport | None:
    """Run optional sklearn attribution without changing diagnosis authority."""

    if not enabled:
        return None
    if error_family != "optimization_setup" or checkpoint_report is None:
        return OptimizerAttributionReport(
            status="insufficient_samples",
            method="random_forest_permutation",
            sample_count=0,
            warnings=["Optimizer attribution is only enabled for optimization_setup checkpoint failures."],
        )
    samples = collect_optimizer_attribution_samples(
        recent_iterations=recent_iterations,
        checkpoint_report=checkpoint_report,
        scenario_fingerprint=scenario_fingerprint,
        memory_store=memory,
    )
    # Include older checkpoint context in the report path through the same rows;
    # the latest failing checkpoint remains the authoritative trigger.
    if checkpoint_reports and not samples:
        samples = collect_optimizer_attribution_samples(
            recent_iterations=recent_iterations,
            checkpoint_report=checkpoint_reports[-1],
            scenario_fingerprint=scenario_fingerprint,
            memory_store=memory,
        )
    return analyze_optimizer_attribution(
        samples,
        min_samples=max(1, int(min_samples)),
        method="random_forest_permutation",
    )


def _build_failure_signature(
    *,
    component_type: str,
    objective_metric: str,
    category: str,
    message: str,
    issues: List[str],
) -> str:
    normalized = _normalize_text(" ".join([message] + issues))
    base = "|".join(
        [
            component_type.strip().lower(),
            objective_metric.strip().lower(),
            category.strip().lower(),
            normalized[:180],
        ]
    )
    digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:12]
    return f"{category}:{digest}"


def _coerce_checkpoint_report(
    checkpoint_report: CheckpointReport | Dict[str, Any] | None,
) -> CheckpointReport | None:
    if checkpoint_report is None:
        return None
    if isinstance(checkpoint_report, CheckpointReport):
        return checkpoint_report
    return CheckpointReport.model_validate(checkpoint_report)


def _coerce_scenario_fingerprint(
    scenario_fingerprint: ScenarioFingerprint | Dict[str, Any] | None,
) -> ScenarioFingerprint | None:
    if scenario_fingerprint is None:
        return None
    if isinstance(scenario_fingerprint, ScenarioFingerprint):
        return scenario_fingerprint
    return ScenarioFingerprint.model_validate(scenario_fingerprint)


def _scenario_fingerprint_key(fingerprint: ScenarioFingerprint | None) -> str:
    if fingerprint is None:
        return ""
    return "|".join(
        [
            fingerprint.component_type,
            fingerprint.objective_metric,
            fingerprint.objective_goal,
            fingerprint.wavelength_band,
            fingerprint.domain_ratio,
            fingerprint.monitor_topology_signature,
            fingerprint.boundary_type,
        ]
    ).strip("|")


def _build_diagnosis_queries(
    *,
    component_type: str,
    objective_metric: str,
    failure_message: str,
    recent_issues: List[str],
    checkpoint_report: CheckpointReport | None = None,
) -> List[str]:
    queries = [
        f"Tidy3D inverse design troubleshooting {component_type or 'photonic device'}",
        f"Tidy3D {component_type or 'device'} {objective_metric or 'objective'} monitor source setup",
    ]
    for issue in recent_issues[:3]:
        queries.append(f"Tidy3D faq {issue.replace('_', ' ')}")
    if failure_message:
        short_message = re.sub(r"\s+", " ", failure_message)[:80]
        queries.append(f"Tidy3D {short_message}")
    if checkpoint_report is not None:
        if checkpoint_report.error_subtype:
            queries.append(
                f"Tidy3D inverse design {checkpoint_report.error_subtype.replace('_', ' ')}"
            )
        if checkpoint_report.error_family == "optimization_setup":
            queries.append("Tidy3D adjoint convergence troubleshooting")
        if checkpoint_report.error_family == "simulation_scene":
            queries.append("Tidy3D monitor placement output flux troubleshooting")

    deduped: List[str] = []
    seen: set[str] = set()
    for query in queries:
        normalized = query.strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        deduped.append(normalized)
        seen.add(key)
    return deduped


def _fetch_evidence(
    queries: List[str],
    *,
    search_fn: SearchFn,
    fetch_fn: FetchFn,
) -> tuple[List[str], List[str]]:
    urls: List[str] = []
    snippets: List[str] = []
    seen: set[str] = set()
    for query in queries[:3]:
        search_result = search_fn(query=query, max_results=2)
        if not search_result.get("ok"):
            continue
        for row in search_result.get("data", {}).get("results", []):
            url = str(row.get("url", "")).strip()
            if not url or url in seen:
                continue
            fetch_result = fetch_fn(url=url)
            if not fetch_result.get("ok"):
                continue
            content = str(fetch_result.get("data", {}).get("content", "")).strip()
            if not content:
                continue
            seen.add(url)
            urls.append(url)
            snippets.append(re.sub(r"\s+", " ", content)[:260])
            if len(urls) >= 4:
                return urls, snippets
    return urls, snippets


def _heuristic_category(
    *,
    message: str,
    issues: List[str],
    checkpoint_report: CheckpointReport | None = None,
) -> str:
    if checkpoint_report is not None:
        if checkpoint_report.error_family == "simulation_scene":
            if checkpoint_report.error_subtype == "monitor_observability":
                return "monitor_invalid"
            return "geometry_conflict"
        if checkpoint_report.error_family == "optimization_setup":
            return "convergence_issue"
    lowered = f"{message} {' '.join(issues)}".lower()
    if any(token in lowered for token in ["geometry", "pml", "width", "height", "domain", "boundary"]):
        return "geometry_conflict"
    if any(token in lowered for token in ["monitor", "mode", "freq", "source"]):
        return "monitor_invalid"
    if any(token in lowered for token in ["converg", "target", "improvement", "objective"]):
        return "convergence_issue"
    if any(
        token in lowered
        for token in [
            "memory",
            "resource",
            "timeout",
            "quota",
            "cost",
            "network",
            "proxy",
            "max retries exceeded",
            "httpsconnectionpool",
            "connecterror",
            "readtimeout",
            "ssl",
            "simulation_data_unavailable",
            "simulation data unavailable",
            "failed to download the simulation data file",
            "download the simulation data file from the server",
            "failed to download simulation data",
            "task was successfully run",
            "insufficient balance",
        ]
    ):
        return "resource_limit"
    return "unknown"


def _infer_error_family(
    *,
    message: str,
    issues: List[str],
    checkpoint_report: CheckpointReport | None = None,
) -> tuple[str, str]:
    if checkpoint_report is not None and checkpoint_report.error_family:
        return checkpoint_report.error_family, checkpoint_report.error_subtype or "unknown"
    lowered = f"{message} {' '.join(issues)}".lower()
    if any(
        token in lowered
        for token in [
            "simulation_data_unavailable",
            "simulation data unavailable",
            "failed to download the simulation data file",
            "download the simulation data file from the server",
            "failed to download simulation data",
            "max retries exceeded",
            "httpsconnectionpool",
            "connecterror",
            "readtimeout",
            "ssl",
            "task was successfully run",
            "insufficient balance",
            "credit",
            "quota",
        ]
    ):
        return "resource_limit", "simulation_data_unavailable"
    if any(token in lowered for token in ["source", "monitor", "boundary", "pml", "domain", "geometry"]):
        return "simulation_scene", "scene_semantics"
    if any(token in lowered for token in ["objective", "metric", "bounds", "termination", "optimizer"]):
        return "optimization_setup", "objective_or_bounds"
    if any(token in lowered for token in ["doc", "documentation", "conflict", "unsupported api"]):
        return "documentation_gap", "missing_or_conflicting_evidence"
    if any(token in lowered for token in ["auth", "quota", "credit", "network", "timeout", "queue", "proxy"]):
        return "cloud_runtime", "transient_or_account"
    if any(token in lowered for token in ["converg", "improvement", "stuck", "oscillat"]):
        return "convergence", "non_improving_objective"
    if any(token in lowered for token in ["backend", "version", "capability", "not implemented"]):
        return "backend_capability", "api_or_feature_mismatch"
    return "unknown", "unknown"


def _infer_root_cause_stage(*, error_family: str, category: str) -> Literal["step2", "step3", "step4", "step5", "unknown"]:
    if error_family == "documentation_gap":
        return "step2"
    if error_family in {"optimization_setup", "backend_capability"}:
        return "step3"
    if error_family == "simulation_scene":
        return "step4"
    if error_family in {"cloud_runtime", "convergence"}:
        return "step5"
    if category in {"geometry_conflict", "monitor_invalid"}:
        return "step4"
    if category == "convergence_issue":
        return "step5"
    if category == "resource_limit":
        return "step5"
    return "unknown"


def _score_rollback_candidates(
    *,
    category: str,
    error_family: str,
    root_cause_stage: str,
    evidence_urls: List[str],
    recalled_entries: List[WorkingMemoryEntry],
    repair_actions: List[DiagnosisRepairAction],
) -> List[RollbackCandidate]:
    base_map: Dict[str, List[tuple[str, float, str]]] = {
        "simulation_scene": [
            ("step4", 0.82, "Most scene-geometry/source/monitor errors originate in Step4 validation layer."),
            ("step3", 0.56, "If Step4 repair repeats, regenerate config generation assumptions from Step3."),
        ],
        "optimization_setup": [
            ("step3", 0.84, "Objective/variable/termination mismatches are compiled in Step3."),
            ("step4", 0.48, "Step4 may provide deterministic repair hints for minor semantic mismatch."),
            ("step2", 0.22, "Escalate to Step2 only when objective semantics lack documentation evidence."),
        ],
        "documentation_gap": [
            ("step2", 0.88, "Doc evidence is missing/conflicting; refresh MCP retrieval first."),
            ("step3", 0.58, "Recompile configuration once documentation context is refreshed."),
        ],
        "cloud_runtime": [
            ("step5", 0.90, "Cloud/runtime issues are usually transient and should retry in Step5 first."),
            ("step4", 0.30, "If retries keep failing, reduce runtime burden via Step4 repair suggestions."),
        ],
        "convergence": [
            ("step5", 0.76, "Adjust optimizer controls locally in Step5 before upstream rollback."),
            ("step3", 0.63, "If no gain after retries, revise objective/bounds/termination in Step3."),
            ("step4", 0.24, "Step4 is less likely root cause for convergence-only failures."),
        ],
        "backend_capability": [
            ("step3", 0.73, "API capability mismatch is often a Step3 compilation mismatch."),
            ("step2", 0.68, "Refresh docs when capability mismatch indicates outdated evidence."),
        ],
        "unknown": [
            ("step4", 0.52, "Nearest-upstream conservative rollback when root cause is uncertain."),
            ("step3", 0.41, "Escalate to Step3 if Step4 repair does not resolve repeated failures."),
            ("step2", 0.19, "Doc refresh as last upstream expansion."),
        ],
    }

    choices = list(base_map.get(error_family, base_map["unknown"]))
    if root_cause_stage in _ROLLBACK_STEPS:
        choices.insert(0, (root_cause_stage, 0.78, "Root-cause stage inferred from deterministic diagnosis."))

    # Repeated identical failure signatures reduce confidence on the same stage.
    repeat_count = sum(1 for entry in recalled_entries if entry.stage == "recovery")
    adjusted: List[RollbackCandidate] = []
    seen_steps: set[str] = set()
    for step, confidence, reason in choices:
        if step not in _ROLLBACK_STEPS or step in seen_steps:
            continue
        conf = confidence
        if repeat_count > 0 and step == root_cause_stage:
            conf = max(0.15, conf - 0.1 * repeat_count)
        if repeat_count > 0 and step != root_cause_stage and step in {"step3", "step2"}:
            conf = min(0.95, conf + 0.05 * repeat_count)
        evidence_refs = list(evidence_urls[:2])
        if not evidence_refs:
            evidence_refs = [entry.key for entry in recalled_entries if entry.key][:2]
        adjusted.append(
            RollbackCandidate(
                step=step,  # type: ignore[arg-type]
                confidence=round(conf, 3),
                reason=reason,
                evidence_refs=evidence_refs,
                direct_patch_actions=list(repair_actions[:2]),
            )
        )
        seen_steps.add(step)

    adjusted.sort(key=lambda item: item.confidence, reverse=True)
    return adjusted


def _select_primary_candidate(candidates: List[RollbackCandidate]) -> tuple[str, float]:
    if not candidates:
        return "", 0.0
    top = candidates[0]
    return top.step, top.confidence


def _heuristic_repairs(
    *,
    category: str,
    issues: List[str],
    checkpoint_report: CheckpointReport | None = None,
) -> List[str]:
    base_by_category = {
        "geometry_conflict": [
            "Increase domain span and enforce PML clearance.",
            "Clamp geometry parameters to physically valid ranges.",
        ],
        "monitor_invalid": [
            "Ensure at least one output-side flux monitor and one mode monitor.",
            "Align monitor frequencies with source center frequency.",
        ],
        "convergence_issue": [
            "Increase max_iterations and relax min_improvement threshold.",
            "Refine objective metric and add diagnostic monitors.",
        ],
        "resource_limit": [
            "Reduce domain and monitor count before re-running optimization.",
            "Use conservative run_time and shutoff values to control cost.",
        ],
        "unknown": [
            "Regenerate config from Step3 with updated constraints.",
            "Run Step4 validation and apply all deterministic repair actions.",
        ],
    }
    repairs = list(base_by_category.get(category, base_by_category["unknown"]))
    if checkpoint_report is not None and checkpoint_report.reasons:
        repairs.extend(checkpoint_report.reasons)
    for issue in issues:
        repairs.append(f"Address validator issue `{issue}` before restarting Step5.")
    return repairs[:6]


def _heuristic_repair_actions(
    *,
    category: str,
    error_family: str,
    error_subtype: str,
) -> List[DiagnosisRepairAction]:
    if error_family == "simulation_scene" and error_subtype == "monitor_observability":
        return [
            DiagnosisRepairAction(
                action="add_item",
                path="simulation_config.monitors",
                value={
                    "name": "through_flux_diagnosis",
                    "monitor_type": "flux",
                    "center_um": [4.0, 0.0, 0.11],
                    "size_um": [0.0, 1.5, 1.0],
                    "freqs_hz": [193414489032258.06],
                    "metric": "transmission",
                },
                reason="Guarantee output observability for objective extraction.",
            )
        ]
    if error_family == "optimization_setup" and error_subtype in {"oscillation", "stalled_convergence", "objective_direction", "objective_regression"}:
        return _optimizer_subtype_patch_actions(error_subtype)
    if category == "geometry_conflict":
        return [
            DiagnosisRepairAction(
                action="set_value",
                path="simulation_config.domain.size_um",
                value=[20.0, 12.0, 4.0],
                reason="Increase domain to avoid boundary coupling and PML clipping.",
            )
        ]
    if category == "resource_limit":
        return [
            DiagnosisRepairAction(
                action="set_value",
                path="simulation_config.run_time_s",
                value=2e-12,
                reason="Reduce simulation horizon to control runtime cost.",
            )
        ]
    return [
        DiagnosisRepairAction(
            action="regenerate",
            path="root",
            value="step3_and_step4",
            reason="Unknown failure type; regenerate validated configuration first.",
        )
    ]


def _build_repair_candidates(
    *,
    category: str,
    error_family: str,
    error_subtype: str,
    checkpoint_report: CheckpointReport | None,
    evidence_urls: List[str],
    suggested_queries: List[str],
    scenario_entries: List[WorkingMemoryEntry],
    objective_metric: str,
    objective_goal: str,
    attribution_report: OptimizerAttributionReport | None = None,
) -> List[RepairCandidate]:
    candidates: List[RepairCandidate] = []
    candidates.extend(_repair_candidates_from_memory(scenario_entries))

    if error_family == "optimization_setup":
        if attribution_report is not None:
            candidates.extend(
                build_attribution_guided_candidates(
                    error_subtype=error_subtype,
                    attribution_report=attribution_report,
                    objective_goal=objective_goal,
                )
            )
        candidates.extend(
            [
                _optimizer_subtype_repair_candidate(
                    error_subtype=error_subtype,
                    evidence_urls=evidence_urls,
                    attribution_report=attribution_report,
                ),
                RepairCandidate(
                    candidate_id="step3_patience_relax",
                    target_step="step3",
                    confidence=0.71,
                    patch_actions=[
                        DiagnosisRepairAction(
                            action="set_value",
                            path="optimization_config.termination.patience",
                            value=5,
                            reason="Avoid premature termination during checkpoint monitoring.",
                        )
                    ],
                    expected_effect="Reduce short-window noise sensitivity before rollback escalates further upstream.",
                    risk="medium",
                    evidence_refs=list(evidence_urls[:2]),
                    rationale=f"Objective `{objective_metric}` / `{objective_goal}` may need more patience.",
                ),
            ]
        )

    if error_family == "simulation_scene":
        candidates.extend(
            [
                RepairCandidate(
                    candidate_id="step4_observability_restore",
                    target_step="step4",
                    confidence=0.86,
                    patch_actions=[
                        DiagnosisRepairAction(
                            action="add_item",
                            path="simulation_config.monitors",
                            value={
                                "name": "through_flux_checkpoint",
                                "monitor_type": "flux",
                                "center_um": [4.0, 0.0, 0.11],
                                "size_um": [0.0, 1.5, 1.0],
                                "freqs_hz": [193414489032258.06],
                                "metric": "transmission",
                            },
                            reason="Restore output observability for Step6 checkpoint monitoring.",
                        )
                    ],
                    expected_effect="Recover monitor signal coverage so checkpoint metrics become reliable.",
                    risk="low",
                    evidence_refs=list(evidence_urls[:2]),
                    rationale="Scene-level recovery is preferred when checkpoint observability collapses.",
                ),
                RepairCandidate(
                    candidate_id="step4_domain_expand",
                    target_step="step4",
                    confidence=0.73,
                    patch_actions=[
                        DiagnosisRepairAction(
                            action="set_value",
                            path="simulation_config.domain.size_um",
                            value=[20.0, 12.0, 4.0],
                            reason="Increase domain margin to prevent scene clipping and weak response.",
                        )
                    ],
                    expected_effect="Improve monitor signal and reduce boundary contamination.",
                    risk="medium",
                    evidence_refs=list(evidence_urls[:2]),
                    rationale="Domain-margin increase is a safe fallback when scene semantics are weak.",
                ),
            ]
        )

    if error_family == "documentation_gap":
        candidates.append(
            RepairCandidate(
                candidate_id="step2_doc_refresh",
                target_step="step2",
                confidence=0.82,
                patch_actions=[
                    DiagnosisRepairAction(
                        action="add_item",
                        path="queries",
                        value=(suggested_queries[0] if suggested_queries else "Tidy3D inverse design troubleshooting"),
                        reason="Inject an extra MCP query before regenerating downstream config.",
                    )
                ],
                expected_effect="Refresh Step2 evidence before rerunning Step3/Step4 compilation.",
                risk="low",
                evidence_refs=list(evidence_urls[:2]),
                rationale="Documentation-gap diagnosis requires upstream evidence refresh first.",
            )
        )

    if not candidates:
        candidates.append(
            RepairCandidate(
                candidate_id="generic_regenerate_step3",
                target_step="step3",
                confidence=0.45,
                patch_actions=[
                    DiagnosisRepairAction(
                        action="regenerate",
                        path="root",
                        value="step3_and_step4",
                        reason="Fallback regeneration when diagnosis confidence remains low.",
                    )
                ],
                expected_effect="Regenerate validated config with a conservative upstream rollback.",
                risk="medium",
                evidence_refs=list(evidence_urls[:2]),
                rationale=f"No stronger Step6 repair candidate was found for category `{category}`.",
            )
        )

    return _apply_counterexample_penalties(
        _dedupe_repair_candidates(candidates),
        scenario_entries,
    )


def _optimizer_subtype_repair_candidate(
    *,
    error_subtype: str,
    evidence_urls: List[str],
    attribution_report: OptimizerAttributionReport | None,
) -> RepairCandidate:
    subtype = error_subtype or "optimization_setup"
    metadata: Dict[str, Any] = {}
    if attribution_report is not None:
        metadata = {
            "attribution_status": attribution_report.status,
            "method": attribution_report.method,
            "sample_count": attribution_report.sample_count,
            "most_important_param": attribution_report.most_important_param,
            "importance_scores": dict(attribution_report.importance_scores),
            "warnings": list(attribution_report.warnings),
        }
    return RepairCandidate(
        candidate_id=f"step3_{subtype}_tune",
        target_step="step3",
        confidence=_optimizer_subtype_confidence(subtype, attribution_report),
        patch_actions=_optimizer_subtype_patch_actions(subtype),
        expected_effect="Apply subtype-specific optimizer recovery without treating attribution as the root-cause authority.",
        risk="low" if subtype in {"oscillation", "objective_regression", "objective_direction"} else "medium",
        evidence_refs=list(evidence_urls[:2]),
        rationale=f"Checkpoint subtype={subtype} maps to a bounded Step3 optimizer-hints repair.",
        analysis_metadata=metadata,
    )


def _optimizer_subtype_confidence(
    subtype: str,
    attribution_report: OptimizerAttributionReport | None,
) -> float:
    base = {
        "objective_regression": 0.84,
        "objective_direction": 0.82,
        "oscillation": 0.84,
        "stalled_convergence": 0.78,
    }.get(subtype, 0.70)
    if attribution_report is not None and attribution_report.status == "ok":
        base = min(0.93, base + 0.05 * attribution_report.confidence)
    return round(base, 3)


def _optimizer_subtype_patch_actions(subtype: str) -> List[DiagnosisRepairAction]:
    subtype = str(subtype or "").strip().lower()
    if subtype == "objective_regression":
        return [
            _set_action("optimization_config.optimizer_hints.learning_rate", 0.03, "Reduce step size after FOM regression."),
            _set_action("optimization_config.optimizer_hints.beta", 10.0, "Use mild projection pressure during regression recovery."),
            _set_action("optimization_config.optimizer_hints.penalty_weight", 0.5, "Avoid extra penalty pressure while restoring FOM."),
            _set_action("optimization_config.termination.max_iterations", 40, "Allow a bounded recovery run."),
        ]
    if subtype == "objective_direction":
        return [
            _set_action("optimization_config.optimizer_hints.learning_rate", 0.03, "Reduce step size to correct wrong-direction updates."),
            _set_action("optimization_config.optimizer_hints.beta", 6.0, "Keep projection pressure neutral until direction recovers."),
            _set_action("optimization_config.optimizer_hints.penalty_weight", 0.5, "Keep penalty pressure neutral while correcting direction."),
            _set_action("optimization_config.termination.max_iterations", 40, "Allow a bounded recovery run."),
        ]
    if subtype == "oscillation":
        return [
            _set_action("optimization_config.optimizer_hints.learning_rate", 0.05, "Lower learning rate to damp objective oscillation."),
            _set_action("optimization_config.optimizer_hints.beta", 8.0, "Increase projection pressure gently rather than jumping to beta=30."),
            _set_action("optimization_config.optimizer_hints.penalty_weight", 0.6, "Add mild regularization to reduce oscillatory updates."),
            _set_action("optimization_config.termination.max_iterations", 40, "Allow a bounded recovery run."),
        ]
    if subtype == "stalled_convergence":
        return [
            _set_action("optimization_config.optimizer_hints.learning_rate", 0.08, "Keep enough step size to escape a plateau."),
            _set_action("optimization_config.optimizer_hints.beta", 10.0, "Add moderate projection pressure to change the design landscape."),
            _set_action("optimization_config.optimizer_hints.penalty_weight", 0.6, "Add mild manufacturability pressure during plateau recovery."),
            _set_action("optimization_config.termination.max_iterations", 60, "Give plateau recovery more room."),
            _set_action("optimization_config.termination.patience", 5, "Avoid premature stop during plateau recovery."),
        ]
    return [
        _set_action("optimization_config.termination.max_iterations", 40, "Allow a bounded optimizer recovery run."),
        _set_action("optimization_config.termination.patience", 5, "Avoid premature stop during recovery."),
    ]


def _set_action(path: str, value: Any, reason: str) -> DiagnosisRepairAction:
    return DiagnosisRepairAction(action="set_value", path=path, value=value, reason=reason)


def _apply_counterexample_penalties(
    candidates: List[RepairCandidate],
    scenario_entries: List[WorkingMemoryEntry],
) -> List[RepairCandidate]:
    counterexample_index = _build_counterexample_index(scenario_entries)
    if not counterexample_index["candidate_fail_counts"] and not counterexample_index["patch_fail_counts"]:
        return candidates

    adjusted: List[RepairCandidate] = []
    for candidate in candidates:
        candidate_id_key = str(candidate.candidate_id).strip().lower()
        candidate_failures = int(counterexample_index["candidate_fail_counts"].get(candidate_id_key, 0))
        patch_signature = _patch_signature(candidate.patch_actions)
        patch_failures = int(counterexample_index["patch_fail_counts"].get(patch_signature, 0))
        failure_count = candidate_failures + patch_failures
        if failure_count <= 0:
            adjusted.append(candidate)
            continue

        penalty = min(0.65, 0.22 * failure_count)
        confidence = max(0.05, float(candidate.confidence) - penalty)
        evidence_refs = list(candidate.evidence_refs)
        counter_refs = list(counterexample_index["evidence_refs"].get(candidate_id_key, []))
        if patch_signature:
            counter_refs.extend(counterexample_index["evidence_refs"].get(patch_signature, []))
        for ref in counter_refs:
            if ref and ref not in evidence_refs:
                evidence_refs.append(ref)
            if len(evidence_refs) >= 4:
                break

        rationale_prefix = (
            f"Downranked by scenario counter-evidence ({failure_count} failed repair attempt(s) "
            "in similar fingerprint)."
        )
        rationale = f"{rationale_prefix} {candidate.rationale}".strip()
        adjusted.append(
            RepairCandidate(
                candidate_id=candidate.candidate_id,
                target_step=candidate.target_step,
                confidence=round(confidence, 3),
                patch_actions=list(candidate.patch_actions),
                expected_effect=candidate.expected_effect,
                risk=candidate.risk,
                evidence_refs=evidence_refs,
                rationale=rationale,
            )
        )

    adjusted.sort(key=lambda item: item.confidence, reverse=True)
    return adjusted


def _build_counterexample_index(entries: List[WorkingMemoryEntry]) -> Dict[str, Dict[str, Any]]:
    candidate_fail_counts: Dict[str, int] = {}
    patch_fail_counts: Dict[str, int] = {}
    evidence_refs: Dict[str, List[str]] = {}
    failed_outcomes = {"failed", "retry", "escalate", "fallback", "budget_exhausted"}

    for entry in entries:
        metadata = entry.metadata or {}
        outcome = str(metadata.get("recovery_outcome", "")).strip().lower()
        if outcome not in failed_outcomes:
            continue

        candidate_id = str(metadata.get("candidate_id", "")).strip().lower()
        patch_payload = metadata.get("patch_actions") or metadata.get("replan_actions") or []
        patch_actions = _coerce_patch_actions(patch_payload)
        patch_signature = _patch_signature(patch_actions)
        evidence_ref = entry.key or entry.failure_signature

        if candidate_id:
            candidate_fail_counts[candidate_id] = candidate_fail_counts.get(candidate_id, 0) + 1
            if evidence_ref:
                refs = evidence_refs.setdefault(candidate_id, [])
                if evidence_ref not in refs:
                    refs.append(evidence_ref)
        if patch_signature:
            patch_fail_counts[patch_signature] = patch_fail_counts.get(patch_signature, 0) + 1
            if evidence_ref:
                refs = evidence_refs.setdefault(patch_signature, [])
                if evidence_ref not in refs:
                    refs.append(evidence_ref)

    return {
        "candidate_fail_counts": candidate_fail_counts,
        "patch_fail_counts": patch_fail_counts,
        "evidence_refs": evidence_refs,
    }


def _coerce_patch_actions(payload: Any) -> List[DiagnosisRepairAction]:
    if not isinstance(payload, list):
        return []
    coerced: List[DiagnosisRepairAction] = []
    for item in payload:
        try:
            coerced.append(
                item
                if isinstance(item, DiagnosisRepairAction)
                else DiagnosisRepairAction.model_validate(item)
            )
        except Exception:
            continue
    return coerced


def _patch_signature(actions: List[DiagnosisRepairAction]) -> str:
    if not actions:
        return ""
    parts: List[str] = []
    for action in actions:
        try:
            raw_value = json.dumps(action.value, ensure_ascii=True, sort_keys=True)
        except Exception:
            raw_value = str(action.value)
        value_hash = hashlib.sha1(raw_value.encode("utf-8")).hexdigest()[:8]
        parts.append(f"{action.action}:{action.path}:{value_hash}")
    return "|".join(sorted(parts))


def _repair_candidates_from_memory(entries: List[WorkingMemoryEntry]) -> List[RepairCandidate]:
    recalled: List[RepairCandidate] = []
    for index, entry in enumerate(entries, start=1):
        metadata = entry.metadata or {}
        if str(metadata.get("recovery_outcome", "")).lower() not in {"success", "retry"}:
            continue
        patch_payload = metadata.get("patch_actions") or metadata.get("replan_actions") or []
        if not isinstance(patch_payload, list) or not patch_payload:
            continue
        try:
            patch_actions = [
                item
                if isinstance(item, DiagnosisRepairAction)
                else DiagnosisRepairAction.model_validate(item)
                for item in patch_payload
            ]
        except Exception:
            continue
        target_step = str(
            metadata.get("target_step")
            or metadata.get("selected_rollback_step")
            or metadata.get("resume_from_step")
            or "step4"
        ).strip()
        if target_step not in {"step2", "step3", "step4"}:
            continue
        recalled.append(
            RepairCandidate(
                candidate_id=f"scenario_memory_{index}",
                target_step=target_step,  # type: ignore[arg-type]
                confidence=0.77,
                patch_actions=patch_actions,
                expected_effect="Reuse a previously successful Step6 repair pattern from scenario memory.",
                risk="low",
                evidence_refs=[entry.key] if entry.key else [],
                rationale=entry.summary,
            )
        )
    return recalled


def _dedupe_repair_candidates(candidates: List[RepairCandidate]) -> List[RepairCandidate]:
    deduped: List[RepairCandidate] = []
    seen: set[tuple[str, str]] = set()
    for candidate in candidates:
        signature = (
            candidate.target_step,
            "|".join(action.path for action in candidate.patch_actions),
        )
        if signature in seen:
            continue
        deduped.append(candidate)
        seen.add(signature)
    return deduped


def _select_repair_candidate_id(
    repair_candidates: List[RepairCandidate],
    selected_step: str,
) -> str:
    if not repair_candidates:
        return ""
    filtered = [candidate for candidate in repair_candidates if candidate.target_step == selected_step]
    pool = filtered or repair_candidates
    pool.sort(key=lambda item: item.confidence, reverse=True)
    return pool[0].candidate_id


def _select_replan_actions(
    *,
    repair_candidates: List[RepairCandidate],
    repair_actions: List[DiagnosisRepairAction],
    selected_repair_candidate_id: str,
) -> List[DiagnosisRepairAction]:
    if selected_repair_candidate_id:
        candidate = next(
            (item for item in repair_candidates if item.candidate_id == selected_repair_candidate_id),
            None,
        )
        if candidate is not None and candidate.patch_actions:
            return list(candidate.patch_actions)
    return list(repair_actions)


def _apply_llm_advisor(
    diagnosis: FailureDiagnosis,
    *,
    failure_message: str,
    issues: List[str],
    llm_call_fn: LLMCallFn | None,
    llm_model: str,
) -> FailureDiagnosis:
    caller = llm_call_fn or _default_llm_call
    if caller is None:
        return diagnosis

    prompt = (
        "Refine this failure diagnosis for inverse design troubleshooting. "
        "Return strict JSON: "
        "{category, summary, recommended_repairs, repair_actions, suggested_queries, "
        "error_family, error_subtype, root_cause_stage, requires_doc_refresh, "
        "rollback_candidates, repair_candidates}.\n"
        f"Failure message: {failure_message}\n"
        f"Issues: {json.dumps(issues, ensure_ascii=True)}\n"
        f"Current diagnosis: {diagnosis.model_dump_json()}"
    )
    sys_prompt = (
        "You are a Tidy3D troubleshooting assistant. "
        "Return JSON only and keep repair actions executable."
    )

    try:
        raw = caller(prompt, sys_prompt, llm_model)
        llm_result = LLMDiagnosisResult.model_validate(_extract_json_object(str(raw)))
    except Exception:
        return diagnosis

    category = diagnosis.category
    if llm_result.category in {
        "geometry_conflict",
        "monitor_invalid",
        "convergence_issue",
        "resource_limit",
        "unknown",
    }:
        category = llm_result.category  # type: ignore[assignment]

    root_cause_stage = diagnosis.root_cause_stage
    if llm_result.root_cause_stage in _ROLLBACK_STEPS:
        root_cause_stage = llm_result.root_cause_stage  # type: ignore[assignment]

    candidates = list(diagnosis.rollback_candidates)
    if llm_result.rollback_candidates:
        candidates = _normalize_candidates(llm_result.rollback_candidates, diagnosis.rollback_candidates)
    repair_candidates = list(diagnosis.repair_candidates)
    if llm_result.repair_candidates:
        repair_candidates = _dedupe_repair_candidates(list(llm_result.repair_candidates))

    selected_step, selected_confidence = _select_primary_candidate(candidates)
    selected_repair_candidate_id = _select_repair_candidate_id(repair_candidates, selected_step)
    requires_doc_refresh = diagnosis.requires_doc_refresh
    if llm_result.requires_doc_refresh is not None:
        requires_doc_refresh = bool(llm_result.requires_doc_refresh)

    return FailureDiagnosis(
        category=category,
        summary=llm_result.summary or diagnosis.summary,
        recalled_memories=list(diagnosis.recalled_memories),
        evidence_urls=list(diagnosis.evidence_urls),
        evidence_snippets=list(diagnosis.evidence_snippets),
        recommended_repairs=list(llm_result.recommended_repairs or diagnosis.recommended_repairs),
        repair_actions=list(llm_result.repair_actions or diagnosis.repair_actions),
        suggested_queries=list(llm_result.suggested_queries or diagnosis.suggested_queries),
        failure_signature=diagnosis.failure_signature,
        failed_step=diagnosis.failed_step,
        failure_stage=diagnosis.failure_stage,
        root_cause_stage=root_cause_stage,
        error_family=llm_result.error_family or diagnosis.error_family,
        error_subtype=llm_result.error_subtype or diagnosis.error_subtype,
        rollback_candidates=candidates,
        checkpoint_report=diagnosis.checkpoint_report,
        checkpoint_reports=list(diagnosis.checkpoint_reports),
        scenario_fingerprint=diagnosis.scenario_fingerprint,
        scenario_memories=list(diagnosis.scenario_memories),
        repair_candidates=repair_candidates,
        attribution_report=diagnosis.attribution_report,
        selected_rollback_step=selected_step,
        selected_repair_candidate_id=selected_repair_candidate_id,
        replan_actions=_select_replan_actions(
            repair_candidates=repair_candidates,
            repair_actions=list(llm_result.repair_actions or diagnosis.replan_actions),
            selected_repair_candidate_id=selected_repair_candidate_id,
        ),
        resume_from_step=selected_step,
        recovery_attempt=diagnosis.recovery_attempt,
        recovery_outcome=diagnosis.recovery_outcome,
        requires_doc_refresh=requires_doc_refresh,
        confidence=selected_confidence,
    )


def _normalize_candidates(
    candidates: List[RollbackCandidate],
    fallback: List[RollbackCandidate],
) -> List[RollbackCandidate]:
    normalized: List[RollbackCandidate] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate.step not in _ROLLBACK_STEPS:
            continue
        if candidate.step in seen:
            continue
        normalized.append(candidate)
        seen.add(candidate.step)
    return normalized or fallback


def _advisor_enabled(use_llm_advisor: bool | None) -> bool:
    if use_llm_advisor is not None:
        return use_llm_advisor
    value = os.getenv("INVERSE_DESIGN_ENABLE_LLM_DIAGNOSIS", "0").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _default_llm_call(prompt: str, sys_prompt: str, llm_model: str) -> str:
    from PhotonicsAI.Photon.llm_api import call_llm

    return call_llm(prompt, sys_prompt, llm_api_selection=llm_model)


def _extract_json_object(text: str) -> Dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if len(lines) >= 3:
            candidate = "\n".join(lines[1:-1])
    match = re.search(r"\{[\s\S]*\}", candidate)
    if match:
        candidate = match.group(0)
    return json.loads(candidate)


def _summarize_memory_entry(entry: WorkingMemoryEntry) -> str:
    key = f" [{entry.key}]" if entry.key else ""
    signature = f" {{{entry.failure_signature}}}" if entry.failure_signature else ""
    scenario = f" <{entry.scenario_fingerprint}>" if entry.scenario_fingerprint else ""
    return f"{entry.stage}{key}{signature}{scenario}: {entry.summary}"


def _normalize_text(text: str) -> str:
    lowered = text.lower()
    cleaned = re.sub(r"[^a-z0-9_\-\s]+", " ", lowered)
    return re.sub(r"\s+", " ", cleaned).strip()
