"""Step6 optimizer-hyperparameter attribution helpers.

This module is deliberately advisory.  It never changes the failure family or
rollback target; it only produces evidence that can help rank Step6
``RepairCandidate`` objects.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from PhotonicsAI.tools.inverse_design_types import (
    CheckpointReport,
    DiagnosisRepairAction,
    OptimizerAttributionReport,
    RepairCandidate,
    ScenarioFingerprint,
)
from PhotonicsAI.tools.inverse_design_working_memory import (
    InverseDesignWorkingMemory,
    WorkingMemoryEntry,
    get_inverse_design_working_memory,
)

FEATURE_NAMES = [
    "learning_rate",
    "beta",
    "penalty_weight",
    "pixel_size",
    "filter_radius",
]

DEFAULT_HINTS = {
    "learning_rate": 0.1,
    "beta": 6.0,
    "penalty_weight": 0.5,
    "pixel_size": 0.0443,
    "filter_radius": 0.1329,
}


def collect_optimizer_attribution_samples(
    *,
    run_result: Dict[str, Any] | None = None,
    recent_iterations: List[Dict[str, Any]] | None = None,
    checkpoint_report: CheckpointReport | Dict[str, Any] | None = None,
    scenario_fingerprint: ScenarioFingerprint | Dict[str, Any] | None = None,
    memory_store: InverseDesignWorkingMemory | None = None,
    csv_path: str | None = None,
) -> List[Dict[str, Any]]:
    """Collect optimizer-attribution samples from run data and memory.

    Long-term truth remains the run result, trace data, and working memory.  If
    ``csv_path`` is provided, CSV is only an audit export of the collected rows.
    """

    samples: List[Dict[str, Any]] = []
    typed_checkpoint = _coerce_checkpoint_report(checkpoint_report)
    scenario_key = _scenario_key(scenario_fingerprint)

    for iteration in recent_iterations or []:
        sample = _sample_from_iteration(iteration, typed_checkpoint=typed_checkpoint)
        if sample:
            sample.setdefault("scenario_fingerprint", scenario_key)
            samples.append(sample)

    if isinstance(run_result, dict):
        for iteration in run_result.get("iterations", []) or []:
            if not isinstance(iteration, dict):
                continue
            sample = _sample_from_iteration(iteration, typed_checkpoint=typed_checkpoint)
            if sample:
                sample.setdefault("scenario_fingerprint", scenario_key)
                samples.append(sample)

    memory = memory_store or get_inverse_design_working_memory()
    memory_entries = _recall_memory_entries(memory, scenario_key)
    for entry in memory_entries:
        sample = _sample_from_memory_entry(entry, scenario_key=scenario_key)
        if sample:
            samples.append(sample)

    deduped = _dedupe_samples(samples)
    if csv_path:
        _write_samples_csv(deduped, csv_path)
    return deduped


def analyze_optimizer_attribution(
    samples: List[Dict[str, Any]],
    *,
    min_samples: int = 12,
    random_state: int = 13,
    method: str = "random_forest_permutation",
) -> OptimizerAttributionReport:
    """Analyze hyperparameter importance when enough trustworthy samples exist."""

    rows = [_normalize_sample(row) for row in samples]
    rows = [row for row in rows if row is not None]
    if len(rows) < int(min_samples):
        return OptimizerAttributionReport(
            status="insufficient_samples",
            method=method,
            sample_count=len(rows),
            feature_names=list(FEATURE_NAMES),
            warnings=[f"Need at least {int(min_samples)} samples; got {len(rows)}."],
        )

    varied_features = [
        name for name in FEATURE_NAMES
        if len({round(float(row[name]), 12) for row in rows}) >= 2
    ]
    if len(varied_features) < 1:
        return OptimizerAttributionReport(
            status="insufficient_samples",
            method=method,
            sample_count=len(rows),
            feature_names=list(FEATURE_NAMES),
            warnings=["No optimizer hyperparameter has enough variation for attribution."],
        )

    y = [float(row["target"]) for row in rows]
    if len({round(value, 12) for value in y}) < 2:
        return OptimizerAttributionReport(
            status="insufficient_samples",
            method=method,
            sample_count=len(rows),
            feature_names=varied_features,
            warnings=["Attribution target is constant across samples."],
        )

    warnings = _correlation_warnings(rows, varied_features)
    if method == "correlation_fallback":
        return _correlation_report(rows, varied_features, warnings=warnings)

    try:
        import numpy as np
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.inspection import permutation_importance
    except Exception as exc:
        return OptimizerAttributionReport(
            status="unavailable",
            method=method,
            sample_count=len(rows),
            feature_names=varied_features,
            warnings=[f"scikit-learn attribution unavailable: {exc}"],
        )

    try:
        x = np.asarray([[float(row[name]) for name in varied_features] for row in rows], dtype=float)
        y_arr = np.asarray(y, dtype=float)
        model = RandomForestRegressor(
            n_estimators=64,
            min_samples_leaf=2,
            random_state=random_state,
        )
        model.fit(x, y_arr)
        result = permutation_importance(
            model,
            x,
            y_arr,
            n_repeats=8,
            random_state=random_state,
        )
        raw_scores = {
            name: max(0.0, float(score))
            for name, score in zip(varied_features, result.importances_mean)
        }
    except Exception as exc:
        return OptimizerAttributionReport(
            status="failed",
            method=method,
            sample_count=len(rows),
            feature_names=varied_features,
            warnings=[f"optimizer attribution failed: {exc}"],
        )

    scores = _normalize_scores(raw_scores)
    most_important = max(scores, key=scores.get) if scores else ""
    confidence = float(scores.get(most_important, 0.0)) if most_important else 0.0
    return OptimizerAttributionReport(
        status="ok",
        method=method,
        sample_count=len(rows),
        feature_names=varied_features,
        importance_scores=scores,
        most_important_param=most_important,
        confidence=round(confidence, 4),
        warnings=warnings,
        recommended_focus=_recommended_focus(scores),
    )


def build_attribution_guided_candidates(
    *,
    error_subtype: str,
    attribution_report: OptimizerAttributionReport,
    objective_goal: str,
) -> List[RepairCandidate]:
    """Build advisory repair candidates from an attribution report."""

    if attribution_report.status != "ok":
        return []
    focus = attribution_report.recommended_focus or [attribution_report.most_important_param]
    focus = [item for item in focus if item in FEATURE_NAMES]
    if not focus:
        return []

    actions = _subtype_patch_actions(error_subtype)
    if not actions:
        return []

    most_important = attribution_report.most_important_param
    confidence = min(0.9, max(0.52, 0.58 + 0.25 * attribution_report.confidence))
    if most_important in {"learning_rate", "beta", "penalty_weight"}:
        confidence = min(0.94, confidence + 0.06)

    metadata = {
        "attribution_status": attribution_report.status,
        "method": attribution_report.method,
        "sample_count": attribution_report.sample_count,
        "most_important_param": most_important,
        "importance_scores": dict(attribution_report.importance_scores),
        "recommended_focus": list(focus),
        "warnings": list(attribution_report.warnings),
    }
    return [
        RepairCandidate(
            candidate_id=f"step3_attribution_{error_subtype or 'optimization'}",
            target_step="step3",
            confidence=round(confidence, 3),
            patch_actions=actions,
            expected_effect=(
                "Use sample-backed optimizer attribution to choose a subtype-specific "
                "hyperparameter recovery patch."
            ),
            risk="medium" if attribution_report.confidence < 0.5 else "low",
            rationale=(
                f"Attribution focus={most_important or 'unknown'} for subtype "
                f"{error_subtype or 'unknown'}; failure diagnosis remains authoritative."
            ),
            analysis_metadata=metadata,
        )
    ]


def _subtype_patch_actions(error_subtype: str) -> List[DiagnosisRepairAction]:
    subtype = str(error_subtype or "").strip().lower()
    if subtype == "objective_regression":
        return [
            _set("optimization_config.optimizer_hints.learning_rate", 0.03, "Reduce step size after FOM regression."),
            _set("optimization_config.optimizer_hints.beta", 10.0, "Use only mild projection pressure during regression recovery."),
            _set("optimization_config.optimizer_hints.penalty_weight", 0.5, "Avoid adding extra penalty pressure while restoring FOM direction."),
            _set("optimization_config.termination.max_iterations", 40, "Allow a bounded recovery run."),
        ]
    if subtype == "objective_direction":
        return [
            _set("optimization_config.optimizer_hints.learning_rate", 0.03, "Reduce step size to correct wrong-direction updates."),
            _set("optimization_config.optimizer_hints.beta", 6.0, "Keep projection pressure neutral until objective direction recovers."),
            _set("optimization_config.optimizer_hints.penalty_weight", 0.5, "Keep penalty pressure neutral while correcting objective direction."),
            _set("optimization_config.termination.max_iterations", 40, "Allow a bounded recovery run."),
        ]
    if subtype == "oscillation":
        return [
            _set("optimization_config.optimizer_hints.learning_rate", 0.05, "Lower learning rate to damp objective oscillation."),
            _set("optimization_config.optimizer_hints.beta", 8.0, "Increase projection pressure gently rather than jumping to beta=30."),
            _set("optimization_config.optimizer_hints.penalty_weight", 0.6, "Add mild regularization to reduce oscillatory updates."),
            _set("optimization_config.termination.max_iterations", 40, "Allow a bounded recovery run."),
        ]
    if subtype == "stalled_convergence":
        return [
            _set("optimization_config.optimizer_hints.learning_rate", 0.08, "Keep enough step size to escape a plateau."),
            _set("optimization_config.optimizer_hints.beta", 10.0, "Add moderate projection pressure to change the design landscape."),
            _set("optimization_config.optimizer_hints.penalty_weight", 0.6, "Add mild manufacturability pressure during plateau recovery."),
            _set("optimization_config.termination.max_iterations", 60, "Give plateau recovery more room."),
            _set("optimization_config.termination.patience", 5, "Avoid premature stop during plateau recovery."),
        ]
    return []


def _set(path: str, value: Any, reason: str) -> DiagnosisRepairAction:
    return DiagnosisRepairAction(action="set_value", path=path, value=value, reason=reason)


def _coerce_checkpoint_report(
    checkpoint_report: CheckpointReport | Dict[str, Any] | None,
) -> CheckpointReport | None:
    if checkpoint_report is None:
        return None
    if isinstance(checkpoint_report, CheckpointReport):
        return checkpoint_report
    try:
        return CheckpointReport.model_validate(checkpoint_report)
    except Exception:
        return None


def _scenario_key(scenario_fingerprint: ScenarioFingerprint | Dict[str, Any] | None) -> str:
    if scenario_fingerprint is None:
        return ""
    if isinstance(scenario_fingerprint, dict):
        try:
            scenario_fingerprint = ScenarioFingerprint.model_validate(scenario_fingerprint)
        except Exception:
            return ""
    return "|".join(
        [
            scenario_fingerprint.component_type,
            scenario_fingerprint.objective_metric,
            scenario_fingerprint.objective_goal,
            scenario_fingerprint.wavelength_band,
            scenario_fingerprint.domain_ratio,
            scenario_fingerprint.monitor_topology_signature,
            scenario_fingerprint.boundary_type,
        ]
    ).strip("|")


def _sample_from_iteration(
    iteration: Dict[str, Any],
    *,
    typed_checkpoint: CheckpointReport | None,
) -> Dict[str, Any] | None:
    hints = _extract_optimizer_hints(iteration)
    target = _first_float(
        iteration.get("short_run_score"),
        iteration.get("fom_delta"),
        iteration.get("fom_value"),
        iteration.get("objective_value"),
        iteration.get("score"),
        _nested(iteration, "metrics", "adjoint_trace", "post_process_val"),
        _nested(iteration, "metrics", "adjoint_trace", "objective_fn_val"),
    )
    if target is None:
        return None
    sample = dict(hints)
    sample.update(
        {
            "target": target,
            "fom_value": target,
            "fom_delta": 0.0 if typed_checkpoint is None else typed_checkpoint.objective_delta,
            "source": "iteration",
        }
    )
    if typed_checkpoint is not None:
        sample.update(
            {
                "checkpoint_iteration": typed_checkpoint.checkpoint_iteration,
                "error_family": typed_checkpoint.error_family,
                "error_subtype": typed_checkpoint.error_subtype,
                "oscillation_ratio": typed_checkpoint.oscillation_ratio,
                "parameter_update_norm": typed_checkpoint.parameter_update_norm,
                "manufacturability_score": typed_checkpoint.manufacturability_score,
                "observability_score": typed_checkpoint.observability_score,
            }
        )
    return sample


def _sample_from_memory_entry(
    entry: WorkingMemoryEntry,
    *,
    scenario_key: str,
) -> Dict[str, Any] | None:
    metadata = entry.metadata or {}
    target = _first_float(
        metadata.get("short_run_score"),
        metadata.get("fom_delta"),
        metadata.get("fom_value"),
    )
    if target is None:
        return None
    hints = dict(DEFAULT_HINTS)
    hints.update(_hints_from_patch_actions(metadata.get("patch_actions") or metadata.get("replan_actions") or []))
    sample = dict(hints)
    sample.update(
        {
            "target": target,
            "short_run_score": target,
            "scenario_fingerprint": entry.scenario_fingerprint or scenario_key,
            "failure_signature": entry.failure_signature,
            "candidate_id": str(metadata.get("candidate_id", "")),
            "outcome": str(metadata.get("recovery_outcome", metadata.get("outcome", ""))),
            "patch_signature": _patch_signature(metadata.get("patch_actions") or []),
            "source": f"memory:{entry.stage}",
        }
    )
    return sample


def _extract_optimizer_hints(payload: Dict[str, Any]) -> Dict[str, float]:
    hints = dict(DEFAULT_HINTS)
    raw_hints = payload.get("optimizer_hints")
    if not isinstance(raw_hints, dict):
        raw_hints = _nested(payload, "metrics", "optimizer_hints")
    if isinstance(raw_hints, dict):
        for name in FEATURE_NAMES:
            value = _as_float(raw_hints.get(name))
            if value is not None:
                hints[name] = value
    hints.update(_hints_from_patch_actions(payload.get("patch_actions") or []))
    return hints


def _hints_from_patch_actions(actions: Any) -> Dict[str, float]:
    hints: Dict[str, float] = {}
    if not isinstance(actions, list):
        return hints
    for action in actions:
        if hasattr(action, "model_dump"):
            action = action.model_dump()
        if not isinstance(action, dict):
            continue
        path = str(action.get("path", ""))
        name = path.rsplit(".", 1)[-1]
        if name not in FEATURE_NAMES:
            continue
        value = _as_float(action.get("value"))
        if value is not None:
            hints[name] = value
    return hints


def _normalize_sample(row: Dict[str, Any]) -> Dict[str, float] | None:
    target = _first_float(row.get("short_run_score"), row.get("fom_delta"), row.get("fom_value"), row.get("target"))
    if target is None:
        return None
    normalized: Dict[str, float] = {"target": target}
    for name in FEATURE_NAMES:
        value = _as_float(row.get(name))
        if value is None:
            value = DEFAULT_HINTS[name]
        normalized[name] = value
    return normalized


def _correlation_report(
    rows: List[Dict[str, float]],
    feature_names: List[str],
    *,
    warnings: List[str],
) -> OptimizerAttributionReport:
    y = [float(row["target"]) for row in rows]
    raw_scores = {
        name: abs(_pearson([float(row[name]) for row in rows], y))
        for name in feature_names
    }
    scores = _normalize_scores(raw_scores)
    most_important = max(scores, key=scores.get) if scores else ""
    return OptimizerAttributionReport(
        status="ok",
        method="correlation_fallback",
        sample_count=len(rows),
        feature_names=feature_names,
        importance_scores=scores,
        most_important_param=most_important,
        confidence=round(float(scores.get(most_important, 0.0)), 4) if most_important else 0.0,
        warnings=warnings,
        recommended_focus=_recommended_focus(scores),
    )


def _correlation_warnings(rows: List[Dict[str, float]], feature_names: List[str]) -> List[str]:
    warnings: List[str] = []
    for i, left in enumerate(feature_names):
        for right in feature_names[i + 1:]:
            corr = _pearson([float(row[left]) for row in rows], [float(row[right]) for row in rows])
            if abs(corr) > 0.85:
                warnings.append(f"High hyperparameter correlation: {left} vs {right} r={corr:.3f}.")
    return warnings


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    left_var = sum((a - left_mean) ** 2 for a in left)
    right_var = sum((b - right_mean) ** 2 for b in right)
    denom = math.sqrt(left_var * right_var)
    if denom <= 0:
        return 0.0
    return numerator / denom


def _normalize_scores(raw_scores: Dict[str, float]) -> Dict[str, float]:
    total = sum(max(0.0, float(value)) for value in raw_scores.values())
    if total <= 0:
        return {name: 0.0 for name in raw_scores}
    return {
        name: round(max(0.0, float(value)) / total, 6)
        for name, value in raw_scores.items()
    }


def _recommended_focus(scores: Dict[str, float]) -> List[str]:
    return [
        name for name, value in sorted(scores.items(), key=lambda item: item[1], reverse=True)
        if value > 0
    ][:3]


def _recall_memory_entries(memory: InverseDesignWorkingMemory, scenario_key: str) -> List[WorkingMemoryEntry]:
    entries: List[WorkingMemoryEntry] = []
    if scenario_key:
        entries.extend(memory.recall_by_scenario_fingerprint(scenario_key, stage="step6_repair_trial", limit=50))
        entries.extend(memory.recall_by_scenario_fingerprint(scenario_key, stage="recovery", limit=50))
    if not entries:
        entries.extend(memory.list_entries(stage="step6_repair_trial", limit=50))
        entries.extend(memory.list_entries(stage="recovery", limit=50))
    return entries


def _dedupe_samples(samples: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for sample in samples:
        key = "|".join(
            [
                str(sample.get("source", "")),
                str(sample.get("candidate_id", "")),
                str(sample.get("patch_signature", "")),
                str(sample.get("target", sample.get("short_run_score", sample.get("fom_value", "")))),
            ]
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(sample)
    return deduped


def _write_samples_csv(samples: List[Dict[str, Any]], csv_path: str) -> None:
    if not csv_path:
        return
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "run_tag",
        "scenario_fingerprint",
        "component_type",
        "objective_metric",
        "objective_goal",
        "iteration",
        "checkpoint_iteration",
        "error_family",
        "error_subtype",
        *FEATURE_NAMES,
        "fom_value",
        "fom_delta",
        "oscillation_ratio",
        "parameter_update_norm",
        "manufacturability_score",
        "observability_score",
        "patch_signature",
        "candidate_id",
        "short_run_score",
        "outcome",
        "source",
    ]
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for sample in samples:
            writer.writerow(sample)


def _patch_signature(actions: Any) -> str:
    if not isinstance(actions, list):
        return ""
    parts: List[str] = []
    for action in actions:
        if hasattr(action, "model_dump"):
            action = action.model_dump()
        if not isinstance(action, dict):
            continue
        parts.append(f"{action.get('action', '')}:{action.get('path', '')}:{action.get('value', '')}")
    return "|".join(sorted(parts))


def _nested(payload: Dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _first_float(*values: Any) -> float | None:
    for value in values:
        numeric = _as_float(value)
        if numeric is not None:
            return numeric
    return None


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, str):
        try:
            numeric = float(value)
        except ValueError:
            return None
        return numeric if math.isfinite(numeric) else None
    return None
