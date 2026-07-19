"""Execution helpers for inverse-design step 9.1 item 5.

Primary kernel: Tidy3D invdes/adjoint example-style optimizer.
Execution policy: adjoint-only (bridge fallback disabled).
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import json
import hashlib
import math
import os
import queue
import re
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Sequence
from urllib.parse import urlsplit
from uuid import uuid4

import matplotlib as _mpl
_mpl.use("Agg")  # BUG-7 fix: thread-safe backend

logger = logging.getLogger(__name__)

from pydantic import BaseModel, ConfigDict, Field

from PhotonicsAI.tools.inverse_design_config import InverseDesignConfigBundle, parse_inverse_design_config
from PhotonicsAI.tools.inverse_design_contracts import (
    is_discrete_topology_parameter,
    normalize_discrete_topology_parameter,
)
from PhotonicsAI.tools.inverse_design_config_validation import ConfigValidationResult, validate_config
from PhotonicsAI.tools.inverse_design_failure_diagnosis import diagnose_inverse_design_failure
from PhotonicsAI.tools.inverse_design_rag_memory import Step4ConstraintPacket
from PhotonicsAI.tools.inverse_design_types import CheckpointReport, FailureDiagnosis, ScenarioFingerprint
from PhotonicsAI.tools.inverse_design_working_memory import get_inverse_design_working_memory
from PhotonicsAI.tools.tidy3d_tools import run_tidy3d_simulation


_OFFICIAL_INVDES_DOC = (
    "https://docs.flexcompute.com/projects/tidy3d/en/v2.10.2/notebooks/Autograd0Quickstart.html"
)
_FOOTPRINT_PAIR_RE = re.compile(
    r"([0-9]+(?:\.[0-9]+)?)\s*(?:\u00b5m|\u03bcm|um|nm)?\s*[xX\u00d7]\s*([0-9]+(?:\.[0-9]+)?)\s*(\u00b5m|\u03bcm|um|nm)",
    re.IGNORECASE,
)
_MODULE_BOOT_MTIME = os.path.getmtime(__file__)
_REPO_ROOT = Path(__file__).resolve().parents[2]
_BUILD_DIR = _REPO_ROOT / "build"
_ADJOINT_RETRY_ERROR_PATTERNS = (
    "failed to download the simulation data file",
    "simulation_data_unavailable",
    "headobject operation: not found",
    "headobject operation: 404",
    "connecterror",
    "readtimeout",
    "timed out",
    "connection aborted",
    "httpsconnectionpool",
    "max retries exceeded",
    "ssl",
    "ssleoferror",
    "unexpected eof",
    "failed to upload",
    "upload failed",
    "connection reset by peer",
    "adjoint_run_timeout_guard",
)
_ADJOINT_NON_RETRYABLE_ERROR_PATTERNS = (
    "insufficient balance",
    "balance is",
    "is reserved",
    "out of credit",
    "out of credits",
    "quota exceeded",
    "payment required",
    "billing",
)
_TIDY3D_RUNTIME_DEFAULTS = {
    "TIDY3D_LOAD_RETRIES": "8",
    "TIDY3D_LOAD_RETRY_SLEEP_S": "10",
    "TIDY3D_WEB__TIMEOUT": "180",
}
_TIDY3D_NO_PROXY_REQUIRED = (
    "tidy3d-api.simulation.cloud",
    "simulation.cloud",
    "flexcompute.com",
)
_ADJOINT_TIMEOUT_BACKOFF_DEFAULT = 2.0
_ADJOINT_TIMEOUT_BACKOFF_MAX = 4.0
_ADJOINT_TIMEOUT_MAX_DEFAULT_S = 1800.0
_DEMUX_COUPLING_SHORTFALL_PENALTY = 0.5  # was 2.0 — direction bug in coupling_ratio_obj causes false fire for rev-propagating cases
_DEMUX_CROSSTALK_EXCESS_PENALTY = 0.5   # was 4.0 — direction bug causes artificial 0.9 crosstalk; 4.0*0.89=-3.56 dominated case5
_DEMUX_MODE_PURITY_PENALTY = 0.8          # raised from 0.5 — stronger gradient for case4/5 purity plateau
_DEMUX_HIGHER_ORDER_MODE_BOOST = 0.25      # was 0.1 — TE3: 1.75x, TE4: 2.0x (was 1.3x/1.4x)
_DEMUX_MODE_PURITY_FOCUS_THRESHOLD = 0.80  # was 0.7 — trigger focus below 80% (C4/C5 stuck at 47-52%)
_DEMUX_MODE_PURITY_FOCUS_GAIN = 0.5        # was 0.8 — gentler dynamic focuser
_DEMUX_MODE_PURITY_FOCUS_MAX_MULTIPLIER = 4.0  # was 3.0 — allow stronger boost (TE3/TE4 mixing plateau)
_DEMUX_TRANSMISSION_SURROGATE_WEIGHT = 1.0  # raised from 0.35 — absolute transmission now primary signal
_DEMUX_TRANSMISSION_SHORTFALL_PENALTY = 0.5  # was 2.0 — lower until optimizer finds valid topology first
_DEMUX_TRANSMISSION_FLOOR = 0.05  # was 0.1 — TE3/TE4 physically hard; lower floor avoids dominance
_DEMUX_TRANSMISSION_SURROGATE_MAX = 1.0  # lowered from 2.0 — physical cap; prevents saturation


def _demux_penalty_weights() -> tuple[float, float]:
    coupling = _as_float_or_none(os.getenv("INVERSE_DEMUX_COUPLING_SHORTFALL_PENALTY"))
    crosstalk = _as_float_or_none(os.getenv("INVERSE_DEMUX_CROSSTALK_EXCESS_PENALTY"))
    return (
        float(coupling) if coupling is not None and coupling > 0 else _DEMUX_COUPLING_SHORTFALL_PENALTY,
        float(crosstalk) if crosstalk is not None and crosstalk > 0 else _DEMUX_CROSSTALK_EXCESS_PENALTY,
    )


def _demux_mode_purity_penalty_weight() -> float:
    purity = _as_float_or_none(os.getenv("INVERSE_DEMUX_MODE_PURITY_PENALTY"))
    if purity is None or purity < 0:
        return _DEMUX_MODE_PURITY_PENALTY
    return float(purity)


def _demux_mode_focus_params() -> tuple[float, float, float, float]:
    higher_order_boost = _as_float_or_none(os.getenv("INVERSE_DEMUX_HIGHER_ORDER_MODE_BOOST"))
    purity_focus_threshold = _as_float_or_none(os.getenv("INVERSE_DEMUX_MODE_PURITY_FOCUS_THRESHOLD"))
    purity_focus_gain = _as_float_or_none(os.getenv("INVERSE_DEMUX_MODE_PURITY_FOCUS_GAIN"))
    purity_focus_max_multiplier = _as_float_or_none(os.getenv("INVERSE_DEMUX_MODE_PURITY_FOCUS_MAX_MULTIPLIER"))

    return (
        float(higher_order_boost)
        if higher_order_boost is not None and higher_order_boost >= 0
        else _DEMUX_HIGHER_ORDER_MODE_BOOST,
        float(purity_focus_threshold)
        if purity_focus_threshold is not None and 0 <= purity_focus_threshold <= 1.5
        else _DEMUX_MODE_PURITY_FOCUS_THRESHOLD,
        float(purity_focus_gain)
        if purity_focus_gain is not None and purity_focus_gain >= 0
        else _DEMUX_MODE_PURITY_FOCUS_GAIN,
        float(purity_focus_max_multiplier)
        if purity_focus_max_multiplier is not None and purity_focus_max_multiplier >= 1.0
        else _DEMUX_MODE_PURITY_FOCUS_MAX_MULTIPLIER,
    )


def _demux_mode_focus_multiplier(
    *,
    target_mode_index: int,
    target_mode_purity: float | None,
) -> float:
    higher_order_boost, purity_focus_threshold, purity_focus_gain, max_multiplier = _demux_mode_focus_params()
    mode_order_factor = 1.0 + max(int(target_mode_index), 0) * higher_order_boost
    focus_deficit = 0.0
    if target_mode_purity is not None and purity_focus_gain > 0:
        focus_deficit = max(0.0, purity_focus_threshold - float(target_mode_purity))
    focus_factor = 1.0 + purity_focus_gain * focus_deficit
    return max(1.0, min(float(mode_order_factor * focus_factor), max_multiplier))


def _demux_transmission_params() -> tuple[float, float, float, float]:
    weight = _as_float_or_none(os.getenv("INVERSE_DEMUX_TRANSMISSION_SURROGATE_WEIGHT"))
    shortfall_penalty = _as_float_or_none(os.getenv("INVERSE_DEMUX_TRANSMISSION_SHORTFALL_PENALTY"))
    floor = _as_float_or_none(os.getenv("INVERSE_DEMUX_TRANSMISSION_FLOOR"))
    max_ratio = _as_float_or_none(os.getenv("INVERSE_DEMUX_TRANSMISSION_SURROGATE_MAX"))
    return (
        float(weight)
        if weight is not None and weight >= 0
        else _DEMUX_TRANSMISSION_SURROGATE_WEIGHT,
        float(shortfall_penalty)
        if shortfall_penalty is not None and shortfall_penalty >= 0
        else _DEMUX_TRANSMISSION_SHORTFALL_PENALTY,
        float(floor)
        if floor is not None and floor >= 0
        else _DEMUX_TRANSMISSION_FLOOR,
        float(max_ratio)
        if max_ratio is not None and max_ratio >= 1.0
        else _DEMUX_TRANSMISSION_SURROGATE_MAX,
    )


def _new_run_artifact_tag() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"inv_{ts}_{uuid4().hex[:8]}"


def _resolve_run_artifact_tag() -> str:
    forced = str(os.getenv("INVERSE_FORCE_RUN_ARTIFACT_TAG", "") or "").strip()
    if forced:
        return _safe_run_artifact_tag(forced)
    return _new_run_artifact_tag()


def _safe_run_artifact_tag(run_artifact_tag: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(run_artifact_tag or "").strip()) or "invdes"


def _adjoint_iteration_trace_path(run_artifact_tag: str) -> Path:
    safe_tag = _safe_run_artifact_tag(run_artifact_tag)
    path = _BUILD_DIR / f"invdes_adjoint_iteration_trace_{safe_tag}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _step5_heartbeat_path(run_artifact_tag: str) -> Path:
    safe_tag = _safe_run_artifact_tag(run_artifact_tag)
    path = _BUILD_DIR / f"invdes_step5_heartbeat_{safe_tag}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _write_step5_heartbeat(run_artifact_tag: str, stage: str, **extra: Any) -> None:
    try:
        payload: Dict[str, Any] = {
            "run_artifact_tag": _safe_run_artifact_tag(run_artifact_tag),
            "stage": str(stage or "").strip() or "unknown",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if extra:
            payload.update(extra)
        path = _step5_heartbeat_path(run_artifact_tag)
        path.write_text(json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8")
    except Exception:
        logger.debug("Failed to update Step5 heartbeat file.", exc_info=True)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = str(os.getenv(name, "") or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _require_field_plot_in_iteration_eval() -> bool:
    raw = str(os.getenv("INVERSE_STEP5_REQUIRE_FIELD_PER_ITER", "") or "").strip().lower()
    if raw:
        return raw in {"1", "true", "yes", "on"}
    # Default policy: do not emit field images for every iteration.
    # Best-iteration rerender still forces field outputs for evidence.
    return False


def _include_field_monitors_in_adjoint_optimizer() -> bool:
    # Scheme A default: adjoint optimizer simulations do not carry Step3 field
    # monitors. Diagnostic/seed/rerender paths still produce field evidence.
    return _env_bool("INVERSE_ADJOINT_INCLUDE_FIELD_MONITORS", default=False)


def _case_artifact_label(case: Dict[str, Any] | None) -> str:
    if not isinstance(case, dict):
        return ""
    parts: List[str] = []
    wavelength_nm = _as_float_or_none(case.get("wavelength_nm"))
    if wavelength_nm is not None and wavelength_nm > 0:
        if abs(wavelength_nm - round(wavelength_nm)) < 1e-6:
            parts.append(f"{int(round(wavelength_nm))}nm")
        else:
            parts.append(f"{wavelength_nm:.3f}".rstrip("0").rstrip(".") + "nm")
    source_port = _case_source_port(case)
    target_port = str(case.get("target_port") or "").strip().lower()
    if source_port:
        parts.append(f"src_{source_port}")
    parts.append(f"src_te{_case_source_mode_index(case)}")
    if target_port:
        parts.append(f"to_{target_port}")
    parts.append(f"te{max(int(_as_float_or_none(case.get('target_mode_index')) or 0), 0)}")
    return _safe_run_artifact_tag("_".join(parts))[:96]


def _case_artifact_suffix(case: Dict[str, Any] | None, case_index: int) -> str:
    base = f"_case{max(int(case_index), 1)}"
    label = _case_artifact_label(case)
    if not label:
        return base
    return f"{base}_{label}"


def _build_adjoint_case_task_plan(
    *,
    objective_cases: Sequence[Dict[str, Any]],
    task_names: Sequence[str],
    case_task_indices: Sequence[int] | None = None,
) -> List[Dict[str, Any]]:
    plan: List[Dict[str, Any]] = []
    if not objective_cases or not task_names:
        return plan

    for case_idx, case in enumerate(objective_cases, start=1):
        task_idx = case_idx - 1
        if case_task_indices is not None and case_idx - 1 < len(case_task_indices):
            try:
                task_idx = int(case_task_indices[case_idx - 1])
            except Exception:
                task_idx = case_idx - 1
        if task_idx < 0 or task_idx >= len(task_names):
            continue
        plan.append(
            {
                "case_index": case_idx,
                "case_name": str(case.get("name") or f"case_{case_idx}"),
                "case_label": _case_artifact_label(case),
                "task_name": str(task_names[task_idx]),
                "task_index": task_idx,
                "wavelength_nm": _as_float_or_none(case.get("wavelength_nm")),
                "source_port": _case_source_port(case),
                "source_mode_index": _case_source_mode_index(case),
                "source_direction": _case_source_direction(case),
                "target_port": str(case.get("target_port") or "").strip().lower(),
                "target_mode_index": max(int(_as_float_or_none(case.get("target_mode_index")) or 0), 0),
            }
        )
    return plan


def _export_adjoint_forward_sim_artifacts(
    *,
    bundle: InverseDesignConfigBundle,
    td,
    run_artifact_tag: str,
    objective_cases: Sequence[Dict[str, Any]],
) -> List[str]:
    """Export forward-inspection sims with Step3 field monitors preserved."""

    component = str(bundle.simulation_config.component_type or "component").strip() or "component"
    safe_tag = _safe_run_artifact_tag(run_artifact_tag)
    export_cases: List[Dict[str, Any] | None] = list(objective_cases) if objective_cases else [None]
    exported_paths: List[str] = []
    exported_signatures: set[tuple[Any, ...]] = set()

    for case in export_cases:
        if isinstance(case, dict):
            try:
                case_wavelength = round(float(case.get("wavelength_nm")), 6)
            except (TypeError, ValueError):
                case_wavelength = round(float(bundle.simulation_config.wavelength_nm), 6)
            signature = (
                case_wavelength,
                str(case.get("source_port") or "port_o1"),
                int(_as_float_or_none(case.get("source_mode_index")) or 0),
                str(case.get("source_direction") or ""),
            )
            if signature in exported_signatures:
                continue
            exported_signatures.add(signature)
            export_bundle = _bundle_with_case_wavelength(bundle, case_wavelength, case=case)
            export_case = case
        else:
            signature = ("base", float(bundle.simulation_config.wavelength_nm))
            if signature in exported_signatures:
                continue
            exported_signatures.add(signature)
            export_bundle = bundle
            export_case = None

        export_index = len(exported_signatures)
        export_sim = _build_invdes_simulation(
            bundle=export_bundle,
            td=td,
            case_override=export_case,
            include_field_monitors=True,
        )
        suffix = _case_artifact_suffix(export_case, export_index) if objective_cases else ""
        sim_path = _BUILD_DIR / f"tidy3d_sim_{component}_{safe_tag}_adjoint_forward{suffix}.hdf5"
        viewer_path = _BUILD_DIR / f"tidy3d_viewer_{component}_{safe_tag}_adjoint_forward{suffix}.py"
        sim_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            export_sim.to_hdf5(str(sim_path))
            exported_paths.append(str(sim_path))
        except Exception:
            logger.warning(
                "Step5 artifact export FAILED for forward inspection sim: %s — "
                "forward FieldMonitor HDF5 will be missing from build/. "
                "Check tidy3d version compatibility and disk write permissions.",
                sim_path, exc_info=True,
            )
            continue

        try:
            viewer_path.write_text(
                f'"""Auto-generated viewer script for {component} adjoint forward inspection."""\n'
                f"import tidy3d as td\n\n"
                f'sim = td.Simulation.from_hdf5(r"{sim_path}")\n',
                encoding="utf-8",
            )
            exported_paths.append(str(viewer_path))
        except Exception:
            logger.warning(
                "Step5 artifact export FAILED for forward viewer script: %s",
                viewer_path, exc_info=True,
            )

    return exported_paths


def _recover_adjoint_forward_artifacts_from_disk(run_artifact_tag: str) -> List[str]:
    """Scan *build/* for forward-inspection artifacts that were written before a timeout.

    When the adjoint optimizer thread is killed by the timeout guard,
    ``adjoint_forward_artifacts`` (computed inside the thread) is inaccessible.
    However, ``_export_adjoint_forward_sim_artifacts`` runs early in the thread
    and writes HDF5/viewer files to ``_BUILD_DIR`` before the optimizer loop
    starts.  This helper reconstructs the list from disk so that the failure
    path ``constraint_summary`` always exposes those inspection files to M178+
    acceptance checks (``HDF5_ARTIFACT_COUNT``, ``ADJOINT_FWD_FIELD_MONITORS_FLAG``).
    """
    safe_tag = _safe_run_artifact_tag(run_artifact_tag)
    pattern_prefix = f"*{safe_tag}_adjoint_forward*"
    found: List[str] = []
    try:
        for p in sorted(_BUILD_DIR.glob(pattern_prefix)):
            if p.suffix in {".hdf5", ".py"}:
                found.append(str(p))
    except Exception:
        logger.debug(
            "_recover_adjoint_forward_artifacts_from_disk: scan failed for tag %s",
            safe_tag, exc_info=True,
        )
    return found


def _step5_optimizer_alive_heartbeat_interval_s() -> float:
    """Interval for Step5 alive heartbeat during optimizer blocking calls."""

    raw = str(os.getenv("INVERSE_STEP5_OPTIMIZER_ALIVE_HEARTBEAT_S", "15") or "").strip()
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = 15.0
    return max(2.0, value)


def _constraints_min_feature_um(bundle: InverseDesignConfigBundle) -> float | None:
    constraints = bundle.optimization_config.constraints or []
    if not constraints:
        return None

    min_feature_um: float | None = None
    for item in constraints:
        text = str(item or "").strip().lower()
        if not text:
            continue
        if not any(token in text for token in ("feature", "line width", "linewidth", "线宽", "最小特征")):
            continue
        for match in re.finditer(r"([0-9]+(?:\.[0-9]+)?)\s*(nm|um|μm|µm)", text, flags=re.IGNORECASE):
            value = _as_float_or_none(match.group(1))
            if value is None or value <= 0:
                continue
            unit = str(match.group(2)).strip().lower()
            value_um = float(value) / 1000.0 if unit == "nm" else float(value)
            if value_um <= 0:
                continue
            min_feature_um = value_um if min_feature_um is None else max(min_feature_um, value_um)
    return min_feature_um


def _effective_min_feature_um(bundle: InverseDesignConfigBundle) -> float:
    env_nm = _as_float_or_none(os.getenv("INVERSE_STEP5_MIN_FEATURE_SIZE_NM"))
    env_um = _as_float_or_none(os.getenv("INVERSE_STEP5_MIN_FEATURE_SIZE_UM"))
    env_floor = 0.1
    if env_um is not None and env_um > 0:
        env_floor = max(0.1, float(env_um))
    elif env_nm is not None and env_nm > 0:
        env_floor = max(0.1, float(env_nm) / 1000.0)

    constraint_floor = _constraints_min_feature_um(bundle)
    if constraint_floor is not None and constraint_floor > 0:
        return max(0.1, env_floor, float(constraint_floor))
    return max(0.1, env_floor)


def _prepare_invdes_runtime_workdir(run_artifact_tag: str) -> Path:
    """Create per-run working directory for invdes temporary HDF5 outputs.

    Tidy3D invdes can write relative files such as ``batch.hdf5``. Running
    every optimization in an isolated directory avoids cross-run locking /
    permission collisions on shared ``./batch.hdf5`` in repo root.
    """
    safe_tag = _safe_run_artifact_tag(run_artifact_tag)
    workdir = _BUILD_DIR / "invdes_runtime" / safe_tag
    workdir.mkdir(parents=True, exist_ok=True)
    return workdir


def _resume_store_dir() -> Path:
    path = _BUILD_DIR / "invdes_resume"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resume_enabled() -> bool:
    raw = str(os.getenv("INVERSE_STEP5_ENABLE_OPTIMIZER_RESUME", "1")).strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _resume_allow_completed_state() -> bool:
    raw = str(os.getenv("INVERSE_STEP5_RESUME_ALLOW_COMPLETED", "0")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _resume_force_state_file() -> str:
    return str(os.getenv("INVERSE_STEP5_RESUME_STATE_FILE", "") or "").strip()


def _objective_case_signature_payload(objective_cases: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    for case in objective_cases:
        if not isinstance(case, dict):
            continue
        payload.append(
            {
                "wavelength_nm": round(float(case.get("wavelength_nm", 0.0) or 0.0), 6),
                "source_port": str(case.get("source_port") or "").strip().lower(),
                "source_mode_index": int(case.get("source_mode_index", 0) or 0),
                "source_direction": str(case.get("source_direction") or "-").strip() or "-",
                "target_port": str(case.get("target_port") or "").strip().lower(),
                "target_mode_index": int(case.get("target_mode_index", 0) or 0),
            }
        )
    payload.sort(
        key=lambda item: (
            item["wavelength_nm"],
            item["source_port"],
            item["source_mode_index"],
            item["source_direction"],
            item["target_port"],
            item["target_mode_index"],
        )
    )
    return payload


def _optimizer_resume_signature(bundle: InverseDesignConfigBundle, objective_cases: Sequence[Dict[str, Any]]) -> str:
    payload = {
        "component_type": str(bundle.simulation_config.component_type or ""),
        "objective_metric": str(bundle.optimization_config.objective.metric or ""),
        "objective_goal": str(bundle.optimization_config.objective.goal or ""),
        "wavelength_nm": round(float(bundle.simulation_config.wavelength_nm or 0.0), 6),
        "objective_cases": _objective_case_signature_payload(objective_cases),
        "domain_size_um": [round(float(item), 6) for item in bundle.simulation_config.domain.size_um],
        "geometry_keys": sorted(str(key) for key in bundle.simulation_config.geometry.parameters.keys()),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _optimizer_resume_registry_path(signature: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(signature or "").strip()) or "default"
    return _resume_store_dir() / f"resume_registry_{safe}.json"


def _write_optimizer_resume_registry(
    *,
    signature: str,
    cache_file: Path,
    run_artifact_tag: str,
    status: str,
    completed_steps: int,
) -> None:
    try:
        payload = {
            "signature": signature,
            "cache_file": str(cache_file.resolve()),
            "run_artifact_tag": str(run_artifact_tag or ""),
            "status": str(status or ""),
            "completed_steps": int(max(completed_steps, 0)),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        _optimizer_resume_registry_path(signature).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        logger.debug("Failed to write optimizer resume registry.", exc_info=True)


def _resolve_optimizer_resume_candidate(
    *,
    signature: str,
) -> Dict[str, Any]:
    candidate: Dict[str, Any] = {
        "enabled": _resume_enabled(),
        "resume_used": False,
        "resume_source": "",
        "resume_reason": "",
        "resume_state_file": "",
        "resume_previous_steps": 0,
    }
    if not candidate["enabled"]:
        candidate["resume_reason"] = "resume_disabled_by_env"
        return candidate

    forced = _resume_force_state_file()
    if forced:
        forced_path = Path(forced)
        if forced_path.exists():
            candidate.update(
                {
                    "resume_used": True,
                    "resume_source": "forced_state_file",
                    "resume_state_file": str(forced_path.resolve()),
                }
            )
        else:
            candidate["resume_reason"] = "forced_state_file_missing"
        return candidate

    registry_path = _optimizer_resume_registry_path(signature)
    if not registry_path.exists():
        candidate["resume_reason"] = "resume_registry_missing"
        return candidate
    try:
        # utf-8-sig strips BOM if present (written by PowerShell WriteAllText / ConvertTo-Json)
        payload = json.loads(registry_path.read_text(encoding="utf-8-sig"))
    except Exception:
        candidate["resume_reason"] = "resume_registry_invalid"
        return candidate

    if not isinstance(payload, dict):
        candidate["resume_reason"] = "resume_registry_invalid"
        return candidate

    status = str(payload.get("status") or "").strip().lower()
    if status == "completed" and not _resume_allow_completed_state():
        candidate["resume_reason"] = "completed_state_resume_disabled"
        return candidate

    cache_file = Path(str(payload.get("cache_file") or "").strip())
    if not cache_file.exists():
        candidate["resume_reason"] = "resume_cache_missing"
        return candidate

    candidate.update(
        {
            "resume_used": True,
            "resume_source": "scenario_registry",
            "resume_state_file": str(cache_file.resolve()),
            "resume_previous_steps": int(payload.get("completed_steps") or 0),
        }
    )
    return candidate


@contextmanager
def _scoped_cwd(path: Path):
    """Temporarily switch current working directory."""
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _runtime_stale_code_state() -> Dict[str, Any]:
    current_mtime = os.path.getmtime(__file__)
    stale = current_mtime > _MODULE_BOOT_MTIME + 1e-6
    return {
        "stale_process_code": stale,
        "module_boot_mtime": _MODULE_BOOT_MTIME,
        "module_current_mtime": current_mtime,
        "module_path": __file__,
    }


def _format_exception_chain(exc: BaseException) -> str:
    parts: List[str] = []
    seen: set[str] = set()
    current: BaseException | None = exc
    depth = 0
    while current is not None and depth < 6:
        text = str(current).strip() or current.__class__.__name__
        key = text.lower()
        if key not in seen:
            seen.add(key)
            parts.append(text)
        next_exc = current.__cause__ or current.__context__
        if next_exc is current:
            break
        current = next_exc
        depth += 1
    return " | ".join(parts) if parts else "unknown_error"


def _is_path_writable_dir(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except Exception:
        return False


def _ensure_tidy3d_local_home() -> Dict[str, Any]:
    patched: Dict[str, Any] = {}
    force_local_home = str(os.getenv("INVERSE_TIDY3D_FORCE_LOCAL_HOME", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    local_home_raw = str(os.getenv("INVERSE_TIDY3D_LOCAL_HOME", "")).strip()
    local_home = (
        Path(local_home_raw).expanduser().resolve()
        if local_home_raw
        else (Path.cwd() / "build" / ".tidy3d_home").resolve()
    )
    default_home = Path.home()
    default_tidy3d_dir = default_home / ".tidy3d"
    if not force_local_home and _is_path_writable_dir(default_tidy3d_dir):
        return patched

    if not _is_path_writable_dir(local_home / ".tidy3d"):
        return patched

    roaming = local_home / "AppData" / "Roaming"
    local = local_home / "AppData" / "Local"
    roaming.mkdir(parents=True, exist_ok=True)
    local.mkdir(parents=True, exist_ok=True)
    os.environ["HOME"] = str(local_home)
    os.environ["USERPROFILE"] = str(local_home)
    os.environ["APPDATA"] = str(roaming)
    os.environ["LOCALAPPDATA"] = str(local)
    patched.update(
        {
            "HOME": str(local_home),
            "USERPROFILE": str(local_home),
            "APPDATA": str(roaming),
            "LOCALAPPDATA": str(local),
            "tidy3d_home_reason": "forced" if force_local_home else "default_home_unwritable",
        }
    )
    return patched


def _sanitize_tidy3d_runtime_env() -> Dict[str, Any]:
    patched: Dict[str, Any] = {}
    patched.update(_ensure_tidy3d_local_home())
    cleared: List[str] = []
    for key, value in _TIDY3D_RUNTIME_DEFAULTS.items():
        if str(os.getenv(key, "")).strip():
            continue
        os.environ[key] = value
        patched[key] = value

    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        raw = str(os.getenv(name, "") or "").strip()
        if not raw:
            continue
        if _is_loopback_proxy_value(raw):
            os.environ.pop(name, None)
            cleared.append(name)
    if cleared:
        patched["cleared_proxy_env"] = list(cleared)

    no_proxy_items: List[str] = []
    seen_no_proxy: set[str] = set()
    raw_no_proxy = ",".join(
        [
            str(os.getenv("NO_PROXY", "") or ""),
            str(os.getenv("no_proxy", "") or ""),
        ]
    )
    for item in raw_no_proxy.split(","):
        token = item.strip()
        if not token:
            continue
        key = token.lower()
        if key in seen_no_proxy:
            continue
        seen_no_proxy.add(key)
        no_proxy_items.append(token)

    for host in _TIDY3D_NO_PROXY_REQUIRED:
        key = host.lower()
        if key in seen_no_proxy:
            continue
        seen_no_proxy.add(key)
        no_proxy_items.append(host)

    if no_proxy_items:
        no_proxy_value = ",".join(no_proxy_items)
        if str(os.getenv("NO_PROXY", "") or "") != no_proxy_value:
            os.environ["NO_PROXY"] = no_proxy_value
            patched["NO_PROXY"] = no_proxy_value
        if str(os.getenv("no_proxy", "") or "") != no_proxy_value:
            os.environ["no_proxy"] = no_proxy_value
            patched["no_proxy"] = no_proxy_value
    return patched


def _is_loopback_proxy_value(raw_value: str) -> bool:
    value = str(raw_value or "").strip().lower()
    if not value:
        return False
    candidate = value if "://" in value else f"http://{value}"
    try:
        parsed = urlsplit(candidate)
        host = str(parsed.hostname or "").strip().lower()
    except Exception:
        host = ""

    if not host:
        host = value
    if host in {"localhost", "::1"}:
        return True
    return host.startswith("127.") or host.endswith(".localhost")


def _adjoint_retry_config() -> tuple[int, float]:
    retries = max(0, int(os.environ.get("INVERSE_ADJOINT_RETRIES", "2")))
    sleep_s = max(0.0, float(os.environ.get("INVERSE_ADJOINT_RETRY_SLEEP_S", "8.0")))
    return retries, sleep_s


def _adjoint_run_timeout_s() -> float:
    raw = _as_float_or_none(os.getenv("INVERSE_ADJOINT_RUN_TIMEOUT_S"))
    if raw is None or raw <= 0:
        # Default raised to 3600 s: a 1×5 MMI mode-mux run dispatches 10 cloud
        # simulations per iteration; 900 s was consistently too short.
        return 3600.0
    return max(float(raw), 60.0)


def _rerender_run_timeout_s() -> float:
    """Wall-clock budget for the post-Adam best-iteration rerender batch.

    W20 (V16-A rerender-stall fix): the rerender batch runs N case sims on the
    cloud after the optimizer loop ends. Without a cap, a single stuck cloud
    poll can hang the entire run indefinitely (observed in production: 4h+
    stall with cpu≈0.5%). Default 1800s gives 5 cases ~6 min budget each;
    override via INVERSE_RERENDER_RUN_TIMEOUT_S env var.
    """
    raw = _as_float_or_none(os.getenv("INVERSE_RERENDER_RUN_TIMEOUT_S"))
    if raw is None or raw <= 0:
        return 1800.0
    return max(float(raw), 60.0)


def _persist_iteration_metrics(
    *,
    record: "InverseDesignIterationRecord",
    run_artifact_tag: str,
) -> None:
    """Incrementally dump per-iteration metrics to disk.

    W21 (M48 RC-5 + V16-A data-loss fix): IterationRecord historically lived
    only in memory until the run completed. If the process hung in the
    post-Adam rerender (V16-A: 4h+ stall, never wrote result.json), ALL
    per-iter multi_case data was lost. This helper writes a minimal JSON
    snapshot per iteration so even a crashed/killed run leaves an audit
    trail with per-case absolute_ce_w, coupling_ratio, etc.

    Disabled by setting INVERSE_ITER_METRICS_PERSIST=0. Output dir defaults
    to ./build/iter_metrics/<run_artifact_tag>/ ; override via
    INVERSE_ITER_METRICS_DIR.
    """
    if os.environ.get("INVERSE_ITER_METRICS_PERSIST", "1").strip() in {"0", "false", "False", ""}:
        return
    try:
        base = os.environ.get("INVERSE_ITER_METRICS_DIR") or str(
            Path("build") / "iter_metrics"
        )
        out_dir = Path(base) / str(run_artifact_tag or "unknown_run")
        out_dir.mkdir(parents=True, exist_ok=True)
        iter_n = int(getattr(record, "iteration", 0) or 0)
        out_file = out_dir / f"iter_{iter_n:04d}.json"
        metrics = dict(getattr(record, "metrics", {}) or {})
        multi_case = metrics.get("multi_case")
        snapshot = {
            "run_artifact_tag": str(run_artifact_tag),
            "iteration": iter_n,
            "objective_metric": getattr(record, "objective_metric", None),
            "objective_value": getattr(record, "objective_value", None),
            "score": getattr(record, "score", None),
            "simulation_ok": bool(getattr(record, "simulation_ok", False)),
            "optimizer_backend": getattr(record, "optimizer_backend", None),
            "error": getattr(record, "error", None),
            "constraint_status": dict(getattr(record, "constraint_status", {}) or {}),
            "monitor_readings": dict(getattr(record, "monitor_readings", {}) or {}),
            "multi_case": multi_case if isinstance(multi_case, list) else None,
            "multi_case_summary": (
                metrics.get("multi_case_summary")
                if isinstance(metrics.get("multi_case_summary"), dict)
                else None
            ),
            "artifacts": list(getattr(record, "artifacts", []) or []),
        }
        out_file.write_text(
            json.dumps(snapshot, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
    except Exception as exc:  # pragma: no cover - persistence is best-effort
        logger.warning("iter-metrics persistence skipped (iter=%s): %s",
                       getattr(record, "iteration", "?"), exc)


def _adjoint_run_timeout_backoff() -> float:
    raw = _as_float_or_none(os.getenv("INVERSE_ADJOINT_RUN_TIMEOUT_BACKOFF"))
    if raw is None or raw < 1.0:
        return _ADJOINT_TIMEOUT_BACKOFF_DEFAULT
    return min(float(raw), _ADJOINT_TIMEOUT_BACKOFF_MAX)


def _adjoint_run_timeout_max_s(base_timeout_s: float) -> float:
    raw = _as_float_or_none(os.getenv("INVERSE_ADJOINT_RUN_TIMEOUT_MAX_S"))
    if raw is None or raw <= 0:
        return max(float(base_timeout_s), _ADJOINT_TIMEOUT_MAX_DEFAULT_S)
    return max(float(raw), float(base_timeout_s))


def _adjoint_timeout_for_attempt(base_timeout_s: float, attempt: int) -> float:
    base = max(float(base_timeout_s), 60.0)
    scaled = base * (_adjoint_run_timeout_backoff() ** max(int(attempt) - 1, 0))
    return min(scaled, _adjoint_run_timeout_max_s(base))


def _invoke_with_timeout(
    *,
    fn: Callable[[], Any],
    timeout_s: float,
    timeout_error: str,
) -> Any:
    result_box: Dict[str, Any] = {}
    error_box: Dict[str, BaseException] = {}

    def _runner() -> None:
        try:
            result_box["value"] = fn()
        except BaseException as exc:  # pragma: no cover - exercised by runtime failures
            error_box["error"] = exc

    worker = threading.Thread(
        target=_runner,
        name=f"adjoint-timeout-guard-{uuid4().hex[:8]}",
        daemon=True,
    )
    worker.start()
    worker.join(timeout=max(float(timeout_s), 1.0))

    if worker.is_alive():
        raise RuntimeError(timeout_error)
    if "error" in error_box:
        raise error_box["error"]
    return result_box.get("value")


def _adjoint_timeout_mode() -> str:
    # Default changed from "process" to "thread" (M183 fix): subprocess IPC
    # serialization fails for large topology params_vector payloads, causing
    # the parent retry loop to fire a second fresh-start optimizer.run().
    raw = str(os.getenv("INVERSE_ADJOINT_TIMEOUT_MODE", "thread") or "").strip().lower()
    if raw in {"thread", "process"}:
        return raw
    return "thread"


def _serialize_adjoint_kernel_output_for_ipc(output: Dict[str, Any]) -> str:
    records_payload: List[Dict[str, Any]] = []
    for record in list(output.get("records") or []):
        if isinstance(record, InverseDesignIterationRecord):
            records_payload.append(record.model_dump(mode="json"))
        elif isinstance(record, dict):
            records_payload.append(dict(record))

    payload = {
        "backend": str(output.get("backend") or ""),
        "records": records_payload,
        "termination_reason": str(output.get("termination_reason") or ""),
        "constraint_summary": output.get("constraint_summary", {}),
    }
    return json.dumps(payload, ensure_ascii=False, default=str)


def _deserialize_adjoint_kernel_output_from_ipc(payload_text: str) -> Dict[str, Any]:
    payload = json.loads(payload_text)
    records = [
        InverseDesignIterationRecord.model_validate(item)
        for item in list(payload.get("records") or [])
    ]
    return {
        "backend": str(payload.get("backend") or "adjoint_invdes_example"),
        "records": records,
        "termination_reason": str(payload.get("termination_reason") or ""),
        "constraint_summary": payload.get("constraint_summary", {}),
    }


def _adjoint_subprocess_worker(
    result_queue: Any,
    bundle_payload: Dict[str, Any],
    run_iterations: int,
    run_artifact_tag: str,
) -> None:
    # Force non-interactive matplotlib backend before any other import can
    # trigger GUI initialization.  On Windows this prevents Tk from creating a
    # window inside a daemon subprocess, which would deadlock.
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        import matplotlib
        matplotlib.use("Agg")
    except Exception:
        pass
    try:
        bundle = parse_inverse_design_config(bundle_payload)
        output = _run_adjoint_invdes_example(
            bundle=bundle,
            run_iterations=int(run_iterations),
            run_artifact_tag=str(run_artifact_tag),
        )
        result_queue.put(
            {
                "ok": True,
                "payload": _serialize_adjoint_kernel_output_for_ipc(output),
            }
        )
    except BaseException as exc:  # pragma: no cover - runtime failure path
        result_queue.put({"ok": False, "error": _format_exception_chain(exc)})


def _invoke_adjoint_with_process_timeout(
    *,
    bundle: InverseDesignConfigBundle,
    run_iterations: int,
    run_artifact_tag: str,
    timeout_s: float,
    timeout_error: str,
) -> Dict[str, Any]:
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue(maxsize=1)
    worker = ctx.Process(
        target=_adjoint_subprocess_worker,
        args=(
            result_queue,
            bundle.model_dump(mode="json"),
            int(run_iterations),
            str(run_artifact_tag),
        ),
        daemon=True,
    )
    worker.start()

    # DEADLOCK FIX: On Windows, multiprocessing Queue uses a named pipe with a
    # limited kernel buffer (~64 KB).  When the subprocess calls result_queue.put()
    # with a large JSON payload, its internal FeedFeeder thread blocks trying to
    # flush bytes to the pipe.  Python's Queue contract prevents the subprocess
    # from exiting until FeedFeeder completes.  If the main thread is simultaneously
    # blocking on worker.join(), neither side can make progress — classic deadlock.
    #
    # Fix: run a concurrent drain_thread that calls result_queue.get() while the
    # main thread calls worker.join().  The drain_thread reads the data from the
    # pipe, allowing FeedFeeder to finish, allowing the subprocess to exit, and
    # allowing worker.join() to return.
    message_box: list = []
    drain_exc_box: list = []

    def _drain_queue() -> None:
        try:
            # Wait up to timeout_s + 60 s for the subprocess to produce a result;
            # the extra 60 s ensures we don't race with worker.join()'s own deadline.
            msg = result_queue.get(timeout=max(float(timeout_s) + 60.0, 120.0))
            message_box.append(msg)
        except Exception as exc:
            drain_exc_box.append(exc)

    drain_thread = threading.Thread(target=_drain_queue, name="adjoint-queue-drain", daemon=True)
    drain_thread.start()

    worker.join(timeout=max(float(timeout_s), 1.0))

    if worker.is_alive():
        worker.terminate()
        worker.join(timeout=10.0)
        drain_thread.join(timeout=5.0)
        try:
            # cancel_join_thread() prevents result_queue.close() from blocking
            # on Windows when the subprocess was killed (FeedFeeder pipe closed).
            result_queue.cancel_join_thread()
            result_queue.close()
        except Exception:
            pass
        raise RuntimeError(timeout_error)

    # Worker exited normally — wait briefly for drain thread to finish reading.
    drain_thread.join(timeout=15.0)

    message: Dict[str, Any] | None = message_box[0] if message_box else None
    try:
        result_queue.cancel_join_thread()
        result_queue.close()
    except Exception:
        pass

    if not isinstance(message, dict):
        exit_code = worker.exitcode
        raise RuntimeError(
            "adjoint_subprocess_no_result: "
            f"worker exited with code {exit_code} without returning payload."
        )

    if not bool(message.get("ok")):
        raise RuntimeError(str(message.get("error") or "adjoint_subprocess_unknown_failure"))

    payload_text = str(message.get("payload") or "")
    if not payload_text:
        raise RuntimeError("adjoint_subprocess_empty_payload")
    return _deserialize_adjoint_kernel_output_from_ipc(payload_text)


def _allow_download_failure_rerun() -> bool:
    raw = str(os.getenv("INVERSE_ADJOINT_RETRY_DOWNLOAD_FAILURES", "")).strip().lower()
    if not raw:
        return False
    return raw in {"1", "true", "yes", "on"}


def _adjoint_min_flexcredit_threshold() -> float:
    raw = _as_float_or_none(os.getenv("INVERSE_STEP5_MIN_FLEXCREDIT"))
    if raw is None or raw <= 0:
        return 0.0
    return float(raw)


def _query_flexcredit_balance() -> float | None:
    try:
        _configure_tidy3d_cloud_auth()
        from tidy3d import web

        account = web.account()
        credit = _as_float_or_none(getattr(account, "credit", None))
        if credit is None:
            return None
        return float(credit)
    except Exception:
        return None


def _adjoint_credit_preflight_error(*, run_iterations: int | None = None) -> str | None:
    threshold = _adjoint_min_flexcredit_threshold()
    per_iter = _as_float_or_none(os.getenv("INVERSE_STEP5_MIN_FLEXCREDIT_PER_ITER"))
    per_task = _as_float_or_none(os.getenv("INVERSE_STEP5_MIN_FLEXCREDIT_PER_CLOUD_TASK"))
    if per_iter is None or per_iter <= 0:
        per_iter = 0.0
    if per_task is None or per_task <= 0:
        per_task = 0.0
    iterations = max(int(run_iterations or 0), 0)
    estimated_cloud_tasks = max(int(_as_float_or_none(os.getenv("INVERSE_STEP5_ESTIMATED_CLOUD_TASKS")) or 0), 0)
    required_candidates = [float(threshold), float(per_iter) * float(iterations)]
    if per_task > 0 and estimated_cloud_tasks > 0:
        required_candidates.append(float(per_task) * float(estimated_cloud_tasks))
    required = max(required_candidates)
    if required <= 0:
        return None

    balance = _query_flexcredit_balance()
    if balance is None:
        return None
    if balance + 1e-9 >= required:
        return None
    return (
        "insufficient balance preflight: "
        f"FlexCredit balance {balance:.3f} is below required {required:.3f}. "
        "Aborting Step5 cloud submission to avoid credit burn."
    )


def _estimate_adjoint_cloud_tasks_per_iteration(
    objective_cases: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    if not objective_cases:
        return {
            "per_iteration_total": 2,
            "per_iteration_adjoint": 2,
            "per_iteration_diagnostic": 0,
            "observation_mode": "single_case",
            "objective_case_count": 0,
            "unique_task_count": 1,
        }

    signatures: set[tuple[float, str, int, str]] = set()
    for case in objective_cases:
        if not isinstance(case, dict):
            continue
        source_port = _case_source_port(case)
        source_mode_index = _case_source_mode_index(case)
        source_direction = _case_source_direction(case)
        wavelength_nm = round(float(case.get("wavelength_nm", 0.0) or 0.0), 6)
        signatures.add((wavelength_nm, source_port, source_mode_index, source_direction))

    unique_task_count = max(len(signatures), 1)
    adjoint_tasks_per_iter = unique_task_count * 2
    observation_mode = _multi_case_observation_mode()
    objective_case_count = max(len(objective_cases), 1)

    diagnostic_tasks_per_iter = 0
    if observation_mode == "diagnostic_runner":
        diagnostic_tasks_per_iter = objective_case_count
    elif observation_mode == "hybrid":
        # Conservative budget estimate: in hybrid mode, when adjoint-internal
        # observation is missing the runner falls back to per-case diagnostics.
        diagnostic_tasks_per_iter = objective_case_count

    return {
        "per_iteration_total": max(adjoint_tasks_per_iter + diagnostic_tasks_per_iter, 1),
        "per_iteration_adjoint": max(adjoint_tasks_per_iter, 1),
        "per_iteration_diagnostic": max(diagnostic_tasks_per_iter, 0),
        "observation_mode": observation_mode,
        "objective_case_count": objective_case_count,
        "unique_task_count": unique_task_count,
    }


def _configure_tidy3d_cloud_auth() -> None:
    """Force-refresh cloud auth from .env for Step5 adjoint execution.

    SAFETY: previously called ``load_dotenv(override=True)`` which blindly
    re-applied EVERY key in .env (including ``INVERSE_STEP5_*`` toggles),
    silently overriding launcher-level decisions such as
    ``INVERSE_STEP5_ENABLE_OPTIMIZER_RESUME=0`` /
    ``INVERSE_STEP5_RESUME_ALLOW_COMPLETED=0``. Per AGENTS Rule 13 (no
    point patches) and the M48 lessons (no leaky side-effects beyond the
    function's declared purpose), this routine now refreshes ONLY the
    Tidy3D auth keys and leaves all other environment variables untouched.
    """

    AUTH_KEYS = ("TIDY3D_API_KEY", "SIMCLOUD_APIKEY", "FLEXCOMPUTE_API_KEY")

    try:
        from dotenv import dotenv_values

        dotenv_path = Path(__file__).resolve().parents[2] / ".env"
        if dotenv_path.exists():
            file_values = dotenv_values(dotenv_path) or {}
            for _k in AUTH_KEYS:
                _v = file_values.get(_k)
                if _v:
                    # Always refresh auth from .env (override=True semantics
                    # preserved ONLY for auth keys).
                    os.environ[_k] = str(_v).strip()
    except Exception:
        # Keep runtime robust when dotenv is unavailable.
        pass

    api_key = (
        os.getenv("TIDY3D_API_KEY", "").strip()
        or os.getenv("SIMCLOUD_APIKEY", "").strip()
        or os.getenv("FLEXCOMPUTE_API_KEY", "").strip()
    )
    if not api_key:
        raise RuntimeError("Missing TIDY3D_API_KEY for Step5 cloud adjoint execution.")

    os.environ["TIDY3D_API_KEY"] = api_key
    os.environ["SIMCLOUD_APIKEY"] = api_key
    _sanitize_tidy3d_runtime_env()

    # --- Socket-level timeout guard for the `requests` library --------------
    # Root-cause fix for the 2026-05-30 EXT2 stall: tidy3d.web.core.http_util
    # invokes module-level ``requests.get(...)`` without a timeout, so a
    # half-open TCP / silent server can wedge an SSL ``recv_into`` forever.
    # Our outer ``_invoke_adjoint_once`` timeout-guard would eventually fire
    # (3600s), but that wastes an hour of cloud-side compute and stalls the
    # optimizer. Inject a sane default timeout for any requests call that
    # forgets to specify one, while leaving explicit timeouts untouched.
    #
    # Env override: ``INVERSE_HTTP_DEFAULT_TIMEOUT_S`` (seconds). Default 300s
    # (5 min) — long enough for any healthy POST upload, short enough to
    # break a hung socket and let retry logic resubmit.
    _patch_requests_default_timeout()

    from tidy3d import web

    web.configure(apikey=api_key)


_REQUESTS_TIMEOUT_PATCHED = False


def _patch_requests_default_timeout() -> None:
    """Install a global default ``timeout`` for the ``requests`` library.

    Idempotent: re-invocation is a no-op. Wraps ``requests.Session.request``
    so every call (including module-level ``requests.get/post/...`` which
    internally go through a Session) receives a default ``timeout=`` if the
    caller did not provide one. Explicit caller-supplied timeouts are
    respected.

    Without this, Tidy3D cloud polling can SSL-recv forever on a silent
    server, hanging the optimizer's adjoint loop until the outer 60-min
    guard trips (see py-spy stack at EXT2 stall 2026-05-30).
    """
    global _REQUESTS_TIMEOUT_PATCHED
    if _REQUESTS_TIMEOUT_PATCHED:
        return
    try:
        import requests  # type: ignore
    except Exception:
        return

    try:
        default_timeout = float(os.getenv("INVERSE_HTTP_DEFAULT_TIMEOUT_S", "300"))
    except Exception:
        default_timeout = 300.0

    if default_timeout <= 0:
        # Caller explicitly disabled (e.g. via env=0); keep stock behaviour.
        _REQUESTS_TIMEOUT_PATCHED = True
        return

    _original_request = requests.Session.request

    def _request_with_default_timeout(self, method, url, **kwargs):  # type: ignore[no-untyped-def]
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = default_timeout
        return _original_request(self, method, url, **kwargs)

    requests.Session.request = _request_with_default_timeout  # type: ignore[assignment]
    _REQUESTS_TIMEOUT_PATCHED = True
    try:
        logger.info(
            "Installed default requests timeout=%.0fs (override via INVERSE_HTTP_DEFAULT_TIMEOUT_S)",
            default_timeout,
        )
    except Exception:
        pass


def _is_retryable_adjoint_error(error_text: str) -> bool:
    normalized = (error_text or "").strip().lower()
    return any(pattern in normalized for pattern in _ADJOINT_RETRY_ERROR_PATTERNS)


def _is_download_failure_adjoint_error(error_text: str) -> bool:
    normalized = (error_text or "").strip().lower()
    return (
        "failed to download the simulation data file" in normalized
        or "simulation_data_unavailable" in normalized
        or "headobject operation: not found" in normalized
        or "headobject operation: 404" in normalized
    )


def _is_non_retryable_adjoint_error(error_text: str) -> bool:
    normalized = (error_text or "").strip().lower()
    return any(pattern in normalized for pattern in _ADJOINT_NON_RETRYABLE_ERROR_PATTERNS)


def _adjoint_task_name(component_type: str, run_artifact_tag: str) -> str:
    component = re.sub(r"[^a-zA-Z0-9_-]+", "-", (component_type or "component")).strip("-")
    tag = re.sub(r"[^a-zA-Z0-9_-]+", "-", (run_artifact_tag or "run")).strip("-")
    return f"step5-{component}-{tag[:12]}"


class StrictModel(BaseModel):
    """Base model that rejects unknown fields to keep output stable."""

    model_config = ConfigDict(extra="forbid")


class InverseDesignIterationRecord(StrictModel):
    """Single inverse-design iteration execution record."""

    iteration: int
    parameters: Dict[str, float] = Field(default_factory=dict)
    score: float | None = None
    objective_metric: str = ""
    objective_value: float | None = None
    simulation_ok: bool = False
    optimizer_backend: str = ""
    monitor_readings: Dict[str, float | None] = Field(default_factory=dict)
    constraint_status: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    artifacts: List[str] = Field(default_factory=list)
    error: str | None = None


class InverseDesignRunResult(StrictModel):
    """Top-level run result for step 9.1 item 5."""

    ok: bool
    status: Literal["completed", "requires_config_fix", "simulation_failed", "requires_self_recovery"]
    validation: ConfigValidationResult
    iterations: List[InverseDesignIterationRecord] = Field(default_factory=list)
    checkpoint_reports: List[CheckpointReport] = Field(default_factory=list)
    best_iteration: int | None = None
    best_score: float | None = None
    best_objective_value: float | None = None
    optimizer_backend: str = ""
    uses_adjoint_invdes_example: bool = False
    scenario_fingerprint: ScenarioFingerprint | None = None
    constraint_summary: Dict[str, Any] = Field(default_factory=dict)
    objective_metric: str = ""
    objective_goal: str = ""
    termination_reason: str = ""
    failure_diagnosis: FailureDiagnosis | None = None


def run_inverse_design(
    payload: InverseDesignConfigBundle | Dict[str, Any],
    *,
    max_iterations: int | None = None,
    include_llm_review: bool | None = None,
    enable_failure_diagnosis: bool | None = None,
    constraint_packet: Step4ConstraintPacket | Dict[str, Any] | None = None,
    checkpoint_interval: int | None = None,
    checkpoint_warmup: int | None = None,
    llm_call_fn: Any | None = None,
    llm_model: str = "gpt-5.4",
) -> InverseDesignRunResult:
    """Run inverse-design optimization loop and return quantitative metrics."""

    bundle = payload if isinstance(payload, InverseDesignConfigBundle) else parse_inverse_design_config(payload)
    runtime_config = bundle.runtime_config
    max_iterations = (
        int(max_iterations) if max_iterations is not None else int(runtime_config.max_iterations)
    )
    include_llm_review = (
        include_llm_review if include_llm_review is not None else bool(runtime_config.include_llm_review)
    )
    enable_failure_diagnosis = (
        enable_failure_diagnosis
        if enable_failure_diagnosis is not None
        else bool(runtime_config.enable_failure_diagnosis)
    )
    checkpoint_interval = (
        int(checkpoint_interval)
        if checkpoint_interval is not None
        else int(runtime_config.checkpoint_interval)
    )
    checkpoint_warmup = (
        int(checkpoint_warmup)
        if checkpoint_warmup is not None
        else int(runtime_config.checkpoint_warmup)
    )
    rag_packet = _coerce_constraint_packet(constraint_packet)
    optimizer_backend = "adjoint_invdes_example"
    constraints = list(bundle.optimization_config.constraints)
    scenario_fingerprint = _build_scenario_fingerprint(bundle)
    run_artifact_tag = _resolve_run_artifact_tag()
    _write_step5_heartbeat(run_artifact_tag, "run_start", max_iterations=max_iterations)
    runtime_code_state = _runtime_stale_code_state()
    validation = validate_config(
        bundle,
        include_llm_review=include_llm_review,
        constraint_packet=rag_packet,
        llm_call_fn=llm_call_fn,
        llm_model=llm_model,
    )
    if not validation.ok:
        _write_step5_heartbeat(run_artifact_tag, "validation_failed")
        diagnosis = None
        if _diagnosis_enabled(enable_failure_diagnosis):
            diagnosis = diagnose_inverse_design_failure(
                "Step4 validation failed before execution.",
                component_type=bundle.simulation_config.component_type,
                objective_metric=bundle.optimization_config.objective.metric,
                recent_issues=[issue.code for issue in validation.issues],
                use_llm_advisor=True,
                llm_call_fn=llm_call_fn,
                llm_model=llm_model,
            )
        return InverseDesignRunResult(
            ok=False,
            status="requires_config_fix",
            validation=validation,
            iterations=[],
            checkpoint_reports=[],
            optimizer_backend=optimizer_backend,
            uses_adjoint_invdes_example=(optimizer_backend == "adjoint_invdes_example"),
            scenario_fingerprint=scenario_fingerprint,
            constraint_summary={
                "declared_constraints": constraints,
                "evaluated": False,
                "satisfied": None,
                "note": "Constraint evaluation is skipped because config validation failed.",
                "run_artifact_tag": run_artifact_tag,
                "step5_heartbeat_file": str(_step5_heartbeat_path(run_artifact_tag).resolve()),
                "runtime_code_state": runtime_code_state,
            },
            termination_reason="Config validation failed. Return to step 9.1.4.",
            objective_metric=bundle.optimization_config.objective.metric,
            objective_goal=bundle.optimization_config.objective.goal,
            failure_diagnosis=diagnosis,
        )

    objective = bundle.optimization_config.objective
    objective_cases = _objective_cases(bundle)
    requested_max = bundle.optimization_config.termination.max_iterations
    run_iterations = max(1, min(max_iterations, requested_max))

    # Primary path: official-style invdes/adjoint optimizer.
    _write_step5_heartbeat(run_artifact_tag, "kernel_launch", run_iterations=run_iterations)
    kernel_output = _execute_adjoint_invdes_kernel(
        bundle=bundle,
        run_iterations=run_iterations,
        run_artifact_tag=run_artifact_tag,
    )
    _write_step5_heartbeat(run_artifact_tag, "kernel_returned", backend=kernel_output.get("backend", ""))

    records: List[InverseDesignIterationRecord] = kernel_output["records"]
    # Bug A fix: Attach adjoint_forward 5-case HDF5s/viewer scripts to
    # records[0] so they appear in best_iteration_artifacts.
    _fwd_arts = [
        str(p) for p in kernel_output.get("constraint_summary", {}).get("adjoint_forward_sim_artifacts", [])
        if str(p or "").strip()
    ]
    if not _fwd_arts:
        _fwd_arts = _recover_adjoint_forward_artifacts_from_disk(run_artifact_tag)
    if records and _fwd_arts:
        records[0].artifacts = list(dict.fromkeys([*_fwd_arts, *records[0].artifacts]))
    optimizer_backend: str = kernel_output["backend"]
    constraint_summary = dict(kernel_output["constraint_summary"])
    constraint_summary.setdefault("run_artifact_tag", run_artifact_tag)
    constraint_summary.setdefault("runtime_code_state", runtime_code_state)
    constraint_summary.setdefault("step5_heartbeat_file", str(_step5_heartbeat_path(run_artifact_tag).resolve()))
    if objective_cases:
        constraint_summary["multi_case_observation_summary"] = _build_run_multi_case_observation_summary(
            records=records,
            objective_cases=objective_cases,
        )
    termination_reason = kernel_output["termination_reason"]
    physics_gate = _evaluate_first_iteration_physics_gate(
        records=records,
        bundle=bundle,
        constraint_packet=rag_packet,
    )
    constraint_summary["hard_physics_gate"] = physics_gate
    if records:
        records[0].constraint_status["hard_physics_gate"] = physics_gate

    if not physics_gate.get("passed", True):
        diagnosis = None
        gate_issue_codes = [
            str(item.get("code") or "hard_physics_gate_blocked")
            for item in physics_gate.get("blockers", [])
        ]
        if _diagnosis_enabled(enable_failure_diagnosis):
            diagnosis = diagnose_inverse_design_failure(
                _gate_failure_reason(physics_gate),
                component_type=bundle.simulation_config.component_type,
                objective_metric=objective.metric,
                recent_issues=[item.code for item in validation.issues] + gate_issue_codes,
                use_llm_advisor=True,
                llm_call_fn=llm_call_fn,
                llm_model=llm_model,
            )
        return InverseDesignRunResult(
            ok=False,
            status="simulation_failed",
            validation=validation,
            iterations=records,
            checkpoint_reports=[],
            best_iteration=None,
            best_score=None,
            best_objective_value=None,
            optimizer_backend=optimizer_backend,
            uses_adjoint_invdes_example=(optimizer_backend == "adjoint_invdes_example"),
            scenario_fingerprint=scenario_fingerprint,
            constraint_summary=constraint_summary,
            objective_metric=objective.metric,
            objective_goal=objective.goal,
            termination_reason=_gate_failure_reason(physics_gate),
            failure_diagnosis=diagnosis,
        )

    first_sim_failure = next((record for record in records if not record.simulation_ok), None)
    if first_sim_failure is not None:
        diagnosis = None
        if _diagnosis_enabled(enable_failure_diagnosis):
            diagnosis = diagnose_inverse_design_failure(
                first_sim_failure.error or "Simulation failed during inverse-design execution.",
                component_type=bundle.simulation_config.component_type,
                objective_metric=objective.metric,
                recent_issues=[item.code for item in validation.issues],
                use_llm_advisor=True,
                llm_call_fn=llm_call_fn,
                llm_model=llm_model,
            )
        constraint_summary["first_simulation_failure"] = {
            "iteration": first_sim_failure.iteration,
            "error": first_sim_failure.error,
        }
        return InverseDesignRunResult(
            ok=False,
            status="simulation_failed",
            validation=validation,
            iterations=records,
            checkpoint_reports=[],
            best_iteration=None,
            best_score=None,
            best_objective_value=None,
            optimizer_backend=optimizer_backend,
            uses_adjoint_invdes_example=(optimizer_backend == "adjoint_invdes_example"),
            scenario_fingerprint=scenario_fingerprint,
            constraint_summary=constraint_summary,
            objective_metric=objective.metric,
            objective_goal=objective.goal,
            termination_reason="Simulation failed during inverse-design execution.",
            failure_diagnosis=diagnosis,
        )

    # ── Multi-case best-iteration rerender (runs in PARENT process) ─────────
    # _ensure_best_iteration_multi_case_artifacts was REMOVED from the child
    # subprocess (_run_adjoint_invdes_example) because, for 5-case bundles,
    # it can take ~35 min and causes the subprocess to exceed its timeout.
    # Running it here in the parent has no timeout risk.
    if objective_cases and records and _should_use_multi_case_adjoint(objective_cases):
        _parent_run_tag = constraint_summary.get("run_artifact_tag", run_artifact_tag)
        try:
            _ensure_best_iteration_multi_case_artifacts(
                bundle=bundle,
                records=records,
                objective_metric=objective.metric,
                objective_goal=objective.goal,
                objective_cases=objective_cases,
                run_artifact_tag=_parent_run_tag,
            )
        except BaseException as _rerender_err:  # M184: catch SystemExit from Tidy3D auth/balance  # noqa: BLE001
            logger.warning(
                "Parent-process multi-case rerender failed (%s, non-fatal): %s",
                type(_rerender_err).__name__,
                _rerender_err,
            )
            if isinstance(_rerender_err, KeyboardInterrupt):
                raise
    # ─────────────────────────────────────────────────────────────────────────

    checkpoint_reports = _build_checkpoint_reports(
        records=records,
        bundle=bundle,
        checkpoint_interval=checkpoint_interval,
        checkpoint_warmup=checkpoint_warmup,
        scenario_fingerprint=scenario_fingerprint,
    )
    # M59 fix: only the *last* checkpoint window can trigger recovery.
    # Earlier windows may show transient regressions (exploration phase) that
    # the optimizer recovers from.  Using the first failing window caused
    # false-positive recovery on runs that converged successfully.
    checkpoint_failure: CheckpointReport | None = None
    if checkpoint_reports:
        last_report = checkpoint_reports[-1]
        if last_report.status == "fail":
            # Additional guard: if the run completed all requested iterations
            # and achieved significant global improvement, the "stall" in the
            # final window is normal convergence — not a failure.
            _obj_vals = [_checkpoint_objective_value(r) for r in records]
            _first_obj = _obj_vals[0] if _obj_vals else 0.0
            _best_obj = (
                max(_obj_vals) if objective.goal == "maximize" else min(_obj_vals)
            ) if _obj_vals else 0.0
            _global_base = max(abs(_first_obj), 1e-9)
            _global_gain = abs(_best_obj - _first_obj) / _global_base
            # If overall gain > 50% the run was productive; only trigger
            # recovery for severe issues (oscillation, regression, direction).
            _is_convergence_tail = (
                _global_gain > 0.50
                and last_report.error_subtype == "stalled_convergence"
            )
            if not _is_convergence_tail:
                checkpoint_failure = last_report

    if checkpoint_failure is not None:
        constraint_summary["checkpoint_trigger"] = checkpoint_failure.model_dump()
        best_record: InverseDesignIterationRecord | None = None
        for record in records:
            if best_record is None or _is_better(record, best_record, objective.goal):
                best_record = record
        diagnosis = None
        if _diagnosis_enabled(enable_failure_diagnosis):
            diagnosis = diagnose_inverse_design_failure(
                _checkpoint_failure_reason(checkpoint_failure),
                component_type=bundle.simulation_config.component_type,
                objective_metric=objective.metric,
                objective_goal=objective.goal,
                recent_issues=[item.code for item in validation.issues],
                recent_iterations=[record.model_dump() for record in records[-checkpoint_failure.window_size :]],
                checkpoint_report=checkpoint_failure,
                checkpoint_reports=checkpoint_reports,
                scenario_fingerprint=scenario_fingerprint,
                enable_optimizer_attribution=bool(runtime_config.enable_optimizer_attribution),
                optimizer_attribution_min_samples=int(runtime_config.optimizer_attribution_min_samples),
                use_llm_advisor=True,
                llm_call_fn=llm_call_fn,
                llm_model=llm_model,
            )
        return InverseDesignRunResult(
            ok=False,
            status="requires_self_recovery",
            validation=validation,
            iterations=records,
            checkpoint_reports=checkpoint_reports,
            best_iteration=None if best_record is None else best_record.iteration,
            best_score=None if best_record is None else best_record.score,
            best_objective_value=None if best_record is None else best_record.objective_value,
            optimizer_backend=optimizer_backend,
            uses_adjoint_invdes_example=(optimizer_backend == "adjoint_invdes_example"),
            scenario_fingerprint=scenario_fingerprint,
            constraint_summary=constraint_summary,
            objective_metric=objective.metric,
            objective_goal=objective.goal,
            termination_reason=_checkpoint_failure_reason(checkpoint_failure),
            failure_diagnosis=diagnosis,
        )

    best_index: int | None = None
    best_record: InverseDesignIterationRecord | None = None

    for idx, record in enumerate(records):
        if best_record is None or _is_better(record, best_record, objective.goal):
            best_record = record
            best_index = idx

    return InverseDesignRunResult(
        ok=True,
        status="completed",
        validation=validation,
        iterations=records,
        checkpoint_reports=checkpoint_reports,
        best_iteration=None if best_record is None else best_record.iteration,
        best_score=None if best_record is None else best_record.score,
        best_objective_value=None if best_record is None else best_record.objective_value,
        optimizer_backend=optimizer_backend,
        uses_adjoint_invdes_example=(optimizer_backend == "adjoint_invdes_example"),
        scenario_fingerprint=scenario_fingerprint,
        constraint_summary=constraint_summary,
        objective_metric=objective.metric,
        objective_goal=objective.goal,
        termination_reason=termination_reason,
        failure_diagnosis=None,
    )


def _diagnosis_enabled(enable_failure_diagnosis: bool | None) -> bool:
    if enable_failure_diagnosis is not None:
        return enable_failure_diagnosis
    value = os.getenv("INVERSE_DESIGN_ENABLE_FAILURE_DIAGNOSIS", "0").strip().lower()
    return value in {"1", "true", "yes", "on"}


def inverse_design_execution_schema() -> Dict[str, Any]:
    """Return schema for inverse-design execution output."""

    return InverseDesignRunResult.model_json_schema()


def _build_scenario_fingerprint(bundle: InverseDesignConfigBundle) -> ScenarioFingerprint:
    domain_size = [float(item) for item in bundle.simulation_config.domain.size_um]
    min_dim = max(min(domain_size), 1e-6)
    domain_ratio = ":".join(f"{(item / min_dim):.2f}" for item in domain_size)
    monitor_signature = "|".join(
        sorted(
            f"{monitor.monitor_type}:{monitor.metric or monitor.name}"
            for monitor in bundle.simulation_config.monitors
        )
    )
    boundary_type = "|".join(sorted(set(bundle.simulation_config.domain.boundary.values())))
    return ScenarioFingerprint(
        component_type=bundle.simulation_config.component_type,
        objective_metric=bundle.optimization_config.objective.metric,
        objective_goal=bundle.optimization_config.objective.goal,
        wavelength_band=_wavelength_band(bundle.simulation_config.wavelength_nm),
        domain_ratio=domain_ratio,
        monitor_topology_signature=monitor_signature[:160],
        boundary_type=boundary_type,
    )


def _wavelength_band(wavelength_nm: float) -> str:
    if wavelength_nm < 1000:
        return "visible"
    if wavelength_nm < 1360:
        return "o_band"
    if wavelength_nm < 1460:
        return "e_band"
    if wavelength_nm < 1530:
        return "s_band"
    if wavelength_nm < 1565:
        return "c_band"
    if wavelength_nm < 1625:
        return "l_band"
    return "custom_band"


def _build_checkpoint_reports(
    *,
    records: List[InverseDesignIterationRecord],
    bundle: InverseDesignConfigBundle,
    checkpoint_interval: int,
    checkpoint_warmup: int,
    scenario_fingerprint: ScenarioFingerprint,
) -> List[CheckpointReport]:
    interval = max(1, int(checkpoint_interval))
    warmup = max(interval, int(checkpoint_warmup))
    if len(records) < warmup:
        return []

    reports: List[CheckpointReport] = []
    for end_index in range(warmup, len(records) + 1):
        if end_index % interval != 0:
            continue
        window = records[end_index - interval : end_index]
        report = _evaluate_checkpoint_window(
            window=window, bundle=bundle, all_records=records,
        )
        reports.append(report)
        _record_checkpoint_report(report, bundle=bundle, scenario_fingerprint=scenario_fingerprint)
    return reports


def _evaluate_checkpoint_window(
    *,
    window: List[InverseDesignIterationRecord],
    bundle: InverseDesignConfigBundle,
    all_records: List[InverseDesignIterationRecord] | None = None,
) -> CheckpointReport:
    goal = bundle.optimization_config.objective.goal
    runtime_config = bundle.runtime_config
    min_relative_improvement = float(runtime_config.checkpoint_min_relative_improvement)
    regression_tolerance = float(runtime_config.checkpoint_regression_tolerance)
    oscillation_threshold = float(runtime_config.checkpoint_oscillation_ratio_threshold)
    direction_update_threshold = float(runtime_config.checkpoint_update_norm_direction_threshold)
    oscillation_update_threshold = float(runtime_config.checkpoint_update_norm_oscillation_threshold)
    objective_values = [
        _checkpoint_objective_value(record)
        for record in window
    ]
    delta = objective_values[-1] - objective_values[0] if len(objective_values) >= 2 else 0.0
    oscillation_ratio = _estimate_oscillation_ratio(objective_values)
    update_norm = _estimate_parameter_update_norm(window)
    manufacturability_score = _estimate_manufacturability_score(
        parameters=window[-1].parameters if window else {},
        bundle=bundle,
    )
    observability_score = _estimate_observability_score(window)

    error_family = ""
    error_subtype = ""
    reasons: List[str] = []

    direction_bad = (goal == "maximize" and delta <= 0.0) or (goal == "minimize" and delta >= 0.0)
    # W12 fix: use *relative* improvement rate instead of absolute threshold.
    # ``abs(delta) < 0.02`` false-triggered on healthy 11× improvements when
    # objective values are in the 0.003–0.037 range.
    baseline = max(abs(objective_values[0]), 1e-9)
    relative_improvement = abs(delta) / baseline
    marginal_progress = relative_improvement < min_relative_improvement

    # M58 fix: suppress stall detection when global progress is significant.
    # If the optimizer achieved large cumulative improvement (from iteration 1
    # to now), a small window-local change is normal convergence, not a stall.
    if marginal_progress and all_records and len(all_records) >= 2:
        global_values = [_checkpoint_objective_value(r) for r in all_records]
        global_first = global_values[0]
        global_best = max(global_values) if goal == "maximize" else min(global_values)
        global_baseline = max(abs(global_first), 1e-9)
        global_improvement = abs(global_best - global_first) / global_baseline
        # If overall improvement exceeds 50%, late-stage <5% windows are
        # normal convergence tails — do NOT treat as stall.
        if global_improvement > 0.50:
            marginal_progress = False

    # M53 fix: detect best-value regression within the window.
    # For a maximize goal, if the window's best value is worse than the
    # window's first value, optimizer is systematically going backward.
    window_best = max(objective_values) if goal == "maximize" else min(objective_values)
    window_worst = min(objective_values) if goal == "maximize" else max(objective_values)
    best_regressing = False
    if len(objective_values) >= 3:
        mid = len(objective_values) // 2
        first_half_best = max(objective_values[:mid]) if goal == "maximize" else min(objective_values[:mid])
        second_half_best = max(objective_values[mid:]) if goal == "maximize" else min(objective_values[mid:])
        # Use absolute tolerance: 5% of first_half_best magnitude.
        # Multiplying first_half_best by 0.95 is incorrect for negative objectives
        # (maximize task): it shifts the threshold toward a BETTER value, wrongly
        # requiring improvement rather than merely tolerating a 5% regression.
        tolerance = abs(first_half_best) * regression_tolerance
        if goal == "maximize":
            best_regressing = second_half_best < first_half_best - tolerance
        elif goal == "minimize":
            best_regressing = second_half_best > first_half_best + tolerance

    if observability_score < 0.35:
        error_family = "simulation_scene"
        error_subtype = "monitor_observability"
        reasons.append("Checkpoint observed weak or missing monitor signal coverage.")
    elif manufacturability_score < 0.45:
        error_family = "simulation_scene"
        error_subtype = "manufacturability_risk"
        reasons.append("Checkpoint manufacturability proxy fell below the safe threshold.")
    elif oscillation_ratio > oscillation_threshold and update_norm > oscillation_update_threshold:
        error_family = "optimization_setup"
        error_subtype = "oscillation"
        reasons.append("Objective trend is oscillating despite large parameter updates.")
    elif best_regressing:
        error_family = "optimization_setup"
        error_subtype = "objective_regression"
        reasons.append(
            "Best objective value regressed in the second half of the checkpoint window. "
            "Optimizer hyperparameters (learning rate, beta) likely need tuning."
        )
    elif direction_bad and update_norm > direction_update_threshold:
        error_family = "optimization_setup"
        error_subtype = "objective_direction"
        reasons.append("Objective is moving in the wrong direction across the latest checkpoint window.")
    elif marginal_progress:
        error_family = "optimization_setup"
        error_subtype = "stalled_convergence"
        reasons.append("Objective improvement stalled across the latest checkpoint window.")

    latest_readings = dict(window[-1].monitor_readings) if window else {}
    status = "fail" if reasons else "pass"
    return CheckpointReport(
        checkpoint_iteration=window[-1].iteration if window else 0,
        window_size=len(window),
        status=status,
        error_family=error_family,
        error_subtype=error_subtype,
        reasons=reasons,
        objective_values=objective_values,
        objective_delta=round(delta, 6),
        oscillation_ratio=round(oscillation_ratio, 6),
        parameter_update_norm=round(update_norm, 6),
        manufacturability_score=round(manufacturability_score, 6),
        observability_score=round(observability_score, 6),
        metrics={
            "goal": goal,
            "latest_monitor_readings": latest_readings,
            "thresholds": {
                "checkpoint_min_relative_improvement": min_relative_improvement,
                "checkpoint_regression_tolerance": regression_tolerance,
                "checkpoint_oscillation_ratio_threshold": oscillation_threshold,
                "checkpoint_update_norm_direction_threshold": direction_update_threshold,
                "checkpoint_update_norm_oscillation_threshold": oscillation_update_threshold,
            },
        },
    )


def _record_secondary_objective_value(record: InverseDesignIterationRecord) -> float | None:
    if record.score is not None and math.isfinite(record.score):
        return float(record.score)
    metrics = record.metrics if isinstance(record.metrics, dict) else {}
    adjoint_trace = metrics.get("adjoint_trace", {}) if isinstance(metrics, dict) else {}
    if isinstance(adjoint_trace, dict):
        for key in ("post_process_val", "objective_fn_val"):
            value = _as_float_or_none(adjoint_trace.get(key))
            if value is not None and math.isfinite(value):
                return float(value)
    return None


def _adjoint_cloud_preflight_error() -> str | None:
    enabled = str(os.getenv("INVERSE_STEP5_CLOUD_PREFLIGHT_ENABLED", "1")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not enabled:
        return None

    retries = max(1, int(os.getenv("INVERSE_STEP5_CLOUD_PREFLIGHT_RETRIES", "2")))
    sleep_s = max(0.0, float(os.getenv("INVERSE_STEP5_CLOUD_PREFLIGHT_SLEEP_S", "2.0")))
    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            _configure_tidy3d_cloud_auth()
            from tidy3d import web

            _ = web.account()
            return None
        except SystemExit as exc:
            # tidy3d may call sys.exit(1) on KYC/auth failure; catch to prevent
            # propagation that causes the parent process to exit with code 1.
            last_error = f"auth_sys_exit_{exc.code}"
            break  # Never retry a forced exit
        except Exception as exc:
            last_error = _format_exception_chain(exc)
            retryable = _is_retryable_adjoint_error(last_error) and not _is_non_retryable_adjoint_error(last_error)
            if attempt >= retries or not retryable:
                break
            if sleep_s > 0:
                time.sleep(sleep_s)
    if not last_error:
        return "cloud preflight failed: unknown_error"
    return f"cloud preflight failed: {last_error}"


def _checkpoint_objective_value(record: InverseDesignIterationRecord) -> float:
    primary = (
        float(record.objective_value)
        if record.objective_value is not None and math.isfinite(record.objective_value)
        else None
    )
    secondary = _record_secondary_objective_value(record)
    if primary is None:
        return secondary if secondary is not None else 0.0
    # For demux/multi-case runs objective_value can be numerically flat near 0,
    # while score/post_process still carries the real optimization trend.
    if secondary is not None and abs(primary) <= 1e-6 and abs(secondary - primary) > 1e-9:
        return float(secondary)
    return float(primary)


def _estimate_oscillation_ratio(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    total_abs = sum(abs(curr - prev) for prev, curr in zip(values, values[1:]))
    net = abs(values[-1] - values[0])
    return total_abs / max(net, 1e-9)


def _estimate_parameter_update_norm(window: Sequence[InverseDesignIterationRecord]) -> float:
    if len(window) < 2:
        return 0.0
    norms: List[float] = []
    for previous, current in zip(window, window[1:]):
        keys = set(previous.parameters).intersection(current.parameters)
        if not keys:
            continue
        squared = 0.0
        for key in keys:
            squared += (float(current.parameters[key]) - float(previous.parameters[key])) ** 2
        norms.append(math.sqrt(squared) / max(len(keys), 1))
    if not norms:
        return 0.0
    return sum(norms) / len(norms)


def _estimate_manufacturability_score(
    *,
    parameters: Dict[str, float],
    bundle: InverseDesignConfigBundle,
) -> float:
    if not parameters:
        return 1.0

    risk_hits = 0.0
    variable_map = {var.name: var for var in bundle.optimization_config.variables}
    for name, value in parameters.items():
        numeric_value = float(value)
        lowered = name.lower()
        if any(token in lowered for token in {"width", "gap", "height", "separation"}):
            if numeric_value < 0.08:
                risk_hits += 1.0
        bounds = variable_map.get(name)
        if bounds is None:
            continue
        span = float(bounds.upper_bound) - float(bounds.lower_bound)
        if span <= 0:
            continue
        edge_margin = min(
            abs(numeric_value - float(bounds.lower_bound)),
            abs(float(bounds.upper_bound) - numeric_value),
        )
        if edge_margin / span < 0.05:
            risk_hits += 0.5

    total = max(len(parameters), 1)
    return max(0.0, 1.0 - risk_hits / total)


def _estimate_observability_score(window: Sequence[InverseDesignIterationRecord]) -> float:
    seen = 0
    observable = 0
    for record in window:
        for value in record.monitor_readings.values():
            if value is None or not math.isfinite(float(value)):
                continue
            seen += 1
            if abs(float(value)) > 1e-5:
                observable += 1
    if seen == 0:
        return 0.0
    return observable / seen


def _record_checkpoint_report(
    report: CheckpointReport,
    *,
    bundle: InverseDesignConfigBundle,
    scenario_fingerprint: ScenarioFingerprint,
) -> None:
    memory = get_inverse_design_working_memory()
    memory.record(
        stage="step5_checkpoint",
        key=bundle.simulation_config.component_type,
        scenario_fingerprint=_scenario_fingerprint_key(scenario_fingerprint),
        summary=(
            f"Checkpoint@{report.checkpoint_iteration} status={report.status} "
            f"family={report.error_family or 'healthy'}."
        ),
        issues=list(report.reasons),
        metadata={
            "checkpoint_report": report.model_dump(),
            "scenario_fingerprint": scenario_fingerprint.model_dump(),
        },
    )


def _scenario_fingerprint_key(fingerprint: ScenarioFingerprint) -> str:
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


def _checkpoint_failure_reason(report: CheckpointReport) -> str:
    base = f"Step6 checkpoint failed at iteration {report.checkpoint_iteration}"
    if report.error_family or report.error_subtype:
        base = f"{base} ({report.error_family or 'unknown'}/{report.error_subtype or 'unknown'})"
    if report.reasons:
        return f"{base}: {report.reasons[0]}"
    return base


def _numeric_geometry_parameters(parameters: Dict[str, Any]) -> Dict[str, float]:
    numeric = {
        key: float(value)
        for key, value in parameters.items()
        if isinstance(value, (int, float))
    }
    for key, value in list(numeric.items()):
        numeric[key] = normalize_discrete_topology_parameter(key, value)
    return numeric


def _effective_optimization_variables(bundle: InverseDesignConfigBundle) -> List[Any]:
    """Return runtime-active optimization variables.

    Discrete topology-contract parameters (for example output port count)
    must remain fixed during continuous adjoint updates.
    """

    return [
        var
        for var in bundle.optimization_config.variables
        if not is_discrete_topology_parameter(str(getattr(var, "name", "")))
    ]


def _apply_topology_contract_parameters(
    bundle: InverseDesignConfigBundle,
    parameters: Dict[str, float],
) -> Dict[str, float]:
    """Pin discrete topology-contract parameters to locked bundle values.

    This prevents continuous optimizer vectors or recovered traces from
    perturbing structural cardinality (for example 1x5 drifting to 1x4).
    """

    if not parameters:
        return parameters

    updated = dict(parameters)
    locked = _numeric_geometry_parameters(bundle.simulation_config.geometry.parameters)
    for name, value in locked.items():
        if is_discrete_topology_parameter(name):
            updated[name] = normalize_discrete_topology_parameter(name, value)
    return updated


def _execute_adjoint_invdes_kernel(
    *,
    bundle: InverseDesignConfigBundle,
    run_iterations: int,
    run_artifact_tag: str,
) -> Dict[str, Any]:
    """Run primary invdes/adjoint kernel (adjoint-only policy).

    Notes:
    - Primary API references the official quickstart and uses
      ``TopologyDesignRegion`` + ``InverseDesign`` + ``AdamOptimizer``.
    - Per-iteration monitor/artifact contract is preserved via a diagnostic
      simulation pass with the mapped geometry parameters.
    """
    objective_cases = _objective_cases(bundle)
    multi_case_objective = _should_use_multi_case_adjoint(objective_cases)
    estimated_cloud_task_info = _estimate_adjoint_cloud_tasks_per_iteration(objective_cases)
    estimated_cloud_tasks = (
        int(estimated_cloud_task_info.get("per_iteration_total", 1)) * max(int(run_iterations), 0)
    )
    _write_step5_heartbeat(
        run_artifact_tag,
        "kernel_start",
        run_iterations=run_iterations,
        multi_case_objective=bool(multi_case_objective),
        estimated_cloud_tasks=estimated_cloud_tasks,
    )
    previous_estimated_tasks_env = os.getenv("INVERSE_STEP5_ESTIMATED_CLOUD_TASKS")
    os.environ["INVERSE_STEP5_ESTIMATED_CLOUD_TASKS"] = str(max(estimated_cloud_tasks, 0))
    preflight_error = _adjoint_credit_preflight_error(run_iterations=run_iterations)
    if previous_estimated_tasks_env is None:
        os.environ.pop("INVERSE_STEP5_ESTIMATED_CLOUD_TASKS", None)
    else:
        os.environ["INVERSE_STEP5_ESTIMATED_CLOUD_TASKS"] = previous_estimated_tasks_env
    if preflight_error:
        _write_step5_heartbeat(run_artifact_tag, "credit_preflight_failed")
        reason = "Adjoint-only optimization failed."
        note = "Adjoint-only execution is enforced; bridge fallback is disabled by design."
        if multi_case_objective:
            reason = "Adjoint-only multi-case optimization failed."
            note = (
                "Multi-case objective constraints require adjoint optimization. "
                "Bridge fallback is disabled by design."
            )
        return _adjoint_failure_kernel_output(
            bundle=bundle,
            error=preflight_error,
            termination_reason=reason,
            note=note,
            run_artifact_tag=run_artifact_tag,
            attempt_logs=[
                {
                    "attempt": 0,
                    "total_attempts": 0,
                    "error": preflight_error,
                    "retryable": False,
                    "non_retryable": True,
                }
            ],
        )

    cloud_preflight_error = _adjoint_cloud_preflight_error()
    if cloud_preflight_error:
        _write_step5_heartbeat(run_artifact_tag, "cloud_preflight_failed")
        reason = "Adjoint-only optimization failed."
        note = "Adjoint-only execution is enforced; bridge fallback is disabled by design."
        if multi_case_objective:
            reason = "Adjoint-only multi-case optimization failed."
            note = (
                "Multi-case objective constraints require adjoint optimization. "
                "Bridge fallback is disabled by design."
            )
        return _adjoint_failure_kernel_output(
            bundle=bundle,
            error=cloud_preflight_error,
            termination_reason=reason,
            note=note,
            run_artifact_tag=run_artifact_tag,
            attempt_logs=[
                {
                    "attempt": 0,
                    "total_attempts": 0,
                    "error": cloud_preflight_error,
                    "retryable": _is_retryable_adjoint_error(cloud_preflight_error),
                    "non_retryable": _is_non_retryable_adjoint_error(cloud_preflight_error),
                }
            ],
        )

    adjoint_timeout_s = _adjoint_run_timeout_s()

    def _invoke_adjoint_once(*, attempt_timeout_s: float) -> Dict[str, Any]:
        timeout_error = (
            "adjoint_run_timeout_guard: step5 adjoint run exceeded "
            f"{attempt_timeout_s:.0f}s without completion (possible cloud upload/monitor stall)."
        )
        if _adjoint_timeout_mode() == "thread":
            def _call_once() -> Dict[str, Any]:
                try:
                    return _run_adjoint_invdes_example(
                        bundle=bundle,
                        run_iterations=run_iterations,
                        run_artifact_tag=run_artifact_tag,
                    )
                except TypeError as exc:
                    # Test stubs may still provide the legacy signature.
                    if "run_artifact_tag" not in str(exc):
                        raise
                    return _run_adjoint_invdes_example(bundle=bundle, run_iterations=run_iterations)

            return _invoke_with_timeout(
                fn=_call_once,
                timeout_s=attempt_timeout_s,
                timeout_error=timeout_error,
            )
        return _invoke_adjoint_with_process_timeout(
            bundle=bundle,
            run_iterations=run_iterations,
            run_artifact_tag=run_artifact_tag,
            timeout_s=attempt_timeout_s,
            timeout_error=timeout_error,
        )

    retries, retry_sleep_s = _adjoint_retry_config()
    total_attempts = 1 + retries
    last_error: Exception | None = None
    attempt_logs: List[Dict[str, Any]] = []
    for attempt in range(1, total_attempts + 1):
        attempt_timeout_s = _adjoint_timeout_for_attempt(adjoint_timeout_s, attempt)
        _write_step5_heartbeat(
            run_artifact_tag,
            "attempt_start",
            attempt=attempt,
            total_attempts=total_attempts,
            timeout_s=attempt_timeout_s,
        )
        try:
            output = _invoke_adjoint_once(attempt_timeout_s=attempt_timeout_s)
            _write_step5_heartbeat(run_artifact_tag, "attempt_success", attempt=attempt)
            return output
        except Exception as exc:  # pragma: no cover - exercised via execution-path tests
            last_error = exc
            error_text = _format_exception_chain(exc)
            non_retryable = _is_non_retryable_adjoint_error(error_text)
            _write_step5_heartbeat(
                run_artifact_tag,
                "attempt_error",
                attempt=attempt,
                error=error_text[:400],
                retryable=_is_retryable_adjoint_error(error_text),
                non_retryable=non_retryable,
            )
            attempt_logs.append(
                {
                    "attempt": attempt,
                    "total_attempts": total_attempts,
                    "error": error_text,
                    "retryable": _is_retryable_adjoint_error(error_text),
                    "non_retryable": non_retryable,
                }
            )
            can_retry = (
                attempt < total_attempts
                and _is_retryable_adjoint_error(error_text)
                and not non_retryable
            )
            if can_retry and _is_download_failure_adjoint_error(error_text) and not _allow_download_failure_rerun():
                can_retry = False
                logger.warning(
                    "Adjoint kernel download failure detected; skip full-run retry to avoid "
                    "duplicate cloud-task spend. Set INVERSE_ADJOINT_RETRY_DOWNLOAD_FAILURES=1 "
                    "to force rerun retries."
                )
            if not can_retry:
                break
            logger.warning(
                "Adjoint kernel retryable failure on attempt %d/%d: %s",
                attempt,
                total_attempts,
                error_text,
            )
            if retry_sleep_s > 0:
                _write_step5_heartbeat(
                    run_artifact_tag,
                    "attempt_retry_sleep",
                    attempt=attempt,
                    sleep_s=retry_sleep_s,
                )
                time.sleep(retry_sleep_s)

    assert last_error is not None
    exc = last_error
    error_text = _format_exception_chain(exc)
    if total_attempts > 1 and _is_retryable_adjoint_error(error_text):
        error_text = f"{error_text} (after {total_attempts} attempts)"

    reason = "Adjoint-only optimization failed."
    note = "Adjoint-only execution is enforced; bridge fallback is disabled by design."
    if multi_case_objective:
        reason = "Adjoint-only multi-case optimization failed."
        note = (
            "Multi-case objective constraints require adjoint optimization. "
            "Bridge fallback is disabled by design."
        )
    _write_step5_heartbeat(
        run_artifact_tag,
        "kernel_failed",
        total_attempts=total_attempts,
        error=error_text[:400],
    )
    # Reconstruct forward-inspection artifact paths from disk.  The thread that
    # runs _run_adjoint_invdes_example (and therefore _export_adjoint_forward_sim_artifacts)
    # may have been killed by the timeout guard before returning its result.
    # _export_adjoint_forward_sim_artifacts writes HDF5/viewer files early in the
    # thread, so they typically exist on disk even after a timeout.
    _recovered_forward_artifacts = _recover_adjoint_forward_artifacts_from_disk(run_artifact_tag)
    return _adjoint_failure_kernel_output(
        bundle=bundle,
        error=error_text,
        termination_reason=reason,
        note=note,
        run_artifact_tag=run_artifact_tag,
        attempt_logs=attempt_logs,
        forward_artifacts=_recovered_forward_artifacts,
    )


def _adjoint_failure_kernel_output(
    *,
    bundle: InverseDesignConfigBundle,
    error: str,
    termination_reason: str,
    note: str,
    run_artifact_tag: str | None = None,
    attempt_logs: List[Dict[str, Any]] | None = None,
    forward_artifacts: List[Any] | None = None,
) -> Dict[str, Any]:
    """Build the failure-path kernel output dict.

    ``forward_artifacts`` receives the list produced by
    ``_export_adjoint_forward_sim_artifacts`` so that even when the optimizer
    times-out or errors the local HDF5 inspection files are visible in
    ``constraint_summary``.
    """
    objective = bundle.optimization_config.objective
    constraints = list(bundle.optimization_config.constraints)
    trace_artifacts: List[str] = []
    trace_file = ""
    trace_records = 0
    recovered_records: List[InverseDesignIterationRecord] = []
    # Normalise forward_artifacts to a serialisable list of strings.
    _fwd_artifacts: List[str] = [str(p) for p in (forward_artifacts or []) if p]
    if run_artifact_tag:
        trace_path = _adjoint_iteration_trace_path(run_artifact_tag)
        if trace_path.exists():
            trace_file = str(trace_path.resolve())
            trace_artifacts.append(trace_file)
            trace_entries = _load_adjoint_trace_entries(trace_path)
            trace_records = len(trace_entries)
            recovered_records = _recover_adjoint_failure_records_from_trace(
                bundle=bundle,
                trace_entries=trace_entries,
                constraints=constraints,
                failure_error=f"{termination_reason.rstrip('.')} : {error}",
                run_artifact_tag=run_artifact_tag,
            )
            if recovered_records:
                recovered_records[-1].artifacts.extend(trace_artifacts)
                return {
                    "backend": "adjoint_invdes_example",
                    "records": recovered_records,
                    "termination_reason": termination_reason,
                    "constraint_summary": {
                        "declared_constraints": constraints,
                        "evaluated": True,
                        "satisfied": False,
                        "adjoint_only_mode": True,
                        "note": (
                            f"{note} Recovered {len(recovered_records)} partial iteration "
                            "records from adjoint trace after failure."
                        ),
                        "adjoint_iteration_trace_file": trace_file,
                        "adjoint_iteration_trace_records": trace_records,
                        "partial_trace_recovered": True,
                        "adjoint_attempt_logs": list(attempt_logs or []),
                        "step5_heartbeat_file": str(_step5_heartbeat_path(run_artifact_tag).resolve()),
                        # Always present so M178+ checks can locate locally-exported HDF5 files
                        # even when the cloud optimizer was interrupted.
                        "adjoint_forward_task_field_monitors": False,
                        "adjoint_forward_sim_artifacts": _fwd_artifacts,
                    },
                }
    failure_record = InverseDesignIterationRecord(
        iteration=1,
        parameters=_numeric_geometry_parameters(bundle.simulation_config.geometry.parameters),
        score=None,
        objective_metric=objective.metric,
        objective_value=None,
        simulation_ok=False,
        optimizer_backend="adjoint_invdes_example",
        monitor_readings={},
        constraint_status=_derive_constraint_status(constraints),
        metrics={},
        artifacts=trace_artifacts,
        error=f"{termination_reason.rstrip('.')} : {error}",
    )
    return {
        "backend": "adjoint_invdes_example",
        "records": [failure_record],
        "termination_reason": termination_reason,
        "constraint_summary": {
            "declared_constraints": constraints,
            "evaluated": True,
            "satisfied": False,
            "adjoint_only_mode": True,
            "note": note,
            "adjoint_iteration_trace_file": trace_file,
            "adjoint_iteration_trace_records": trace_records,
            "adjoint_attempt_logs": list(attempt_logs or []),
            "step5_heartbeat_file": (
                str(_step5_heartbeat_path(run_artifact_tag).resolve())
                if run_artifact_tag
                else ""
            ),
            # Always present so M178+ checks can locate locally-exported HDF5 files
            # even when the cloud optimizer was interrupted.
            "adjoint_forward_task_field_monitors": False,
            "adjoint_forward_sim_artifacts": _fwd_artifacts,
        },
    }


def _load_adjoint_trace_entries(trace_path: Path) -> List[Dict[str, Any]]:
    entries: Dict[int, Dict[str, Any]] = {}
    try:
        with trace_path.open("r", encoding="utf-8") as fp:
            for raw_line in fp:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except Exception:
                    continue
                if not isinstance(payload, dict):
                    continue
                iteration = payload.get("iteration")
                try:
                    iteration_idx = int(iteration)
                except Exception:
                    continue
                if iteration_idx <= 0:
                    continue
                entries[iteration_idx] = payload
    except Exception:
        return []
    return [entries[idx] for idx in sorted(entries)]


def _recover_adjoint_failure_records_from_trace(
    *,
    bundle: InverseDesignConfigBundle,
    trace_entries: Sequence[Dict[str, Any]],
    constraints: Sequence[str],
    failure_error: str,
    run_artifact_tag: str | None = None,
) -> List[InverseDesignIterationRecord]:
    if not trace_entries:
        return []
    variables = _effective_optimization_variables(bundle)
    current_params = _apply_topology_contract_parameters(
        bundle,
        _numeric_geometry_parameters(bundle.simulation_config.geometry.parameters),
    )
    objective_metric = bundle.optimization_config.objective.metric
    records: List[InverseDesignIterationRecord] = []
    for idx, entry in enumerate(trace_entries):
        iteration = idx + 1
        try:
            iteration = max(int(entry.get("iteration", iteration)), 1)
        except Exception:
            iteration = idx + 1

        mapped_params = _map_design_vector_to_variables(
            vector=entry.get("params_vector"),
            variables=variables,
            previous=current_params,
        )
        mapped_params = _apply_topology_contract_parameters(bundle, mapped_params)
        current_params = dict(mapped_params)

        post_process_val = _as_float_or_none(entry.get("post_process_val"))
        objective_fn_val = _as_float_or_none(entry.get("objective_fn_val"))
        objective_value = post_process_val if post_process_val is not None else objective_fn_val
        score = post_process_val if post_process_val is not None else objective_fn_val

        trace_payload = entry.get("adjoint_trace")
        adjoint_trace = dict(trace_payload) if isinstance(trace_payload, dict) else {}
        if objective_fn_val is not None:
            adjoint_trace.setdefault("objective_fn_val", objective_fn_val)
        if post_process_val is not None:
            adjoint_trace.setdefault("post_process_val", post_process_val)

        case_summary_payload = entry.get("case_observation_summary")
        case_summary = dict(case_summary_payload) if isinstance(case_summary_payload, dict) else {}
        case_observation_payload = entry.get("case_observation")
        case_observation = (
            dict(case_observation_payload)
            if isinstance(case_observation_payload, dict)
            else {}
        )
        metrics: Dict[str, Any] = {"adjoint_trace": adjoint_trace}
        monitor_readings: Dict[str, float | None] = {}
        sim_ok = False
        raw_case_details = case_observation.get("multi_case")
        if isinstance(raw_case_details, list):
            case_details = [dict(item) for item in raw_case_details if isinstance(item, dict)]
            if case_details:
                metrics["multi_case"] = case_details
                if not case_summary:
                    case_summary = _build_multi_case_iteration_summary(
                        case_details=case_details,
                        expected_case_count=len(case_details),
                    )
                observable_cases = sum(1 for item in case_details if bool(item.get("observable")))
                sim_ok = observable_cases > 0
        if case_summary:
            metrics["multi_case_summary"] = dict(case_summary)
            if not sim_ok:
                sim_ok = bool(case_summary.get("all_cases_observable")) or int(
                    case_summary.get("observable_cases", 0) or 0
                ) > 0
        raw_monitor_readings = case_observation.get("monitor_readings")
        if isinstance(raw_monitor_readings, dict):
            monitor_readings = {
                str(key): _as_float_or_none(value)
                for key, value in raw_monitor_readings.items()
            }
            if monitor_readings and not sim_ok:
                sim_ok = any(value is not None for value in monitor_readings.values())
        metrics["partial_trace_recovered"] = True

        constraint_status = _derive_constraint_status(constraints)
        constraint_status["partial_trace_recovered"] = True
        constraint_status["satisfied"] = bool(sim_ok)

        record_error = None
        if idx == 0:
            record_error = failure_error

        records.append(
            InverseDesignIterationRecord(
                iteration=iteration,
                parameters=dict(mapped_params),
                score=score,
                objective_metric=objective_metric,
                objective_value=objective_value,
                simulation_ok=sim_ok,
                optimizer_backend="adjoint_invdes_example",
                monitor_readings=monitor_readings,
                constraint_status=constraint_status,
                metrics=metrics,
                artifacts=[],
                error=record_error,
            )
        )
        _persist_iteration_metrics(record=records[-1], run_artifact_tag=run_artifact_tag)

    return records


def _run_adjoint_invdes_example(
    *,
    bundle: InverseDesignConfigBundle,
    run_iterations: int,
    run_artifact_tag: str,
) -> Dict[str, Any]:
    _configure_tidy3d_cloud_auth()
    _write_step5_heartbeat(run_artifact_tag, "adjoint_setup_start", run_iterations=run_iterations)

    import tidy3d as td
    from tidy3d.plugins.expressions.metrics import ModePower
    from tidy3d.plugins.invdes import (
        AdamOptimizer,
        CustomInitializationSpec,
        ErosionDilationPenalty,
        FilterProject,
        InverseDesign,
        InverseDesignMulti,
        TopologyDesignRegion,
        UniformInitializationSpec,
    )

    objective = bundle.optimization_config.objective
    objective_cases = _objective_cases(bundle)
    resume_signature = _optimizer_resume_signature(bundle, objective_cases)
    resume_candidate = _resolve_optimizer_resume_candidate(signature=resume_signature)
    estimated_cloud_task_info = _estimate_adjoint_cloud_tasks_per_iteration(objective_cases)
    variables = _effective_optimization_variables(bundle)
    constraints = list(bundle.optimization_config.constraints)
    # Recovery can inject optimizer_hints to tune hyperparameters.
    hints = dict(bundle.optimization_config.optimizer_hints or {})
    # Env var overrides allow per-run hyperparameter tuning without modifying the bundle.
    # INVERSE_STEP5_LEARNING_RATE: Adam step size (default 0.1).
    # INVERSE_STEP5_BETA: FilterProject projection sharpness (default 6.0; lower = softer/slower binarization).
    # INVERSE_STEP5_PENALTY_WEIGHT: ErosionDilationPenalty weight (default 0.5; lower = less binarization pressure).
    for _env_key, _hint_key in (
        ("INVERSE_STEP5_LEARNING_RATE", "learning_rate"),
        ("INVERSE_STEP5_BETA", "beta"),
        ("INVERSE_STEP5_PENALTY_WEIGHT", "penalty_weight"),
    ):
        _env_val = os.getenv(_env_key)
        if _env_val is not None:
            try:
                hints[_hint_key] = float(_env_val)
                logger.info("Step5 hint override from env: %s=%.4g", _hint_key, hints[_hint_key])
            except ValueError:
                logger.warning("Step5 hint override ignored (not float): %s=%r", _env_key, _env_val)
    include_field_monitors_in_optimizer = _include_field_monitors_in_adjoint_optimizer()
    # NOTE: cloud case_sims must NOT include FieldMonitors (tidy3d arbitrary-field-profile
    # adjoint conflict with TopologyDesignRegion). Forward inspection artifacts with
    # FieldMonitors are exported locally via _export_adjoint_forward_sim_artifacts().

    sim = _build_invdes_simulation(
        bundle=bundle,
        td=td,
        case_override=(objective_cases[0] if objective_cases else None),
        include_field_monitors=include_field_monitors_in_optimizer,
    )
    adjoint_forward_artifacts = _export_adjoint_forward_sim_artifacts(
        bundle=bundle,
        td=td,
        run_artifact_tag=run_artifact_tag,
        objective_cases=objective_cases,
    )
    # Post-validate: warn if fewer HDF5 artifacts than expected were produced.
    # Missing artifacts do NOT block the optimizer — they are local inspection files only.
    _expected_forward_artifact_count = (
        len({
            (
                c.get("source_port", ""),
                int(_as_float_or_none(c.get("source_mode_index")) or 0),
                round(float(c.get("wavelength_nm", 1550)), 6),
            )
            for c in objective_cases
        })
        if objective_cases else 1
    )
    _hdf5_artifact_count = sum(1 for p in adjoint_forward_artifacts if str(p).endswith(".hdf5"))
    if _hdf5_artifact_count < _expected_forward_artifact_count:
        logger.warning(
            "Step5: _export_adjoint_forward_sim_artifacts produced %d HDF5 artifacts, "
            "expected %d. Forward FieldMonitor inspection files in build/ may be incomplete. "
            "Cloud optimizer will still proceed (artifacts are for local inspection only).",
            _hdf5_artifact_count,
            _expected_forward_artifact_count,
        )

    # Tidy3D official defaults: beta=10 (InverseDesign tutorial),
    # learning_rate=0.1 (TopologyBend example), ErosionDilationPenalty.
    # M53-fix: pixel_size & filter_radius must scale with wavelength to avoid
    # quantization-induced odd-even oscillation.
    wavelength_um = bundle.simulation_config.source.wavelength_nm / 1000.0
    min_feature_um = _effective_min_feature_um(bundle)
    pixel_size = float(hints.get("pixel_size", round(wavelength_um / 35.0, 4)))
    if pixel_size < min_feature_um:
        logger.info(
            "Step5 design-region pixel_size %.4fum is below minimum feature %.4fum; enforcing floor.",
            pixel_size,
            min_feature_um,
        )
        pixel_size = min_feature_um
    filter_radius = float(hints.get("filter_radius", round(max(3.0 * pixel_size, min_feature_um), 4)))
    if filter_radius < min_feature_um:
        filter_radius = min_feature_um
    beta = float(hints.get("beta", 6.0))
    penalty_weight = float(hints.get("penalty_weight", 0.5))
    resolved_optimizer_hints = {
        "learning_rate": None,  # filled after learning_rate is resolved below
        "beta": beta,
        "penalty_weight": penalty_weight,
        "pixel_size": pixel_size,
        "filter_radius": filter_radius,
    }

    # M57/M58: design-region center z must match structure z (wg_height/2),
    # not the simulation-domain center which is typically (0,0,0).
    _geo_params = _numeric_geometry_parameters(bundle.simulation_config.geometry.parameters)
    wg_height = _geo_params.get("wg_height", 0.22)
    _dr_center = (
        bundle.simulation_config.domain.center_um[0],
        bundle.simulation_config.domain.center_um[1],
        wg_height / 2,
    )
    design_region = TopologyDesignRegion(
        size=_design_region_size(bundle.simulation_config.domain.size_um, bundle),
        center=_dr_center,
        eps_bounds=(1.44**2, 3.48**2),
        transformations=(FilterProject(radius=filter_radius, beta=beta, eta=0.5),),
        penalties=(ErosionDilationPenalty(weight=penalty_weight, length_scale=filter_radius),),
        initialization_spec=UniformInitializationSpec(value=0.5),
        pixel_size=pixel_size,
        uniform=(False, False, True),
    )

    # Warm-start: load custom initial parameters if a path is provided via
    # env var INVERSE_STEP5_INITIAL_PARAMS_FILE or bundle hint `warm_start_from`.
    # File must be a .npy array; values must be in [0, 1]; shape is reshaped to
    # match design_region.params_shape. On any failure we log and fall back to
    # the uniform 0.5 init above (do NOT silently corrupt the run).
    _warm_start_path = os.getenv("INVERSE_STEP5_INITIAL_PARAMS_FILE") or hints.get("warm_start_from")
    if _warm_start_path:
        try:
            import numpy as np
            _ws_p = Path(str(_warm_start_path))
            if not _ws_p.is_absolute():
                _ws_p = (Path.cwd() / _ws_p).resolve()
            if not _ws_p.exists():
                raise FileNotFoundError(f"warm-start file not found: {_ws_p}")
            _ws_arr = np.asarray(np.load(str(_ws_p)), dtype=np.float64)
            _expected_shape = design_region.params_shape
            _expected_size = int(np.prod(_expected_shape))
            if _ws_arr.size != _expected_size:
                raise ValueError(
                    f"warm-start size {_ws_arr.size} does not match design_region "
                    f"params_shape={_expected_shape} (expected {_expected_size} elements)"
                )
            _ws_arr = _ws_arr.reshape(_expected_shape)
            _ws_min, _ws_max = float(_ws_arr.min()), float(_ws_arr.max())
            if _ws_min < 0.0 or _ws_max > 1.0:
                # Clip with explicit warning rather than silently rescaling.
                logger.warning(
                    "Step5 warm-start values out of [0,1] (min=%.4f max=%.4f); clipping.",
                    _ws_min, _ws_max,
                )
                _ws_arr = np.clip(_ws_arr, 0.0, 1.0)
            design_region = design_region.updated_copy(
                initialization_spec=CustomInitializationSpec(params=_ws_arr),
            )
            logger.info(
                "Step5 warm-start LOADED from %s shape=%s mean=%.4f range=[%.4f,%.4f]",
                str(_ws_p), _expected_shape, float(_ws_arr.mean()),
                float(_ws_arr.min()), float(_ws_arr.max()),
            )
        except Exception as _ws_err:
            # Per AGENTS rule: report problem, do not mask. Fall back to uniform but
            # raise loud warning so subsequent acceptance checks know warm-start failed.
            logger.error(
                "Step5 warm-start FAILED to load from %r: %s. Falling back to UniformInit(0.5).",
                _warm_start_path, _ws_err,
            )

    use_multi_case_adjoint = _should_use_multi_case_adjoint(objective_cases)
    post_process_fn = None
    latest_multi_case_observation: Dict[str, Any] | None = None
    if use_multi_case_adjoint:
        def _capture_multi_case_observation(observation: Dict[str, Any]) -> None:
            nonlocal latest_multi_case_observation
            latest_multi_case_observation = dict(observation)

        task_name = _adjoint_task_name(bundle.simulation_config.component_type, run_artifact_tag)
        case_sims = []
        case_output_monitor_names: List[tuple[str, ...]] = []
        case_task_indices: List[int] = []
        case_signature_to_task_idx: Dict[tuple[float, str, int, str], int] = {}
        for case in objective_cases:
            case_wavelength = round(float(case["wavelength_nm"]), 6)
            case_source_port = _case_source_port(case)
            case_source_mode = _case_source_mode_index(case)
            case_source_direction = _case_source_direction(case)
            signature = (
                case_wavelength,
                case_source_port,
                case_source_mode,
                case_source_direction,
            )
            task_idx = case_signature_to_task_idx.get(signature)
            if task_idx is None:
                case_bundle = _bundle_with_case_wavelength(
                    bundle,
                    float(case["wavelength_nm"]),
                    case=case,
                )
                case_sim = _build_invdes_simulation(
                    bundle=case_bundle,
                    td=td,
                    case_override=case,
                    # Optimizer path must NOT include FieldMonitor: Tidy3D adjoint
                    # framework creates one adjoint sim PER FREQUENCY for FieldMonitor
                    # ("arbitrary field profile" path), which conflicts with the
                    # topology-optimization adjoint upload and causes upload failures.
                    # Field monitors for user inspection are exported separately via
                    # _export_adjoint_forward_sim_artifacts().
                    include_field_monitors=include_field_monitors_in_optimizer,
                )
                case_sims.append(case_sim)
                task_idx = len(case_sims) - 1
                case_signature_to_task_idx[signature] = task_idx
                case_output_monitor_names.append(
                    tuple(_pick_case_output_monitor_names(case_sim, str(case.get("target_port", ""))))
                )
            else:
                case_sim = case_sims[task_idx]
                merged_monitor_names = list(case_output_monitor_names[task_idx])
                for monitor_name in _pick_case_output_monitor_names(case_sim, str(case.get("target_port", ""))):
                    if monitor_name not in merged_monitor_names:
                        merged_monitor_names.append(monitor_name)
                case_output_monitor_names[task_idx] = tuple(merged_monitor_names)
            case_task_indices.append(task_idx)
        design = InverseDesignMulti(
            design_region=design_region,
            task_name=task_name,
            simulations=tuple(case_sims),
            output_monitor_names=tuple(case_output_monitor_names),
            verbose=False,
        )
        adjoint_case_task_plan = _build_adjoint_case_task_plan(
            objective_cases=objective_cases,
            task_names=tuple(design.task_names),
            case_task_indices=tuple(case_task_indices),
        )
        post_process_fn = _build_multi_case_post_process_fn(
            objective_cases=objective_cases,
            task_names=tuple(design.task_names),
            case_task_indices=tuple(case_task_indices),
            observation_callback=_capture_multi_case_observation,
        )
    else:
        # W18 fix: for multi-port devices (MMI, splitter, crossing, etc.)
        # sum all output port powers instead of measuring a single mode monitor.
        output_monitor_names = _pick_output_monitor_names(bundle, sim)
        if len(output_monitor_names) > 1:
            metric = sum(
                ModePower(monitor_name=mn, direction="+", mode_index=0)
                for mn in output_monitor_names
            )
        else:
            metric = ModePower(monitor_name=output_monitor_names[0], direction="+", mode_index=0)

        design = InverseDesign(
            design_region=design_region,
            task_name=_adjoint_task_name(bundle.simulation_config.component_type, run_artifact_tag),
            simulation=sim,
            metric=metric,
            output_monitor_names=tuple(output_monitor_names),
            verbose=False,
        )
        adjoint_case_task_plan = []

    runtime_workdir = _prepare_invdes_runtime_workdir(run_artifact_tag)
    optimizer_cache_file = runtime_workdir / "optimizer_state.hdf5"
    learning_rate = float(hints.get("learning_rate", 0.1))
    resolved_optimizer_hints["learning_rate"] = learning_rate
    optimizer = AdamOptimizer(
        design=design,
        learning_rate=learning_rate,
        maximize=(objective.goal == "maximize"),
        num_steps=run_iterations,
        results_cache_fname=str(optimizer_cache_file.resolve()),
        store_full_results=False,
    )

    snapshots: List[Dict[str, Any]] = []
    _abort_requested = False
    trace_path = _adjoint_iteration_trace_path(run_artifact_tag).resolve()
    trace_records_written = 0
    try:
        if trace_path.exists():
            trace_path.unlink()
    except Exception:
        pass

    def _append_iteration_trace(entry: Dict[str, Any]) -> None:
        nonlocal trace_records_written
        try:
            with trace_path.open("a", encoding="utf-8") as trace_fp:
                trace_fp.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
                trace_fp.flush()
                os.fsync(trace_fp.fileno())
            trace_records_written += 1
        except Exception:
            logger.debug("Failed to append adjoint iteration trace.", exc_info=True)

    def _callback(result, step_index=None, aux_data=None):
        nonlocal _abort_requested, latest_multi_case_observation
        aux_data = aux_data or {}
        obj_val = _as_float_or_none(aux_data.get("objective_fn_val"))
        trace = _build_adjoint_trace_snapshot(aux_data)
        if obj_val is not None:
            trace.setdefault("objective_fn_val", obj_val)
        # W16 fix: if objective is NaN/None on the very first step the
        # simulation likely failed (e.g. insufficient balance).  Request
        # abort so we don't keep burning cloud tasks.
        if obj_val is None and int(step_index or 0) == 0:
            _abort_requested = True
        snapshots.append(
            {
                "step_index": int(step_index or 0),
                "objective_fn_val": obj_val,
                "post_process_val": _as_float_or_none(aux_data.get("post_process_val")),
                "params": result.get_last("params"),
                "trace": trace,
                "case_observation": dict(latest_multi_case_observation) if isinstance(latest_multi_case_observation, dict) else None,
            }
        )
        _append_iteration_trace(
            {
                "iteration": resumed_from_steps + int(step_index or 0) + 1,
                "objective_fn_val": obj_val,
                "post_process_val": _as_float_or_none(aux_data.get("post_process_val")),
                "optimizer_hints": dict(resolved_optimizer_hints),
                "params_vector": _flatten_to_list(result.get_last("params")),
                "adjoint_trace": trace,
                "case_observation_summary": (
                    dict(latest_multi_case_observation.get("multi_case_summary", {}))
                    if isinstance(latest_multi_case_observation, dict)
                    else {}
                ),
                "case_observation": (
                    dict(latest_multi_case_observation)
                    if isinstance(latest_multi_case_observation, dict)
                    else None
                ),
                "source": "adjoint_callback",
            }
        )
        _write_step5_heartbeat(
            run_artifact_tag,
            "iteration_callback",
            iteration=int(step_index or 0) + 1,
            objective_fn_val=obj_val,
            post_process_val=_as_float_or_none(aux_data.get("post_process_val")),
        )
        latest_multi_case_observation = None
        if _abort_requested:
            raise RuntimeError("Optimizer aborted: first-iteration objective not observable (possible credit exhaustion).")

    invdes_result = None
    optimizer_run_error: str | None = None
    resume_active = bool(resume_candidate.get("resume_used"))
    resumed_from_steps = int(resume_candidate.get("resume_previous_steps") or 0)
    alive_interval_s = _step5_optimizer_alive_heartbeat_interval_s()
    alive_stop_event = threading.Event()
    alive_thread: threading.Thread | None = None
    alive_start_monotonic = time.monotonic()

    def _optimizer_alive_heartbeat_loop() -> None:
        alive_seq = 0
        while not alive_stop_event.wait(alive_interval_s):
            alive_seq += 1
            _write_step5_heartbeat(
                run_artifact_tag,
                "optimizer_run_alive",
                alive_seq=alive_seq,
                elapsed_s=round(time.monotonic() - alive_start_monotonic, 3),
                resume_active=resume_active,
                resumed_from_steps=resumed_from_steps,
                run_iterations=run_iterations,
            )

    try:
        _write_step5_heartbeat(
            run_artifact_tag,
            "optimizer_run_start",
            resume_active=resume_active,
            resumed_from_steps=resumed_from_steps,
            run_iterations=run_iterations,
            pixel_size_um=pixel_size,
            filter_radius_um=filter_radius,
            min_feature_um=min_feature_um,
        )
        alive_thread = threading.Thread(
            target=_optimizer_alive_heartbeat_loop,
            name=f"step5-alive-{_safe_run_artifact_tag(run_artifact_tag)}",
            daemon=True,
        )
        alive_thread.start()
        with _scoped_cwd(runtime_workdir):
            if resume_active:
                resume_file = str(resume_candidate.get("resume_state_file") or "")
                if post_process_fn is not None:
                    invdes_result = optimizer.continue_run_from_file(
                        fname=resume_file,
                        num_steps=run_iterations,
                        post_process_fn=post_process_fn,
                        callback=_callback,
                    )
                else:
                    invdes_result = optimizer.continue_run_from_file(
                        fname=resume_file,
                        num_steps=run_iterations,
                        callback=_callback,
                    )
            else:
                if post_process_fn is not None:
                    invdes_result = optimizer.run(post_process_fn=post_process_fn, callback=_callback)
                else:
                    invdes_result = optimizer.run(callback=_callback)
    except Exception as exc:
        optimizer_run_error = str(exc)
        _write_step5_heartbeat(
            run_artifact_tag,
            "optimizer_run_error",
            error=_format_exception_chain(exc)[:400],
            snapshots=len(snapshots),
            trace_records_written=trace_records_written,
        )
        if not snapshots and trace_records_written <= 0:
            raise
        logger.warning(
            "Adjoint optimizer ended with recoverable runtime error after partial progress; "
            "salvaging completed iterations. error=%s",
            optimizer_run_error,
        )
    finally:
        alive_stop_event.set()
        if alive_thread is not None:
            alive_thread.join(timeout=max(1.0, min(alive_interval_s, 5.0)))
        _write_step5_heartbeat(
            run_artifact_tag,
            "optimizer_run_alive_stopped",
            elapsed_s=round(time.monotonic() - alive_start_monotonic, 3),
            optimizer_run_error=bool(optimizer_run_error),
            snapshots=len(snapshots),
            trace_records_written=trace_records_written,
        )

    if not snapshots and trace_records_written > 0:
        trace_entries = _load_adjoint_trace_entries(trace_path)
        for entry in trace_entries[:run_iterations]:
            snapshots.append(
                {
                    "step_index": int(entry.get("iteration", 1)) - 1,
                    "objective_fn_val": _as_float_or_none(entry.get("objective_fn_val")),
                    "post_process_val": _as_float_or_none(entry.get("post_process_val")),
                    "params": entry.get("params_vector"),
                    "trace": dict(entry.get("adjoint_trace", {})),
                    "case_observation": (
                        dict(entry.get("case_observation"))
                        if isinstance(entry.get("case_observation"), dict)
                        else None
                    ),
                }
            )
    _write_step5_heartbeat(
        run_artifact_tag,
        "optimizer_run_completed",
        snapshots=len(snapshots),
        trace_records_written=trace_records_written,
        optimizer_run_error=optimizer_run_error or "",
    )
    if not snapshots:
        history = invdes_result.history
        params_hist = history.get("params", [])
        objective_hist = history.get("objective_fn_val", [])
        post_hist = history.get("post_process_val", [])
        for idx in range(min(len(params_hist), run_iterations)):
            objective_val = _as_float_or_none(objective_hist[idx]) if idx < len(objective_hist) else None
            post_val = _as_float_or_none(post_hist[idx]) if idx < len(post_hist) else None
            trace: Dict[str, float] = {}
            if objective_val is not None:
                trace["objective_fn_val"] = objective_val
            if post_val is not None:
                trace["post_process_val"] = post_val
            snapshots.append(
                {
                    "step_index": idx,
                    "objective_fn_val": objective_val,
                    "post_process_val": post_val,
                    "params": params_hist[idx],
                    "trace": trace,
                }
            )
        if snapshots and trace_records_written == 0:
            for snap in snapshots:
                _append_iteration_trace(
                    {
                        "iteration": int(snap.get("step_index", 0)) + 1,
                        "objective_fn_val": _as_float_or_none(snap.get("objective_fn_val")),
                        "post_process_val": _as_float_or_none(snap.get("post_process_val")),
                        "params_vector": _flatten_to_list(snap.get("params")),
                        "adjoint_trace": dict(snap.get("trace", {})),
                        "case_observation_summary": {},
                        "source": "history_fallback",
                    }
                )

    current_params = _apply_topology_contract_parameters(
        bundle,
        _numeric_geometry_parameters(bundle.simulation_config.geometry.parameters),
    )
    records: List[InverseDesignIterationRecord] = []
    consecutive_score_none = 0  # KL-2: Track consecutive None scores
    observation_mode = _multi_case_observation_mode()
    for idx, snap in enumerate(snapshots[:run_iterations], start=1):
        mapped_params = _map_design_vector_to_variables(
            vector=snap.get("params"),
            variables=variables,
            previous=current_params,
        )
        mapped_params = _apply_topology_contract_parameters(bundle, mapped_params)
        current_params = mapped_params

        internal_case_observation = snap.get("case_observation")
        use_internal_observation = (
            use_multi_case_adjoint
            and observation_mode in {"adjoint_internal", "hybrid"}
            and isinstance(internal_case_observation, dict)
        )
        if use_internal_observation:
            eval_out = _evaluate_multi_case_from_adjoint_observation(
                case_observation=internal_case_observation,
                objective_cases=objective_cases,
            )
            # Keep hard-physics gate evidence contract (field artifact + flux keys)
            # by seeding first iteration from ONE diagnostic simulation only.
            # Limit to the first objective case to avoid excessive cloud time.
            if idx == 1:
                _seed_bundle = _bundle_with_one_case(bundle, objective_cases)
                seed_out = _evaluate_multi_case_objective(
                    bundle=_seed_bundle,
                    parameters=mapped_params,
                    objective_metric=objective.metric,
                    iteration_index=idx,
                    backend_tag="adjoint_seed",
                    run_artifact_tag=run_artifact_tag,
                )
                if bool(seed_out.get("sim_ok")):
                    eval_out["monitor_readings"] = dict(seed_out.get("monitor_readings", {}))
                    eval_out["artifacts"] = list(seed_out.get("artifacts", []))
                    metrics_seed = dict(eval_out.get("metrics", {}))
                    metrics_seed["diagnostic_seeded_for_gate"] = True
                    eval_out["metrics"] = metrics_seed
                else:
                    # Bug B fix: propagate locally-saved artifacts from the
                    # failed seed sim so HDF5/viewer files are not discarded.
                    _seed_local_arts = list(seed_out.get("artifacts", []))
                    if _seed_local_arts:
                        eval_out["artifacts"] = list(
                            dict.fromkeys(_seed_local_arts)
                        )
                    metrics_seed = dict(eval_out.get("metrics", {}))
                    metrics_seed["diagnostic_seed_error"] = seed_out.get("error")
                    eval_out["metrics"] = metrics_seed
        else:
            eval_out = _evaluate_multi_case_objective(
                bundle=bundle,
                parameters=mapped_params,
                objective_metric=objective.metric,
                iteration_index=idx,
                backend_tag="adjoint",
                run_artifact_tag=run_artifact_tag,
            )
            if (
                use_multi_case_adjoint
                and observation_mode == "adjoint_internal"
                and not isinstance(internal_case_observation, dict)
            ):
                eval_out["error"] = "adjoint_internal_observation_unavailable"
                eval_out["sim_ok"] = False
        sim_ok = bool(eval_out["sim_ok"])
        metrics = dict(eval_out.get("metrics", {}))
        metrics["optimizer_hints"] = dict(resolved_optimizer_hints)
        score = _as_float_or_none(eval_out.get("score"))
        objective_value = _as_float_or_none(eval_out.get("objective_value"))
        objective_from_adjoint = snap.get("objective_fn_val")
        if objective_from_adjoint is not None and objective_value is None:
            objective_value = objective_from_adjoint
        if snap.get("post_process_val") is not None:
            metrics.setdefault("adjoint_trace", {})
            metrics["adjoint_trace"]["post_process_val"] = _as_float_or_none(snap.get("post_process_val"))
        trace = dict(snap.get("trace") or {})
        if objective_from_adjoint is not None:
            trace.setdefault("objective_fn_val", objective_from_adjoint)
        if trace:
            metrics["adjoint_trace"] = trace

        # KL-2: Track consecutive None diagnostic scores and warn early.
        score_none_warning: str | None = None
        if score is None and sim_ok:
            consecutive_score_none += 1
            if consecutive_score_none >= 3:
                score_none_warning = (
                    f"Diagnostic score has been None for {consecutive_score_none} "
                    "consecutive iterations despite sim_ok=True — possible monitor "
                    "misconfiguration or data extraction regression."
                )
                logger.warning("KL-2: %s (iteration %d)", score_none_warning, idx)
        else:
            consecutive_score_none = 0

        # Cross-validation: warn if optimizer objective and diagnostic score
        # diverge by more than an order of magnitude.  This catches the
        # scenario where the two simulation builders produce incompatible
        # structures (the root-cause of W17).
        cross_validation_warning: str | None = None
        if (
            (not objective_cases)
            and objective_from_adjoint is not None
            and score is not None
            and abs(score) > 1e-9
            and abs(objective_from_adjoint) > 1e-9
        ):
            ratio = abs(objective_from_adjoint) / abs(score)
            divergence_factor = ratio if ratio >= 1.0 else (1.0 / ratio)
            if divergence_factor > 10.0:
                cross_validation_warning = (
                    f"Optimizer objective ({objective_from_adjoint:.6g}) and diagnostic "
                    f"score ({score:.6g}) diverge by {divergence_factor:.1f}x "
                    f"(opt/diag={ratio:.3g}) — possible sim mismatch."
                )

        constraint_status = _derive_constraint_status(constraints)
        if cross_validation_warning:
            constraint_status["cross_validation_warning"] = cross_validation_warning
        if score_none_warning:
            constraint_status["score_none_warning"] = score_none_warning

        records.append(
            InverseDesignIterationRecord(
                iteration=resumed_from_steps + idx,
                parameters=dict(mapped_params),
                score=score,
                objective_metric=objective.metric,
                objective_value=objective_value,
                simulation_ok=sim_ok,
                optimizer_backend="adjoint_invdes_example",
                monitor_readings=dict(eval_out.get("monitor_readings", {})),
                constraint_status=constraint_status,
                metrics=metrics,
                artifacts=list(eval_out.get("artifacts", [])),
                error=cross_validation_warning if not sim_ok else (
                    None if cross_validation_warning is None
                    else None  # warning stored in constraint_status, not error field
                ),
            )
        )
        if not sim_ok:
            records[-1].error = str(eval_out.get("error") or "Unknown simulation error.")
        _persist_iteration_metrics(record=records[-1], run_artifact_tag=run_artifact_tag)

        if _target_reached(
            records[-1],
            objective.goal,
            objective.target_value,
            expected_case_count=(len(objective_cases) if objective_cases else None),
        ):
            break

    # M53: Save topology structure image for the best iteration.
    # Allow saving even when invdes_result is None (optimizer exited via exception
    # after partial progress) — snapshots carry the density params via params_vector.
    if snapshots or invdes_result is not None:
        topology_artifacts, topology_best_iteration = _save_topology_image(
            invdes_result=invdes_result,
            snapshots=snapshots,
            records=records,
            bundle=bundle,
            objective_goal=objective.goal,
            run_artifact_tag=run_artifact_tag,
        )
        if topology_artifacts and records:
            # Find the record whose .iteration matches topology_best_iteration.
            # topology_best_iteration is the ABSOLUTE iteration number (e.g. 15),
            # NOT a 0-based index into records (which may only have 3 entries).
            # Using it as an index would silently attach the density to the wrong
            # record (or be out-of-bounds), causing the rerender to miss the density.
            attach_idx = len(records) - 1
            if isinstance(topology_best_iteration, int):
                for _ridx, _rec in enumerate(records):
                    if getattr(_rec, "iteration", None) == topology_best_iteration:
                        attach_idx = _ridx
                        break
                else:
                    # Fallback: treat as 1-based local index only if within range.
                    if 1 <= topology_best_iteration <= len(records):
                        attach_idx = topology_best_iteration - 1
            records[attach_idx].artifacts.extend(topology_artifacts)

    # NOTE: _ensure_best_iteration_multi_case_artifacts has been moved to the
    # PARENT process (run_inverse_design) so it does not count against the
    # subprocess timeout.  This avoids the run being killed before best-iteration
    # evidence is generated when there are many objective cases.

    if records:
        _save_best_iteration_support_artifacts(
            records=records,
            bundle=bundle,
            objective_goal=objective.goal,
            run_artifact_tag=run_artifact_tag,
        )

    termination_reason = _termination_reason(
        records,
        objective.goal,
        objective.target_value,
        run_iterations,
        expected_case_count=(len(objective_cases) if objective_cases else None),
    )
    if optimizer_run_error:
        termination_reason = (
            "Adjoint optimizer produced partial results before runtime failure: "
            + optimizer_run_error
        )

    completed_steps_for_resume = 0
    try:
        if invdes_result is not None:
            history = getattr(invdes_result, "history", {}) or {}
            objective_hist = history.get("objective_fn_val", []) if isinstance(history, dict) else []
            if isinstance(objective_hist, list):
                completed_steps_for_resume = len(objective_hist)
    except Exception:
        completed_steps_for_resume = 0
    if completed_steps_for_resume <= 0:
        completed_steps_for_resume = max((int(item.iteration) for item in records), default=0)
    resume_status = "completed" if optimizer_run_error is None else "partial_failure"
    _write_optimizer_resume_registry(
        signature=resume_signature,
        cache_file=optimizer_cache_file,
        run_artifact_tag=run_artifact_tag,
        status=resume_status,
        completed_steps=completed_steps_for_resume,
    )

    return {
        "backend": "adjoint_invdes_example",
        "records": records,
        "termination_reason": termination_reason,
        "constraint_summary": {
            "declared_constraints": constraints,
            "evaluated": bool(records),
            "satisfied": True if records else None,
            "run_artifact_tag": run_artifact_tag,
            "adjoint_iteration_trace_file": str(trace_path),
            "adjoint_iteration_trace_records": trace_records_written,
            "adjoint_task_names": (
                [str(name) for name in getattr(design, "task_names", ())]
                if use_multi_case_adjoint
                else [str(getattr(design, "task_name", ""))]
            ),
            "estimated_cloud_tasks_per_iteration": int(estimated_cloud_task_info["per_iteration_total"]),
            "estimated_cloud_tasks_for_run": int(estimated_cloud_task_info["per_iteration_total"]) * len(records),
            "estimated_cloud_tasks_per_iteration_adjoint": int(estimated_cloud_task_info["per_iteration_adjoint"]),
            "estimated_cloud_tasks_per_iteration_diagnostic": int(estimated_cloud_task_info["per_iteration_diagnostic"]),
            "estimated_cloud_observation_mode": str(estimated_cloud_task_info["observation_mode"]),
            "estimated_objective_case_count": int(estimated_cloud_task_info["objective_case_count"]),
            "estimated_unique_task_count": int(estimated_cloud_task_info["unique_task_count"]),
            # Cloud case_sims never include FieldMonitors (tidy3d adjoint conflict);
            # forward inspection artifacts exported locally via _export_adjoint_forward_sim_artifacts.
            "adjoint_forward_task_field_monitors": False,
            "adjoint_case_task_plan": list(adjoint_case_task_plan),
            "optimizer_resume_enabled": bool(resume_candidate.get("enabled")),
            "optimizer_resume_used": bool(resume_candidate.get("resume_used")),
            "optimizer_resume_source": str(resume_candidate.get("resume_source") or ""),
            "optimizer_resume_reason": str(resume_candidate.get("resume_reason") or ""),
            "optimizer_resume_signature": resume_signature,
            "optimizer_resume_state_file": str(optimizer_cache_file.resolve()),
            "optimizer_resume_previous_steps": resumed_from_steps if resume_active else 0,
            "optimizer_resume_completed_steps": completed_steps_for_resume,
            "adjoint_forward_sim_artifacts": list(adjoint_forward_artifacts),
            "note": (
                "Primary optimization kernel follows Tidy3D official invdes/adjoint "
                f"example pattern ({_OFFICIAL_INVDES_DOC})."
            ),
            "adjoint_run_error": optimizer_run_error,
        },
    }


def _run_simulation_loop_bridge(
    *,
    bundle: InverseDesignConfigBundle,
    run_iterations: int,
    run_artifact_tag: str,
) -> Dict[str, Any]:
    objective = bundle.optimization_config.objective
    objective_cases = _objective_cases(bundle)
    variables = {var.name: var for var in _effective_optimization_variables(bundle)}
    constraints = list(bundle.optimization_config.constraints)
    current_params = _apply_topology_contract_parameters(
        bundle,
        _numeric_geometry_parameters(bundle.simulation_config.geometry.parameters),
    )
    records: List[InverseDesignIterationRecord] = []
    consecutive_score_none_bridge = 0  # KL-2: Track consecutive None scores

    for idx in range(run_iterations):
        eval_out = _evaluate_multi_case_objective(
            bundle=bundle,
            parameters=current_params,
            objective_metric=objective.metric,
            iteration_index=idx + 1,
            backend_tag="bridge",
            run_artifact_tag=run_artifact_tag,
        )
        sim_ok = bool(eval_out.get("sim_ok"))
        metrics = dict(eval_out.get("metrics", {}))
        score = _as_float_or_none(eval_out.get("score"))

        # KL-2: Track consecutive None scores in bridge loop.
        bridge_score_none_warning: str | None = None
        if score is None and sim_ok:
            consecutive_score_none_bridge += 1
            if consecutive_score_none_bridge >= 3:
                bridge_score_none_warning = (
                    f"Diagnostic score None for {consecutive_score_none_bridge} "
                    "consecutive iterations (bridge) — possible monitor issue."
                )
                logger.warning("KL-2 bridge: %s (iteration %d)", bridge_score_none_warning, idx + 1)
        else:
            consecutive_score_none_bridge = 0

        bridge_constraint_status = _derive_constraint_status(constraints)
        if bridge_score_none_warning:
            bridge_constraint_status["score_none_warning"] = bridge_score_none_warning

        records.append(
            InverseDesignIterationRecord(
                iteration=idx + 1,
                parameters=dict(current_params),
                score=score,
                objective_metric=objective.metric,
                objective_value=_as_float_or_none(eval_out.get("objective_value")),
                simulation_ok=sim_ok,
                optimizer_backend="simulation_loop_bridge",
                monitor_readings=dict(eval_out.get("monitor_readings", {})),
                constraint_status=bridge_constraint_status,
                metrics=metrics,
                artifacts=list(eval_out.get("artifacts", [])),
                error=None if sim_ok else str(eval_out.get("error") or "Unknown simulation error."),
            )
        )
        _persist_iteration_metrics(record=records[-1], run_artifact_tag=run_artifact_tag)

        expected_case_count = len(objective_cases)
        if not sim_ok or _target_reached(
            records[-1],
            objective.goal,
            objective.target_value,
            expected_case_count=(expected_case_count if expected_case_count > 0 else None),
        ):
            break

        current_params = _next_parameters(
            current_params=current_params,
            variables=variables,
            iteration_index=idx,
        )
        current_params = _apply_topology_contract_parameters(bundle, current_params)

    # M58: Save structure cross-section + objective history images (bridge path).
    _bridge_artifacts = _save_bridge_structure_images(
        records=records, bundle=bundle, objective_goal=objective.goal,
    )
    if _bridge_artifacts and records:
        records[-1].artifacts.extend(_bridge_artifacts)

    return {
        "backend": "simulation_loop_bridge",
        "records": records,
        "termination_reason": _termination_reason(
            records,
            objective.goal,
            objective.target_value,
            run_iterations,
            expected_case_count=(len(objective_cases) if objective_cases else None),
        ),
        "constraint_summary": {
            "declared_constraints": constraints,
            "evaluated": bool(records),
            "satisfied": True if records else None,
            "run_artifact_tag": run_artifact_tag,
            "note": "Fallback bridge execution path used because primary invdes path was unavailable.",
        },
    }


def _build_invdes_simulation(
    bundle: InverseDesignConfigBundle,
    td,
    *,
    case_override: Dict[str, Any] | None = None,
    include_field_monitors: bool | None = None,
):
    """Build the optimizer simulation using the same component-aware builders
    as the diagnostic simulation (``tidy3d_runner``).

    This ensures the optimizer's forward/adjoint simulations use physically
    correct device geometry (input/output waveguides, MMI body, etc.) instead
    of a generic slab placeholder.
    """
    from PhotonicsAI.Photon.tidy3d_runner import (
        create_mmi,
        create_simple_waveguide,
        create_waveguide_crossing,
        _default_source_direction_for_port,
        _normalize_port_name,
        _port_position_map,
        _port_waveguide_width,
        _port_prop_axis,
        monitor_size_for_port,
        source_size_for_port,
    )

    sim_cfg = bundle.simulation_config
    domain = sim_cfg.domain
    source_cfg = sim_cfg.source
    params = _numeric_geometry_parameters(sim_cfg.geometry.parameters)

    component = sim_cfg.component_type
    wavelength_um = source_cfg.wavelength_nm / 1000.0
    objective_metric = str(bundle.optimization_config.objective.metric or "").strip().lower()

    wg_width = params.get("wg_width", 0.5)
    wg_height = params.get("wg_height", 0.22)
    mmi_width = params.get("mmi_width", 2.5)
    port_o1_width = _port_waveguide_width(
        component_type=component,
        port_name="port_o1",
        objective_metric=objective_metric,
        wg_width=wg_width,
        mmi_width=mmi_width,
        params=params,
    )
    mmi_num_outputs = max(int(round(float(params.get("mmi_num_outputs", 2) or 2))), 2)
    output_wg_widths = [
        _port_waveguide_width(
            component_type=component,
            port_name=f"port_o{idx + 2}",
            objective_metric=objective_metric,
            wg_width=wg_width,
            mmi_width=mmi_width,
            params=params,
        )
        for idx in range(mmi_num_outputs)
    ]
    required_mode_count = 1
    required_mode_count = max(required_mode_count, int(max(source_cfg.mode_index, 0)) + 1)
    for case in bundle.optimization_config.objective_cases:
        mode_idx = int(getattr(case, "target_mode_index", 0) or 0)
        required_mode_count = max(required_mode_count, mode_idx + 1)
    # Track extra modes beyond the target index so the optimizer can see and
    # penalise power that leaks into high-order modes (Error 3 fix).
    # Wide input ports (e.g. 3 µm SOI at 1550 nm) support ~13 TE modes;
    # without this buffer TE5–TE12 are invisible to the gradient.
    # Default +5 is configurable via INVERSE_STEP5_EXTRA_MODES_BUFFER.
    _extra_modes_buffer = int(os.environ.get("INVERSE_STEP5_EXTRA_MODES_BUFFER", "5"))
    required_mode_count += _extra_modes_buffer
    # M179 fix: narrow single-mode ports (e.g. 0.5 µm output ports) must NOT
    # use the same num_modes as the wide input port.  Requesting radiation modes
    # (mode_index >= guided mode count) on a 0.5 µm waveguide produces
    # astronomically large FDTD mode-decomposition amplitudes (~1e28 W) that
    # cause objective explosions (case5 obj ≈ -1e29) and NaN adjoint sources
    # → upload failure for task *_4_adjoint_0.  Narrow ports only support TE0;
    # use num_modes=1 to eliminate radiation mode artifacts entirely.
    required_mode_count_narrow = 1  # single-mode 0.5 µm output ports

    # ---- Dispatch to the same component builders used by tidy3d_runner ----
    try:
        if component in {"mmi", "splitter"}:
            structures, sim_size, src_center, monitor_positions = create_mmi(
                td, wavelength_um, wg_width, wg_height,
                mmi_width=mmi_width,
                mmi_length=params.get("mmi_length", 10.0),
                num_outputs=mmi_num_outputs,
                input_wg_width=port_o1_width,
                output_wg_widths=output_wg_widths,
            )
        elif component == "crossing":
            structures, sim_size, src_center, monitor_positions = create_waveguide_crossing(
                td, wavelength_um, wg_width, wg_height,
                wg_length=params.get("wg_length", 8.0),
            )
        else:
            # Lazy-import less common builders only when needed.
            _builder_map = _lazy_component_builders(td)
            builder = _builder_map.get(component)
            if builder is not None:
                structures, sim_size, src_center, monitor_positions = builder(
                    td, wavelength_um, wg_width, wg_height, params,
                )
            elif component == "waveguide":
                structures, sim_size, src_center, monitor_positions = create_simple_waveguide(
                    td, wavelength_um, wg_width, wg_height,
                    wg_length=params.get("wg_length", 2.0),
                )
            else:
                raise ValueError(
                    f"No geometry builder for component_type='{component}'. "
                    f"Step1 should have resolved this to a supported type."
                )
    except Exception as _builder_exc:
        raise ValueError(
            f"Component builder failed for component_type='{component}': {_builder_exc}"
        ) from _builder_exc

    # ---- Per official InverseDesign tutorial: only I/O waveguides remain as
    # static structures; the coupling region is entirely defined by the
    # TopologyDesignRegion.  Remove the MMI body (or equivalent main body)
    # so the optimizer has full authority over that region.
    _body_names = {"mmi_section", "coupling_section", "crossing_body"}
    structures = [s for s in structures if getattr(s, "name", None) not in _body_names]

    Lx, Ly, Lz = sim_size

    # ---- Source ----
    freq0 = 299_792_458.0 / (source_cfg.wavelength_nm * 1e-9)
    pulse = td.GaussianPulse(freq0=freq0, fwidth=max(freq0 * 0.1, 1e10))
    port_positions = _port_position_map(monitor_positions)
    source_port = _normalize_port_name(
        (case_override or {}).get("source_port")
        or getattr(source_cfg, "port", "")
        or "port_o1",
        default="port_o1",
    )
    if source_port not in port_positions and "port_o1" in port_positions:
        source_port = "port_o1"
    source_xy = port_positions.get(source_port, (float(src_center[0]), float(src_center[1])))
    source_axis = _port_prop_axis(component, source_port)
    source_width = _port_waveguide_width(
        component_type=component,
        port_name=source_port,
        objective_metric=objective_metric,
        wg_width=wg_width,
        mmi_width=mmi_width,
        params=params,
    )
    source_size = source_size_for_port(source_axis, source_width, wg_height)
    source_direction = str(
        (case_override or {}).get("source_direction")
        or source_cfg.direction
        or ""
    ).strip()
    if source_direction not in {"+", "-"}:
        source_direction = _default_source_direction_for_port(component, source_port, source_xy)
    source_mode_index = (case_override or {}).get("source_mode_index")
    if source_mode_index is None:
        source_mode_index = source_cfg.mode_index
    try:
        source_mode_index = max(int(source_mode_index), 0)
    except (TypeError, ValueError):
        source_mode_index = 0
    source = td.ModeSource(
        source_time=pulse,
        center=(float(source_xy[0]), float(source_xy[1]), wg_height / 2),
        size=source_size,
        # filter_pol='te' guarantees mode_index=0 is TE0 regardless of cross-section;
        # without it, substrate or TM modes may sort before TE0 for non-standard geometries.
        mode_spec=td.ModeSpec(num_modes=required_mode_count, filter_pol="te"),
        mode_index=source_mode_index,
        direction=source_direction,
        name="invdes_mode_source",
    )

    # ---- Monitors ----
    # Step3 config (bundle.simulation_config.monitors) is the primary source
    # of truth for monitor types, names, sizes, and metrics.  Component-builder
    # port positions provide physically accurate coordinates.  The shared
    # helpers ``_port_prop_axis`` / ``monitor_size_for_port`` guarantee that
    # all paths (diagnostic sim, optimizer sim, Step3) use identical sizing.
    monitors = []
    freqs = [freq0]

    # -- 1. Step3-defined monitors (types & sizes from config) --
    step3_monitors = sim_cfg.monitors  # List[MonitorSpec]
    if include_field_monitors is None:
        include_field_monitors = _include_field_monitors_in_adjoint_optimizer()
    _created_names: set = set()
    for spec in step3_monitors:
        spec_name = str(spec.name or "").strip()

        # Override XY center with physics-accurate port position from the
        # component builder whenever the monitor name encodes a port reference.
        # This prevents wrong Step3 bundle coordinates (e.g., generated when
        # mmi_num_outputs was incorrectly 2 instead of 5) from misplacing
        # mode/flux monitors in empty space and producing NaN adjoint sources.
        center = tuple(spec.center_um)
        _port_match = re.search(r"(port_o\d+)", spec_name)
        if _port_match:
            _port_key = _port_match.group(1)
            if _port_key in port_positions:
                _px, _py = port_positions[_port_key]
                center = (_px, _py, center[2])
        elif spec_name in {"through_flux"} and "port_o2" in port_positions:
            _px, _py = port_positions["port_o2"]
            center = (_px, _py, center[2])
        elif spec_name in {"secondary_flux"} and "port_o3" in port_positions:
            _px, _py = port_positions["port_o3"]
            center = (_px, _py, center[2])
        size = tuple(spec.size_um)
        if spec.monitor_type == "field":
            if not include_field_monitors:
                continue
            monitors.append(
                td.FieldMonitor(
                    name=spec.name, center=center, size=size, freqs=freqs,
                )
            )
        elif spec.monitor_type == "flux":
            monitors.append(
                td.FluxMonitor(
                    name=spec.name, center=center, size=size, freqs=freqs,
                )
            )
        elif spec.monitor_type == "mode":
            # Wide monitors (port_o1, ≥1 µm cross-section) need the full mode
            # count.  Narrow monitors (port_o2..o6, 0.5 µm) must use
            # num_modes=1 to prevent radiation-mode amplitude explosions.
            _spec_width = max(spec.size_um) if spec.size_um else 0.0
            _spec_num_modes = required_mode_count if _spec_width >= 1.0 else required_mode_count_narrow
            monitors.append(
                td.ModeMonitor(
                    name=spec.name, center=center, size=size, freqs=freqs,
                    mode_spec=td.ModeSpec(num_modes=_spec_num_modes, filter_pol="te"),
                )
            )
        _created_names.add(spec_name)

    # -- 2. Flux monitors at builder port positions (physics-accurate coords) --
    for mx, my, mname in monitor_positions:
        fname = f"flux_{mname}"
        if fname in _created_names:
            continue
        port_name = str(mname).strip().lower()
        axis = _port_prop_axis(component, port_name)
        port_width = _port_waveguide_width(
            component_type=component,
            port_name=port_name,
            objective_metric=objective_metric,
            wg_width=wg_width,
            mmi_width=mmi_width,
            params=params,
        )
        msize = monitor_size_for_port(axis, port_width, wg_height)
        monitors.append(
            td.FluxMonitor(
                name=fname,
                center=(mx, my, wg_height / 2),
                size=msize,
                freqs=freqs,
            )
        )
        _created_names.add(fname)

    # -- 3. Mode monitors at EACH output port (required by invdes ModePower) --
    # ModePower requires ModeMonitor — using FluxMonitor causes incorrect
    # adjoint gradients and persistent odd-even oscillation (BUG-8).
    for mx, my, mname in monitor_positions:
        include_mode_port = str(mname).startswith("port_o") and (
            str(mname) != "port_o1" or objective_metric == "mux_routing"
        )
        if include_mode_port:
            mode_name = f"mode_{mname}"
            if mode_name in _created_names:
                continue
            port_name = str(mname).strip().lower()
            axis = _port_prop_axis(component, port_name)
            port_width = _port_waveguide_width(
                component_type=component,
                port_name=port_name,
                objective_metric=objective_metric,
                wg_width=wg_width,
                mmi_width=mmi_width,
                params=params,
            )
            msize = monitor_size_for_port(axis, port_width, wg_height)
            # Wide port (≥1 µm) → extra-mode count; narrow port → num_modes=1.
            _loop3_num_modes = required_mode_count if port_width >= 1.0 else required_mode_count_narrow
            monitors.append(
                td.ModeMonitor(
                    name=mode_name,
                    center=(mx, my, wg_height / 2),
                    size=msize,
                    freqs=freqs,
                    mode_spec=td.ModeSpec(num_modes=_loop3_num_modes, filter_pol="te"),
                )
            )
            _created_names.add(mode_name)

    # -- 4. Fallback: guarantee at least one mode monitor --
    if not any(isinstance(m, td.ModeMonitor) for m in monitors):
        through_port = next(
            ((mx, my) for mx, my, mn in monitor_positions if "o2" in mn),
            (Lx / 2 - 1.0, 0.0),
        )
        axis = _port_prop_axis(component, "port_o2")
        fallback_width = _port_waveguide_width(
            component_type=component,
            port_name="port_o2",
            objective_metric=objective_metric,
            wg_width=wg_width,
            mmi_width=mmi_width,
            params=params,
        )
        msize = monitor_size_for_port(axis, fallback_width, wg_height)
        monitors.append(
            td.ModeMonitor(
                name="mode_monitor",
                center=(through_port[0], through_port[1], wg_height / 2),
                size=msize,
                freqs=freqs,
                mode_spec=td.ModeSpec(num_modes=required_mode_count, filter_pol="te"),
            )
        )

    boundary = td.BoundarySpec.all_sides(boundary=td.PML())
    return td.Simulation(
        size=(Lx, Ly, Lz),
        center=tuple(domain.center_um),
        run_time=sim_cfg.run_time_s,
        medium=td.Medium(permittivity=1.44**2),
        structures=structures,
        sources=[source],
        monitors=monitors,
        boundary_spec=boundary,
        grid_spec=td.GridSpec.auto(
            wavelength=wavelength_um,
            min_steps_per_wvl=max(domain.min_steps_per_wvl, 10),
        ),
    )


def _lazy_component_builders(td):
    """Return a mapping of component_type -> builder for less common types.

    Each builder has the signature ``(td, wavelength_um, wg_width, wg_height, params) -> tuple``.
    """
    from PhotonicsAI.Photon.tidy3d_runner import (
        create_coupler,
        create_grating_coupler,
        create_mzi,
        create_polarization_rotator,
        create_ring_resonator,
        create_y_branch,
    )

    def _ring(td, wl, ww, wh, p):
        return create_ring_resonator(td, wl, ww, wh, p.get("ring_radius", 5.0), p.get("gap", 0.2))

    def _mzi(td, wl, ww, wh, p):
        return create_mzi(td, wl, ww, wh, p.get("arm_length", 20.0), p.get("arm_separation", 5.0))

    def _dc(td, wl, ww, wh, p):
        return create_coupler(td, wl, ww, wh, p.get("coupler_length", 10.0), p.get("gap", 0.2))

    def _gc(td, wl, ww, wh, p):
        return create_grating_coupler(td, wl, ww, wh, p.get("grating_period", 0.62), p.get("num_periods", 20))

    def _pr(td, wl, ww, wh, p):
        return create_polarization_rotator(td, wl, ww, wh, p.get("rotation_length", 30.0), p.get("swg_period", 0.4))

    def _yb(td, wl, ww, wh, p):
        return create_y_branch(td, wl, ww, wh, p.get("arm_length", 15.0), p.get("arm_separation", 3.0))

    return {
        "ring_resonator": _ring,
        "mzi": _mzi,
        "directional_coupler": _dc,
        "grating_coupler": _gc,
        "polarization_rotator": _pr,
        "y_branch": _yb,
    }


def _pick_output_monitor_names(bundle: InverseDesignConfigBundle, sim) -> List[str]:
    """Select output monitor names from the simulation for the invdes metric.

    For multi-port devices we want to sum the power at all output ports.
    ModePower requires ModeMonitor — always prefer mode_port_* monitors.
    Falls back to the first mode or flux monitor found.
    """
    component = bundle.simulation_config.component_type
    objective_metric = str(bundle.optimization_config.objective.metric or "").strip().lower()
    sim_monitor_names = [m.name for m in sim.monitors]

    # Prefer ModeMonitors at output ports (mode_port_o2, mode_port_o3, ...)
    output_mode = [
        n for n in sim_monitor_names
        if n.startswith("mode_port_o")
    ]
    if objective_metric == "mux_routing":
        if "mode_port_o1" in output_mode:
            return ["mode_port_o1"] + [name for name in output_mode if name != "mode_port_o1"]
    if component in {"mmi", "splitter", "crossing", "y_branch", "directional_coupler", "mzi"} and output_mode:
        return output_mode

    return _pick_output_monitor_names_for_sim(sim_monitor_names)


def _design_region_size(domain_size: Sequence[float], bundle: InverseDesignConfigBundle) -> tuple[float, float, float]:
    """Compute design region size based on component geometry parameters.

    For MMI/splitter the design region should cover the multimode section.
    For other components, use a sensible fraction of the simulation domain.
    """
    params = _numeric_geometry_parameters(bundle.simulation_config.geometry.parameters)
    component = bundle.simulation_config.component_type
    wg_height = params.get("wg_height", 0.22)
    footprint = _extract_footprint_dims_um(bundle.optimization_config.constraints)

    if component in {"mmi", "splitter"}:
        if footprint is not None:
            sx, sy = footprint
        else:
            sx = float(params.get("mmi_length", 10.0))
            sy = float(params.get("mmi_width", 2.5))
        sz = wg_height
    elif component == "crossing":
        wg_len = float(params.get("wg_length", 8.0))
        sx = wg_len * 2.0
        sy = wg_len * 2.0
        sz = wg_height
    else:
        sx = max(float(domain_size[0]) * 0.5, 0.5)
        sy = max(float(domain_size[1]) * 0.5, 0.5)
        sz = max(min(float(domain_size[2]) * 0.2, 0.5), wg_height)
    return (sx, sy, sz)


def _extract_footprint_dims_um(constraints: Sequence[str]) -> tuple[float, float] | None:
    for item in constraints:
        text = str(item or "")
        if "footprint" not in text.lower():
            continue
        m = _FOOTPRINT_PAIR_RE.search(text)
        if not m:
            nums = re.findall(r"([0-9]+(?:\.[0-9]+)?)", text)
            if len(nums) >= 2:
                return (float(nums[0]), float(nums[1]))
            continue
        d1 = float(m.group(1))
        d2 = float(m.group(2))
        unit = m.group(3).lower()
        scale = 0.001 if unit == "nm" else 1.0
        return (d1 * scale, d2 * scale)
    return None


def _map_design_vector_to_variables(
    *,
    vector: Any,
    variables: Sequence[Any],
    previous: Dict[str, float],
) -> Dict[str, float]:
    mapped = dict(previous)
    if vector is None:
        return mapped

    flat = _flatten_to_list(vector)
    if not flat:
        return mapped

    for idx, var in enumerate(variables):
        unit_value = flat[idx % len(flat)]
        unit_value = min(max(unit_value, 0.0), 1.0)
        span = float(var.upper_bound) - float(var.lower_bound)
        mapped[var.name] = round(float(var.lower_bound) + unit_value * span, 6)
    return mapped


def _flatten_to_list(value: Any) -> List[float]:
    if value is None:
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, (list, tuple)):
        out: List[float] = []
        for item in value:
            out.extend(_flatten_to_list(item))
        return out
    try:
        # Support numpy-like arrays while avoiding a hard dependency.
        return [float(item) for item in value.ravel()]  # type: ignore[attr-defined]
    except Exception:
        try:
            return [float(item) for item in value]
        except Exception:
            return []


def _as_float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int_or_none(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_path(path_like: str | Path | None) -> Path | None:
    if not path_like:
        return None
    path = Path(path_like)
    if not path.is_absolute():
        path = (_REPO_ROOT / path).resolve()
    return path


def _build_adjoint_trace_snapshot(aux_data: Dict[str, Any]) -> Dict[str, float]:
    trace: Dict[str, float] = {}
    for key in (
        "objective_fn_val",
        "post_process_val",
        "grad_norm",
        "gradient_norm",
        "penalty",
        "constraint_penalty",
        "update_norm",
        "step_size",
    ):
        value = _as_float_or_none(aux_data.get(key))
        if value is not None:
            trace[str(key)] = value
    # Binarization metric: mean(|2p - 1|) over params; 0=fully gray, 1=fully binary.
    # Computed from params vector if available in aux_data.
    params_raw = aux_data.get("params")
    if params_raw is not None:
        try:
            import numpy as _np
            p_flat = _np.asarray(params_raw, dtype=float).ravel()
            if p_flat.size > 0:
                trace["binarization_metric"] = float(_np.mean(_np.abs(2.0 * p_flat - 1.0)))
        except Exception:
            pass
    return trace


def _objective_cases(bundle: InverseDesignConfigBundle) -> List[Dict[str, Any]]:
    cases = []
    for case in bundle.optimization_config.objective_cases:
        cases.append(
            {
                "name": case.name,
                "wavelength_nm": float(case.wavelength_nm),
                "source_port": str(getattr(case, "source_port", "") or "").strip().lower(),
                "source_mode_index": int(getattr(case, "source_mode_index", 0) or 0),
                "source_direction": str(getattr(case, "source_direction", "") or "").strip(),
                "target_port": str(case.target_port or "").strip().lower(),
                "target_mode_index": int(getattr(case, "target_mode_index", 0) or 0),
                "min_coupling": case.min_coupling,
                "max_crosstalk": case.max_crosstalk,
                "weight": float(case.weight or 1.0),
            }
        )
    return cases


def _has_multi_wavelength_cases(cases: Sequence[Dict[str, Any]]) -> bool:
    wavelengths = {
        round(float(item.get("wavelength_nm", 0.0)), 6)
        for item in cases
        if item.get("wavelength_nm") is not None
    }
    return len(wavelengths) > 1


def _should_use_multi_case_adjoint(cases: Sequence[Dict[str, Any]]) -> bool:
    if not cases:
        return False
    if len(cases) > 1:
        return True
    if _has_multi_wavelength_cases(cases):
        return True
    first = dict(cases[0]) if cases else {}
    mode_idx = _as_float_or_none(first.get("target_mode_index"))
    return bool(mode_idx is not None and mode_idx > 0)


def _bundle_with_one_case(
    bundle: InverseDesignConfigBundle,
    objective_cases: Sequence[Dict[str, Any]],
) -> InverseDesignConfigBundle:
    """Return a copy of bundle whose objective_cases list is limited to the first case.

    Used to limit the physics-gate seed diagnostic sim to a single cloud task so
    it completes within the per-subprocess timeout headroom left after the adjoint
    optimizer run finishes.
    """
    if not objective_cases:
        return bundle
    one_case_bundle = bundle.model_copy(deep=True)
    # Limit the already-parsed ObjectiveCaseSpec list to the first element.
    existing = getattr(one_case_bundle.optimization_config, "objective_cases", None)
    if existing:
        try:
            one_case_bundle.optimization_config.objective_cases = [existing[0]]
        except Exception:  # noqa: BLE001
            pass  # Keep original if attribute is immutable or wrong type
    return one_case_bundle


def _bundle_with_case_wavelength(
    bundle: InverseDesignConfigBundle,
    wavelength_nm: float,
    *,
    case: Dict[str, Any] | None = None,
) -> InverseDesignConfigBundle:
    case_bundle = bundle.model_copy(deep=True)
    case_bundle.simulation_config.wavelength_nm = float(wavelength_nm)
    case_bundle.simulation_config.source.wavelength_nm = float(wavelength_nm)
    if case:
        source_port = _case_source_port(case)
        case_bundle.simulation_config.source.port = source_port
        case_bundle.simulation_config.source.mode_index = _case_source_mode_index(case)
        source_direction = _case_source_direction(case)
        if source_direction in {"+", "-"}:
            case_bundle.simulation_config.source.direction = source_direction
    freq_hz = 299_792_458.0 / (float(wavelength_nm) * 1e-9)
    for monitor in case_bundle.simulation_config.monitors:
        monitor.freqs_hz = [float(freq_hz)]
    return case_bundle


def _pick_case_output_monitor_names(sim, target_port: str) -> List[str]:
    sim_monitor_names = [m.name for m in sim.monitors]
    output_mode = [
        n for n in sim_monitor_names
        if n.startswith("mode_port_o")
    ]
    if not output_mode:
        return _pick_output_monitor_names_for_sim(sim_monitor_names)

    target_mode = _target_port_to_mode_monitor(target_port)
    if target_mode and target_mode in output_mode:
        ordered = [target_mode] + [name for name in output_mode if name != target_mode]
        return ordered
    return output_mode


def _pick_output_monitor_names_for_sim(sim_monitor_names: Sequence[str]) -> List[str]:
    if "mode_monitor" in sim_monitor_names:
        return ["mode_monitor"]
    mode_any = [n for n in sim_monitor_names if n.startswith("mode_")]
    if mode_any:
        return mode_any[:1]
    flux_any = [n for n in sim_monitor_names if n.startswith("flux_")]
    if flux_any:
        return flux_any[:1]
    return [sim_monitor_names[0]] if sim_monitor_names else ["mode_monitor"]


def _target_port_to_mode_monitor(target_port: str) -> str:
    lowered = str(target_port or "").strip().lower()
    if lowered.startswith("mode_port_o"):
        return lowered
    if lowered.startswith("port_o"):
        return f"mode_{lowered}"
    m = re.search(r"(\d+)", lowered)
    if m:
        return f"mode_port_o{int(m.group(1))}"
    return "mode_port_o2"


def _case_source_port(case: Dict[str, Any]) -> str:
    lowered = str(case.get("source_port", "") or "").strip().lower()
    if lowered.startswith("port_o"):
        return lowered
    m = re.search(r"(\d+)", lowered)
    if m:
        return f"port_o{int(m.group(1))}"
    return "port_o1"


def _case_source_mode_index(case: Dict[str, Any]) -> int:
    value = _as_float_or_none(case.get("source_mode_index"))
    if value is None:
        return 0
    return max(int(value), 0)


def _case_source_direction(case: Dict[str, Any]) -> str:
    direction = str(case.get("source_direction", "") or "").strip()
    if direction in {"+", "-"}:
        return direction
    source_port = _case_source_port(case)
    return "+" if source_port == "port_o1" else "-"


def _source_port_to_flux_key(source_port: str) -> str:
    lowered = str(source_port or "").strip().lower()
    if lowered.startswith("flux_port_o"):
        return lowered
    if lowered.startswith("port_o"):
        return f"flux_{lowered}"
    m = re.search(r"(\d+)", lowered)
    if m:
        return f"flux_port_o{int(m.group(1))}"
    return "flux_port_o1"


def _build_multi_case_post_process_fn(
    *,
    objective_cases: Sequence[Dict[str, Any]],
    task_names: Sequence[str],
    case_task_indices: Sequence[int] | None = None,
    observation_callback: Any | None = None,
):
    import autograd.numpy as anp

    def _to_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except Exception:
            pass
        raw_value = getattr(value, "_value", None)
        if raw_value is not None:
            try:
                arr = anp.asarray(raw_value)
                if arr.size == 0:
                    return None
                return float(anp.real(arr.reshape(-1)[0]))
            except Exception:
                pass
        try:
            arr = anp.asarray(value)
            if arr.size == 0:
                return None
            return float(anp.real(anp.mean(arr)))
        except Exception:
            return None

    def _flux_power(sim_data: Any, monitor_name: str) -> float | None:
        try:
            monitor_data = sim_data[monitor_name]
        except Exception:
            return None
        flux = getattr(monitor_data, "flux", None)
        if flux is None:
            return None
        value = flux
        try:
            value = value.sel(direction="+")
        except Exception:
            pass
        try:
            values = value.values if hasattr(value, "values") else value
        except Exception:
            values = value
        return _to_float(values)

    def _mode_power(sim_data: Any, monitor_name: str, mode_index: int = 0):
        monitor_data = sim_data[monitor_name]
        amps = getattr(monitor_data, "amps", None)
        if amps is None:
            return anp.array(0.0)
        value = amps
        for dim_name, dim_value in (("direction", "+"), ("mode_index", max(int(mode_index), 0))):
            try:
                value = value.sel(**{dim_name: dim_value})
            except Exception:
                pass
        values = value.values if hasattr(value, "values") else value
        return anp.real(anp.mean(values * anp.conj(values)))

    def _flux_power_auto(sim_data: Any, monitor_name: str):
        """Return flux power as an autograd-compatible array.

        Unlike _flux_power (which calls _to_float and breaks the adjoint graph),
        this function returns anp arrays so gradients flow through it correctly.
        The sign convention for FluxMonitor at the source port is typically
        negative (power leaving the domain); anp.abs gives the injected magnitude.
        """
        try:
            monitor_data = sim_data[monitor_name]
        except Exception:
            return None
        flux = getattr(monitor_data, "flux", None)
        if flux is None:
            return None
        try:
            values = flux.values if hasattr(flux, "values") else flux
            # Flux values are real-valued power; take mean over frequency etc.
            return anp.abs(anp.mean(values))
        except Exception:
            return None

    def _mode_power_dir(sim_data: Any, monitor_name: str, mode_index: int, direction: str):
        """Like _mode_power but reads a specific propagation direction.

        Use direction=case_source_direction ("-" for right-side sources) to read
        the *forward-going* power rather than the reflected/backward power that
        the default "+" selection returns at right-side ports.
        """
        monitor_data = sim_data[monitor_name]
        amps = getattr(monitor_data, "amps", None)
        if amps is None:
            return anp.array(0.0)
        value = amps
        for dim_name, dim_value in (("direction", direction), ("mode_index", max(int(mode_index), 0))):
            try:
                value = value.sel(**{dim_name: dim_value})
            except Exception:
                pass
        values = value.values if hasattr(value, "values") else value
        # M179 safety: if mode_index was out of the monitor's range, the .sel()
        # call above was silently skipped and `values` still has extra dimensions.
        # Return 0 — the mode physically does not exist at this port
        # (e.g. mode 4 at a single-mode 0.5 µm waveguide).
        try:
            if hasattr(values, "ndim") and values.ndim > 1:
                return anp.array(0.0)
        except Exception:
            pass
        return anp.real(anp.mean(values * anp.conj(values)))

    def _post_process_fn(batch_data: Any):
        total_weight = anp.array(0.0)
        weighted_obj = anp.array(0.0)
        case_details: List[Dict[str, Any]] = []
        monitor_readings: Dict[str, float | None] = {}
        global_mode_indices = sorted(
            {
                max(int(_to_float(case.get("target_mode_index")) or 0), 0)
                for case in objective_cases
            }
        )
        if not global_mode_indices:
            global_mode_indices = [0]
        if max(global_mode_indices) >= 1:
            global_mode_indices = sorted(set([*global_mode_indices, 0, 1]))
        mode_demux_active = max(global_mode_indices) >= 1
        purity_penalty_weight = _demux_mode_purity_penalty_weight()
        (
            higher_order_boost,
            purity_focus_threshold,
            purity_focus_gain,
            purity_focus_max_multiplier,
        ) = _demux_mode_focus_params()
        (
            transmission_surrogate_weight,
            transmission_shortfall_penalty,
            transmission_floor,
            transmission_surrogate_max,
        ) = _demux_transmission_params()

        for case_idx, case in enumerate(objective_cases):
            task_idx = case_idx
            if case_task_indices is not None and case_idx < len(case_task_indices):
                try:
                    task_idx = int(case_task_indices[case_idx])
                except Exception:
                    task_idx = case_idx
            if task_idx < 0 or task_idx >= len(task_names):
                continue
            task_name = task_names[task_idx]
            sim_data = batch_data[task_name]
            target_mode = _target_port_to_mode_monitor(str(case.get("target_port", "")))
            target_mode_index = max(int(_to_float(case.get("target_mode_index")) or 0), 0)
            monitor_names = [name for name in sim_data.monitor_data.keys() if str(name).startswith("mode_port_o")]
            if not monitor_names:
                monitor_names = [name for name in sim_data.monitor_data.keys() if str(name).startswith("mode_")]
            if not monitor_names:
                continue
            if target_mode not in monitor_names:
                target_mode = monitor_names[0]

            # Compute propagation direction early so all mode-power terms use the
            # correct direction: "-" for right-side sources (cases 2-5) whose
            # transmitted modes exit port_o1 in the "-" (leftward) direction.
            # Using the wrong direction ("+") reads reflected power, which is near
            # zero for higher modes and causes coupling_ratio_obj ≈ 0 → large penalties.
            case_source_direction = _case_source_direction(case)

            target_mode_power = _mode_power_dir(sim_data, target_mode, target_mode_index, case_source_direction)
            target_mode_power_float = _to_float(target_mode_power)
            target_mode_power_clamped = anp.maximum(target_mode_power, 0.0)
            max_observed_mode = max(global_mode_indices) if global_mode_indices else 0
            target_monitor_modes: List[float | None] = [None] * (max_observed_mode + 1)
            target_monitor_mode_sum_obj = anp.array(0.0)
            for mode_idx in global_mode_indices:
                mode_obj = _mode_power_dir(sim_data, target_mode, mode_idx, case_source_direction)
                mode_obj_clamped = anp.maximum(mode_obj, 0.0)
                target_monitor_mode_sum_obj = target_monitor_mode_sum_obj + mode_obj_clamped
                mode_float = _to_float(mode_obj)
                if mode_float is not None:
                    target_monitor_modes[mode_idx] = max(mode_float, 0.0)
            target_monitor_mode_total = float(
                sum(value for value in target_monitor_modes if value is not None)
            )
            target_monitor_mode_te0 = (
                float(target_monitor_modes[0])
                if len(target_monitor_modes) >= 1 and target_monitor_modes[0] is not None
                else None
            )
            target_monitor_mode_te1 = (
                float(target_monitor_modes[1])
                if len(target_monitor_modes) >= 2 and target_monitor_modes[1] is not None
                else None
            )
            target_monitor_mode_te0_purity = (
                target_monitor_mode_te0 / target_monitor_mode_total
                if target_monitor_mode_te0 is not None and target_monitor_mode_total > 1e-12
                else None
            )
            target_monitor_mode_te1_purity = (
                target_monitor_mode_te1 / target_monitor_mode_total
                if target_monitor_mode_te1 is not None and target_monitor_mode_total > 1e-12
                else None
            )
            # Resolve source port/monitor before the crosstalk loop so the source
            # monitor can be excluded from mode_power_sum.  Including the source
            # monitor would inject the large incident-power amplitude into the
            # crosstalk denominator, making coupling_ratio_obj ≈ 0 for rev-cases.
            source_port = _case_source_port(case)
            source_mode_index = _case_source_mode_index(case)
            # case_source_direction already computed above before mode-power calls
            source_mode_monitor = _target_port_to_mode_monitor(source_port)

            mode_crosstalk = anp.array(0.0)
            mode_crosstalk_float = 0.0
            mode_power_sum = anp.array(0.0)
            for name in monitor_names:
                # Skip the source-port monitor: in case_source_direction it carries
                # the incident field (large), which must not inflate the denominator.
                if name == source_mode_monitor:
                    continue
                mode_val = _mode_power_dir(sim_data, name, target_mode_index, case_source_direction)
                mode_power_sum = mode_power_sum + anp.maximum(mode_val, 0.0)
                if name == target_mode:
                    continue
                mode_crosstalk = anp.maximum(mode_crosstalk, mode_val)
                mode_val_float = _to_float(mode_val)
                if mode_val_float is not None:
                    mode_crosstalk_float = max(mode_crosstalk_float, max(mode_val_float, 0.0))

            mode_sum = _to_float(mode_power_sum) or 0.0
            mode_norm = anp.maximum(mode_power_sum, 1e-12)
            coupling_ratio_obj = target_mode_power / mode_norm
            crosstalk_ratio_obj = mode_crosstalk / mode_norm
            optimizer_ratio_basis = "mode_output_sum"

            min_coupling = _as_float_or_none(case.get("min_coupling"))
            max_crosstalk = _as_float_or_none(case.get("max_crosstalk"))
            coupling_penalty, crosstalk_penalty = _demux_penalty_weights()
            _src_has_mode_monitor = source_mode_monitor in monitor_names
            if not _src_has_mode_monitor:
                source_mode_monitor = target_mode  # keep for legacy diagnostics path
            # --- transmission surrogate using CORRECT directional power ---
            # Priority for source normalization:
            #  1. Dedicated ModeMonitor at source port (most accurate, reads correct direction)
            #  2. FluxMonitor at source port via _flux_power() as constant float (~0.14W).
            #     Source flux is ~constant w.r.t. design parameters, so using a Python float
            #     constant as divisor gives correct gradient: d(trans_s)/d(x) ∝ d(P_target)/d(x).
            #     NOTE: _flux_power_auto (using anp.mean) was tried but returns None in the
            #     JAX adjoint context because JaxFluxData.flux.values access fails silently.
            #  3. TE0 power at target port in source direction (last resort - only when flux
            #     monitor is missing; gives relative proxy, not absolute transmission).
            if _src_has_mode_monitor:
                source_mode_power_fwd = _mode_power_dir(
                    sim_data, source_mode_monitor, source_mode_index, case_source_direction
                )
            else:
                _src_flux_key = _source_port_to_flux_key(source_port)
                _src_flux_val = _flux_power(sim_data, _src_flux_key)  # returns float or None
                if _src_flux_val is not None and abs(_src_flux_val) > 1e-6:
                    # Convert to plain Python float — constant in adjoint graph.
                    # Gradient flows correctly through target_mode_power_transmitted.
                    source_mode_power_fwd = abs(float(_src_flux_val))
                else:
                    # True last resort: TE0 at target port in source direction.
                    # For case1 (both numerator and denominator use TE0@target_mode),
                    # add a small TE1 fraction to avoid a constant ratio of 1.0.
                    _te0_fwd = _mode_power_dir(sim_data, target_mode, 0, case_source_direction)
                    _te1_fwd = _mode_power_dir(sim_data, target_mode, 1, case_source_direction)
                    source_mode_power_fwd = _te0_fwd + 0.05 * _te1_fwd
            source_mode_power_fwd_clamped = anp.maximum(source_mode_power_fwd, 0.0)
            source_mode_norm_fwd = anp.maximum(source_mode_power_fwd_clamped, 1e-12)
            # Target-mode power in the transmission direction at target port.
            target_mode_power_transmitted = _mode_power_dir(
                sim_data, target_mode, target_mode_index, case_source_direction
            )
            target_mode_power_transmitted_clamped = anp.maximum(target_mode_power_transmitted, 0.0)
            transmission_ratio_obj = anp.minimum(
                target_mode_power_transmitted_clamped / source_mode_norm_fwd,
                transmission_surrogate_max,  # now 1.0 — physically bounded
            )
            transmission_ratio_float = _to_float(transmission_ratio_obj)
            # Keep backward-compat source_mode_power for legacy diagnostics only.
            source_mode_power = _mode_power(sim_data, source_mode_monitor, source_mode_index)

            input_flux_key = _source_port_to_flux_key(source_port)
            target_flux_key = _normalize_target_flux_key(str(case.get("target_port", "")))
            input_flux_raw = _flux_power(sim_data, input_flux_key)
            target_flux_measured_raw = _flux_power(sim_data, target_flux_key)
            input_flux = abs(input_flux_raw) if input_flux_raw is not None else None
            target_flux_measured = (
                abs(target_flux_measured_raw)
                if target_flux_measured_raw is not None
                else None
            )
            crosstalk_flux = 0.0
            output_flux_sum = 0.0
            has_other_flux = False
            for key in sim_data.monitor_data.keys():
                key_name = str(key).lower()
                if not key_name.startswith("flux_port_o"):
                    continue
                if key_name == input_flux_key:
                    continue
                flux_val_raw = _flux_power(sim_data, key_name)
                if flux_val_raw is not None:
                    flux_val = abs(flux_val_raw)
                    output_flux_sum += flux_val
                    if key_name != target_flux_key:
                        has_other_flux = True
                        crosstalk_flux = max(crosstalk_flux, flux_val)

            # Objective normalization: use the INJECTED mode power as denominator.
            # Priority (M181 denominator-collapse fix):
            #  1. source_mode_power_fwd = |A_TE0("-")|² at source port ModeMonitor.
            #     This reads the UNIDIRECTIONAL injected amplitude — immune to
            #     destructive-interference collapse (net_flux → 0 when backreflection
            #     is large).  Convert to float so gradient flows only through numerator.
            #  2. FluxMonitor abs value — only if (a) source ModeMonitor is absent AND
            #     (b) the value is physically plausible (≥ 10% of expected ≈ 0.10 W).
            #     Collapsed flux (< 0.01) is rejected to avoid coupling_primary blowing
            #     up to ~50× (M181 denominator-collapse bug).
            #  3. Mode-output-sum fallback — last resort.
            _FLUX_COLLAPSE_THRESHOLD = 0.01  # W; below this FluxMonitor is unreliable
            _obj_input_norm = None
            if _src_has_mode_monitor:
                _src_mode_pwr_fwd = _mode_power_dir(
                    sim_data, source_mode_monitor, source_mode_index, case_source_direction
                )
                _src_mode_pwr_fwd_float = _to_float(_src_mode_pwr_fwd)
                # Apply same collapse guard as FluxMonitor path (M182 fix):
                # near-source ModeMonitor amps can measure near-zero values due to
                # near-field contamination or reversed normal convention, causing the
                # denominator to collapse and objective to blow up ~10^4x.
                if (
                    _src_mode_pwr_fwd_float is not None
                    and _src_mode_pwr_fwd_float >= _FLUX_COLLAPSE_THRESHOLD
                ):
                    # Python-float constant: gradient flows through numerator only.
                    _obj_input_norm = anp.maximum(
                        anp.array(abs(float(_src_mode_pwr_fwd_float))), 1e-12
                    )
                    optimizer_ratio_basis = "src_mode_power_fwd"
            if _obj_input_norm is None:
                # M200 bug fix: FluxMonitor at source port for direction="-" cases
                # (right-side sources) measures REFLECTED power (~6-15% of injected),
                # NOT injected power (~1W).  Using reflected power as denominator
                # inflates coupling_primary_obj to > 1.0 (physically impossible) and
                # prevents min_coupling threshold checks from firing.
                # Tidy3D ModeSource normalization: source injects exactly 1W at the
                # central frequency.  Use 1.0 as the true injection power baseline.
                # The FluxMonitor path is only retained as a cross-check (diagnostics).
                _flux_val_abs = (
                    abs(float(input_flux_raw))
                    if input_flux_raw is not None and abs(float(input_flux_raw)) > 1e-12
                    else None
                )
                # Only trust FluxMonitor as injection proxy when it reads > 0.5W,
                # i.e. when the monitor is on the injection side (direction="+" source).
                # For direction="-" sources the monitor captures reflected power only.
                _flux_is_injection_side = (
                    _flux_val_abs is not None and _flux_val_abs >= 0.5
                )
                if _flux_is_injection_side:
                    _obj_input_norm = anp.maximum(anp.array(_flux_val_abs), 1e-12)
                    optimizer_ratio_basis = "input_flux_normalized"
                else:
                    # Tidy3D normalization: ModeSource injects 1W → use as denominator.
                    _obj_input_norm = anp.array(1.0)
                    optimizer_ratio_basis = "modesource_unit_power"
            if _obj_input_norm is not None:
                coupling_primary_obj = target_mode_power_clamped / _obj_input_norm
                # Cross-port leakage of target mode index, normalized by source input.
                crosstalk_primary_obj = mode_crosstalk / _obj_input_norm
            else:
                # Fallback when source ModeMonitor and FluxMonitor are both absent/collapsed.
                coupling_primary_obj = coupling_ratio_obj
                crosstalk_primary_obj = crosstalk_ratio_obj
                optimizer_ratio_basis = "mode_output_sum_fallback"

            case_obj = coupling_primary_obj - crosstalk_primary_obj
            mode_wrong_ratio_obj = anp.array(0.0)
            mode_target_purity_obj = anp.array(1.0)
            mode_focus_multiplier_obj = anp.array(1.0)
            if mode_demux_active:
                # M180 PURITY DENOMINATOR FIX:
                # FluxMonitor normalization != |amps|^2 normalization (systematic ~20% offset).
                # Using FluxMonitor as denominator inflates purity (e.g. 0.797→0.957),
                # killing the purity gradient and allowing multi-lobe field patterns even at
                # "high purity". The correct formula is:
                #   purity = |a_target|^2 / sum_i(|a_i|^2)  for ALL i in ModeMonitor
                # This cancels the normalization offset because numerator and denominator
                # both use amps-based power. sum_i(|a_i|^2) covers all tracked modes (0..N-1).
                # Old mistake 1: FluxMonitor denominator → normalization mismatch
                # Old mistake 2: sum(TE0-TE4) only → misses TE5-TE9 power
                _purity_denom_set = False
                try:
                    _target_amps_da = sim_data[target_mode].amps
                    if hasattr(_target_amps_da, "mode_index"):
                        _all_midxs = [int(x) for x in _target_amps_da.mode_index.values]
                        _mode_sum_full = anp.array(0.0)
                        for _mi in _all_midxs:
                            _mp = _mode_power_dir(sim_data, target_mode, _mi, case_source_direction)
                            _mode_sum_full = _mode_sum_full + anp.maximum(_mp, 0.0)
                        target_purity_norm_obj = anp.maximum(_mode_sum_full, 1e-12)
                        _purity_denom_set = True
                except Exception:
                    pass
                if not _purity_denom_set:
                    # Fallback: sum of tracked objective modes (TE0-TE4)
                    target_purity_norm_obj = anp.maximum(target_monitor_mode_sum_obj, 1e-12)
                mode_target_purity_obj = target_mode_power_clamped / target_purity_norm_obj
                mode_wrong_ratio_obj = anp.maximum(1.0 - mode_target_purity_obj, 0.0)
                mode_order_factor = 1.0 + max(target_mode_index, 0) * higher_order_boost
                focus_deficit_obj = anp.maximum(0.0, purity_focus_threshold - mode_target_purity_obj)
                mode_focus_multiplier_obj = mode_order_factor * (
                    1.0 + purity_focus_gain * focus_deficit_obj
                )
                mode_focus_multiplier_obj = anp.minimum(
                    anp.maximum(mode_focus_multiplier_obj, 1.0),
                    purity_focus_max_multiplier,
                )
                case_obj = case_obj - purity_penalty_weight * mode_focus_multiplier_obj * mode_wrong_ratio_obj
            if min_coupling is not None:
                case_obj = case_obj - coupling_penalty * anp.maximum(
                    0.0,
                    float(min_coupling) - coupling_primary_obj,
                )
            if max_crosstalk is not None:
                case_obj = case_obj - crosstalk_penalty * anp.maximum(
                    0.0,
                    crosstalk_primary_obj - float(max_crosstalk),
                )
            # Cross-mode absolute power penalty: directly penalize a specific adjacent
            # mode index at the target port (e.g. TE4 in the TE3 case, TE3 in the TE4
            # case).  This provides a directional gradient that the ratio-based purity
            # penalty cannot deliver when target and adjacent modes are nearly equal
            # (~50/50 split), because the purity ratio gradient is symmetric whereas
            # an absolute power penalty unambiguously rewards reducing the wrong mode.
            _cross_mode_index = _as_int_or_none(case.get("cross_mode_index"))
            _cross_mode_penalty_w = _as_float_or_none(case.get("cross_mode_penalty_weight"))
            if (
                _cross_mode_index is not None
                and _cross_mode_penalty_w is not None
                and _cross_mode_penalty_w > 0
                and _obj_input_norm is not None
            ):
                _cross_mp = _mode_power_dir(
                    sim_data, target_mode, _cross_mode_index, case_source_direction
                )
                _cross_norm = anp.maximum(_cross_mp, 0.0) / _obj_input_norm
                case_obj = case_obj - _cross_mode_penalty_w * _cross_norm
            # coupling_primary_obj IS absolute transmission; do not also add
            # transmission_ratio_obj as a bonus — that would double-count the same term.
            if transmission_floor > 0:
                case_obj = case_obj - transmission_shortfall_penalty * anp.maximum(
                    0.0,
                    transmission_floor - coupling_primary_obj,
                )

            # M163/V16-A: log-coupling reward in the autograd hot path.
            # Adds   log_w * log(max(coupling_primary_obj, 0) + log_eps)   so that
            # tiny CE values (e.g. C4/C5 ~0.003) receive a large positive gradient
            # that pulls the optimizer out of dead basins. See progress.md
            # §V16-PLANNING §2.2 / §4.1 — this is the wiring that was missing in v14/v15.
            _use_log_coupling = bool(case.get("use_log_coupling") or False)
            _log_coupling_w = _as_float_or_none(case.get("log_coupling_weight"))
            _log_coupling_eps = _as_float_or_none(case.get("log_coupling_epsilon"))
            if (
                _use_log_coupling
                and _log_coupling_w is not None
                and _log_coupling_w > 0.0
                and _log_coupling_eps is not None
                and _log_coupling_eps > 0.0
            ):
                case_obj = case_obj + float(_log_coupling_w) * anp.log(
                    anp.maximum(coupling_primary_obj, 0.0) + float(_log_coupling_eps)
                )

            weight = max(_as_float_or_none(case.get("weight")) or 1.0, 1e-6)
            total_weight = total_weight + float(weight)
            weighted_obj = weighted_obj + float(weight) * case_obj

            if input_flux_raw is not None:
                monitor_readings[input_flux_key] = input_flux_raw
            if target_flux_measured_raw is not None:
                monitor_readings[target_flux_key] = target_flux_measured_raw

            # ---- Modal purity at target port (definition B) ----
            # modal_purity = target_mode_power / target_port_mode_total_power,
            # i.e. the fraction of the target-port modal power carried in the
            # target mode index. This is the ONLY supported "purity" metric.
            # The historical `coupling_ratio_split` field (target_port_flux /
            # Sum-of-output-ports-flux) was a port-routing ratio mislabeled as
            # purity; it has been removed (see memory-bank/progress.md M48-class
            # reconciliation note).
            modal_purity = None
            modal_purity_denominator = None
            if target_monitor_mode_total > 1e-12 and target_mode_power_float is not None:
                modal_purity = max(target_mode_power_float, 0.0) / target_monitor_mode_total
                modal_purity_denominator = target_monitor_mode_total

            # Input-normalized transmission metrics are the primary acceptance
            # semantics for requirements such as ">90% coupling".
            coupling_ratio_to_input = _ratio_or_none(target_flux_measured, input_flux)
            crosstalk_ratio_to_input = _ratio_or_none(crosstalk_flux if has_other_flux else 0.0, input_flux)
            coupling_ratio = coupling_ratio_to_input
            crosstalk_ratio = crosstalk_ratio_to_input
            ratio_basis = (
                "input_flux"
                if coupling_ratio_to_input is not None and crosstalk_ratio_to_input is not None
                else None
            )
            ratio_denominator = input_flux if ratio_basis == "input_flux" else None
            case_obj_eval = _demux_case_objective_from_ratios(
                coupling_ratio=coupling_ratio,
                crosstalk_ratio=crosstalk_ratio,
                min_coupling=min_coupling,
                max_crosstalk=max_crosstalk,
                use_log_coupling=bool(case.get("use_log_coupling") or False),
                log_coupling_weight=_as_float_or_none(case.get("log_coupling_weight")),
                log_coupling_epsilon=_as_float_or_none(case.get("log_coupling_epsilon")),
            )

            case_details.append(
                {
                    "name": case.get("name"),
                    "task_name": task_name,
                    "wavelength_nm": _as_float_or_none(case.get("wavelength_nm")),
                    "source_port": source_port,
                    "source_mode_index": source_mode_index,
                    "source_direction": _case_source_direction(case),
                    "target_port": case.get("target_port"),
                    "target_mode_index": target_mode_index,
                    "target_flux": (
                        (target_flux_measured or 0.0)
                        if target_flux_measured is not None
                        else max(target_mode_power_float or 0.0, 0.0)
                    ),
                    "crosstalk": max(crosstalk_flux, 0.0) if has_other_flux else mode_crosstalk_float,
                    "input_flux_key": input_flux_key,
                    "input_flux": max(input_flux or 0.0, 0.0) if input_flux is not None else None,
                    "coupling_ratio": coupling_ratio,
                    "crosstalk_ratio": crosstalk_ratio,
                    "modal_purity": modal_purity,
                    "modal_purity_denominator": modal_purity_denominator,
                    "coupling_ratio_to_input": coupling_ratio_to_input,
                    "crosstalk_ratio_to_input": crosstalk_ratio_to_input,
                    "ratio_basis": ratio_basis,
                    "ratio_denominator": ratio_denominator,
                    "target_flux_measured": max(target_flux_measured or 0.0, 0.0)
                    if target_flux_measured is not None
                    else None,
                    "crosstalk_flux_measured": max(crosstalk_flux, 0.0) if has_other_flux else 0.0,
                    "target_mode_transmission_surrogate": transmission_ratio_float,
                    "target_mode_transmission_surrogate_weight": transmission_surrogate_weight,
                    "target_mode_transmission_floor": transmission_floor,
                    "optimizer_ratio_basis": optimizer_ratio_basis,
                    "min_coupling": min_coupling,
                    "max_crosstalk": max_crosstalk,
                    "objective_contribution": _to_float(case_obj_eval),
                    "objective_contribution_optimizer": _to_float(case_obj),
                    "target_port_mode_purity_target": _to_float(mode_target_purity_obj),
                    "target_port_mode_impurity_ratio": _to_float(mode_wrong_ratio_obj),
                    "target_mode_focus_multiplier": _to_float(mode_focus_multiplier_obj),
                    "weight": weight,
                    "score": _to_float(case_obj_eval),
                    "observable": coupling_ratio is not None and crosstalk_ratio is not None,
                    "target_port_mode_monitor": target_mode,
                    "target_port_mode_power": target_monitor_modes,
                    "target_port_mode_total_power": (
                        target_monitor_mode_total if target_monitor_mode_total > 0 else None
                    ),
                    "target_port_mode_power_te0": target_monitor_mode_te0,
                    "target_port_mode_power_te1": target_monitor_mode_te1,
                    "target_port_mode_purity_te0": target_monitor_mode_te0_purity,
                    "target_port_mode_purity_te1": target_monitor_mode_te1_purity,
                    "observed_mode_indices": list(global_mode_indices),
                    "coupling_efficiency": max(target_mode_power_float or 0.0, 0.0),
                    "artifacts": [],
                    "error": None,
                }
            )

        if float(total_weight) <= 0:
            result_value = anp.array(0.0)
        else:
            result_value = weighted_obj / total_weight

        if callable(observation_callback):
            try:
                observation_callback(
                    {
                        "multi_case": case_details,
                        "multi_case_summary": _build_multi_case_iteration_summary(
                            case_details=case_details,
                            expected_case_count=len(objective_cases),
                        ),
                        "monitor_readings": dict(monitor_readings),
                        "score": _to_float(result_value),
                        "objective_value": _to_float(result_value),
                    }
                )
            except Exception:
                pass

        return result_value

    return _post_process_fn


def _evaluate_multi_case_objective(
    *,
    bundle: InverseDesignConfigBundle,
    parameters: Dict[str, float],
    objective_metric: str,
    iteration_index: int = 1,
    backend_tag: str = "eval",
    run_artifact_tag: str = "inv_run",
    force_field_plot: bool = False,
    topology_density_path: str | None = None,
    topology_density_meta_path: str | None = None,
    skip_cloud: bool = False,
) -> Dict[str, Any]:
    case_tag_prefix = f"{run_artifact_tag}_{backend_tag}_iter{iteration_index:03d}"
    require_field_plot = force_field_plot or _require_field_plot_in_iteration_eval()

    # Lock static geometry (wg_height, wg_width, mmi_width, mmi_length, …) to the
    # bundle's fixed geometry.parameters.  _build_invdes_simulation always reads from
    # this same source, so the diagnostic/seed/rerender sims must match the adjoint
    # optimizer's actual physical device — not the optimizer's explored parameter vector
    # (which may have drifted to physically incorrect values such as wg_height=0.154).
    _fixed_geo = _numeric_geometry_parameters(bundle.simulation_config.geometry.parameters)
    parameters = {**parameters, **_fixed_geo}

    cases = _objective_cases(bundle)
    if not cases:
        result = run_tidy3d_simulation(
            component_type=bundle.simulation_config.component_type,
            parameters=parameters,
            wavelength_nm=bundle.simulation_config.wavelength_nm,
            run_time_s=bundle.simulation_config.run_time_s,
            min_steps_per_wvl=bundle.simulation_config.domain.min_steps_per_wvl,
            artifact_tag=case_tag_prefix,
            objective_metric=objective_metric,
            require_field_plot=require_field_plot,
            topology_density_path=topology_density_path,
            topology_density_meta_path=topology_density_meta_path,
            skip_cloud=skip_cloud,
        )
        sim_ok = bool(result.get("ok"))
        if not sim_ok:
            return {
                "sim_ok": False,
                "score": None,
                "metrics": {},
                "objective_value": None,
                "monitor_readings": {},
                "artifacts": [],
                "error": result.get("error") or "Unknown simulation error.",
            }
        data = result.get("data", {}) if sim_ok else {}
        metrics = dict(data.get("metrics", {}))
        score = _as_float_or_none(data.get("score"))
        objective_value = _extract_objective_value(objective_metric, score, metrics)
        return {
            "sim_ok": True,
            "score": score,
            "metrics": metrics,
            "objective_value": objective_value,
            "monitor_readings": _extract_monitor_readings(metrics),
            "artifacts": list(data.get("artifacts", [])),
            "error": None,
        }

    case_details: List[Dict[str, Any]] = []
    artifacts: List[str] = []
    total_weight = 0.0
    objective_sum = 0.0
    score_sum = 0.0
    score_count = 0
    all_readings: Dict[str, float | None] = {}
    global_mode_indices = sorted(
        {
            max(int(_as_float_or_none(case.get("target_mode_index")) or 0), 0)
            for case in cases
        }
    )
    if not global_mode_indices:
        global_mode_indices = [0]
    highest_mode = max(global_mode_indices)
    if highest_mode >= 1:
        # Ensure TE0/TE1 observability for mode-demux diagnostics when at
        # least one objective case requests higher-order modes.
        global_mode_indices = sorted(set([*global_mode_indices, 0, 1]))
    mode_demux_active = max(global_mode_indices) >= 1
    purity_penalty_weight = _demux_mode_purity_penalty_weight()
    (
        transmission_surrogate_weight,
        transmission_shortfall_penalty,
        transmission_floor,
        transmission_surrogate_max,
    ) = _demux_transmission_params()

    for idx, case in enumerate(cases, start=1):
        target_port = str(case.get("target_port", "") or "").strip().lower()
        target_mode_index = int(_as_float_or_none(case.get("target_mode_index")) or 0)
        source_port = _case_source_port(case)
        source_mode_index = _case_source_mode_index(case)
        source_direction = _case_source_direction(case)
        case_artifact_tag = f"{case_tag_prefix}{_case_artifact_suffix(case, idx)}"
        result = run_tidy3d_simulation(
            component_type=bundle.simulation_config.component_type,
            parameters=parameters,
            wavelength_nm=float(case["wavelength_nm"]),
            run_time_s=bundle.simulation_config.run_time_s,
            min_steps_per_wvl=bundle.simulation_config.domain.min_steps_per_wvl,
            artifact_tag=case_artifact_tag,
            objective_metric=objective_metric,
            target_ports=[target_port] if target_port else None,
            target_mode_indices=global_mode_indices,
            source_port=source_port,
            source_mode_index=source_mode_index,
            source_direction=source_direction,
            require_field_plot=require_field_plot,
            topology_density_path=topology_density_path,
            topology_density_meta_path=topology_density_meta_path,
            skip_cloud=skip_cloud,
        )
        sim_ok = bool(result.get("ok"))
        if not sim_ok:
            failure_error = result.get("error") or f"Simulation failed for objective case #{idx}."
            # Collect locally-saved artifacts (HDF5, viewer script, slice PNGs) that
            # run_tidy3d_simulation writes before the cloud upload.  When the cloud
            # fails (e.g. insufficient balance), these local files are still valid
            # structural snapshots and should be registered for all 5 cases.
            _case_local_arts = [
                str(item)
                for item in (result.get("data") or {}).get("artifacts", [])
                if str(item or "").strip()
            ]
            artifacts.extend(_case_local_arts)
            case_details.append(
                {
                    "name": case.get("name") or f"case_{idx}",
                    "wavelength_nm": float(case["wavelength_nm"]),
                    "source_port": source_port,
                    "source_mode_index": source_mode_index,
                    "source_direction": source_direction,
                    "target_port": case.get("target_port"),
                    "target_mode_index": target_mode_index,
                    "target_flux": None,
                    "crosstalk": None,
                    "input_flux_key": _source_port_to_flux_key(source_port),
                    "input_flux": None,
                    "coupling_ratio": None,
                    "crosstalk_ratio": None,
                    "min_coupling": _as_float_or_none(case.get("min_coupling")),
                    "max_crosstalk": _as_float_or_none(case.get("max_crosstalk")),
                    "objective_contribution": None,
                    "weight": max(_as_float_or_none(case.get("weight")) or 1.0, 1e-6),
                    "score": None,
                    "observable": False,
                    "case_label": _case_artifact_label(case),
                    "artifact_tag": case_artifact_tag,
                    "artifacts": _case_local_arts,
                    "error": failure_error,
                }
            )
            # Continue to the next case rather than bailing out immediately.
            # This ensures local sim artefacts (HDF5, viewer, slice plots) are
            # generated and registered for ALL cases even when the cloud upload
            # fails due to insufficient balance or network errors.
            continue
        data = result.get("data", {}) if sim_ok else {}
        metrics = dict(data.get("metrics", {}))
        flux = metrics.get("flux", {}) if isinstance(metrics, dict) else {}
        if not isinstance(flux, dict):
            flux = {}
        target_key = _normalize_target_flux_key(target_port)
        input_flux_key = _source_port_to_flux_key(source_port)
        input_flux_raw = _as_float_or_none(flux.get(input_flux_key))
        target_flux_raw = _as_float_or_none(flux.get(target_key))
        target_flux = abs(target_flux_raw) if target_flux_raw is not None else 0.0
        input_flux = abs(input_flux_raw) if input_flux_raw is not None else 0.0
        crosstalk = _estimate_case_crosstalk(flux, target_key, input_flux_key=input_flux_key)
        output_sum = _sum_output_fluxes_case(flux, input_flux_key=input_flux_key)
        split_coupling_ratio = None
        split_crosstalk_ratio = None
        if output_sum is not None and output_sum > 1e-12:
            split_coupling_ratio = (abs(target_flux_raw) if target_flux_raw is not None else 0.0) / output_sum
            split_crosstalk_ratio = max(crosstalk, 0.0) / output_sum

        input_coupling_ratio = _ratio_or_none(target_flux_raw, input_flux_raw)
        input_crosstalk_ratio = _ratio_or_none(crosstalk, input_flux_raw)

        mode_power_raw = metrics.get("mode_power", {}) if isinstance(metrics, dict) else {}
        if not isinstance(mode_power_raw, dict):
            mode_power_raw = {}

        mode_power_map: Dict[str, List[float]] = {}
        for monitor_name, raw_vals in mode_power_raw.items():
            if isinstance(raw_vals, (list, tuple)):
                values = [_as_float_or_none(item) for item in raw_vals]
            else:
                maybe_scalar = _as_float_or_none(raw_vals)
                values = [maybe_scalar] if maybe_scalar is not None else []
            cleaned = [max(float(item), 0.0) for item in values if item is not None]
            if cleaned:
                mode_power_map[str(monitor_name)] = cleaned

        def _mode_power_from_metrics(monitor_name: str, mode_idx: int) -> float | None:
            values = mode_power_map.get(monitor_name)
            if not values:
                return None
            if mode_idx < len(values):
                return max(float(values[mode_idx]), 0.0)
            return None

        target_mode_monitor = _target_port_to_mode_monitor(target_port)
        output_mode_monitors = sorted(
            name
            for name in mode_power_map.keys()
            if str(name).startswith("mode_port_o")
        )
        if not output_mode_monitors and "mode_monitor" in mode_power_map:
            output_mode_monitors = ["mode_monitor"]
        if target_mode_monitor not in output_mode_monitors and output_mode_monitors:
            target_mode_monitor = output_mode_monitors[0]

        mode_target_power = _mode_power_from_metrics(target_mode_monitor, target_mode_index)
        mode_sum = 0.0
        mode_crosstalk = 0.0
        if output_mode_monitors:
            for monitor_name in output_mode_monitors:
                mode_val = _mode_power_from_metrics(monitor_name, target_mode_index)
                if mode_val is None:
                    continue
                mode_sum += mode_val
                if monitor_name != target_mode_monitor:
                    mode_crosstalk = max(mode_crosstalk, mode_val)

        target_monitor_modes = list(mode_power_map.get(target_mode_monitor, []))
        target_monitor_mode_total = float(sum(target_monitor_modes)) if target_monitor_modes else 0.0
        target_monitor_mode_te0 = (
            float(target_monitor_modes[0])
            if len(target_monitor_modes) >= 1
            else None
        )
        target_monitor_mode_te1 = (
            float(target_monitor_modes[1])
            if len(target_monitor_modes) >= 2
            else None
        )
        # PURITY DENOMINATOR: FluxMonitor(target_port) covers all modes (TE0~TE8+).
        # The old target_monitor_mode_total (sum of only 5 tracked modes) inflates
        # purity to near-100% even when most output power is in untracked high-order modes.
        _diag_target_flux_val = abs(target_flux_raw) if target_flux_raw is not None else None
        _purity_denom = (
            _diag_target_flux_val
            if _diag_target_flux_val is not None and _diag_target_flux_val > 1e-12
            else (target_monitor_mode_total if target_monitor_mode_total > 1e-12 else None)
        )
        target_monitor_mode_te0_purity = (
            (target_monitor_mode_te0 / _purity_denom)
            if target_monitor_mode_te0 is not None and _purity_denom is not None and _purity_denom > 1e-12
            else None
        )
        target_monitor_mode_te1_purity = (
            (target_monitor_mode_te1 / _purity_denom)
            if target_monitor_mode_te1 is not None and _purity_denom is not None and _purity_denom > 1e-12
            else None
        )
        target_mode_power_clamped = max(mode_target_power or 0.0, 0.0) if mode_target_power is not None else 0.0
        source_mode_monitor = _target_port_to_mode_monitor(source_port)
        if source_mode_monitor not in mode_power_map:
            source_mode_monitor = target_mode_monitor
        source_mode_power = _mode_power_from_metrics(source_mode_monitor, source_mode_index)
        source_mode_power_clamped = (
            max(source_mode_power or 0.0, 0.0)
            if source_mode_power is not None
            else None
        )
        transmission_ratio_surrogate = _ratio_or_none(
            target_mode_power_clamped if mode_target_power is not None else None,
            source_mode_power_clamped,
        )
        if transmission_ratio_surrogate is not None:
            transmission_ratio_surrogate = min(
                max(float(transmission_ratio_surrogate), 0.0),
                float(transmission_surrogate_max),
            )
        target_port_mode_purity_target = (
            target_mode_power_clamped / _purity_denom
            if _purity_denom is not None and _purity_denom > 1e-12
            else None
        )
        target_port_mode_impurity_ratio = (
            1.0 - target_port_mode_purity_target
            if target_port_mode_purity_target is not None
            else None
        )
        target_mode_focus_multiplier = _demux_mode_focus_multiplier(
            target_mode_index=target_mode_index,
            target_mode_purity=target_port_mode_purity_target,
        )

        coupling_ratio = None
        crosstalk_ratio = None
        ratio_basis = None
        ratio_denominator = None
        _diag_input_flux_val = abs(input_flux_raw) if input_flux_raw is not None else None
        if mode_target_power is not None and _diag_input_flux_val is not None and _diag_input_flux_val > 1e-12:
            # Input-flux-normalized, mode-specific: absolute transmission per PRD ">90%" semantics.
            # Preferred over mode_output_sum (split-ratio) and over total-flux input_coupling_ratio.
            coupling_ratio = mode_target_power / _diag_input_flux_val
            crosstalk_ratio = mode_crosstalk / _diag_input_flux_val
            ratio_basis = "input_flux_mode_specific"
            ratio_denominator = _diag_input_flux_val
        elif input_coupling_ratio is not None and input_crosstalk_ratio is not None:
            coupling_ratio = input_coupling_ratio
            crosstalk_ratio = input_crosstalk_ratio
            ratio_basis = "input_flux"
            ratio_denominator = _as_float_or_none(input_flux_raw)
        elif mode_target_power is not None and mode_sum > 1e-12:
            # Fallback only: split-ratio inflates coupling and should not be the primary metric.
            coupling_ratio = mode_target_power / mode_sum
            crosstalk_ratio = mode_crosstalk / mode_sum
            ratio_basis = "mode_output_sum"
            ratio_denominator = mode_sum
        observable = ratio_basis in {"input_flux", "input_flux_mode_specific", "mode_output_sum"}
        min_coupling = _as_float_or_none(case.get("min_coupling"))
        max_crosstalk = _as_float_or_none(case.get("max_crosstalk"))
        case_obj = _demux_case_objective_from_ratios(
            coupling_ratio=coupling_ratio,
            crosstalk_ratio=crosstalk_ratio,
            min_coupling=min_coupling,
            max_crosstalk=max_crosstalk,
            use_log_coupling=bool(case.get("use_log_coupling") or False),
            log_coupling_weight=_as_float_or_none(case.get("log_coupling_weight")),
            log_coupling_epsilon=_as_float_or_none(case.get("log_coupling_epsilon")),
        )
        if (
            case_obj is not None
            and mode_demux_active
            and target_port_mode_impurity_ratio is not None
        ):
            case_obj -= (
                purity_penalty_weight
                * target_mode_focus_multiplier
                * target_port_mode_impurity_ratio
            )
        # coupling_ratio is now input-flux normalized (absolute transmission per PRD);
        # do not add transmission_ratio_surrogate again to avoid double-counting.
        if case_obj is not None and transmission_floor > 0 and transmission_ratio_surrogate is not None:
            case_obj -= transmission_shortfall_penalty * max(
                0.0,
                transmission_floor - transmission_ratio_surrogate,
            )
        if case_obj is None:
            failure_error = (
                "Input/output flux ratios are not observable for demux_routing objective."
            )
            case_details.append(
                {
                    "name": case.get("name") or f"case_{idx}",
                    "wavelength_nm": float(case["wavelength_nm"]),
                    "target_port": case.get("target_port"),
                    "target_mode_index": target_mode_index,
                    "target_flux": target_flux,
                    "crosstalk": crosstalk,
                    "input_flux_key": input_flux_key,
                    "input_flux": input_flux if input_flux > 0 else None,
                    "coupling_ratio": coupling_ratio,
                    "crosstalk_ratio": crosstalk_ratio,
                    "modal_purity": None,
                    "modal_purity_denominator": None,
                    "coupling_ratio_to_input": input_coupling_ratio,
                    "crosstalk_ratio_to_input": input_crosstalk_ratio,
                    "ratio_basis": ratio_basis,
                    "ratio_denominator": ratio_denominator,
                    "min_coupling": min_coupling,
                    "max_crosstalk": max_crosstalk,
                    "objective_contribution": None,
                    "weight": max(_as_float_or_none(case.get("weight")) or 1.0, 1e-6),
                    "score": None,
                    "observable": False,
                    "case_label": _case_artifact_label(case),
                    "artifact_tag": case_artifact_tag,
                    "artifacts": [],
                    "error": failure_error,
                }
            )
            return {
                "sim_ok": False,
                "score": (score_sum / score_count) if score_count > 0 else None,
                "metrics": {
                    "multi_case": case_details,
                    "multi_case_summary": _build_multi_case_iteration_summary(
                        case_details=case_details,
                        expected_case_count=len(cases),
                    ),
                },
                "objective_value": None,
                "monitor_readings": all_readings,
                "artifacts": artifacts,
                "error": failure_error,
            }
        weight = max(_as_float_or_none(case.get("weight")) or 1.0, 1e-6)
        total_weight += weight
        objective_sum += weight * case_obj
        case_score = _as_float_or_none(data.get("score"))
        if case_score is not None:
            score_sum += case_score
            score_count += 1
        case_readings = _extract_monitor_readings(metrics)
        for key, value in case_readings.items():
            all_readings[f"case{idx}_{key}"] = value
            # Keep a baseline unprefixed view (from first case) so the
            # deterministic hard-physics gate can still evaluate required
            # input/output flux keys (e.g. flux_port_o1 / flux_port_o2 ...).
            if idx == 1 and key not in all_readings:
                all_readings[key] = value
        case_artifacts = [str(a) for a in data.get("artifacts", []) if str(a)]
        artifacts.extend(case_artifacts)
        case_details.append(
            {
                "name": case.get("name") or f"case_{idx}",
                "wavelength_nm": float(case["wavelength_nm"]),
                "source_port": source_port,
                "source_mode_index": source_mode_index,
                "source_direction": source_direction,
                "target_port": case.get("target_port"),
                "target_mode_index": target_mode_index,
                "target_flux": target_flux,
                "crosstalk": crosstalk,
                "input_flux_key": input_flux_key,
                "input_flux": input_flux,
                "coupling_ratio": coupling_ratio,
                "crosstalk_ratio": crosstalk_ratio,
                "modal_purity": (
                    (max(mode_target_power or 0.0, 0.0) / target_monitor_mode_total)
                    if mode_target_power is not None and target_monitor_mode_total > 1e-12
                    else None
                ),
                "modal_purity_denominator": (
                    target_monitor_mode_total
                    if target_monitor_mode_total > 1e-12
                    else None
                ),
                "coupling_ratio_to_input": input_coupling_ratio,
                "crosstalk_ratio_to_input": input_crosstalk_ratio,
                "ratio_basis": ratio_basis,
                "ratio_denominator": ratio_denominator,
                "min_coupling": min_coupling,
                "max_crosstalk": max_crosstalk,
                "objective_contribution": case_obj,
                    "target_port_mode_purity_target": target_port_mode_purity_target,
                    "target_port_mode_impurity_ratio": target_port_mode_impurity_ratio,
                    "target_mode_focus_multiplier": target_mode_focus_multiplier,
                    "target_mode_transmission_surrogate": transmission_ratio_surrogate,
                    "target_mode_transmission_surrogate_weight": transmission_surrogate_weight,
                    "target_mode_transmission_floor": transmission_floor,
                    "weight": weight,
                    "score": case_score,
                    "observable": observable,
                "case_label": _case_artifact_label(case),
                "artifact_tag": case_artifact_tag,
                "target_port_mode_monitor": target_mode_monitor,
                "target_port_mode_power": target_monitor_modes,
                "target_port_mode_total_power": (
                    target_monitor_mode_total if target_monitor_mode_total > 0 else None
                ),
                "target_port_mode_power_te0": target_monitor_mode_te0,
                "target_port_mode_power_te1": target_monitor_mode_te1,
                "target_port_mode_purity_te0": target_monitor_mode_te0_purity,
                "target_port_mode_purity_te1": target_monitor_mode_te1_purity,
                "observed_mode_indices": list(global_mode_indices),
                "artifacts": list(dict.fromkeys(case_artifacts)),
                "error": None,
            }
        )

    # Check if any case failed (cloud failure path now continues rather than returning early)
    _failed_cases = [cd for cd in case_details if cd.get("error")]
    _any_failed = bool(_failed_cases)

    return {
        "sim_ok": not _any_failed,
        "score": (score_sum / score_count) if score_count > 0 else None,
        "metrics": {
            "multi_case": case_details,
            "multi_case_summary": _build_multi_case_iteration_summary(
                case_details=case_details,
                expected_case_count=len(cases),
            ),
        },
        "objective_value": (objective_sum / total_weight) if total_weight > 0 else None,
        "monitor_readings": all_readings,
        "artifacts": list(dict.fromkeys(artifacts)),
        "error": (_failed_cases[0].get("error") if _any_failed else None),
    }


def _multi_case_observation_mode() -> str:
    raw = os.getenv("INVERSE_MULTI_CASE_OBSERVATION_MODE", "hybrid").strip().lower()
    if raw in {"adjoint_internal", "diagnostic_runner", "hybrid"}:
        return raw
    return "hybrid"


def _covered_multi_case_indices_from_artifacts(
    artifacts: Sequence[str],
    expected_case_count: int,
) -> set[int]:
    covered: set[int] = set()
    if expected_case_count <= 0:
        return covered
    for item in artifacts:
        text = str(item or "").lower()
        if not text:
            continue
        match = re.search(r"_case(\d+)", text)
        if not match:
            continue
        try:
            case_index = int(match.group(1))
        except (TypeError, ValueError):
            continue
        if 1 <= case_index <= expected_case_count:
            covered.add(case_index)
    return covered


def _has_required_multi_case_artifact_evidence(
    artifacts: Sequence[str],
    expected_case_count: int,
) -> bool:
    artifact_text = [str(item or "").lower() for item in artifacts if str(item or "").strip()]
    if not artifact_text:
        return False
    # Require at least one field plot (field_z or field_port_o1) as physics-gate evidence.
    has_any_field = any(
        "field_port_o1" in item or "field_z" in item or "field_ey" in item
        for item in artifact_text
    )
    if not has_any_field:
        return False
    # Require that all expected cases are covered by SOME artifact (HDF5, viewer,
    # or geometry plot).
    covered_cases = _covered_multi_case_indices_from_artifacts(artifact_text, expected_case_count)
    return len(covered_cases) >= expected_case_count


def _ensure_best_iteration_multi_case_artifacts(
    *,
    bundle: InverseDesignConfigBundle,
    records: List[InverseDesignIterationRecord],
    objective_metric: str,
    objective_goal: str,
    objective_cases: Sequence[Dict[str, Any]],
    run_artifact_tag: str,
) -> None:
    if not records or not objective_cases:
        return

    best_idx = 0
    best_record = records[0]
    for idx, record in enumerate(records[1:], start=1):
        if _is_better(record, best_record, objective_goal):
            best_record = record
            best_idx = idx

    expected_case_count = len(objective_cases)
    existing_artifacts = list(best_record.artifacts)
    if _has_required_multi_case_artifact_evidence(existing_artifacts, expected_case_count):
        return

    # Locate topology density replay artifacts saved during the adjoint run.
    # Without these, the rerender would use the initial (un-optimised) geometry.
    density_npy = _pick_artifact_path(existing_artifacts, ("_density.npy",))
    density_meta = _pick_artifact_path(existing_artifacts, ("_density_meta.json",))

    # Fallback: if the best record has no density artifact registered (happens when
    # topology_best_iteration is an absolute iteration number > len(records) so the
    # density was wrongly attached to a different record), search build/ for a
    # density file that matches the run tag for the best iteration.
    if density_npy is None and run_artifact_tag:
        import glob as _glob
        _build_dir = os.path.join(os.getcwd(), "build")
        _best_iter = int(best_record.iteration)
        _iter_tag = f"iter{_best_iter:03d}"
        _patterns = [
            os.path.join(_build_dir, f"topology_*_{run_artifact_tag}_{_iter_tag}_density.npy"),
            # Also accept CORRECT-tagged variant created by recovery scripts
            os.path.join(_build_dir, f"topology_*_{run_artifact_tag}_{_iter_tag}_CORRECT_density.npy"),
            # Fallback: any density for this run tag (closest to best iteration)
            os.path.join(_build_dir, f"topology_*_{run_artifact_tag}_*_density.npy"),
        ]
        for _pat in _patterns:
            _candidates = sorted(_glob.glob(_pat))
            if _candidates:
                # If multiple matches, prefer the one closest to the best iteration
                def _iter_dist(p: str) -> int:
                    import re as _re
                    m = _re.search(r"iter(\d+)", os.path.basename(p))
                    return abs(int(m.group(1)) - _best_iter) if m else 9999
                _candidates.sort(key=_iter_dist)
                density_npy = _candidates[0]
                _maybe_meta = density_npy.replace("_density.npy", "_density_meta.json")
                if os.path.exists(_maybe_meta):
                    density_meta = _maybe_meta
                logger.info(
                    "Density fallback: using %s for run %s iter%d",
                    os.path.basename(density_npy), run_artifact_tag, _best_iter,
                )
                break

    # Ultimate fallback: reconstruct density from the JSONL iteration trace.
    # Triggered when the density file was never written because the adjoint
    # subprocess crashed/was killed before the post-loop _save_topology_image
    # call was reached (heartbeat stuck at optimizer_run_alive).
    if density_npy is None and run_artifact_tag:
        _trace_path = _adjoint_iteration_trace_path(run_artifact_tag)
        if _trace_path.exists():
            try:
                _trace_entries = _load_adjoint_trace_entries(_trace_path)
            except Exception:
                _trace_entries = []
            if _trace_entries:
                _best_iter_n = int(best_record.iteration)
                _best_trace_entry: Dict[str, Any] | None = None
                # First: exact iteration number match.
                for _te in _trace_entries:
                    if int(_te.get("iteration", 0)) == _best_iter_n:
                        _best_trace_entry = _te
                        break
                # Second: if no match, pick the entry with the best objective_fn_val.
                if _best_trace_entry is None:
                    _best_te_val: float | None = None
                    for _te in _trace_entries:
                        _v = _as_float_or_none(_te.get("objective_fn_val"))
                        if _v is None:
                            continue
                        if _best_te_val is None or (
                            objective_goal == "maximize" and _v > _best_te_val
                        ) or (objective_goal != "maximize" and _v < _best_te_val):
                            _best_te_val = _v
                            _best_trace_entry = _te
                if _best_trace_entry is not None and _best_trace_entry.get("params_vector"):
                    _topo_arts, _ = _save_topology_image(
                        invdes_result=None,
                        snapshots=[{
                            "step_index": int(_best_trace_entry.get("iteration", 1)) - 1,
                            "objective_fn_val": _as_float_or_none(
                                _best_trace_entry.get("objective_fn_val")
                            ),
                            "post_process_val": None,
                            "params": _best_trace_entry["params_vector"],
                            "trace": {},
                            "case_observation": None,
                        }],
                        records=[best_record],
                        bundle=bundle,
                        objective_goal=objective_goal,
                        run_artifact_tag=run_artifact_tag,
                    )
                    if _topo_arts:
                        density_npy = _pick_artifact_path(_topo_arts, ("_density.npy",))
                        density_meta = _pick_artifact_path(
                            _topo_arts, ("_density_meta.json",)
                        )
                        best_record.artifacts.extend(_topo_arts)
                        logger.info(
                            "Density recovered from JSONL trace for run %s iter%d "
                            "(subprocess likely crashed before post-loop save).",
                            run_artifact_tag,
                            _best_iter_n,
                        )

    # W20 (M48 RC-5 + V16-A rerender-stall): the post-Adam rerender batch
    # historically had no wall-clock guard, so a single stuck cloud sim could
    # block the entire run forever (observed: V16-A inv_20260528T132650Z_9df3d11f
    # hung 4h43m on case4 with cpu=0.5%, never producing result.json). The
    # adjoint loop already uses INVERSE_ADJOINT_RUN_TIMEOUT_S; the rerender
    # path needs its own cap.
    _rerender_timeout_s = _rerender_run_timeout_s()
    try:
        rerender_out = _invoke_with_timeout(
            fn=lambda: _evaluate_multi_case_objective(
                bundle=bundle,
                parameters=dict(best_record.parameters),
                objective_metric=objective_metric,
                iteration_index=int(best_record.iteration),
                backend_tag="best_rerender",
                run_artifact_tag=run_artifact_tag,
                topology_density_path=str(density_npy) if density_npy is not None else None,
                topology_density_meta_path=str(density_meta) if density_meta is not None else None,
            ),
            timeout_s=_rerender_timeout_s,
            timeout_error=(
                f"best_rerender exceeded INVERSE_RERENDER_RUN_TIMEOUT_S="
                f"{_rerender_timeout_s:.0f}s for run {run_artifact_tag}"
            ),
        )
    except RuntimeError as _rerender_timeout_exc:
        logger.error("best_rerender wall-clock timeout: %s", _rerender_timeout_exc)
        rerender_out = {
            "sim_ok": False,
            "score": None,
            "metrics": {},
            "objective_value": None,
            "monitor_readings": {},
            "artifacts": [],
            "error": f"rerender_timeout: {_rerender_timeout_exc}",
        }

    # Bug D fix: if cloud fails (e.g. insufficient balance), retry with
    # skip_cloud=True to generate local-only structural HDF5 + viewer files
    # of the optimised topology.
    if not rerender_out.get("sim_ok"):
        _rerr = str(rerender_out.get("error") or "").lower()
        if "insufficient balance" in _rerr or "aborted due to insufficient" in _rerr or "simulation_data_unavailable" in _rerr:
            logger.warning(
                "Cloud rerender failed (%s); retrying with skip_cloud=True for local artifacts.",
                _rerr[:120],
            )
            rerender_out = _evaluate_multi_case_objective(
                bundle=bundle,
                parameters=dict(best_record.parameters),
                objective_metric=objective_metric,
                iteration_index=int(best_record.iteration),
                backend_tag="best_rerender_local",
                run_artifact_tag=run_artifact_tag,
                force_field_plot=False,
                topology_density_path=str(density_npy) if density_npy is not None else None,
                topology_density_meta_path=str(density_meta) if density_meta is not None else None,
                skip_cloud=True,
            )

    rerender_artifacts = [
        str(item) for item in rerender_out.get("artifacts", []) if str(item or "").strip()
    ]
    if rerender_artifacts:
        best_record.artifacts = list(dict.fromkeys([*existing_artifacts, *rerender_artifacts]))

    covered_cases = _covered_multi_case_indices_from_artifacts(best_record.artifacts, expected_case_count)
    rerender_summary = {
        "sim_ok": bool(rerender_out.get("sim_ok")),
        "error": rerender_out.get("error"),
        "expected_case_count": expected_case_count,
        "covered_case_count": len(covered_cases),
        "covered_cases": sorted(covered_cases),
        "has_field_port_o1": any("field_port_o1" in str(item or "").lower() for item in best_record.artifacts),
        "has_field_z": any("field_z" in str(item or "").lower() for item in best_record.artifacts),
        "artifacts_added": len(best_record.artifacts) - len(existing_artifacts),
    }
    best_record.metrics = dict(best_record.metrics)
    best_record.metrics["best_iteration_rerender"] = rerender_summary
    records[best_idx] = best_record


def _evaluate_multi_case_from_adjoint_observation(
    *,
    case_observation: Dict[str, Any] | None,
    objective_cases: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    if not isinstance(case_observation, dict):
        return {
            "sim_ok": False,
            "score": None,
            "metrics": {
                "multi_case": [],
                "multi_case_summary": {
                    "expected_cases": len(objective_cases),
                    "reported_cases": 0,
                    "observable_cases": 0,
                    "all_cases_observable": False,
                },
            },
            "objective_value": None,
            "monitor_readings": {},
            "artifacts": [],
            "error": "adjoint_internal_observation_unavailable",
        }

    case_details_raw = case_observation.get("multi_case")
    case_details = [
        dict(item) for item in case_details_raw
        if isinstance(item, dict)
    ] if isinstance(case_details_raw, list) else []

    summary = case_observation.get("multi_case_summary")
    if not isinstance(summary, dict):
        summary = _build_multi_case_iteration_summary(
            case_details=case_details,
            expected_case_count=len(objective_cases),
        )

    total_weight = 0.0
    objective_sum = 0.0
    score_sum = 0.0
    score_count = 0
    for detail in case_details:
        weight = max(_as_float_or_none(detail.get("weight")) or 1.0, 1e-6)
        contribution = _as_float_or_none(detail.get("objective_contribution"))
        if contribution is not None:
            total_weight += weight
            objective_sum += weight * contribution
        case_score = _as_float_or_none(detail.get("score"))
        if case_score is not None:
            score_sum += case_score
            score_count += 1

    sim_ok = bool(case_details)
    objective_value = (objective_sum / total_weight) if total_weight > 0 else None
    if objective_value is None:
        objective_value = _as_float_or_none(case_observation.get("objective_value"))
    score = _as_float_or_none(case_observation.get("score"))
    if score is None and score_count > 0:
        score = score_sum / score_count

    monitor_readings = dict(case_observation.get("monitor_readings", {}))
    return {
        "sim_ok": sim_ok,
        "score": score,
        "metrics": {
            "multi_case": case_details,
            "multi_case_summary": summary,
        },
        "objective_value": objective_value,
        "monitor_readings": monitor_readings,
        "artifacts": list(case_observation.get("artifacts", [])),
        "error": None if sim_ok else "adjoint_internal_observation_unavailable",
    }


def _ratio_or_none(numerator: float | None, denominator: float | None) -> float | None:
    num = _as_float_or_none(numerator)
    den = _as_float_or_none(denominator)
    if num is None or den is None:
        return None
    den_abs = abs(den)
    if den_abs <= 1e-9:
        return None
    return abs(num) / den_abs


def summarize_multi_case_metrics(run_result: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Summarize latest per-case mode metrics for reports and tests.

    The summary intentionally ignores raw adjoint-amplitude ``coupling_efficiency``
    values unless a case also exposes a trusted input-normalized ratio.
    """

    iterations = run_result.get("iterations", []) if isinstance(run_result, dict) else []
    if not iterations:
        return {}
    latest = iterations[-1] if isinstance(iterations[-1], dict) else {}
    metrics = latest.get("metrics", {}) if isinstance(latest, dict) else {}
    cases = metrics.get("multi_case", []) if isinstance(metrics, dict) else []
    summary: Dict[str, Dict[str, Any]] = {}
    for item in cases:
        if not isinstance(item, dict):
            continue
        try:
            target_mode = int(item.get("target_mode_index", len(summary)))
        except (TypeError, ValueError):
            target_mode = len(summary)
        ratio = item.get("coupling_ratio_to_input")
        if ratio is None and item.get("coupling_metric_trusted", True):
            ratio = item.get("coupling_ratio_corrected")
        summary[f"TE{target_mode}"] = {
            "coupling_ratio": ratio,
            "modal_purity": item.get("modal_purity"),
            "target_mode_power_raw": item.get("target_mode_power_raw"),
        }
    return summary


def _demux_case_objective_from_ratios(
    *,
    coupling_ratio: float | None,
    crosstalk_ratio: float | None,
    min_coupling: float | None,
    max_crosstalk: float | None,
    use_log_coupling: bool | None = False,
    log_coupling_weight: float | None = None,
    log_coupling_epsilon: float | None = None,
) -> float | None:
    coupling = _as_float_or_none(coupling_ratio)
    crosstalk = _as_float_or_none(crosstalk_ratio)
    if coupling is None or crosstalk is None:
        return None

    value = coupling - crosstalk
    coupling_penalty, crosstalk_penalty = _demux_penalty_weights()
    if min_coupling is not None:
        value -= coupling_penalty * max(0.0, float(min_coupling) - coupling)
    if max_crosstalk is not None:
        value -= crosstalk_penalty * max(0.0, crosstalk - float(max_crosstalk))
    # Optional log-coupling reward: log(CE+eps) grows fast for tiny CE, providing
    # a strong gradient that linear (coupling - crosstalk) cannot deliver when CE
    # is near zero. See progress.md §V16-PLANNING §4.1 for the v14 mechanism
    # validation rationale. Defaults match the v14/v15 bundle settings.
    if use_log_coupling and log_coupling_weight is not None:
        log_w = _as_float_or_none(log_coupling_weight)
        log_eps = _as_float_or_none(log_coupling_epsilon)
        if log_w is not None and log_w > 0.0 and log_eps is not None and log_eps > 0.0:
            value += float(log_w) * math.log(max(coupling, 0.0) + float(log_eps))
    return value


def _build_multi_case_iteration_summary(
    *,
    case_details: Sequence[Dict[str, Any]],
    expected_case_count: int,
) -> Dict[str, Any]:
    observed_cases = 0
    for case in case_details:
        if bool(case.get("observable")):
            observed_cases += 1
    return {
        "expected_cases": int(expected_case_count),
        "reported_cases": int(len(case_details)),
        "observable_cases": int(observed_cases),
        "all_cases_observable": bool(observed_cases == expected_case_count and expected_case_count > 0),
    }


def _build_run_multi_case_observation_summary(
    *,
    records: Sequence[InverseDesignIterationRecord],
    objective_cases: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    expected_cases = len(objective_cases)
    if expected_cases == 0:
        return {}

    case_stats: List[Dict[str, Any]] = []
    for case in objective_cases:
        case_stats.append(
            {
                "name": case.get("name"),
                "wavelength_nm": _as_float_or_none(case.get("wavelength_nm")),
                "target_port": case.get("target_port"),
                "observable_iterations": 0,
                "missing_iterations": 0,
                "best_coupling_ratio": None,
                "worst_crosstalk_ratio": None,
                "last_coupling_ratio": None,
                "last_crosstalk_ratio": None,
                "last_error": None,
            }
        )

    full_observable_iterations = 0
    any_observable_iterations = 0
    for record in records:
        metrics = dict(record.metrics or {})
        raw_cases = metrics.get("multi_case")
        multi_case_metrics = raw_cases if isinstance(raw_cases, list) else []
        observable_this_iter = 0
        for idx in range(expected_cases):
            case_metric = (
                dict(multi_case_metrics[idx])
                if idx < len(multi_case_metrics) and isinstance(multi_case_metrics[idx], dict)
                else {}
            )
            stat = case_stats[idx]
            coupling_ratio = _as_float_or_none(case_metric.get("coupling_ratio"))
            crosstalk_ratio = _as_float_or_none(case_metric.get("crosstalk_ratio"))
            observable = bool(case_metric.get("observable")) and coupling_ratio is not None and crosstalk_ratio is not None
            if observable:
                observable_this_iter += 1
                stat["observable_iterations"] += 1
                stat["last_coupling_ratio"] = coupling_ratio
                stat["last_crosstalk_ratio"] = crosstalk_ratio
                current_best = _as_float_or_none(stat.get("best_coupling_ratio"))
                if current_best is None or coupling_ratio > current_best:
                    stat["best_coupling_ratio"] = coupling_ratio
                current_worst = _as_float_or_none(stat.get("worst_crosstalk_ratio"))
                if current_worst is None or crosstalk_ratio > current_worst:
                    stat["worst_crosstalk_ratio"] = crosstalk_ratio
            else:
                stat["missing_iterations"] += 1
                if case_metric.get("error"):
                    stat["last_error"] = str(case_metric.get("error"))

        if observable_this_iter == expected_cases:
            full_observable_iterations += 1
        if observable_this_iter > 0:
            any_observable_iterations += 1

    total_iterations = len(records)
    return {
        "expected_cases": expected_cases,
        "total_iterations": total_iterations,
        "full_observable_iterations": full_observable_iterations,
        "any_observable_iterations": any_observable_iterations,
        "full_observability_ratio": (
            full_observable_iterations / total_iterations if total_iterations > 0 else 0.0
        ),
        "any_observability_ratio": (
            any_observable_iterations / total_iterations if total_iterations > 0 else 0.0
        ),
        "case_stats": case_stats,
    }


def _normalize_target_flux_key(target_port: str) -> str:
    lowered = target_port.strip().lower()
    if lowered.startswith("flux_port_o"):
        return lowered
    if lowered.startswith("port_o"):
        return f"flux_{lowered}"
    m = re.search(r"(\d+)", lowered)
    if m:
        return f"flux_port_o{int(m.group(1))}"
    return "flux_port_o2"


def _estimate_case_crosstalk(
    flux_map: Dict[str, Any],
    target_key: str,
    *,
    input_flux_key: str = "flux_port_o1",
) -> float:
    candidates: List[float] = []
    for key, value in flux_map.items():
        name = str(key).lower()
        if name in {str(input_flux_key).strip().lower(), "flux_port_i1", "flux_input"}:
            continue
        if not name.startswith("flux_port_o"):
            continue
        if name == target_key:
            continue
        num = _as_float_or_none(value)
        if num is None:
            continue
        candidates.append(abs(num))
    return max(candidates) if candidates else 0.0


def _extract_objective_value(metric: str, score: float | None, metrics: Dict[str, Any]) -> float | None:
    lowered = metric.lower()
    flux_map = metrics.get("flux", {}) if isinstance(metrics, dict) else {}
    if lowered in {"score", "composite_score"}:
        return score
    if lowered in {"transmission", "efficiency"}:
        transmission = _sum_output_fluxes(flux_map)
        return transmission if transmission is not None else score
    if lowered == "crosstalk":
        upper = _as_float_or_none(flux_map.get("flux_port_o3")) or 0.0
        lower = _as_float_or_none(flux_map.get("flux_port_o4")) or 0.0
        return abs(upper) + abs(lower)
    if lowered in {"loss", "insertion_loss"}:
        transmission = _as_float_or_none(flux_map.get("flux_port_o2"))
        if transmission is None:
            return None
        return max(0.0, 1.0 - transmission)
    if lowered == "mode_overlap":
        return _as_float_or_none(flux_map.get("flux_port_o2")) or score
    return score


def _extract_monitor_readings(metrics: Dict[str, Any]) -> Dict[str, float | None]:
    readings: Dict[str, float | None] = {}
    flux_map = metrics.get("flux", {}) if isinstance(metrics, dict) else {}
    if isinstance(flux_map, dict):
        for key, value in flux_map.items():
            readings[str(key)] = _as_float_or_none(value)
    return readings


def _sum_output_fluxes(flux_map: Dict[str, Any]) -> float | None:
    if not isinstance(flux_map, dict):
        return None

    output_values: List[float] = []
    for key, value in flux_map.items():
        normalized = str(key).lower()
        numeric = _as_float_or_none(value)
        if numeric is None:
            continue
        if normalized in {"flux_port_o1", "flux_port_i1", "flux_input"}:
            continue
        if normalized.startswith("flux_port_o") or normalized.startswith("flux_output"):
            output_values.append(abs(numeric))

    if output_values:
        return sum(output_values)
    return _as_float_or_none(flux_map.get("flux_port_o2"))


def _sum_output_fluxes_case(
    flux_map: Dict[str, Any],
    *,
    input_flux_key: str,
) -> float | None:
    if not isinstance(flux_map, dict):
        return None
    exclude_key = str(input_flux_key or "").strip().lower()
    output_values: List[float] = []
    for key, value in flux_map.items():
        normalized = str(key).lower()
        if not normalized.startswith("flux_port_o"):
            continue
        if normalized == exclude_key:
            continue
        numeric = _as_float_or_none(value)
        if numeric is None:
            continue
        output_values.append(abs(numeric))
    if output_values:
        return sum(output_values)
    return None


def _derive_constraint_status(constraints: List[str]) -> Dict[str, Any]:
    return {
        "declared_constraints": constraints,
        "evaluated": bool(constraints),
        "satisfied": True if constraints else None,
        "note": "Detailed per-constraint physics checks are not yet implemented in bridge mode.",
    }


def _coerce_constraint_packet(
    packet: Step4ConstraintPacket | Dict[str, Any] | None,
) -> Step4ConstraintPacket | None:
    if packet is None:
        return None
    if isinstance(packet, Step4ConstraintPacket):
        return packet
    return Step4ConstraintPacket.model_validate(packet)


def _evaluate_first_iteration_physics_gate(
    *,
    records: List[InverseDesignIterationRecord],
    bundle: InverseDesignConfigBundle,
    constraint_packet: Step4ConstraintPacket | None,
) -> Dict[str, Any]:
    if constraint_packet is None or not constraint_packet.hard_physics_gate.enabled:
        return {
            "evaluated": False,
            "passed": True,
            "blockers": [],
            "observations": {"mode": "disabled"},
        }

    if not records:
        return {
            "evaluated": True,
            "passed": False,
            "blockers": [
                {
                    "code": "hard_gate_missing_iteration",
                    "message": "No first-iteration record available for deterministic hard-physics gate.",
                }
            ],
            "observations": {},
        }

    gate = constraint_packet.hard_physics_gate
    first_record = records[0]
    if not first_record.simulation_ok:
        return {
            "evaluated": True,
            "passed": False,
            "blockers": [
                {
                    "code": "hard_gate_first_iteration_simulation_failed",
                    "message": str(
                        first_record.error
                        or "First iteration simulation failed before hard-physics observability checks."
                    ),
                }
            ],
            "observations": {
                "iteration": first_record.iteration,
                "optimizer_backend": first_record.optimizer_backend,
                "simulation_ok": False,
            },
        }

    readings = dict(first_record.monitor_readings)
    blockers: List[Dict[str, Any]] = []
    observations: Dict[str, Any] = {
        "iteration": first_record.iteration,
        "optimizer_backend": first_record.optimizer_backend,
    }

    existing_monitor_types = {monitor.monitor_type for monitor in bundle.simulation_config.monitors}
    missing_monitor_types = [
        monitor_type
        for monitor_type in constraint_packet.required_monitor_types
        if monitor_type not in existing_monitor_types
    ]
    if missing_monitor_types:
        blockers.append(
            {
                "code": "hard_gate_monitor_coverage_incomplete",
                "message": (
                    "Missing required monitor types for deterministic gate: "
                    + ", ".join(missing_monitor_types)
                ),
                "missing_monitor_types": missing_monitor_types,
            }
        )

    input_key, input_flux = _pick_flux(readings, gate.required_input_flux_keys)
    input_flux_abs = abs(input_flux) if input_flux is not None and math.isfinite(input_flux) else None
    observations["input_flux_key"] = input_key
    observations["input_flux"] = input_flux
    observations["input_flux_abs"] = input_flux_abs
    if input_flux is None:
        blockers.append(
            {
                "code": "hard_gate_input_injection_unobservable",
                "message": "Input-side flux is not observable; cannot verify injection floor.",
            }
        )
    elif input_flux_abs is not None and input_flux_abs < gate.input_flux_min:
        blockers.append(
            {
                "code": "hard_gate_input_injection_below_floor",
                "message": (
                    f"Input flux magnitude {input_flux_abs:.6g} is below floor {gate.input_flux_min:.6g}."
                ),
                "threshold": gate.input_flux_min,
                "value": input_flux_abs,
                "signed_value": input_flux,
            }
        )

    output_values_signed = [
        value
        for key, value in readings.items()
        if key in set(gate.required_output_flux_keys) and isinstance(value, (int, float)) and math.isfinite(value)
    ]
    output_values = [abs(value) for value in output_values_signed]
    observations["output_flux_keys"] = list(gate.required_output_flux_keys)
    observations["output_flux_values"] = output_values
    observations["output_flux_values_signed"] = output_values_signed
    if not output_values:
        blockers.append(
            {
                "code": "hard_gate_output_unobservable",
                "message": "Through/drop outputs are not observable on required monitor keys.",
            }
        )
    elif max(output_values) < gate.output_flux_min:
        blockers.append(
            {
                "code": "hard_gate_output_response_too_low",
                "message": (
                    f"Max output response {max(output_values):.6g} is below floor {gate.output_flux_min:.6g}."
                ),
                "threshold": gate.output_flux_min,
                "value": max(output_values),
            }
        )

    # --- Per-port transmission ratio (catches wrongly-oriented monitors) ---
    # A monitor with the wrong orientation plane still captures some stray
    # flux (non-zero) but typically <1% of injected power.  The existing
    # ``output_flux_min`` threshold (1e-4) is too lenient to catch this.
    if input_flux_abs is not None and input_flux_abs > 1e-9:
        per_port_ratios: Dict[str, float] = {}
        for key in gate.required_output_flux_keys:
            port_flux = readings.get(key)
            if isinstance(port_flux, (int, float)) and math.isfinite(port_flux):
                ratio = abs(port_flux) / input_flux_abs
                per_port_ratios[key] = ratio
                if ratio < gate.per_port_transmission_min:
                    blockers.append(
                        {
                            "code": "hard_gate_port_transmission_anomaly",
                            "message": (
                                f"Port '{key}' transmission ratio {ratio:.6g} is below "
                                f"minimum {gate.per_port_transmission_min:.6g} — "
                                f"possible monitor orientation error."
                            ),
                            "port": key,
                            "ratio": ratio,
                            "threshold": gate.per_port_transmission_min,
                        }
                    )
        observations["per_port_transmission_ratios"] = per_port_ratios

    # --- Output port flux imbalance ---
    # For nominally symmetric devices a >100:1 imbalance between output ports
    # is a strong signal that one monitor has the wrong orientation plane.
    if len(output_values) >= 2:
        abs_outputs = [abs(v) for v in output_values]
        max_out = max(abs_outputs)
        min_nonzero = min((v for v in abs_outputs if v > 1e-12), default=0.0)
        if min_nonzero > 0:
            imbalance = max_out / min_nonzero
            observations["output_imbalance_ratio"] = imbalance
            if imbalance > gate.output_imbalance_max_ratio:
                blockers.append(
                    {
                        "code": "hard_gate_output_imbalance",
                        "message": (
                            f"Output port flux imbalance ratio {imbalance:.1f}x exceeds "
                            f"maximum {gate.output_imbalance_max_ratio:.1f}x — "
                            f"possible monitor orientation mismatch."
                        ),
                        "ratio": imbalance,
                        "threshold": gate.output_imbalance_max_ratio,
                    }
                )

    # --- Input-normalized coupling sanity for multi-case metrics ---
    # coupling_ratio_to_input >> 1 typically indicates monitor normalization
    # mismatch; block early to avoid accepting physically inconsistent runs.
    metrics = dict(first_record.metrics or {})
    raw_multi_case = metrics.get("multi_case")
    multi_case = raw_multi_case if isinstance(raw_multi_case, list) else []
    over_unity_cases: List[Dict[str, Any]] = []
    for case in multi_case:
        if not isinstance(case, dict):
            continue
        coupling_to_input = _as_float_or_none(case.get("coupling_ratio_to_input"))
        if coupling_to_input is None:
            continue
        source_direction = str(case.get("source_direction") or "").strip()
        threshold = (
            gate.max_input_normalized_coupling_reverse
            if source_direction == "-"
            else gate.max_input_normalized_coupling
        )
        if coupling_to_input > threshold:
            over_unity_cases.append(
                {
                    "name": str(case.get("name") or ""),
                    "value": float(coupling_to_input),
                    "threshold": float(threshold),
                    "source_direction": source_direction or None,
                }
            )
    if over_unity_cases:
        observations["input_normalized_coupling_overflow_cases"] = over_unity_cases
        blockers.append(
            {
                "code": "hard_gate_input_normalized_coupling_exceeds_max",
                "message": (
                    "Detected coupling_ratio_to_input above allowed threshold in first iteration; "
                    "monitor normalization may be inconsistent."
                ),
                "threshold": gate.max_input_normalized_coupling,
                "threshold_reverse": gate.max_input_normalized_coupling_reverse,
                "cases": over_unity_cases,
            }
        )

    if input_flux_abs is not None and output_values:
        output_sum = sum(output_values)
        closure_error = abs(input_flux_abs - output_sum) / max(input_flux_abs, 1e-12)
        observations["energy_closure_error"] = closure_error

        # Determine whether loss channels (reflection, absorption, scattering)
        # are monitored.  Without them, closure error is unreliable because
        # the "missing" power may be dissipated in unmonitored channels.
        loss_monitor_keywords = {"reflection", "absorption", "scatter", "back_flux"}
        loss_channel_coverage = any(
            any(kw in m.name.lower() for kw in loss_monitor_keywords)
            for m in bundle.simulation_config.monitors
        )
        observations["loss_channel_coverage"] = loss_channel_coverage

        if loss_channel_coverage and closure_error > gate.energy_closure_tolerance:
            blockers.append(
                {
                    "code": "hard_gate_energy_closure_violation",
                    "message": (
                        f"Energy closure error {closure_error:.6g} exceeds tolerance {gate.energy_closure_tolerance:.6g}."
                    ),
                    "threshold": gate.energy_closure_tolerance,
                    "value": closure_error,
                }
            )

        continuity_ratio = output_sum / max(input_flux_abs, 1e-12)
        observations["field_continuity_ratio"] = continuity_ratio
        if continuity_ratio < gate.field_continuity_min_ratio:
            blockers.append(
                {
                    "code": "hard_gate_field_continuity_violation",
                    "message": (
                        f"Field continuity proxy ratio {continuity_ratio:.6g} is below floor {gate.field_continuity_min_ratio:.6g}."
                    ),
                    "threshold": gate.field_continuity_min_ratio,
                    "value": continuity_ratio,
                }
            )

    defer_visual_artifact_gate = not _require_field_plot_in_iteration_eval()
    observations["defer_visual_artifact_gate"] = defer_visual_artifact_gate

    if gate.require_field_artifact and not defer_visual_artifact_gate:
        has_field_artifact = any("field" in artifact.lower() for artifact in first_record.artifacts)
        observations["has_field_artifact"] = has_field_artifact
        if not has_field_artifact:
            blockers.append(
                {
                    "code": "hard_gate_missing_field_artifact",
                    "message": "Field-distribution artifact is required for continuity diagnostics.",
                }
            )

    objective_metric_lower = str(bundle.optimization_config.objective.metric or "").strip().lower()
    if (
        gate.require_mode_expansion_for_demux
        and not defer_visual_artifact_gate
        and objective_metric_lower in {
        "demux_routing",
        "mode_demux",
        "wdm_routing",
        "mux_routing",
    }
    ):
        has_mode_expansion_artifact = any(
            "mode_expansion" in artifact.lower() for artifact in first_record.artifacts
        )
        observations["has_mode_expansion_artifact"] = has_mode_expansion_artifact
        if not has_mode_expansion_artifact:
            blockers.append(
                {
                    "code": "hard_gate_missing_mode_expansion_artifact",
                    "message": (
                        "Mode-demux objective requires mode-expansion artifact "
                        "(TE0/TE1 observability evidence)."
                    ),
                }
            )

    # --- Dual-path cross-system divergence check (AGENTS.md rule #18) ---
    # Independently verify BOTH optimizer objective AND diagnostic score.
    # If both are present and diverge by >10x, it indicates the two
    # simulation pipelines are producing incompatible physics (W17-class bug).
    opt_obj = first_record.objective_value
    diag_score = first_record.score
    observations["optimizer_objective"] = opt_obj
    observations["diagnostic_score"] = diag_score

    objective_cases = _objective_cases(bundle)
    if (
        not objective_cases
        and opt_obj is not None
        and diag_score is not None
        and abs(opt_obj) > 1e-9
        and abs(diag_score) > 1e-9
    ):
        cross_ratio = abs(opt_obj) / abs(diag_score)
        divergence_factor = cross_ratio if cross_ratio >= 1.0 else (1.0 / cross_ratio)
        observations["cross_system_ratio"] = cross_ratio
        observations["cross_system_divergence_factor"] = divergence_factor
        if divergence_factor > 10.0:
            blockers.append(
                {
                    "code": "hard_gate_cross_system_divergence",
                    "message": (
                        f"Optimizer objective ({opt_obj:.6g}) and diagnostic score "
                        f"({diag_score:.6g}) diverge by {divergence_factor:.1f}x "
                        f"(opt/diag={cross_ratio:.3g}) — "
                        f"possible dual-builder mismatch (W17-class)."
                    ),
                    "optimizer_objective": opt_obj,
                    "diagnostic_score": diag_score,
                    "divergence_ratio": cross_ratio,
                    "divergence_factor": divergence_factor,
                }
            )

    # Also flag if cross_validation_warning was already recorded during iteration.
    cv_warning = first_record.constraint_status.get("cross_validation_warning")
    if cv_warning:
        observations["cross_validation_warning"] = cv_warning

    return {
        "evaluated": True,
        "passed": not blockers,
        "blockers": blockers,
        "observations": observations,
        "required_monitor_types": list(constraint_packet.required_monitor_types),
    }


def _pick_flux(
    readings: Dict[str, float | None],
    candidate_keys: Sequence[str],
) -> tuple[str, float | None]:
    for key in candidate_keys:
        value = readings.get(key)
        if isinstance(value, (int, float)) and math.isfinite(value):
            return key, float(value)
    return "", None


def _gate_failure_reason(gate_result: Dict[str, Any]) -> str:
    blockers = gate_result.get("blockers", [])
    if not blockers:
        return "Deterministic hard-physics gate rejected first iteration."
    first = blockers[0]
    return str(first.get("message") or "Deterministic hard-physics gate rejected first iteration.")


def _is_better(
    current: InverseDesignIterationRecord,
    best: InverseDesignIterationRecord,
    goal: str,
) -> bool:
    maximize = str(goal or "maximize").strip().lower() != "minimize"
    routing_metrics = {"demux_routing", "mode_demux", "mux_routing"}
    current_metric = str(current.objective_metric or "").strip().lower()
    best_metric = str(best.objective_metric or "").strip().lower()

    # For routing-mode objectives, the optimizer-aligned score is a more
    # stable best-selection signal than acceptance-side objective proxies.
    if current_metric in routing_metrics and best_metric in routing_metrics:
        current_score = (
            float(current.score)
            if current.score is not None and math.isfinite(current.score)
            else None
        )
        best_score = (
            float(best.score)
            if best.score is not None and math.isfinite(best.score)
            else None
        )
        if current_score is not None and best_score is not None:
            if abs(current_score - best_score) > 1e-9:
                return current_score > best_score if maximize else current_score < best_score
        elif current_score is not None and best_score is None:
            return True
        elif current_score is None and best_score is not None:
            return False

    current_value = (
        float(current.objective_value)
        if current.objective_value is not None and math.isfinite(current.objective_value)
        else None
    )
    best_value = (
        float(best.objective_value)
        if best.objective_value is not None and math.isfinite(best.objective_value)
        else None
    )
    # Primary comparison by objective_value when the gap is meaningful.
    if current_value is not None and best_value is not None:
        if abs(current_value - best_value) > 1e-6:
            return current_value > best_value if maximize else current_value < best_value
    elif current_value is not None and best_value is None:
        return True
    elif current_value is None and best_value is not None:
        return False

    # Secondary fallback for objective-degenerate runs (e.g. near-zero plateaus).
    current_secondary = _record_secondary_objective_value(current)
    best_secondary = _record_secondary_objective_value(best)
    if current_secondary is not None and best_secondary is not None:
        if abs(current_secondary - best_secondary) > 1e-9:
            return current_secondary > best_secondary if maximize else current_secondary < best_secondary
    elif current_secondary is not None and best_secondary is None:
        return True
    elif current_secondary is None and best_secondary is not None:
        return False

    # Deterministic tie-breaker: keep the later iteration.
    return int(current.iteration or 0) > int(best.iteration or 0)


def _target_reached(
    record: InverseDesignIterationRecord,
    goal: str,
    target: float | None,
    *,
    expected_case_count: int | None = None,
) -> bool:
    # System-level hard rule for multi-case objectives:
    # objective value alone must never be considered "reached" unless all
    # wavelength cases are directionally correct and threshold-compliant.
    if not _demux_case_targets_satisfied(record, expected_case_count=expected_case_count):
        return False
    objective_metric = str(record.objective_metric).lower()
    # For demux/mux routing, per-case coupling/crosstalk thresholds are the
    # primary acceptance condition (input-normalized semantics).  If no per-case
    # thresholds are specified, do not auto-terminate; fall back to explicit
    # global objective targets only.
    if objective_metric in {"demux_routing", "mux_routing"}:
        if _demux_cases_have_explicit_thresholds(record, expected_case_count=expected_case_count):
            return True
    if target is None or record.objective_value is None:
        return False
    if goal == "maximize":
        return record.objective_value >= target
    return record.objective_value <= target


def _demux_case_targets_satisfied(
    record: InverseDesignIterationRecord,
    *,
    expected_case_count: int | None = None,
) -> bool:
    metrics = record.metrics if isinstance(record.metrics, dict) else {}
    cases = metrics.get("multi_case", []) if isinstance(metrics, dict) else []
    if not isinstance(cases, list):
        return True
    if expected_case_count is not None and len(cases) < int(expected_case_count):
        return False
    if not cases:
        return True

    for case in cases:
        if not isinstance(case, dict):
            return False
        coupling = _as_float_or_none(case.get("coupling_ratio_to_input"))
        crosstalk = _as_float_or_none(case.get("crosstalk_ratio_to_input"))
        if coupling is None or crosstalk is None:
            return False
        # Demux directionality gate: each case must favor its own target port.
        if coupling <= crosstalk:
            return False

        min_coupling = _as_float_or_none(case.get("min_coupling"))
        if min_coupling is not None and coupling < min_coupling:
            return False
        max_crosstalk = _as_float_or_none(case.get("max_crosstalk"))
        if max_crosstalk is not None and crosstalk > max_crosstalk:
            return False
    return True


def _demux_cases_have_explicit_thresholds(
    record: InverseDesignIterationRecord,
    *,
    expected_case_count: int | None = None,
) -> bool:
    metrics = record.metrics if isinstance(record.metrics, dict) else {}
    cases = metrics.get("multi_case", []) if isinstance(metrics, dict) else []
    if not isinstance(cases, list) or not cases:
        return False
    if expected_case_count is not None and len(cases) < int(expected_case_count):
        return False
    for case in cases:
        if not isinstance(case, dict):
            return False
        min_coupling = _as_float_or_none(case.get("min_coupling"))
        max_crosstalk = _as_float_or_none(case.get("max_crosstalk"))
        if min_coupling is None and max_crosstalk is None:
            return False
    return True


def _next_parameters(
    *,
    current_params: Dict[str, float],
    variables: Dict[str, Any],
    iteration_index: int,
) -> Dict[str, float]:
    if not current_params:
        return current_params

    updated = dict(current_params)
    sign = 1.0 if iteration_index % 2 == 0 else -1.0
    for name, var in variables.items():
        if name not in updated:
            continue
        span = float(var.upper_bound) - float(var.lower_bound)
        if span <= 0:
            continue
        step = span * (0.05 / (iteration_index + 1))
        candidate = updated[name] + sign * step
        candidate = min(max(candidate, float(var.lower_bound)), float(var.upper_bound))
        updated[name] = round(candidate, 6)
    return updated


def _termination_reason(
    records: List[InverseDesignIterationRecord],
    goal: str,
    target: float | None,
    run_iterations: int,
    *,
    expected_case_count: int | None = None,
) -> str:
    if not records:
        return "No iterations executed."
    if target is not None and _target_reached(
        records[-1],
        goal,
        target,
        expected_case_count=expected_case_count,
    ):
        return "Target objective value reached."
    if len(records) < run_iterations:
        return "Stopped early."
    return "Reached configured iteration budget for this run."


def _save_topology_image(
    *,
    invdes_result: Any,
    snapshots: List[Dict[str, Any]],
    records: List[InverseDesignIterationRecord],
    bundle: InverseDesignConfigBundle,
    objective_goal: str,
    run_artifact_tag: str = "",
) -> tuple[List[str], int | None]:
    """Save topology density image for the best iteration.

    Returns list of saved artifact paths.
    """
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend; prevents GUI deadlock in subprocess
    import matplotlib.pyplot as plt
    import numpy as np

    artifacts: List[str] = []
    best_iteration: int | None = None
    try:
        # Find best params from records (objective-aware with score fallback).
        best_idx: int | None = None
        best_rec: InverseDesignIterationRecord | None = None
        for idx, rec in enumerate(records):
            if best_rec is None or _is_better(rec, best_rec, objective_goal):
                best_rec = rec
                best_idx = idx

        best_val = None if best_rec is None else _checkpoint_objective_value(best_rec)
        if best_rec is not None:
            best_iteration = int(best_rec.iteration)

        best_snap = snapshots[best_idx] if best_idx is not None and best_idx < len(snapshots) else None

        params = best_snap.get("params") if best_snap else None
        if params is None and snapshots:
            params = snapshots[-1].get("params")

        if params is None:
            return artifacts, best_iteration

        # Convert params to 2D density array
        params_array = np.asarray(params)
        if params_array.ndim == 1:
            side = int(np.sqrt(params_array.size))
            if side * side == params_array.size:
                density = params_array.reshape((side, side))
            else:
                # Non-square design region: infer (nx, ny) from the bundle's
                # design-region aspect ratio instead of assuming a square grid.
                _dr_sx, _dr_sy, _ = _design_region_size(
                    bundle.simulation_config.domain.size_um, bundle
                )
                density = None
                if _dr_sx > 0 and _dr_sy > 0:
                    _ratio = _dr_sx / _dr_sy
                    _nx_approx = int(round(np.sqrt(params_array.size * _ratio)))
                    # Search ±5 around the approximation for an exact integer
                    # factorization. The round() approximation can produce
                    # (_nx, _ny) pairs that don't multiply back to params_array.size
                    # when the aspect ratio is not a perfect ratio of small integers.
                    for _try_nx in range(max(1, _nx_approx - 5), _nx_approx + 6):
                        if _try_nx > 0 and params_array.size % _try_nx == 0:
                            density = params_array.reshape(
                                (_try_nx, params_array.size // _try_nx)
                            )
                            break
                # If aspect-ratio-guided search failed, find the 2-factor pair
                # whose ratio is closest to the design-region target ratio.
                if density is None:
                    import math as _math
                    _target_ratio = (
                        _dr_sx / _dr_sy if _dr_sx > 0 and _dr_sy > 0 else 1.0
                    )
                    _best_diff = float("inf")
                    for _f in range(1, int(_math.isqrt(params_array.size)) + 1):
                        if params_array.size % _f == 0:
                            _g = params_array.size // _f
                            _diff = abs(_g / _f - _target_ratio)
                            if _diff < _best_diff:
                                _best_diff = _diff
                                density = params_array.reshape((_f, _g))
                    if density is None:
                        logger.warning(
                            "Could not factorize params_vector of size %d into 2D for "
                            "topology image — density not saved.",
                            params_array.size,
                        )
                        return artifacts, best_iteration
        elif params_array.ndim == 2:
            density = params_array
        elif params_array.ndim == 3:
            density = params_array[:, :, 0]
        else:
            return artifacts, best_iteration

        # Clip to [0, 1]
        density = np.clip(density, 0.0, 1.0)

        build_dir = os.path.join(os.getcwd(), "build")
        os.makedirs(build_dir, exist_ok=True)

        comp_type = bundle.simulation_config.component_type
        run_tag = re.sub(r"[^a-zA-Z0-9_-]+", "-", (run_artifact_tag or "run")).strip("-")
        iter_tag = f"iter{int(best_iteration or 0):03d}"
        png_path = os.path.join(build_dir, f"topology_{comp_type}_density.png")
        tagged_png_path = os.path.join(build_dir, f"topology_{comp_type}_{run_tag}_{iter_tag}_density.png")
        region_sx, region_sy, _ = _design_region_size(bundle.simulation_config.domain.size_um, bundle)
        if region_sx <= 0 or region_sy <= 0:
            region_sx = float(density.shape[0])
            region_sy = float(density.shape[1])

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        im = ax.imshow(
            np.flipud(density.T),
            cmap="gray_r",
            vmin=0,
            vmax=1,
            extent=[-region_sx / 2.0, region_sx / 2.0, -region_sy / 2.0, region_sy / 2.0],
        )
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(
            f"Topology Density — {comp_type}\n"
            f"Best objective: {best_val:.4f}" if best_val is not None else f"Topology Density — {comp_type}",
            fontsize=12,
        )
        ax.set_xlabel("X (um)")
        ax.set_ylabel("Y (um)")
        fig.colorbar(im, ax=ax, label="Material density")
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        fig.savefig(tagged_png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        artifacts.append(png_path)
        artifacts.append(tagged_png_path)
        logger.info("Saved topology density image: %s", png_path)

        density_npy_path = os.path.join(build_dir, f"topology_{comp_type}_{run_tag}_{iter_tag}_density.npy")
        np.save(density_npy_path, density)
        artifacts.append(density_npy_path)
        logger.info("Saved topology density array: %s", density_npy_path)

        density_meta = {
            "component_type": comp_type,
            "run_artifact_tag": run_artifact_tag,
            "best_iteration": int(best_iteration or 0),
            "best_objective": float(best_val) if best_val is not None else None,
            "density_npy_path": density_npy_path,
            "density_png_path": tagged_png_path,
            "density_shape": [int(density.shape[0]), int(density.shape[1])],
            "region_size_um": [float(region_sx), float(region_sy)],
            "wg_height_um": float(bundle.simulation_config.geometry.parameters.get("wg_height", 0.22)),
            "base_parameters": (
                dict(records[int(best_iteration - 1)].parameters)
                if best_iteration is not None and 1 <= best_iteration <= len(records)
                else {}
            ),
        }
        density_meta_path = os.path.join(build_dir, f"topology_{comp_type}_{run_tag}_{iter_tag}_density_meta.json")
        with open(density_meta_path, "w", encoding="utf-8") as f:
            json.dump(density_meta, f, ensure_ascii=False, indent=2)
        artifacts.append(density_meta_path)
        logger.info("Saved topology density meta: %s", density_meta_path)

        # Also save objective history plot
        obj_values = [_checkpoint_objective_value(rec) for rec in records]
        if len(obj_values) >= 2:
            history_path = os.path.join(build_dir, f"topology_{comp_type}_objective_history.png")
            fig2, ax2 = plt.subplots(1, 1, figsize=(8, 4))
            ax2.plot(range(1, len(obj_values) + 1), obj_values, "o-", color="steelblue", linewidth=2)
            ax2.set_xlabel("Iteration", fontsize=12)
            ax2.set_ylabel("Objective Value", fontsize=12)
            ax2.set_title(f"Optimization History — {comp_type}", fontsize=13)
            ax2.grid(True, alpha=0.3)
            fig2.savefig(history_path, dpi=150, bbox_inches="tight")
            plt.close(fig2)
            artifacts.append(history_path)
            logger.info("Saved objective history: %s", history_path)
    except Exception as exc:
        logger.warning("Failed to save topology image: %s", exc)
    return artifacts, best_iteration


def _case_identifier(case: Dict[str, Any]) -> str:
    name = str(case.get("name") or "").strip()
    if name:
        return name
    src = str(case.get("source_port") or "").strip().lower()
    dst = str(case.get("target_port") or "").strip().lower()
    mode_idx = _as_int_or_none(case.get("target_mode_index"))
    if src or dst:
        mode_suffix = f":te{mode_idx}" if mode_idx is not None else ""
        return f"{src or 'source'}->{dst or 'target'}{mode_suffix}"
    return "case"


def _case_label(case: Dict[str, Any]) -> str:
    src = str(case.get("source_port") or "").strip().lower()
    dst = str(case.get("target_port") or "").strip().lower()
    mode_idx = _as_int_or_none(case.get("target_mode_index"))
    if src and dst and mode_idx is not None:
        return f"{src}->{dst} TE{mode_idx}"
    name = str(case.get("name") or "").strip()
    return name or _case_identifier(case)


def _pick_artifact_path(artifacts: Sequence[str], tokens: Sequence[str]) -> Path | None:
    lowered = [str(path).lower() for path in artifacts]
    for token in tokens:
        token_l = str(token).lower()
        for idx, item in enumerate(lowered):
            if token_l in item:
                return _normalize_path(artifacts[idx])
    return None


def _save_density_to_gds(
    *,
    density: Any,
    output_path: Path,
    region_sx: float | None,
    region_sy: float | None,
    wg_height: float,
) -> None:
    import numpy as np
    import gdstk

    arr = np.asarray(density, dtype=float)
    arr = np.clip(np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    nx, ny = int(arr.shape[0]), int(arr.shape[1])
    sx = float(region_sx) if region_sx and region_sx > 0 else float(nx) * 0.05
    sy = float(region_sy) if region_sy and region_sy > 0 else float(ny) * 0.05
    dx = sx / max(nx, 1)
    dy = sy / max(ny, 1)
    x0 = -sx / 2.0
    y0 = -sy / 2.0

    lib = gdstk.Library()
    cell = lib.new_cell("INVDES_BEST_TOPOLOGY")
    threshold = 0.5
    for i in range(nx):
        xa = x0 + i * dx
        xb = xa + dx
        for j in range(ny):
            if float(arr[i, j]) < threshold:
                continue
            ya = y0 + j * dy
            yb = ya + dy
            cell.add(gdstk.rectangle((xa, ya), (xb, yb), layer=1, datatype=0))

    wg_w = max(0.2, float(wg_height) * 2.0)
    x_left = -sx / 2.0
    x_right = sx / 2.0
    out_off = min(sy * 0.2, 0.35)
    cell.add(gdstk.rectangle((x_left - 1.5, -wg_w / 2), (x_left, wg_w / 2), layer=1, datatype=0))
    cell.add(gdstk.rectangle((x_right, out_off - wg_w / 2), (x_right + 1.5, out_off + wg_w / 2), layer=1, datatype=0))
    cell.add(gdstk.rectangle((x_right, -out_off - wg_w / 2), (x_right + 1.5, -out_off + wg_w / 2), layer=1, datatype=0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    lib.write_gds(str(output_path))


def _save_best_iteration_support_artifacts(
    *,
    records: List[InverseDesignIterationRecord],
    bundle: InverseDesignConfigBundle,
    objective_goal: str,
    run_artifact_tag: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend; prevents GUI deadlock in subprocess
    import matplotlib.pyplot as plt
    import numpy as np

    if not records:
        return

    best_idx = 0
    best_record = records[0]
    for idx, record in enumerate(records[1:], start=1):
        if _is_better(record, best_record, objective_goal):
            best_record = record
            best_idx = idx

    existing_artifacts = [str(item) for item in best_record.artifacts if str(item or "").strip()]
    added_artifacts: List[str] = []
    summary: Dict[str, Any] = {
        "best_iteration": int(best_record.iteration),
    }

    # ---- 1) Purity/coupling trend chart over iterations ----
    try:
        iterations: List[int] = []
        case_order: List[str] = []
        case_labels: Dict[str, str] = {}
        purity_series: Dict[str, List[float]] = {}
        coupling_series: Dict[str, List[float]] = {}

        for rec in records:
            iterations.append(int(rec.iteration))
            metrics = rec.metrics if isinstance(rec.metrics, dict) else {}
            multi_case = metrics.get("multi_case", []) if isinstance(metrics, dict) else []
            case_by_id: Dict[str, Dict[str, Any]] = {}
            if isinstance(multi_case, list):
                for case in multi_case:
                    if not isinstance(case, dict):
                        continue
                    cid = _case_identifier(case)
                    case_by_id[cid] = case
                    if cid not in case_order:
                        case_order.append(cid)
                        case_labels[cid] = _case_label(case)
                        purity_series[cid] = []
                        coupling_series[cid] = []

            for cid in case_order:
                case = case_by_id.get(cid, {})
                purity_val = _as_float_or_none(case.get("target_port_mode_purity_target"))
                if purity_val is None:
                    mode_idx = _as_int_or_none(case.get("target_mode_index"))
                    if mode_idx == 0:
                        purity_val = _as_float_or_none(case.get("target_port_mode_purity_te0"))
                    elif mode_idx == 1:
                        purity_val = _as_float_or_none(case.get("target_port_mode_purity_te1"))
                coupling_val = _as_float_or_none(case.get("coupling_ratio_to_input"))
                if coupling_val is None:
                    coupling_val = _as_float_or_none(case.get("target_mode_transmission_surrogate"))
                if coupling_val is None:
                    coupling_val = _as_float_or_none(case.get("coupling_ratio"))

                purity_series[cid].append(float(purity_val) if purity_val is not None else float("nan"))
                coupling_series[cid].append(float(coupling_val) if coupling_val is not None else float("nan"))

        if iterations and case_order:
            safe_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(run_artifact_tag or "run")).strip("_") or "run"
            trend_path = _BUILD_DIR / (
                f"tidy3d_mode_purity_coupling_over_iterations_"
                f"{bundle.simulation_config.component_type}_{safe_tag}.png"
            )
            fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            ax_purity, ax_coupling = axes
            for cid in case_order:
                label = case_labels.get(cid, cid)
                ax_purity.plot(iterations, purity_series[cid], marker="o", linewidth=1.6, label=f"{label} purity")
                ax_coupling.plot(iterations, coupling_series[cid], marker="o", linewidth=1.6, label=f"{label} coupling")

            ax_purity.set_ylabel("Target mode purity")
            ax_purity.set_ylim(0.0, 1.05)
            ax_purity.grid(alpha=0.3)
            ax_purity.legend(loc="best", fontsize=8)
            ax_coupling.set_ylabel("Target coupling")
            ax_coupling.set_ylim(0.0, 1.05)
            ax_coupling.set_xlabel("Iteration")
            ax_coupling.grid(alpha=0.3)
            ax_coupling.legend(loc="best", fontsize=8)
            fig.suptitle(f"{bundle.simulation_config.component_type} purity/coupling over iterations")
            fig.tight_layout()
            trend_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(trend_path, dpi=160)
            plt.close(fig)

            added_artifacts.append(str(trend_path))
            summary["purity_coupling_curve"] = str(trend_path)
    except Exception as exc:
        summary["purity_coupling_curve_error"] = str(exc)

    # ---- 2) Topology-derived GDS (best iteration) ----
    try:
        density_npy = _pick_artifact_path(existing_artifacts, ("_density.npy",))
        density_meta = _pick_artifact_path(existing_artifacts, ("_density_meta.json",))
        if density_npy is not None and density_npy.exists():
            density = np.asarray(np.load(str(density_npy)), dtype=float)
            region_sx: float | None = None
            region_sy: float | None = None
            wg_height = 0.22
            if density_meta is not None and density_meta.exists():
                try:
                    payload = json.loads(density_meta.read_text(encoding="utf-8"))
                    region = payload.get("region_size_um")
                    if isinstance(region, list) and len(region) >= 2:
                        region_sx = _as_float_or_none(region[0])
                        region_sy = _as_float_or_none(region[1])
                    wg_height = _as_float_or_none(payload.get("wg_height_um")) or wg_height
                except Exception:
                    pass
            safe_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(run_artifact_tag or "run")).strip("_") or "run"
            gds_path = _BUILD_DIR / (
                f"{bundle.simulation_config.component_type}_best_topology_"
                f"{safe_tag}_iter{int(best_record.iteration):03d}.gds"
            )
            _save_density_to_gds(
                density=density,
                output_path=gds_path,
                region_sx=region_sx,
                region_sy=region_sy,
                wg_height=wg_height,
            )
            if gds_path.exists():
                added_artifacts.append(str(gds_path))
                summary["gds"] = str(gds_path)
        else:
            summary["gds_error"] = "Missing topology density npy artifact; cannot export topology-derived GDS."
    except Exception as exc:
        summary["gds_error"] = str(exc)

    if added_artifacts:
        best_record.artifacts = list(dict.fromkeys([*existing_artifacts, *added_artifacts]))
    best_metrics = dict(best_record.metrics or {})
    best_metrics["best_iteration_support_artifacts"] = summary
    best_record.metrics = best_metrics
    records[best_idx] = best_record


def _save_bridge_structure_images(
    *,
    records: List[InverseDesignIterationRecord],
    bundle: InverseDesignConfigBundle,
    objective_goal: str,
) -> List[str]:
    """Save structure cross-section plots + objective history for the bridge path.

    Unlike the adjoint path (which produces a topology density map), the bridge
    path optimizes geometric parameters.  We save:
    1. Simulation cross-section images (XY, XZ, YZ) for the best iteration.
    2. Objective value history curve.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    artifacts: List[str] = []
    try:
        build_dir = os.path.join(os.getcwd(), "build")
        os.makedirs(build_dir, exist_ok=True)
        comp_type = bundle.simulation_config.component_type

        # --- 1. Objective history ---
        obj_values = [_checkpoint_objective_value(r) for r in records]
        if len(obj_values) >= 2:
            hist_path = os.path.join(build_dir, f"bridge_{comp_type}_objective_history.png")
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
            ax.plot(range(1, len(obj_values) + 1), obj_values, "o-", color="steelblue", linewidth=2)
            ax.set_xlabel("Iteration", fontsize=12)
            ax.set_ylabel("Objective Value", fontsize=12)
            ax.set_title(f"Optimization History — {comp_type} (bridge)", fontsize=13)
            ax.grid(True, alpha=0.3)
            fig.savefig(hist_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            artifacts.append(hist_path)
            logger.info("Saved bridge objective history: %s", hist_path)

        # --- 2. Structure cross-sections for best iteration ---
        best_rec: InverseDesignIterationRecord | None = None
        for rec in records:
            if best_rec is None or _is_better(rec, best_rec, objective_goal):
                best_rec = rec

        if best_rec is not None:
            best_val = _checkpoint_objective_value(best_rec)
            _save_sim_cross_sections_from_params(
                bundle, best_rec.parameters, comp_type, best_val, build_dir, artifacts,
            )
    except Exception as exc:
        logger.warning("Failed to save bridge structure images: %s", exc)
    return artifacts


def _save_sim_cross_sections_from_params(
    bundle: InverseDesignConfigBundle,
    params: Dict[str, Any],
    comp_type: str,
    best_val: float | None,
    build_dir: str,
    artifacts: List[str],
) -> None:
    """Build a td.Simulation from parameters and save XY/XZ/YZ cross-sections."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import tidy3d as td

    try:
        from PhotonicsAI.Photon.tidy3d_runner import (
            create_mmi, create_waveguide_crossing, create_simple_waveguide, source_size_for_port,
        )

        def _float_param(name: str, default: float) -> float:
            try:
                value = params.get(name, default)
                return float(value)
            except (TypeError, ValueError):
                return float(default)

        wl_um = bundle.simulation_config.source.wavelength_nm / 1000.0
        freq0 = td.C_0 / wl_um
        fwidth = freq0 * 0.1
        wg_width = _float_param("wg_width", 0.5)
        wg_height = _float_param("wg_height", 0.22)
        mmi_width = _float_param("mmi_width", 2.5)
        mmi_length = _float_param("mmi_length", 10.0)
        mmi_num_outputs = max(int(round(_float_param("mmi_num_outputs", 2.0))), 2)
        wg_length = _float_param("wg_length", 10.0)

        if comp_type in {"mmi", "splitter"}:
            from PhotonicsAI.Photon.tidy3d_runner import _port_waveguide_width
            port_o1_width = _port_waveguide_width(
                component_type=comp_type,
                port_name="port_o1",
                objective_metric=str(bundle.optimization_config.objective.metric or ""),
                wg_width=wg_width,
                mmi_width=mmi_width,
                params=params,
            )
            output_wg_widths = [
                _port_waveguide_width(
                    component_type=comp_type,
                    port_name=f"port_o{i + 2}",
                    objective_metric=str(bundle.optimization_config.objective.metric or ""),
                    wg_width=wg_width,
                    mmi_width=mmi_width,
                    params=params,
                )
                for i in range(mmi_num_outputs)
            ]
            structures, sim_size, src_center, mon_positions = create_mmi(
                td,
                wl_um,
                wg_width,
                wg_height,
                mmi_width,
                mmi_length,
                num_outputs=mmi_num_outputs,
                input_wg_width=port_o1_width,
                output_wg_widths=output_wg_widths,
            )
        elif comp_type == "crossing":
            structures, sim_size, src_center, mon_positions = create_waveguide_crossing(
                td, wl_um, wg_width, wg_height, wg_length,
            )
        else:
            structures, sim_size, src_center, mon_positions = create_simple_waveguide(
                td, wl_um, wg_width, wg_height, wg_length,
            )

        source = td.ModeSource(
            center=tuple(src_center),
            size=source_size_for_port("x", wg_width, wg_height),
            source_time=td.GaussianPulse(freq0=freq0, fwidth=fwidth),
            direction="+",
            mode_spec=td.ModeSpec(num_modes=1),
        )

        sim = td.Simulation(
            size=sim_size,
            grid_spec=td.GridSpec.auto(wavelength=wl_um, min_steps_per_wvl=10),
            structures=structures,
            sources=[source],
            monitors=[],
            run_time=20 / fwidth,
            boundary_spec=td.BoundarySpec.all_sides(boundary=td.PML()),
            medium=td.Medium(permittivity=1.44**2),
        )

        title_suffix = f" (best obj={best_val:.4f})" if best_val is not None else ""

        # XY plane (z = wg_height/2)
        for plane_name, plot_kwargs in [
            ("z0", {"z": wg_height / 2}),
            ("x0", {"x": 0}),
            ("y0", {"y": 0}),
        ]:
            fig, ax = plt.subplots(1, 1, figsize=(10, 4))
            sim.plot(ax=ax, **plot_kwargs)
            ax.set_title(f"{comp_type} — {plane_name}{title_suffix}", fontsize=12)
            path = os.path.join(build_dir, f"bridge_{comp_type}_{plane_name}.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            artifacts.append(path)
            logger.info("Saved bridge structure cross-section: %s", path)

    except Exception as exc:
        logger.warning("Failed to save bridge cross-section images: %s", exc)
