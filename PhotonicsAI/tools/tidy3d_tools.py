"""Tidy3D tool wrappers for simulation and doc lookup via MCP."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence

from PhotonicsAI.Photon import tidy3d_runner
from PhotonicsAI.config import PATH
from PhotonicsAI.core.tooling import Tool
from PhotonicsAI.tools.mcp_client import (
    search_docs_sync,
    fetch_doc_sync,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Parameter-name translation: gdsfactory template → tidy3d_runner kwargs
# ---------------------------------------------------------------------------

# Names that tidy3d_runner.run_tidy3d_simulation already accepts directly
_RUNNER_KWARGS = {
    "wg_width", "wg_height", "wg_length",
    "ring_radius", "gap",
    "mmi_width", "mmi_length", "mmi_num_outputs",
    "port_o1_wg_width", "side_port_wg_width",
    "arm_length", "arm_separation",
    "coupler_length",
    "grating_period", "num_periods",
    "rotation_length", "swg_period",
    "target_ports", "target_mode_indices",
    "source_port", "source_mode_index", "source_direction",
}

# Generic aliases shared across component types
_COMMON_ALIASES: Dict[str, str] = {
    "width": "wg_width",
    "height": "wg_height",
    "radius": "ring_radius",
    "coupling_length": "coupler_length",
    "n_periods": "num_periods",
}

# Per-component-type overrides (checked before _COMMON_ALIASES)
_COMPONENT_ALIASES: Dict[str, Dict[str, str]] = {
    "crossing":              {"length": "wg_length"},
    "waveguide":             {"length": "wg_length"},
    "ring_resonator":        {},
    "mmi":                   {"length": "mmi_length", "width_mmi": "mmi_width"},
    "splitter":              {"length": "mmi_length", "width_mmi": "mmi_width"},
    "mzi":                   {"length": "arm_length", "gap": "arm_separation"},
    "directional_coupler":   {"length": "coupler_length"},
    "grating_coupler":       {"period": "grating_period"},
    "polarization_rotator":  {"length": "rotation_length", "period": "swg_period"},
    "y_branch":              {"length": "arm_length", "gap": "arm_separation"},
}


def _translate_params(
    component_type: str,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """Map gdsfactory template parameter names to tidy3d_runner kwargs.

    Priority: component-specific override > already-valid runner kwarg > common alias.
    Unmapped keys (e.g. ``angle``) are silently dropped.
    """
    overrides = _COMPONENT_ALIASES.get(component_type, {})
    out: Dict[str, Any] = {}
    for key, value in params.items():
        if key in overrides:
            mapped = overrides[key]
            out[mapped] = value
            logger.debug("param translate [%s]: %s -> %s = %s", component_type, key, mapped, value)
        elif key in _RUNNER_KWARGS:
            out[key] = value
        elif key in _COMMON_ALIASES:
            mapped = _COMMON_ALIASES[key]
            out[mapped] = value
            logger.debug("param translate [common]: %s -> %s = %s", key, mapped, value)
        else:
            logger.debug("param translate: dropping unmapped key %s", key)
    return out


_DEMUX_METRICS = {"demux_routing", "mode_demux", "mux_routing"}


def _pick_artifact(artifacts: List[str], tokens: Sequence[str]) -> str:
    lowered = [path.lower() for path in artifacts]
    for token in tokens:
        t = token.lower()
        for idx, item in enumerate(lowered):
            if t in item:
                return artifacts[idx]
    return ""


def select_visualization_artifacts(
    artifacts: List[str],
    *,
    component_type: str,
    objective_metric: str | None = None,
    ensure_field: bool = True,
) -> Dict[str, Any]:
    normalized = sorted({str(path) for path in artifacts if str(path)})
    metric = str(objective_metric or "").strip().lower()
    is_demux_like = metric in _DEMUX_METRICS

    field = _pick_artifact(normalized, ("tidy3d_field_",))
    field_port_o1 = _pick_artifact(normalized, ("tidy3d_field_port_o1",))
    field_port_o2 = _pick_artifact(normalized, ("tidy3d_field_port_o2",))
    field_port_o3 = _pick_artifact(normalized, ("tidy3d_field_port_o3",))
    field_profile_y_phase_aligned = _pick_artifact(normalized, ("field_profile_y_phase_aligned",))
    mode_expansion = _pick_artifact(normalized, ("mode_expansion",))
    mode_expansion_port_o1 = _pick_artifact(normalized, ("mode_expansion_port_o1",))
    mode_components = _pick_artifact(normalized, ("mode_components",))
    mode_profile_port_o1 = _pick_artifact(normalized, ("mode_profile_port_o1",))
    mode_profile_port_o2 = _pick_artifact(normalized, ("mode_profile_port_o2",))
    mode_profile_port_o3 = _pick_artifact(normalized, ("mode_profile_port_o3",))
    mode_purity_curve = _pick_artifact(normalized, ("mode_purity_over_iterations",))
    mode_purity_best = _pick_artifact(normalized, ("mode_purity_best",))
    flux_output_only = _pick_artifact(normalized, ("flux_output_only",))
    flux_ratio_to_input = _pick_artifact(normalized, ("flux_ratio_to_input",))
    flux = _pick_artifact(normalized, ("tidy3d_flux_",))
    structure = _pick_artifact(normalized, ("tidy3d_sim_x0", "tidy3d_sim_z0", "tidy3d_sim_y0", "tidy3d_sim_"))

    secondary_priority = (
        [
            mode_expansion,
            mode_expansion_port_o1,
            mode_components,
            mode_profile_port_o1,
            mode_profile_port_o2,
            mode_profile_port_o3,
            flux_output_only,
            flux_ratio_to_input,
            flux,
        ]
        if is_demux_like
        else [flux, flux_output_only, flux_ratio_to_input, mode_expansion]
    )
    secondary = next((item for item in secondary_priority if item), "")

    recommended = field or secondary or structure
    warnings: List[str] = []
    if ensure_field and not field:
        warnings.append(
            f"Missing mandatory field artifact for `{component_type}`."
        )
        if not recommended:
            recommended = secondary or structure

    return {
        "field": field,
        "field_port_o1": field_port_o1,
        "field_port_o2": field_port_o2,
        "field_port_o3": field_port_o3,
        "field_profile_y_phase_aligned": field_profile_y_phase_aligned,
        "secondary": secondary,
        "mode_expansion": mode_expansion,
        "mode_expansion_port_o1": mode_expansion_port_o1,
        "mode_components": mode_components,
        "mode_profile_port_o1": mode_profile_port_o1,
        "mode_profile_port_o2": mode_profile_port_o2,
        "mode_profile_port_o3": mode_profile_port_o3,
        "mode_purity_curve": mode_purity_curve,
        "mode_purity_best": mode_purity_best,
        "flux_output_only": flux_output_only,
        "flux_ratio_to_input": flux_ratio_to_input,
        "flux": flux,
        "structure": structure,
        "recommended_image": recommended,
        "warnings": warnings,
        "all_artifacts": normalized,
        "objective_metric": metric,
    }


def run_tidy3d_simulation(
    component_type: str,
    parameters: Dict[str, Any] | None = None,
    wavelength_nm: float = 1550.0,
    run_time_s: float | None = None,
    min_steps_per_wvl: int | None = None,
    artifact_tag: str | None = None,
    topology_density_path: str | None = None,
    topology_density_meta_path: str | None = None,
    objective_metric: str | None = None,
    target_ports: List[str] | None = None,
    target_mode_indices: List[int] | None = None,
    source_port: str | None = None,
    source_mode_index: int | None = None,
    source_direction: str | None = None,
    require_field_plot: bool = True,
    skip_field_monitors: bool = False,
    skip_cloud: bool = False,
) -> Dict[str, Any]:
    params = _translate_params(component_type, parameters or {})
    if run_time_s is not None:
        params["run_time_s"] = float(run_time_s)
    if min_steps_per_wvl is not None:
        params["min_steps_per_wvl"] = int(min_steps_per_wvl)
    if artifact_tag:
        params["artifact_tag"] = str(artifact_tag)
    if topology_density_path:
        params["topology_density_path"] = str(topology_density_path)
    if topology_density_meta_path:
        params["topology_density_meta_path"] = str(topology_density_meta_path)
    if objective_metric:
        params["objective_metric"] = str(objective_metric)
    if target_ports:
        params["target_ports"] = [str(item) for item in target_ports if str(item).strip()]
    if target_mode_indices:
        safe_indices = []
        for item in target_mode_indices:
            try:
                safe_indices.append(max(int(item), 0))
            except (TypeError, ValueError):
                continue
        if safe_indices:
            params["target_mode_indices"] = safe_indices
    if source_port:
        params["source_port"] = str(source_port).strip().lower()
    if source_mode_index is not None:
        try:
            params["source_mode_index"] = max(int(source_mode_index), 0)
        except (TypeError, ValueError):
            pass
    if source_direction in {"+", "-"}:
        params["source_direction"] = source_direction
    # Propagate skip_field_monitors to the underlying runner so rerender sims
    # omit FieldMonitors (matching the adjoint optimizer forward cost).
    if skip_field_monitors:
        params["skip_field_monitors"] = True
    if skip_cloud:
        params["skip_cloud"] = True
    try:
        metrics = tidy3d_runner.run_tidy3d_simulation(
            component_type=component_type,
            wavelength_nm=wavelength_nm,
            **params,
        )
    except Exception as exc:
        return {"ok": False, "data": {}, "error": f"Tidy3D simulation failed: {exc}"}

    log_path = PATH.build / "tidy3d.log"
    artifacts: List[str] = []
    if not isinstance(metrics, dict):
        return {
            "ok": False,
            "data": {
                "component_type": component_type,
                "artifacts": artifacts,
                "log_path": str(log_path) if log_path.exists() else "",
                "metrics": {},
                "score": None,
            },
            "error": "Tidy3D simulation returned invalid metrics payload.",
        }

    # Prefer explicit artifact list returned by the runner to avoid mixed-run
    # filename globbing. This is critical for multi-case acceptance evidence.
    artifacts = [str(path) for path in metrics.get("_artifacts", []) if isinstance(path, str)]
    if not artifacts:
        artifacts = [str(artifact) for artifact in PATH.build.glob(f"tidy3d_*_{component_type}.png")]
        if artifact_tag:
            artifacts.extend(
                str(artifact)
                for artifact in PATH.build.glob(f"tidy3d_*_{component_type}_*{artifact_tag}*.png")
            )

    if artifact_tag:
        tag = str(artifact_tag)
        tagged_only = [a for a in artifacts if tag in Path(a).stem]
        if tagged_only:
            artifacts = tagged_only
    artifacts = sorted(set(artifacts))
    selected = select_visualization_artifacts(
        artifacts,
        component_type=component_type,
        objective_metric=objective_metric,
        ensure_field=require_field_plot,
    )

    if metrics.get("_error"):
        return {
            "ok": False,
            "data": {
                "component_type": component_type,
                "artifacts": artifacts,
                "selected_artifacts": selected,
                "log_path": str(log_path) if log_path.exists() else "",
                "metrics": metrics,
                "score": None,
            },
            "error": f"Tidy3D simulation failed before metrics extraction: {metrics.get('_error')}",
        }

    if require_field_plot and not selected.get("field"):
        return {
            "ok": False,
            "data": {
                "component_type": component_type,
                "artifacts": artifacts,
                "selected_artifacts": selected,
                "log_path": str(log_path) if log_path.exists() else "",
                "metrics": metrics if isinstance(metrics, dict) else {},
                "score": None,
            },
            "error": "Tidy3D simulation missing mandatory field artifact.",
        }

    flux_map = metrics.get("flux")
    score = metrics.get("score")
    flux_ok = isinstance(flux_map, dict) and len(flux_map) > 0
    score_ok = isinstance(score, (int, float))
    if not flux_ok or not score_ok:
        return {
            "ok": False,
            "data": {
                "component_type": component_type,
                "artifacts": artifacts,
                "selected_artifacts": selected,
                "log_path": str(log_path) if log_path.exists() else "",
                "metrics": metrics,
                "score": score if score_ok else None,
            },
            "error": (
                "Tidy3D diagnostic metrics are incomplete "
                f"(flux_ok={flux_ok}, score_ok={score_ok})."
            ),
        }

    return {
        "ok": True,
        "data": {
            "component_type": component_type,
            "artifacts": artifacts,
            "selected_artifacts": selected,
            "log_path": str(log_path) if log_path.exists() else "",
            "metrics": metrics,
            "score": score,
        },
        "error": None,
    }


def search_tidy3d_docs(
    query: str,
    max_results: int = 5,
    timeout_s: float | None = None,
) -> Dict[str, Any]:
    """Search Tidy3D / Flexcompute docs via tidy3d-mcp (``search_flexcompute_docs``)."""
    return search_docs_sync(query, max_results=max_results, timeout_s=timeout_s)


def fetch_tidy3d_doc(url: str, timeout_s: float | None = None) -> Dict[str, Any]:
    """Fetch a single Tidy3D doc page via tidy3d-mcp (``fetch_flexcompute_doc``)."""
    if not url.startswith("http"):
        return {"ok": False, "data": {}, "error": f"Invalid URL: {url}"}
    return fetch_doc_sync(url, timeout_s=timeout_s)


def start_viewer(component_type: str) -> Dict[str, Any]:
    """Collect simulation artifacts and write a local viewer manifest."""
    viewer_script = PATH.build / f"tidy3d_viewer_{component_type}.py"
    # Local-only: collect all artifacts and write a manifest
    artifacts = [str(p) for p in sorted(PATH.build.glob(f"tidy3d_*_{component_type}.png"))]
    sim_hdf5 = PATH.build / f"tidy3d_sim_{component_type}.hdf5"
    manifest_path = PATH.build / f"tidy3d_viewer_{component_type}.json"
    manifest = {
        "component_type": component_type,
        "artifacts": artifacts,
        "sim_hdf5": str(sim_hdf5) if sim_hdf5.exists() else "",
        "viewer_script": str(viewer_script) if viewer_script.exists() else "",
        "log_path": str(PATH.build / "tidy3d.log"),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "ok": True,
        "data": {
            "component_type": component_type,
            "manifest_path": str(manifest_path),
            "artifacts": artifacts,
            "viewer_id": None,
            "mode": "local",
        },
        "error": None,
    }


def capture_viewer(
    component_type: str,
    viewer_id: str | None = None,
    objective_metric: str | None = None,
) -> Dict[str, Any]:
    """Return the latest local visualization artifact for a component."""
    artifacts = sorted(PATH.build.glob(f"tidy3d_*_{component_type}.png"))
    selected = select_visualization_artifacts(
        [str(item) for item in artifacts],
        component_type=component_type,
        objective_metric=objective_metric,
        ensure_field=True,
    )
    best = selected.get("recommended_image", "")
    return {
        "ok": bool(best),
        "data": {
            "component_type": component_type,
            "objective_metric": objective_metric or "",
            "image_path": str(best) if best else "",
            "all_artifacts": [str(a) for a in artifacts],
            "selected_artifacts": selected,
            "mode": "local",
        },
        "error": None if best else "No viewer artifact available.",
    }


def check_simulation(component_type: str, viewer_id: str | None = None) -> Dict[str, Any]:
    """Check that local simulation artifacts exist for a component."""
    artifacts = [str(p) for p in PATH.build.glob(f"tidy3d_*_{component_type}.png")]
    log_path = PATH.build / "tidy3d.log"
    sim_hdf5 = PATH.build / f"tidy3d_sim_{component_type}.hdf5"
    return {
        "ok": bool(artifacts) or log_path.exists(),
        "data": {
            "component_type": component_type,
            "artifacts": sorted(artifacts),
            "log_path": str(log_path) if log_path.exists() else "",
            "sim_hdf5": str(sim_hdf5) if sim_hdf5.exists() else "",
            "mode": "local",
        },
        "error": None if artifacts or log_path.exists() else "No Tidy3D artifacts found yet.",
    }


def show_structures(
    viewer_id: str,
    visibility: Dict[str, bool] | None = None,
) -> Dict[str, Any]:
    """Toggle structure visibility via the MCP viewer bridge.

    Falls back to a local 'not available' message when the viewer bridge
    is not reachable (no running Tidy3D Viewer instance).
    """
    try:
        from PhotonicsAI.tools.mcp_client import get_client
        client = get_client()
        if client.viewer_available:
            import asyncio as _aio
            result = _aio.get_event_loop().run_until_complete(
                client.show_structures(viewer_id, visibility)
            )
            if result.get("ok"):
                return result
    except Exception:
        pass
    return {
        "ok": False,
        "data": {},
        "error": (
            "show_structures not available: no active Tidy3D Viewer bridge. "
            "Set TIDY3D_VIEWER_BRIDGE_URL to a running viewer instance."
        ),
    }


def plot_results(component_type: str) -> Dict[str, Any]:
    """Collect all generated visualization artifacts for display.

    Returns categorised lists of field plots, flux plots, and structure cross-sections
    so the UI can render them in logical groups.
    """
    field_plots = sorted(str(p) for p in PATH.build.glob(f"tidy3d_field_*_{component_type}.png"))
    mode_expansion_plots = sorted(str(p) for p in PATH.build.glob(f"tidy3d_mode_expansion*_{component_type}.png"))
    flux_plots = sorted(
        {
            *(str(p) for p in PATH.build.glob(f"tidy3d_flux_{component_type}.png")),
            *(str(p) for p in PATH.build.glob(f"tidy3d_flux_*_{component_type}.png")),
        }
    )
    structure_plots = sorted(str(p) for p in PATH.build.glob(f"tidy3d_sim_*_{component_type}.png"))
    sim_hdf5 = PATH.build / f"tidy3d_sim_{component_type}.hdf5"
    data_hdf5 = PATH.build / "tidy3d_data.hdf5"
    return {
        "ok": bool(field_plots or flux_plots or structure_plots),
        "data": {
            "component_type": component_type,
            "field_plots": field_plots,
            "mode_expansion_plots": mode_expansion_plots,
            "flux_plots": flux_plots,
            "structure_plots": structure_plots,
            "sim_hdf5": str(sim_hdf5) if sim_hdf5.exists() else "",
            "data_hdf5": str(data_hdf5) if data_hdf5.exists() else "",
        },
        "error": None,
    }


def get_tidy3d_tools() -> List[Tool]:
    return [
        Tool(
            name="run_tidy3d_simulation",
            description="Run Tidy3D simulation for a detected photonic component.",
            parameters={
                "type": "object",
                "properties": {
                    "component_type": {"type": "string"},
                    "parameters": {"type": "object"},
                    "wavelength_nm": {"type": "number"},
                    "artifact_tag": {"type": "string"},
                    "topology_density_path": {"type": "string"},
                    "topology_density_meta_path": {"type": "string"},
                    "objective_metric": {"type": "string"},
                    "target_ports": {"type": "array", "items": {"type": "string"}},
                    "target_mode_indices": {"type": "array", "items": {"type": "integer", "minimum": 0}},
                    "require_field_plot": {"type": "boolean"},
                },
                "required": ["component_type"],
            },
            fn=run_tidy3d_simulation,
        ),
        Tool(
            name="search_tidy3d_docs",
            description="Search Tidy3D / Flexcompute documentation via MCP (search_flexcompute_docs).",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
            fn=search_tidy3d_docs,
        ),
        Tool(
            name="fetch_tidy3d_doc",
            description="Fetch a Tidy3D doc page as plain text via MCP (fetch_flexcompute_doc).",
            parameters={
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
            fn=fetch_tidy3d_doc,
        ),
        Tool(
            name="start_viewer",
            description="Collect simulation artifacts and create a local viewer manifest for a component.",
            parameters={
                "type": "object",
                "properties": {"component_type": {"type": "string"}},
                "required": ["component_type"],
            },
            fn=start_viewer,
        ),
        Tool(
            name="capture_viewer",
            description="Return the latest local visualization artifact (PNG) for a component.",
            parameters={
                "type": "object",
                "properties": {
                    "component_type": {"type": "string"},
                    "viewer_id": {"type": "string", "description": "viewer_id from start_viewer (optional)"},
                    "objective_metric": {"type": "string", "description": "Objective metric for goal-aware image selection (optional)."},
                },
                "required": ["component_type"],
            },
            fn=capture_viewer,
        ),
        Tool(
            name="check_simulation",
            description="Check that local simulation artifacts and logs exist for a component.",
            parameters={
                "type": "object",
                "properties": {
                    "component_type": {"type": "string"},
                    "viewer_id": {"type": "string", "description": "viewer_id from start_viewer (optional)"},
                },
                "required": ["component_type"],
            },
            fn=check_simulation,
        ),
        Tool(
            name="show_structures",
            description="(Not available) Toggle structure visibility — tidy3d-mcp server does not expose viewer tools.",
            parameters={
                "type": "object",
                "properties": {
                    "viewer_id": {"type": "string"},
                    "visibility": {"type": "object", "description": "Map of structure_name→visible (bool)"},
                },
                "required": ["viewer_id"],
            },
            fn=show_structures,
        ),
        Tool(
            name="plot_results",
            description="Collect all simulation visualization artifacts (field plots, flux plots, structure cross-sections) for display.",
            parameters={
                "type": "object",
                "properties": {"component_type": {"type": "string"}},
                "required": ["component_type"],
            },
            fn=plot_results,
        ),
    ]
