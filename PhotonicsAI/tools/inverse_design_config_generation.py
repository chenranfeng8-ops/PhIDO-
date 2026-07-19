"""Config generation helpers for inverse-design step 9.1 item 3."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List

from pydantic import BaseModel, ConfigDict, Field

from PhotonicsAI.Photon.component_detector import get_component_sim_params
from PhotonicsAI.Photon.tidy3d_runner import (
    _port_prop_axis,
    monitor_size_for_port,
    source_size_for_port,
)
from PhotonicsAI.tools.inverse_design_config import (
    DocumentationReference,
    DomainSpec,
    GeometrySpec,
    InverseDesignConfigBundle,
    MonitorSpec,
    ObjectiveSpec,
    OptimizationConfig,
    SimulationConfig,
    SourceSpec,
    TerminationSpec,
    VariableBounds,
    build_default_runtime_config,
    parse_inverse_design_config,
)
from PhotonicsAI.tools.inverse_design_doc_context import (
    DocumentationReference as Step2DocumentationReference,
    DocumentationRuleSummary as Step2DocumentationRuleSummary,
    InverseDesignDocContext,
    retrieve_inverse_design_doc_context,
)
from PhotonicsAI.tools.inverse_design_requirements import InverseDesignRequirement
from PhotonicsAI.tools.inverse_design_working_memory import (
    InverseDesignWorkingMemory,
    get_inverse_design_working_memory,
)

_LIGHT_SPEED_M_PER_S = 299_792_458.0
_DEFAULT_GROUP_INDEX = 4.0
_DEFAULT_MAX_ITERATIONS = 20
_DEFAULT_MULTI_CASE_MAX_ITERATIONS = 30
_DEFAULT_MULTI_CASE_RUN_TIME_MULTIPLIER = 1.5
_DEFAULT_MULTI_CASE_MIN_RUN_TIME_S = 2.2e-12
_DEFAULT_STEP5_SLICE_ITERATIONS = 3
_UNICODE_SUBSCRIPT_DIGITS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
_DEGRADED_DOC_SUMMARY = (
    "No external MCP docs were retrieved. Use conservative inverse-design defaults: "
    "PML boundaries, mode source, field monitor and flux monitor, then enforce Step4 hard validation."
)


class StrictModel(BaseModel):
    """Base model that rejects unknown fields to keep planner output stable."""

    model_config = ConfigDict(extra="forbid")


class Step3PlannerDecision(StrictModel):
    """Optional LLM planning decision used to tune generated config fields."""

    source_type: str = ""
    boundary: str = ""
    monitor_types: List[str] = Field(default_factory=list)
    mesh_strategy: str = ""
    min_steps_per_wvl: int | None = None
    run_time_multiplier: float | None = None
    shutoff: float | None = None
    max_iterations: int | None = None
    target_score: float | None = None
    rationale: str = ""


def generate_inverse_design_config(
    doc_context: InverseDesignDocContext,
    *,
    use_llm_planner: bool | None = None,
    llm_call_fn: Any | None = None,
    llm_model: str = "gpt-5.4",
    memory_store: InverseDesignWorkingMemory | None = None,
    search_fn: Any | None = None,
    fetch_fn: Any | None = None,
) -> InverseDesignConfigBundle:
    """Generate a strict inverse-design config bundle from requirement + doc context."""

    requirement = doc_context.requirement
    if not requirement.is_complete:
        missing = ", ".join(requirement.missing_critical_fields)
        raise ValueError(f"Requirement is incomplete: {missing}")
    doc_context_refs = list(doc_context.references or [])
    if not doc_context_refs:
        doc_context_refs = [
            Step2DocumentationReference(
                url="memory://step2/degraded/no_mcp_docs",
                title="Step2 degraded-mode fallback guidance",
                summary=_DEGRADED_DOC_SUMMARY,
                rules=[
                    Step2DocumentationRuleSummary(rule="Use PML boundaries on all simulation sides."),
                    Step2DocumentationRuleSummary(rule="Use a mode source for guided-wave excitation."),
                    Step2DocumentationRuleSummary(rule="Keep both field and flux monitors enabled."),
                ],
            )
        ]
        (memory_store or get_inverse_design_working_memory()).record(
            stage="step3_config_generation",
            key=requirement.component_type,
            summary="Step3 auto-injected degraded doc reference because Step2 references were empty.",
            metadata={"degraded_mode": True, "reason": "empty_doc_context_references"},
        )

    component_type = requirement.component_type or "waveguide"
    wavelength_nm = requirement.wavelength_nm or 1550.0
    geometry_params = _base_geometry_params(component_type)
    structure_template_path = _structure_template_path(doc_context)
    _apply_footprint_overrides(geometry_params, component_type, requirement.constraints)
    objective_cases = _build_objective_cases(requirement)
    objective_metric = _infer_objective_metric(requirement, objective_cases)
    if component_type in {"mmi", "splitter"}:
        geometry_params["mmi_num_outputs"] = _infer_mmi_side_port_count(requirement, objective_cases)
    _apply_port_width_contract_overrides(
        geometry_params,
        component_type=component_type,
        objective_cases=objective_cases,
        constraints=requirement.constraints,
        raw_request=requirement.raw_request,
    )
    domain_size = _domain_size(component_type, geometry_params)
    freqs_hz = [_freq_from_wavelength_nm(wavelength_nm)]
    doc_refs = _convert_doc_references(doc_context_refs)
    source = _build_source(
        component_type,
        wavelength_nm,
        geometry_params,
        domain_size,
        doc_context,
        objective_cases=objective_cases,
        objective_metric=objective_metric,
    )
    monitors = _build_monitors_port_aligned(
        component_type,
        geometry_params,
        domain_size,
        freqs_hz,
        doc_context,
        objective_cases=objective_cases,
        objective_metric=objective_metric,
    )
    variables = _build_variable_bounds(
        component_type,
        geometry_params,
        constraints=requirement.constraints,
    )
    is_multi_case_objective = _is_multi_case_objective(objective_cases)
    min_steps_per_wvl = _infer_min_steps_per_wvl(doc_context)
    if is_multi_case_objective:
        # Multi-wavelength demux/WDM configs are more sensitive to mesh quality;
        # keep a deterministic floor aligned with the validated M98 setup.
        min_steps_per_wvl = max(min_steps_per_wvl, 28)
    run_time_s = _recommended_run_time_s(domain_size, source.wavelength_nm, source.bandwidth_nm)
    if is_multi_case_objective:
        run_time_s = _recommended_multi_case_run_time_s(run_time_s)
    default_max_iterations = _default_max_iterations(is_multi_case_objective)

    objective_target_value = _derive_objective_target_value(requirement, objective_cases)

    payload = {
        "simulation_config": {
            "component_type": component_type,
            "wavelength_nm": wavelength_nm,
            "geometry": {
                "component_type": component_type,
                "template_path": structure_template_path,
                "parameters": geometry_params,
                "variable_regions": ["design_region"],
            },
            "domain": {
                "size_um": domain_size,
                "center_um": [0.0, 0.0, 0.0],
                "mesh_strategy": "auto",
                "min_steps_per_wvl": min_steps_per_wvl,
                "boundary": {
                    "x_min": _boundary_type(doc_context),
                    "x_max": _boundary_type(doc_context),
                    "y_min": _boundary_type(doc_context),
                    "y_max": _boundary_type(doc_context),
                    "z_min": _boundary_type(doc_context),
                    "z_max": _boundary_type(doc_context),
                },
            },
            "source": source,
            "monitors": monitors,
            "run_time_s": run_time_s,
            "shutoff": 1e-5,
            "doc_references": doc_refs,
        },
        "optimization_config": {
            "optimizer": "inverse_design",
            "objective": {
                "metric": objective_metric,
                "goal": "maximize" if objective_cases else requirement.objective.goal,
                "target_value": objective_target_value,
                "description": requirement.objective_function or requirement.objective.description,
            },
            "objective_cases": objective_cases,
            "variables": variables,
            "termination": {
                "max_iterations": default_max_iterations,
                "target_score": _derive_target_score(requirement),
                "min_improvement": 0.005,
                "patience": 3,
            },
            "constraints": _build_constraint_descriptions(requirement),
            "doc_references": doc_refs,
        },
        "runtime_config": build_default_runtime_config(
            max_iterations=min(default_max_iterations, _DEFAULT_STEP5_SLICE_ITERATIONS)
        ).model_dump(),
    }
    memory = memory_store or get_inverse_design_working_memory()
    scenario_memory_context = _build_step3_scenario_memory_context(requirement, memory)
    if scenario_memory_context["entry_count"] > 0:
        memory.record(
            stage="step3_config_generation",
            key=requirement.component_type,
            summary=(
                f"Loaded {scenario_memory_context['entry_count']} scenario-memory entries "
                "for Step3 configuration planning."
            ),
            metadata={
                "scenario_memory_fingerprint": scenario_memory_context["fingerprint"],
                "scenario_memory_query": scenario_memory_context["query"],
                "scenario_memory_entry_count": scenario_memory_context["entry_count"],
                "scenario_memory_stages": scenario_memory_context["stages"],
            },
        )
    tuned_payload = _apply_step3_llm_decision(
        payload,
        doc_context,
        use_llm_planner=use_llm_planner,
        llm_call_fn=llm_call_fn,
        llm_model=llm_model,
        memory_store=memory,
        search_fn=search_fn,
        fetch_fn=fetch_fn,
        scenario_memory_prompt=scenario_memory_context["prompt"],
    )
    bundle = parse_inverse_design_config(tuned_payload)
    memory.record(
        stage="step3_config_generation",
        key=requirement.component_type,
        summary=(
            f"Step3 generated config for {requirement.component_type} "
            f"with {len(bundle.simulation_config.monitors)} monitors."
        ),
        evidence_urls=[ref.url for ref in bundle.simulation_config.doc_references],
        metadata={
            "wavelength_nm": bundle.simulation_config.wavelength_nm,
            "mesh_strategy": bundle.simulation_config.domain.mesh_strategy,
            "min_steps_per_wvl": bundle.simulation_config.domain.min_steps_per_wvl,
            "objective_case_count": len(bundle.optimization_config.objective_cases),
            "scenario_memory_fingerprint": scenario_memory_context["fingerprint"],
            "scenario_memory_entry_count": scenario_memory_context["entry_count"],
            "target_mode_indices": [
                int(getattr(item, "target_mode_index", 0) or 0)
                for item in bundle.optimization_config.objective_cases
            ],
            "mode_monitors": [
                monitor.name
                for monitor in bundle.simulation_config.monitors
                if monitor.monitor_type == "mode"
            ],
        },
    )
    objective_cases_dump = [
        {
            "source_port": str(getattr(item, "source_port", "") or ""),
            "source_mode_index": int(getattr(item, "source_mode_index", 0) or 0),
            "target_port": str(getattr(item, "target_port", "") or ""),
            "target_mode_index": int(getattr(item, "target_mode_index", 0) or 0),
        }
        for item in bundle.optimization_config.objective_cases
    ]
    max_mode_index = max(
        (item["target_mode_index"] for item in objective_cases_dump),
        default=0,
    )
    if objective_cases_dump and max_mode_index > 0:
        has_mux_case = any(
            str(item.get("source_port", "")).strip().lower() not in {"", "port_o1"}
            and str(item.get("target_port", "")).strip().lower() == "port_o1"
            for item in objective_cases_dump
        )
        memory.record(
            stage="step3_config_generation",
            key=f"{requirement.component_type}_{'mode_mux' if has_mux_case else 'mode_demux'}_contract",
            summary=(
                "Mode-routing config contract applied: per-target mode monitors "
                "and multi-mode observability requirements recorded."
            ),
            proposed_fixes=[
                "Keep mode_port_* monitors for every target_port in objective_cases.",
                "Ensure diagnostic/acceptance rerender uses num_modes >= max(target_mode_index)+1.",
            ],
            metadata={
                "objective_cases": objective_cases_dump,
                "required_num_modes": max_mode_index + 1,
            },
        )
    return bundle


def build_inverse_design_config_from_request(
    requirement: InverseDesignRequirement,
    *,
    use_llm_planner: bool | None = None,
    llm_call_fn: Any | None = None,
    llm_model: str = "gpt-5.4",
    memory_store: InverseDesignWorkingMemory | None = None,
) -> InverseDesignConfigBundle:
    """Convenience helper that runs doc retrieval then config generation."""

    if (
        use_llm_planner is None
        and llm_call_fn is None
        and llm_model == "gpt-5.4"
        and memory_store is None
    ):
        doc_context = retrieve_inverse_design_doc_context(requirement)
    else:
        doc_context = retrieve_inverse_design_doc_context(
            requirement,
            use_llm_planner=use_llm_planner,
            llm_call_fn=llm_call_fn,
            llm_model=llm_model,
            memory_store=memory_store,
        )
    return generate_inverse_design_config(
        doc_context,
        use_llm_planner=use_llm_planner,
        llm_call_fn=llm_call_fn,
        llm_model=llm_model,
        memory_store=memory_store,
    )


def inverse_design_config_generation_schema() -> Dict[str, Any]:
    """Return schema for the generated config bundle."""

    return InverseDesignConfigBundle.model_json_schema()



def _structure_template_path(doc_context: InverseDesignDocContext) -> str | None:
    """Return the Step2-selected reusable structure code reference, when present."""

    structure_context = getattr(doc_context, "structure_context", None)
    if structure_context is None:
        return None
    builder_reference = str(getattr(structure_context, "builder_reference", "") or "").strip()
    if builder_reference:
        return builder_reference
    builder_module = str(getattr(structure_context, "builder_module", "") or "").strip()
    builder_function = str(getattr(structure_context, "builder_function", "") or "").strip()
    if builder_module and builder_function:
        return f"{builder_module}.{builder_function}"
    return None

def _base_geometry_params(component_type: str) -> Dict[str, Any]:
    params = dict(get_component_sim_params(component_type))
    if component_type == "crossing":
        params.setdefault("wg_length", 8.0)
    elif component_type == "waveguide":
        params.setdefault("wg_length", 12.0)
    elif component_type == "ring_resonator":
        params.setdefault("ring_radius", 5.0)
        params.setdefault("gap", 0.2)
    elif component_type in {"mmi", "splitter"}:
        params.setdefault("mmi_width", 2.5)
        params.setdefault("mmi_length", 10.0)
    elif component_type == "y_branch":
        params.setdefault("arm_length", 15.0)
        params.setdefault("arm_separation", 3.0)
    return params


_FOOTPRINT_DIM_RE = re.compile(
    r"([0-9]+(?:\.[0-9]+)?)\s*(\u00b5m|\u03bcm|um|nm)",
    re.IGNORECASE,
)


def _constraint_value_to_um(value: Any, unit: str) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    normalized = str(unit or "").strip().lower()
    if normalized == "nm":
        return numeric / 1000.0
    return numeric


def _constraint_payload(item: Any) -> Dict[str, Any]:
    if isinstance(item, dict):
        return dict(item)
    if hasattr(item, "model_dump"):
        try:
            return dict(item.model_dump())
        except Exception:
            pass
    return {
        "name": getattr(item, "name", ""),
        "raw_value": getattr(item, "raw_value", ""),
        "description": getattr(item, "description", ""),
        "unit": getattr(item, "unit", ""),
        "target_value": getattr(item, "target_value", None),
    }


def _extract_footprint_dims_um(constraints: List[Any]) -> tuple[float, float] | None:
    pair_names = {"footprint", "initial_footprint"}
    x_names = {"footprint_x", "initial_structure_size_x"}
    y_names = {"footprint_y", "initial_structure_size_y"}

    for constraint in constraints:
        payload = _constraint_payload(constraint)
        name = str(payload.get("name", "") or "").strip().lower()
        raw = str(payload.get("raw_value", "") or "").strip()
        description = str(payload.get("description", "") or "").strip()
        if name not in pair_names:
            continue
        text = raw or description
        dims = _FOOTPRINT_DIM_RE.findall(text)
        if len(dims) >= 2:
            values_um: List[float] = []
            for val_str, unit_str in dims[:2]:
                value_um = _constraint_value_to_um(val_str, unit_str)
                if value_um is None:
                    continue
                values_um.append(float(value_um))
            if len(values_um) >= 2:
                return float(values_um[0]), float(values_um[1])
        nums = re.findall(r"([0-9]+(?:\.[0-9]+)?)", text)
        if len(nums) >= 2:
            return float(nums[0]), float(nums[1])

    footprint_x_um: float | None = None
    footprint_y_um: float | None = None
    for constraint in constraints:
        payload = _constraint_payload(constraint)
        name = str(payload.get("name", "") or "").strip().lower()
        if name not in x_names.union(y_names):
            continue
        unit = str(payload.get("unit", "") or "")
        raw = payload.get("raw_value")
        target_value = payload.get("target_value")
        value_um = _constraint_value_to_um(target_value, unit)
        if value_um is None:
            value_um = _constraint_value_to_um(raw, unit)
        if value_um is None:
            continue
        if name in x_names:
            footprint_x_um = float(value_um)
        elif name in y_names:
            footprint_y_um = float(value_um)

    if footprint_x_um is not None and footprint_y_um is not None:
        return float(footprint_x_um), float(footprint_y_um)
    return None


def _apply_footprint_overrides(
    params: Dict[str, Any],
    component_type: str,
    constraints: List[Any],
) -> None:
    """Override geometry params with user-specified footprint dimensions (in-place)."""
    footprint_dims = _extract_footprint_dims_um(constraints)
    if footprint_dims is None:
        return

    dim1, dim2 = footprint_dims
    if component_type in {"mmi", "splitter"}:
        params["mmi_length"] = dim1
        params["mmi_width"] = dim2
    elif component_type == "crossing":
        params["wg_length"] = max(dim1, dim2)
    elif component_type == "waveguide":
        params["wg_length"] = dim1
    elif component_type == "y_branch":
        params["arm_length"] = dim1
        params["arm_separation"] = dim2


def _apply_port_width_contract_overrides(
    params: Dict[str, Any],
    *,
    component_type: str,
    objective_cases: List[Dict[str, Any]],
    constraints: List[Any],
    raw_request: str = "",
) -> None:
    if component_type not in {"mmi", "splitter"}:
        return
    has_mux_case = any(
        str(item.get("source_port", "")).strip().lower() not in {"", "port_o1"}
        and str(item.get("target_port", "")).strip().lower() == "port_o1"
        for item in objective_cases
    )
    if not has_mux_case:
        return

    input_width_um: float | None = None
    output_width_um: float | None = None
    for constraint in constraints:
        payload = _constraint_payload(constraint)
        name = str(payload.get("name", "") or "").strip().lower()
        unit = str(payload.get("unit", "") or "")
        value_um = _constraint_value_to_um(payload.get("target_value"), unit)
        if value_um is None:
            value_um = _constraint_value_to_um(payload.get("raw_value"), unit)
        if value_um is None:
            continue
        if name == "input_waveguide_width":
            input_width_um = float(value_um)
        elif name == "output_waveguide_width":
            output_width_um = float(value_um)

    if input_width_um is None or output_width_um is None:
        text = str(raw_request or "")
        width_patterns = [
            # input/output ... 500 nm wide
            re.compile(
                r"(input|output)[^.;,\n]{0,120}?([0-9]+(?:\.[0-9]+)?)\s*(nm|um|μm|µm)\s*(?:wide|width|w)",
                re.IGNORECASE,
            ),
            # input/output ... width 500 nm
            re.compile(
                r"(input|output)[^.;,\n]{0,120}?(?:wide|width)\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)\s*(nm|um|μm|µm)",
                re.IGNORECASE,
            ),
            # 输入/出射 ... 500 nm 宽
            re.compile(
                r"(输入|入射|输出|出射)[^。；，,\n]{0,120}?([0-9]+(?:\.[0-9]+)?)\s*(nm|um|μm|µm)\s*(?:宽|宽度)?",
                re.IGNORECASE,
            ),
            # 输入/出射 ... 宽度500 nm
            re.compile(
                r"(输入|入射|输出|出射)[^。；，,\n]{0,120}?(?:宽|宽度)\s*[:：=]?\s*([0-9]+(?:\.[0-9]+)?)\s*(nm|um|μm|µm)",
                re.IGNORECASE,
            ),
        ]
        for pattern in width_patterns:
            for match in pattern.finditer(text):
                role = str(match.group(1) or "").strip().lower()
                numeric = _constraint_value_to_um(match.group(2), match.group(3))
                if numeric is None:
                    continue
                if role in {"input", "输入", "入射"} and input_width_um is None:
                    input_width_um = float(numeric)
                if role in {"output", "输出", "出射"} and output_width_um is None:
                    output_width_um = float(numeric)

    if input_width_um is not None:
        params["side_port_wg_width"] = max(input_width_um, 1e-3)
    if output_width_um is not None:
        params["port_o1_wg_width"] = max(output_width_um, 1e-3)


def _domain_size(component_type: str, params: Dict[str, Any]) -> List[float]:
    if component_type == "crossing":
        wg_length = float(params.get("wg_length", 8.0))
        span = wg_length * 2.0 + 2.0
        return [span, span, 4.0]
    if component_type == "waveguide":
        length = float(params.get("wg_length", 12.0))
        return [length * 2.0 + 2.0, 4.0, 4.0]
    if component_type == "ring_resonator":
        radius = float(params.get("ring_radius", 5.0))
        return [radius * 5.0, radius * 5.0, 4.0]
    if component_type in {"mmi", "splitter"}:
        mmi_length = float(params.get("mmi_length", 10.0))
        mmi_width = float(params.get("mmi_width", 2.5))
        return [mmi_length + 24.0, mmi_width + 4.0, 4.0]
    if component_type == "directional_coupler":
        coupler_length = float(params.get("coupler_length", 10.0))
        gap = float(params.get("gap", 0.2))
        wg_width = float(params.get("wg_width", 0.5))
        return [coupler_length + 24.0, gap + wg_width * 2.0 + 4.0, 4.0]
    if component_type == "mzi":
        arm_length = float(params.get("arm_length", 20.0))
        arm_separation = float(params.get("arm_separation", 5.0))
        return [arm_length + 30.0, arm_separation + 6.0, 4.0]
    if component_type == "y_branch":
        arm_length = float(params.get("arm_length", 15.0))
        arm_separation = float(params.get("arm_separation", 3.0))
        return [arm_length * 3.0 + 25.0, arm_separation + 8.0, 4.0]
    return [12.0, 6.0, 4.0]


def _build_source(
    component_type: str,
    wavelength_nm: float,
    params: Dict[str, Any],
    domain_size: List[float],
    doc_context: InverseDesignDocContext,
    *,
    objective_cases: List[Dict[str, Any]] | None = None,
    objective_metric: str = "",
) -> SourceSpec:
    wg_width = float(params.get("wg_width", 0.5))
    wg_height = float(params.get("wg_height", 0.22))
    ports = _output_port_positions(component_type, params, domain_size)
    source_port = "port_o1"
    source_mode_index = 0
    if objective_cases:
        first = objective_cases[0]
        case_source_port = str(first.get("source_port", "") or "").strip().lower()
        if case_source_port:
            source_port = case_source_port
        try:
            source_mode_index = max(int(first.get("source_mode_index", 0) or 0), 0)
        except (TypeError, ValueError):
            source_mode_index = 0

    source_xy = ports.get(source_port)
    if source_xy is None:
        source_xy = (-domain_size[0] / 2 + 1.0, 0.0)
    axis = _port_prop_axis(component_type, source_port)
    source_width = _waveguide_width_for_port(
        component_type=component_type,
        params=params,
        port_name=source_port,
        objective_metric=objective_metric,
    )
    source_size = list(source_size_for_port(axis, source_width, wg_height))
    direction = _default_port_direction(component_type, source_port, source_xy)
    return SourceSpec(
        source_type=_source_type(doc_context),
        port=source_port,
        center_um=[source_xy[0], source_xy[1], wg_height / 2],
        size_um=source_size,
        direction=direction,
        mode_index=source_mode_index,
        wavelength_nm=wavelength_nm,
        bandwidth_nm=max(wavelength_nm * 0.05, 50.0),
    )


def _build_monitors(
    component_type: str,
    params: Dict[str, Any],
    domain_size: List[float],
    freqs_hz: List[float],
    doc_context: InverseDesignDocContext,
) -> List[MonitorSpec]:
    wg_width = float(params.get("wg_width", 0.5))
    wg_height = float(params.get("wg_height", 0.22))
    monitors: List[MonitorSpec] = [
        MonitorSpec(
            name="field_monitor",
            monitor_type="field",
            center_um=[0.0, 0.0, wg_height / 2],
            size_um=[domain_size[0] * 0.9, domain_size[1] * 0.9, 0.0],
            freqs_hz=freqs_hz,
            field_component="Ey",
        )
    ]
    # Explicit global z-normal plane field monitor for debugging/inspection.
    monitors.append(
        MonitorSpec(
            name="field_z",
            monitor_type="field",
            center_um=[0.0, 0.0, wg_height / 2],
            size_um=[domain_size[0] * 0.9, domain_size[1] * 0.9, 0.0],
            freqs_hz=freqs_hz,
            field_component="Ey",
        )
    )

    output_x = domain_size[0] / 2 - 1.0
    # Use shared helper for monitor sizing (single source of truth).
    flux_size_x = list(monitor_size_for_port("x", wg_width, wg_height))
    flux_size_y = list(monitor_size_for_port("y", wg_width, wg_height))
    # Flux monitors are always needed for quantitative optimization —
    # add unconditionally for the primary through port.
    monitors.append(
        MonitorSpec(
            name="through_flux",
            monitor_type="flux",
            center_um=[output_x, 0.0, wg_height / 2],
            size_um=flux_size_x,
            freqs_hz=freqs_hz,
            metric="transmission",
        )
    )

    # Multi-port split devices always need a secondary output flux monitor.
    if component_type in {"crossing", "y_branch", "mmi", "splitter", "directional_coupler", "mzi"}:
        offset_y = max(float(params.get("arm_separation", params.get("mmi_width", 2.5))) / 2, 1.25)
        # Use _port_prop_axis to determine correct orientation per component.
        sec_port = "port_o3" if component_type != "crossing" else "port_o3"
        sec_axis = _port_prop_axis(component_type, sec_port)
        sec_size = list(monitor_size_for_port(sec_axis, wg_width, wg_height))
        monitors.append(
            MonitorSpec(
                name="secondary_flux",
                monitor_type="flux",
                center_um=[output_x if component_type != "crossing" else 0.0, offset_y, wg_height / 2],
                size_um=sec_size,
                freqs_hz=freqs_hz,
                metric="secondary_output",
            )
        )

    if _monitor_enabled(doc_context, "mode"):
        monitors.append(
            MonitorSpec(
                name="mode_monitor",
                monitor_type="mode",
                center_um=[output_x, 0.0, wg_height / 2],
                size_um=flux_size_x,
                freqs_hz=freqs_hz,
                metric="mode_overlap",
            )
        )

    if not any(m.monitor_type == "flux" for m in monitors):
        monitors.append(
            MonitorSpec(
                name="through_flux",
                monitor_type="flux",
                center_um=[output_x, 0.0, wg_height / 2],
                size_um=flux_size_x,
                freqs_hz=freqs_hz,
                metric="transmission",
            )
        )

    if not any(m.monitor_type == "mode" for m in monitors):
        monitors.append(
            MonitorSpec(
                name="mode_monitor",
                monitor_type="mode",
                center_um=[output_x, 0.0, wg_height / 2],
                size_um=flux_size_x,
                freqs_hz=freqs_hz,
                metric="mode_overlap",
            )
        )
    return monitors


def _build_monitors_port_aligned(
    component_type: str,
    params: Dict[str, Any],
    domain_size: List[float],
    freqs_hz: List[float],
    doc_context: InverseDesignDocContext,
    objective_cases: List[Dict[str, Any]] | None = None,
    objective_metric: str = "",
) -> List[MonitorSpec]:
    """Build monitors aligned to builder port coordinates (Step3->Step5 consistency)."""
    wg_height = float(params.get("wg_height", 0.22))
    objective_metric = str(objective_metric or "").strip().lower()
    demux_like_metrics = {"demux_routing", "mode_demux", "wdm_routing", "mux_routing"}
    ports = _output_port_positions(component_type, params, domain_size)
    port_o2 = ports.get("port_o2", (domain_size[0] / 2 - 1.0, 0.0))
    port_o3 = ports.get("port_o3")

    monitors: List[MonitorSpec] = [
        MonitorSpec(
            name="field_monitor",
            monitor_type="field",
            center_um=[0.0, 0.0, wg_height / 2],
            size_um=[domain_size[0] * 0.9, domain_size[1] * 0.9, 0.0],
            freqs_hz=freqs_hz,
            field_component="Ey",
        )
    ]
    # Explicit global z-normal plane field monitor for debugging/inspection.
    monitors.append(
        MonitorSpec(
            name="field_z",
            monitor_type="field",
            center_um=[0.0, 0.0, wg_height / 2],
            size_um=[domain_size[0] * 0.9, domain_size[1] * 0.9, 0.0],
            freqs_hz=freqs_hz,
            field_component="Ey",
        )
    )

    primary_axis = _port_prop_axis(component_type, "port_o2")
    primary_width = _waveguide_width_for_port(
        component_type=component_type,
        params=params,
        port_name="port_o2",
        objective_metric=objective_metric,
    )
    primary_size = list(monitor_size_for_port(primary_axis, primary_width, wg_height))
    monitors.append(
        MonitorSpec(
            name="through_flux",
            monitor_type="flux",
            center_um=[port_o2[0], port_o2[1], wg_height / 2],
            size_um=primary_size,
            freqs_hz=freqs_hz,
            metric="transmission",
        )
    )

    if component_type in {"crossing", "y_branch", "mmi", "splitter", "directional_coupler", "mzi"} and port_o3 is not None:
        sec_axis = _port_prop_axis(component_type, "port_o3")
        sec_width = _waveguide_width_for_port(
            component_type=component_type,
            params=params,
            port_name="port_o3",
            objective_metric=objective_metric,
        )
        sec_size = list(monitor_size_for_port(sec_axis, sec_width, wg_height))
        monitors.append(
            MonitorSpec(
                name="secondary_flux",
                monitor_type="flux",
                center_um=[port_o3[0], port_o3[1], wg_height / 2],
                size_um=sec_size,
                freqs_hz=freqs_hz,
                metric="secondary_output",
            )
        )

    # Demux/mode-demux requires per-target output ModeMonitors so TE0/TE1
    # observability is explicit in Step3 config and downstream rerender.
    objective_mode_ports: List[str] = []
    for case in objective_cases or []:
        port_name = str(case.get("target_port", "")).strip().lower()
        if not port_name.startswith("port_o"):
            continue
        if port_name not in objective_mode_ports:
            objective_mode_ports.append(port_name)

    output_port_targets: List[str] = []
    if objective_metric in demux_like_metrics:
        for port_name in sorted(ports.keys()):
            if not port_name.startswith("port_o"):
                continue
            if objective_metric != "mux_routing" and port_name == "port_o1":
                continue
            if port_name not in output_port_targets:
                output_port_targets.append(port_name)
    for port_name in objective_mode_ports:
        if port_name not in output_port_targets:
            output_port_targets.append(port_name)
    # Keep input-port observability explicit for mode-mux validation.
    if "port_o1" in ports and "port_o1" not in output_port_targets:
        output_port_targets.append("port_o1")

    if output_port_targets:
        for port_name in output_port_targets:
            port_xy = ports.get(port_name)
            if port_xy is None:
                continue
            axis = _port_prop_axis(component_type, port_name)
            port_width = _waveguide_width_for_port(
                component_type=component_type,
                params=params,
                port_name=port_name,
                objective_metric=objective_metric,
            )
            field_size = list(monitor_size_for_port(axis, port_width, wg_height))
            monitors.append(
                MonitorSpec(
                    name=f"field_{port_name}",
                    monitor_type="field",
                    center_um=[port_xy[0], port_xy[1], wg_height / 2],
                    size_um=field_size,
                    freqs_hz=freqs_hz,
                    field_component="Ey",
                )
            )

    if _monitor_enabled(doc_context, "mode") or output_port_targets:
        target_ports = output_port_targets or ["port_o2"]
        for port_name in target_ports:
            port_xy = ports.get(port_name)
            if port_xy is None:
                continue
            axis = _port_prop_axis(component_type, port_name)
            port_width = _waveguide_width_for_port(
                component_type=component_type,
                params=params,
                port_name=port_name,
                objective_metric=objective_metric,
            )
            mode_size = list(monitor_size_for_port(axis, port_width, wg_height))
            monitors.append(
                MonitorSpec(
                    name=f"mode_{port_name}",
                    monitor_type="mode",
                    center_um=[port_xy[0], port_xy[1], wg_height / 2],
                    size_um=mode_size,
                    freqs_hz=freqs_hz,
                    metric="mode_overlap",
                )
            )

    if not any(m.monitor_type == "flux" for m in monitors):
        monitors.append(
            MonitorSpec(
                name="through_flux",
                monitor_type="flux",
                center_um=[port_o2[0], port_o2[1], wg_height / 2],
                size_um=primary_size,
                freqs_hz=freqs_hz,
                metric="transmission",
            )
        )

    if not any(m.monitor_type == "mode" for m in monitors):
        monitors.append(
            MonitorSpec(
                name="mode_port_o2",
                monitor_type="mode",
                center_um=[port_o2[0], port_o2[1], wg_height / 2],
                size_um=primary_size,
                freqs_hz=freqs_hz,
                metric="mode_overlap",
            )
        )
    return monitors


def _output_port_positions(
    component_type: str,
    params: Dict[str, Any],
    domain_size: List[float],
) -> Dict[str, tuple[float, float]]:
    """Estimate component port centers aligned to tidy3d_runner builders."""
    fallback_x = float(domain_size[0]) / 2 - 1.0
    mapping: Dict[str, tuple[float, float]] = {
        "port_o1": (-float(domain_size[0]) / 2 + 1.0, 0.0),
        "port_o2": (fallback_x, 0.0),
    }

    if component_type in {"mmi", "splitter"}:
        mmi_width = float(params.get("mmi_width", 2.5))
        mmi_length = float(params.get("mmi_length", 10.0))
        num_outputs = max(int(round(float(params.get("mmi_num_outputs", 2) or 2))), 2)
        output_spacing = mmi_width / float(num_outputs + 1)
        x_in = -mmi_length / 2.0 - 9.0
        x_out = mmi_length / 2.0 + 9.0
        mapping["port_o1"] = (x_in, 0.0)
        for port_offset in range(num_outputs):
            port_name = f"port_o{port_offset + 2}"
            y_port = -mmi_width / 2.0 + output_spacing * float(port_offset + 1)
            mapping[port_name] = (x_out, y_port)
    elif component_type == "y_branch":
        arm_length = float(params.get("arm_length", 15.0))
        arm_separation = float(params.get("arm_separation", 3.0))
        x_in = -arm_length - 9.0
        x_out = arm_length + 9.0
        mapping["port_o1"] = (x_in, 0.0)
        mapping["port_o2"] = (x_out, arm_separation / 2.0)
        mapping["port_o3"] = (x_out, -arm_separation / 2.0)
    elif component_type == "directional_coupler":
        coupler_length = float(params.get("coupler_length", 10.0))
        gap = float(params.get("gap", 0.2))
        wg_width = float(params.get("wg_width", 0.5))
        y_upper = gap / 2.0 + wg_width / 2.0
        y_lower = -y_upper
        x_in = -coupler_length / 2.0 - 9.0
        x_out = coupler_length / 2.0 + 9.0
        mapping["port_o1"] = (x_in, y_upper)
        mapping["port_o2"] = (x_out, y_upper)
        mapping["port_o3"] = (x_out, y_lower)
        mapping["port_o4"] = (x_in, y_lower)
    elif component_type == "crossing":
        wg_length = float(params.get("wg_length", 8.0))
        mapping["port_o1"] = (-wg_length + 1.0, 0.0)
        mapping["port_o2"] = (wg_length - 1.0, 0.0)
        mapping["port_o3"] = (0.0, -wg_length + 1.0)
        mapping["port_o4"] = (0.0, wg_length - 1.0)

    return mapping


def _infer_mmi_side_port_count(
    requirement: InverseDesignRequirement,
    objective_cases: List[Dict[str, Any]],
) -> int:
    port_numbers = {2, 3}
    for target in requirement.routing_targets:
        for raw_port in (getattr(target, "source_port", ""), getattr(target, "target_port", "")):
            port_num = _extract_port_number(str(raw_port or ""))
            if port_num > 1:
                port_numbers.add(port_num)
    for case in objective_cases:
        for raw_port in (case.get("source_port", ""), case.get("target_port", "")):
            port_num = _extract_port_number(str(raw_port or ""))
            if port_num > 1:
                port_numbers.add(port_num)
    inferred_from_ports = max(max(port_numbers) - 1, 2)
    declared_outputs = _extract_declared_mmi_outputs(requirement.raw_request)
    if declared_outputs is None:
        return inferred_from_ports
    return max(inferred_from_ports, int(declared_outputs))


def _extract_declared_mmi_outputs(text: str) -> int | None:
    lowered = str(text or "").lower()
    patterns = (
        # Use digit-boundary lookarounds instead of word-boundary \b so patterns
        # also match in CJK text like "优化一个1x5 mmi splitter".
        re.compile(r"(?<![0-9])1\s*[x×]\s*([0-9]+)(?![0-9])", re.IGNORECASE),
        re.compile(r"\bmmi\s*1\s*[x×]\s*([0-9]+)(?![0-9])", re.IGNORECASE),
    )
    for pattern in patterns:
        match = pattern.search(lowered)
        if not match:
            continue
        try:
            value = int(match.group(1))
        except (TypeError, ValueError):
            continue
        if value >= 2:
            return value
    return None


def _default_port_direction(
    component_type: str,
    port_name: str,
    port_xy: tuple[float, float],
) -> str:
    axis = _port_prop_axis(component_type, port_name)
    if axis == "y":
        return "+" if float(port_xy[1]) <= 0 else "-"
    return "+" if float(port_xy[0]) <= 0 else "-"


def _waveguide_width_for_port(
    *,
    component_type: str,
    params: Dict[str, Any],
    port_name: str,
    objective_metric: str = "",
) -> float:
    default_width = float(params.get("wg_width", 0.5))
    lowered_metric = str(objective_metric or "").strip().lower()
    lowered_port = str(port_name or "").strip().lower()
    explicit_o1 = params.get("port_o1_wg_width")
    explicit_side = params.get("side_port_wg_width")
    try:
        if lowered_port == "port_o1" and explicit_o1 is not None:
            return max(float(explicit_o1), 1e-3)
        if lowered_port != "port_o1" and explicit_side is not None:
            return max(float(explicit_side), 1e-3)
    except (TypeError, ValueError):
        pass
    if component_type in {"mmi", "splitter"} and lowered_metric == "mux_routing":
        mmi_width = float(params.get("mmi_width", 2.5))
        if lowered_port == "port_o1":
            candidate = max(default_width * 2.0, 1.0)
            upper = max(default_width + 0.05, mmi_width - 0.2)
            return min(candidate, upper)
    return default_width


def _build_variable_bounds(
    component_type: str,
    params: Dict[str, Any],
    constraints: List[Any] | None = None,
) -> List[VariableBounds]:
    footprint_dims = _extract_footprint_dims_um(constraints or [])
    footprint_x_um = float(footprint_dims[0]) if footprint_dims is not None else None
    footprint_y_um = float(footprint_dims[1]) if footprint_dims is not None else None
    variables: List[VariableBounds] = []
    fixed_contract_keys = {"port_o1_wg_width", "side_port_wg_width", "mmi_num_outputs"}
    for name, value in params.items():
        if not isinstance(value, (int, float)):
            continue
        if str(name).strip().lower() in fixed_contract_keys:
            continue
        lower, upper = _bound_for_parameter(name, float(value))
        if component_type in {"mmi", "splitter"}:
            lowered_name = str(name).strip().lower()
            if lowered_name == "mmi_length" and footprint_x_um is not None:
                upper = min(float(upper), float(footprint_x_um))
            elif lowered_name == "mmi_width" and footprint_y_um is not None:
                upper = min(float(upper), float(footprint_y_um))
            if lower > upper:
                lower = upper
        initial_value = min(max(float(value), float(lower)), float(upper))
        variables.append(
            VariableBounds(
                name=name,
                lower_bound=lower,
                upper_bound=upper,
                initial_value=initial_value,
            )
        )

    if variables:
        return variables

    fallback_name = "wg_width" if component_type != "ring_resonator" else "gap"
    fallback_value = 0.5 if fallback_name == "wg_width" else 0.2
    lower, upper = _bound_for_parameter(fallback_name, fallback_value)
    return [
        VariableBounds(
            name=fallback_name,
            lower_bound=lower,
            upper_bound=upper,
            initial_value=fallback_value,
        )
    ]


def _build_objective_cases(requirement: InverseDesignRequirement) -> List[Dict[str, Any]]:
    mode_mux_cases = _extract_mode_mux_case_specs(requirement)
    if mode_mux_cases:
        return mode_mux_cases

    cases: List[Dict[str, Any]] = []
    mode_targets = _extract_mode_target_index_by_port(requirement.raw_request)
    mode_sequence = _extract_mode_sequence(requirement.raw_request)
    for idx, target in enumerate(requirement.routing_targets):
        port_key = str(target.target_port or "").strip().lower()
        source_key = str(getattr(target, "source_port", "") or "").strip().lower()
        cases.append(
            {
                "name": f"case_{idx + 1}_{target.target_port}",
                "wavelength_nm": float(target.wavelength_nm),
                "source_port": source_key,
                "source_mode_index": int(getattr(target, "source_mode_index", 0) or 0),
                "target_port": target.target_port,
                "target_mode_index": int(
                    getattr(target, "target_mode_index", 0)
                    or mode_targets.get(port_key, 0)
                ),
                "min_coupling": target.min_coupling,
                "max_crosstalk": target.max_crosstalk,
                "weight": 1.0,
            }
        )
    has_source_switched_cases = any(
        str(item.get("source_port", "")).strip().lower() not in {"", "port_o1"}
        and str(item.get("target_port", "")).strip().lower() == "port_o1"
        for item in cases
    )
    if mode_targets:
        if cases:
            existing_ports = {
                str(item.get("target_port", "")).strip().lower()
                for item in cases
            }
            anchor_wl = float(cases[0]["wavelength_nm"])
            for item in cases:
                port_key = str(item.get("target_port", "")).strip().lower()
                if port_key in mode_targets:
                    item["target_mode_index"] = int(mode_targets[port_key])
            if not has_source_switched_cases:
                add_index = len(cases) + 1
                for port_key, mode_idx in mode_targets.items():
                    if port_key in existing_ports:
                        continue
                    cases.append(
                        {
                            "name": f"case_{add_index}_{port_key}",
                            "wavelength_nm": anchor_wl,
                            "source_port": "",
                            "source_mode_index": 0,
                            "target_port": port_key,
                            "target_mode_index": int(mode_idx),
                            "min_coupling": None,
                            "max_crosstalk": None,
                            "weight": 1.0,
                        }
                    )
                    add_index += 1
        else:
            # Fallback for single-wavelength mode demux prompts where Step1 routing
            # targets are absent but explicit "portX -> TEy" semantics exist.
            anchor_wl = None
            if requirement.wavelengths_nm:
                anchor_wl = float(requirement.wavelengths_nm[0])
            elif requirement.wavelength_nm is not None:
                anchor_wl = float(requirement.wavelength_nm)
            if anchor_wl is not None:
                for idx, (port_key, mode_idx) in enumerate(mode_targets.items(), start=1):
                    cases.append(
                        {
                            "name": f"case_{idx}_{port_key}",
                            "wavelength_nm": anchor_wl,
                            "source_port": "",
                            "source_mode_index": 0,
                            "target_port": port_key,
                            "target_mode_index": int(mode_idx),
                            "min_coupling": None,
                            "max_crosstalk": None,
                            "weight": 1.0,
                        }
                    )
    elif len(cases) >= 2 and mode_sequence:
        ordered_cases = sorted(
            cases,
            key=lambda item: str(item.get("target_port", "")),
        )
        for idx, case in enumerate(ordered_cases):
            case["target_mode_index"] = int(mode_sequence[min(idx, len(mode_sequence) - 1)])
    return cases


def _infer_objective_metric(
    requirement: InverseDesignRequirement,
    objective_cases: List[Dict[str, Any]],
) -> str:
    if objective_cases:
        has_mux_case = any(
            str(case.get("source_port", "")).strip().lower() not in {"", "port_o1"}
            and str(case.get("target_port", "")).strip().lower() == "port_o1"
            for case in objective_cases
        )
        return "mux_routing" if has_mux_case else "demux_routing"
    return str(requirement.objective.metric or "transmission")


def _extract_mode_mux_case_specs(requirement: InverseDesignRequirement) -> List[Dict[str, Any]]:
    text = str(requirement.raw_request or "")
    if not text:
        return []
    lowered = text.lower().translate(_UNICODE_SUBSCRIPT_DIGITS)

    clauses = [
        item.strip()
        for item in re.split(r"[。；;\n]+", lowered)
        if item.strip()
    ]
    if len(clauses) <= 1:
        clauses = [
            item.strip()
            for item in re.split(r"(?:,|，)(?=(?:\s*)(?:port|端口)\s*[0-9]+)", lowered)
            if item.strip()
        ]

    parsed: List[Dict[str, int]] = []
    for clause in clauses:
        source = _extract_clause_port_mode(clause, role="source")
        target = _extract_clause_port_mode(clause, role="target")
        if source is None or target is None:
            continue
        parsed.append(
            {
                "source_port_num": source["port_num"],
                "source_mode_index": source["mode_index"],
                "target_port_num": target["port_num"],
                "target_mode_index": target["mode_index"],
            }
        )

    if not parsed:
        return []

    source_ports = {item["source_port_num"] for item in parsed}
    target_ports = {item["target_port_num"] for item in parsed}
    if len(source_ports) < 2 or target_ports != {1}:
        return []

    if requirement.wavelengths_nm:
        wavelength_nm = float(requirement.wavelengths_nm[0])
    elif requirement.wavelength_nm is not None:
        wavelength_nm = float(requirement.wavelength_nm)
    else:
        wavelength_nm = 1550.0

    targets_by_port: Dict[str, Any] = {}
    for item in requirement.routing_targets:
        key = str(getattr(item, "target_port", "") or "").strip().lower()
        if key and key not in targets_by_port:
            targets_by_port[key] = item

    cases: List[Dict[str, Any]] = []
    for idx, item in enumerate(parsed, start=1):
        source_port = f"port_o{int(item['source_port_num'])}"
        target_port = f"port_o{int(item['target_port_num'])}"
        target_cfg = targets_by_port.get(target_port)
        min_coupling = getattr(target_cfg, "min_coupling", None) if target_cfg is not None else None
        max_crosstalk = getattr(target_cfg, "max_crosstalk", None) if target_cfg is not None else None
        cases.append(
            {
                "name": f"case_{idx}_{source_port}_to_{target_port}",
                "wavelength_nm": wavelength_nm,
                "source_port": source_port,
                "source_mode_index": int(item["source_mode_index"]),
                "target_port": target_port,
                "target_mode_index": int(item["target_mode_index"]),
                "min_coupling": min_coupling,
                "max_crosstalk": max_crosstalk,
                "weight": 1.0,
            }
        )

    def _sort_key(case: Dict[str, Any]) -> tuple[int, int]:
        src = _extract_port_number(str(case.get("source_port", "")))
        tgt_mode = int(case.get("target_mode_index", 0) or 0)
        return (src, tgt_mode)

    cases.sort(key=_sort_key)
    return cases


def _extract_clause_port_mode(clause: str, *, role: str) -> Dict[str, int] | None:
    role_tokens = (
        ("输入", "注入", "入射", "source", "inject", "launch", "input")
        if role == "source"
        else ("输出", "output", "to", "out")
    )
    token_expr = "|".join(re.escape(token) for token in role_tokens)
    port_expr = r"(?:port|端口)\s*(?P<port>[0-9]+)"
    mode_expr = r"(?:te|tm)\s*[_-]?\s*(?P<mode>[0-9]+)"
    no_new_port = r"(?:(?!(?:port|端口)\s*[0-9]).)"
    patterns = [
        re.compile(
            rf"{port_expr}{no_new_port}{{0,48}}?(?:{token_expr}){no_new_port}{{0,24}}?{mode_expr}",
            re.IGNORECASE,
        ),
        re.compile(
            rf"{port_expr}{no_new_port}{{0,48}}?{mode_expr}{no_new_port}{{0,24}}?(?:{token_expr})",
            re.IGNORECASE,
        ),
        re.compile(
            rf"(?:{token_expr}){no_new_port}{{0,24}}?{port_expr}{no_new_port}{{0,24}}?{mode_expr}",
            re.IGNORECASE,
        ),
        re.compile(
            rf"(?:{token_expr}){no_new_port}{{0,24}}?{mode_expr}{no_new_port}{{0,24}}?{port_expr}",
            re.IGNORECASE,
        ),
    ]
    for pattern in patterns:
        match = pattern.search(clause)
        if not match:
            continue
        port_num = _extract_port_number(match.group("port"))
        if port_num <= 0:
            continue
        try:
            mode_index = max(int(match.group("mode")), 0)
        except (TypeError, ValueError):
            mode_index = 0
        return {"port_num": port_num, "mode_index": mode_index}
    return None


def _extract_port_number(raw: str) -> int:
    m = re.search(r"(\d+)", str(raw or ""))
    if not m:
        return 0
    try:
        return int(m.group(1))
    except ValueError:
        return 0


def _extract_mode_target_index_by_port(raw_request: str) -> Dict[str, int]:
    """Extract mappings like port2->TE0 / port3->TE1 from NL prompts."""
    text = str(raw_request or "")
    if not text:
        return {}
    lowered = text.lower()
    mapping: Dict[str, int] = {}

    def _parse_mode_index(raw_mode: str) -> int | None:
        token = str(raw_mode or "").strip().lower().translate(_UNICODE_SUBSCRIPT_DIGITS)
        token = re.sub(r"[^0-9]", "", token)
        if not token:
            return None
        try:
            idx = int(token)
        except ValueError:
            return None
        return idx if idx >= 0 else None

    patterns = [
        re.compile(
            r"(?:port|端口)\s*([0-9]+)[^。；;,\n，、]{0,32}?(?:te|tm)\s*[_-]?\s*([0-9₀₁₂₃₄₅₆₇₈₉]+)",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?:te|tm)\s*[_-]?\s*([0-9₀₁₂₃₄₅₆₇₈₉]+)[^。；;,\n，、]{0,24}?(?:port|端口)\s*([0-9]+)",
            re.IGNORECASE,
        ),
    ]
    for pattern in patterns:
        for match in pattern.finditer(lowered):
            if pattern is patterns[0]:
                port_raw, mode_raw = match.group(1), match.group(2)
            else:
                mode_raw, port_raw = match.group(1), match.group(2)
            try:
                port_num = int(port_raw)
            except ValueError:
                continue
            if port_num <= 1:
                continue
            mode_idx = _parse_mode_index(mode_raw)
            if mode_idx is None:
                continue
            port_key = f"port_o{port_num}"
            # Prefer direct "port -> mode" matches over reverse/contextual
            # matches to avoid cross-clause overwrite in prompts like:
            # "port2 output TE0、port3 output TE1".
            if pattern is patterns[0] or port_key not in mapping:
                mapping[port_key] = mode_idx
    return mapping


def _extract_mode_sequence(raw_request: str) -> List[int]:
    """Extract ordered TE/TM mode indices from request text."""
    text = str(raw_request or "").lower().translate(_UNICODE_SUBSCRIPT_DIGITS)
    if not text:
        return []
    values: List[int] = []
    seen: set[int] = set()
    for match in re.finditer(r"(?:te|tm)\s*[_-]?\s*([0-9]+)", text, re.IGNORECASE):
        try:
            idx = int(match.group(1))
        except ValueError:
            continue
        if idx in seen:
            continue
        seen.add(idx)
        values.append(idx)
    return values


def _derive_objective_target_value(
    requirement: InverseDesignRequirement,
    objective_cases: List[Dict[str, Any]],
) -> float | None:
    """Map user-level targets to optimizer-level target values.

    For demux routing, Step1 often extracts percent-based coupling goals
    (for example 90).  Step5's ``demux_routing`` objective is a bounded
    flux-difference style score, so percent values are not directly usable.
    """
    if not objective_cases:
        return requirement.objective.target_value

    expected_case_values: List[float] = []
    for target in requirement.routing_targets:
        if target.min_coupling is None and target.max_crosstalk is None:
            continue
        min_coupling = float(target.min_coupling or 0.0)
        max_crosstalk = float(target.max_crosstalk or 0.0)
        expected_case_values.append(min_coupling - max_crosstalk)

    if not expected_case_values:
        return None

    mean_target = sum(expected_case_values) / len(expected_case_values)
    return round(min(max(mean_target, -2.0), 1.0), 6)


def _bound_for_parameter(name: str, value: float) -> tuple[float, float]:
    if name in {"wg_width", "wg_height", "gap", "arm_separation"}:
        delta = max(value * 0.3, 0.05)
        lower = max(value - delta, 0.05)
        upper = value + delta
        return round(lower, 4), round(upper, 4)
    delta = max(value * 0.25, 0.5)
    lower = max(value - delta, 0.1)
    upper = value + delta
    return round(lower, 4), round(upper, 4)


def _convert_doc_references(raw_refs: List[Any]) -> List[DocumentationReference]:
    converted: List[DocumentationReference] = []
    for ref in raw_refs:
        rules: List[str] = []
        for rule in getattr(ref, "rules", []) or []:
            if hasattr(rule, "rule"):
                rules.append(str(getattr(rule, "rule")))
            else:
                rules.append(str(rule))
        converted.append(
            DocumentationReference(
                url=str(getattr(ref, "url", "")),
                title=str(getattr(ref, "title", "")),
                summary=str(getattr(ref, "summary", "")),
                rules=rules,
            )
        )
    return converted


def _build_constraint_descriptions(requirement: InverseDesignRequirement) -> List[str]:
    descriptions: List[str] = []

    constraints = list(requirement.constraints or [])
    has_multi_case_targets = bool(requirement.routing_targets)

    def _as_constraint_dict(item: Any) -> Dict[str, Any]:
        if isinstance(item, dict):
            return item
        try:
            return item.model_dump()
        except Exception:
            return {}

    def _to_um(value: Any, unit: str) -> float | None:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        normalized = str(unit or "").strip().lower()
        if normalized == "nm":
            return numeric / 1000.0
        return numeric

    # --- Canonical footprint constraint (supports footprint / footprint_x+y) ---
    footprint_raw: str = ""
    footprint_x_um: float | None = None
    footprint_y_um: float | None = None
    crosstalk_constraint: Dict[str, Any] | None = None

    for item in constraints:
        payload = _as_constraint_dict(item)
        name = str(payload.get("name", "") or "").strip()
        if name == "footprint":
            footprint_raw = str(payload.get("raw_value", "") or "").strip() or str(payload.get("description", "") or "").strip()
        elif name in {"footprint_x", "footprint_y"}:
            unit = str(payload.get("unit", "") or "")
            raw = payload.get("raw_value", "")
            target_value = payload.get("target_value")
            value_um = _to_um(target_value, unit)
            if value_um is None:
                value_um = _to_um(raw, unit)
            if value_um is None:
                continue
            if name == "footprint_x":
                footprint_x_um = float(value_um)
            else:
                footprint_y_um = float(value_um)
        elif name == "crosstalk":
            crosstalk_constraint = payload

    footprint_dims: tuple[float, float] | None = None
    if footprint_raw:
        dims = _FOOTPRINT_DIM_RE.findall(str(footprint_raw))
        if len(dims) >= 2:
            values_um = []
            for val_str, unit_str in dims[:2]:
                v = _to_um(val_str, unit_str)
                if v is None:
                    continue
                values_um.append(v)
            if len(values_um) >= 2:
                footprint_dims = (float(values_um[0]), float(values_um[1]))
        else:
            nums = re.findall(r"([0-9]+(?:\.[0-9]+)?)", str(footprint_raw))
            if len(nums) >= 2:
                footprint_dims = (float(nums[0]), float(nums[1]))
    elif footprint_x_um is not None and footprint_y_um is not None:
        footprint_dims = (float(footprint_x_um), float(footprint_y_um))

    if footprint_dims is not None:
        fx, fy = footprint_dims
        component = (requirement.component_type or "").strip().lower()
        if component in {"mmi", "splitter"}:
            descriptions.append(f"a footprint of {fx:g} x {fy:g} um^2 based on 1 x MMI")
        else:
            descriptions.append(f"footprint {fx:g} x {fy:g} um^2")

    # --- Multi-case demux: prefer the crosstalk-style objective constraint ---
    if has_multi_case_targets:
        comparator = ""
        target_value = None
        unit = ""
        if isinstance(crosstalk_constraint, dict):
            comparator = str(crosstalk_constraint.get("comparator", "") or "")
            target_value = crosstalk_constraint.get("target_value")
            unit = str(crosstalk_constraint.get("unit", "") or "")
        if comparator and isinstance(target_value, (int, float)):
            descriptions.append(f"Objective target: crosstalk {comparator} {float(target_value):g} {unit}".strip())
        else:
            max_values = [
                float(target.max_crosstalk)
                for target in requirement.routing_targets
                if target.max_crosstalk is not None
            ]
            if max_values:
                percent = round(max_values[0] * 100.0, 6)
                descriptions.append(f"Objective target: crosstalk < {percent:g} %")

    # --- Other constraints: keep their original descriptions ---
    skip_names = {"footprint", "footprint_x", "footprint_y", "crosstalk", "max_iterations"}
    for item in constraints:
        payload = _as_constraint_dict(item)
        name = str(payload.get("name", "") or "").strip()
        if name in skip_names:
            continue
        desc = str(payload.get("description", "") or "").strip()
        if desc:
            descriptions.append(desc)

    # For single-case objectives, keep the legacy objective target echo.
    if not has_multi_case_targets and requirement.objective.target_value is not None and requirement.objective.comparator:
        descriptions.append(
            f"Objective target: {requirement.objective.metric} {requirement.objective.comparator} "
            f"{requirement.objective.target_value:g} {requirement.objective.unit}".strip()
        )

    return descriptions


def _derive_target_score(requirement: InverseDesignRequirement) -> float | None:
    if requirement.objective.goal != "maximize":
        return None
    if requirement.objective.metric not in {"transmission", "efficiency", "mode_overlap"}:
        return None
    if requirement.objective.target_value is None:
        return None
    return requirement.objective.target_value


def _infer_min_steps_per_wvl(doc_context: InverseDesignDocContext) -> int:
    if doc_context.guidance.mesh_advice == "mesh_override":
        return 24
    if doc_context.guidance.mesh_advice == "fine_mesh":
        return 26
    if doc_context.guidance.mesh_advice == "auto_grid":
        return 20

    for ref in doc_context.references:
        text = f"{ref.summary} " + " ".join(rule.rule for rule in ref.rules)
        lowered = text.lower()
        if "mesh override" in lowered or "adjoint" in lowered:
            return 24
        if "fine mesh" in lowered:
            return 26
    return 20


def _freq_from_wavelength_nm(wavelength_nm: float) -> float:
    return _LIGHT_SPEED_M_PER_S / (wavelength_nm * 1e-9)


def _recommended_run_time_s(domain_size_um: List[float], wavelength_nm: float, bandwidth_nm: float) -> float:
    largest_dim_m = max(domain_size_um) * 1e-6
    propagation_time_s = 3.0 * largest_dim_m * _DEFAULT_GROUP_INDEX / _LIGHT_SPEED_M_PER_S

    wl_um = wavelength_nm * 1e-3
    bw_um = max(bandwidth_nm * 1e-3, 1e-6)
    wl_min = max(wl_um - bw_um / 2, 1e-6)
    wl_max = wl_um + bw_um / 2
    fwidth = 0.5 * (_LIGHT_SPEED_M_PER_S / (wl_min * 1e-6) - _LIGHT_SPEED_M_PER_S / (wl_max * 1e-6))
    source_tail_s = 2.0 / max(fwidth, 1e9)

    return round(max(propagation_time_s + source_tail_s, 1e-13), 15)


def _is_multi_case_objective(objective_cases: List[Dict[str, Any]]) -> bool:
    return len(objective_cases) >= 2


def _safe_env_float(name: str, default: float, *, min_value: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    return max(value, min_value)


def _safe_env_int(name: str, default: int, *, min_value: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return max(value, min_value)


def _recommended_multi_case_run_time_s(base_run_time_s: float) -> float:
    multiplier = _safe_env_float(
        "INVERSE_MULTI_CASE_RUN_TIME_MULTIPLIER",
        _DEFAULT_MULTI_CASE_RUN_TIME_MULTIPLIER,
        min_value=1.0,
    )
    min_runtime_s = _safe_env_float(
        "INVERSE_MULTI_CASE_MIN_RUN_TIME_S",
        _DEFAULT_MULTI_CASE_MIN_RUN_TIME_S,
        min_value=1e-13,
    )
    return round(max(base_run_time_s * multiplier, min_runtime_s), 15)


def _default_max_iterations(is_multi_case_objective: bool) -> int:
    if is_multi_case_objective:
        return _safe_env_int(
            "INVERSE_MULTI_CASE_MAX_ITERATIONS",
            _DEFAULT_MULTI_CASE_MAX_ITERATIONS,
            min_value=_DEFAULT_MAX_ITERATIONS,
        )
    return _safe_env_int(
        "INVERSE_DEFAULT_MAX_ITERATIONS",
        _DEFAULT_MAX_ITERATIONS,
        min_value=1,
    )


def _source_type(doc_context: InverseDesignDocContext) -> str:
    source_type = doc_context.guidance.source_type
    if source_type in {"mode", "gaussian", "plane_wave"}:
        return source_type
    return "mode"


def _monitor_enabled(doc_context: InverseDesignDocContext, monitor_type: str) -> bool:
    return monitor_type in set(doc_context.guidance.recommended_monitors)


def _boundary_type(doc_context: InverseDesignDocContext) -> str:
    return "pml" if doc_context.guidance.require_pml else "periodic"


def _fetch_mcp_evidence_for_step3(
    doc_context: InverseDesignDocContext,
    memory_store: InverseDesignWorkingMemory,
    *,
    search_fn: Any | None = None,
    fetch_fn: Any | None = None,
) -> str:
    """Query MCP for step3-specific configuration evidence.

    Returns formatted documentation text to inject into the LLM prompt.
    Records MCP usage/failure in working memory.
    """
    if search_fn is None or fetch_fn is None:
        from PhotonicsAI.tools.tidy3d_tools import fetch_tidy3d_doc, search_tidy3d_docs
        if search_fn is None:
            search_fn = search_tidy3d_docs
        if fetch_fn is None:
            fetch_fn = fetch_tidy3d_doc

    component_type = doc_context.requirement.component_type or "photonic device"
    queries = [
        f"Tidy3D {component_type} inverse design configuration source monitor",
        f"Tidy3D inverse design mesh boundary run_time best practices",
    ]
    requirement = doc_context.requirement
    if len(requirement.routing_targets) >= 2:
        queries.extend(
            [
                "Tidy3D AdjointPlugin9WDM notebook",
                "Tidy3D Autograd9WDM notebook",
                "Tidy3D MultiplexingMMI notebook",
                f"Tidy3D {component_type} dual wavelength monitor placement port_o2 port_o3",
            ]
        )

    evidence_parts: List[str] = []
    mcp_urls: List[str] = []
    mcp_errors: List[str] = []

    for query in queries:
        try:
            search_result = search_fn(query=query, max_results=2)
            if not search_result.get("ok"):
                mcp_errors.append(f"MCP search returned ok=False for: {query}")
                continue
            for raw_hit in search_result.get("data", {}).get("results", []):
                url = str(raw_hit.get("url", "")).strip()
                if not url or url in mcp_urls:
                    continue
                try:
                    fetch_result = fetch_fn(url=url)
                    if not fetch_result.get("ok"):
                        continue
                    content = str(fetch_result.get("data", {}).get("content", "")).strip()
                    if content:
                        evidence_parts.append(f"[MCP Doc: {url}]\n{content[:2000]}")
                        mcp_urls.append(url)
                except Exception as fetch_exc:
                    mcp_errors.append(f"MCP fetch failed for {url}: {fetch_exc}")
        except Exception as search_exc:
            mcp_errors.append(f"MCP search error for '{query}': {search_exc}")

    # Also surface Step2 doc reference summaries + rules into the prompt.
    for ref in doc_context.references:
        rules_text = "; ".join(r.rule for r in ref.rules)
        if ref.summary or rules_text:
            evidence_parts.append(
                f"[Step2 Doc: {ref.url}]\nSummary: {ref.summary}\nRules: {rules_text}"
            )

    if mcp_errors:
        memory_store.record(
            stage="step3_mcp",
            key=doc_context.requirement.component_type,
            summary=f"Step3 MCP evidence retrieval had {len(mcp_errors)} errors, {len(mcp_urls)} docs retrieved.",
            issues=mcp_errors,
            evidence_urls=mcp_urls,
            metadata={"queries": queries, "path": "mcp_pre_fetch"},
        )

    if mcp_urls:
        memory_store.record(
            stage="step3_mcp",
            key=doc_context.requirement.component_type,
            summary=f"Step3 MCP retrieved {len(mcp_urls)} docs for LLM planner enrichment.",
            evidence_urls=mcp_urls,
            metadata={"queries": queries, "path": "mcp_main"},
        )

    return "\n\n".join(evidence_parts)


def _apply_step3_llm_decision(
    payload: Dict[str, Any],
    doc_context: InverseDesignDocContext,
    *,
    use_llm_planner: bool | None,
    llm_call_fn: Any | None,
    llm_model: str,
    memory_store: InverseDesignWorkingMemory,
    search_fn: Any | None = None,
    fetch_fn: Any | None = None,
    scenario_memory_prompt: str = "",
) -> Dict[str, Any]:
    if not _planner_enabled(use_llm_planner):
        return payload

    caller = llm_call_fn or _default_llm_call
    if caller is None:
        return payload

    simulation_raw = payload["simulation_config"]
    optimization_raw = payload["optimization_config"]
    simulation = _to_plain_dict(simulation_raw)
    optimization = _to_plain_dict(optimization_raw)

    # --- MCP evidence retrieval (pre-fetch + inject) ---
    mcp_evidence = _fetch_mcp_evidence_for_step3(
        doc_context, memory_store, search_fn=search_fn, fetch_fn=fetch_fn
    )

    prompt_parts = [
        "You are planning Step3 inverse-design configuration refinement. "
        "Return strict JSON with optional keys: source_type, boundary, monitor_types, "
        "mesh_strategy, min_steps_per_wvl, run_time_multiplier, shutoff, max_iterations, target_score, rationale.",
        f"Requirement: {doc_context.requirement.model_dump_json()}",
        f"Guidance: {doc_context.guidance.model_dump_json()}",
        f"Current simulation_config summary: {json.dumps(_summary_simulation(simulation), ensure_ascii=True)}",
        f"Current optimization_config summary: {json.dumps(_summary_optimization(optimization), ensure_ascii=True)}",
    ]
    if mcp_evidence:
        prompt_parts.append(
            "The following Tidy3D documentation evidence was retrieved via MCP. "
            "Use it to inform your configuration decisions:\n" + mcp_evidence
        )
    if scenario_memory_prompt.strip():
        prompt_parts.append(
            "The following scenario memory was recalled from prior successful/failed "
            "runs. Reuse applicable fixes and avoid repeated mistakes:\n"
            + scenario_memory_prompt.strip()
        )
    prompt = "\n".join(prompt_parts)

    sys_prompt = (
        "You are a Tidy3D inverse-design planner. "
        "Only return JSON, do not include markdown. Keep conservative values."
    )

    try:
        raw = caller(prompt, sys_prompt, llm_model)
        decision = Step3PlannerDecision.model_validate(_extract_json_object(str(raw)))
    except Exception as exc:
        memory_store.record(
            stage="step3_config_generation",
            key=doc_context.requirement.component_type,
            summary="Step3 LLM planner failed; kept deterministic generated config.",
            issues=[str(exc)],
            metadata={
                "fallback": "deterministic_generation",
                "mcp_evidence_available": bool(mcp_evidence),
            },
        )
        return payload

    tuned = dict(payload)
    tuned_sim = dict(simulation)
    tuned_domain = dict(tuned_sim["domain"])
    tuned_source = dict(tuned_sim["source"])
    tuned_opt = dict(optimization)
    tuned_termination = dict(tuned_opt["termination"])

    if decision.source_type in {"mode", "gaussian", "plane_wave"}:
        tuned_source["source_type"] = decision.source_type

    if decision.boundary in {"pml", "periodic"}:
        tuned_domain["boundary"] = {
            "x_min": decision.boundary,
            "x_max": decision.boundary,
            "y_min": decision.boundary,
            "y_max": decision.boundary,
            "z_min": decision.boundary,
            "z_max": decision.boundary,
        }

    if decision.mesh_strategy in {"auto", "uniform", "override"}:
        tuned_domain["mesh_strategy"] = decision.mesh_strategy

    if isinstance(decision.min_steps_per_wvl, int):
        tuned_domain["min_steps_per_wvl"] = max(10, min(decision.min_steps_per_wvl, 40))

    if isinstance(decision.run_time_multiplier, (int, float)):
        multiplier = max(0.5, min(float(decision.run_time_multiplier), 4.0))
        tuned_sim["run_time_s"] = float(tuned_sim["run_time_s"]) * multiplier

    if isinstance(decision.shutoff, (int, float)):
        tuned_sim["shutoff"] = max(1e-8, min(float(decision.shutoff), 1e-2))

    if isinstance(decision.max_iterations, int):
        tuned_termination["max_iterations"] = max(5, min(decision.max_iterations, 100))

    if isinstance(decision.target_score, (int, float)):
        tuned_termination["target_score"] = float(decision.target_score)

    monitor_types = [m for m in decision.monitor_types if m in {"field", "flux", "mode"}]
    if monitor_types:
        tuned_sim["monitors"] = _ensure_monitor_types(
            monitors=list(tuned_sim["monitors"]),
            monitor_types=monitor_types,
            wavelength_nm=float(tuned_sim["wavelength_nm"]),
            domain_size=list(tuned_domain["size_um"]),
            wg_height=float(tuned_sim["geometry"]["parameters"].get("wg_height", 0.22)),
            wg_width=float(tuned_sim["geometry"]["parameters"].get("wg_width", 0.5)),
            component_type=str(tuned_sim.get("component_type", "waveguide")),
            geometry_params=dict(tuned_sim.get("geometry", {}).get("parameters", {})),
        )

    tuned_sim["domain"] = tuned_domain
    tuned_sim["source"] = tuned_source
    tuned_opt["termination"] = tuned_termination
    tuned["simulation_config"] = tuned_sim
    tuned["optimization_config"] = tuned_opt

    memory_store.record(
        stage="step3_config_generation",
        key=doc_context.requirement.component_type,
        summary="Applied Step3 LLM planner decision on top of deterministic baseline.",
        metadata={
            "planner_rationale": decision.rationale,
            "decision": decision.model_dump(),
        },
    )
    return tuned


def _ensure_monitor_types(
    *,
    monitors: List[Dict[str, Any]],
    monitor_types: List[str],
    wavelength_nm: float,
    domain_size: List[float],
    wg_height: float,
    wg_width: float,
    component_type: str = "waveguide",
    geometry_params: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    existing = {str(m.get("monitor_type", "")).strip().lower() for m in monitors}
    ports = _output_port_positions(component_type, geometry_params or {}, domain_size)
    port_o2 = ports.get("port_o2", (domain_size[0] / 2 - 1.0, 0.0))
    freq0 = _freq_from_wavelength_nm(wavelength_nm)
    axis = _port_prop_axis(component_type, "port_o2")
    mode_or_flux_size = list(monitor_size_for_port(axis, wg_width, wg_height))

    if "field" in monitor_types and "field" not in existing:
        monitors.append(
            {
                "name": "field_monitor_planner",
                "monitor_type": "field",
                "center_um": [0.0, 0.0, wg_height / 2],
                "size_um": [domain_size[0] * 0.8, domain_size[1] * 0.8, 0.0],
                "freqs_hz": [freq0],
                "field_component": "Ey",
            }
        )
    if "flux" in monitor_types and "flux" not in existing:
        monitors.append(
            {
                "name": "through_flux_planner",
                "monitor_type": "flux",
                "center_um": [port_o2[0], port_o2[1], wg_height / 2],
                "size_um": mode_or_flux_size,
                "freqs_hz": [freq0],
                "metric": "transmission",
            }
        )
    if "mode" in monitor_types and "mode" not in existing:
        monitors.append(
            {
                "name": "mode_monitor_planner",
                "monitor_type": "mode",
                "center_um": [port_o2[0], port_o2[1], wg_height / 2],
                "size_um": mode_or_flux_size,
                "freqs_hz": [freq0],
                "metric": "mode_overlap",
            }
        )
    return monitors


def _summary_simulation(simulation: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "component_type": simulation.get("component_type"),
        "wavelength_nm": simulation.get("wavelength_nm"),
        "source_type": simulation.get("source", {}).get("source_type"),
        "boundary": simulation.get("domain", {}).get("boundary", {}).get("x_min"),
        "mesh_strategy": simulation.get("domain", {}).get("mesh_strategy"),
        "min_steps_per_wvl": simulation.get("domain", {}).get("min_steps_per_wvl"),
        "monitor_types": [m.get("monitor_type") for m in simulation.get("monitors", [])],
        "run_time_s": simulation.get("run_time_s"),
        "shutoff": simulation.get("shutoff"),
    }


def _summary_optimization(optimization: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "optimizer": optimization.get("optimizer"),
        "objective": optimization.get("objective", {}),
        "max_iterations": optimization.get("termination", {}).get("max_iterations"),
        "target_score": optimization.get("termination", {}).get("target_score"),
    }


def _build_step3_scenario_memory_context(
    requirement: InverseDesignRequirement,
    memory_store: InverseDesignWorkingMemory,
    *,
    limit: int = 6,
) -> Dict[str, Any]:
    fingerprint = _step3_scenario_memory_fingerprint(requirement)
    query = _step3_scenario_memory_query(requirement)
    stages = [
        "step5_diagnosis",
        "step3_config_generation",
        "step2_doc_retrieval",
    ]
    entries = []
    for stage in stages:
        entries.extend(
            memory_store.recall_by_scenario_fingerprint(
                fingerprint,
                stage=stage,
                limit=max(2, limit // 2),
                exact=False,
            )
        )
    entries = _filter_scenario_memory_entries_by_component(entries, requirement.component_type)
    if not entries:
        entries = _filter_scenario_memory_entries_by_component(
            memory_store.recall(query=query, limit=limit),
            requirement.component_type,
        )
    deduped = _dedupe_working_memory_entries(entries, limit=limit)
    return {
        "fingerprint": fingerprint,
        "query": query,
        "entry_count": len(deduped),
        "stages": [entry.stage for entry in deduped],
        "prompt": _format_step3_scenario_memory_prompt(deduped),
    }


def _step3_scenario_memory_fingerprint(requirement: InverseDesignRequirement) -> str:
    component = str(requirement.component_type or "unknown").strip().lower()
    goal = str(requirement.objective.goal or "maximize").strip().lower()
    raw_request = str(requirement.raw_request or "").lower()
    is_mode_mux = any(
        token in raw_request
        for token in ("mode multiplexer", "mode mux", "模式复用", "模式复用器", "模复用")
    )
    metric = (
        "mux_routing"
        if is_mode_mux
        else (
            "demux_routing"
            if len(requirement.routing_targets) >= 2
            else str(requirement.objective.metric or "transmission").strip().lower()
        )
    )
    ports = _collect_requirement_ports(requirement, include_text_ports=True)
    if is_mode_mux:
        ports = sorted(set([*ports, "port_o1"]))
    band = _infer_requirement_band(requirement)
    return "|".join(
        [
            f"component={component}",
            f"metric={metric}",
            f"goal={goal}",
            f"band={band}",
            f"ports={','.join(ports)}",
        ]
    )


def _step3_scenario_memory_query(requirement: InverseDesignRequirement) -> str:
    component = str(requirement.component_type or "device").strip().lower()
    goal = str(requirement.objective.goal or "maximize").strip().lower()
    metric = str(requirement.objective.metric or "transmission").strip().lower()
    raw_request = str(requirement.raw_request or "").lower()
    query_parts = [component, metric, goal]
    is_mode_mux = any(
        token in raw_request
        for token in ("mode multiplexer", "mode mux", "模式复用", "模式复用器", "模复用")
    )
    if is_mode_mux:
        query_parts.append("mode_mux")
        query_parts.extend(_collect_requirement_ports(requirement, include_text_ports=True))
    elif len(requirement.routing_targets) >= 2:
        query_parts.extend(["demux", "mode", "port_o2", "port_o3"])
    if "te0" in raw_request:
        query_parts.append("te0")
    if "te1" in raw_request:
        query_parts.append("te1")
    return " ".join(part for part in query_parts if part)


def _collect_requirement_ports(
    requirement: InverseDesignRequirement,
    *,
    include_text_ports: bool = True,
) -> List[str]:
    ports: set[str] = set()
    for target in requirement.routing_targets:
        dst = str(getattr(target, "target_port", "") or "").strip().lower()
        src = str(getattr(target, "source_port", "") or "").strip().lower()
        if dst.startswith("port_o"):
            ports.add(dst)
        if src.startswith("port_o"):
            ports.add(src)
    if include_text_ports:
        raw = str(requirement.raw_request or "").lower()
        for match in re.finditer(r"port\s*_?\s*o?\s*(\d+)", raw):
            try:
                ports.add(f"port_o{int(match.group(1))}")
            except (TypeError, ValueError):
                continue
        for match in re.finditer(r"端口\s*(\d+)", raw):
            try:
                ports.add(f"port_o{int(match.group(1))}")
            except (TypeError, ValueError):
                continue
    return sorted(ports)


def _infer_requirement_band(requirement: InverseDesignRequirement) -> str:
    wavelengths = []
    if isinstance(requirement.wavelength_nm, (int, float)):
        wavelengths.append(float(requirement.wavelength_nm))
    for value in requirement.wavelengths_nm:
        if isinstance(value, (int, float)):
            wavelengths.append(float(value))
    if not wavelengths:
        return "unknown"
    center = sum(wavelengths) / len(wavelengths)
    if 1260.0 <= center <= 1360.0:
        return "o_band"
    if 1520.0 <= center <= 1570.0:
        return "c_band"
    if 1570.0 < center <= 1625.0:
        return "l_band"
    return "custom"


def _dedupe_working_memory_entries(entries: List[Any], *, limit: int) -> List[Any]:
    output: List[Any] = []
    seen: set[str] = set()
    for entry in entries:
        signature = "|".join(
            [
                str(getattr(entry, "stage", "")),
                str(getattr(entry, "key", "")),
                str(getattr(entry, "summary", "")),
            ]
        )
        if signature in seen:
            continue
        seen.add(signature)
        output.append(entry)
        if len(output) >= max(1, int(limit)):
            break
    return output


def _filter_scenario_memory_entries_by_component(
    entries: List[Any],
    component_type: str,
) -> List[Any]:
    component = str(component_type or "").strip().lower()
    if not component:
        return entries

    filtered: List[Any] = []
    for entry in entries:
        fingerprint = str(getattr(entry, "scenario_fingerprint", "") or "").lower()
        key = str(getattr(entry, "key", "") or "").lower()
        summary = str(getattr(entry, "summary", "") or "").lower()
        if f"component={component}" in fingerprint:
            filtered.append(entry)
            continue
        if component in key or component in summary:
            filtered.append(entry)
    return filtered


def _format_step3_scenario_memory_prompt(entries: List[Any]) -> str:
    if not entries:
        return ""
    lines: List[str] = []
    for idx, entry in enumerate(entries[:6], start=1):
        lines.append(f"[ScenarioMemory {idx}] stage={entry.stage} key={entry.key}")
        lines.append(f"summary: {entry.summary}")
        if entry.issues:
            lines.append("issues: " + "; ".join(str(item) for item in entry.issues[:3]))
        if entry.proposed_fixes:
            lines.append("fixes: " + "; ".join(str(item) for item in entry.proposed_fixes[:4]))
        if entry.evidence_urls:
            lines.append("evidence: " + "; ".join(str(item) for item in entry.evidence_urls[:3]))
    return "\n".join(lines).strip()


def _planner_enabled(use_llm_planner: bool | None) -> bool:
    if use_llm_planner is not None:
        return use_llm_planner
    value = os.getenv("INVERSE_DESIGN_ENABLE_LLM_PLANNER", "0").strip().lower()
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


def _to_plain_dict(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _to_plain_dict(value.model_dump())
    if isinstance(value, dict):
        return {key: _to_plain_dict(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_plain_dict(item) for item in value]
    return value


