"""Step3 wrapper: generate strict simulation and optimization configuration."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

from PhotonicsAI.tools.inverse_design_config_generation import generate_inverse_design_config
from PhotonicsAI.tools.inverse_design_doc_context import InverseDesignDocContext
from PhotonicsAI.tools.inverse_design_replan import apply_patch_actions


def inverse_step3_generate_config(
    doc_context: Dict[str, Any] | None = None,
    *,
    use_llm_planner: bool = True,
    llm_model: str = "gpt-5.4",
    selected_rollback_step: str | None = None,
    recovery_context: Dict[str, Any] | None = None,
    workflow_state: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build Step3 config bundle from Step2 context."""

    try:
        effective_context = doc_context
        if effective_context is None and workflow_state:
            effective_context = (
                workflow_state.get("step2", {})
                .get("data", {})
                .get("doc_context")
            )
        if effective_context is None:
            effective_context = _build_degraded_context_from_step1(workflow_state)
        if effective_context is None:
            raise ValueError("missing Step2 doc context input for Step3")
        effective_context = _normalize_doc_context_payload(effective_context)
        effective_context = _merge_step1_requirement_context(
            effective_context,
            workflow_state=workflow_state,
        )

        typed_context = InverseDesignDocContext.model_validate(effective_context)
        bundle = generate_inverse_design_config(
            typed_context,
            use_llm_planner=use_llm_planner,
            llm_model=llm_model,
        )
        bundle_payload = bundle.model_dump()
        contract_error = _validate_mode_mux_contract(
            typed_context.model_dump(),
            bundle_payload,
        )
        if contract_error:
            return {"ok": False, "data": {}, "error": f"Step3 config generation failed: {contract_error}"}
        applied_patch_paths = []
        if isinstance(recovery_context, dict):
            applied_patch_paths = apply_patch_actions(
                bundle_payload,
                recovery_context.get("patch_actions", []),
            )
            # W13 fix: recovery patch_actions may delete or corrupt flux
            # monitors.  Re-inject them from the original bundle when missing.
            _ensure_flux_monitors_present(bundle_payload, bundle)
    except Exception as exc:
        return {"ok": False, "data": {}, "error": f"Step3 config generation failed: {exc}"}

    return {
        "ok": True,
        "data": {
            "config_bundle": bundle_payload,
            "component_type": bundle_payload.get("simulation_config", {}).get("component_type", bundle.simulation_config.component_type),
            "objective_metric": (
                bundle_payload.get("optimization_config", {})
                .get("objective", {})
                .get("metric", bundle.optimization_config.objective.metric)
            ),
            "input_dependencies": ["step2.doc_context"],
            "recovery_context_used": bool(recovery_context),
            "selected_rollback_step": selected_rollback_step or "",
            "applied_patch_paths": applied_patch_paths,
        },
        "error": None,
    }


def _build_degraded_context_from_step1(
    workflow_state: Dict[str, Any] | None,
) -> Dict[str, Any] | None:
    if not isinstance(workflow_state, dict):
        return None
    step1_requirement = (
        workflow_state.get("step1", {})
        .get("data", {})
        .get("requirement")
    )
    if not isinstance(step1_requirement, dict) or not step1_requirement:
        return None
    return {
        "requirement": deepcopy(step1_requirement),
        "queries": [],
        "references": [],
        "guidance": {
            "source_type": "mode",
            "require_pml": True,
            "recommended_monitors": ["field", "flux", "mode"],
            "mesh_advice": "",
            "inverse_design_hint": "Step3 degraded context synthesized from Step1.",
        },
    }


def _normalize_doc_context_payload(doc_context: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(doc_context, dict):
        return doc_context
    normalized = deepcopy(doc_context)
    if not isinstance(normalized.get("queries"), list):
        normalized["queries"] = []
    if not isinstance(normalized.get("references"), list):
        normalized["references"] = []
    guidance = normalized.get("guidance")
    if not isinstance(guidance, dict):
        guidance = {}
    guidance.setdefault("source_type", "mode")
    guidance.setdefault("require_pml", True)
    guidance.setdefault("recommended_monitors", ["field", "flux", "mode"])
    guidance.setdefault("mesh_advice", "")
    guidance.setdefault("inverse_design_hint", "")
    normalized["guidance"] = guidance
    return normalized


def _validate_mode_mux_contract(
    doc_context_payload: Dict[str, Any],
    bundle_payload: Dict[str, Any],
) -> str:
    requirement = doc_context_payload.get("requirement", {}) if isinstance(doc_context_payload, dict) else {}
    raw_request = str(requirement.get("raw_request", "") or "").lower()
    has_mode_mux_intent = any(
        token in raw_request
        for token in ("mode multiplexer", "mode mux", "模式复用", "模式复用器", "模复用")
    )
    if not has_mode_mux_intent:
        return ""

    objective_metric = (
        bundle_payload.get("optimization_config", {})
        .get("objective", {})
        .get("metric", "")
    )
    objective_cases = (
        bundle_payload.get("optimization_config", {})
        .get("objective_cases", [])
    )
    has_mux_case = any(
        str(item.get("source_port", "")).strip().lower() not in {"", "port_o1"}
        and str(item.get("target_port", "")).strip().lower() == "port_o1"
        for item in objective_cases
        if isinstance(item, dict)
    )
    if objective_metric != "mux_routing" or not has_mux_case:
        return (
            "mode-mux contract violation: expected mux_routing with source-switched "
            "objective_cases targeting port_o1."
        )
    return ""


def _merge_step1_requirement_context(
    doc_context: Dict[str, Any],
    *,
    workflow_state: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Merge deterministic Step1 requirement fields into Step2 doc context.

    Step2 retrieval may occasionally return a shortened requirement payload
    (for example dropping routing_targets). Step3 must preserve Step1 as the
    single source of truth for routing/mode intent.
    """

    if not isinstance(doc_context, dict):
        return doc_context
    if not isinstance(workflow_state, dict):
        return doc_context

    step1_requirement = (
        workflow_state.get("step1", {})
        .get("data", {})
        .get("requirement")
    )
    if not isinstance(step1_requirement, dict):
        return doc_context

    merged_context = deepcopy(doc_context)
    requirement = merged_context.get("requirement")
    if not isinstance(requirement, dict):
        requirement = {}

    merged_requirement = dict(requirement)
    step1_targets = step1_requirement.get("routing_targets")
    current_targets = requirement.get("routing_targets")
    if isinstance(step1_targets, list) and step1_targets:
        current_count = len(current_targets) if isinstance(current_targets, list) else 0
        if current_count < len(step1_targets):
            merged_requirement["routing_targets"] = deepcopy(step1_targets)

    if step1_requirement.get("wavelengths_nm") and not requirement.get("wavelengths_nm"):
        merged_requirement["wavelengths_nm"] = deepcopy(step1_requirement.get("wavelengths_nm"))
    if step1_requirement.get("wavelength_nm") and not requirement.get("wavelength_nm"):
        merged_requirement["wavelength_nm"] = step1_requirement.get("wavelength_nm")
    if step1_requirement.get("raw_request") and not requirement.get("raw_request"):
        merged_requirement["raw_request"] = step1_requirement.get("raw_request")

    merged_context["requirement"] = merged_requirement
    return merged_context


def _ensure_flux_monitors_present(
    bundle_payload: Dict[str, Any],
    original_bundle: Any,
) -> None:
    """Re-inject flux monitors if recovery patches deleted them (W13 fix)."""
    monitors = (
        bundle_payload
        .get("simulation_config", {})
        .get("monitors", [])
    )
    has_flux = any(
        m.get("monitor_type") == "flux"
        for m in monitors
        if isinstance(m, dict)
    )
    if has_flux:
        return
    # Restore flux monitors from the pre-patch bundle.
    for mon in original_bundle.simulation_config.monitors:
        if mon.monitor_type == "flux":
            monitors.append(mon.model_dump())
    bundle_payload.setdefault("simulation_config", {})["monitors"] = monitors
