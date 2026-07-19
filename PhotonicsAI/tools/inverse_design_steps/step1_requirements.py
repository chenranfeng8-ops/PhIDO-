"""Step1 wrapper: parse natural-language inverse-design requirements."""

from __future__ import annotations

import json
import re
from typing import Any, Dict

from PhotonicsAI.Photon.component_detector import detect_component_type
from PhotonicsAI.Photon.llm_api import call_llm
from PhotonicsAI.tools.inverse_design_requirements import InverseDesignRequirement, parse_inverse_design_requirement


_CANONICAL_METRICS = {
    "transmission": {"transmission", "throughput", "efficiency", "透过率", "透射率", "传输效率", "total transmission", "总透过率"},
    "insertion_loss": {"insertion_loss", "insertion loss", "loss", "插入损耗", "损耗"},
    "reflection": {"reflection", "return loss", "反射", "回波损耗"},
    "q_factor": {"q_factor", "q factor", "q-factor", "quality factor", "品质因子"},
    "extinction_ratio": {"extinction_ratio", "extinction ratio", "消光比"},
    "crosstalk": {"crosstalk", "串扰"},
    "bandwidth": {"bandwidth", "带宽"},
}

_GOAL_ALIASES = {
    "maximize": {"maximize", "maximise", "最大化", "提高", "增大", "high", "higher", "increase"},
    "minimize": {"minimize", "minimise", "最小化", "降低", "减小", "low", "lower", "reduce"},
}

_COMPARATOR_ALIASES = {
    ">": {">", "gt", "greater than", "higher than", "above", "over", "高于", "大于"},
    ">=": {">=", "gte", "ge", "at least", "no less than", "not less than",
            "equals", "equal", "equal to", "=", "==", "eq",
            "不小于", "大于等于", "至少", "等于"},
    "<": {"<", "lt", "less than", "lower than", "below", "under", "低于", "小于"},
    "<=": {"<=", "lte", "le", "at most", "no more than", "not more than",
            "不大于", "小于等于", "至多"},
}

# Known component builders at Step5 — only these types have geometry constructors.
_KNOWN_BUILDERS = frozenset({
    "mmi", "splitter", "crossing", "ring_resonator", "mzi",
    "directional_coupler", "grating_coupler", "polarization_rotator",
    "y_branch", "waveguide",
})

_SYS_PROMPT_FREE = (
    "You are a photonics inverse-design requirement parser. "
    "Return JSON only. No markdown, no explanation. "
    "Use your photonics domain knowledge to identify the exact device type "
    "and optimization target."
)

_SYS_PROMPT_REFINE = (
    "You are correcting a previous photonics requirement parse. "
    "Return ONLY a JSON object with the corrected/missing fields. "
    "Do NOT repeat fields that are already correct."
)


def inverse_step1_parse_requirements(
    requirement_text: str,
    *,
    require_complete: bool = False,
    use_llm_parser: bool = True,
    llm_model: str = "gpt-5.4",
) -> Dict[str, Any]:
    """Parse requirement text into a structured Step1 object."""

    try:
        deterministic_requirement = parse_inverse_design_requirement(requirement_text)
        llm_requirement = None
        llm_error = None
        if use_llm_parser:
            try:
                llm_requirement = _parse_requirement_with_llm(requirement_text, llm_model)
            except Exception as exc:
                llm_error = str(exc)
        requirement = _merge_requirements(deterministic_requirement, llm_requirement)
        _enforce_mode_mux_source_switch_contract(requirement_text, requirement)
    except Exception as exc:
        return {"ok": False, "data": {}, "error": f"Step1 parsing failed: {exc}"}

    if require_complete and not requirement.is_complete:
        missing = ", ".join(requirement.missing_critical_fields)
        return {
            "ok": False,
            "data": {"requirement": requirement.model_dump()},
            "error": f"Step1 requirement is incomplete: {missing}",
        }

    return {
        "ok": True,
        "data": {
            "requirement": requirement.model_dump(),
            "is_complete": requirement.is_complete,
            "missing_critical_fields": list(requirement.missing_critical_fields),
            "llm_parser_used": bool(use_llm_parser),
            "llm_parser_succeeded": llm_requirement is not None,
            "llm_model": llm_model if use_llm_parser else "",
            "llm_parser_error": llm_error,
            "parser_backend": "free_semantic_with_iterative_refinement" if use_llm_parser else "deterministic_only",
        },
        "error": None,
    }


def _parse_requirement_with_llm(
    requirement_text: str,
    llm_model: str,
    *,
    max_retries: int = 2,
) -> InverseDesignRequirement:
    """Parse with free semantic LLM extraction and iterative refinement.

    Pass 1: LLM freely identifies device type, optimization target, etc.
            No ALLOWED VALUES constraint — LLM uses its domain knowledge.
    Pass 2+: If critical fields remain unresolved after 3-tier resolution,
             a diagnostic prompt tells the LLM what failed and asks for
             correction with specific guidance.
    """
    deterministic = parse_inverse_design_requirement(requirement_text)

    # Pass 1: Free semantic extraction
    raw = call_llm(
        _free_semantic_prompt(requirement_text),
        _SYS_PROMPT_FREE,
        llm_api_selection=llm_model,
    )
    payload = _extract_json_object(str(raw))
    merged = _build_merged_from_payload(payload, deterministic)

    # Iterative refinement loop — only fires when critical fields are unresolved
    for _attempt in range(max_retries):
        unresolved = _find_unresolved_fields(merged, payload)
        if not unresolved:
            break

        try:
            refinement_raw = call_llm(
                _build_refinement_prompt(requirement_text, merged, unresolved),
                _SYS_PROMPT_REFINE,
                llm_api_selection=llm_model,
            )
            patch = _extract_json_object(str(refinement_raw))
        except Exception:
            break  # Refinement failed, use best-effort

        merged = _apply_refinement_patch(merged, patch, deterministic)
        payload = {**payload, **patch}

    return merged


def _merge_requirements(
    deterministic_requirement: InverseDesignRequirement,
    llm_requirement: InverseDesignRequirement | None,
) -> InverseDesignRequirement:
    if llm_requirement is None:
        return deterministic_requirement

    merged = deterministic_requirement.model_dump()
    llm_dump = llm_requirement.model_dump()
    for key in ("component_type", "wavelength_nm"):
        if llm_dump.get(key) not in (None, ""):
            merged[key] = llm_dump[key]

    merged_objective = dict(merged.get("objective") or {})
    for obj_key, value in (llm_dump.get("objective") or {}).items():
        if value not in (None, "", []):
            merged_objective[obj_key] = value
    merged["objective"] = merged_objective

    llm_constraints = llm_dump.get("constraints")
    if isinstance(llm_constraints, list) and llm_constraints:
        merged["constraints"] = llm_constraints
    llm_targets = llm_dump.get("routing_targets")
    if isinstance(llm_targets, list) and llm_targets:
        merged["routing_targets"] = _merge_routing_targets(
            deterministic_targets=deterministic_requirement.routing_targets,
            llm_targets=llm_targets,
        )

    normalized = InverseDesignRequirement.model_validate(merged)
    # Recompute missing_critical_fields based on actual merged values
    missing = []
    if not normalized.component_type:
        missing.append("component_type")
    if not normalized.objective.metric:
        missing.append("objective.metric")
    if not normalized.objective.goal:
        missing.append("objective.goal")
    normalized.missing_critical_fields = missing
    normalized.is_complete = not missing
    return normalized


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


def _normalize_objective_payload(payload: Any, fallback: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return dict(fallback)

    normalized = dict(fallback)
    normalized["metric"] = _normalize_metric(payload.get("metric"), fallback.get("metric"))
    normalized["goal"] = _normalize_goal(payload.get("goal"), fallback.get("goal"))
    normalized["comparator"] = _normalize_comparator(payload.get("comparator"), fallback.get("comparator"))

    # Preserve LLM's original text for downstream search enrichment.
    # These are kept even when canonical normalization succeeds — the
    # raw text may contain richer semantics useful for MCP queries.
    llm_metric_raw = str(payload.get("metric", "") or "").strip()
    llm_goal_raw = str(payload.get("goal", "") or "").strip()
    normalized["llm_metric_raw"] = llm_metric_raw
    normalized["llm_goal_raw"] = llm_goal_raw

    target_value = payload.get("target_value")
    if isinstance(target_value, (int, float)):
        normalized["target_value"] = float(target_value)

    for key in ("unit", "description"):
        value = payload.get(key)
        normalized[key] = "" if value is None else str(value)

    return normalized


def _normalize_constraint_payload(payload: Any, fallback: Any) -> Any:
    if not isinstance(payload, list):
        return fallback

    normalized_constraints = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        description = item.get("description")
        if not name or not description:
            continue

        normalized_item = {
            "name": str(name),
            "description": str(description),
            "comparator": _normalize_comparator(item.get("comparator"), None),
            "target_value": float(item["target_value"]) if isinstance(item.get("target_value"), (int, float)) else None,
            "unit": "" if item.get("unit") is None else str(item.get("unit")),
            "raw_value": "" if item.get("raw_value") is None else str(item.get("raw_value")),
        }
        normalized_constraints.append(normalized_item)

    return normalized_constraints or fallback


def _normalize_component_type(value: Any, fallback: str | None) -> str | None:
    """Three-tier component type resolution.

    Tier 1: LLM value is directly a known builder name → use it.
    Tier 2: LLM value resolves to a known builder via fuzzy keyword detection.
    Tier 3: Deterministic fallback if it maps to a known builder.
    Returns None if no tier matches (triggers refinement loop or pipeline block).
    """
    from PhotonicsAI.Photon.component_detector import COMPONENT_RULES

    _priority_map = {rule[0]: rule[3] for rule in COMPONENT_RULES}

    def _pick_specific(a: str, b: str) -> str:
        return a if _priority_map.get(a, 99) <= _priority_map.get(b, 99) else b

    # Tier 1: Direct match — LLM returned an exact known builder name
    if value not in (None, ""):
        lowered = str(value).strip().lower().replace("-", "_").replace(" ", "_")
        if lowered in _KNOWN_BUILDERS:
            if fallback and fallback in _KNOWN_BUILDERS:
                return _pick_specific(lowered, fallback)
            return lowered

    # Tier 2: Fuzzy match via keyword detection
    if value not in (None, ""):
        det, conf = detect_component_type(str(value))
        if det and det != "unknown" and conf > 0 and det in _KNOWN_BUILDERS:
            if fallback and fallback in _KNOWN_BUILDERS:
                return _pick_specific(det, fallback)
            return det

    # Tier 3: Deterministic fallback (only if it maps to a known builder)
    if fallback and fallback in _KNOWN_BUILDERS:
        return fallback

    return None


def _normalize_metric(value: Any, fallback: Any) -> Any:
    if value in (None, ""):
        return fallback
    lowered = str(value).strip().lower()
    for canonical, aliases in _CANONICAL_METRICS.items():
        if lowered in aliases:
            return canonical
    return fallback or lowered


def _normalize_goal(value: Any, fallback: Any) -> Any:
    if value in (None, ""):
        return fallback
    lowered = str(value).strip().lower()
    for canonical, aliases in _GOAL_ALIASES.items():
        if lowered in aliases:
            return canonical
    return fallback


def _normalize_comparator(value: Any, fallback: Any) -> Any:
    if value in (None, ""):
        return fallback
    lowered = str(value).strip().lower()
    for canonical, aliases in _COMPARATOR_ALIASES.items():
        if lowered in aliases:
            return canonical
    return fallback


# ---------------------------------------------------------------------------
# Helper functions for free semantic extraction + iterative refinement
# ---------------------------------------------------------------------------


def _free_semantic_prompt(requirement_text: str) -> str:
    """Build free semantic extraction prompt — no ALLOWED VALUES constraint."""
    return (
        "Parse the inverse-design request into strict JSON with these keys:\n"
        "\n"
        "- component_type: identify the photonic device type using your optics knowledge.\n"
        "  Common types: mmi, y_branch, ring_resonator, grating_coupler, mzi, "
        "directional_coupler, crossing, waveguide, splitter, modulator, bragg, "
        "polarization_rotator, subwavelength_grating.\n"
        "  If the device is something else (e.g. photonic_crystal_cavity, metalens), "
        "return that. Do NOT force-fit.\n"
        "- component_type_reasoning: one sentence explaining your identification.\n"
        "- wavelength_nm: operating wavelength in nanometers (number or null).\n"
        "- wavelengths_nm: list of all wavelengths explicitly mentioned in the request "
        "(numbers in nm, sorted by mention order).\n"
        "- objective: {metric, goal, comparator, target_value, unit, description}\n"
        "  - metric: the physical quantity to optimize. Standard names: transmission, "
        "insertion_loss, reflection, q_factor, extinction_ratio, crosstalk, bandwidth. "
        "You may use other metrics if more appropriate.\n"
        "  - goal: 'maximize' or 'minimize'.\n"
        "  - comparator: one of >, >=, <, <=. For equality use >=.\n"
        "  - target_value: numeric target if mentioned, else null.\n"
        "  - unit: physical unit if applicable, else empty string.\n"
        "  - description: one-sentence description.\n"
        "- constraints: list of {name, description, comparator, target_value, unit, raw_value}\n"
        "- routing_targets: optional list for wavelength/mode routing tasks.\n"
        "  item schema: {wavelength_nm, source_port(optional), source_mode_index(optional), "
        "target_port, target_mode_index(optional), min_coupling, max_crosstalk}\n"
        "  target_port format: port_o2 / port_o3 ...\n"
        f"\nRequest: {requirement_text}"
    )


def _build_merged_from_payload(
    payload: dict,
    deterministic: InverseDesignRequirement,
) -> InverseDesignRequirement:
    """Build merged requirement from LLM payload and deterministic parse."""
    objective_payload = _normalize_objective_payload(
        payload.get("objective"), deterministic.objective.model_dump()
    )
    constraint_payload = _normalize_constraint_payload(
        payload.get("constraints"), deterministic.constraints
    )
    routing_payload = _normalize_routing_targets_payload(
        payload.get("routing_targets"), deterministic.routing_targets
    )

    llm_component_raw = str(payload.get("component_type", "") or "").strip()
    canonical_component = _normalize_component_type(
        payload.get("component_type"), deterministic.component_type
    )

    merged = {
        **deterministic.model_dump(),
        "component_type": canonical_component,
        "llm_component_type_raw": llm_component_raw,
        "wavelength_nm": (
            payload.get("wavelength_nm")
            if payload.get("wavelength_nm") is not None
            else deterministic.wavelength_nm
        ),
        "wavelengths_nm": _normalize_wavelengths_payload(
            payload.get("wavelengths_nm"),
            deterministic.wavelengths_nm,
        ),
        "objective": objective_payload,
        "constraints": constraint_payload,
        "routing_targets": routing_payload,
    }
    # Recompute missing_critical_fields based on actual merged values
    missing = []
    if not merged.get("component_type"):
        missing.append("component_type")
    obj = merged.get("objective") or {}
    if not obj.get("metric"):
        missing.append("objective.metric")
    if not obj.get("goal"):
        missing.append("objective.goal")
    merged["missing_critical_fields"] = missing
    merged["is_complete"] = not missing

    return InverseDesignRequirement.model_validate(merged)


def _enforce_mode_mux_source_switch_contract(
    requirement_text: str,
    requirement: InverseDesignRequirement,
) -> None:
    """Block silent demux fallback when user intent is source-switched mode mux."""
    lowered = str(requirement_text or "").lower()
    has_mode_mux_intent = any(
        token in lowered
        for token in ("mode multiplexer", "mode mux", "模式复用", "模式复用器", "模复用")
    )
    if not has_mode_mux_intent:
        return

    # Source-switched mux should include at least two distinct source ports
    # targeting port_o1 with explicit source_port in routing targets.
    targets = list(requirement.routing_targets or [])
    source_ports = {
        str(item.source_port or "").strip().lower()
        for item in targets
        if str(item.source_port or "").strip()
    }
    target_ports = {
        str(item.target_port or "").strip().lower()
        for item in targets
    }
    has_source_switched_cases = len(source_ports) >= 2 and target_ports == {"port_o1"}
    has_single_case_smoke_hint = any(
        token in lowered
        for token in (
            "one-case",
            "one case",
            "single-case",
            "single case",
            "smoke test",
            "冒烟",
        )
    )
    has_single_source_case = (
        len(source_ports) == 1
        and target_ports == {"port_o1"}
        and has_single_case_smoke_hint
    )
    if has_source_switched_cases or has_single_source_case:
        return

    missing = list(requirement.missing_critical_fields or [])
    missing_key = "routing_targets.source_port"
    if missing_key not in missing:
        missing.append(missing_key)
    requirement.missing_critical_fields = missing
    requirement.is_complete = False


def _find_unresolved_fields(
    merged: InverseDesignRequirement,
    payload: dict,
) -> dict:
    """Identify critical fields that failed resolution and need refinement."""
    unresolved: dict = {}

    if merged.component_type is None:
        llm_raw = str(payload.get("component_type", "") or "").strip()
        unresolved["component_type"] = {
            "llm_raw": llm_raw,
            "reason": f"no builder for '{llm_raw}'" if llm_raw else "not identified",
        }

    if merged.objective.metric is None:
        unresolved["objective.metric"] = {
            "llm_raw": str((payload.get("objective") or {}).get("metric", "") or ""),
            "reason": "not identified",
        }

    if merged.objective.goal is None:
        unresolved["objective.goal"] = {
            "llm_raw": str((payload.get("objective") or {}).get("goal", "") or ""),
            "reason": "not identified",
        }

    if merged.wavelength_nm is None:
        unresolved["wavelength_nm"] = {"reason": "not identified"}

    return unresolved


def _build_refinement_prompt(
    requirement_text: str,
    merged: InverseDesignRequirement,
    unresolved: dict,
) -> str:
    """Build a diagnostic refinement prompt targeting only unresolved fields."""
    lines = [f"Original request: {requirement_text}\n"]

    lines.append("Already extracted (do NOT change):")
    if "component_type" not in unresolved and merged.component_type:
        lines.append(f"  component_type: {merged.component_type}")
    if "wavelength_nm" not in unresolved and merged.wavelength_nm:
        lines.append(f"  wavelength_nm: {merged.wavelength_nm}")
    if "objective.metric" not in unresolved and merged.objective.metric:
        lines.append(f"  objective.metric: {merged.objective.metric}")
    if "objective.goal" not in unresolved and merged.objective.goal:
        lines.append(f"  objective.goal: {merged.objective.goal}")

    lines.append("\nFields to correct/fill:")
    for field, info in unresolved.items():
        if field == "component_type":
            llm_raw = info.get("llm_raw", "")
            if llm_raw:
                lines.append(
                    f"  component_type: You returned '{llm_raw}' but the system "
                    f"has no geometry builder for it. Available builders: "
                    f"{', '.join(sorted(_KNOWN_BUILDERS))}. "
                    f"Pick the closest match or explain why none fits."
                )
            else:
                lines.append(
                    f"  component_type: Could not identify the device. "
                    f"Available: {', '.join(sorted(_KNOWN_BUILDERS))}."
                )
        elif field == "objective.metric":
            lines.append(
                "  objective.metric: Not identified. Common metrics: "
                "transmission, insertion_loss, reflection, q_factor, "
                "extinction_ratio, crosstalk, bandwidth."
            )
        elif field == "objective.goal":
            lines.append("  objective.goal: Must be 'maximize' or 'minimize'.")
        elif field == "wavelength_nm":
            lines.append("  wavelength_nm: Not detected. Provide in nm.")

    lines.append("\nReturn ONLY a JSON patch with the corrected fields.")
    return "\n".join(lines)


def _apply_refinement_patch(
    current: InverseDesignRequirement,
    patch: dict,
    deterministic: InverseDesignRequirement,
) -> InverseDesignRequirement:
    """Apply a refinement patch to the current merged result."""
    merged = current.model_dump()

    # Patch component_type
    if "component_type" in patch:
        patched_ct = _normalize_component_type(
            patch["component_type"], deterministic.component_type
        )
        if patched_ct is not None:
            merged["component_type"] = patched_ct
            merged["llm_component_type_raw"] = str(
                patch.get("component_type", "") or ""
            ).strip()

    # Patch wavelength
    if "wavelength_nm" in patch and patch["wavelength_nm"] is not None:
        try:
            merged["wavelength_nm"] = float(patch["wavelength_nm"])
        except (TypeError, ValueError):
            pass
    if "wavelengths_nm" in patch:
        merged["wavelengths_nm"] = _normalize_wavelengths_payload(
            patch.get("wavelengths_nm"),
            merged.get("wavelengths_nm") or [],
        )

    # Patch objective fields — accept nested {"objective": {...}} or flat {"metric": "..."}
    obj = dict(merged.get("objective") or {})
    p_obj = patch.get("objective") if isinstance(patch.get("objective"), dict) else {}
    if not p_obj:
        p_obj = {}
        for flat_key in ("metric", "goal", "comparator"):
            if flat_key in patch:
                p_obj[flat_key] = patch[flat_key]

    if "metric" in p_obj:
        val = _normalize_metric(p_obj["metric"], obj.get("metric"))
        if val:
            obj["metric"] = val
    if "goal" in p_obj:
        val = _normalize_goal(p_obj["goal"], obj.get("goal"))
        if val:
            obj["goal"] = val
    if "comparator" in p_obj:
        val = _normalize_comparator(p_obj["comparator"], obj.get("comparator"))
        if val:
            obj["comparator"] = val
    merged["objective"] = obj
    if "routing_targets" in patch:
        merged["routing_targets"] = _normalize_routing_targets_payload(
            patch.get("routing_targets"),
            merged.get("routing_targets") or [],
        )

    # Recompute missing fields
    missing = []
    if not merged.get("component_type"):
        missing.append("component_type")
    obj_dict = merged.get("objective") or {}
    if not obj_dict.get("metric"):
        missing.append("objective.metric")
    if not obj_dict.get("goal"):
        missing.append("objective.goal")
    merged["missing_critical_fields"] = missing
    merged["is_complete"] = not missing

    return InverseDesignRequirement.model_validate(merged)


def _normalize_routing_targets_payload(payload: Any, fallback: Any) -> Any:
    if not isinstance(payload, list):
        return fallback

    out = []
    for item in payload:
        if not isinstance(item, dict):
            try:
                item = item.model_dump()
            except Exception:
                continue
        try:
            wl = float(item.get("wavelength_nm"))
        except (TypeError, ValueError):
            continue

        def _normalize_port(raw_port: Any) -> str:
            port_raw = str(raw_port or "").strip().lower()
            if not port_raw:
                return ""
            if port_raw.startswith("port_o"):
                return port_raw
            m = re.search(r"(\d+)", port_raw)
            if not m:
                return ""
            return f"port_o{int(m.group(1))}"

        target_port = _normalize_port(item.get("target_port"))
        if not target_port:
            continue
        source_port = _normalize_port(item.get("source_port"))

        def _mode_idx(key: str) -> int:
            raw = item.get(key)
            if raw in (None, ""):
                return 0
            try:
                idx = int(float(raw))
            except (TypeError, ValueError):
                return 0
            return max(idx, 0)

        def _frac(key: str) -> float | None:
            value = item.get(key)
            if value in (None, ""):
                return None
            raw = str(value).strip().lower()
            try:
                num = float(value)
            except (TypeError, ValueError):
                return None

            is_percent_literal = "%" in raw or "percent" in raw
            if key == "max_crosstalk":
                # Crosstalk in prompts is commonly stated in percent.  Treat
                # 1 as 1% (0.01), while preserving already-normalized values.
                if num >= 1.0 or is_percent_literal:
                    num /= 100.0
            else:
                # Coupling is usually expressed either as fraction (0~1) or
                # percentage (>1 or explicit percent literal).
                if num > 1.0 or is_percent_literal:
                    num /= 100.0
            return min(max(num, 0.0), 1.0)

        out.append(
            {
                "wavelength_nm": wl,
                "source_port": source_port,
                "source_mode_index": _mode_idx("source_mode_index"),
                "target_port": target_port,
                "target_mode_index": _mode_idx("target_mode_index"),
                "min_coupling": _frac("min_coupling"),
                "max_crosstalk": _frac("max_crosstalk"),
            }
        )
    return out or fallback


def _merge_routing_targets(
    *,
    deterministic_targets: Any,
    llm_targets: Any,
) -> Any:
    """Merge routing targets with deterministic extraction as the base."""
    deterministic_list = _normalize_routing_targets_payload(deterministic_targets, [])
    llm_list = _normalize_routing_targets_payload(llm_targets, [])

    if not deterministic_list:
        return llm_list
    if not llm_list:
        return deterministic_list

    merged: Dict[tuple[float, str], Dict[str, Any]] = {}
    for item in deterministic_list:
        key = (
            round(float(item.get("wavelength_nm", 0.0)), 6),
            str(item.get("source_port", "")),
            str(item.get("target_port", "")),
        )
        merged[key] = dict(item)
    for item in llm_list:
        key = (
            round(float(item.get("wavelength_nm", 0.0)), 6),
            str(item.get("source_port", "")),
            str(item.get("target_port", "")),
        )
        if key not in merged:
            merged[key] = dict(item)
            continue
        existing = merged[key]
        for mode_key in ("source_mode_index", "target_mode_index"):
            try:
                existing_mode = int(float(existing.get(mode_key) or 0))
            except (TypeError, ValueError):
                existing_mode = 0
            try:
                incoming_mode = int(float(item.get(mode_key) or 0))
            except (TypeError, ValueError):
                incoming_mode = 0
            if existing_mode <= 0 and incoming_mode > 0:
                existing[mode_key] = incoming_mode
        for score_key in ("min_coupling", "max_crosstalk"):
            if existing.get(score_key) in (None, "") and item.get(score_key) not in (None, ""):
                existing[score_key] = item.get(score_key)

    return list(merged.values())


def _normalize_wavelengths_payload(payload: Any, fallback: Any) -> Any:
    if not isinstance(payload, list):
        return fallback
    out = []
    seen = set()
    for item in payload:
        try:
            wl = float(item)
        except (TypeError, ValueError):
            continue
        key = round(wl, 6)
        if key in seen:
            continue
        seen.add(key)
        out.append(wl)
    return out or fallback
