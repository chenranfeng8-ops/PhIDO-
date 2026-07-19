"""Structured parsing for inverse-design natural-language requirements."""

from __future__ import annotations

import re
from typing import Dict, List, Literal

from pydantic import BaseModel, ConfigDict, Field

from PhotonicsAI.Photon.component_detector import (
    detect_component_type,
    get_component_display_name,
)


class StrictModel(BaseModel):
    """Base model that rejects unknown fields to keep parser output stable."""

    model_config = ConfigDict(extra="forbid")


class RequirementObjective(StrictModel):
    """Primary optimization target extracted from user intent."""

    metric: str | None = None
    goal: Literal["maximize", "minimize"] | None = None
    comparator: Literal[">", ">=", "<", "<="] | None = None
    target_value: float | None = None
    unit: str = ""
    description: str = ""
    # LLM raw outputs — preserved even when canonical normalization fails,
    # so downstream steps (e.g. Step2 search queries) can use the richer text.
    llm_metric_raw: str = ""
    llm_goal_raw: str = ""


class RequirementConstraint(StrictModel):
    """Explicit design or fabrication constraint from the requirement text."""

    name: str
    description: str
    comparator: Literal[">", ">=", "<", "<="] | None = None
    target_value: float | None = None
    unit: str = ""
    raw_value: str = ""


class RequirementRoutingTarget(StrictModel):
    """Routing target for multi-wavelength demux-like objectives."""

    wavelength_nm: float
    source_port: str = ""
    source_mode_index: int = Field(default=0, ge=0)
    target_port: str
    target_mode_index: int = Field(default=0, ge=0)
    min_coupling: float | None = None
    max_crosstalk: float | None = None


class InverseDesignRequirement(StrictModel):
    """Structured target object used by implementation step 9.1 item 1."""

    raw_request: str
    component_type: str | None = None
    component_display_name: str = ""
    component_confidence: float = 0.0
    wavelength_nm: float | None = None
    wavelengths_nm: List[float] = Field(default_factory=list)
    objective: RequirementObjective = Field(default_factory=RequirementObjective)
    objective_function: str = ""
    constraints: List[RequirementConstraint] = Field(default_factory=list)
    routing_targets: List[RequirementRoutingTarget] = Field(default_factory=list)
    missing_critical_fields: List[str] = Field(default_factory=list)
    is_complete: bool = False
    # LLM raw output — preserved even when canonical mapping fails.
    # Used by Step2 to build richer MCP search queries.
    llm_component_type_raw: str = ""


_CLAUSE_SPLIT_RE = re.compile(
    r"\s*(?:,|，|;|；|、|\bwhile\b|\bsubject to\b|\bwith\b)\s*",
    re.IGNORECASE,
)
_METRIC_RULES = [
    ("insertion_loss", ["insertion loss", "loss", "插入损耗", "损耗"], "minimize", "dB"),
    ("transmission", ["transmission", "throughput", "efficiency", "透过率", "透射率", "传输效率"], "maximize", ""),
    ("reflection", ["reflection", "return loss", "反射", "回波损耗"], "minimize", "dB"),
    ("q_factor", ["q factor", "q-factor", "quality factor", "品质因子"], "maximize", ""),
    ("extinction_ratio", ["extinction ratio", "消光比"], "maximize", "dB"),
    ("crosstalk", ["crosstalk", "串扰"], "minimize", "dB"),
    ("bandwidth", ["bandwidth", "带宽"], "maximize", "nm"),
]
_CONSTRAINT_HINTS = {
    "minimum_feature_size": ["minimum feature size", "feature size", "min feature"],
    "footprint": ["footprint", "device area", "size limit", "方块", "初始结构", "尺寸"],
    "crosstalk": ["crosstalk", "串扰"],
    "fabrication": ["fabrication", "fab rule", "manufacturing"],
    "symmetry": ["symmetry", "symmetric"],
    "thickness": ["thickness", "etch depth"],
}
_COMPARATOR_RE = [
    (re.compile(r"(?:at least|no less than|not less than|greater than or equal to|大于等于|不小于|>=)\s*([0-9]+(?:\.[0-9]+)?)(?:\s*(nm|um|db|%))?", re.IGNORECASE), ">="),
    (re.compile(r"(?:above|over|greater than|more than|大于|高于|>)\s*([0-9]+(?:\.[0-9]+)?)(?:\s*(nm|um|db|%))?", re.IGNORECASE), ">"),
    (re.compile(r"(?:at most|no more than|less than or equal to|小于等于|不大于|<=)\s*([0-9]+(?:\.[0-9]+)?)(?:\s*(nm|um|db|%))?", re.IGNORECASE), "<="),
    (re.compile(r"(?:below|under|less than|fewer than|小于|低于|<)\s*([0-9]+(?:\.[0-9]+)?)(?:\s*(nm|um|db|%))?", re.IGNORECASE), "<"),
]
_UNIT_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*(\u00b5m|\u03bcm|nm|um|db|%)", re.IGNORECASE)
_FOOTPRINT_PAIR_RE = re.compile(
    r"([0-9]+(?:\.[0-9]+)?)\s*(?:(\u00b5m|\u03bcm|um|nm))?\s*[\*xX\u00d7]\s*"
    r"([0-9]+(?:\.[0-9]+)?)\s*(\u00b5m|\u03bcm|um|nm)(?:\^?2|\u00b2)?",
    re.IGNORECASE,
)
_FOOTPRINT_RE = re.compile(
    r"([0-9]+(?:\.[0-9]+)?\s*(?:\u00b5m|\u03bcm|um|nm))\s*[\*xX×]\s*([0-9]+(?:\.[0-9]+)?\s*(?:\u00b5m|\u03bcm|um|nm))",
    re.IGNORECASE,
)
_ROUTING_TARGET_RE = re.compile(
    r"([0-9]+(?:\.[0-9]+)?)\s*nm[^.]{0,120}?(?:to|->)\s*port\s*([0-9]+)",
    re.IGNORECASE,
)
_COUPLING_MIN_RE = re.compile(
    r"(?:>|>=|above|over|at least|no less than|大于|高于|不小于|大于等于)\s*([0-9]+(?:\.[0-9]+)?)\s*%",
    re.IGNORECASE,
)
_COUPLING_ASSIGN_RE = re.compile(
    r"(?:min(?:imum)?\s*coupling|min[_\s-]*coupling|耦合率(?:下限)?)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)\s*%",
    re.IGNORECASE,
)
_CROSSTALK_MAX_RE = re.compile(
    r"(?:crosstalk|串扰)[^.。]{0,80}?(?:<|<=|below|under|at most|no more than|小于|低于|不大于|小于等于)\s*([0-9]+(?:\.[0-9]+)?)\s*%",
    re.IGNORECASE,
)
_CROSSTALK_ASSIGN_RE = re.compile(
    r"(?:max(?:imum)?\s*crosstalk|max[_\s-]*crosstalk|串扰(?:上限)?)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)\s*%",
    re.IGNORECASE,
)
_MODE_MUX_INTENT_TOKENS = (
    "mode multiplexer",
    "mode mux",
    "模式复用",
    "模式复用器",
    "模复用",
)


def parse_inverse_design_requirement(requirement_text: str) -> InverseDesignRequirement:
    """Parse a natural-language inverse-design request into a target object."""

    normalized = " ".join((requirement_text or "").split())
    # Normalize grouped numbers like "1,300 nm" -> "1300 nm".
    normalized = re.sub(r"(?<=\d),(?=\d)", "", normalized)
    component_type, confidence = detect_component_type(normalized)
    if component_type == "unknown":
        component_type = None
        display_name = ""
        confidence = 0.0
    else:
        display_name = get_component_display_name(component_type)

    wavelengths_nm = _extract_all_wavelengths_nm(normalized)
    wavelength_nm = _extract_wavelength_nm(normalized)
    if wavelength_nm is None and wavelengths_nm:
        wavelength_nm = wavelengths_nm[0]
    objective = _extract_objective(normalized)
    constraints = _extract_constraints(normalized)
    routing_targets = _extract_routing_targets(normalized)
    if not objective.metric or not objective.goal:
        inferred_objective = _infer_routing_objective(normalized, routing_targets)
        if inferred_objective is not None:
            objective = inferred_objective

    missing = []
    if not component_type:
        missing.append("component_type")
    if not objective.metric:
        missing.append("objective.metric")
    if not objective.goal:
        missing.append("objective.goal")

    objective_function = _build_objective_function(objective, wavelength_nm)

    return InverseDesignRequirement(
        raw_request=normalized,
        component_type=component_type,
        component_display_name=display_name,
        component_confidence=confidence,
        wavelength_nm=wavelength_nm,
        wavelengths_nm=wavelengths_nm,
        objective=objective,
        objective_function=objective_function,
        constraints=constraints,
        routing_targets=routing_targets,
        missing_critical_fields=missing,
        is_complete=not missing,
    )


def inverse_design_requirement_schema() -> Dict[str, object]:
    """Return JSON schema for the requirement-target object."""

    return InverseDesignRequirement.model_json_schema()


def require_complete_inverse_design_requirement(
    requirement_text: str,
) -> InverseDesignRequirement:
    """Parse a requirement and raise if critical fields are missing."""

    requirement = parse_inverse_design_requirement(requirement_text)
    if requirement.is_complete:
        return requirement

    missing = ", ".join(requirement.missing_critical_fields)
    raise ValueError(f"Missing critical fields: {missing}")


_WAVELENGTH_HINT_RE = re.compile(
    r"(?:wavelength|波长|工作波长|工作中心波长|中心波长|波长为|wvl|lambda|operating(?:\s+at)?|working(?:\s+wavelength)?|center(?:\s+wavelength)?)"
    r"\D{0,14}([0-9]+(?:\.[0-9]+)?)\s*(\u00b5m|\u03bcm|um|nm)",
    re.IGNORECASE,
)

_WAVELENGTH_CONTEXT_TOKENS = (
    "wavelength",
    "lambda",
    "operat",
    "working wavelength",
    "center wavelength",
    "波长",
    "工作波长",
    "工作中心波长",
    "中心波长",
    "波长为",
)

_DIMENSION_CONTEXT_TOKENS = (
    "width",
    "waveguide width",
    "port width",
    "single-mode waveguide width",
    "multimode waveguide width",
    "feature size",
    "thickness",
    "height",
    "宽度",
    "波导宽度",
    "端口宽度",
    "单模波导宽度",
    "多模波导宽度",
    "入射单模波导宽度",
    "出射多模波导宽度",
    "线宽",
    "厚度",
    "高度",
)

_DIMENSION_CONTEXT_RE = re.compile(
    r"(?:width|waveguide\s+width|port\s+width|single-mode\s+waveguide\s+width|multimode\s+waveguide\s+width|"
    r"feature\s+size|thickness|height|宽度|波导宽度|端口宽度|单模波导宽度|多模波导宽度|"
    r"入射单模波导宽度|出射多模波导宽度|线宽|厚度|高度)\s*$",
    re.IGNORECASE,
)
_DIMENSION_TRAILING_RE = re.compile(
    r"^\s*(?:wide|width|thick(?:ness)?|height|long|宽|宽度|厚|高度)",
    re.IGNORECASE,
)


def _extract_wavelength_nm(text: str) -> float | None:
    # Priority: explicit wavelength hints in local context.
    hint_match = _WAVELENGTH_HINT_RE.search(text)
    if hint_match:
        return _to_nm(float(hint_match.group(1)), hint_match.group(2))

    # Fallback: score all unit-bearing candidates and pick the most likely
    # wavelength value. This avoids footprint dimensions such as 2.5x2.5 um.
    candidates: List[tuple[float, int, float]] = []
    for idx, match in enumerate(_UNIT_RE.finditer(text)):
        if _in_footprint_span(match.start(), match.end(), text):
            continue
        if _is_dimension_unit_match(text, match.start(), match.end()):
            continue
        value = float(match.group(1))
        unit = match.group(2).lower()
        nm = _to_nm(value, unit)
        if nm is None or not (100.0 <= nm <= 20000.0):
            continue

        score = 0.0
        if unit == "nm":
            score += 2.0
        if 300.0 <= nm <= 2000.0:
            score += 1.0

        context = _unit_context_window(text, match.start(), match.end())
        if any(token in context for token in _WAVELENGTH_CONTEXT_TOKENS):
            score += 2.0

        candidates.append((score, idx, nm))

    if candidates:
        # Higher score first, then earlier occurrence in text.
        candidates.sort(key=lambda item: (-item[0], item[1]))
        return candidates[0][2]
    return None


def _to_nm(value: float, unit: str) -> float | None:
    u = unit.lower()
    if u == "nm":
        return value
    if u in {"um", "\u00b5m", "\u03bcm"}:
        return value * 1000.0
    return None


def _extract_all_wavelengths_nm(text: str) -> List[float]:
    values: List[float] = []
    seen = set()
    for match in _WAVELENGTH_HINT_RE.finditer(text):
        nm = _to_nm(float(match.group(1)), match.group(2))
        if nm is None or not (100.0 <= nm <= 20000.0):
            continue
        key = round(nm, 6)
        if key in seen:
            continue
        seen.add(key)
        values.append(nm)
    has_explicit_wavelength_hints = bool(values)
    for match in _UNIT_RE.finditer(text):
        if _in_footprint_span(match.start(), match.end(), text):
            continue
        if _is_dimension_unit_match(text, match.start(), match.end()):
            continue
        if has_explicit_wavelength_hints:
            context = _unit_context_window(text, match.start(), match.end())
            has_wavelength_context = any(
                token in context for token in _WAVELENGTH_CONTEXT_TOKENS
            )
            has_routing_context = bool(
                re.search(r"(?:port|端口|->|to|输出|输入|target)", context, re.IGNORECASE)
            )
            if not has_wavelength_context and not has_routing_context:
                continue
        nm = _to_nm(float(match.group(1)), match.group(2))
        if nm is None or not (100.0 <= nm <= 20000.0):
            continue
        key = round(nm, 6)
        if key in seen:
            continue
        seen.add(key)
        values.append(nm)
    return values


def _extract_objective(text: str) -> RequirementObjective:
    clauses = [clause.strip() for clause in _CLAUSE_SPLIT_RE.split(text) if clause.strip()]

    for clause in clauses or [text]:
        lowered = clause.lower()
        for metric, keywords, default_goal, default_unit in _METRIC_RULES:
            if not any(keyword in lowered for keyword in keywords):
                continue

            goal = _detect_goal(lowered, default_goal)
            comparator, target_value, target_unit = _extract_comparator_and_value(lowered)
            unit = target_unit or default_unit

            return RequirementObjective(
                metric=metric,
                goal=goal,
                comparator=comparator,
                target_value=target_value,
                unit=unit,
                description=clause,
            )

    return RequirementObjective(description=text)


def _infer_routing_objective(
    text: str,
    routing_targets: List[RequirementRoutingTarget],
) -> RequirementObjective | None:
    if not routing_targets:
        return None

    lowered = str(text or "").lower()
    has_mode_mux_case = any(str(item.source_port or "").strip() for item in routing_targets)
    if has_mode_mux_case:
        return RequirementObjective(
            metric="transmission",
            goal="maximize",
            description="maximize target-mode transmission inferred from mode-mux routing request",
        )

    if any(
        token in lowered
        for token in (
            "demux",
            "demultiplexer",
            "routing",
            "coupling",
            "transmission",
            "throughput",
            "efficiency",
            "透过率",
            "透射率",
            "传输效率",
        )
    ):
        return RequirementObjective(
            metric="transmission",
            goal="maximize",
            description="maximize routed transmission inferred from routing request",
        )
    return None


def _extract_constraints(text: str) -> List[RequirementConstraint]:
    constraints: List[RequirementConstraint] = []
    clauses = [clause.strip() for clause in _CLAUSE_SPLIT_RE.split(text) if clause.strip()]

    for clause in clauses:
        lowered = clause.lower()
        matched_names = set()
        for name, hints in _CONSTRAINT_HINTS.items():
            if not any(hint in lowered for hint in hints):
                continue
            if name in matched_names:
                continue

            comparator, target_value, target_unit = _extract_comparator_and_value(lowered)
            unit = target_unit
            raw_value = ""
            if name == "footprint":
                raw_value = _extract_footprint_raw_value(clause)
            elif name == "crosstalk":
                xtalk_match = _CROSSTALK_MAX_RE.search(clause)
                if xtalk_match:
                    comparator = "<"
                    target_value = float(xtalk_match.group(1))
                    unit = "%"
                    raw_value = xtalk_match.group(0)

            constraints.append(
                RequirementConstraint(
                    name=name,
                    description=clause,
                    comparator=comparator,
                    target_value=target_value,
                    unit=unit,
                    raw_value=raw_value,
                )
            )
            matched_names.add(name)

    has_any_footprint = any(
        constraint.name in {"footprint", "footprint_x", "footprint_y"}
        for constraint in constraints
    )
    if not has_any_footprint:
        raw_value = _extract_footprint_raw_value(text)
        if raw_value:
            constraints.append(
                RequirementConstraint(
                    name="footprint",
                    description=f"footprint {raw_value}",
                    comparator="<=",
                    target_value=None,
                    unit="",
                    raw_value=raw_value,
                )
            )

    # Deduplicate by constraint name while preserving order.
    deduped: List[RequirementConstraint] = []
    seen: set[str] = set()
    for item in constraints:
        if item.name in seen:
            continue
        deduped.append(item)
        seen.add(item.name)
    return deduped


def _detect_goal(text: str, default_goal: Literal["maximize", "minimize"]) -> Literal["maximize", "minimize"]:
    if any(token in text for token in ["maximize", "maximise", "high", "higher", "increase", "最大化", "提高", "增大"]):
        return "maximize"
    if any(token in text for token in ["minimize", "minimise", "low", "lower", "reduce", "最小化", "降低", "减小"]):
        return "minimize"
    return default_goal


def _extract_comparator_and_value(
    text: str,
) -> tuple[Literal[">", ">=", "<", "<="] | None, float | None, str]:
    for pattern, comparator in _COMPARATOR_RE:
        match = pattern.search(text)
        if match:
            return comparator, float(match.group(1)), _normalize_unit(match.group(2) or "")
    return None, None, ""


def _normalize_unit(unit: str) -> str:
    normalized = unit.lower()
    if normalized == "db":
        return "dB"
    return normalized


def _build_objective_function(
    objective: RequirementObjective,
    wavelength_nm: float | None,
) -> str:
    if not objective.metric or not objective.goal:
        return ""

    parts = [objective.goal, objective.metric.replace("_", " ")]
    if objective.target_value is not None and objective.comparator:
        target = f"{objective.comparator} {objective.target_value:g}"
        if objective.unit:
            target = f"{target} {objective.unit}"
        parts.append(target)
    if wavelength_nm is not None:
        parts.append(f"@ {wavelength_nm:g} nm")
    return " ".join(parts)


def _extract_footprint_raw_value(text: str) -> str:
    """Normalize footprint dimension text as '<dim> <unit> x <dim> <unit>'."""
    pair_match = _FOOTPRINT_PAIR_RE.search(text)
    if pair_match:
        dim1 = pair_match.group(1)
        unit1 = pair_match.group(2) or pair_match.group(4)
        dim2 = pair_match.group(3)
        unit2 = pair_match.group(4)
        return f"{dim1} {unit1} x {dim2} {unit2}"

    legacy_match = _FOOTPRINT_RE.search(text)
    if legacy_match:
        return f"{legacy_match.group(1)} x {legacy_match.group(2)}"
    return ""


def _in_footprint_span(start: int, end: int, text: str) -> bool:
    """Return True when a unit-bearing number is part of an NxN footprint."""
    for pattern in (_FOOTPRINT_PAIR_RE, _FOOTPRINT_RE):
        for match in pattern.finditer(text):
            if match.start() <= start and end <= match.end():
                return True
    return False


def _unit_context_window(text: str, start: int, end: int, window: int = 28) -> str:
    return text[max(0, start - window):min(len(text), end + window)].lower()


def _is_dimension_unit_match(text: str, start: int, end: int) -> bool:
    context = _unit_context_window(text, start, end)
    before = text[max(0, start - 24):start].lower()
    after = text[end:min(len(text), end + 24)].lower()
    if _DIMENSION_CONTEXT_RE.search(before) or _DIMENSION_TRAILING_RE.search(after):
        return True
    if any(token in context for token in _DIMENSION_CONTEXT_TOKENS):
        return True
    if any(token in context for token in _WAVELENGTH_CONTEXT_TOKENS):
        return False
    return False


def _extract_routing_targets(text: str) -> List[RequirementRoutingTarget]:
    targets: List[RequirementRoutingTarget] = []
    global_xtalk = None
    crosstalk_match = _CROSSTALK_MAX_RE.search(text) or _CROSSTALK_ASSIGN_RE.search(text)
    if crosstalk_match:
        global_xtalk = float(crosstalk_match.group(1)) / 100.0
    mode_mux_targets = _extract_mode_mux_routing_targets(text, global_xtalk=global_xtalk)
    if mode_mux_targets:
        return mode_mux_targets

    port_re = re.compile(r"(?:port|端口)\s*([0-9]+)", re.IGNORECASE)

    # 1) Primary: scan wavelength mentions and pair with the nearest port hint.
    for wl_match in re.finditer(r"([0-9]+(?:\.[0-9]+)?)\s*nm", text, re.IGNORECASE):
        wl_nm = float(wl_match.group(1))
        if not (100.0 <= wl_nm <= 20000.0):
            continue

        span_start = max(0, wl_match.start() - 120)
        span_end = min(len(text), wl_match.end() + 160)
        local = text[span_start:span_end]

        port_candidates: List[tuple[int, int, int, int]] = []
        for pm in port_re.finditer(local):
            try:
                port_num = int(pm.group(1))
            except ValueError:
                continue
            if port_num <= 0:
                continue
            global_port_pos = span_start + pm.start()
            distance = abs(global_port_pos - wl_match.start())
            between_text = (
                text[wl_match.end():global_port_pos]
                if global_port_pos >= wl_match.end()
                else text[global_port_pos:wl_match.start()]
            )
            crosses_other_wavelength = bool(
                re.search(r"([0-9]+(?:\.[0-9]+)?)\s*nm", between_text, re.IGNORECASE)
            )
            # Prefer downstream ("wavelength ... to portX") matches when present.
            direction_priority = 0 if global_port_pos >= wl_match.end() else 1
            # Prefer pairings that do not cross another wavelength mention.
            wavelength_cross_priority = 1 if crosses_other_wavelength else 0
            port_candidates.append(
                (wavelength_cross_priority, direction_priority, distance, port_num)
            )
        if not port_candidates:
            continue
        port_candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        if port_candidates[0][0] > 0:
            # Ambiguous pairing: every nearby port requires crossing another
            # wavelength mention, so skip this wavelength occurrence.
            continue
        port = f"port_o{port_candidates[0][3]}"

        coupling_candidates: List[tuple[int, float]] = []
        for cm in _COUPLING_MIN_RE.finditer(local):
            value = float(cm.group(1)) / 100.0
            distance = abs((span_start + cm.start()) - wl_match.start())
            coupling_candidates.append((distance, value))
        for cm in _COUPLING_ASSIGN_RE.finditer(local):
            value = float(cm.group(1)) / 100.0
            distance = abs((span_start + cm.start()) - wl_match.start())
            coupling_candidates.append((distance, value))
        coupling_candidates.sort(key=lambda item: item[0])
        min_coupling = coupling_candidates[0][1] if coupling_candidates else None

        xtalk_candidates: List[tuple[int, float]] = []
        for xm in _CROSSTALK_MAX_RE.finditer(local):
            value = float(xm.group(1)) / 100.0
            distance = abs((span_start + xm.start()) - wl_match.start())
            xtalk_candidates.append((distance, value))
        for xm in _CROSSTALK_ASSIGN_RE.finditer(local):
            value = float(xm.group(1)) / 100.0
            distance = abs((span_start + xm.start()) - wl_match.start())
            xtalk_candidates.append((distance, value))
        xtalk_candidates.sort(key=lambda item: item[0])
        max_crosstalk = xtalk_candidates[0][1] if xtalk_candidates else global_xtalk

        targets.append(
            RequirementRoutingTarget(
                wavelength_nm=wl_nm,
                target_port=port,
                min_coupling=min_coupling,
                max_crosstalk=max_crosstalk,
            )
        )

    # 2) Compatibility: keep the legacy "1300 nm -> port 2" extraction.
    if not targets:
        for match in _ROUTING_TARGET_RE.finditer(text):
            wl_nm = float(match.group(1))
            port = f"port_o{int(match.group(2))}"
            local_span_start = max(0, match.start() - 120)
            local_span_end = min(len(text), match.end() + 40)
            local = text[local_span_start:local_span_end]
            coupling_match = _COUPLING_MIN_RE.search(local) or _COUPLING_ASSIGN_RE.search(local)
            min_coupling = (
                float(coupling_match.group(1)) / 100.0
                if coupling_match
                else None
            )
            targets.append(
                RequirementRoutingTarget(
                    wavelength_nm=wl_nm,
                    target_port=port,
                    min_coupling=min_coupling,
                    max_crosstalk=global_xtalk,
                )
            )

    # 3) Single-wavelength mode/path routing often appears as
    # "port2 output TE0, port3 output TE1" without repeating wavelength
    # for each port. When only one wavelength is present, map all output-port
    # mentions to that wavelength so Step3 can build multi-case objectives.
    all_wavelengths = _extract_all_wavelengths_nm(text)
    if len(all_wavelengths) == 1:
        anchor_wl = float(all_wavelengths[0])
        for port_num in _extract_output_port_mentions(text):
            port = f"port_o{port_num}"
            exists = any(
                round(item.wavelength_nm, 6) == round(anchor_wl, 6)
                and item.target_port == port
                for item in targets
            )
            if exists:
                continue
            targets.append(
                RequirementRoutingTarget(
                    wavelength_nm=anchor_wl,
                    target_port=port,
                    min_coupling=None,
                    max_crosstalk=global_xtalk,
                )
            )

    # Deduplicate by (wavelength, port) while preserving order.
    deduped: List[RequirementRoutingTarget] = []
    seen: set[tuple[float, str]] = set()
    for item in targets:
        key = (round(item.wavelength_nm, 6), item.target_port)
        if key in seen:
            continue
        deduped.append(item)
        seen.add(key)
    return deduped


def _extract_mode_mux_routing_targets(
    text: str,
    *,
    global_xtalk: float | None,
) -> List[RequirementRoutingTarget]:
    lowered = str(text or "").lower().translate(str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789"))

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

    parsed: List[tuple[int, int, int, int]] = []
    for clause in clauses:
        source = _extract_port_mode_from_clause(clause, role="source")
        target = _extract_port_mode_from_clause(clause, role="target")
        if source is None or target is None:
            case_pair = _extract_mode_mux_case_pair(clause)
            if case_pair is not None:
                parsed.append(case_pair)
            continue
        parsed.append((source[0], source[1], target[0], target[1]))
    parsed.extend(_extract_mode_mux_pairs_global(lowered))

    deduped_pairs: List[tuple[int, int, int, int]] = []
    seen_pairs: set[tuple[int, int, int, int]] = set()
    for pair in parsed:
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        deduped_pairs.append(pair)
    parsed = deduped_pairs

    if not parsed:
        return []
    source_ports = {item[0] for item in parsed}
    target_ports = {item[2] for item in parsed}
    if target_ports != {1}:
        return []

    if len(source_ports) < 2:
        has_one_case_hint = any(
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
        has_mode_mux_intent = any(
            token in lowered
            for token in (
                "mode multiplexer",
                "mode mux",
                "模式复用",
            )
        )
        if not (len(source_ports) == 1 and (has_one_case_hint or has_mode_mux_intent)):
            return []

    wavelengths = _extract_all_wavelengths_nm(text)
    if wavelengths:
        wl_nm = float(wavelengths[0])
    else:
        wl_single = _extract_wavelength_nm(text)
        wl_nm = float(wl_single) if wl_single is not None else 1550.0

    coupling_match = _COUPLING_MIN_RE.search(text) or _COUPLING_ASSIGN_RE.search(text)
    global_min_coupling = (
        float(coupling_match.group(1)) / 100.0
        if coupling_match
        else None
    )

    targets: List[RequirementRoutingTarget] = []
    for source_port_num, source_mode_idx, target_port_num, target_mode_idx in sorted(parsed):
        targets.append(
            RequirementRoutingTarget(
                wavelength_nm=wl_nm,
                source_port=f"port_o{source_port_num}",
                source_mode_index=max(int(source_mode_idx), 0),
                target_port=f"port_o{target_port_num}",
                target_mode_index=max(int(target_mode_idx), 0),
                min_coupling=global_min_coupling,
                max_crosstalk=global_xtalk,
            )
        )
    return targets


def _extract_mode_mux_pairs_global(text: str) -> List[tuple[int, int, int, int]]:
    """Extract source-switched mode-mux mappings from full prompt text.

    This complements clause-level parsing when punctuation/formatting causes
    local matching to miss some `portN input TEi -> port1 output TEj` pairs.
    """
    pairs: List[tuple[int, int, int, int]] = []
    patterns = [
        re.compile(
            r"(?:port|端口)\s*(?P<src_port>[0-9]+)\s*"
            r"(?:input|输入|inject|source|launch)\s*"
            r"(?:te|tm)\s*[_-]?\s*(?P<src_mode>[0-9]+)"
            r"[^.;,\n，。；]{0,80}?(?:to|->|到)\s*"
            r"(?:port|端口)\s*(?P<tgt_port>[0-9]+)\s*"
            r"(?:output|输出|target)?\s*"
            r"(?:te|tm)\s*[_-]?\s*(?P<tgt_mode>[0-9]+)",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?:port|端口)\s*(?P<src_port>[0-9]+)\s*"
            r"(?:te|tm)\s*[_-]?\s*(?P<src_mode>[0-9]+)"
            r"[^.;,\n，。；]{0,80}?(?:to|->|到)\s*"
            r"(?:port|端口)\s*(?P<tgt_port>[0-9]+)\s*"
            r"(?:te|tm)\s*[_-]?\s*(?P<tgt_mode>[0-9]+)",
            re.IGNORECASE,
        ),
    ]
    for pattern in patterns:
        for match in pattern.finditer(text):
            try:
                src_port = int(match.group("src_port"))
                src_mode = max(int(match.group("src_mode")), 0)
                tgt_port = int(match.group("tgt_port"))
                tgt_mode = max(int(match.group("tgt_mode")), 0)
            except (TypeError, ValueError):
                continue
            if src_port <= 0 or tgt_port <= 0:
                continue
            pairs.append((src_port, src_mode, tgt_port, tgt_mode))
    return pairs


def _extract_mode_mux_case_pair(clause: str) -> tuple[int, int, int, int] | None:
    patterns = [
        re.compile(
            r"(?:port|端口)\s*(?P<src_port>[0-9]+)[^.;,\n，。；]{0,40}?"
            r"(?:输入|注入|入射|source|inject|launch)[^.;,\n，。；]{0,20}?"
            r"(?:te|tm)\s*[_-]?\s*(?P<src_mode>[0-9]+)"
            r"[^.;,\n，。；]{0,60}?(?:到|to|->)[^.;,\n，。；]{0,20}?"
            r"(?:port|端口)\s*(?P<tgt_port>[0-9]+)[^.;,\n，。；]{0,40}?"
            r"(?:输出|output|target)[^.;,\n，。；]{0,20}?"
            r"(?:te|tm)\s*[_-]?\s*(?P<tgt_mode>[0-9]+)",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?:launch|inject|input|source)\s*(?:te|tm)\s*[_-]?\s*(?P<src_mode>[0-9]+)"
            r"[^.;,\n]{0,80}?(?:from|in)\s*(?:port(?:_?o)?|端口)\s*[_\s]*?(?P<src_port>[0-9]+)"
            r"[^.;,\n]{0,120}?(?:maximize|target|output|at)\s*(?:te|tm)\s*[_-]?\s*(?P<tgt_mode>[0-9]+)"
            r"[^.;,\n]{0,80}?(?:at|to|in)\s*(?:port(?:_?o)?|端口)\s*[_\s]*?(?P<tgt_port>[0-9]+)",
            re.IGNORECASE,
        ),
    ]
    for pattern in patterns:
        match = pattern.search(clause)
        if not match:
            continue
        try:
            src_mode = max(int(match.group("src_mode")), 0)
            src_port = int(match.group("src_port"))
            tgt_mode = max(int(match.group("tgt_mode")), 0)
            tgt_port = int(match.group("tgt_port"))
        except (TypeError, ValueError):
            continue
        if src_port <= 0 or tgt_port <= 0:
            continue
        return src_port, src_mode, tgt_port, tgt_mode
    return None


def _extract_port_mode_from_clause(
    clause: str,
    *,
    role: str,
) -> tuple[int, int] | None:
    role_tokens = (
        ("输入", "注入", "入射", "source", "inject", "launch", "input")
        if role == "source"
        else ("输出", "output", "to", "out", "target")
    )
    token_expr = "|".join(re.escape(token) for token in role_tokens)
    port_expr = r"(?:port|端口)\s*(?P<port>[0-9]+)"
    mode_expr = r"(?:te|tm)\s*[_-]?\s*(?P<mode>[0-9]+)"
    # Do not cross another "portN" mention when binding role/mode to a port.
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
        try:
            port_num = int(match.group("port"))
            mode_idx = int(match.group("mode"))
        except (TypeError, ValueError):
            continue
        if port_num <= 0:
            continue
        return port_num, max(mode_idx, 0)
    return None


def _extract_output_port_mentions(text: str) -> List[int]:
    """Extract output-port numbers mentioned in routing/output context."""
    ports: List[int] = []
    seen: set[int] = set()

    def _add(port_num: int) -> None:
        if port_num <= 1 or port_num in seen:
            return
        seen.add(port_num)
        ports.append(port_num)

    contextual_patterns = [
        re.compile(
            r"(?:port|端口)\s*([0-9]+)[^。；;,\n]{0,32}(?:输出|output|to)",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?:输出|output|to)[^。；;,\n]{0,24}(?:port|端口)\s*([0-9]+)",
            re.IGNORECASE,
        ),
    ]
    for pattern in contextual_patterns:
        for match in pattern.finditer(text):
            try:
                _add(int(match.group(1)))
            except Exception:
                continue

    if ports:
        return ports

    # Fallback: collect explicit port mentions (excluding input port 1).
    for match in re.finditer(r"(?:port|端口)\s*([0-9]+)", text, re.IGNORECASE):
        try:
            _add(int(match.group(1)))
        except Exception:
            continue
    return ports
