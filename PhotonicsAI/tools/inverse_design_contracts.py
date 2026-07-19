"""Shared contracts for the Step1-Step5 inverse-design pipeline."""

from __future__ import annotations

from typing import Iterable, List, Tuple


DISCRETE_TOPOLOGY_PARAMETER_NAMES: Tuple[str, ...] = ("mmi_num_outputs",)

DEFAULT_REQUIRED_INPUT_FLUX_KEYS: Tuple[str, ...] = (
    "flux_port_o1",
    "flux_port_i1",
    "flux_input",
)
DEFAULT_REQUIRED_OUTPUT_FLUX_KEYS: Tuple[str, ...] = ("flux_port_o2", "flux_port_o3")

OBJECTIVE_FLUX_KEY_CONTRACTS = {
    "crosstalk": (DEFAULT_REQUIRED_INPUT_FLUX_KEYS, ("flux_port_o3", "flux_port_o4")),
    "cross_talk": (DEFAULT_REQUIRED_INPUT_FLUX_KEYS, ("flux_port_o3", "flux_port_o4")),
    "secondary_output": (DEFAULT_REQUIRED_INPUT_FLUX_KEYS, ("flux_port_o3", "flux_port_o4")),
    "mux_routing": (("flux_port_o2", "flux_port_o3", "flux_port_o1"), ("flux_port_o1",)),
}

OBJECTIVE_SEMANTIC_CONTRACT_RULES = {
    "demux_routing": (
        "Optimizer and diagnostic simulations must share the same component-aware builder.",
        "Design region size and alignment must be derived from footprint or component parameters.",
        "Flux monitors must cover input and all output ports; mode monitors must cover every output port.",
        "Nonzero target modes require ModeSource/ModeMonitor num_modes >= max(target_mode_index)+1.",
        "Field, flux, and topology artifacts must be scoped to the current run tag, iteration, and case.",
        "Acceptance must not be repaired by hand-editing simulation artifacts; fixes must land in Step1-Step5 logic.",
    ),
    "wdm_routing": (
        "Optimizer and diagnostic simulations must share the same component-aware builder.",
        "Design region size and alignment must be derived from footprint or component parameters.",
        "Flux monitors must cover input and all output ports; mode monitors must cover every output port.",
        "Nonzero target modes require ModeSource/ModeMonitor num_modes >= max(target_mode_index)+1.",
        "Field, flux, and topology artifacts must be scoped to the current run tag, iteration, and case.",
        "Acceptance must not be repaired by hand-editing simulation artifacts; fixes must land in Step1-Step5 logic.",
    ),
    "mux_routing": (
        "Optimizer and diagnostic simulations must share the same component-aware builder.",
        "Design region size and alignment must be derived from footprint or component parameters.",
        "Mode-mux objectives must use per-case source switching and target-mode checks at the output mode monitor.",
        "Nonzero target modes require ModeSource/ModeMonitor num_modes >= max(target_mode_index)+1.",
        "Each case must report source-port input flux and target-port transmission ratio to input.",
        "Field, flux, and topology artifacts must be scoped to the current run tag, iteration, and case.",
    ),
}


def objective_flux_key_contract(objective_metric: str) -> tuple[List[str], List[str]]:
    """Return required input/output flux keys for an objective metric."""

    normalized = _normalize_objective_metric(objective_metric)
    input_keys, output_keys = OBJECTIVE_FLUX_KEY_CONTRACTS.get(
        normalized,
        (DEFAULT_REQUIRED_INPUT_FLUX_KEYS, DEFAULT_REQUIRED_OUTPUT_FLUX_KEYS),
    )
    return list(input_keys), list(output_keys)


def objective_semantic_contract_rules(objective_metric: str) -> List[str]:
    """Return reusable Step4 semantic contract rules for an objective metric."""

    normalized = _normalize_objective_metric(objective_metric)
    return list(OBJECTIVE_SEMANTIC_CONTRACT_RULES.get(normalized, ()))


def is_discrete_topology_parameter(name: str) -> bool:
    """Return true when a design parameter controls structural cardinality."""

    return name.strip().lower() in DISCRETE_TOPOLOGY_PARAMETER_NAMES


def normalize_discrete_topology_parameter(name: str, value: float) -> float:
    """Normalize a discrete topology parameter to its executable numeric form."""

    if not is_discrete_topology_parameter(name):
        return float(value)
    return float(max(int(round(value)), 2))


def dedupe_contract_rules(rules: Iterable[str], *, max_items: int = 20) -> List[str]:
    """Deduplicate contract rules while preserving order."""

    seen: set[str] = set()
    deduped: List[str] = []
    for rule in rules:
        text = str(rule).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        deduped.append(text)
        if len(deduped) >= max_items:
            break
    return deduped


def _normalize_objective_metric(objective_metric: str) -> str:
    return str(objective_metric or "").strip().lower()
