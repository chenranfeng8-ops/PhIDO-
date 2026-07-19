"""Structured schemas for inverse-design simulation and optimization configs."""

from __future__ import annotations

from typing import Any, Dict, List, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class StrictModel(BaseModel):
    """Base model that rejects unknown fields to keep LLM output constrained."""

    model_config = ConfigDict(extra="forbid")


class DocumentationReference(StrictModel):
    """Traceable documentation evidence used to justify config decisions."""

    url: str
    title: str = ""
    summary: str = ""
    rules: List[str] = Field(default_factory=list)


class GeometrySpec(StrictModel):
    """Geometry definition for the base simulation device."""

    component_type: str
    template_path: str | None = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    variable_regions: List[str] = Field(default_factory=list)


class DomainSpec(StrictModel):
    """Simulation domain and discretization settings."""

    size_um: List[float] = Field(..., min_length=3, max_length=3)
    center_um: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0], min_length=3, max_length=3)
    mesh_strategy: Literal["auto", "uniform", "override"] = "auto"
    min_steps_per_wvl: int = Field(default=20, ge=6)
    boundary: Dict[str, str] = Field(
        default_factory=lambda: {
            "x_min": "pml",
            "x_max": "pml",
            "y_min": "pml",
            "y_max": "pml",
            "z_min": "pml",
            "z_max": "pml",
        }
    )


class SourceSpec(StrictModel):
    """Excitation source specification."""

    source_type: Literal["mode", "gaussian", "plane_wave"] = "mode"
    port: str = "port_o1"
    center_um: List[float] = Field(..., min_length=3, max_length=3)
    size_um: List[float] = Field(..., min_length=3, max_length=3)
    direction: Literal["+", "-"] = "+"
    mode_index: int = Field(default=0, ge=0)
    wavelength_nm: float = Field(..., gt=0)
    bandwidth_nm: float = Field(default=50.0, gt=0)


class MonitorSpec(StrictModel):
    """Requested simulation monitor."""

    name: str
    monitor_type: Literal["field", "flux", "mode"]
    center_um: List[float] = Field(..., min_length=3, max_length=3)
    size_um: List[float] = Field(..., min_length=3, max_length=3)
    freqs_hz: List[float] = Field(..., min_length=1)
    field_component: str | None = None
    metric: str | None = None

    @model_validator(mode="after")
    def _check_monitor_details(self) -> "MonitorSpec":
        if self.monitor_type == "field" and not self.field_component:
            raise ValueError("field monitors require `field_component`.")
        if self.monitor_type in {"flux", "mode"} and not self.metric:
            raise ValueError("flux/mode monitors require `metric`.")
        return self


class SimulationConfig(StrictModel):
    """LLM-generated simulation configuration before semantic validation."""

    component_type: str
    wavelength_nm: float = Field(..., gt=0)
    geometry: GeometrySpec
    domain: DomainSpec
    source: SourceSpec
    monitors: List[MonitorSpec] = Field(..., min_length=1)
    run_time_s: float = Field(default=2e-12, gt=0)
    shutoff: float = Field(default=1e-5, ge=0, le=1)
    doc_references: List[DocumentationReference] = Field(..., min_length=1)


class ObjectiveSpec(StrictModel):
    """Optimization target to maximize or minimize."""

    metric: str
    goal: Literal["maximize", "minimize"]
    target_value: float | None = None
    description: str = ""


class ObjectiveCaseSpec(StrictModel):
    """One operating case in a multi-condition inverse-design objective."""

    name: str = ""
    wavelength_nm: float = Field(..., gt=0)
    source_port: str = ""
    source_mode_index: int = Field(default=0, ge=0)
    source_direction: Literal["+", "-", ""] = ""
    target_port: str = ""
    target_mode_index: int = Field(default=0, ge=0)
    # v16 canonical absolute design goals (W, ModeSource injects 1 W reference).
    # These supersede the v14-era `min_coupling`/`max_crosstalk` ratio fields
    # which were derived from FluxMonitor / FluxMonitor and broke for
    # direction="-" sources (denominator measured reflection, not injection).
    # See memory-bank/progress.md §V16-METRIC-CLEANUP.
    min_absolute_ce_w: float | None = Field(default=None, ge=0.0, le=1.0)
    max_wrong_mode_leakage_w: float | None = Field(default=None, ge=0.0, le=1.0)
    max_cross_port_leakage_w: float | None = Field(default=None, ge=0.0, le=1.0)
    max_reflection_w: float | None = Field(default=None, ge=0.0, le=1.0)
    # Legacy v14/v15 ratio fields — DEPRECATED, kept only so old bundles still
    # parse. New bundles MUST populate `min_absolute_ce_w` etc. Step5 emits a
    # WARN and auto-translates if these legacy fields are present without v16
    # fields. Removal scheduled after Step3 generator (config_generation.py)
    # is migrated to the canonical schema.
    min_coupling: float | None = None
    max_crosstalk: float | None = None
    weight: float = Field(default=1.0, gt=0)
    cross_mode_index: int | None = None
    cross_mode_penalty_weight: float | None = None
    # v14 narrow-mode gradient boost: opt-in log-coupling reward whose
    # gradient = log_coupling_weight / (CE + log_coupling_epsilon) explodes
    # for small CE, restoring design-space pull for narrow-mode cases that
    # would otherwise be neglected due to small ∂CE/∂x_design.
    use_log_coupling: bool = False
    log_coupling_weight: float | None = Field(default=None, ge=0)
    log_coupling_epsilon: float | None = Field(default=None, gt=0)


class VariableBounds(StrictModel):
    """Continuous design variable bounds for inverse design."""

    name: str
    lower_bound: float
    upper_bound: float
    initial_value: float | None = None

    @model_validator(mode="after")
    def _check_bounds(self) -> "VariableBounds":
        if self.lower_bound >= self.upper_bound:
            raise ValueError("`lower_bound` must be smaller than `upper_bound`.")
        if self.initial_value is not None and not (self.lower_bound <= self.initial_value <= self.upper_bound):
            raise ValueError("`initial_value` must stay within the declared bounds.")
        return self


class TerminationSpec(StrictModel):
    """Conditions that stop the optimization loop."""

    max_iterations: int = Field(..., ge=1)
    target_score: float | None = None
    min_improvement: float | None = Field(default=None, ge=0)
    patience: int | None = Field(default=None, ge=1)


class RuntimeConfig(StrictModel):
    """Execution-time controls for the Step5 run wrapper.

    These values are intentionally separated from the optimization objective
    contract. They describe how one Step5 execution should be launched, while
    ``optimization_config.termination`` still describes the broader optimizer
    budget/stop criteria for the design problem itself.
    """

    max_iterations: int = Field(default=3, ge=1)
    include_llm_review: bool = True
    enable_failure_diagnosis: bool = True
    checkpoint_interval: int = Field(default=5, ge=1)
    checkpoint_warmup: int = Field(default=5, ge=1)
    checkpoint_min_relative_improvement: float = Field(default=0.05, ge=0.0)
    checkpoint_regression_tolerance: float = Field(default=0.05, ge=0.0)
    checkpoint_oscillation_ratio_threshold: float = Field(default=4.0, ge=0.0)
    checkpoint_update_norm_direction_threshold: float = Field(default=0.01, ge=0.0)
    checkpoint_update_norm_oscillation_threshold: float = Field(default=0.02, ge=0.0)
    enable_optimizer_attribution: bool = False
    optimizer_attribution_min_samples: int = Field(default=12, ge=1)
    optimizer_attribution_method: Literal[
        "random_forest_permutation",
        "correlation_fallback",
    ] = "random_forest_permutation"


def build_default_runtime_config(*, max_iterations: int) -> RuntimeConfig:
    """Create the canonical Step5 runtime defaults for a generated bundle.

    This helper keeps the execution-default contract in one place so Step3,
    Step5 wrappers, and direct execution callers do not drift into separate
    copies of the same defaults.
    """

    return RuntimeConfig(
        max_iterations=max_iterations,
        include_llm_review=True,
        enable_failure_diagnosis=True,
        checkpoint_interval=5,
        checkpoint_warmup=5,
        checkpoint_min_relative_improvement=0.05,
        checkpoint_regression_tolerance=0.05,
        checkpoint_oscillation_ratio_threshold=4.0,
        checkpoint_update_norm_direction_threshold=0.01,
        checkpoint_update_norm_oscillation_threshold=0.02,
        enable_optimizer_attribution=False,
        optimizer_attribution_min_samples=12,
        optimizer_attribution_method="random_forest_permutation",
    )


class OptimizationConfig(StrictModel):
    """Inverse-design optimization configuration."""

    optimizer: Literal["inverse_design", "adjoint"] = "inverse_design"
    objective: ObjectiveSpec
    objective_cases: List[ObjectiveCaseSpec] = Field(default_factory=list)
    variables: List[VariableBounds] = Field(..., min_length=1)
    termination: TerminationSpec
    constraints: List[str] = Field(default_factory=list)
    doc_references: List[DocumentationReference] = Field(..., min_length=1)
    optimizer_hints: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Runtime hints for the adjoint optimizer: learning_rate, beta, "
            "beta_max, penalty_weight, etc.  Populated by recovery to tune "
            "stalled/oscillating optimization."
        ),
    )


class InverseDesignConfigBundle(StrictModel):
    """Top-level payload expected from the LLM for the new optimization path."""

    simulation_config: SimulationConfig
    optimization_config: OptimizationConfig
    runtime_config: RuntimeConfig = Field(default_factory=RuntimeConfig)


def parse_inverse_design_config(payload: Dict[str, Any]) -> InverseDesignConfigBundle:
    """Parse a raw payload into the strict inverse-design config bundle."""

    return InverseDesignConfigBundle.model_validate(payload)


def inverse_design_config_schema() -> Dict[str, Any]:
    """Return the JSON schema used to guide LLM config generation."""

    return InverseDesignConfigBundle.model_json_schema()


def format_config_validation_error(exc: ValidationError) -> str:
    """Convert a pydantic validation error into a short readable summary."""

    parts = []
    for error in exc.errors():
        location = ".".join(str(item) for item in error.get("loc", []))
        parts.append(f"{location}: {error.get('msg', 'invalid value')}")
    return "; ".join(parts)
