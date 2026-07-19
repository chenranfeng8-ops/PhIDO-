"""Step5 wrapper: execute inverse-design optimization with diagnosis fallback."""

from __future__ import annotations

from typing import Any, Dict

from PhotonicsAI.tools.inverse_design_config import InverseDesignConfigBundle
from PhotonicsAI.tools.inverse_design_execution import run_inverse_design


def inverse_step5_execute(
    config_bundle: Dict[str, Any] | None = None,
    *,
    max_iterations: int | None = None,
    include_llm_review: bool | None = None,
    enable_failure_diagnosis: bool | None = None,
    constraint_packet: Dict[str, Any] | None = None,
    checkpoint_interval: int | None = None,
    checkpoint_warmup: int | None = None,
    llm_model: str = "gpt-5.4",
    selected_rollback_step: str | None = None,
    recovery_context: Dict[str, Any] | None = None,
    workflow_state: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Execute Step5 inverse design using LLM-first review and diagnosis settings."""

    try:
        effective_bundle = config_bundle
        if effective_bundle is None and workflow_state:
            effective_bundle = (
                workflow_state.get("step3", {})
                .get("data", {})
                .get("config_bundle")
            )
        effective_constraint_packet = constraint_packet
        if effective_constraint_packet is None and workflow_state:
            effective_constraint_packet = (
                workflow_state.get("step4", {})
                .get("data", {})
                .get("constraint_packet")
            )
        if effective_bundle is None:
            raise ValueError("missing Step3 config bundle input for Step5")

        typed_bundle = InverseDesignConfigBundle.model_validate(effective_bundle)
        runtime_config = typed_bundle.runtime_config
        result = run_inverse_design(
            typed_bundle,
            max_iterations=(
                int(max_iterations)
                if max_iterations is not None
                else int(runtime_config.max_iterations)
            ),
            include_llm_review=(
                include_llm_review
                if include_llm_review is not None
                else bool(runtime_config.include_llm_review)
            ),
            enable_failure_diagnosis=(
                enable_failure_diagnosis
                if enable_failure_diagnosis is not None
                else bool(runtime_config.enable_failure_diagnosis)
            ),
            constraint_packet=effective_constraint_packet,
            checkpoint_interval=(
                int(checkpoint_interval)
                if checkpoint_interval is not None
                else int(runtime_config.checkpoint_interval)
            ),
            checkpoint_warmup=(
                int(checkpoint_warmup)
                if checkpoint_warmup is not None
                else int(runtime_config.checkpoint_warmup)
            ),
            llm_model=llm_model,
        )
    except Exception as exc:
        return {"ok": False, "data": {}, "error": f"Step5 execution failed: {exc}"}

    return {
        "ok": bool(result.ok),
        "data": {
            "run_result": result.model_dump(),
            "input_dependencies": ["step3.config_bundle", "step4.validation", "step4.constraint_packet"],
            "recovery_context_used": bool(recovery_context),
            "selected_rollback_step": selected_rollback_step or "",
        },
        "error": None if result.ok else result.termination_reason,
    }
