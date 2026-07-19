"""Step4 wrapper: deterministic validation plus RAG semantic constraints."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

from PhotonicsAI.tools.inverse_design_config import InverseDesignConfigBundle
from PhotonicsAI.tools.inverse_design_config_validation import validate_config
from PhotonicsAI.tools.inverse_design_doc_context import InverseDesignDocContext
from PhotonicsAI.tools.inverse_design_replan import apply_patch_actions
from PhotonicsAI.tools.inverse_design_rag_memory import (
    build_step4_constraint_packet,
    get_inverse_design_step4_rag_memory,
)


def inverse_step4_validate_config(
    config_bundle: Dict[str, Any] | None = None,
    *,
    doc_context: Dict[str, Any] | None = None,
    include_llm_review: bool = True,
    llm_model: str = "gpt-5.4",
    selected_rollback_step: str | None = None,
    recovery_context: Dict[str, Any] | None = None,
    workflow_state: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Validate Step3 config with Step4 hard constraints and semantic RAG packet."""

    try:
        effective_bundle = config_bundle
        if effective_bundle is None and workflow_state:
            effective_bundle = (
                workflow_state.get("step3", {})
                .get("data", {})
                .get("config_bundle")
            )
        if effective_bundle is None:
            raise ValueError("missing Step3 config bundle input for Step4")

        effective_context = doc_context
        if effective_context is None and workflow_state:
            effective_context = (
                workflow_state.get("step2", {})
                .get("data", {})
                .get("doc_context")
            )

        bundle_payload = deepcopy(effective_bundle)
        applied_patch_paths = []
        if isinstance(recovery_context, dict):
            applied_patch_paths = apply_patch_actions(
                bundle_payload,
                recovery_context.get("patch_actions", []),
            )

        typed_bundle = InverseDesignConfigBundle.model_validate(bundle_payload)
        typed_context = (
            InverseDesignDocContext.model_validate(effective_context)
            if isinstance(effective_context, dict)
            else None
        )
        packet = build_step4_constraint_packet(
            doc_context=typed_context,
            config_bundle=typed_bundle,
        )
        rag_memory = get_inverse_design_step4_rag_memory()
        rag_memory.record(
            query=f"{packet.component_type} {packet.objective_metric}".strip(),
            packet=packet,
        )
        validation = validate_config(
            typed_bundle,
            include_llm_review=include_llm_review,
            llm_model=llm_model,
            constraint_packet=packet,
        )
    except Exception as exc:
        return {"ok": False, "data": {}, "error": f"Step4 validation failed: {exc}"}

    return {
        "ok": True,
        "data": {
            "validation": validation.model_dump(),
            "constraint_packet": packet.model_dump(),
            "input_dependencies": ["step3.config_bundle", "step2.doc_context"],
            "recovery_context_used": bool(recovery_context),
            "selected_rollback_step": selected_rollback_step or "",
            "applied_patch_paths": applied_patch_paths,
        },
        "error": None,
    }
