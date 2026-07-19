"""Step2 wrapper: retrieve documentation context with LLM-first planning."""

from __future__ import annotations

import json
from typing import Any, Dict

from PhotonicsAI.tools.inverse_design_doc_context import (
    InverseDesignDocContext,
    retrieve_inverse_design_doc_context,
)
from PhotonicsAI.tools.inverse_design_requirements import InverseDesignRequirement


def inverse_step2_retrieve_doc_context(
    requirement: Dict[str, Any] | str | None = None,
    *,
    max_results: int = 3,
    use_llm_planner: bool = True,
    llm_model: str = "gpt-5.4",
    selected_rollback_step: str | None = None,
    recovery_context: Dict[str, Any] | None = None,
    workflow_state: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build Step2 document context from Step1 output."""

    try:
        effective_requirement: Dict[str, Any] | str | None = requirement
        if effective_requirement is None and workflow_state:
            effective_requirement = (
                workflow_state.get("step1", {})
                .get("data", {})
                .get("requirement")
            )
        if effective_requirement is None:
            raise ValueError("missing Step1 requirement input for Step2")
        if isinstance(effective_requirement, str):
            stripped = effective_requirement.strip()
            if stripped.startswith("{") and stripped.endswith("}"):
                try:
                    parsed = json.loads(stripped)
                except Exception:
                    parsed = None
                if isinstance(parsed, dict):
                    effective_requirement = parsed

        extra_queries = []
        if isinstance(recovery_context, dict):
            extra_queries = [str(item) for item in recovery_context.get("suggested_queries", [])]

        typed_requirement = (
            InverseDesignRequirement.model_validate(effective_requirement)
            if isinstance(effective_requirement, dict)
            else effective_requirement
        )
        doc_context = retrieve_inverse_design_doc_context(
            typed_requirement,
            max_results=max_results,
            extra_queries=extra_queries,
            use_llm_planner=use_llm_planner,
            llm_model=llm_model,
        )
    except Exception as exc:
        return {"ok": False, "data": {}, "error": f"Step2 doc retrieval failed: {exc}"}

    assert isinstance(doc_context, InverseDesignDocContext)
    return {
        "ok": True,
        "data": {
            "doc_context": doc_context.model_dump(),
            "reference_count": len(doc_context.references),
            "queries": list(doc_context.queries),
            "input_dependencies": ["step1.requirement"],
            "recovery_context_used": bool(recovery_context),
            "selected_rollback_step": selected_rollback_step or "",
        },
        "error": None,
    }
