"""Unified tool surface for Step1-Step5 inverse-design workflow."""

from __future__ import annotations

import os
from typing import List

from PhotonicsAI.core.tooling import Tool
from PhotonicsAI.tools.inverse_design_steps.mcp_surface import get_inverse_design_mcp_tools
from PhotonicsAI.tools.inverse_design_steps.step1_requirements import inverse_step1_parse_requirements
from PhotonicsAI.tools.inverse_design_steps.step2_doc_context import inverse_step2_retrieve_doc_context
from PhotonicsAI.tools.inverse_design_steps.step3_config_generation import inverse_step3_generate_config
from PhotonicsAI.tools.inverse_design_steps.step4_validation import inverse_step4_validate_config
from PhotonicsAI.tools.inverse_design_steps.step5_execution import inverse_step5_execute


def _step5_tool_timeout_seconds() -> float:
    """Return configurable Step5 tool timeout to avoid long-run false timeouts."""

    raw = os.getenv("INVERSE_STEP5_TOOL_TIMEOUT_S", "").strip()
    if not raw:
        timeout_s = 14400.0
    else:
        try:
            timeout_s = float(raw)
        except (TypeError, ValueError):
            timeout_s = 14400.0
    timeout_s = max(600.0, timeout_s)
    watchdog_raw = os.getenv("INVERSE_STEP5_NO_EVENT_TIMEOUT_S", "").strip()
    if not watchdog_raw:
        return timeout_s
    try:
        watchdog_timeout = float(watchdog_raw)
    except (TypeError, ValueError):
        return timeout_s
    if watchdog_timeout <= 0:
        return timeout_s
    # Keep tool timeout bounded by no-event watchdog to avoid fake-alive stalls.
    return max(120.0, min(timeout_s, watchdog_timeout))


def get_inverse_design_step_tools() -> List[Tool]:
    """Return explicit Step1-Step5 tool chain for ReAct orchestration."""

    return [
        Tool(
            name="inverse_step1_parse_requirements",
            description="Step1: parse natural-language inverse-design requirement into structured target fields.",
            parameters={
                "type": "object",
                "properties": {
                    "requirement_text": {"type": "string"},
                    "require_complete": {"type": "boolean", "default": False},
                    "use_llm_parser": {"type": "boolean", "default": True},
                    "llm_model": {"type": "string", "default": "gpt-5.4"},
                },
                "required": ["requirement_text"],
            },
            fn=inverse_step1_parse_requirements,
        ),
        Tool(
            name="inverse_step2_retrieve_doc_context",
            description="Step2: retrieve MCP documentation context with LLM-first query planning.",
            parameters={
                "type": "object",
                "properties": {
                    "requirement": {
                        "oneOf": [
                            {"type": "object"},
                            {"type": "string"},
                        ]
                    },
                    "max_results": {"type": "integer", "default": 3},
                    "use_llm_planner": {"type": "boolean", "default": True},
                    "llm_model": {"type": "string", "default": "gpt-5.4"},
                    "selected_rollback_step": {"type": "string"},
                    "recovery_context": {"type": "object"},
                    "workflow_state": {"type": "object"},
                },
            },
            fn=inverse_step2_retrieve_doc_context,
            timeout=300.0,  # Bounded to keep LLM main path responsive on MCP instability
        ),
        Tool(
            name="inverse_step3_generate_config",
            description="Step3: generate strict simulation and optimization config bundle from doc context.",
            parameters={
                "type": "object",
                "properties": {
                    "doc_context": {"type": "object"},
                    "use_llm_planner": {"type": "boolean", "default": True},
                    "llm_model": {"type": "string", "default": "gpt-5.4"},
                    "selected_rollback_step": {"type": "string"},
                    "recovery_context": {"type": "object"},
                    "workflow_state": {"type": "object"},
                },
            },
            fn=inverse_step3_generate_config,
        ),
        Tool(
            name="inverse_step4_validate_config",
            description="Step4: validate config with deterministic hard checks plus Step4 RAG semantic constraints.",
            parameters={
                "type": "object",
                "properties": {
                    "config_bundle": {"type": "object"},
                    "doc_context": {"type": "object"},
                    "include_llm_review": {"type": "boolean", "default": True},
                    "llm_model": {"type": "string", "default": "gpt-5.4"},
                    "selected_rollback_step": {"type": "string"},
                    "recovery_context": {"type": "object"},
                    "workflow_state": {"type": "object"},
                },
            },
            fn=inverse_step4_validate_config,
        ),
        Tool(
            name="inverse_step5_execute",
            description=(
                "Step5: run inverse-design optimization. "
                "Execution-time defaults come from config_bundle.runtime_config; "
                "tool arguments only override that bundle contract when explicitly provided."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "config_bundle": {"type": "object"},
                    "max_iterations": {"type": "integer"},
                    "include_llm_review": {"type": "boolean"},
                    "enable_failure_diagnosis": {"type": "boolean"},
                    "constraint_packet": {"type": "object"},
                    "checkpoint_interval": {"type": "integer"},
                    "checkpoint_warmup": {"type": "integer"},
                    "llm_model": {"type": "string", "default": "gpt-5.4"},
                    "selected_rollback_step": {"type": "string"},
                    "recovery_context": {"type": "object"},
                    "workflow_state": {"type": "object"},
                },
            },
            fn=inverse_step5_execute,
            timeout=_step5_tool_timeout_seconds(),  # Cloud optimization may exceed 2h for 40+ iterations
        ),
    ]


def get_inverse_design_orchestration_tools() -> List[Tool]:
    """Return Step1-Step5 chain plus explicit MCP helper tools."""

    return get_inverse_design_step_tools() + get_inverse_design_mcp_tools()
