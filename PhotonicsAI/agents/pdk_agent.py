"""PDK Agent assembly for generation and Tidy3D optimization workflows."""

from __future__ import annotations

import os
from copy import deepcopy
import inspect
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Tuple

from PhotonicsAI.agents.react_loop import Memory, ReActEngine, build_recovery_event
from PhotonicsAI.agents.llm_client_factory import build_tool_calling_client

if TYPE_CHECKING:
    from PhotonicsAI.tools.inverse_design_types import FailureDiagnosis


SYSTEM_PROMPT_TEMPLATE = """
You are PhIDO PDK Agent.
Follow four phases:
1) component generation,
2) simulation setup,
3) optimization with reflection after each simulation,
4) result summary.
Stop when target metrics are reached, or iterations >= 8, or no meaningful gain.

Available tools:
{tool_descriptions}
""".strip()

INVERSE_DESIGN_SYSTEM_PROMPT_TEMPLATE = """
You are PhIDO inverse-design orchestration agent.
Always orchestrate the explicit Step1-Step5 chain through tools:
1) inverse_step1_parse_requirements
2) inverse_step2_retrieve_doc_context
3) inverse_step3_generate_config
4) inverse_step4_validate_config
5) inverse_step5_execute

Stepwise MCP policy (Step1-Step6):
- Step1: do not call MCP tools.
- Step2: use inverse_mcp_search_flexcompute_docs / inverse_mcp_fetch_flexcompute_doc only when evidence is missing or conflicting.
- Step3: generate config first; call MCP docs tools only to fill missing documented rules.
- Step4: run validation first; use inverse_mcp_validate_simulation only when file/viewer_id exists.
- Step5: execute optimization through inverse_step5_execute; do not replace Step5 with raw MCP calls.
- Step6 (recovery): use MCP docs/viewer helpers to gather evidence, then return to Step tools for rollback + replan.
- Step7 is not implemented in this repository. Never claim Step7 execution.

Policy:
- LLM-first: keep planner/reviewer/diagnosis enabled.
- Fallback is allowed only when tool or model execution fails.
- Keep each step output machine-readable for downstream steps.
- A run is complete only if inverse_step5_execute returns ok=true with a completed status.
- Do not skip step order unless rollback recovery explicitly re-enters from Step2/Step3/Step4.

Available Step tools:
{step_tool_descriptions}

Available MCP tools:
{mcp_tool_descriptions}
""".strip()

# Default model for all LLM-driven planning.
_DEFAULT_MODEL = os.getenv("DEFAULT_LLM_MODEL", "").strip() or os.getenv("RESOURCEPACK_DEFAULT_MODEL", "gpt-5.4")


class PDKAgent:
    """Agent facade exposing event-stream workflows for Streamlit integration."""

    def __init__(self) -> None:
        from PhotonicsAI.tools.pdk_tools import get_pdk_tools
        from PhotonicsAI.tools.tidy3d_tools import get_tidy3d_tools
        from PhotonicsAI.tools.inverse_design_steps import (
            get_inverse_design_mcp_tools,
            get_inverse_design_orchestration_tools,
            get_inverse_design_step_tools,
        )

        tools = get_pdk_tools() + get_tidy3d_tools()
        inverse_step_tools = get_inverse_design_step_tools()
        inverse_mcp_tools = get_inverse_design_mcp_tools()
        inverse_tools = get_inverse_design_orchestration_tools()
        self.memory = Memory()

        client_spec = build_tool_calling_client(_DEFAULT_MODEL)
        self.llm_client = client_spec.client
        self.model = client_spec.model
        self.llm_backend = client_spec.backend
        self.llm_backend_reason = client_spec.reason

        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            tool_descriptions="\n".join(f"- {tool.name}: {tool.description}" for tool in tools)
        )
        self.engine = ReActEngine(
            tools=tools,
            system_prompt=system_prompt,
            llm_client=self.llm_client,
            max_steps=30,
            reflection_trigger=lambda name, _: name == "run_tidy3d_simulation",
        )
        inverse_system_prompt = INVERSE_DESIGN_SYSTEM_PROMPT_TEMPLATE.format(
            step_tool_descriptions="\n".join(
                f"- {tool.name}: {tool.description}" for tool in inverse_step_tools
            ),
            mcp_tool_descriptions="\n".join(
                f"- {tool.name}: {tool.description}" for tool in inverse_mcp_tools
            ),
        )
        self.inverse_engine = ReActEngine(
            tools=inverse_tools,
            system_prompt=inverse_system_prompt,
            llm_client=self.llm_client,
            max_steps=40,
        )
        self.system_prompt = system_prompt
        self.inverse_system_prompt = inverse_system_prompt

    def _emit(self, event: str, **payload: Any) -> Dict[str, Any]:
        return {"event": event, **payload}

    def run_optimization(
        self,
        component_type: str,
        template_path: str,
        wavelength_nm: float = 1550.0,
    ) -> Generator[Dict[str, Any], None, Dict[str, Any]]:
        self.memory.add_message("system", self.system_prompt)
        self.memory.add_message(
            "user",
            f"Optimize component={component_type} using template={template_path}",
        )

        # Phase 1: parse template defaults (unchanged).
        plan = [
            {
                "tool": "parse_template_defaults",
                "args": {"template_path": template_path},
                "thought": "Parsing template defaults for simulation parameters.",
            },
        ]

        parsed = None
        for event in self.engine.run_plan(self.memory, plan):
            if event.get("event") == "observation":
                parsed = event.get("result")
            if event.get("event") == "answer":
                break
            yield event

        if not parsed or not parsed.get("ok"):
            answer = {
                "ok": False,
                "error": (parsed or {}).get("error") or "Failed to parse template defaults.",
            }
            yield self._emit("answer", result=answer)
            return answer

        params = parsed.get("data", {}).get("params", {})

        # Phase 2: multi-round optimisation loop.
        from PhotonicsAI.agents.param_perturbation import perturb_params

        def build_iteration_plan(
            current_params: Dict[str, Any], iteration: int
        ) -> list:
            return [
                {
                    "tool": "run_tidy3d_simulation",
                    "args": {
                        "component_type": component_type,
                        "parameters": current_params,
                        "wavelength_nm": wavelength_nm,
                    },
                    "thought": f"Iteration {iteration}: running Tidy3D simulation.",
                    "reflection": (
                        "<reflection>Inspect simulation artifacts and score "
                        "before deciding next parameter changes.</reflection>"
                    ),
                },
                {
                    "tool": "check_simulation",
                    "args": {"component_type": component_type},
                    "thought": f"Iteration {iteration}: validating simulation output.",
                },
                {
                    "tool": "start_viewer",
                    "args": {"component_type": component_type},
                    "thought": f"Iteration {iteration}: opening 3D viewer (MCP or local).",
                },
                {
                    "tool": "plot_results",
                    "args": {"component_type": component_type},
                    "thought": f"Iteration {iteration}: collecting field/flux/structure plots.",
                    "stop_on_error": False,
                },
                {
                    "tool": "capture_viewer",
                    "args": {"component_type": component_type},
                    "thought": f"Iteration {iteration}: capturing best visualization.",
                    "stop_on_error": False,
                },
            ]

        for event in self.engine.run_optimization_loop(
            memory=self.memory,
            initial_params=params,
            build_iteration_plan=build_iteration_plan,
            perturb_params=perturb_params,
            max_iterations=8,
            convergence_threshold=0.05,
        ):
            yield event

    def run_generation(
        self,
        component_name: str,
        provided_pdf_path: str,
    ) -> Generator[Dict[str, Any], None, Dict[str, Any]]:
        """Run the generation workflow.

        Delegates to LLM-driven planning when ``llm_client`` is available,
        otherwise falls back to the deterministic path.
        """
        if self.llm_client is not None:
            return self.run_generation_llm(component_name, provided_pdf_path)
        return self._run_generation_deterministic(component_name, provided_pdf_path)

    def run_generation_llm(
        self,
        component_name: str,
        provided_pdf_path: str,
    ) -> Generator[Dict[str, Any], None, Dict[str, Any]]:
        """LLM-driven generation: GPT-5.4 decides which tools to call."""
        user_input = (
            f"Generate a new component template for component={component_name} "
            f"using pdf={provided_pdf_path}.\n"
            "Use the available tools in the right order: "
            "resolve_device_type → retrieve_papers_multi_source → "
            "extract_or_aggregate_params → generate_template_file."
        )

        last_answer: Dict[str, Any] = {"ok": False, "error": "No answer produced."}
        for event in self.engine.run(self.memory, user_input, model=self.model):
            if event.get("event") == "answer":
                last_answer = event.get("result", last_answer)
            yield event

        return last_answer

    def _run_generation_deterministic(
        self,
        component_name: str,
        provided_pdf_path: str,
    ) -> Generator[Dict[str, Any], None, Dict[str, Any]]:
        self.memory.add_message("system", self.system_prompt)
        self.memory.add_message(
            "user",
            f"Generate a new component template for component={component_name} using pdf={provided_pdf_path}",
        )

        yield self._emit("thought", content="Resolving device type from requested component name.")
        resolved = self.engine.execute_tool(
            self.memory,
            "resolve_device_type",
            component_name=component_name,
        )
        yield self._emit("observation", tool="resolve_device_type", result=resolved)
        if not resolved.get("ok"):
            answer = {"ok": False, "error": resolved.get("error")}
            yield self._emit("answer", result=answer)
            return answer

        device_type = resolved.get("data", {}).get("device_type", "")

        yield self._emit("action", tool="retrieve_papers_multi_source", args={"device_type": device_type, "provided_pdf_path": provided_pdf_path})
        papers_result = self.engine.execute_tool(
            self.memory,
            "retrieve_papers_multi_source",
            device_type=device_type,
            provided_pdf_path=provided_pdf_path,
        )
        yield self._emit("observation", tool="retrieve_papers_multi_source", result=papers_result)
        if not papers_result.get("ok"):
            answer = {"ok": False, "error": papers_result.get("error")}
            yield self._emit("answer", result=answer)
            return answer

        papers = papers_result.get("data", {}).get("papers", [])
        yield self._emit("action", tool="extract_or_aggregate_params", args={"papers": papers, "device_type": device_type})
        params_result = self.engine.execute_tool(
            self.memory,
            "extract_or_aggregate_params",
            papers=papers,
            device_type=device_type,
        )
        yield self._emit("observation", tool="extract_or_aggregate_params", result=params_result)
        if not params_result.get("ok"):
            answer = {"ok": False, "error": params_result.get("error")}
            yield self._emit("answer", result=answer)
            return answer

        params = params_result.get("data", {}).get("params", {})
        confidence_note = params_result.get("data", {}).get("confidence_note", "")
        yield self._emit(
            "action",
            tool="generate_template_file",
            args={
                "device_type": device_type,
                "params": params,
                "papers": papers,
                "confidence_note": confidence_note,
            },
        )
        template_result = self.engine.execute_tool(
            self.memory,
            "generate_template_file",
            device_type=device_type,
            params=params,
            papers=papers,
            confidence_note=confidence_note,
        )
        yield self._emit("observation", tool="generate_template_file", result=template_result)

        answer = {
            "ok": bool(template_result.get("ok")),
            "component_name": component_name,
            "device_type": device_type,
            "papers": papers_result,
            "params": params_result,
            "template": template_result,
        }
        yield self._emit("answer", result=answer)
        return answer

    def run_generation_and_optimization(
        self,
        component_name: str,
        provided_pdf_path: str,
        wavelength_nm: float = 1550.0,
    ) -> Generator[Dict[str, Any], None, Dict[str, Any]]:
        generation_answer = None
        for event in self.run_generation(component_name, provided_pdf_path):
            if event.get("event") == "answer":
                generation_answer = event.get("result")
            yield event

        if not generation_answer or not generation_answer.get("ok"):
            answer = generation_answer or {"ok": False, "error": "Generation failed."}
            yield self._emit("answer", result=answer)
            return answer

        device_type = generation_answer.get("device_type") or component_name
        template_path = (
            generation_answer.get("template", {})
            .get("data", {})
            .get("filepath", "")
        )
        optimization_answer = None
        for event in self.run_optimization(device_type, template_path, wavelength_nm):
            if event.get("event") == "answer":
                optimization_answer = event.get("result")
            yield event

        final_answer = {
            "ok": bool(optimization_answer and optimization_answer.get("ok")),
            "generation": generation_answer,
            "optimization": optimization_answer,
        }
        yield self._emit("answer", result=final_answer)
        return final_answer

    def run_inverse_design_react(
        self,
        requirement_text: str,
        *,
        max_iterations: int = 3,
        max_recovery_attempts: int = 2,
        rollback_confidence_threshold: float = 0.7,
        recover_on_target_miss: bool | None = None,
    ) -> Generator[Dict[str, Any], None, Dict[str, Any]]:
        """Run Step1-Step5 inverse-design through the ReAct main chain."""

        workflow_memory = Memory()
        user_input = (
            "Execute inverse-design workflow via explicit step tools. "
            f"Requirement: {requirement_text}\n"
            f"Use max_iterations={max_iterations}. "
            "Call step tools in order unless a tool failure requires fallback repair."
        )
        force_deterministic_chain = os.getenv("INVERSE_FORCE_DETERMINISTIC_CHAIN", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        disable_deterministic_fallback = os.getenv(
            "INVERSE_DISABLE_DETERMINISTIC_FALLBACK", "0"
        ).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if self.llm_client is not None and not force_deterministic_chain:
            last_answer: Dict[str, Any] = {"ok": False, "error": "No answer produced."}
            step_outputs: Dict[str, Dict[str, Any]] = {}
            used_mcp_tools: List[str] = []
            llm_path_broken = False
            llm_path_break_reason = ""
            # Seed the workflow state with the requirement text so Step1
            # receives it even if the LLM omits it from the tool args.
            initial_ws = {"requirement_text": requirement_text}
            for event in self.inverse_engine.run(
                workflow_memory,
                user_input,
                model=self.model,
                initial_workflow_state=initial_ws,
            ):
                if event.get("event") == "action":
                    tool_name = str(event.get("tool") or "")
                    if tool_name.startswith("inverse_mcp_") and tool_name not in used_mcp_tools:
                        used_mcp_tools.append(tool_name)
                if event.get("event") == "observation":
                    tool_name = str(event.get("tool") or "")
                    result = event.get("result", {})
                    step_key = self._tool_to_step_key(tool_name)
                    if step_key:
                        step_outputs[step_key] = result
                        if not bool(result.get("ok")):
                            llm_path_broken = True
                            llm_path_break_reason = (
                                f"{step_key} failed on LLM path: "
                                f"{result.get('error') or 'unknown error'}"
                            )
                    elif tool_name.startswith("inverse_mcp_") and tool_name not in used_mcp_tools:
                        used_mcp_tools.append(tool_name)
                if event.get("event") == "answer":
                    last_answer = event.get("result", last_answer)
                yield event
                if llm_path_broken:
                    break
            step5 = step_outputs.get("step5", {})
            run_result = step5.get("data", {}).get("run_result", {}) if isinstance(step5, dict) else {}
            if not llm_path_broken and step5.get("ok") and run_result.get("status") == "completed":
                answer = {
                    "ok": True,
                    "error": None,
                    "steps": step_outputs,
                    "run_result": run_result,
                    "recovery": [],
                    "llm_main_path_used": True,
                    "llm_client_backend": self.llm_backend,
                    "llm_client_reason": self.llm_backend_reason,
                    "used_mcp_tools": used_mcp_tools,
                }
                yield self._emit("answer", result=answer)
                return answer
            if disable_deterministic_fallback:
                answer = {
                    "ok": False,
                    "error": llm_path_break_reason or (last_answer.get("error") if isinstance(last_answer, dict) else "LLM main path did not complete."),
                    "steps": step_outputs,
                    "run_result": run_result if isinstance(run_result, dict) else {},
                    "recovery": [],
                    "llm_main_path_used": True,
                    "llm_client_backend": self.llm_backend,
                    "llm_client_reason": self.llm_backend_reason,
                    "used_mcp_tools": used_mcp_tools,
                    "fallback_skipped": True,
                }
                yield self._emit(
                    "thought",
                    content=(
                        "LLM main path failed and deterministic fallback is disabled "
                        "(INVERSE_DISABLE_DETERMINISTIC_FALLBACK=1)."
                    ),
                )
                yield self._emit("answer", result=answer)
                return answer

            yield self._emit(
                "thought",
                content=(
                    "Inverse ReAct LLM path did not return a successful result. "
                    "Switching to deterministic ReAct fallback chain with recovery, "
                    "reusing existing step outputs from LLM path."
                    + (f" {llm_path_break_reason}" if llm_path_break_reason else "")
                )
            )
        else:
            step_outputs = {}
            deterministic_reason = (
                "forced by INVERSE_FORCE_DETERMINISTIC_CHAIN=1"
                if force_deterministic_chain and self.llm_client is not None
                else "LLM client unavailable"
            )
            yield self._emit(
                "thought",
                content=f"Running deterministic ReAct fallback chain ({deterministic_reason}).",
            )

        # Determine the earliest failed step so recovery resumes from there
        # rather than wastefully re-running successful upstream steps.
        _resume_step = "step1"
        if step_outputs:
            for _sk in ("step1", "step2", "step3", "step4", "step5"):
                _sr = step_outputs.get(_sk, {})
                if not _sr or not _sr.get("ok"):
                    _resume_step = _sk
                    break
            else:
                _resume_step = "step5"  # all ok but Step5 not completed

        fallback_answer = None
        for event in self._run_inverse_design_with_recovery(
            requirement_text=requirement_text,
            max_iterations=max_iterations,
            max_recovery_attempts=max_recovery_attempts,
            rollback_confidence_threshold=rollback_confidence_threshold,
            recover_on_target_miss=recover_on_target_miss,
            initial_step_outputs=step_outputs if step_outputs else None,
            resume_from_step=_resume_step if step_outputs else "step1",
        ):
            if event.get("event") == "answer":
                fallback_answer = event.get("result")
            yield event

        return fallback_answer or {"ok": False, "error": "Inverse-design fallback did not complete."}

    @staticmethod
    def _tool_to_step_key(tool_name: str) -> str | None:
        mapping = {
            "inverse_step1_parse_requirements": "step1",
            "inverse_step2_retrieve_doc_context": "step2",
            "inverse_step3_generate_config": "step3",
            "inverse_step4_validate_config": "step4",
            "inverse_step5_execute": "step5",
        }
        return mapping.get(tool_name)

    def _run_inverse_design_with_recovery(
        self,
        *,
        requirement_text: str,
        max_iterations: int,
        max_recovery_attempts: int,
        rollback_confidence_threshold: float,
        recover_on_target_miss: bool | None = None,
        initial_step_outputs: Dict[str, Dict[str, Any]] | None = None,
        resume_from_step: str = "step1",
    ) -> Generator[Dict[str, Any], None, Dict[str, Any]]:
        """Deterministic chain with section-11 recovery orchestration (R1-R6).

        When ``initial_step_outputs`` is provided (e.g. reused from the ReAct
        LLM path), the initial run resumes from ``resume_from_step`` instead of
        re-running the entire Step1-Step5 sequence from scratch.
        """

        from PhotonicsAI.tools.inverse_design_failure_diagnosis import diagnose_inverse_design_failure
        from PhotonicsAI.tools.inverse_design_working_memory import get_inverse_design_working_memory

        step_outputs: Dict[str, Dict[str, Any]] = dict(initial_step_outputs) if initial_step_outputs else {}
        chain_memory = Memory()
        working_memory = get_inverse_design_working_memory()
        recovery_history: List[Dict[str, Any]] = []
        attempts_by_step: Dict[str, int] = {}

        # KL-1: Budget guard — cap total cloud tasks across all recovery
        # attempts to avoid runaway cloud credit consumption.
        cloud_tasks_budget = max(
            max_iterations,
            self._estimate_cloud_tasks_budget(
                step_outputs=step_outputs,
                max_iterations=max_iterations,
            ),
        )
        cloud_tasks_spent = 0

        # Reusing a failed Step5 snapshot from the LLM path should not trigger
        # an immediate duplicate Step5 cloud rerun before recovery diagnosis.
        # Otherwise we burn extra credits and overwrite the more informative
        # first-failure trace.
        skip_initial_step5_rerun = (
            bool(initial_step_outputs)
            and resume_from_step == "step5"
            and isinstance(step_outputs.get("step5"), dict)
            and bool(step_outputs.get("step5"))
        )
        if skip_initial_step5_rerun:
            yield self._emit(
                "thought",
                content=(
                    "Reusing existing Step5 output from prior LLM execution; "
                    "skipping duplicate Step5 rerun before recovery diagnosis."
                ),
            )
        else:
            yield from self._run_inverse_step_sequence(
                chain_memory=chain_memory,
                step_outputs=step_outputs,
                requirement_text=requirement_text,
                max_iterations=max_iterations,
                start_step=resume_from_step,
            )

        # Count initial run's iteration expenditure toward the budget.
        cloud_tasks_budget = max(
            cloud_tasks_budget,
            self._estimate_cloud_tasks_budget(
                step_outputs=step_outputs,
                max_iterations=max_iterations,
            ),
        )
        cloud_tasks_spent += self._estimate_step5_cloud_tasks_spent(step_outputs)

        recover_target_miss = recover_on_target_miss
        if recover_target_miss is None:
            recover_target_miss = os.getenv("INVERSE_RECOVER_ON_TARGET_MISS", "1").strip().lower() not in {
                "0",
                "false",
                "no",
                "off",
            }

        if not self._step5_needs_recovery(step_outputs, recover_on_target_miss=recover_target_miss):
            run_result = step_outputs.get("step5", {}).get("data", {}).get("run_result", {})
            answer = {
                "ok": bool(step_outputs.get("step5", {}).get("ok")),
                "error": step_outputs.get("step5", {}).get("error"),
                "steps": step_outputs,
                "run_result": run_result,
                "recovery": recovery_history,
            }
            yield self._emit("answer", result=answer)
            return answer

        low_credit_abort_reason = self._low_credit_recovery_abort_reason()
        if low_credit_abort_reason:
            recovery_history.append(
                {
                    "failure_signature": "resource_limit_low_credit_guard",
                    "failed_step": "step5",
                    "rollback_candidates": [],
                    "selected_rollback_step": "",
                    "selected_repair_candidate_id": "",
                    "replan_actions": [],
                    "resume_from_step": "",
                    "recovery_attempt": 0,
                    "recovery_outcome": "aborted_low_credit",
                    "selection_reason": low_credit_abort_reason,
                    "requires_doc_refresh": False,
                    "confidence": 1.0,
                    "evidence_refs": [],
                    "scenario_fingerprint": self._extract_step5_scenario_fingerprint(step_outputs),
                    "trial_results": [],
                }
            )
            yield self._emit("thought", content=low_credit_abort_reason)
            final_step5 = step_outputs.get("step5", {})
            run_result = final_step5.get("data", {}).get("run_result", {})
            answer = {
                "ok": bool(final_step5.get("ok"))
                and not self._step5_needs_recovery(
                    step_outputs,
                    recover_on_target_miss=recover_target_miss,
                ),
                "error": final_step5.get("error") or self._derive_failure_message(step_outputs),
                "steps": step_outputs,
                "run_result": run_result,
                "recovery": recovery_history,
            }
            yield self._emit("answer", result=answer)
            return answer

        max_attempts = max(0, int(max_recovery_attempts))
        if max_attempts > 0:
            per_attempt_floor = max(1, self._step5_cloud_tasks_per_iteration(step_outputs))
            cloud_tasks_budget = max(cloud_tasks_budget, max_attempts * per_attempt_floor)
        if max_attempts == 0:
            final_step5 = step_outputs.get("step5", {})
            answer = {
                "ok": False,
                "error": final_step5.get("error") or "Step5 failed and recovery is disabled (max_recovery_attempts=0).",
                "steps": step_outputs,
                "run_result": final_step5.get("data", {}).get("run_result", {}),
                "recovery": recovery_history,
            }
            yield self._emit(
                "thought",
                content=(
                    "Recovery disabled by configuration (max_recovery_attempts=0); "
                    "returning first Step5 failure without rerun to protect cloud budget."
                ),
            )
            yield self._emit("answer", result=answer)
            return answer

        for recovery_attempt in range(1, max_attempts + 1):
            failure_message = self._derive_failure_message(step_outputs)
            objective_metric = (
                step_outputs.get("step3", {})
                .get("data", {})
                .get("objective_metric", "")
            )
            objective_goal = (
                step_outputs.get("step3", {})
                .get("data", {})
                .get("config_bundle", {})
                .get("optimization_config", {})
                .get("objective", {})
                .get("goal", "")
            )
            runtime_config = self._extract_step5_runtime_config(step_outputs)
            diagnosis = self._extract_step5_failure_diagnosis(step_outputs)
            if diagnosis is None:
                diagnosis = diagnose_inverse_design_failure(
                    failure_message,
                    component_type=(
                        step_outputs.get("step3", {})
                        .get("data", {})
                        .get("component_type", "")
                    ),
                    objective_metric=objective_metric,
                    objective_goal=str(objective_goal),
                    recent_issues=self._collect_recent_step4_issues(step_outputs),
                    recent_iterations=self._extract_recent_iterations(step_outputs),
                    checkpoint_report=self._extract_step5_checkpoint_report(step_outputs),
                    checkpoint_reports=self._extract_step5_checkpoint_reports(step_outputs),
                    scenario_fingerprint=self._extract_step5_scenario_fingerprint(step_outputs),
                    failed_step="step5",
                    recovery_attempt=recovery_attempt,
                    confidence_threshold=rollback_confidence_threshold,
                    use_llm_advisor=True,
                    llm_model=self.model,
                    memory_store=working_memory,
                    enable_optimizer_attribution=bool(runtime_config.get("enable_optimizer_attribution", False)),
                    optimizer_attribution_min_samples=int(runtime_config.get("optimizer_attribution_min_samples", 12) or 12),
                )
            yield build_recovery_event(
                "recovery_diagnosis",
                step="R2",
                recovery_attempt=recovery_attempt,
                failure_signature=diagnosis.failure_signature,
                diagnosis=diagnosis.model_dump(),
            )

            # KL-1: Even when the recovery budget is already exhausted, emit one
            # structured diagnosis event for the current failure state before
            # stopping the loop. This preserves recovery observability and keeps
            # the event contract aligned with the documented R1-R6 sequence.
            if cloud_tasks_spent >= cloud_tasks_budget:
                budget_reason = (
                    f"Cloud task budget exhausted: {cloud_tasks_spent}/{cloud_tasks_budget} "
                    "tasks consumed. Aborting recovery to prevent runaway costs."
                )
                budget_record = self._build_recovery_record(
                    diagnosis=diagnosis,
                    selected_rollback_step="",
                    replan_actions=[],
                    resume_from_step="",
                    recovery_attempt=recovery_attempt,
                    recovery_outcome="budget_exhausted",
                    selection_reason=budget_reason,
                )
                recovery_history.append(budget_record)
                yield self._emit(
                    "thought",
                    content=(
                        f"Recovery budget exhausted ({cloud_tasks_spent}/{cloud_tasks_budget} "
                        "cloud tasks). Stopping recovery loop."
                    ),
                )
                break

            selected_step, selection_reason = self._select_rollback_step(
                diagnosis=diagnosis,
                confidence_threshold=rollback_confidence_threshold,
                attempts_by_step=attempts_by_step,
            )
            candidate_dump = [item.model_dump() for item in diagnosis.rollback_candidates]
            yield build_recovery_event(
                "rollback_selection",
                step="R3",
                recovery_attempt=recovery_attempt,
                failure_signature=diagnosis.failure_signature,
                rollback_candidates=candidate_dump,
                selected_rollback_step=selected_step,
                selection_reason=selection_reason,
            )

            if not selected_step:
                record = self._build_recovery_record(
                    diagnosis=diagnosis,
                    selected_rollback_step="",
                    replan_actions=[],
                    resume_from_step="",
                    recovery_attempt=recovery_attempt,
                    recovery_outcome="fallback",
                    selection_reason=selection_reason,
                )
                recovery_history.append(record)
                working_memory.record(
                    stage="recovery",
                    key=requirement_text[:80],
                    failure_signature=diagnosis.failure_signature,
                    scenario_fingerprint=self._scenario_fingerprint_key_from_step_outputs(step_outputs),
                    summary=f"Recovery attempt {recovery_attempt} fell back without rollback target.",
                    evidence_urls=list(diagnosis.evidence_urls),
                    issues=[diagnosis.category, diagnosis.error_family, selection_reason],
                    proposed_fixes=[action.path for action in diagnosis.replan_actions],
                    metadata=record,
                )
                break

            attempts_by_step[selected_step] = attempts_by_step.get(selected_step, 0) + 1
            selected_candidate, trial_results = self._evaluate_repair_candidates(
                chain_memory=chain_memory,
                requirement_text=requirement_text,
                step_outputs=step_outputs,
                diagnosis=diagnosis,
                selected_step=selected_step,
                max_iterations=max_iterations,
                remaining_cloud_tasks_budget=max(0, cloud_tasks_budget - cloud_tasks_spent),
            )
            trial_cloud_tasks_spent = int(getattr(self, "_last_trial_cloud_tasks_spent", 0) or 0)
            cloud_tasks_spent += trial_cloud_tasks_spent
            replan_actions = (
                list(selected_candidate.patch_actions)
                if selected_candidate is not None and selected_candidate.patch_actions
                else list(diagnosis.replan_actions or diagnosis.repair_actions)
            )
            selected_repair_candidate_id = (
                selected_candidate.candidate_id
                if selected_candidate is not None
                else diagnosis.selected_repair_candidate_id
            )
            applied_paths = [action.path for action in replan_actions if getattr(action, "path", "")]
            recovery_context = self._build_recovery_context(
                diagnosis=diagnosis,
                selected_step=selected_step,
                replan_actions=replan_actions,
                selected_repair_candidate_id=selected_repair_candidate_id,
                trial_results=trial_results,
            )
            yield build_recovery_event(
                "replan",
                step="R4",
                recovery_attempt=recovery_attempt,
                failure_signature=diagnosis.failure_signature,
                selected_rollback_step=selected_step,
                selected_repair_candidate_id=selected_repair_candidate_id,
                replan_actions=[action.model_dump() for action in replan_actions],
                applied_paths=applied_paths,
                trial_results=trial_results,
            )

            yield build_recovery_event(
                "resume",
                step="R5",
                recovery_attempt=recovery_attempt,
                failure_signature=diagnosis.failure_signature,
                resume_from_step=selected_step,
            )
            yield from self._run_inverse_step_sequence(
                chain_memory=chain_memory,
                step_outputs=step_outputs,
                requirement_text=requirement_text,
                # Recovery re-runs use a reduced iteration budget when the
                # original count is large (>10) to avoid burning N × full
                # cloud tasks per recovery cycle.  For small counts, keep
                # the original so the optimizer has enough room to converge.
                max_iterations=min(max_iterations, max(10, max_iterations // 2)),
                start_step=selected_step,
                recovery_context=recovery_context,
            )

            # KL-1: Track recovery iteration expenditure toward budget.
            cloud_tasks_spent += self._estimate_step5_cloud_tasks_spent(step_outputs)

            recovered = not self._step5_needs_recovery(
                step_outputs,
                recover_on_target_miss=recover_target_miss,
            )
            outcome = "success" if recovered else ("retry" if recovery_attempt < max_attempts else "escalate")
            record = self._build_recovery_record(
                diagnosis=diagnosis,
                selected_rollback_step=selected_step,
                replan_actions=replan_actions,
                resume_from_step=selected_step,
                recovery_attempt=recovery_attempt,
                recovery_outcome=outcome,
                selection_reason=selection_reason,
                selected_repair_candidate_id=selected_repair_candidate_id,
                trial_results=trial_results,
            )
            recovery_history.append(record)
            working_memory.record(
                stage="recovery",
                key=requirement_text[:80],
                failure_signature=diagnosis.failure_signature,
                scenario_fingerprint=self._scenario_fingerprint_key_from_step_outputs(step_outputs),
                summary=f"Recovery attempt {recovery_attempt} outcome={outcome} via {selected_step}.",
                evidence_urls=list(diagnosis.evidence_urls),
                issues=[diagnosis.category, diagnosis.error_family],
                proposed_fixes=[action.path for action in replan_actions],
                metadata=record,
            )

            if recovered:
                break

        final_step5 = step_outputs.get("step5", {})
        run_result = final_step5.get("data", {}).get("run_result", {})
        answer = {
            "ok": bool(final_step5.get("ok"))
            and not self._step5_needs_recovery(
                step_outputs,
                recover_on_target_miss=recover_target_miss,
            ),
            "error": final_step5.get("error") or self._derive_failure_message(step_outputs),
            "steps": step_outputs,
            "run_result": run_result,
            "recovery": recovery_history,
        }
        yield self._emit("answer", result=answer)
        return answer

    def _run_inverse_step_sequence(
        self,
        *,
        chain_memory: Memory,
        step_outputs: Dict[str, Dict[str, Any]],
        requirement_text: str,
        max_iterations: int,
        start_step: str,
        recovery_context: Dict[str, Any] | None = None,
        use_llm_step_tools: bool = False,
    ) -> Generator[Dict[str, Any], None, Dict[str, Any]]:
        """Execute Step1-Step5 from ``start_step`` while preserving upstream outputs."""

        step_plan: List[Tuple[str, str, str]] = [
            ("step1", "inverse_step1_parse_requirements", "Step1: parsing inverse-design requirement."),
            ("step2", "inverse_step2_retrieve_doc_context", "Step2: retrieving MCP documentation context."),
            ("step3", "inverse_step3_generate_config", "Step3: generating simulation and optimization config."),
            ("step4", "inverse_step4_validate_config", "Step4: validating config with RAG hard constraints and semantic checks."),
            ("step5", "inverse_step5_execute", "Step5: executing inverse-design optimization and diagnosis loop."),
        ]
        step_names = [item[0] for item in step_plan]
        if start_step not in step_names:
            raise ValueError(f"Unknown start_step `{start_step}` for inverse step sequence.")

        start_index = step_names.index(start_step)
        for step_key, tool_name, thought in step_plan[start_index:]:
            yield self._emit("thought", content=thought)
            try:
                kwargs = self._build_inverse_step_kwargs(
                    step_key=step_key,
                    step_outputs=step_outputs,
                    requirement_text=requirement_text,
                    max_iterations=max_iterations,
                    recovery_context=recovery_context,
                    use_llm_step_tools=use_llm_step_tools,
                )
            except Exception as exc:
                result = {"ok": False, "data": {}, "error": f"{step_key} input dependency error: {exc}"}
                step_outputs[step_key] = result
                yield self._emit("observation", tool=tool_name, result=result)
                return result

            yield self._emit("action", tool=tool_name, args=kwargs)
            result = self._execute_inverse_tool(
                chain_memory=chain_memory,
                tool_name=tool_name,
                kwargs=kwargs,
            )
            step_outputs[step_key] = result
            yield self._emit("observation", tool=tool_name, result=result)
            if not result.get("ok"):
                return result

        return step_outputs.get("step5", {"ok": False, "data": {}, "error": "Step5 not executed."})

    def _build_inverse_step_kwargs(
        self,
        *,
        step_key: str,
        step_outputs: Dict[str, Dict[str, Any]],
        requirement_text: str,
        max_iterations: int,
        recovery_context: Dict[str, Any] | None = None,
        use_llm_step_tools: bool = False,
    ) -> Dict[str, Any]:
        if step_key == "step1":
            return {
                "requirement_text": requirement_text,
                "require_complete": True,
                "use_llm_parser": use_llm_step_tools,
                "llm_model": self.model,
            }

        if step_key == "step2":
            requirement = (
                step_outputs.get("step1", {})
                .get("data", {})
                .get("requirement")
            )
            if requirement is None:
                raise ValueError("missing step1.requirement")
            max_results_env = (
                "INVERSE_STEP2_MAX_RESULTS"
                if use_llm_step_tools
                else "INVERSE_FALLBACK_STEP2_MAX_RESULTS"
            )
            try:
                max_results = int(os.getenv(max_results_env, "3"))
            except Exception:
                max_results = 3
            max_results = max(0, max_results)
            return {
                "requirement": requirement,
                "max_results": max_results,
                "use_llm_planner": use_llm_step_tools,
                "llm_model": self.model,
                "selected_rollback_step": (recovery_context or {}).get("selected_rollback_step", ""),
                "recovery_context": recovery_context,
                "workflow_state": step_outputs,
            }

        if step_key == "step3":
            doc_context = (
                step_outputs.get("step2", {})
                .get("data", {})
                .get("doc_context")
            )
            if doc_context is None:
                raise ValueError("missing step2.doc_context")
            return {
                "doc_context": doc_context,
                "use_llm_planner": use_llm_step_tools,
                "llm_model": self.model,
                "selected_rollback_step": (recovery_context or {}).get("selected_rollback_step", ""),
                "recovery_context": recovery_context,
                "workflow_state": step_outputs,
            }

        if step_key == "step4":
            config_bundle = (
                step_outputs.get("step3", {})
                .get("data", {})
                .get("config_bundle")
            )
            doc_context = (
                step_outputs.get("step2", {})
                .get("data", {})
                .get("doc_context")
            )
            if config_bundle is None:
                raise ValueError("missing step3.config_bundle")
            return {
                "config_bundle": config_bundle,
                "doc_context": doc_context,
                "include_llm_review": use_llm_step_tools,
                "llm_model": self.model,
                "selected_rollback_step": (recovery_context or {}).get("selected_rollback_step", ""),
                "recovery_context": recovery_context,
                "workflow_state": step_outputs,
            }

        if step_key == "step5":
            from copy import deepcopy
            from PhotonicsAI.tools.inverse_design_replan import apply_patch_actions

            config_bundle = (
                step_outputs.get("step3", {})
                .get("data", {})
                .get("config_bundle")
            )
            constraint_packet = (
                step_outputs.get("step4", {})
                .get("data", {})
                .get("constraint_packet")
            )
            if config_bundle is None:
                raise ValueError("missing step3.config_bundle")

            # Apply Step4 repair_actions to config_bundle before Step5
            step4_validation = (
                step_outputs.get("step4", {})
                .get("data", {})
                .get("validation", {})
            )
            all_repairs = list(step4_validation.get("repair_actions", []))
            all_repairs.extend(step4_validation.get("review_repair_actions", []))
            if all_repairs:
                config_bundle = deepcopy(config_bundle)
                applied = apply_patch_actions(config_bundle, all_repairs)
                if applied:
                    print(f"[Step5-prep] Applied {len(applied)} Step4 repair actions: {applied}")

            step5_kwargs = {
                "config_bundle": config_bundle,
                "constraint_packet": constraint_packet,
                "max_iterations": max_iterations,
                "llm_model": self.model,
                "selected_rollback_step": (recovery_context or {}).get("selected_rollback_step", ""),
                "recovery_context": recovery_context,
                "workflow_state": step_outputs,
            }
            if not use_llm_step_tools:
                # Keep Step5 review policy override explicit only for fallback
                # execution. All other runtime defaults come from the bundle.
                step5_kwargs["include_llm_review"] = False
            return step5_kwargs

        raise ValueError(f"unsupported step_key `{step_key}`")

    def _extract_step5_failure_diagnosis(self, step_outputs: Dict[str, Dict[str, Any]]) -> Any | None:
        from PhotonicsAI.tools.inverse_design_types import FailureDiagnosis

        payload = (
            step_outputs.get("step5", {})
            .get("data", {})
            .get("run_result", {})
            .get("failure_diagnosis")
        )
        if not isinstance(payload, dict):
            return None
        try:
            return FailureDiagnosis.model_validate(payload)
        except Exception:
            return None

    def _extract_step5_runtime_config(self, step_outputs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        config_bundle = (
            step_outputs.get("step3", {})
            .get("data", {})
            .get("config_bundle", {})
        )
        if not isinstance(config_bundle, dict):
            return {}
        runtime_config = config_bundle.get("runtime_config", {})
        return dict(runtime_config) if isinstance(runtime_config, dict) else {}

    def _extract_step5_checkpoint_report(self, step_outputs: Dict[str, Dict[str, Any]]) -> Dict[str, Any] | None:
        reports = self._extract_step5_checkpoint_reports(step_outputs)
        return reports[0] if reports else None

    def _extract_step5_checkpoint_reports(self, step_outputs: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        run_result = step_outputs.get("step5", {}).get("data", {}).get("run_result", {})
        reports = run_result.get("checkpoint_reports", []) if isinstance(run_result, dict) else []
        return [item for item in reports if isinstance(item, dict)]

    def _extract_step5_scenario_fingerprint(self, step_outputs: Dict[str, Dict[str, Any]]) -> Dict[str, Any] | None:
        run_result = step_outputs.get("step5", {}).get("data", {}).get("run_result", {})
        fingerprint = run_result.get("scenario_fingerprint") if isinstance(run_result, dict) else None
        if isinstance(fingerprint, dict):
            return fingerprint
        return None

    def _extract_recent_iterations(self, step_outputs: Dict[str, Dict[str, Any]], limit: int = 5) -> List[Dict[str, Any]]:
        run_result = step_outputs.get("step5", {}).get("data", {}).get("run_result", {})
        iterations = run_result.get("iterations", []) if isinstance(run_result, dict) else []
        valid = [item for item in iterations if isinstance(item, dict)]
        return valid[-max(1, int(limit)) :]

    def _scenario_fingerprint_key_from_step_outputs(self, step_outputs: Dict[str, Dict[str, Any]]) -> str:
        fingerprint = self._extract_step5_scenario_fingerprint(step_outputs) or {}
        parts = [
            str(fingerprint.get("component_type", "")).strip(),
            str(fingerprint.get("objective_metric", "")).strip(),
            str(fingerprint.get("objective_goal", "")).strip(),
            str(fingerprint.get("wavelength_band", "")).strip(),
            str(fingerprint.get("domain_ratio", "")).strip(),
            str(fingerprint.get("monitor_topology_signature", "")).strip(),
            str(fingerprint.get("boundary_type", "")).strip(),
        ]
        return "|".join(part for part in parts if part)

    def _evaluate_repair_candidates(
        self,
        *,
        chain_memory: Memory,
        requirement_text: str,
        step_outputs: Dict[str, Dict[str, Any]],
        diagnosis: Any,
        selected_step: str,
        max_iterations: int,
        remaining_cloud_tasks_budget: int = 0,
    ) -> Tuple[Any | None, List[Dict[str, Any]]]:
        from PhotonicsAI.tools.inverse_design_working_memory import get_inverse_design_working_memory
        self._last_trial_cloud_tasks_spent = 0

        candidates = [
            candidate
            for candidate in list(getattr(diagnosis, "repair_candidates", []) or [])
            if getattr(candidate, "target_step", "") == selected_step
        ]
        if not candidates:
            return None, []

        if self._should_skip_repair_trials(diagnosis):
            self._last_trial_cloud_tasks_spent = 0
            return (
                None,
                [
                    {
                        "candidate_id": "",
                        "target_step": selected_step,
                        "short_run_iterations": 0,
                        "short_run_score": 0.0,
                        "passed": False,
                        "outcome": "skipped",
                        "reason": "Skipped repair trials for resource/transport failure to avoid credit amplification.",
                        "patch_actions": [],
                        "checkpoint_status": "skipped",
                    }
                ],
            )

        baseline_run_result = (
            step_outputs.get("step5", {})
            .get("data", {})
            .get("run_result", {})
        )
        short_run_iterations = max(3, min(5, int(max_iterations)))
        trial_results: List[Dict[str, Any]] = []
        best_candidate = None
        best_score = float("-inf")
        working_memory = get_inverse_design_working_memory()
        trial_cloud_tasks_spent = 0
        cloud_tasks_per_iteration = self._step5_cloud_tasks_per_iteration(step_outputs)
        per_trial_cloud_tasks = short_run_iterations * max(cloud_tasks_per_iteration, 1)

        for candidate in candidates[:3]:
            if (
                remaining_cloud_tasks_budget > 0
                and trial_cloud_tasks_spent + per_trial_cloud_tasks > remaining_cloud_tasks_budget
            ):
                trial_results.append(
                    {
                        "candidate_id": getattr(candidate, "candidate_id", ""),
                        "target_step": selected_step,
                        "short_run_iterations": 0,
                        "short_run_score": 0.0,
                        "passed": False,
                        "outcome": "skipped",
                        "reason": "Skipped repair trial due to cloud-task budget guard.",
                        "patch_actions": [
                            action.model_dump() if hasattr(action, "model_dump") else dict(action)
                            for action in list(getattr(candidate, "patch_actions", []) or [])
                        ],
                        "checkpoint_status": "skipped",
                    }
                )
                continue
            trial_step_outputs = deepcopy(step_outputs)
            trial_memory = Memory()
            recovery_context = self._build_recovery_context(
                diagnosis=diagnosis,
                selected_step=selected_step,
                replan_actions=list(getattr(candidate, "patch_actions", []) or []),
                selected_repair_candidate_id=getattr(candidate, "candidate_id", ""),
                trial_results=[],
            )
            for _ in self._run_inverse_step_sequence(
                chain_memory=trial_memory,
                step_outputs=trial_step_outputs,
                requirement_text=requirement_text,
                max_iterations=short_run_iterations,
                start_step=selected_step,
                recovery_context=recovery_context,
            ):
                pass

            trial_run_result = (
                trial_step_outputs.get("step5", {})
                .get("data", {})
                .get("run_result", {})
            )
            short_run_score = self._score_short_run_candidate(
                baseline_run_result=baseline_run_result,
                trial_run_result=trial_run_result,
            )
            passed = isinstance(trial_run_result, dict) and bool(trial_run_result.get("ok")) and trial_run_result.get("status") == "completed"
            trial_result = {
                "candidate_id": getattr(candidate, "candidate_id", ""),
                "target_step": selected_step,
                "short_run_iterations": short_run_iterations,
                "short_run_score": short_run_score,
                "passed": passed,
                "outcome": "success" if passed else "failed",
                "reason": str(trial_run_result.get("termination_reason", "") if isinstance(trial_run_result, dict) else ""),
                "patch_actions": [
                    action.model_dump() if hasattr(action, "model_dump") else dict(action)
                    for action in list(getattr(candidate, "patch_actions", []) or [])
                ],
                "checkpoint_status": self._extract_trial_checkpoint_status(trial_run_result),
            }
            analysis_metadata = dict(getattr(candidate, "analysis_metadata", {}) or {})
            trial_results.append(trial_result)
            trial_cloud_tasks_spent += self._estimate_step5_cloud_tasks_spent(trial_step_outputs)
            working_memory.record(
                stage="step6_repair_trial",
                key=requirement_text[:80],
                failure_signature=getattr(diagnosis, "failure_signature", ""),
                scenario_fingerprint=self._scenario_fingerprint_key_from_step_outputs(step_outputs),
                summary=(
                    f"Repair trial {trial_result['candidate_id']} score={short_run_score:.4f} "
                    f"passed={passed}."
                ),
                evidence_urls=list(getattr(candidate, "evidence_refs", []) or []),
                proposed_fixes=[item.get("path", "") for item in trial_result["patch_actions"] if item.get("path")],
                metadata={
                    "candidate_id": trial_result["candidate_id"],
                    "target_step": selected_step,
                    "patch_actions": trial_result["patch_actions"],
                    "short_run_score": short_run_score,
                    "recovery_outcome": trial_result["outcome"],
                    "counterexample": not passed,
                    "counterexample_reason": trial_result["reason"],
                    "analysis_metadata": analysis_metadata,
                    "attribution_status": analysis_metadata.get("attribution_status", ""),
                    "most_important_param": analysis_metadata.get("most_important_param", ""),
                },
            )
            if short_run_score > best_score:
                best_score = short_run_score
                best_candidate = candidate

        self._last_trial_cloud_tasks_spent = trial_cloud_tasks_spent
        return best_candidate, trial_results

    @staticmethod
    def _should_skip_repair_trials(diagnosis: Any) -> bool:
        error_family = str(getattr(diagnosis, "error_family", "") or "").strip().lower()
        error_subtype = str(getattr(diagnosis, "error_subtype", "") or "").strip().lower()
        signature = str(getattr(diagnosis, "failure_signature", "") or "").strip().lower()
        if error_family == "resource_limit":
            return True
        skip_tokens = (
            "insufficient balance",
            "out of credit",
            "quota exceeded",
            "simulation_data_unavailable",
            "failed to download",
            "headobject",
            "ssl",
            "max retries exceeded",
            "httpsconnectionpool",
            "connecterror",
            "readtimeout",
            "timed out",
        )
        haystack = " | ".join([error_subtype, signature])
        return any(token in haystack for token in skip_tokens)

    def _extract_trial_checkpoint_status(self, run_result: Dict[str, Any]) -> str:
        if not isinstance(run_result, dict):
            return ""
        reports = run_result.get("checkpoint_reports", [])
        if not isinstance(reports, list) or not reports:
            return "not_emitted"
        latest = reports[-1]
        if isinstance(latest, dict):
            return str(latest.get("status", ""))
        return ""

    def _score_short_run_candidate(
        self,
        *,
        baseline_run_result: Dict[str, Any],
        trial_run_result: Dict[str, Any],
    ) -> float:
        if not isinstance(trial_run_result, dict):
            return -10.0

        trial_status = str(trial_run_result.get("status", "")).strip().lower()
        baseline_value = self._best_objective_value(baseline_run_result)
        trial_value = self._best_objective_value(trial_run_result)
        goal = str(
            trial_run_result.get("objective_goal")
            or baseline_run_result.get("objective_goal")
            or "maximize"
        ).strip().lower()

        delta = 0.0
        if trial_value is not None and baseline_value is not None:
            delta = trial_value - baseline_value if goal != "minimize" else baseline_value - trial_value

        checkpoint_bonus = 0.0
        checkpoint_trigger = trial_run_result.get("constraint_summary", {}).get("checkpoint_trigger", {})
        if isinstance(checkpoint_trigger, dict) and checkpoint_trigger.get("status") == "fail":
            checkpoint_bonus = -0.5
        elif trial_run_result.get("checkpoint_reports"):
            checkpoint_bonus = 0.2

        status_bonus = 0.6 if trial_status == "completed" and trial_run_result.get("ok") else -0.6
        return round(delta + checkpoint_bonus + status_bonus, 6)

    def _best_objective_value(self, run_result: Dict[str, Any]) -> float | None:
        if not isinstance(run_result, dict):
            return None
        for key in ("best_objective_value", "best_score"):
            value = run_result.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        return None

    def _build_recovery_context(
        self,
        *,
        diagnosis: Any,
        selected_step: str,
        replan_actions: List[Any],
        selected_repair_candidate_id: str,
        trial_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        suggested_queries = list(getattr(diagnosis, "suggested_queries", []) or [])
        for action in replan_actions:
            action_dict = action.model_dump() if hasattr(action, "model_dump") else dict(action)
            if str(action_dict.get("path", "")).strip() == "queries" and action_dict.get("value"):
                suggested_queries.append(str(action_dict["value"]))
        return {
            "selected_rollback_step": selected_step,
            "selected_repair_candidate_id": selected_repair_candidate_id,
            "patch_actions": replan_actions,
            "suggested_queries": suggested_queries,
            "checkpoint_report": getattr(diagnosis, "checkpoint_report", None),
            "scenario_fingerprint": getattr(diagnosis, "scenario_fingerprint", None),
            "trial_results": trial_results,
        }

    def _execute_inverse_tool(
        self,
        *,
        chain_memory: Memory,
        tool_name: str,
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        tool_aliases = {
            "search_flexcompute_docs": "inverse_mcp_search_flexcompute_docs",
            "fetch_flexcompute_doc": "inverse_mcp_fetch_flexcompute_doc",
            "detect_python_environment": "inverse_mcp_detect_python_environment",
            "validate_simulation": "inverse_mcp_validate_simulation",
            "rotate_viewer": "inverse_mcp_rotate_viewer",
            "capture": "inverse_mcp_capture",
            "show_structures": "inverse_mcp_show_structures",
        }
        resolved_tool_name = tool_aliases.get(tool_name, tool_name)
        tool = self.inverse_engine.tool_map.get(resolved_tool_name)
        if tool is None:
            return {"ok": False, "data": {}, "error": f"Unknown tool: {tool_name}"}

        try:
            signature = inspect.signature(tool.fn)
            if any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in signature.parameters.values()
            ):
                filtered_kwargs = kwargs
            else:
                allowed = set(signature.parameters.keys())
                filtered_kwargs = {key: value for key, value in kwargs.items() if key in allowed}
        except Exception:
            filtered_kwargs = kwargs

        return self.inverse_engine.execute_tool(chain_memory, resolved_tool_name, **filtered_kwargs)

    def _step5_needs_recovery(
        self,
        step_outputs: Dict[str, Dict[str, Any]],
        *,
        recover_on_target_miss: bool = True,
    ) -> bool:
        step5 = step_outputs.get("step5", {})
        if not step5.get("ok"):
            return True

        run_result = step5.get("data", {}).get("run_result", {})
        if not isinstance(run_result, dict):
            return True
        if not run_result.get("ok", False):
            return True
        if run_result.get("status") != "completed":
            return True

        objective = (
            step_outputs.get("step3", {})
            .get("data", {})
            .get("config_bundle", {})
            .get("optimization_config", {})
            .get("objective", {})
        )
        target = objective.get("target_value")
        goal = str(objective.get("goal", "maximize")).strip().lower()
        if target is None:
            return False

        try:
            target_value = float(target)
        except (TypeError, ValueError):
            return False

        try:
            best_value = float(run_result.get("best_objective_value"))
        except (TypeError, ValueError):
            return True

        if not recover_on_target_miss:
            return False

        if goal == "minimize":
            return best_value > target_value
        return best_value < target_value

    @staticmethod
    def _count_step5_iterations(step_outputs: Dict[str, Dict[str, Any]]) -> int:
        """Count the number of iterations consumed by the most recent Step5 run."""
        step5 = step_outputs.get("step5", {})
        run_result = step5.get("data", {}).get("run_result", {})
        if not isinstance(run_result, dict):
            return 0
        iterations = run_result.get("iterations")
        if isinstance(iterations, list):
            return len(iterations)
        records = run_result.get("records")
        if isinstance(records, list):  # backward compatibility
            return len(records)
        return 0

    @staticmethod
    def _step5_cloud_tasks_per_iteration(step_outputs: Dict[str, Dict[str, Any]]) -> int:
        step5 = step_outputs.get("step5", {})
        run_result = step5.get("data", {}).get("run_result", {})
        if not isinstance(run_result, dict):
            return 2
        summary = run_result.get("constraint_summary", {})
        if not isinstance(summary, dict):
            summary = {}
        raw = summary.get("estimated_cloud_tasks_per_iteration")
        try:
            value = int(raw)
        except (TypeError, ValueError):
            value = 0
        if value > 0:
            return max(value, 1)

        optimization_cfg = (
            step_outputs.get("step3", {})
            .get("data", {})
            .get("config_bundle", {})
            .get("optimization_config", {})
        )
        objective_cases = optimization_cfg.get("objective_cases", [])
        if not isinstance(objective_cases, list) or not objective_cases:
            return 2

        case_signatures: set[Tuple[float, str, int, str]] = set()
        for case in objective_cases:
            if not isinstance(case, dict):
                continue
            try:
                wavelength_nm = round(float(case.get("wavelength_nm", 0.0) or 0.0), 6)
            except (TypeError, ValueError):
                wavelength_nm = 0.0
            source_port = str(case.get("source_port") or "port_o1").strip().lower()
            try:
                source_mode_index = max(int(case.get("source_mode_index", 0) or 0), 0)
            except (TypeError, ValueError):
                source_mode_index = 0
            source_direction = str(case.get("source_direction") or "-").strip() or "-"
            case_signatures.add((wavelength_nm, source_port, source_mode_index, source_direction))

        unique_task_count = max(len(case_signatures), 1)
        adjoint_tasks_per_iter = unique_task_count * 2
        observation_mode = str(os.getenv("INVERSE_MULTI_CASE_OBSERVATION_MODE", "hybrid")).strip().lower()
        if observation_mode not in {"adjoint_internal", "diagnostic_runner", "hybrid"}:
            observation_mode = "hybrid"
        diagnostic_tasks_per_iter = 0 if observation_mode == "adjoint_internal" else max(len(objective_cases), 1)
        return max(adjoint_tasks_per_iter + diagnostic_tasks_per_iter, 1)

    def _estimate_step5_cloud_tasks_spent(self, step_outputs: Dict[str, Dict[str, Any]]) -> int:
        iterations = self._count_step5_iterations(step_outputs)
        per_iter_tasks = self._step5_cloud_tasks_per_iteration(step_outputs)
        if iterations > 0:
            return iterations * per_iter_tasks

        step5 = step_outputs.get("step5", {})
        if not isinstance(step5, dict) or not step5:
            return 0
        error_text = str(step5.get("error") or "").strip().lower()
        run_result = step5.get("data", {}).get("run_result", {})
        if isinstance(run_result, dict):
            termination_reason = str(run_result.get("termination_reason") or "").strip().lower()
            status = str(run_result.get("status") or "").strip().lower()
        else:
            termination_reason = ""
            status = ""
        combined_error = " ".join(part for part in [error_text, termination_reason] if part)
        if "insufficient balance preflight" in combined_error:
            return 0
        if "step5 not executed" in combined_error:
            return 0
        if status in {"simulation_failed", "requires_self_recovery"} or bool(step5.get("ok")) is False:
            return per_iter_tasks
        return 0

    def _estimate_cloud_tasks_budget(
        self,
        *,
        step_outputs: Dict[str, Dict[str, Any]],
        max_iterations: int,
    ) -> int:
        per_iter = self._step5_cloud_tasks_per_iteration(step_outputs)
        try:
            multiplier = float(os.getenv("INVERSE_RECOVERY_CLOUD_BUDGET_MULTIPLIER", "1.5"))
        except Exception:
            multiplier = 1.0
        if multiplier <= 0:
            multiplier = 1.0
        return max(int(max_iterations * per_iter * multiplier), max_iterations)

    def _derive_failure_message(self, step_outputs: Dict[str, Dict[str, Any]]) -> str:
        step5 = step_outputs.get("step5", {})
        error = str(step5.get("error") or "").strip()
        if error:
            return error

        run_result = step5.get("data", {}).get("run_result", {})
        if isinstance(run_result, dict):
            termination_reason = str(run_result.get("termination_reason") or "").strip()
            if termination_reason:
                return termination_reason
            status = str(run_result.get("status") or "").strip()
            if status and status != "completed":
                return f"Step5 ended with status `{status}`."
            if status == "completed":
                return "Step5 completed but objective target was not reached."

        return "Step5 failed with an unknown recovery-triggering condition."

    @staticmethod
    def _query_flexcredit_balance() -> float | None:
        try:
            from pathlib import Path
            from dotenv import load_dotenv
            from tidy3d import web

            dotenv_path = Path(__file__).resolve().parents[2] / ".env"
            load_dotenv(dotenv_path=dotenv_path, override=False)
            api_key = (
                os.getenv("TIDY3D_API_KEY", "").strip()
                or os.getenv("SIMCLOUD_APIKEY", "").strip()
                or os.getenv("FLEXCOMPUTE_API_KEY", "").strip()
            )
            if not api_key:
                return None
            web.configure(apikey=api_key)
            account = web.account()
            credit = getattr(account, "credit", None)
            if credit is None:
                return None
            return float(credit)
        except Exception:
            return None

    def _low_credit_recovery_abort_reason(self) -> str:
        if os.getenv("INVERSE_ABORT_RECOVERY_ON_LOW_CREDIT", "1").strip().lower() in {
            "0",
            "false",
            "no",
            "off",
        }:
            return ""
        try:
            min_credit = float(os.getenv("INVERSE_RECOVERY_MIN_FLEXCREDIT", "0.6"))
        except Exception:
            min_credit = 0.6
        if min_credit <= 0:
            return ""
        balance = self._query_flexcredit_balance()
        if balance is None or balance + 1e-9 >= min_credit:
            return ""
        return (
            "Recovery aborted by low-credit guard: "
            f"FlexCredit balance {balance:.3f} is below configured recovery minimum {min_credit:.3f}. "
            "Returning first Step5 result to avoid extra cloud-credit burn."
        )

    def _select_rollback_step(
        self,
        *,
        diagnosis: FailureDiagnosis,
        confidence_threshold: float,
        attempts_by_step: Dict[str, int],
    ) -> Tuple[str, str]:
        error_family = str(diagnosis.error_family or "").strip().lower()
        signature = str(diagnosis.failure_signature or "").strip().lower()
        summary = str(diagnosis.summary or "").strip().lower()
        if (
            error_family == "resource_limit"
            or "insufficient balance" in signature
            or "insufficient balance" in summary
            or "credit" in signature
        ):
            return "", "resource_limit_no_retry"

        candidates = list(diagnosis.rollback_candidates)
        if not candidates:
            return "", "no_rollback_candidates"

        mapped_step = self._mapped_rollback_priority(diagnosis)
        if mapped_step:
            mapped_candidate = next(
                (item for item in candidates if item.step == mapped_step),
                None,
            )
            if attempts_by_step.get(mapped_step, 0) < 2:
                if mapped_candidate is not None and mapped_candidate.confidence >= confidence_threshold:
                    return mapped_step, "mapped_family_above_threshold"
                return mapped_step, "mapped_family_override"

        best = candidates[0]
        if best.confidence >= confidence_threshold and attempts_by_step.get(best.step, 0) < 2:
            return best.step, "top_candidate_above_threshold"

        # Nearest-upstream conservative policy for low-confidence decisions.
        nearest_order = ["step4", "step3", "step2", "step5"]
        for step in nearest_order:
            candidate = next((item for item in candidates if item.step == step), None)
            if candidate is None:
                continue
            if attempts_by_step.get(step, 0) >= 2:
                continue
            return step, "nearest_upstream_low_confidence"

        return "", "recovery_budget_exhausted_for_candidates"

    def _mapped_rollback_priority(self, diagnosis: FailureDiagnosis) -> str:
        """Apply hardened rollback routing for known error families."""

        error_family = str(diagnosis.error_family or "").strip().lower()
        error_subtype = str(diagnosis.error_subtype or "").strip().lower()
        category = str(diagnosis.category or "").strip().lower()
        root_cause_stage = str(diagnosis.root_cause_stage or "").strip().lower()

        if error_family == "simulation_scene" or category in {"geometry_conflict", "monitor_invalid"}:
            return "step4"
        if error_family == "optimization_setup":
            return "step3"
        if error_family == "documentation_gap" or "doc" in error_subtype:
            return "step2"

        if root_cause_stage in {"step2", "step3", "step4"}:
            return root_cause_stage
        return ""

    def _collect_recent_step4_issues(self, step_outputs: Dict[str, Dict[str, Any]]) -> List[str]:
        issues = (
            step_outputs.get("step4", {})
            .get("data", {})
            .get("validation", {})
            .get("issues", [])
        )
        collected: List[str] = []
        for issue in issues:
            if isinstance(issue, dict):
                code = str(issue.get("code", "")).strip()
                if code:
                    collected.append(code)
            elif isinstance(issue, str) and issue.strip():
                collected.append(issue.strip())
        return collected

    def _apply_replan_actions(
        self,
        step_outputs: Dict[str, Dict[str, Any]],
        actions: List[Any],
    ) -> List[str]:
        applied_paths: List[str] = []
        for action in actions:
            action_dict = action.model_dump() if hasattr(action, "model_dump") else dict(action)
            path = str(action_dict.get("path", "")).strip()
            if not path or path == "root":
                continue
            if self._apply_single_replan_action(step_outputs, action_dict):
                applied_paths.append(path)
        return applied_paths

    def _apply_single_replan_action(
        self,
        step_outputs: Dict[str, Dict[str, Any]],
        action: Dict[str, Any],
    ) -> bool:
        path = str(action.get("path", "")).strip()
        operation = str(action.get("action", "set_value")).strip().lower()
        value = action.get("value")

        config_bundle = (
            step_outputs.get("step3", {})
            .get("data", {})
            .get("config_bundle")
        )
        doc_context = (
            step_outputs.get("step2", {})
            .get("data", {})
            .get("doc_context")
        )

        target, relative_path = None, path
        if path.startswith("simulation_config") or path.startswith("optimization_config"):
            target = config_bundle
        elif path.startswith("doc_context."):
            target = doc_context
            relative_path = path.split(".", 1)[1]
        elif path.split(".", 1)[0] in {"queries", "references", "guidance", "requirement"}:
            target = doc_context
        else:
            target = config_bundle or doc_context

        if not isinstance(target, (dict, list)):
            return False

        if operation == "set_value":
            return self._set_path_value(target, relative_path, value)
        if operation == "add_item":
            return self._add_path_value(target, relative_path, value)
        if operation == "remove_item":
            return self._remove_path_value(target, relative_path)
        return False

    def _set_path_value(self, root: Any, path: str, value: Any) -> bool:
        parent, key = self._resolve_parent(root, path, create=True)
        if parent is None:
            return False
        if isinstance(parent, list) and isinstance(key, int):
            while len(parent) <= key:
                parent.append(None)
            parent[key] = value
            return True
        if isinstance(parent, dict) and isinstance(key, str):
            parent[key] = value
            return True
        return False

    def _add_path_value(self, root: Any, path: str, value: Any) -> bool:
        container = self._get_path_value(root, path)
        if isinstance(container, list):
            container.append(value)
            return True

        parent, key = self._resolve_parent(root, path, create=True)
        if parent is None:
            return False
        if isinstance(parent, dict) and isinstance(key, str):
            existing = parent.get(key)
            if isinstance(existing, list):
                existing.append(value)
            elif existing is None:
                parent[key] = [value]
            else:
                parent[key] = [existing, value]
            return True
        if isinstance(parent, list) and isinstance(key, int):
            while len(parent) <= key:
                parent.append([])
            existing = parent[key]
            if isinstance(existing, list):
                existing.append(value)
            elif existing is None:
                parent[key] = [value]
            else:
                parent[key] = [existing, value]
            return True
        return False

    def _remove_path_value(self, root: Any, path: str) -> bool:
        parent, key = self._resolve_parent(root, path, create=False)
        if parent is None:
            return False
        if isinstance(parent, dict) and isinstance(key, str):
            if key in parent:
                parent.pop(key, None)
                return True
            return False
        if isinstance(parent, list) and isinstance(key, int):
            if 0 <= key < len(parent):
                parent.pop(key)
                return True
            return False
        return False

    def _get_path_value(self, root: Any, path: str) -> Any:
        current = root
        for part in self._parse_path(path):
            if isinstance(part, int):
                if not isinstance(current, list) or part >= len(current):
                    return None
                current = current[part]
            else:
                if not isinstance(current, dict) or part not in current:
                    return None
                current = current[part]
        return current

    def _resolve_parent(self, root: Any, path: str, *, create: bool) -> Tuple[Any, str | int] | Tuple[None, None]:
        parts = self._parse_path(path)
        if not parts:
            return None, None
        current = root
        for idx, part in enumerate(parts[:-1]):
            next_part = parts[idx + 1]
            if isinstance(part, int):
                if not isinstance(current, list):
                    return None, None
                while create and len(current) <= part:
                    current.append({} if isinstance(next_part, str) else [])
                if part >= len(current):
                    return None, None
                if current[part] is None and create:
                    current[part] = {} if isinstance(next_part, str) else []
                current = current[part]
                continue

            if not isinstance(current, dict):
                return None, None
            if part not in current or current[part] is None:
                if not create:
                    return None, None
                current[part] = {} if isinstance(next_part, str) else []
            current = current[part]
        return current, parts[-1]

    def _parse_path(self, path: str) -> List[str | int]:
        cleaned = str(path).strip().strip(".")
        if not cleaned:
            return []
        parts: List[str | int] = []
        for token in cleaned.split("."):
            token = token.strip()
            if not token:
                continue
            if token.isdigit():
                parts.append(int(token))
            else:
                parts.append(token)
        return parts

    def _build_recovery_record(
        self,
        *,
        diagnosis: FailureDiagnosis,
        selected_rollback_step: str,
        replan_actions: List[Any],
        resume_from_step: str,
        recovery_attempt: int,
        recovery_outcome: str,
        selection_reason: str,
        selected_repair_candidate_id: str = "",
        trial_results: List[Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        patch_actions = [
            action.model_dump() if hasattr(action, "model_dump") else dict(action)
            for action in replan_actions
        ]
        return {
            "failure_signature": diagnosis.failure_signature,
            "failed_step": diagnosis.failed_step,
            "rollback_candidates": [item.model_dump() for item in diagnosis.rollback_candidates],
            "selected_rollback_step": selected_rollback_step,
            "selected_repair_candidate_id": selected_repair_candidate_id,
            "candidate_id": selected_repair_candidate_id,
            "replan_actions": patch_actions,
            "patch_actions": patch_actions,
            "resume_from_step": resume_from_step,
            "recovery_attempt": recovery_attempt,
            "recovery_outcome": recovery_outcome,
            "selection_reason": selection_reason,
            "counterexample": recovery_outcome not in {"success"},
            "counterexample_reason": selection_reason if recovery_outcome != "success" else "",
            "requires_doc_refresh": diagnosis.requires_doc_refresh,
            "confidence": diagnosis.confidence,
            "evidence_refs": list(diagnosis.evidence_urls),
            "scenario_fingerprint": (
                diagnosis.scenario_fingerprint.model_dump()
                if diagnosis.scenario_fingerprint is not None
                else None
            ),
            "trial_results": list(trial_results or []),
        }


def create_pdk_agent() -> PDKAgent:
    return PDKAgent()
