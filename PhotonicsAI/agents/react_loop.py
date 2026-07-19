"""Lightweight ReAct primitives for tool-driven agent workflows."""

from __future__ import annotations

import concurrent.futures
import importlib
import json
import logging
import multiprocessing as mp
import os
import queue
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional

from PhotonicsAI.core.tooling import Envelope, Tool

logger = logging.getLogger(__name__)

# Regex to extract the step number from a tool name like
# "inverse_step2_retrieve_doc_context" -> "step2"
import re as _re

_STEP_KEY_RE = _re.compile(r"inverse_(step\d+)")


def _extract_step_key(tool_name: str) -> str | None:
    """Return e.g. ``'step2'`` from ``'inverse_step2_retrieve_doc_context'``."""
    m = _STEP_KEY_RE.search(tool_name)
    return m.group(1) if m else None


RECOVERY_EVENT_TYPES = {
    "recovery_diagnosis",
    "rollback_selection",
    "replan",
    "resume",
}


def build_recovery_event(event: str, **payload: Any) -> Dict[str, Any]:
    """Build a standardized recovery event payload for UI streaming."""

    if event not in RECOVERY_EVENT_TYPES:
        raise ValueError(f"Unsupported recovery event type: {event}")
    return {"event": event, **payload}


def _step5_watchdog_enabled() -> bool:
    raw = str(os.getenv("INVERSE_STEP5_SUBPROCESS_WATCHDOG", "1") or "").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _step5_watchdog_poll_s() -> float:
    raw = str(os.getenv("INVERSE_STEP5_WATCHDOG_POLL_S", "2.0") or "").strip()
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = 2.0
    return max(0.2, value)


def _step5_watchdog_track_progress_files() -> bool:
    raw = str(os.getenv("INVERSE_STEP5_WATCHDOG_TRACK_PROGRESS_FILES", "1") or "").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _latest_step5_progress_mtime(start_time_s: float) -> float:
    """Return latest mtime for Step5 progress artifacts created after run start."""

    build_dir = Path("build")
    if not build_dir.exists():
        return 0.0

    latest = 0.0
    for pattern in ("invdes_step5_heartbeat_*.json", "invdes_adjoint_iteration_trace_*.jsonl"):
        for path in build_dir.glob(pattern):
            try:
                mtime = float(path.stat().st_mtime)
            except OSError:
                continue
            if mtime >= (start_time_s - 1.0) and mtime > latest:
                latest = mtime
    return latest


def _tool_subprocess_worker(
    module_name: str,
    function_name: str,
    kwargs: Dict[str, Any],
    out_queue: Any,
) -> None:
    try:
        module = importlib.import_module(module_name)
        fn = getattr(module, function_name)
        result = fn(**kwargs)
        out_queue.put({"ok": True, "result": result})
    except Exception as exc:
        out_queue.put(
            {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )


def _execute_tool_in_subprocess(
    *,
    tool_name: str,
    module_name: str,
    function_name: str,
    kwargs: Dict[str, Any],
    timeout_s: float,
) -> Envelope:
    ctx = mp.get_context("spawn")
    out_queue = ctx.Queue(maxsize=1)
    process = ctx.Process(
        target=_tool_subprocess_worker,
        args=(module_name, function_name, kwargs, out_queue),
        daemon=True,
    )
    process.start()
    poll_s = _step5_watchdog_poll_s()
    start_time_s = time.time()
    deadline = start_time_s + max(timeout_s, 1.0)
    watch_progress_files = (
        tool_name == "inverse_step5_execute" and _step5_watchdog_track_progress_files()
    )
    last_progress_time_s = start_time_s
    last_progress_mtime = (
        _latest_step5_progress_mtime(start_time_s)
        if watch_progress_files
        else 0.0
    )
    payload: Dict[str, Any] | None = None
    timed_out = False
    try:
        while True:
            now = time.time()
            if watch_progress_files:
                remaining = (last_progress_time_s + max(timeout_s, 1.0)) - now
            else:
                remaining = deadline - now
            if remaining <= 0:
                timed_out = True
                break
            try:
                payload = out_queue.get(timeout=min(poll_s, max(remaining, 0.1)))
                break
            except queue.Empty:
                if not process.is_alive():
                    break
                if watch_progress_files:
                    latest_progress_mtime = _latest_step5_progress_mtime(start_time_s)
                    if latest_progress_mtime > (last_progress_mtime + 1e-6):
                        last_progress_mtime = latest_progress_mtime
                        last_progress_time_s = time.time()
                continue
    finally:
        if process.is_alive():
            if payload is None and timed_out:
                process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=2)

    if payload is None:
        if timed_out:
            return {
                "ok": False,
                "data": {},
                "error": f"Tool {tool_name} no-event watchdog timeout after {timeout_s}s",
            }
        return {
            "ok": False,
            "data": {},
            "error": f"Tool {tool_name} subprocess exited without result.",
        }

    if payload.get("ok"):
        raw_result = payload.get("result")
        if isinstance(raw_result, dict):
            return raw_result
        return {"ok": False, "data": {}, "error": f"Tool {tool_name} returned non-dict envelope."}

    error_text = str(payload.get("error") or "subprocess execution failed")
    tb = str(payload.get("traceback") or "").strip()
    if tb:
        error_text = f"{error_text} | traceback: {tb}"
    return {"ok": False, "data": {}, "error": f"Tool {tool_name} failed: {error_text}"}


def _can_run_tool_in_subprocess(tool_fn: Callable[..., Any]) -> bool:
    fn_name = str(getattr(tool_fn, "__name__", "") or "")
    if not fn_name.isidentifier():
        return False
    module_name = str(getattr(tool_fn, "__module__", "") or "")
    if not module_name:
        return False
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return False
    return getattr(module, fn_name, None) is tool_fn


@dataclass
class Memory:
    """Conversation and iteration memory with bounded history."""

    messages: List[Dict[str, Any]] = field(default_factory=list)
    iterations: List[Dict[str, Any]] = field(default_factory=list)
    max_full_iterations: int = 5

    def add_message(self, role: str, content: str, **extra: Any) -> None:
        entry: Dict[str, Any] = {
            "role": role,
            "content": content,
            "ts": datetime.now().isoformat(timespec="seconds"),
        }
        if extra:
            entry.update(extra)
        self.messages.append(entry)

    def add_iteration(self, payload: Dict[str, Any]) -> None:
        self.iterations.append(payload)
        self._compress_iterations()

    def _compress_iterations(self) -> None:
        overflow = len(self.iterations) - self.max_full_iterations
        if overflow <= 0:
            return

        # Keep a compact summary for older iterations to avoid context bloat.
        for idx in range(overflow):
            item = self.iterations[idx]
            if item.get("compressed"):
                continue
            summary = {
                "compressed": True,
                "iteration": item.get("iteration"),
                "tool": item.get("tool"),
                "ok": item.get("ok"),
                "note": item.get("note") or item.get("error") or "completed",
            }
            self.iterations[idx] = summary


class ReActEngine:
    """Reusable helper that standardizes event stream and tool execution."""

    def __init__(
        self,
        tools: Iterable[Tool],
        system_prompt: str,
        llm_client: Any | None = None,
        max_steps: int = 30,
        max_retries: int = 2,
        reflection_trigger: Optional[Callable[[str, Envelope], bool]] = None,
        tool_timeout: float | None = 300.0,
    ) -> None:
        self.tools: List[Tool] = list(tools)
        self.tool_map: Dict[str, Tool] = {tool.name: tool for tool in tools}
        self.system_prompt = system_prompt
        self.llm_client = llm_client
        self.max_steps = max_steps
        self.max_retries = max_retries
        self.reflection_trigger = reflection_trigger
        self.tool_timeout = tool_timeout  # KL-5: per-tool timeout in seconds

    @property
    def tool_descriptions(self) -> str:
        lines = []
        for tool in self.tools:
            lines.append(f"- {tool.name}: {tool.description}")
        return "\n".join(lines)

    def execute_tool(self, memory: Memory, tool_name: str, **kwargs: Any) -> Envelope:
        tool = self.tool_map.get(tool_name)
        if tool is None:
            return {
                "ok": False,
                "data": {},
                "error": f"Unknown tool: {tool_name}",
            }

        attempt = 0
        last_error = None
        result: Envelope = {"ok": False, "data": {}, "error": None}
        while attempt <= self.max_retries:
            try:
                # KL-5: Per-tool timeout to prevent a single hung tool from
                # blocking the entire ReAct pipeline.
                # Use tool-specific timeout if set, else fall back to default.
                effective_timeout = (
                    tool.timeout if tool.timeout is not None else self.tool_timeout
                )
                use_step5_watchdog = (
                    tool_name == "inverse_step5_execute"
                    and effective_timeout is not None
                    and effective_timeout > 0
                    and _step5_watchdog_enabled()
                    and _can_run_tool_in_subprocess(tool.fn)
                )
                if use_step5_watchdog:
                    try:
                        result = _execute_tool_in_subprocess(
                            tool_name=tool_name,
                            module_name=tool.fn.__module__,
                            function_name=tool.fn.__name__,
                            kwargs=kwargs,
                            timeout_s=float(effective_timeout),
                        )
                    except Exception as subproc_exc:
                        logger.warning(
                            "Step5 subprocess watchdog unavailable, falling back to in-process timeout: %s",
                            subproc_exc,
                        )
                        use_step5_watchdog = False
                        if tool_name == "inverse_step5_execute":
                            raw_timeout = str(os.getenv("INVERSE_STEP5_TOOL_TIMEOUT_S", "") or "").strip()
                            try:
                                fallback_timeout = float(raw_timeout)
                            except (TypeError, ValueError):
                                fallback_timeout = 0.0
                            if fallback_timeout > 0 and (
                                effective_timeout is None or fallback_timeout > float(effective_timeout)
                            ):
                                effective_timeout = float(fallback_timeout)
                if use_step5_watchdog:
                    pass
                elif effective_timeout is not None and effective_timeout > 0:
                    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                    timed_out = False
                    try:
                        future = executor.submit(tool.call, **kwargs)
                        result = future.result(timeout=effective_timeout)
                    except concurrent.futures.TimeoutError:
                        timed_out = True
                        future.cancel()
                        executor.shutdown(wait=False, cancel_futures=True)
                        raise
                    finally:
                        if not timed_out:
                            executor.shutdown(wait=True, cancel_futures=True)
                else:
                    result = tool.call(**kwargs)
                break
            except concurrent.futures.TimeoutError:
                last_error = TimeoutError(
                    f"Tool {tool_name} timed out after {effective_timeout}s"
                )
                result = {
                    "ok": False,
                    "data": {},
                    "error": str(last_error),
                }
                logger.warning("KL-5: %s", last_error)
                break  # Do not retry timeouts — the tool is likely stuck.
            except Exception as exc:  # Explicit envelope conversion for caller simplicity.
                last_error = exc
                result = {
                    "ok": False,
                    "data": {},
                    "error": f"Tool {tool_name} failed: {exc}",
                }
            attempt += 1

        memory.add_iteration(
            {
                "iteration": len(memory.iterations) + 1,
                "tool": tool_name,
                "ok": bool(result.get("ok")),
                "error": result.get("error"),
                "attempts": attempt + 1,
                "note": None if last_error is None else str(last_error),
            }
        )
        return result

    def should_reflect(self, tool_name: str, result: Envelope) -> bool:
        if self.reflection_trigger is None:
            return False
        return bool(self.reflection_trigger(tool_name, result))

    def has_converged(self, score_history: List[float], threshold: float = 0.05) -> bool:
        if len(score_history) < 4:
            return False

        last_scores = score_history[-4:]
        improvements = []
        for previous, current in zip(last_scores, last_scores[1:]):
            if previous == 0:
                improvements.append(1.0 if current > 0 else 0.0)
            else:
                improvements.append(abs(current - previous) / abs(previous))
        return all(improvement < threshold for improvement in improvements)

    # ------------------------------------------------------------------
    # OpenAI function-calling helpers
    # ------------------------------------------------------------------

    def _build_openai_tools(self) -> List[Dict[str, Any]]:
        """Convert registered Tool list to OpenAI ``tools`` format."""
        openai_tools: List[Dict[str, Any]] = []
        for tool in self.tools:
            openai_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
            )
        return openai_tools

    def _build_llm_messages(self, memory: Memory) -> List[Dict[str, Any]]:
        """Build a clean message list suitable for ``chat.completions.create``.

        Strips internal fields (``ts``) and keeps only OpenAI-compatible
        keys: ``role``, ``content``, ``tool_call_id``, ``name``.
        """
        msgs: List[Dict[str, Any]] = []
        for m in memory.messages:
            entry: Dict[str, Any] = {"role": m["role"], "content": m.get("content") or ""}
            if m.get("tool_call_id"):
                entry["tool_call_id"] = m["tool_call_id"]
            if m.get("name"):
                entry["name"] = m["name"]
            if m.get("tool_calls"):
                entry["tool_calls"] = m["tool_calls"]
                # assistant messages with tool_calls may have null content
                if entry["content"] == "":
                    entry["content"] = None
            msgs.append(entry)
        return msgs

    @staticmethod
    def _serialize_tool_result(result: Any) -> str:
        """Serialize a tool result envelope to a JSON string for the LLM."""
        if isinstance(result, str):
            return result
        try:
            return json.dumps(result, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return str(result)

    # ------------------------------------------------------------------
    # Real LLM function-calling loop
    # ------------------------------------------------------------------

    def run(
        self,
        memory: Memory,
        user_input: str,
        *,
        model: str = "gpt-5.4",
        initial_workflow_state: Dict[str, Any] | None = None,
    ) -> Generator[Dict[str, Any], None, Envelope]:
        """Run a full ReAct loop with real LLM function calling.

        Requires ``self.llm_client`` to be a configured ``openai.OpenAI``
        instance.  The loop continues until the LLM stops issuing tool
        calls or ``max_steps`` is reached.

        ``initial_workflow_state`` seeds the accumulated state dict so that
        context values (e.g. ``requirement_text``) are available to the
        first step even if the LLM omits them in the tool-call arguments.
        """
        if self.llm_client is None:
            raise RuntimeError("llm_client is required for run(). Pass an OpenAI client.")

        if not memory.messages:
            memory.add_message("system", self.system_prompt)
        memory.add_message("user", user_input)

        openai_tools = self._build_openai_tools()

        # Accumulated step outputs used to auto-inject workflow_state into
        # subsequent inverse-step tool calls so the LLM doesn't need to
        # manually thread step outputs between calls.
        _workflow_state: Dict[str, Any] = dict(initial_workflow_state or {})

        for step in range(1, self.max_steps + 1):
            messages = self._build_llm_messages(memory)

            try:
                response = self.llm_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=openai_tools,
                    tool_choice="auto",
                )
            except Exception as exc:
                logger.error("LLM call failed at step %d: %s", step, exc)
                error_answer: Envelope = {
                    "ok": False,
                    "data": {},
                    "error": f"LLM call failed: {exc}",
                }
                yield {"event": "answer", "result": error_answer, "step": step}
                return error_answer

            message = response.choices[0].message

            # Emit any textual thought the LLM produced alongside tool calls.
            assistant_text = getattr(message, "content", None) or ""
            tool_calls = getattr(message, "tool_calls", None) or []

            if assistant_text:
                yield {"event": "thought", "content": assistant_text, "step": step}

            # Record the full assistant message in memory (including tool_calls).
            assistant_entry: Dict[str, Any] = {
                "role": "assistant",
                "content": assistant_text or None,
            }
            if tool_calls:
                assistant_entry["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ]
            memory.messages.append(
                {**assistant_entry, "ts": datetime.now().isoformat(timespec="seconds")}
            )

            # No tool calls → LLM is done; emit final answer.
            if not tool_calls:
                answer: Envelope = {
                    "ok": True,
                    "data": {"response": assistant_text},
                    "error": None,
                }
                yield {"event": "answer", "result": answer, "step": step}
                return answer

            # Execute each tool call and feed results back.
            for tc in tool_calls:
                tool_name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}

                # Auto-inject accumulated workflow_state for step tools so
                # each step can find previous steps' outputs without relying
                # on the LLM to manually thread the data.  Only inject keys
                # that the tool's schema actually declares to avoid passing
                # unexpected keyword arguments.
                if tool_name.startswith("inverse_step"):
                    tool_obj = self.tool_map.get(tool_name)
                    if tool_obj:
                        declared = set(
                            tool_obj.parameters.get("properties", {}).keys()
                        )
                        for key, value in _workflow_state.items():
                            if key in declared and key not in args:
                                args[key] = value
                        # Pass the full accumulated state as workflow_state
                        # so step functions can extract outputs from prior
                        # steps (e.g. step3 reads step2.data.doc_context).
                        if "workflow_state" in declared:
                            args.setdefault("workflow_state", dict(_workflow_state))

                yield {"event": "action", "tool": tool_name, "args": args, "step": step}

                result = self.execute_tool(memory, tool_name, **args)
                yield {"event": "observation", "tool": tool_name, "result": result, "step": step}

                # Track step outputs for workflow_state accumulation.
                _step_key = _extract_step_key(tool_name)
                if _step_key:
                    _workflow_state[_step_key] = result

                # Write the tool result as a tool message for the next LLM turn.
                memory.add_message(
                    "tool",
                    self._serialize_tool_result(result),
                    tool_call_id=tc.id,
                    name=tool_name,
                )

                # Reflection trigger (optional).
                if self.should_reflect(tool_name, result):
                    reflection_text = self.generate_reflection(memory, tool_name, result)
                    yield {"event": "reflection", "content": reflection_text, "step": step}

        # Exhausted max_steps without a final answer.
        timeout_answer: Envelope = {
            "ok": False,
            "data": {},
            "error": f"Reached max_steps ({self.max_steps}) without final answer.",
        }
        yield {"event": "answer", "result": timeout_answer, "step": self.max_steps}
        return timeout_answer

    # ------------------------------------------------------------------
    # LLM-based reflection
    # ------------------------------------------------------------------

    def generate_reflection(
        self,
        memory: Memory,
        tool_name: str,
        result: Envelope,
        *,
        model: str = "gpt-5.4",
    ) -> str:
        """Generate a reflection using the LLM instead of a static string."""
        if self.llm_client is None:
            # Fallback to static when no client is available.
            return (
                "<reflection>Review the latest simulation outcome "
                "before the next action.</reflection>"
            )

        result_summary = self._serialize_tool_result(result)
        # Truncate very large results to avoid token overflow.
        if len(result_summary) > 4000:
            result_summary = result_summary[:4000] + "… (truncated)"

        reflection_prompt = (
            f"Tool `{tool_name}` just ran.\n"
            f"Result:\n{result_summary}\n\n"
            "Briefly reflect on the result: what does it imply about "
            "the current design? What should be adjusted next? "
            "Return only one <reflection>...</reflection> block."
        )

        try:
            response = self.llm_client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an engineering reflection assistant for "
                            "photonic device optimization."
                        ),
                    },
                    {"role": "user", "content": reflection_prompt},
                ],
                max_tokens=512,
            )
            text = response.choices[0].message.content or ""
            memory.add_message("assistant", text)
            return text
        except Exception as exc:
            logger.warning("Reflection LLM call failed: %s", exc)
            fallback = (
                "<reflection>Review the latest simulation outcome "
                "before the next action.</reflection>"
            )
            memory.add_message("assistant", fallback)
            return fallback

    def run_plan(
        self,
        memory: Memory,
        plan: List[Dict[str, Any]],
    ) -> Generator[Dict[str, Any], None, Envelope]:
        """Execute a concrete plan as an event stream.

        This is the deterministic fallback path used before full LLM planning is wired.
        """

        if not memory.messages:
            memory.add_message("system", self.system_prompt)

        score_history: List[float] = []
        last_result: Envelope = {"ok": True, "data": {}, "error": None}

        for index, step in enumerate(plan[: self.max_steps], start=1):
            thought = step.get("thought")
            if thought:
                yield {"event": "thought", "content": thought, "step": index}

            tool_name = step["tool"]
            args = step.get("args", {})
            yield {"event": "action", "tool": tool_name, "args": args, "step": index}

            result = self.execute_tool(memory, tool_name, **args)
            last_result = result
            yield {"event": "observation", "tool": tool_name, "result": result, "step": index}

            score = result.get("data", {}).get("score")
            if isinstance(score, (int, float)):
                score_history.append(float(score))

            if self.should_reflect(tool_name, result):
                reflection = step.get(
                    "reflection",
                    "<reflection>Review the latest simulation outcome before the next action.</reflection>",
                )
                memory.add_message("assistant", reflection)
                yield {"event": "reflection", "content": reflection, "step": index}

            if step.get("stop_on_error", True) and not result.get("ok"):
                answer = {
                    "ok": False,
                    "data": result.get("data", {}),
                    "error": result.get("error") or f"Step failed: {tool_name}",
                }
                yield {"event": "answer", "result": answer, "step": index}
                return answer

        # All steps completed — emit final answer.
        answer = {
            "ok": bool(last_result.get("ok", True)),
            "data": last_result.get("data", {}),
            "error": last_result.get("error"),
        }
        yield {"event": "answer", "result": answer, "step": len(plan[: self.max_steps])}
        return answer

    def run_optimization_loop(
        self,
        memory: Memory,
        initial_params: Dict[str, Any],
        build_iteration_plan: Callable[[Dict[str, Any], int], List[Dict[str, Any]]],
        perturb_params: Callable[..., Dict[str, Any]],
        max_iterations: int = 8,
        convergence_threshold: float = 0.05,
        target_score: Optional[float] = None,
    ) -> Generator[Dict[str, Any], None, Envelope]:
        """Multi-round optimisation loop with convergence detection.

        Each iteration: build plan → run plan → extract score →
        check convergence → perturb params → next round.
        """
        if not memory.messages:
            memory.add_message("system", self.system_prompt)

        params = dict(initial_params)
        score_history: List[float] = []
        best_score: float = float("-inf")
        best_params: Dict[str, Any] = dict(params)
        direction_memory: Dict[str, int] = {}

        for iteration in range(1, max_iterations + 1):
            yield {
                "event": "thought",
                "content": f"Starting optimisation iteration {iteration}/{max_iterations}",
                "iteration": iteration,
            }

            plan = build_iteration_plan(params, iteration)
            sim_result: Optional[Envelope] = None

            for event in self.run_plan(memory, plan):
                event["iteration"] = iteration
                if (
                    event.get("event") == "observation"
                    and event.get("tool") == "run_tidy3d_simulation"
                ):
                    sim_result = event.get("result")
                if event.get("event") == "answer" and not event.get("result", {}).get("ok"):
                    # Iteration plan failed — stop the whole loop.
                    return event.get("result", {"ok": False, "error": "iteration plan failed"})
                # Skip the per-plan answer events; only the loop emits the
                # final answer.
                if event.get("event") == "answer":
                    continue
                yield event

            score: float = 0.0
            if sim_result and sim_result.get("ok"):
                score = float(sim_result.get("data", {}).get("score") or 0)
            score_history.append(score)

            if score > best_score:
                best_score = score
                best_params = dict(params)

            yield {
                "event": "thought",
                "content": (
                    f"Iteration {iteration}: score={score:.4f}, "
                    f"best={best_score:.4f}, history={[round(s, 4) for s in score_history]}"
                ),
                "iteration": iteration,
            }

            # Record high-level iteration summary in memory.
            prev_score = score_history[-2] if len(score_history) >= 2 else None
            improvement = (
                (score - prev_score) / abs(prev_score) if prev_score and prev_score != 0 else None
            )
            memory.add_iteration(
                {
                    "iteration": iteration,
                    "tool": "optimization_round",
                    "ok": True,
                    "score": score,
                    "params": params,
                    "improvement": improvement,
                }
            )

            # --- Termination checks ---
            # 1) Target score reached.
            if target_score is not None and score >= target_score:
                yield {
                    "event": "thought",
                    "content": f"Target score {target_score} reached (score={score:.4f}). Stopping.",
                    "iteration": iteration,
                }
                break

            # 2) Convergence (3 consecutive improvements < threshold).
            if self.has_converged(score_history, convergence_threshold):
                yield {
                    "event": "thought",
                    "content": (
                        f"Converged: last 3 improvements all < {convergence_threshold*100:.0f}%. Stopping."
                    ),
                    "iteration": iteration,
                }
                break

            # 3) Max iterations guard (loop range handles this, but
            #    skip perturbation on the last round).
            if iteration >= max_iterations:
                break

            # Reflection before perturbation.
            if sim_result is not None:
                reflection_content = self.generate_reflection(
                    memory, "run_tidy3d_simulation", sim_result
                )
            else:
                reflection_content = (
                    f"<reflection>Score trend: {[round(s, 4) for s in score_history]}. "
                    "Deciding parameter changes for next iteration.</reflection>"
                )
                memory.add_message("assistant", reflection_content)
            yield {"event": "reflection", "content": reflection_content, "iteration": iteration}

            # Perturb parameters.
            params = perturb_params(
                params,
                score,
                iteration,
                direction_memory=direction_memory,
                prev_score=prev_score,
            )
            yield {
                "event": "thought",
                "content": f"Perturbed params for iteration {iteration + 1}: {params}",
                "iteration": iteration,
            }

        answer: Envelope = {
            "ok": True,
            "data": {
                "best_score": best_score,
                "best_params": best_params,
                "score_history": score_history,
                "iterations_used": len(score_history),
            },
            "error": None,
        }
        yield {"event": "answer", "result": answer}
        return answer
