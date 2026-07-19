"""Reusable factory for ReAct tool-calling LLM clients.

This module resolves a tool-calling client for the ReAct main path while
keeping the orchestration layer provider-agnostic. The accepted main-path
providers are GPT/OpenAI-compatible endpoints and Claude/Anthropic.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import anthropic
from dotenv import find_dotenv, load_dotenv
from openai import OpenAI


@dataclass(frozen=True)
class ToolCallingClientSpec:
    """Resolved client/model/backend tuple for ReAct tool calling."""

    client: Any | None
    model: str
    backend: str
    reason: str = ""


class _AnthropicOpenAICompatClient:
    """Wrap Anthropic's messages API behind an OpenAI-style interface."""

    def __init__(self, api_key: str) -> None:
        self._client = anthropic.Anthropic(api_key=api_key)
        self.chat = SimpleNamespace(completions=_AnthropicChatCompletions(self._client))


class _AnthropicChatCompletions:
    def __init__(self, client: anthropic.Anthropic) -> None:
        self._client = client

    def create(self, *, model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None, tool_choice: Any | None = None, max_tokens: int | None = None, **_: Any) -> Any:
        system_prompt, anthropic_messages = _to_anthropic_messages(messages)
        anthropic_tools = _to_anthropic_tools(tools or [])
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": max_tokens or 2048,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools
            kwargs["tool_choice"] = _to_anthropic_tool_choice(tool_choice)

        response = self._client.messages.create(**kwargs)
        return _to_openai_style_response(response)


def _to_anthropic_messages(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    system_parts: list[str] = []
    converted: list[dict[str, Any]] = []

    for message in messages:
        role = str(message.get("role") or "").strip().lower()
        content = message.get("content")

        if role == "system":
            if content:
                system_parts.append(str(content))
            continue

        if role == "assistant":
            blocks: list[Any] = []
            text_content = "" if content is None else str(content)
            if text_content:
                blocks.append({"type": "text", "text": text_content})
            for tool_call in message.get("tool_calls") or []:
                function = tool_call.get("function") or {}
                raw_arguments = function.get("arguments") or "{}"
                try:
                    parsed_arguments = json.loads(raw_arguments)
                except Exception:
                    parsed_arguments = {}
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": tool_call.get("id"),
                        "name": function.get("name"),
                        "input": parsed_arguments,
                    }
                )
            converted.append({"role": "assistant", "content": blocks or text_content or ""})
            continue

        if role == "tool":
            converted.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": message.get("tool_call_id"),
                            "content": "" if content is None else str(content),
                        }
                    ],
                }
            )
            continue

        converted.append({"role": role or "user", "content": "" if content is None else str(content)})

    return "\n\n".join(system_parts).strip(), converted


def _to_anthropic_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for tool in tools:
        function = tool.get("function") or {}
        converted.append(
            {
                "name": function.get("name"),
                "description": function.get("description") or "",
                "input_schema": function.get("parameters") or {"type": "object", "properties": {}},
            }
        )
    return converted


def _to_anthropic_tool_choice(tool_choice: Any) -> Any:
    if tool_choice in (None, "auto"):
        return {"type": "auto"}
    if tool_choice == "none":
        return {"type": "auto"}
    if isinstance(tool_choice, dict):
        return tool_choice
    return {"type": "auto"}


def _to_openai_style_response(response: Any) -> Any:
    text_parts: list[str] = []
    tool_calls: list[Any] = []

    for block in getattr(response, "content", []) or []:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            text_parts.append(getattr(block, "text", "") or "")
            continue
        if block_type == "tool_use":
            tool_calls.append(
                SimpleNamespace(
                    id=getattr(block, "id", ""),
                    function=SimpleNamespace(
                        name=getattr(block, "name", ""),
                        arguments=json.dumps(getattr(block, "input", {}) or {}, ensure_ascii=False),
                    ),
                )
            )

    message = SimpleNamespace(
        content="\n".join(part for part in text_parts if part).strip() or None,
        tool_calls=tool_calls or None,
    )
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice], raw_response=response)


def build_tool_calling_client(preferred_model: str | None = None) -> ToolCallingClientSpec:
    """Resolve the best available client for ReAct tool calling.

    Priority:
    1. ResourcePack unified gateway
    2. Explicit OpenAI-compatible base URL (`OPENAI_BASE_URL`) + `OPENAI_API_KEY`
    3. Native OpenAI endpoint (GPT)
    4. Native Anthropic endpoint (Claude)
    """

    _ensure_env_loaded()

    preferred = (preferred_model or "").strip()

    resourcepack_api_key = os.getenv("RESOURCEPACK_API_KEY", "").strip()
    resourcepack_base_url = os.getenv("RESOURCEPACK_BASE_URL", "").strip()
    resourcepack_model = (
        preferred
        or os.getenv("RESOURCEPACK_DEFAULT_MODEL", "").strip()
        or os.getenv("DEFAULT_LLM_MODEL", "").strip()
        or "gpt-5.4"
    )
    # Guard: reject models requiring /responses endpoint (codex, *-pro)
    # since this client uses /chat/completions.  Fall back to gpt-5.4.
    from PhotonicsAI.Photon.model_capabilities import should_use_responses_api
    if should_use_responses_api(resourcepack_model):
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "Model %r requires /responses endpoint but ReAct uses /chat/completions; "
            "falling back to gpt-5.4.",
            resourcepack_model,
        )
        resourcepack_model = "gpt-5.4"
    if resourcepack_api_key and resourcepack_base_url:
        return ToolCallingClientSpec(
            client=OpenAI(api_key=resourcepack_api_key, base_url=resourcepack_base_url),
            model=resourcepack_model,
            backend="resourcepack_openai_compatible",
            reason="Using RESOURCEPACK_API_KEY + RESOURCEPACK_BASE_URL for ReAct tool calling.",
        )

    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
    openai_base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    if openai_api_key and openai_base_url:
        model = preferred or os.getenv("DEFAULT_LLM_MODEL", "").strip() or "gpt-5.4"
        return ToolCallingClientSpec(
            client=OpenAI(api_key=openai_api_key, base_url=openai_base_url),
            model=model,
            backend="custom_openai_compatible",
            reason="Using OPENAI_API_KEY + OPENAI_BASE_URL as a generic OpenAI-compatible endpoint.",
        )

    if openai_api_key:
        model = preferred if _is_openai_native_model(preferred) else "gpt-4o-mini"
        return ToolCallingClientSpec(
            client=OpenAI(api_key=openai_api_key),
            model=model,
            backend="openai_native",
            reason="Using native OpenAI endpoint for ReAct tool calling.",
        )

    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if anthropic_api_key:
        model = preferred if _is_anthropic_model(preferred) else "claude-3-7-sonnet-20250219"
        return ToolCallingClientSpec(
            client=_AnthropicOpenAICompatClient(api_key=anthropic_api_key),
            model=model,
            backend="anthropic_native",
            reason="Using native Anthropic endpoint through an OpenAI-style adapter for ReAct tool calling.",
        )

    return ToolCallingClientSpec(
        client=None,
        model=preferred or os.getenv("DEFAULT_LLM_MODEL", "").strip() or "gpt-5.4",
        backend="unavailable",
        reason="No GPT/OpenAI-compatible or Claude/Anthropic tool-calling credentials detected.",
    )


def _is_openai_native_model(model: str | None) -> bool:
    name = (model or "").strip().lower()
    return name.startswith("gpt-") or name.startswith("o1") or name.startswith("o3")


def _is_anthropic_model(model: str | None) -> bool:
    name = (model or "").strip().lower()
    return name.startswith("claude")


def _ensure_env_loaded() -> None:
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path, override=False)