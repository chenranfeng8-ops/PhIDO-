"""Shared tool protocol primitives with no agent/tool layer dependency."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict

Envelope = Dict[str, Any]
ToolFn = Callable[..., Envelope]


@dataclass
class Tool:
    """Tool registration entry for agent orchestration."""

    name: str
    description: str
    parameters: Dict[str, Any]
    fn: ToolFn
    timeout: float | None = None  # Optional per-tool timeout override (seconds)

    def call(self, **kwargs: Any) -> Envelope:
        return self.fn(**kwargs)
