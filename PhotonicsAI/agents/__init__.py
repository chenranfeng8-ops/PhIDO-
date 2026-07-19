"""Agent package for photonic design workflows."""

from __future__ import annotations

__all__ = ["PDKAgent", "create_pdk_agent"]


def __getattr__(name: str):
    """Lazy exports to avoid importing heavy agent graph at package import time."""

    if name == "PDKAgent":
        from .pdk_agent import PDKAgent

        return PDKAgent
    if name == "create_pdk_agent":
        from .pdk_agent import create_pdk_agent

        return create_pdk_agent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
