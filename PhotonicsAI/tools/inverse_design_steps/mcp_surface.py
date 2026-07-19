"""Explicit MCP tool surface for inverse-design ReAct orchestration."""

from __future__ import annotations

from typing import Any, Dict, List

from PhotonicsAI.core.tooling import Tool
from PhotonicsAI.tools.mcp_client import (
    capture_sync,
    detect_python_environment_sync,
    fetch_doc_sync,
    rotate_viewer_sync,
    search_docs_sync,
    show_structures_sync,
    validate_simulation_sync,
)


def inverse_mcp_search_flexcompute_docs(
    query_or_queries: str | List[str],
    *,
    max_results: int = 5,
    package: str | None = None,
    version: str | None = None,
) -> Dict[str, Any]:
    """Search Flexcompute docs and keep query-level traceability."""
    queries = [query_or_queries] if isinstance(query_or_queries, str) else list(query_or_queries)
    normalized = [str(item).strip() for item in queries if str(item).strip()]
    if not normalized:
        return {
            "ok": False,
            "data": {},
            "error": "inverse_mcp_search_flexcompute_docs requires at least one non-empty query.",
        }

    merged_results: List[Dict[str, Any]] = []
    errors: List[str] = []
    for query in normalized:
        result = search_docs_sync(
            query,
            max_results=max(1, int(max_results)),
            package=package,
            version=version,
        )
        if not result.get("ok"):
            errors.append(f"{query}: {result.get('error')}")
            continue
        items = result.get("data", {}).get("results", [])
        for item in items if isinstance(items, list) else []:
            merged_results.append({"query": query, "item": item})

    ok = bool(merged_results)
    return {
        "ok": ok,
        "data": {
            "queries": normalized,
            "results": merged_results,
            "count": len(merged_results),
        },
        "error": None if ok else "; ".join(errors) or "No documentation results returned.",
    }


def inverse_mcp_fetch_flexcompute_doc(
    url: str,
    *,
    package: str | None = None,
    version: str | None = None,
) -> Dict[str, Any]:
    """Fetch a Flexcompute doc page."""
    if not str(url).strip():
        return {
            "ok": False,
            "data": {},
            "error": "inverse_mcp_fetch_flexcompute_doc requires a non-empty url.",
        }
    return fetch_doc_sync(str(url).strip(), package=package, version=version)


def inverse_mcp_detect_python_environment(resource: str | None = None) -> Dict[str, Any]:
    """Detect Python environment from tidy3d-mcp."""
    return detect_python_environment_sync(resource=resource)


def inverse_mcp_validate_simulation(
    file: str | None = None,
    symbol: str | None = None,
    index: int | None = None,
    viewer_id: str | None = None,
) -> Dict[str, Any]:
    """Validate simulation or refresh viewer context in tidy3d-mcp."""
    if not any([file, viewer_id]):
        return {
            "ok": False,
            "data": {},
            "error": "inverse_mcp_validate_simulation requires either file or viewer_id.",
        }
    return validate_simulation_sync(file=file, symbol=symbol, index=index, viewer_id=viewer_id)


def inverse_mcp_rotate_viewer(viewer_id: str, direction: str) -> Dict[str, Any]:
    """Rotate MCP viewer camera to a standard direction."""
    if not str(viewer_id).strip():
        return {"ok": False, "data": {}, "error": "inverse_mcp_rotate_viewer requires viewer_id."}
    if not str(direction).strip():
        return {"ok": False, "data": {}, "error": "inverse_mcp_rotate_viewer requires direction."}
    return rotate_viewer_sync(str(viewer_id).strip(), str(direction).strip())


def inverse_mcp_capture(viewer_id: str) -> Dict[str, Any]:
    """Capture a frame from an existing MCP viewer."""
    if not str(viewer_id).strip():
        return {"ok": False, "data": {}, "error": "inverse_mcp_capture requires viewer_id."}
    return capture_sync(str(viewer_id).strip())


def inverse_mcp_show_structures(
    viewer_id: str,
    visibility: Dict[str, bool] | List[Any] | None = None,
) -> Dict[str, Any]:
    """Toggle structure visibility in an MCP viewer session."""
    if not str(viewer_id).strip():
        return {"ok": False, "data": {}, "error": "inverse_mcp_show_structures requires viewer_id."}
    return show_structures_sync(str(viewer_id).strip(), visibility=visibility)


def get_inverse_design_mcp_tools() -> List[Tool]:
    """Return explicit tidy3d MCP tool set exposed to inverse-engine LLM."""

    return [
        Tool(
            name="inverse_mcp_search_flexcompute_docs",
            description="MCP docs search helper (Step2/Step6): search Flexcompute docs with one or multiple queries.",
            parameters={
                "type": "object",
                "properties": {
                    "query_or_queries": {
                        "oneOf": [
                            {"type": "string"},
                            {"type": "array", "items": {"type": "string"}},
                        ]
                    },
                    "max_results": {"type": "integer", "default": 5},
                    "package": {"type": "string"},
                    "version": {"type": "string"},
                },
                "required": ["query_or_queries"],
            },
            fn=inverse_mcp_search_flexcompute_docs,
        ),
        Tool(
            name="inverse_mcp_fetch_flexcompute_doc",
            description="MCP docs fetch helper (Step2/Step6): fetch full content for a selected Flexcompute doc URL.",
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "package": {"type": "string"},
                    "version": {"type": "string"},
                },
                "required": ["url"],
            },
            fn=inverse_mcp_fetch_flexcompute_doc,
        ),
        Tool(
            name="inverse_mcp_detect_python_environment",
            description="MCP viewer helper: detect Python environment details used by tidy3d tooling.",
            parameters={
                "type": "object",
                "properties": {
                    "resource": {"type": "string"},
                },
            },
            fn=inverse_mcp_detect_python_environment,
        ),
        Tool(
            name="inverse_mcp_validate_simulation",
            description="MCP viewer helper (Step4/Step6): validate simulation file or refresh an existing viewer session.",
            parameters={
                "type": "object",
                "properties": {
                    "file": {"type": "string"},
                    "symbol": {"type": "string"},
                    "index": {"type": "integer", "minimum": 0},
                    "viewer_id": {"type": "string"},
                },
            },
            fn=inverse_mcp_validate_simulation,
        ),
        Tool(
            name="inverse_mcp_rotate_viewer",
            description="MCP viewer helper: rotate viewer camera to TOP/BOTTOM/LEFT/RIGHT/FRONT/BACK.",
            parameters={
                "type": "object",
                "properties": {
                    "viewer_id": {"type": "string"},
                    "direction": {"type": "string"},
                },
                "required": ["viewer_id", "direction"],
            },
            fn=inverse_mcp_rotate_viewer,
        ),
        Tool(
            name="inverse_mcp_capture",
            description="MCP viewer helper: capture one frame from the active viewer.",
            parameters={
                "type": "object",
                "properties": {
                    "viewer_id": {"type": "string"},
                },
                "required": ["viewer_id"],
            },
            fn=inverse_mcp_capture,
        ),
        Tool(
            name="inverse_mcp_show_structures",
            description="MCP viewer helper: toggle structure visibility in the active viewer.",
            parameters={
                "type": "object",
                "properties": {
                    "viewer_id": {"type": "string"},
                    "visibility": {
                        "oneOf": [
                            {"type": "object"},
                            {"type": "array", "items": {}},
                        ]
                    },
                },
                "required": ["viewer_id"],
            },
            fn=inverse_mcp_show_structures,
        ),
    ]
