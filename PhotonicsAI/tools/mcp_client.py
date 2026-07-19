"""Thin async client that talks to the local tidy3d-mcp stdio server.

Lifecycle
---------
``Tidy3DMCPClient`` is designed as a module-level singleton.  Call
``get_mcp_client()`` to obtain (and lazily boot) the shared instance.  The
underlying subprocess is kept alive between calls so that the MCP handshake
cost is paid only once.

The client exposes high-level helpers that mirror the remote tools surfaced by
the ``tidy3d-mcp`` proxy server:

* ``search_docs``  → ``search_flexcompute_docs``
* ``fetch_doc``    → ``fetch_flexcompute_doc``
* ``start_viewer`` → ``validate_simulation`` (viewer bootstrap)
* ``capture``      → ``capture``
* ``show_structures`` → ``show_structures``
* ``check_sim``    → ``validate_simulation`` (re-check by ``viewer_id``)

Both return the standard ``{ok, data, error}`` envelope used everywhere else
in the agent/tool layer.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _is_unknown_tool_error(message: str) -> bool:
    msg = str(message).lower()
    return (
        "unknown tool" in msg
        or ("tool '" in msg and "not listed" in msg)
        or "tool not found" in msg
        or "no tool named" in msg
    )

# ---------------------------------------------------------------------------
# Singleton state
# ---------------------------------------------------------------------------
_client_instance: Optional["Tidy3DMCPClient"] = None
_client_lock = asyncio.Lock() if hasattr(asyncio, "Lock") else None  # guard lazy init
_bg_loop: Optional[asyncio.AbstractEventLoop] = None
_bg_loop_thread: Optional[threading.Thread] = None
_bg_loop_guard = threading.Lock()
_cache_lock = threading.Lock()
_search_cache: Dict[tuple, tuple[float, Dict[str, Any]]] = {}
_fetch_cache: Dict[tuple, tuple[float, Dict[str, Any]]] = {}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value >= 0 else default


def _cache_get(cache: Dict[tuple, tuple[float, Dict[str, Any]]], key: tuple, ttl_s: float) -> Dict[str, Any] | None:
    if ttl_s <= 0:
        return None
    with _cache_lock:
        entry = cache.get(key)
    if entry is None:
        return None
    stored_at, value = entry
    if (time.monotonic() - stored_at) > ttl_s:
        with _cache_lock:
            cache.pop(key, None)
        return None
    return dict(value)


def _cache_set(cache: Dict[tuple, tuple[float, Dict[str, Any]]], key: tuple, value: Dict[str, Any]) -> None:
    with _cache_lock:
        cache[key] = (time.monotonic(), dict(value))


class Tidy3DMCPClient:
    """Wraps a long-lived ``fastmcp.Client`` connected to ``tidy3d-mcp``."""

    def __init__(self) -> None:
        self._client: Any = None  # fastmcp.Client, created lazily
        self._connected = False
        self._viewer_tools_available = False
        self._available_tools: set[str] = set()
        self._search_tool_disabled = False
        self._fetch_tool_disabled = False

    # -- connection management -----------------------------------------------

    async def connect(self) -> None:
        """Start the MCP server subprocess and initialise the session."""
        if self._connected:
            return

        from fastmcp import Client  # deferred so module import stays cheap
        from fastmcp.client.transports import StdioTransport

        mcp_exe = self._find_mcp_executable()
        base_args = ["--host", "vscode", "--enable-viewer"]
        args = list(base_args)
        viewer_bridge = (
            os.getenv("TIDY3D_VIEWER_BRIDGE_URL", "").strip()
            or os.getenv("TIDY3D_VIEWER_BRIDGE", "").strip()
        )
        # Always pass --viewer-bridge so the server registers viewer tools.
        # Without this, _bootstrap_viewer_bridge() fails in non-interactive
        # environments and viewer_enabled becomes False → tools not added.
        # When no real bridge is configured we use a placeholder URL; the
        # tools will be registered but invoke_viewer_command will fail at
        # runtime, which our wrappers handle gracefully.
        bridge_url = (
            viewer_bridge
            or os.getenv("TIDY3D_MCP_FORCE_VIEWER_BRIDGE", "").strip()
            or "http://127.0.0.1:0"
        )
        args.extend(["--viewer-bridge", bridge_url])

        async def _open_client(open_args: List[str]) -> Any:
            transport = StdioTransport(command=mcp_exe, args=open_args)
            client_timeout_s = _env_float("TIDY3D_MCP_CLIENT_TIMEOUT_S", 30.0)
            client = Client(transport, timeout=client_timeout_s)
            await client.__aenter__()
            return client

        self._client = await _open_client(args)
        self._connected = True

        # Verify that viewer tools were registered.
        try:
            tools = [t.name for t in await self._client.list_tools()]
        except Exception:
            tools = []
        self._available_tools = set(tools)
        self._search_tool_disabled = False
        self._fetch_tool_disabled = False

        self._viewer_tools_available = "validate_simulation" in tools
        if not self._viewer_tools_available:
            logger.warning(
                "Viewer tools not exposed even with --viewer-bridge %s; "
                "viewer operations will return graceful errors.",
                bridge_url,
            )

        logger.info(
            "tidy3d-mcp client connected (viewer_tools=%s, bridge=%s)",
            self._viewer_tools_available,
            bridge_url,
        )

    async def close(self) -> None:
        """Shut down the MCP server subprocess."""
        if self._client is not None and self._connected:
            try:
                await self._client.__aexit__(None, None, None)
            except Exception:
                pass
            self._connected = False
            self._client = None
            self._viewer_tools_available = False

    @property
    def viewer_available(self) -> bool:
        """Whether the tidy3d-mcp server exposed viewer tools."""
        return self._viewer_tools_available

    # -- tool wrappers -------------------------------------------------------

    async def _call_tool_with_retry(
        self, tool_name: str, args: Dict[str, Any], max_retries: int | None = None
    ) -> Any:
        """Call an MCP tool with automatic retry on transient errors."""
        await self._ensure_connected()
        retries = _env_int("TIDY3D_MCP_TOOL_RETRIES", 2) if max_retries is None else max(0, int(max_retries))
        last_exc: Exception | None = None
        for attempt in range(retries + 1):
            try:
                return await self._client.call_tool(tool_name, args)
            except Exception as exc:
                last_exc = exc
                msg = str(exc).lower()
                if "client is not connected" in msg or "not connected" in msg:
                    logger.debug("MCP client disconnected; reconnecting before retry")
                    self._connected = False
                    await self._ensure_connected()
                if attempt < retries:
                    import asyncio as _asyncio
                    await _asyncio.sleep(0.5 * (attempt + 1))
                    logger.debug(
                        "Retry %d/%d for %s: %s", attempt + 1, retries, tool_name, exc
                    )
        raise last_exc  # type: ignore[misc]

    async def _call_tool_aliases(
        self,
        tool_names: List[str],
        args: Dict[str, Any],
    ) -> Any:
        """Try multiple tool names to stay compatible across tidy3d-mcp versions."""
        if self._available_tools:
            filtered = [name for name in tool_names if name in self._available_tools]
            if filtered:
                tool_names = filtered
        last_exc: Exception | None = None
        for tool_name in tool_names:
            try:
                return await self._call_tool_with_retry(tool_name, args)
            except Exception as exc:
                last_exc = exc
                if _is_unknown_tool_error(str(exc)):
                    continue
                raise
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("No candidate tool names were provided.")

    async def search_docs(
        self,
        query: str,
        *,
        max_results: int = 5,
        package: str | None = None,
        version: str | None = None,
    ) -> Dict[str, Any]:
        """Call the remote ``search_flexcompute_docs`` tool."""
        if self._search_tool_disabled:
            return {
                "ok": False,
                "data": {},
                "error": "MCP search docs tool is unavailable in current server; skipped.",
            }
        await self._ensure_connected()
        args: Dict[str, Any] = {
            "query_or_queries": query,
            "max_results": max_results,
        }
        if package:
            args["package"] = package
        if version:
            args["version"] = version

        search_aliases = [
            "search_flexcompute_docs",
            "search_docs",
            "query_flexcompute_docs",
        ]
        try:
            result = await self._call_tool_aliases(search_aliases, args)
            # The MCP server returns content[0].text as a JSON string containing
            # a list of {"url": ..., "content": ...} objects.
            docs = self._extract_structured(result, fallback_key="result")
            if docs is None:
                # No structured_content; parse the text content as JSON.
                text_parts = self._extract_text(result)
                if text_parts:
                    import json as _json
                    try:
                        docs = _json.loads(text_parts[0])
                    except (ValueError, IndexError):
                        docs = text_parts
            return {
                "ok": True,
                "data": {"query": query, "results": docs if docs else []},
                "error": None,
            }
        except Exception as exc:
            if _is_unknown_tool_error(str(exc)):
                self._search_tool_disabled = True
            logger.warning("search_flexcompute_docs failed: %s", exc)
            return {"ok": False, "data": {}, "error": str(exc)}

    async def fetch_doc(
        self,
        url: str,
        *,
        package: str | None = None,
        version: str | None = None,
    ) -> Dict[str, Any]:
        """Call the remote ``fetch_flexcompute_doc`` tool."""
        if self._fetch_tool_disabled:
            return {
                "ok": False,
                "data": {},
                "error": "MCP fetch doc tool is unavailable in current server; skipped.",
            }
        await self._ensure_connected()
        args: Dict[str, Any] = {"url": url}
        if package:
            args["package"] = package
        if version:
            args["version"] = version

        fetch_aliases = [
            "fetch_flexcompute_doc",
            "fetch_doc",
        ]
        try:
            result = await self._call_tool_aliases(fetch_aliases, args)
            # structured_content.result may contain {"url": ..., "content": ...}
            structured = self._extract_structured(result, fallback_key="result")
            content = ""
            if isinstance(structured, dict) and structured.get("content"):
                content = structured["content"]
            elif isinstance(structured, str):
                content = structured
            else:
                # Fallback: parse text content as JSON
                text_parts = self._extract_text(result)
                if text_parts:
                    import json as _json
                    try:
                        parsed = _json.loads(text_parts[0])
                        if isinstance(parsed, dict):
                            content = parsed.get("content", "")
                        else:
                            content = text_parts[0]
                    except (ValueError, IndexError):
                        content = "\n".join(text_parts)
            return {
                "ok": bool(content),
                "data": {"url": url, "content": content},
                "error": None if content else "Document returned empty content.",
            }
        except Exception as exc:
            if _is_unknown_tool_error(str(exc)):
                self._fetch_tool_disabled = True
            logger.warning("fetch_flexcompute_doc failed: %s", exc)
            return {"ok": False, "data": {}, "error": str(exc)}

    async def list_tools(self) -> List[str]:
        """Return names of tools the MCP server advertises."""
        await self._ensure_connected()
        tools = await self._client.list_tools()
        return [t.name for t in tools]

    # -- viewer tool wrappers ------------------------------------------------

    async def start_viewer(
        self,
        file: str,
        symbol: str = "sim",
    ) -> Dict[str, Any]:
        """Bootstrap viewer session and return ``viewer_id``.

        New tidy3d-mcp versions expose ``validate_simulation`` for viewer start.
        Older deployments may still expose ``tidy3d_start_viewer``.
        """
        try:
            result = await self._call_tool_aliases(
                ["validate_simulation", "tidy3d_start_viewer"],
                {"file": file, "symbol": symbol},
            )
            structured = self._extract_structured(result, fallback_key="result")
            viewer_id = None
            if isinstance(structured, dict):
                viewer_id = structured.get("viewer_id") or structured.get("id")
            elif isinstance(structured, str):
                viewer_id = structured
            text = self._extract_text(result)
            if not viewer_id and text:
                viewer_id = text[0].strip()
            return {
                "ok": bool(viewer_id),
                "data": {"viewer_id": viewer_id, "file": file, "symbol": symbol},
                "error": None if viewer_id else "start_viewer returned no viewer_id.",
            }
        except Exception as exc:
            logger.warning("start_viewer failed: %s", exc)
            return {"ok": False, "data": {}, "error": str(exc)}

    async def capture(self, viewer_id: str) -> Dict[str, Any]:
        """Capture the current 3D viewer frame."""
        try:
            result = await self._call_tool_aliases(
                ["capture", "tidy3d_capture"],
                {"viewer_id": viewer_id},
            )
            structured = self._extract_structured(result, fallback_key="result")
            image_data = None
            if isinstance(structured, dict):
                image_data = (
                    structured.get("images")
                    or structured.get("image")
                    or structured.get("data")
                )
            text = self._extract_text(result)
            return {
                "ok": True,
                "data": {
                    "viewer_id": viewer_id,
                    "image": image_data,
                    "text": text,
                },
                "error": None,
            }
        except Exception as exc:
            logger.warning("capture failed: %s", exc)
            return {"ok": False, "data": {}, "error": str(exc)}

    async def show_structures(
        self,
        viewer_id: str,
        visibility: Any = None,
    ) -> Dict[str, Any]:
        """Toggle structure visibility."""
        args: Dict[str, Any] = {"viewer_id": viewer_id}
        if visibility is not None:
            if isinstance(visibility, dict):
                # Current tidy3d-mcp expects a list of flags.
                args["visibility"] = list(visibility.values())
            else:
                args["visibility"] = visibility
        else:
            # MCP server requires visibility; send empty list as no-op default.
            args["visibility"] = []
        try:
            result = await self._call_tool_aliases(
                ["show_structures", "tidy3d_show_structures"],
                args,
            )
            structured = self._extract_structured(result, fallback_key="result")
            return {
                "ok": True,
                "data": {"viewer_id": viewer_id, "result": structured},
                "error": None,
            }
        except Exception as exc:
            logger.warning("show_structures failed: %s", exc)
            return {"ok": False, "data": {}, "error": str(exc)}

    async def validate_simulation(
        self,
        file: str | None = None,
        symbol: str | None = None,
        index: int | None = None,
        viewer_id: str | None = None,
    ) -> Dict[str, Any]:
        """Direct passthrough for the MCP ``validate_simulation`` tool."""
        args: Dict[str, Any] = {}
        if file:
            args["file"] = file
        if symbol:
            args["symbol"] = symbol
        if index is not None:
            args["index"] = int(index)
        if viewer_id:
            args["viewer_id"] = viewer_id

        try:
            result = await self._call_tool_aliases(
                ["validate_simulation", "tidy3d_start_viewer", "tidy3d_check_simulation"],
                args,
            )
            structured = self._extract_structured(result, fallback_key="result")
            text = self._extract_text(result)
            data: Dict[str, Any] = {
                "result": structured,
                "text": text,
            }
            if isinstance(structured, dict):
                viewer = structured.get("viewer_id") or structured.get("id")
                if viewer:
                    data["viewer_id"] = viewer
            return {"ok": True, "data": data, "error": None}
        except Exception as exc:
            logger.warning("validate_simulation failed: %s", exc)
            return {"ok": False, "data": {}, "error": str(exc)}

    async def check_sim(self, viewer_id: str) -> Dict[str, Any]:
        """Validate simulation health for an existing viewer."""
        try:
            result = await self._call_tool_aliases(
                ["validate_simulation", "tidy3d_check_simulation"],
                {"viewer_id": viewer_id},
            )
            structured = self._extract_structured(result, fallback_key="result")
            text = self._extract_text(result)
            return {
                "ok": True,
                "data": {"viewer_id": viewer_id, "result": structured, "text": text},
                "error": None,
            }
        except Exception as exc:
            logger.warning("check_sim failed: %s", exc)
            return {"ok": False, "data": {}, "error": str(exc)}

    async def rotate_viewer(self, viewer_id: str, direction: str) -> Dict[str, Any]:
        """Rotate viewer camera to a predefined direction."""
        try:
            result = await self._call_tool_with_retry(
                "rotate_viewer", {"viewer_id": viewer_id, "direction": direction}
            )
            structured = self._extract_structured(result, fallback_key="result")
            text = self._extract_text(result)
            return {
                "ok": True,
                "data": {"viewer_id": viewer_id, "result": structured, "text": text},
                "error": None,
            }
        except Exception as exc:
            logger.warning("rotate_viewer failed: %s", exc)
            return {"ok": False, "data": {}, "error": str(exc)}

    async def detect_python_environment(self, resource: str | None = None) -> Dict[str, Any]:
        """Expose tidy3d-mcp's python environment detection tool."""
        args: Dict[str, Any] = {}
        if resource:
            args["resource"] = resource
        try:
            result = await self._call_tool_with_retry("detect_python_environment", args)
            structured = self._extract_structured(result, fallback_key="result")
            text = self._extract_text(result)
            return {
                "ok": True,
                "data": {"result": structured, "text": text},
                "error": None,
            }
        except Exception as exc:
            logger.warning("detect_python_environment failed: %s", exc)
            return {"ok": False, "data": {}, "error": str(exc)}

    # -- internal helpers ----------------------------------------------------

    async def _ensure_connected(self) -> None:
        if not self._connected:
            await self.connect()

    @staticmethod
    def _find_mcp_executable() -> str:
        """Locate the ``tidy3d-mcp`` executable shipped in the venv/conda env."""
        python_dir = Path(sys.executable).parent
        # venv on Windows: Scripts/ is the same dir as python.exe
        # conda on Windows: python.exe is in envs/NAME/, scripts in envs/NAME/Scripts/
        candidates = [
            python_dir / "tidy3d-mcp.exe",
            python_dir / "tidy3d-mcp",
            python_dir / "Scripts" / "tidy3d-mcp.exe",
            python_dir / "Scripts" / "tidy3d-mcp",
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        raise FileNotFoundError(
            "tidy3d-mcp executable not found. "
            "Install it with: pip install tidy3d-mcp"
        )

    @staticmethod
    def _extract_text(result: Any) -> List[str]:
        """Pull plain-text segments from an MCP ToolResult."""
        parts: List[str] = []
        items = result if isinstance(result, list) else [result]
        for item in items:
            if hasattr(item, "text"):
                parts.append(item.text)
            elif hasattr(item, "content"):
                for c in item.content:
                    if hasattr(c, "text"):
                        parts.append(c.text)
            elif isinstance(item, str):
                parts.append(item)
        return parts

    @staticmethod
    def _extract_structured(result: Any, fallback_key: str = "result") -> Any:
        """Extract structured data from a CallToolResult.

        ``fastmcp.Client.call_tool`` returns a ``CallToolResult`` that may carry
        a ``structured_content`` dict and/or a ``data`` list of domain objects.
        """
        # Prefer structured_content → data → None
        if hasattr(result, "structured_content") and result.structured_content:
            sc = result.structured_content
            if isinstance(sc, dict) and fallback_key in sc:
                return sc[fallback_key]
            return sc
        if hasattr(result, "data") and result.data:
            items = result.data
            if isinstance(items, list):
                converted = []
                for item in items:
                    if hasattr(item, "__dict__"):
                        converted.append(
                            {k: v for k, v in item.__dict__.items() if not k.startswith("_")}
                        )
                    elif isinstance(item, dict):
                        converted.append(item)
                    else:
                        converted.append(str(item))
                return converted
            return items
        return None


# ---------------------------------------------------------------------------
# Module-level singleton accessor
# ---------------------------------------------------------------------------

async def get_mcp_client() -> Tidy3DMCPClient:
    """Return the shared ``Tidy3DMCPClient``, connecting on first call."""
    global _client_instance
    if _client_instance is None:
        _client_instance = Tidy3DMCPClient()
        await _client_instance.connect()
    elif not _client_instance._connected:
        await _client_instance.connect()
    return _client_instance


def _sync_call(coro: Any, *, timeout: float = 120.0) -> Any:
    """Run *coro* from synchronous context, reusing a running loop if possible."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        # We are inside an existing event loop (e.g. Streamlit).
        # Schedule the coroutine in a background thread.
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result(timeout=max(1.0, timeout))
    else:
        # Use one persistent background event loop so the MCP stdio client
        # stays bound to the same loop across sequential sync calls.
        bg_loop = _ensure_background_loop()
        future = asyncio.run_coroutine_threadsafe(coro, bg_loop)
        return future.result(timeout=max(1.0, timeout))


def _ensure_background_loop() -> asyncio.AbstractEventLoop:
    """Create/reuse a dedicated background loop for sync MCP wrappers."""
    global _bg_loop, _bg_loop_thread
    with _bg_loop_guard:
        if _bg_loop is not None and _bg_loop.is_running():
            return _bg_loop

        loop = asyncio.new_event_loop()

        def _runner() -> None:
            asyncio.set_event_loop(loop)
            loop.run_forever()

        thread = threading.Thread(
            target=_runner,
            name="tidy3d-mcp-bg-loop",
            daemon=True,
        )
        thread.start()

        _bg_loop = loop
        _bg_loop_thread = thread
        return loop


def search_docs_sync(
    query: str,
    *,
    max_results: int = 5,
    package: str | None = None,
    version: str | None = None,
    timeout_s: float | None = None,
) -> Dict[str, Any]:
    """Synchronous wrapper around ``Tidy3DMCPClient.search_docs``."""
    cache_ttl_s = _env_float("TIDY3D_MCP_SEARCH_CACHE_TTL_S", 300.0)
    cache_key = (query, int(max_results), package or "", version or "")
    cached = _cache_get(_search_cache, cache_key, cache_ttl_s)
    if cached is not None:
        return cached

    async def _inner():
        client = await get_mcp_client()
        return await client.search_docs(
            query, max_results=max_results, package=package, version=version
        )
    timeout_value = timeout_s if (timeout_s is not None and timeout_s > 0) else _env_float("TIDY3D_MCP_SYNC_TIMEOUT_S", 45.0)
    try:
        result = _sync_call(_inner(), timeout=timeout_value)
        if isinstance(result, dict) and result.get("ok"):
            _cache_set(_search_cache, cache_key, result)
        return result
    except Exception as exc:
        return {"ok": False, "data": {}, "error": str(exc)}


def fetch_doc_sync(
    url: str,
    *,
    package: str | None = None,
    version: str | None = None,
    timeout_s: float | None = None,
) -> Dict[str, Any]:
    """Synchronous wrapper around ``Tidy3DMCPClient.fetch_doc``."""
    cache_ttl_s = _env_float("TIDY3D_MCP_FETCH_CACHE_TTL_S", 900.0)
    cache_key = (url, package or "", version or "")
    cached = _cache_get(_fetch_cache, cache_key, cache_ttl_s)
    if cached is not None:
        return cached

    async def _inner():
        client = await get_mcp_client()
        return await client.fetch_doc(url, package=package, version=version)
    timeout_value = timeout_s if (timeout_s is not None and timeout_s > 0) else _env_float("TIDY3D_MCP_SYNC_TIMEOUT_S", 45.0)
    try:
        result = _sync_call(_inner(), timeout=timeout_value)
        if isinstance(result, dict) and result.get("ok"):
            _cache_set(_fetch_cache, cache_key, result)
        return result
    except Exception as exc:
        return {"ok": False, "data": {}, "error": str(exc)}


def start_viewer_sync(file: str, symbol: str = "sim") -> Dict[str, Any]:
    """Synchronous wrapper around ``Tidy3DMCPClient.start_viewer``."""
    async def _inner():
        client = await get_mcp_client()
        return await client.start_viewer(file, symbol)
    return _sync_call(_inner())


def capture_sync(viewer_id: str) -> Dict[str, Any]:
    """Synchronous wrapper around ``Tidy3DMCPClient.capture``."""
    async def _inner():
        client = await get_mcp_client()
        return await client.capture(viewer_id)
    return _sync_call(_inner())


def show_structures_sync(
    viewer_id: str,
    visibility: Dict[str, bool] | None = None,
) -> Dict[str, Any]:
    """Synchronous wrapper around ``Tidy3DMCPClient.show_structures``."""
    async def _inner():
        client = await get_mcp_client()
        return await client.show_structures(viewer_id, visibility)
    return _sync_call(_inner())


def validate_simulation_sync(
    file: str | None = None,
    symbol: str | None = None,
    index: int | None = None,
    viewer_id: str | None = None,
) -> Dict[str, Any]:
    """Synchronous wrapper around ``Tidy3DMCPClient.validate_simulation``."""

    async def _inner():
        client = await get_mcp_client()
        return await client.validate_simulation(
            file=file,
            symbol=symbol,
            index=index,
            viewer_id=viewer_id,
        )

    return _sync_call(_inner())


def check_sim_sync(viewer_id: str) -> Dict[str, Any]:
    """Synchronous wrapper around ``Tidy3DMCPClient.check_sim``."""
    async def _inner():
        client = await get_mcp_client()
        return await client.check_sim(viewer_id)
    return _sync_call(_inner())


def rotate_viewer_sync(viewer_id: str, direction: str) -> Dict[str, Any]:
    """Synchronous wrapper around ``Tidy3DMCPClient.rotate_viewer``."""
    async def _inner():
        client = await get_mcp_client()
        return await client.rotate_viewer(viewer_id, direction)
    return _sync_call(_inner())


def detect_python_environment_sync(resource: str | None = None) -> Dict[str, Any]:
    """Synchronous wrapper around ``Tidy3DMCPClient.detect_python_environment``."""
    async def _inner():
        client = await get_mcp_client()
        return await client.detect_python_environment(resource)
    return _sync_call(_inner())
