"""Shared patch-application helpers for Step6 local replan flows."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple


def apply_patch_actions(root: Any, actions: Iterable[Any]) -> List[str]:
    """Apply machine-readable patch actions onto a nested dict/list payload."""

    applied_paths: List[str] = []
    for action in actions:
        action_dict = action.model_dump() if hasattr(action, "model_dump") else dict(action)
        path = str(action_dict.get("path", "")).strip()
        if not path or path == "root":
            continue
        if apply_single_patch_action(root, action_dict):
            applied_paths.append(path)
    return applied_paths


def apply_single_patch_action(root: Any, action: Dict[str, Any]) -> bool:
    """Apply a single patch action onto a nested dict/list payload."""

    path = str(action.get("path", "")).strip()
    operation = str(action.get("action", "set_value")).strip().lower()
    value = action.get("value")

    if operation == "set_value":
        return _set_path_value(root, path, value)
    if operation == "add_item":
        return _add_path_value(root, path, value)
    if operation == "remove_item":
        return _remove_path_value(root, path)
    if operation == "regenerate":
        return False
    return False


def _set_path_value(root: Any, path: str, value: Any) -> bool:
    parent, key = _resolve_parent(root, path, create=True)
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


def _add_path_value(root: Any, path: str, value: Any) -> bool:
    container = _get_path_value(root, path)
    if isinstance(container, list):
        container.append(value)
        return True

    parent, key = _resolve_parent(root, path, create=True)
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


def _remove_path_value(root: Any, path: str) -> bool:
    parent, key = _resolve_parent(root, path, create=False)
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


def _get_path_value(root: Any, path: str) -> Any:
    current = root
    for part in _parse_path(path):
        if isinstance(part, int):
            if not isinstance(current, list) or part >= len(current):
                return None
            current = current[part]
        else:
            if not isinstance(current, dict) or part not in current:
                return None
            current = current[part]
    return current


def _resolve_parent(root: Any, path: str, *, create: bool) -> Tuple[Any, str | int] | Tuple[None, None]:
    parts = _parse_path(path)
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


def _parse_path(path: str) -> List[str | int]:
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
