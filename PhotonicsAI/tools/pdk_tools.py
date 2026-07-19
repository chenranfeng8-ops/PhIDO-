"""PDK generation tools wrapped for agent invocation."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Dict, List

from PhotonicsAI.config import PATH
from PhotonicsAI.core.tooling import Tool


def _load_auto_pdk_generator():
    repo_root = str(PATH.repo)
    scripts_dir = str(PATH.repo / "scripts")
    if repo_root not in sys.path:
        sys.path.append(repo_root)
    if scripts_dir not in sys.path:
        sys.path.append(scripts_dir)

    try:
        return importlib.import_module("scripts.auto_pdk_generator")
    except ImportError:
        return importlib.import_module("auto_pdk_generator")


def resolve_device_type(component_name: str) -> Dict[str, Any]:
    mod = _load_auto_pdk_generator()
    return mod.resolve_device_type(component_name)


def retrieve_papers_multi_source(device_type: str, provided_pdf_path: str) -> Dict[str, Any]:
    mod = _load_auto_pdk_generator()
    return mod.retrieve_papers_multi_source(
        device_type=device_type,
        provided_pdf_path=provided_pdf_path,
    )


def extract_or_aggregate_params(papers: List[Dict[str, Any]], device_type: str) -> Dict[str, Any]:
    mod = _load_auto_pdk_generator()
    return mod.extract_or_aggregate_params(papers, device_type)


def generate_template_file(
    device_type: str,
    params: Dict[str, Any],
    papers: List[Dict[str, Any]],
    confidence_note: str = "",
) -> Dict[str, Any]:
    mod = _load_auto_pdk_generator()
    return mod.generate_template_file(device_type, params, papers, confidence_note)


def parse_template_defaults(template_path: str) -> Dict[str, Any]:
    p = Path(template_path)
    if not p.exists():
        return {
            "ok": False,
            "data": {"params": {}},
            "error": f"Template file not found: {template_path}",
        }

    import re

    content = p.read_text(encoding="utf-8")
    matches = re.findall(r"(\w+):\s*float\s*=\s*([\d.]+)", content)
    params: Dict[str, Any] = {name: float(value) for name, value in matches}
    return {
        "ok": True,
        "data": {"params": params},
        "error": None,
    }


def get_pdk_tools() -> List[Tool]:
    return [
        Tool(
            name="resolve_device_type",
            description="Resolve normalized photonic device type from component name.",
            parameters={
                "type": "object",
                "properties": {"component_name": {"type": "string"}},
                "required": ["component_name"],
            },
            fn=resolve_device_type,
        ),
        Tool(
            name="retrieve_papers_multi_source",
            description="Retrieve current PDF-based paper input for component generation.",
            parameters={
                "type": "object",
                "properties": {
                    "device_type": {"type": "string"},
                    "provided_pdf_path": {"type": "string"},
                },
                "required": ["device_type", "provided_pdf_path"],
            },
            fn=retrieve_papers_multi_source,
        ),
        Tool(
            name="extract_or_aggregate_params",
            description="Extract or aggregate generation parameters from paper inputs.",
            parameters={
                "type": "object",
                "properties": {
                    "papers": {"type": "array", "items": {"type": "object"}},
                    "device_type": {"type": "string"},
                },
                "required": ["papers", "device_type"],
            },
            fn=extract_or_aggregate_params,
        ),
        Tool(
            name="generate_template_file",
            description="Generate one consensus template file from extracted params and papers.",
            parameters={
                "type": "object",
                "properties": {
                    "device_type": {"type": "string"},
                    "params": {"type": "object"},
                    "papers": {"type": "array", "items": {"type": "object"}},
                    "confidence_note": {"type": "string"},
                },
                "required": ["device_type", "params", "papers"],
            },
            fn=generate_template_file,
        ),
        Tool(
            name="parse_template_defaults",
            description="Parse float default parameters from generated template Python file.",
            parameters={
                "type": "object",
                "properties": {"template_path": {"type": "string"}},
                "required": ["template_path"],
            },
            fn=parse_template_defaults,
        ),
    ]
