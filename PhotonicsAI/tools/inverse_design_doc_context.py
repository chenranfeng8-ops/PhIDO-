"""Documentation retrieval helpers for inverse-design step 9.1 item 2."""

from __future__ import annotations

import json
import os
import re
import time
import concurrent.futures
from typing import Any, Callable, Dict, List

from pydantic import BaseModel, ConfigDict, Field

from PhotonicsAI.tools.inverse_design_requirements import (
    InverseDesignRequirement,
    require_complete_inverse_design_requirement,
)
from PhotonicsAI.tools.inverse_design_working_memory import (
    InverseDesignWorkingMemory,
    get_inverse_design_working_memory,
)
from PhotonicsAI.tools.tidy3d_tools import fetch_tidy3d_doc, search_tidy3d_docs


class StrictModel(BaseModel):
    """Base model that rejects unknown fields to keep parser output stable."""

    model_config = ConfigDict(extra="forbid")


class DocumentationRuleSummary(StrictModel):
    """Compact rule summary extracted from a fetched Tidy3D document."""

    rule: str
    evidence: str = ""


class DocumentationReference(StrictModel):
    """Traceable documentation evidence retained for later config generation."""

    url: str
    title: str = ""
    summary: str = ""
    rules: List[DocumentationRuleSummary] = Field(default_factory=list)
    matched_query: str = ""


class RetrievalGuidance(StrictModel):
    """Structured simulation guidance extracted from retrieved documentation."""

    source_type: str = "mode"
    require_pml: bool = True
    recommended_monitors: List[str] = Field(default_factory=list)
    mesh_advice: str = ""
    inverse_design_hint: str = ""


class StructurePortSpec(StrictModel):
    """Port-level structure hint produced by Step2 for downstream config generation."""

    name: str
    role: str = ""
    direction: str = ""
    expected_monitor_types: List[str] = Field(default_factory=list)


class StructureContext(StrictModel):
    """Step2 structural interpretation for the requested single device.

    This is intentionally limited to Tidy3D documentation plus the current
    local Tidy3D builder path.  It does not perform DesignLibrary retrieval.
    """

    component_type: str
    topology: str = ""
    builder_module: str = "PhotonicsAI.Photon.tidy3d_runner"
    builder_function: str = ""
    builder_reference: str = ""
    parameter_keys: List[str] = Field(default_factory=list)
    ports: List[StructurePortSpec] = Field(default_factory=list)
    source_strategy: str = "mode_source_at_input_port"
    monitor_strategy: str = "field_flux_mode_monitors_at_ports"
    reusable_code_refs: List[str] = Field(default_factory=list)
    evidence_urls: List[str] = Field(default_factory=list)
    design_library_used: bool = False
    rationale: str = ""


class InverseDesignDocContext(StrictModel):
    """Intermediate result for implementation step 9.1 item 2."""

    requirement: InverseDesignRequirement
    queries: List[str] = Field(default_factory=list)
    references: List[DocumentationReference] = Field(default_factory=list)
    guidance: RetrievalGuidance = Field(default_factory=RetrievalGuidance)
    structure_context: StructureContext | None = None


SearchFn = Callable[..., Dict[str, Any]]
FetchFn = Callable[..., Dict[str, Any]]
LLMCallFn = Callable[[str, str, str], str]

_RULE_PATTERNS = [
    re.compile(r"(?P<sentence>[^.]*\bpml\b[^.]*)\.", re.IGNORECASE),
    re.compile(r"(?P<sentence>[^.]*\bmode source\b[^.]*)\.", re.IGNORECASE),
    re.compile(r"(?P<sentence>[^.]*\bflux monitor\b[^.]*)\.", re.IGNORECASE),
    re.compile(r"(?P<sentence>[^.]*\bfield monitor\b[^.]*)\.", re.IGNORECASE),
    re.compile(r"(?P<sentence>[^.]*\bmesh\b[^.]*)\.", re.IGNORECASE),
    re.compile(r"(?P<sentence>[^.]*\binverse design\b[^.]*)\.", re.IGNORECASE),
    re.compile(r"(?P<sentence>[^.]*\badjoint\b[^.]*)\.", re.IGNORECASE),
]


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


def _is_timeout_error(message: str) -> bool:
    msg = (message or "").lower()
    return (
        "timed out" in msg
        or "timeout" in msg
        or "connecterror" in msg
        or "connect error" in msg
    )


def _call_search_with_timeout(
    search_fn: SearchFn,
    *,
    query: str,
    max_results: int,
    timeout_s: float,
) -> Dict[str, Any]:
    try:
        return search_fn(query=query, max_results=max_results, timeout_s=timeout_s)
    except TypeError:
        # Backward compatibility for existing function signatures.
        return search_fn(query=query, max_results=max_results)


def _call_fetch_with_timeout(
    fetch_fn: FetchFn,
    *,
    url: str,
    timeout_s: float,
) -> Dict[str, Any]:
    try:
        return fetch_fn(url=url, timeout_s=timeout_s)
    except TypeError:
        # Backward compatibility for existing function signatures.
        return fetch_fn(url=url)


class DocRetrievalPlan(StrictModel):
    """LLM-generated retrieval plan for Step2 with deterministic fallback."""

    queries: List[str] = Field(default_factory=list)
    rationale: str = ""


def retrieve_inverse_design_doc_context(
    requirement: InverseDesignRequirement | str,
    *,
    max_results: int = 3,
    max_queries: int = 6,
    max_docs_per_query: int = 2,
    max_references: int = 6,
    time_budget_s: float = 180.0,
    search_timeout_s: float | None = None,
    fetch_timeout_s: float | None = None,
    fetch_workers: int | None = None,
    max_consecutive_timeouts: int | None = None,
    extra_queries: List[str] | None = None,
    search_fn: SearchFn = search_tidy3d_docs,
    fetch_fn: FetchFn = fetch_tidy3d_doc,
    use_llm_planner: bool | None = None,
    llm_call_fn: LLMCallFn | None = None,
    llm_model: str = "gpt-5.4",
    memory_store: InverseDesignWorkingMemory | None = None,
) -> InverseDesignDocContext:
    """Retrieve Tidy3D documentation context for a parsed inverse-design requirement."""

    parsed = _coerce_requirement(requirement)
    memory = memory_store or get_inverse_design_working_memory()
    base_queries = build_doc_queries(parsed)
    if extra_queries:
        base_queries = _dedupe_queries(base_queries + [str(query) for query in extra_queries if str(query).strip()])
    queries = _apply_llm_query_plan(
        parsed,
        base_queries,
        use_llm_planner=use_llm_planner,
        llm_call_fn=llm_call_fn,
        llm_model=llm_model,
        memory_store=memory,
    )
    if max_queries > 0:
        queries = queries[:max_queries]
    if max_results <= 0:
        synthetic_reference = DocumentationReference(
            url="memory://step2/degraded/no_mcp_docs",
            title="Step2 degraded-mode fallback guidance",
            summary=(
                "No external MCP docs were retrieved. "
                "Use conservative inverse-design defaults: PML boundaries, mode source, "
                "field monitor and flux monitor, then enforce Step4 hard validation."
            ),
            rules=[
                DocumentationRuleSummary(
                    rule="Use PML boundaries on all simulation sides.",
                    evidence="degraded_mode_fallback",
                ),
                DocumentationRuleSummary(
                    rule="Use a mode source for guided-wave excitation.",
                    evidence="degraded_mode_fallback",
                ),
                DocumentationRuleSummary(
                    rule="Keep both field and flux monitors enabled.",
                    evidence="degraded_mode_fallback",
                ),
            ],
            matched_query=queries[0] if queries else "",
        )
        memory.record(
            stage="step2_doc_retrieval",
            key=parsed.component_type,
            summary="Step2 MCP retrieval skipped by configuration (max_results<=0).",
            metadata={
                "max_results": max_results,
                "queries": queries,
                "degraded_mode": True,
                "skip_reason": "max_results_non_positive",
            },
        )
        return InverseDesignDocContext(
            requirement=parsed,
            queries=queries,
            references=[synthetic_reference],
            guidance=_derive_guidance([synthetic_reference]),
            structure_context=_derive_structure_context(parsed, [synthetic_reference], _derive_guidance([synthetic_reference])),
        )

    references: List[DocumentationReference] = []
    seen_urls: set[str] = set()
    started = time.monotonic()
    budget_exhausted = False
    consecutive_timeouts = 0
    resolved_search_timeout_s = search_timeout_s if (search_timeout_s and search_timeout_s > 0) else _env_float("TIDY3D_STEP2_SEARCH_TIMEOUT_S", 45.0)
    resolved_fetch_timeout_s = fetch_timeout_s if (fetch_timeout_s and fetch_timeout_s > 0) else _env_float("TIDY3D_STEP2_FETCH_TIMEOUT_S", 45.0)
    resolved_fetch_workers = max(1, fetch_workers if (fetch_workers and fetch_workers > 0) else _env_int("TIDY3D_STEP2_FETCH_WORKERS", 2))
    resolved_timeout_threshold = max(1, max_consecutive_timeouts if (max_consecutive_timeouts and max_consecutive_timeouts > 0) else _env_int("TIDY3D_STEP2_MAX_CONSEC_TIMEOUTS", 3))

    for query in queries:
        if time_budget_s > 0 and (time.monotonic() - started) >= time_budget_s:
            budget_exhausted = True
            break
        if consecutive_timeouts >= resolved_timeout_threshold:
            memory.record(
                stage="step2_doc_retrieval",
                key=query,
                summary="Step2 MCP timeout threshold reached; switch to degraded mode for this run.",
                metadata={
                    "timeout_threshold": resolved_timeout_threshold,
                    "consecutive_timeouts": consecutive_timeouts,
                    "degraded_mode": True,
                },
            )
            break

        try:
            search_result = _call_search_with_timeout(
                search_fn,
                query=query,
                max_results=max_results,
                timeout_s=resolved_search_timeout_s,
            )
        except Exception as exc:
            if _is_timeout_error(str(exc)):
                consecutive_timeouts += 1
            memory.record(
                stage="step2_doc_retrieval",
                key=query,
                summary="MCP search call failed; skipping this query.",
                issues=[str(exc)],
                metadata={"phase": "search", "timeout_like": _is_timeout_error(str(exc))},
            )
            continue
        if not search_result.get("ok") and _is_timeout_error(str(search_result.get("error") or "")):
            consecutive_timeouts += 1
        elif search_result.get("ok"):
            consecutive_timeouts = 0
        if not search_result.get("ok"):
            continue

        raw_results = search_result.get("data", {}).get("results", [])
        if not isinstance(raw_results, list):
            raw_results = []
        if max_docs_per_query > 0:
            raw_results = raw_results[:max_docs_per_query]

        urls: List[str] = []
        raw_result_by_url: Dict[str, Dict[str, Any]] = {}
        for raw_result in raw_results:
            url = str(raw_result.get("url", "")).strip()
            if not url or url in seen_urls:
                continue
            urls.append(url)
            raw_result_by_url[url] = dict(raw_result)

        if urls:
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(resolved_fetch_workers, len(urls))) as pool:
                future_map = {
                    pool.submit(
                        _call_fetch_with_timeout,
                        fetch_fn,
                        url=url,
                        timeout_s=resolved_fetch_timeout_s,
                    ): url
                    for url in urls
                }
                for future in concurrent.futures.as_completed(future_map):
                    if max_references > 0 and len(references) >= max_references:
                        break
                    if time_budget_s > 0 and (time.monotonic() - started) >= time_budget_s:
                        budget_exhausted = True
                        break
                    if consecutive_timeouts >= resolved_timeout_threshold:
                        break
                    url = future_map[future]
                    try:
                        fetch_result = future.result()
                    except Exception as exc:
                        if _is_timeout_error(str(exc)):
                            consecutive_timeouts += 1
                        memory.record(
                            stage="step2_doc_retrieval",
                            key=url,
                            summary="MCP fetch call failed; skipping this document.",
                            issues=[str(exc)],
                            metadata={"phase": "fetch", "query": query, "timeout_like": _is_timeout_error(str(exc))},
                        )
                        continue
                    if not fetch_result.get("ok") and _is_timeout_error(str(fetch_result.get("error") or "")):
                        consecutive_timeouts += 1
                    elif fetch_result.get("ok"):
                        consecutive_timeouts = 0
                    if not fetch_result.get("ok"):
                        continue

                    content = str(fetch_result.get("data", {}).get("content", "")).strip()
                    if not content:
                        continue

                    seen_urls.add(url)
                    summary = _summarize_document(content)
                    rules = _extract_rule_summaries(content)
                    references.append(
                        DocumentationReference(
                            url=url,
                            title=_derive_title(url, raw_result_by_url.get(url, {}), content),
                            summary=summary,
                            rules=rules,
                            matched_query=query,
                        )
                    )
                    memory.record(
                        stage="step2_doc_retrieval",
                        key=query,
                        summary=(
                            f"Retrieved doc evidence for {parsed.component_type}: "
                            f"{_derive_title(url, raw_result_by_url.get(url, {}), content)}"
                        ),
                        evidence_urls=[url],
                        proposed_fixes=[rule.rule for rule in rules],
                        metadata={
                            "query": query,
                            "objective_metric": parsed.objective.metric,
                            "objective_goal": parsed.objective.goal,
                        },
                    )

        # Preserve compatibility with previous control-flow semantics.
        if budget_exhausted:
            break
        if max_references > 0 and len(references) >= max_references:
            break

    if budget_exhausted:
        memory.record(
            stage="step2_doc_retrieval",
            key=parsed.component_type,
            summary="Step2 doc retrieval stopped after reaching time budget.",
            metadata={
                "time_budget_s": time_budget_s,
                "queries_considered": len(queries),
                "references_collected": len(references),
            },
        )

    if not references:
        synthetic_reference = DocumentationReference(
            url="memory://step2/degraded/no_mcp_docs",
            title="Step2 degraded-mode fallback guidance",
            summary=(
                "No external MCP docs were retrieved. "
                "Use conservative inverse-design defaults: PML boundaries, mode source, "
                "field monitor and flux monitor, then enforce Step4 hard validation."
            ),
            rules=[
                DocumentationRuleSummary(
                    rule="Use PML boundaries on all simulation sides.",
                    evidence="degraded_mode_fallback",
                ),
                DocumentationRuleSummary(
                    rule="Use a mode source for guided-wave excitation.",
                    evidence="degraded_mode_fallback",
                ),
                DocumentationRuleSummary(
                    rule="Keep both field and flux monitors enabled.",
                    evidence="degraded_mode_fallback",
                ),
            ],
            matched_query=queries[0] if queries else "",
        )
        references = [synthetic_reference]
        memory.record(
            stage="step2_doc_retrieval",
            key=parsed.component_type,
            summary="Step2 MCP retrieval produced no references; injected degraded-mode fallback reference.",
            metadata={
                "queries": queries,
                "degraded_mode": True,
                "skip_reason": "no_references_collected",
            },
        )

    memory.record(
        stage="step2_doc_retrieval",
        key=parsed.component_type,
        summary=(
            f"Step2 completed with {len(references)} references and {len(queries)} queries "
            f"for {parsed.component_type}."
        ),
        evidence_urls=[ref.url for ref in references],
        metadata={
            "queries": queries,
            "wavelength_nm": parsed.wavelength_nm,
        },
    )

    return InverseDesignDocContext(
        requirement=parsed,
        queries=queries,
        references=references,
        guidance=_derive_guidance(references),
        structure_context=_derive_structure_context(parsed, references, _derive_guidance(references)),
    )


def inverse_design_doc_context_schema() -> Dict[str, Any]:
    """Return JSON schema for the doc-retrieval intermediate result."""

    return InverseDesignDocContext.model_json_schema()


def build_doc_queries(requirement: InverseDesignRequirement) -> List[str]:
    """Generate focused Tidy3D doc search queries from requirement intent.

    Uses both canonical enum values (for precise matching) and LLM raw
    semantic text (for broader recall when the canonical value is generic).
    """

    component_label = requirement.component_type or "photonic device"
    objective_metric = requirement.objective.metric or "optimization"
    objective_goal = requirement.objective.goal or "design"

    queries = [
        f"Tidy3D inverse design {component_label}",
        f"Tidy3D {component_label} {objective_goal} {objective_metric}",
    ]

    if requirement.wavelength_nm is not None:
        queries.append(f"Tidy3D {component_label} {requirement.wavelength_nm:g} nm")
    for target in requirement.routing_targets:
        queries.append(
            f"Tidy3D {component_label} {target.wavelength_nm:g} nm {target.target_port} demux"
        )

    # Demux-specific retrieval anchors: force coverage of official multi-case
    # references so Step2 evidence can guide Step3/Step4 toward wavelength-split
    # configuration rather than generic single-case templates.
    if len(requirement.routing_targets) >= 2:
        queries.extend(
            [
                "Tidy3D AdjointPlugin9WDM notebook",
                "Tidy3D Autograd9WDM notebook",
                "Tidy3D MultiplexingMMI notebook",
                f"Tidy3D {component_label} multi wavelength demultiplexer",
                f"Tidy3D {component_label} mode monitor flux monitor port_o2 port_o3",
            ]
        )

    for constraint in requirement.constraints:
        queries.append(f"Tidy3D {component_label} {constraint.name.replace('_', ' ')}")

    # Enrich with LLM raw semantic text when it differs from canonical values.
    # This gives MCP broader search recall for novel/uncommon component types
    # or metric descriptions that the canonical enum doesn't cover.
    llm_comp_raw = (requirement.llm_component_type_raw or "").strip()
    if llm_comp_raw and llm_comp_raw.lower() != component_label.lower():
        queries.append(f"Tidy3D inverse design {llm_comp_raw}")

    llm_metric_raw = (requirement.objective.llm_metric_raw or "").strip()
    if llm_metric_raw and llm_metric_raw.lower() != objective_metric.lower():
        queries.append(f"Tidy3D {component_label} {llm_metric_raw}")

    deduped: List[str] = []
    seen: set[str] = set()
    for query in queries:
        normalized = query.strip()
        if normalized and normalized not in seen:
            deduped.append(normalized)
            seen.add(normalized)
    return deduped


def _apply_llm_query_plan(
    requirement: InverseDesignRequirement,
    base_queries: List[str],
    *,
    use_llm_planner: bool | None,
    llm_call_fn: LLMCallFn | None,
    llm_model: str,
    memory_store: InverseDesignWorkingMemory,
) -> List[str]:
    should_use = _planner_enabled(use_llm_planner)
    if not should_use:
        return base_queries

    caller = llm_call_fn or _default_llm_call
    if caller is None:
        return base_queries

    prompt = (
        "Generate a retrieval query plan for Tidy3D inverse design docs. "
        "Output strict JSON: {\"queries\": [string], \"rationale\": string}.\n"
        f"Requirement JSON: {requirement.model_dump_json()}\n"
        f"Baseline queries: {json.dumps(base_queries)}"
    )
    sys_prompt = (
        "You are a photonics simulation planning assistant. "
        "Return only JSON. Keep queries short and evidence-oriented for MCP search."
    )

    try:
        response = caller(prompt, sys_prompt, llm_model)
        payload = _extract_json_object(str(response))
        plan = DocRetrievalPlan.model_validate(payload)
    except Exception as exc:
        memory_store.record(
            stage="step2_doc_retrieval",
            key=requirement.component_type,
            summary="LLM query planner failed; fallback to deterministic query generation.",
            issues=[str(exc)],
            metadata={"fallback": "deterministic_queries"},
        )
        return base_queries

    planned = [query.strip() for query in plan.queries if query.strip()]
    merged = _dedupe_queries(planned + base_queries)

    memory_store.record(
        stage="step2_doc_retrieval",
        key=requirement.component_type,
        summary="LLM planner produced Step2 retrieval queries.",
        metadata={
            "rationale": plan.rationale,
            "planned_queries": planned,
            "merged_queries": merged,
        },
    )
    return merged


def _planner_enabled(use_llm_planner: bool | None) -> bool:
    if use_llm_planner is not None:
        return use_llm_planner
    value = os.getenv("INVERSE_DESIGN_ENABLE_LLM_PLANNER", "0").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _default_llm_call(prompt: str, sys_prompt: str, llm_model: str) -> str:
    from PhotonicsAI.Photon.llm_api import call_llm

    return call_llm(prompt, sys_prompt, llm_api_selection=llm_model)


def _extract_json_object(text: str) -> Dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if len(lines) >= 3:
            candidate = "\n".join(lines[1:-1])
    match = re.search(r"\{[\s\S]*\}", candidate)
    if match:
        candidate = match.group(0)
    return json.loads(candidate)


def _dedupe_queries(queries: List[str]) -> List[str]:
    deduped: List[str] = []
    seen: set[str] = set()
    for query in queries:
        normalized = query.strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        deduped.append(normalized)
        seen.add(key)
    return deduped


def _coerce_requirement(requirement: InverseDesignRequirement | str) -> InverseDesignRequirement:
    if isinstance(requirement, InverseDesignRequirement):
        if not requirement.is_complete:
            missing = ", ".join(requirement.missing_critical_fields)
            raise ValueError(f"Requirement is incomplete: {missing}")
        return requirement
    return require_complete_inverse_design_requirement(requirement)


def _derive_title(url: str, raw_result: Dict[str, Any], content: str) -> str:
    result_title = str(raw_result.get("title", "")).strip()
    if result_title:
        return result_title

    first_line = content.splitlines()[0].strip() if content.splitlines() else ""
    if first_line:
        return first_line[:120]

    tail = url.rstrip("/").split("/")[-1].replace("-", " ").replace("_", " ").strip()
    return tail.title()


def _summarize_document(content: str, max_sentences: int = 2) -> str:
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", content)
        if sentence.strip()
    ]
    return " ".join(sentences[:max_sentences])[:400]


def _extract_rule_summaries(content: str, max_rules: int = 4) -> List[DocumentationRuleSummary]:
    rules: List[DocumentationRuleSummary] = []
    seen: set[str] = set()

    for pattern in _RULE_PATTERNS:
        match = pattern.search(content)
        if not match:
            continue

        sentence = " ".join(match.group("sentence").split())
        normalized = sentence.lower()
        if normalized in seen:
            continue

        rules.append(
            DocumentationRuleSummary(
                rule=_compress_rule(sentence),
                evidence=sentence[:240],
            )
        )
        seen.add(normalized)

        if len(rules) >= max_rules:
            break

    if rules:
        return rules

    fallback = _summarize_document(content, max_sentences=1)
    if not fallback:
        return []
    return [DocumentationRuleSummary(rule=_compress_rule(fallback), evidence=fallback[:240])]


def _compress_rule(sentence: str) -> str:
    compact = " ".join(sentence.split()).strip()
    if len(compact) <= 140:
        return compact
    return compact[:137].rstrip() + "..."



_STRUCTURE_BUILDER_SPECS: Dict[str, Dict[str, Any]] = {
    "mmi": {
        "topology": "multimode_interference_region_with_input_output_waveguides",
        "builder_function": "create_mmi",
        "parameter_keys": ["wg_width", "wg_height", "mmi_width", "mmi_length", "mmi_num_outputs"],
        "ports": [("port_o1", "input", "+"), ("port_o2", "output", "+"), ("port_o3", "output", "+")],
    },
    "splitter": {
        "topology": "multimode_interference_splitter_with_input_output_waveguides",
        "builder_function": "create_mmi",
        "parameter_keys": ["wg_width", "wg_height", "mmi_width", "mmi_length", "mmi_num_outputs"],
        "ports": [("port_o1", "input", "+"), ("port_o2", "output", "+"), ("port_o3", "output", "+")],
    },
    "directional_coupler": {
        "topology": "two_parallel_waveguides_with_coupling_gap",
        "builder_function": "create_coupler",
        "parameter_keys": ["wg_width", "wg_height", "coupler_length", "gap"],
        "ports": [("port_o1", "input", "+"), ("port_o2", "output", "+"), ("port_o3", "output", "-"), ("port_o4", "output", "-")],
    },
    "ring_resonator": {
        "topology": "bus_waveguide_coupled_to_ring_resonator",
        "builder_function": "create_ring_resonator",
        "parameter_keys": ["wg_width", "wg_height", "ring_radius", "gap"],
        "ports": [("port_o1", "input", "+"), ("port_o2", "through", "+"), ("port_o3", "drop", "+")],
    },
    "y_branch": {
        "topology": "one_input_two_output_s_bend_branch",
        "builder_function": "create_y_branch",
        "parameter_keys": ["wg_width", "wg_height", "arm_length", "arm_separation"],
        "ports": [("port_o1", "input", "+"), ("port_o2", "output", "+"), ("port_o3", "output", "+")],
    },
    "mzi": {
        "topology": "splitter_two_arms_combiner_interferometer",
        "builder_function": "create_mzi",
        "parameter_keys": ["wg_width", "wg_height", "arm_length", "arm_separation"],
        "ports": [("port_o1", "input", "+"), ("port_o2", "output", "+")],
    },
    "grating_coupler": {
        "topology": "waveguide_to_periodic_grating_coupler",
        "builder_function": "create_grating_coupler",
        "parameter_keys": ["wg_width", "wg_height", "grating_period", "num_periods"],
        "ports": [("port_o1", "input", "+"), ("port_o2", "output", "+")],
    },
    "polarization_rotator": {
        "topology": "input_waveguide_swg_rotation_section_output_waveguide",
        "builder_function": "create_polarization_rotator",
        "parameter_keys": ["wg_width", "wg_height", "rotation_length", "swg_period"],
        "ports": [("port_o1", "input", "+"), ("port_o2", "output", "+")],
    },
    "crossing": {
        "topology": "orthogonal_waveguide_crossing",
        "builder_function": "create_waveguide_crossing",
        "parameter_keys": ["wg_width", "wg_height", "wg_length"],
        "ports": [("port_o1", "input", "+"), ("port_o2", "output", "+"), ("port_o3", "cross", "+"), ("port_o4", "cross", "-")],
    },
    "waveguide": {
        "topology": "straight_single_mode_waveguide",
        "builder_function": "create_simple_waveguide",
        "parameter_keys": ["wg_width", "wg_height", "wg_length"],
        "ports": [("port_o1", "input", "+"), ("port_o2", "output", "+")],
    },
}


def _derive_structure_context(
    requirement: InverseDesignRequirement,
    references: List[DocumentationReference],
    guidance: RetrievalGuidance,
) -> StructureContext:
    component = requirement.component_type or "waveguide"
    spec = _STRUCTURE_BUILDER_SPECS.get(component, _STRUCTURE_BUILDER_SPECS["waveguide"])
    builder_function = str(spec.get("builder_function", ""))
    builder_reference = f"PhotonicsAI.Photon.tidy3d_runner.{builder_function}" if builder_function else ""
    monitor_types = list(guidance.recommended_monitors or ["field", "flux", "mode"])
    if "mode" not in monitor_types:
        monitor_types.append("mode")
    ports = [
        StructurePortSpec(
            name=str(name),
            role=str(role),
            direction=str(direction),
            expected_monitor_types=monitor_types,
        )
        for name, role, direction in spec.get("ports", [])
    ]
    evidence_urls = [ref.url for ref in references if ref.url][:6]
    return StructureContext(
        component_type=component,
        topology=str(spec.get("topology", component)),
        builder_function=builder_function,
        builder_reference=builder_reference,
        parameter_keys=[str(item) for item in spec.get("parameter_keys", [])],
        ports=ports,
        source_strategy=f"{guidance.source_type or 'mode'}_source_at_input_port",
        monitor_strategy="field_flux_mode_monitors_at_builder_ports",
        reusable_code_refs=[
            builder_reference,
            "PhotonicsAI.tools.inverse_design_config_generation.generate_inverse_design_config",
            "PhotonicsAI.tools.inverse_design_execution._build_invdes_simulation",
        ],
        evidence_urls=evidence_urls,
        design_library_used=False,
        rationale=(
            "Step2 resolved the requested component to the current Tidy3D builder contract. "
            "This provides structure/config code guidance without using DesignLibrary retrieval."
        ),
    )

def _derive_guidance(references: List[DocumentationReference]) -> RetrievalGuidance:
    combined = " ".join(
        f"{ref.summary} {' '.join(rule.rule for rule in ref.rules)}"
        for ref in references
    ).lower()

    source_type = "mode"
    if "plane wave" in combined:
        source_type = "plane_wave"
    elif "gaussian" in combined:
        source_type = "gaussian"

    recommended_monitors: List[str] = []
    if "field monitor" in combined or "fieldmonitor" in combined:
        recommended_monitors.append("field")
    if "flux monitor" in combined or "fluxmonitor" in combined or "flux" in combined:
        recommended_monitors.append("flux")
    if "mode monitor" in combined or "modemonitor" in combined or "mode overlap" in combined:
        recommended_monitors.append("mode")
    if not recommended_monitors:
        recommended_monitors = ["field", "flux"]

    mesh_advice = ""
    if "mesh override" in combined:
        mesh_advice = "mesh_override"
    elif "fine mesh" in combined:
        mesh_advice = "fine_mesh"
    elif "auto grid" in combined or "gridspec.auto" in combined:
        mesh_advice = "auto_grid"

    inverse_design_hint = ""
    if "adjoint" in combined:
        inverse_design_hint = "adjoint"
    elif "inverse design" in combined:
        inverse_design_hint = "inverse_design"

    require_pml = "pml" in combined or "absorbing boundary" in combined

    return RetrievalGuidance(
        source_type=source_type,
        require_pml=require_pml,
        recommended_monitors=recommended_monitors,
        mesh_advice=mesh_advice,
        inverse_design_hint=inverse_design_hint,
    )


