"""Structured stage context builders and prompt compaction helpers for reproagent."""

import json
import re
from typing import Any

from reproagent.pipeline.schemas import (
    ArchitectureOutput,
    BoundaryRequirementsOutput,
    PaperBenchReproState,
    GlobalContractOutput,
    PackageFilePlanningOutput,
    PipelinePlanOutput,
    ReferenceSelectionOutput,
    TopicProfileOutput,
    WorkPackagePlanningOutput,
)

from ..config import semantic_anchor_disabled
from .dataset_manager import _get_dataset_preparation, _get_resource_manifest
from .evidence_contracts import flatten_evidence_contract, infer_evidence_contract
from .intent_contract import paperbench_prompt_safe_experiment_design, upstream_intent_payload
from .ref_repo_survey import _get_reference_repo_preparation, _get_reference_repo_surveys
from .run_context import _json_default


def _normalize_repo_path(path: str) -> str:
    normalized = str(path or "").strip().replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    while "//" in normalized:
        normalized = normalized.replace("//", "/")
    return normalized.strip("/")


def _dedupe_nonempty(values: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = str(raw or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


_POSITIVE_ACTION_TOKENS = (
    "implement",
    "expose",
    "create",
    "write",
    "support",
    "maintain",
    "compute",
    "save",
    "bind",
    "route",
    "实现",
    "暴露",
    "创建",
    "写出",
    "支持",
    "保存",
    "连接",
    "计算",
    "命名",
    "调度",
    "适配",
    "维护",
)

_PROMPT_INVENTORY_PREFIXES = (
    "obligation_matrix",
    "implementation_scope_inventory",
    "environment_inventory",
    "dataset_inventory",
    "method_inventory",
    "baseline_inventory",
    "measurement_inventory",
    "metric_inventory",
    "parameter_inventory",
    "fixed_hyperparameter_inventory",
    "result_trend_inventory",
    "result_artifact_inventory",
    "artifact_inventory",
    "implementation_obligation_inventory",
    "implementation_surface_inventory",
    "closure_inventory",
    "experiment_inventory",
    "model_inventory",
)


def _prompt_safe_text(value: Any) -> str:
    """Normalize prompt-facing text without rewriting its semantics."""
    return " ".join(str(value or "").split()).strip()


def _prompt_inventory_key(text: str) -> str:
    prefix = str(text or "").split(":", 1)[0].strip().lower()
    return prefix


def _is_prompt_inventory_line(text: str) -> bool:
    return _prompt_inventory_key(text) in _PROMPT_INVENTORY_PREFIXES


def _prompt_inventory_map(payload: Any, *, limit: int = 24) -> dict[str, list[str]]:
    inventories: dict[str, list[str]] = {}
    for key, values in dict(payload or {}).items():
        inventory_key = str(key or "").strip()
        if inventory_key not in _PROMPT_INVENTORY_PREFIXES:
            continue
        items = _prompt_safe_positive_items(values, limit=limit)
        if items:
            inventories[inventory_key] = items
    return inventories


def _prompt_positive_fragments(value: Any) -> list[str]:
    """Select implementation-bearing fragments by positive action/inventory shape.

    Planning prompts receive a compact implementation target: action phrases,
    concrete inventories, and artifact/route ownership fields.
    """
    text = _prompt_safe_text(value)
    if not text:
        return []
    fragments = [
        _prompt_safe_text(fragment)
        for fragment in re.split(r"[\n;；。]+", text)
        if _prompt_safe_text(fragment)
    ]
    selected: list[str] = []
    for fragment in fragments:
        lowered = fragment.lower()
        starts_with_action = any(lowered.startswith(token.lower()) for token in _POSITIVE_ACTION_TOKENS)
        if _is_prompt_inventory_line(fragment) or starts_with_action:
            selected.append(fragment)
    return _dedupe_nonempty(selected)


def _prompt_safe_positive_items(values: Any, *, limit: int = 8) -> list[str]:
    items: list[str] = []
    for raw in list(values or []):
        text = _prompt_safe_text(raw)
        if not text:
            continue
        items.append(text)
    return _dedupe_nonempty(items)[:limit]


def _prompt_safe_source_ids(values: Any, *, limit: int = 8) -> list[str]:
    """Keep active paper/addendum provenance IDs without planning-policy anchors."""
    allowed_prefixes = (
        "target:",
        "addendum:",
        "contract:",
        "chunk_",
        "paper:",
        "section:",
        "paragraph:",
        "table:",
        "figure:",
        "fig:",
        "asset:",
    )
    items: list[str] = []
    for raw in list(values or []):
        text = _prompt_safe_text(raw)
        lowered = text.lower()
        if text and lowered.startswith(allowed_prefixes):
            items.append(text)
    return _dedupe_nonempty(items)[:limit]


def _prompt_positive_obligation_items(values: Any, *, limit: int = 8) -> list[str]:
    items: list[str] = []
    for raw in list(values or []):
        items.extend(_prompt_positive_fragments(raw))
    return _dedupe_nonempty(items)[:limit]


def _prompt_safe_inventory_notes(values: Any, *, limit: int = 8) -> list[str]:
    items: list[str] = []
    for raw in list(values or []):
        text = _prompt_safe_text(raw)
        if not text:
            continue
        if _is_prompt_inventory_line(text):
            items.append(text)
    return _dedupe_nonempty(items)[:limit]


def _prompt_safe_statement(unit_id: str, statement: Any, obligations: list[str], surfaces: list[str], artifacts: list[str]) -> str:
    del statement
    focus = ", ".join(_dedupe_nonempty(obligations[:2] + surfaces[:4] + artifacts[:2]))
    return f"Implement {unit_id} active reproduction surfaces: {focus}" if focus else f"Implement {unit_id} active reproduction surfaces."


def _unit_focus_text(unit_id: str, surfaces: list[str], obligations: list[str], artifacts: list[str]) -> str:
    focus = ", ".join(_dedupe_nonempty(surfaces[:4] + obligations[:2] + artifacts[:2]))
    return focus or unit_id or "active reproduction route"


def _unit_positive_hypothesis(unit_id: str, surfaces: list[str], obligations: list[str], artifacts: list[str]) -> str:
    focus = _unit_focus_text(unit_id, surfaces, obligations, artifacts)
    return f"Implementing {unit_id or 'this unit'} covers the paper-derived active route for {focus}."


def _unit_positive_decision_value(unit_id: str, surfaces: list[str], obligations: list[str], artifacts: list[str]) -> str:
    focus = _unit_focus_text(unit_id, surfaces, obligations, artifacts)
    return f"Confirms executable reproduction coverage for {focus}."


def _planning_source_policy_payload(state: PaperBenchReproState) -> dict[str, Any]:
    """Positive source/provenance policy for prompts."""
    design = _experiment_design_payload(state)
    paperbench = design.get("paperbench") if isinstance(design.get("paperbench"), dict) else {}
    return {
        "input_documents": ["paper.md", "addendum.md when present", "assets when present"],
        "reference_policy": "Prepared local reference repositories provide implementation-pattern provenance.",
        "implementation_scope": "Plan source-derived methods, datasets, metrics, protocols, artifacts, and runnable routes.",
    }


def _prompt_safe_paper_context(paper_context: dict[str, Any]) -> dict[str, Any]:
    """Return a planning-facing paper context focused on source identity."""
    safe = dict(paper_context or {})
    paperbench_input = dict(safe.get("paperbench_input", {}) or {})
    safe.pop("paperbench_addendum", None)
    safe["paperbench_input"] = {
        "title": str(paperbench_input.get("title", "") or ""),
        "assets": list(paperbench_input.get("assets", []) or [])[:80],
        "source_policy": "paper.md/addendum/assets plus prepared local references with provenance.",
    }
    return safe


def _experiment_design_for_planning(state: PaperBenchReproState) -> dict[str, Any]:
    design = _experiment_design_payload(state)
    paperbench = design.get("paperbench") if isinstance(design.get("paperbench"), dict) else {}
    return {
        "task_family": str(design.get("task_family", "") or ""),
        "code_only": bool(design.get("code_only", False)),
        "download_policy": str(design.get("download_policy", "") or ""),
        "paperbench_title": str(design.get("paperbench_title", "") or paperbench.get("title", "") or ""),
        "expected_artifacts": _prompt_safe_positive_items(design.get("expected_artifacts", []), limit=12),
        "execution_model": {
            "bounded_validation": "Provide lightweight validation routes that exercise the same implementation owners.",
            "full_reproduction": "Expose researcher-facing commands, configs, loaders, metrics, and artifact writers for complete runs.",
        },
        "paperbench": {
            "title": str(paperbench.get("title", "") or ""),
            "assets": list(paperbench.get("assets", []) or [])[:80],
            "source_policy": _planning_source_policy_payload(state),
        },
    }


def _experiment_design_payload_for_planning(state: PaperBenchReproState) -> dict[str, Any]:
    return _experiment_design_for_planning(state)


def _target_brief_for_planning(state: PaperBenchReproState) -> str:
    """Render a planning-facing objective from structured positive context."""
    design = _experiment_design_for_planning(state)
    title = str(design.get("paperbench_title", "") or dict(design.get("paperbench", {}) or {}).get("title", "") or "").strip()
    clauses = [
        (
            f"Build a faithful, complete, judgeable code reproduction repository for {title}."
            if title
            else "Build a faithful, complete, judgeable code reproduction repository."
        )
    ]
    unit_contract = _unit_implementation_contract_projection(state, max_units=40)
    surfaces = list(dict(unit_contract).get("surface_counts", {}).keys())[:10]
    artifacts = list(dict(unit_contract).get("artifact_inventory", []) or [])[:8]
    method_terms: list[str] = []
    dataset_terms: list[str] = []
    metric_terms: list[str] = []
    for contract in list(dict(unit_contract).get("contracts", []) or [])[:12]:
        if not isinstance(contract, dict):
            continue
        for note in list(contract.get("implementation_notes", []) or []):
            text = str(note or "")
            lowered = text.lower()
            if lowered.startswith(("method_inventory:", "baseline_inventory:")):
                method_terms.extend(part.strip(" .。") for part in re.split(r"[,;，；]", text.split(":", 1)[-1]) if part.strip())
            elif lowered.startswith(("dataset_inventory:", "environment_inventory:")):
                dataset_terms.extend(part.strip(" .。") for part in re.split(r"[,;，；]", text.split(":", 1)[-1]) if part.strip())
            elif lowered.startswith(("measurement_inventory:", "metric_inventory:")):
                metric_terms.extend(part.strip(" .。") for part in re.split(r"[,;，；]", text.split(":", 1)[-1]) if part.strip())
    if dataset_terms:
        clauses.append("Active datasets/tasks: " + ", ".join(_dedupe_nonempty(dataset_terms)[:10]) + ".")
    if method_terms:
        clauses.append("Active methods/baselines: " + ", ".join(_dedupe_nonempty(method_terms)[:12]) + ".")
    if metric_terms:
        clauses.append("Active measurements: " + ", ".join(_dedupe_nonempty(metric_terms)[:12]) + ".")
    if surfaces:
        clauses.append("Implementation surfaces: " + ", ".join(surfaces) + ".")
    if artifacts:
        clauses.append("Expected artifacts: " + ", ".join(artifacts) + ".")
    return " ".join(clauses)


def _truncate_text(value: Any, max_chars: int) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n...<truncated>..."


def _unit_payload_for_planning(item: Any) -> dict[str, Any]:
    """Return unit context with positive implementation fields only."""
    payload = item.model_dump(mode="json") if hasattr(item, "model_dump") else dict(item)
    unit_id = str(payload.get("unit_id", "") or "").strip()
    obligations = _prompt_positive_obligation_items(payload.get("code_obligations", []), limit=10)
    surfaces = _prompt_safe_positive_items(payload.get("implementation_surfaces", []), limit=8)
    artifacts = _prompt_safe_positive_items(payload.get("expected_artifacts", []), limit=8)
    return {
        "unit_id": unit_id,
        "type": str(payload.get("type", "") or ""),
        "statement": _prompt_safe_statement(unit_id, payload.get("statement", ""), obligations, surfaces, artifacts),
        "hypothesis": _unit_positive_hypothesis(unit_id, surfaces, obligations, artifacts),
        "decision_value": _unit_positive_decision_value(unit_id, surfaces, obligations, artifacts),
        "source_paragraph_ids": _prompt_safe_source_ids(payload.get("source_paragraph_ids", []), limit=8),
        "verification_targets": [
            {
                "kind": str(item.get("kind", "") or ""),
                "description": _prompt_safe_text(item.get("description", "")),
            }
            for item in list(payload.get("verification_targets", []) or [])
            if isinstance(item, dict) and _prompt_positive_fragments(item.get("description", ""))
        ][:4],
        "implementation_surfaces": surfaces,
        "code_obligations": obligations,
        "runtime_interfaces": _prompt_safe_positive_items(payload.get("runtime_interfaces", []), limit=6),
        "expected_artifacts": artifacts,
        "suggested_module_kinds": _prompt_safe_positive_items(payload.get("suggested_module_kinds", []), limit=6),
        "implementation_notes": _prompt_safe_inventory_notes(payload.get("implementation_notes", []), limit=8),
        "status": str(payload.get("status", "active") or "active"),
    }


def _unit_payloads_for_planning(state: PaperBenchReproState) -> list[dict[str, Any]]:
    if not state.unit_extraction:
        return []
    return [_unit_payload_for_planning(item) for item in state.unit_extraction.units]


def _boundary_requirements_for_planning(boundary_output: BoundaryRequirementsOutput | None) -> dict[str, Any]:
    """Project positive paper-derived requirements into planning prompts."""
    if boundary_output is None:
        return {}
    requirements: list[dict[str, Any]] = []
    for item in list(boundary_output.boundary_requirements or []):
        requirement_id = str(item.requirement_id or "").strip()
        if not requirement_id:
            continue
        title = str(item.title or "").strip()
        description = str(item.description or "").strip()
        criteria = _prompt_safe_positive_items(item.acceptance_criteria, limit=8)
        requirements.append(
            {
                "requirement_id": requirement_id,
                "title": title,
                "category": str(item.category or "experiment").strip(),
                "scope": str(item.scope or "").strip(),
                "description": description,
                "source_unit_ids": _prompt_safe_positive_items(item.source_unit_ids, limit=16),
                "acceptance_criteria": criteria,
                "implementation_contract": " ".join(
                    part for part in [title, description, "; ".join(criteria)] if part
                )[:1200],
            }
        )
    return {
        "boundary_requirements": requirements,
        "requirement_scope_items": [
            str(item.get("title") or item.get("description") or item.get("requirement_id") or "").strip()
            for item in requirements
            if str(item.get("title") or item.get("description") or item.get("requirement_id") or "").strip()
        ],
    }


def _work_package_planning_for_prompts(state: PaperBenchReproState | None) -> dict[str, Any]:
    """Return generation/planning-facing work packages without planning-only rationale fields."""
    if state is None or not state.work_package_planning:
        return {}
    return _work_package_planning_output_for_prompts(state.work_package_planning)


def _work_package_planning_output_for_prompts(
    work_package_planning: WorkPackagePlanningOutput | None,
) -> dict[str, Any]:
    """Return generation/planning-facing work packages without planning-only rationale fields."""
    if work_package_planning is None:
        return {}
    work_packages: list[dict[str, Any]] = []
    for package in list(work_package_planning.work_packages or []):
        work_package_id = str(package.work_package_id or "").strip()
        if not work_package_id:
            continue
        inventories = _prompt_inventory_map(package.inventories, limit=24)
        work_packages.append(
            {
                "work_package_id": work_package_id,
                "goal": _prompt_safe_text(package.goal),
                "hypothesis": _unit_positive_hypothesis(
                    work_package_id,
                    _prompt_safe_positive_items(package.tags, limit=8),
                    _prompt_safe_positive_items(package.method_obligations, limit=8),
                    _prompt_safe_positive_items(package.produces, limit=8),
                ),
                "decision_value": _unit_positive_decision_value(
                    work_package_id,
                    _prompt_safe_positive_items(package.tags, limit=8),
                    _prompt_safe_positive_items(package.method_obligations, limit=8),
                    _prompt_safe_positive_items(package.produces, limit=8),
                ),
                "owned_unit_ids": _prompt_safe_positive_items(package.owned_unit_ids, limit=80),
                "tags": _prompt_safe_positive_items(package.tags, limit=16),
                "reference_ids": _prompt_safe_positive_items(package.reference_ids, limit=24),
                "depends_on": _prompt_safe_positive_items(package.depends_on, limit=24),
                "produces": [_normalize_repo_path(path) for path in _prompt_safe_positive_items(package.produces, limit=24)],
                "interface_contract": _prompt_safe_positive_items(package.interface_contract, limit=24),
                "evidence_needs": [
                    f"Cover source unit {unit_id}"
                    for unit_id in _prompt_safe_positive_items(package.owned_unit_ids, limit=24)
                ],
                "inventories": inventories,
                "scope_boundary": {
                    "preserve": _prompt_safe_positive_items(
                        dict(package.scope_boundary or {}).get("preserve", []),
                        limit=24,
                    ),
                    "implementation_focus": _prompt_safe_positive_items(
                        dict(package.scope_boundary or {}).get("implementation_focus", []),
                        limit=24,
                    ),
                },
                "method_obligations": _prompt_safe_positive_items(package.method_obligations, limit=32),
            }
        )
    return {
        "work_packages": work_packages,
        "coverage_summary": work_package_planning.coverage_summary.model_dump(mode="json"),
        "planning_notes": [
            "Work packages are projected as positive implementation ownership contracts."
        ],
    }


def _pipeline_plan_for_prompts(pipeline_plan: PipelinePlanOutput | None) -> dict[str, Any]:
    """Return pipeline plan nodes with implementation-facing text fields only."""
    if pipeline_plan is None:
        return {}
    plan_nodes: list[dict[str, Any]] = []
    for node in list(pipeline_plan.plan_nodes or []):
        node_id = str(node.node_id or "").strip()
        name = _prompt_safe_text(node.name)
        focus = name or node_id
        plan_nodes.append(
            {
                "node_id": node_id,
                "parent_node_id": str(node.parent_node_id or "").strip(),
                "name": name,
                "level": str(node.level or "").strip(),
                "description": _prompt_safe_text(node.description) or f"Implement active plan node {focus}.",
                "hypothesis": f"The {node_id or 'plan node'} route covers the paper-derived implementation contract for {focus}.",
                "decision_value": f"Confirms runnable, reviewable coverage for {focus}.",
                "requirement_ids": _prompt_safe_positive_items(node.requirement_ids, limit=24),
                "ref_id": str(node.ref_id or "").strip(),
                "reusable_module": str(node.reusable_module or "").strip(),
                "depends_on": _prompt_safe_positive_items(node.depends_on, limit=24),
                "traceable": bool(node.traceable),
                "insight": _prompt_safe_text(node.insight),
            }
        )
    return {
        "plan_nodes": plan_nodes,
        "coverage_summary": pipeline_plan.coverage_summary.model_dump(mode="json"),
    }


def _architecture_for_prompts(
    architecture: ArchitectureOutput | None,
    *,
    target_file_tree: list[str] | None = None,
    file_blueprints: list[Any] | None = None,
    dependency_graph: list[Any] | None = None,
    package_layout: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Return architecture context with executable structure and positive routes only."""
    if architecture is None:
        return {}
    selected_file_tree = [
        _normalize_repo_path(path)
        for path in list(target_file_tree if target_file_tree is not None else architecture.target_file_tree)
        if _normalize_repo_path(path)
    ]
    selected_blueprints = []
    for item in list(file_blueprints if file_blueprints is not None else architecture.file_blueprints):
        payload = item.model_dump(mode="json") if hasattr(item, "model_dump") else dict(item)
        path = _normalize_repo_path(str(payload.get("path", "") or ""))
        if not path:
            continue
        selected_blueprints.append(
            {
                "path": path,
                "purpose": _prompt_safe_text(payload.get("purpose", "")),
                "kind": str(payload.get("kind", "source") or "source"),
                "related_node_ids": _prompt_safe_positive_items(payload.get("related_node_ids", []), limit=24),
                "based_on_references": _prompt_safe_positive_items(payload.get("based_on_references", []), limit=24),
                "implementation_strategy": str(payload.get("implementation_strategy", "new") or "new"),
            }
        )
    selected_dependencies = []
    for item in list(dependency_graph if dependency_graph is not None else architecture.dependency_graph):
        payload = item.model_dump(mode="json") if hasattr(item, "model_dump") else dict(item)
        source_path = _normalize_repo_path(str(payload.get("source_path", "") or ""))
        target_path = _normalize_repo_path(str(payload.get("target_path", "") or ""))
        if source_path and target_path:
            selected_dependencies.append(
                {
                    "source_path": source_path,
                    "target_path": target_path,
                    "dependency_type": str(payload.get("dependency_type", "imports") or "imports"),
                }
            )
    selected_layout = {
        str(key): [
            _normalize_repo_path(path)
            for path in list(values or [])
            if _normalize_repo_path(path)
        ]
        for key, values in dict(package_layout if package_layout is not None else architecture.package_layout).items()
        if str(key).strip()
    }
    dependency_rules = [
        f"{item['source_path']} may use {item['target_path']} via {item['dependency_type']}."
        for item in selected_dependencies[:40]
    ]
    if not dependency_rules:
        dependency_rules = ["Files use package-local stable interfaces and explicit entrypoint-to-artifact routes."]
    return {
        "target_stack": _prompt_safe_positive_items(architecture.target_stack, limit=12),
        "target_file_tree": selected_file_tree,
        "file_blueprints": selected_blueprints,
        "dependency_graph": selected_dependencies,
        "stable_interfaces": [
            _normalize_repo_path(path)
            for path in _prompt_safe_positive_items(architecture.stable_interfaces, limit=24)
            if _normalize_repo_path(path)
        ],
        "execution_entrypoints": [
            _normalize_repo_path(path)
            for path in _prompt_safe_positive_items(architecture.execution_entrypoints, limit=12)
            if _normalize_repo_path(path)
        ],
        "config_surfaces": [
            _normalize_repo_path(path)
            for path in _prompt_safe_positive_items(architecture.config_surfaces, limit=24)
            if _normalize_repo_path(path)
        ],
        "package_layout": selected_layout,
        "dependency_rules": dependency_rules,
        "protocol_stages": _prompt_safe_positive_items(architecture.protocol_stages, limit=24),
        "result_targets": [
            _normalize_repo_path(path)
            for path in _prompt_safe_positive_items(architecture.result_targets, limit=24)
            if _normalize_repo_path(path)
        ],
        "architecture_reference_ids": _prompt_safe_positive_items(architecture.architecture_reference_ids, limit=24),
    }


def _normalized_input_for_planning(state: PaperBenchReproState) -> dict[str, Any]:
    if not state.normalized_input:
        return {}
    payload = state.normalized_input.model_dump(mode="json")
    return {
        "task_type": str(payload.get("task_type", "") or ""),
        "expected_outputs": _prompt_safe_positive_items(payload.get("expected_outputs", []), limit=12),
    }


def _compact_reference_repo_payload(item: Any) -> dict[str, Any]:
    payload = item.model_dump(mode="json") if hasattr(item, "model_dump") else dict(item)
    compact_symbol_evidence = []
    for evidence in list(payload.get("symbol_evidence", []) or [])[:6]:
        if not isinstance(evidence, dict):
            continue
        compact_symbol_evidence.append(
            {
                "evidence_id": str(evidence.get("evidence_id", "") or ""),
                "file_path": str(evidence.get("file_path", "") or ""),
                "symbol_name": str(evidence.get("symbol_name", "") or ""),
                "symbol_kind": str(evidence.get("symbol_kind", "") or ""),
                "matched_unit_ids": list(evidence.get("matched_unit_ids", []) or []),
                "matched_requirement_ids": list(evidence.get("matched_requirement_ids", []) or []),
                "matched_surfaces": list(evidence.get("matched_surfaces", []) or []),
                "matched_keywords": list(evidence.get("matched_keywords", []) or [])[:8],
                "relevance_reason": _truncate_text(_prompt_safe_text(evidence.get("relevance_reason", "")), 220),
                "snippet_summary": "Reference-backed code context is available in this file; use provenance markers when adapting patterns.",
                "score": evidence.get("score", 0.0),
            }
        )
    compact_requirement_coverage = []
    for coverage in list(payload.get("requirement_coverage", []) or [])[:8]:
        if not isinstance(coverage, dict):
            continue
        compact_requirement_coverage.append(
            {
                "requirement_id": str(coverage.get("requirement_id", "") or ""),
                "title": str(coverage.get("title", "") or ""),
                "scope": str(coverage.get("scope", "") or ""),
                "source_unit_ids": list(coverage.get("source_unit_ids", []) or []),
                "matched_keywords": list(coverage.get("matched_keywords", []) or [])[:8],
                "matched_files": list(coverage.get("matched_files", []) or [])[:6],
                "match_locations": list(coverage.get("match_locations", []) or [])[:4],
                "code_snippet_count": len(list(coverage.get("code_snippets", []) or [])),
            }
        )
    return {
        **payload,
        "readme_summary": _truncate_text(_prompt_safe_text(payload.get("readme_summary", "")), 900),
        "file_tree_summary": _truncate_text(_prompt_safe_text(payload.get("file_tree_summary", "")), 700),
        "protocol_clues": _prompt_safe_positive_items(payload.get("protocol_clues", []), limit=10),
        "top_python_files": list(payload.get("top_python_files", []) or [])[:10],
        "likely_reusable_files": list(payload.get("likely_reusable_files", []) or [])[:10],
        "symbol_evidence": compact_symbol_evidence,
        "requirement_coverage": compact_requirement_coverage,
    }


def _resource_manifest_for_planning(state: PaperBenchReproState) -> dict[str, Any]:
    manifest = dict(_get_resource_manifest(state) or {})
    ref_repos: list[dict[str, Any]] = []
    for item in list(manifest.get("ref_repos", []) or []):
        if not isinstance(item, dict):
            continue
        survey = dict(item.get("survey", {}) or {})
        ref_repos.append(
            {
                "ref_id": str(item.get("ref_id", "") or ""),
                "title": str(item.get("title", "") or ""),
                "status": str(item.get("status", "") or ""),
                "local_path": str(item.get("local_path", "") or ""),
                "survey": _compact_reference_repo_payload(survey) if survey else {},
            }
        )
    return {
        "schema_version": str(manifest.get("schema_version", "") or ""),
        "prepare_status": str(manifest.get("prepare_status", "") or ""),
        "datasets": list(manifest.get("datasets", []) or []),
        "benchmarks": list(manifest.get("benchmarks", []) or []),
        "baselines": list(manifest.get("baselines", []) or []),
        "ref_repos": ref_repos,
    }


def _paper_chunk_dicts(state: PaperBenchReproState) -> list[dict[str, Any]]:
    if state.paper_chunks:
        return [item.model_dump(mode="json") for item in state.paper_chunks]
    chunks = state.temp_data.get("paper_chunks")
    if isinstance(chunks, list):
        return [dict(item) for item in chunks if isinstance(item, dict)]
    design = _experiment_design_payload(state)
    chunks = design.get("paper_chunks")
    if isinstance(chunks, list):
        return [dict(item) for item in chunks if isinstance(item, dict)]
    return []


def _select_prompt_paper_chunks(
    chunks: list[dict[str, Any]],
    *,
    max_chunks: int = 14,
    preview_chars: int = 1600,
    include_text: bool = False,
) -> list[dict[str, Any]]:
    """Select a compact, paper-faithful chunk set for stage prompts."""
    if not chunks:
        return []
    priority_terms = (
        "abstract",
        "introduction",
        "method",
        "approach",
        "algorithm",
        "model",
        "experiment",
        "implementation",
        "appendix",
        "limitations",
    )
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(item: dict[str, Any]) -> None:
        chunk_id = str(item.get("chunk_id", "") or "")
        if not chunk_id or chunk_id in seen or len(selected) >= max_chunks:
            return
        payload = {
            "chunk_id": chunk_id,
            "section_title": str(item.get("section_title", "") or ""),
            "ordinal": item.get("ordinal", 0),
            "source_path": str(item.get("source_path", "") or ""),
            "token_estimate": item.get("token_estimate", 0),
        }
        if include_text:
            payload["text"] = _truncate_text(_prompt_safe_text(item.get("text", "")), preview_chars)
        selected.append(payload)
        seen.add(chunk_id)

    for item in chunks[:3]:
        add(item)
    for item in chunks:
        title = str(item.get("section_title", "") or "").lower()
        if any(term in title for term in priority_terms):
            add(item)
        if len(selected) >= max_chunks:
            break
    for item in chunks:
        add(item)
        if len(selected) >= max_chunks:
            break
    return selected


def _paper_context_payload(state: PaperBenchReproState, *, include_chunk_text: bool = False) -> dict[str, Any]:
    """Compact paper/PaperBench context shared by planning prompts."""
    design = _experiment_design_payload(state)
    raw_design = state.input.experiment_design if isinstance(state.input.experiment_design, dict) else {}
    raw_paperbench = raw_design.get("paperbench") if isinstance(raw_design.get("paperbench"), dict) else {}
    paperbench = design.get("paperbench") if isinstance(design.get("paperbench"), dict) else {}
    chunks = _paper_chunk_dicts(state)
    paperbench_input = {
        "title": str(paperbench.get("title", "") or design.get("paperbench_title", "") or ""),
        "assets": list(paperbench.get("assets", []) or [])[:80],
    }
    payload = {
        "paper_path": str(getattr(state.input, "paper_path", "") or ""),
        "paper_chunk_count": len(chunks),
        "paper_chunk_index": list(design.get("paper_chunk_index", []) or [])[:80],
        "selected_paper_chunks": _select_prompt_paper_chunks(chunks, include_text=include_chunk_text),
        "paperbench_input": paperbench_input,
    }
    if include_chunk_text:
        payload["paperbench_addendum"] = _truncate_text(raw_paperbench.get("addendum_text", ""), 6000)
    return payload


def _paper_context_payload_for_planning(state: PaperBenchReproState, *, include_chunk_text: bool = False) -> dict[str, Any]:
    payload = _prompt_safe_paper_context(_paper_context_payload(state, include_chunk_text=include_chunk_text))
    payload["source_policy"] = _planning_source_policy_payload(state)
    return payload


def _contract_item_names(contract: dict[str, Any], category: str) -> list[str]:
    names: list[str] = []
    for item in list(contract.get(category, []) or []):
        if isinstance(item, dict):
            name = str(item.get("name", "") or "").strip()
            if category == "parameter_sweeps":
                values = [
                    str(value).strip()
                    for value in list(item.get("values", []) or [])
                    if str(value).strip()
                ]
                names.append(name + (f"[{','.join(values[:12])}]" if values else ""))
            else:
                names.append(name)
        else:
            names.append(str(item or "").strip())
    return _dedupe_nonempty([item for item in names if item])


def _paper_source_text_for_contract(state: PaperBenchReproState) -> str:
    design = _experiment_design_payload(state)
    paperbench = design.get("paperbench") if isinstance(design.get("paperbench"), dict) else {}
    text_parts = [
        str(getattr(state.input, "paper_text", "") or ""),
        str(paperbench.get("addendum_text", "") or ""),
    ]
    for chunk in _paper_chunk_dicts(state):
        text_parts.append(str(chunk.get("text", "") or ""))
    return "\n".join(part for part in text_parts if part.strip())


_FORMULA_ALGORITHM_SECTION_TERMS = (
    "method",
    "methods",
    "approach",
    "algorithm",
    "model",
    "adapter",
    "pruning",
    "tuning",
    "training",
    "optimization",
    "objective",
    "loss",
    "metric",
    "evaluation",
    "implementation",
    "方法",
    "算法",
    "模型",
    "剪枝",
    "调参",
    "训练",
    "优化",
    "目标",
    "损失",
    "指标",
)

_FORMULA_ALGORITHM_TEXT_TERMS = (
    "equation",
    "eq.",
    "algorithm",
    "formula",
    "objective",
    "loss",
    "gradient",
    "salience",
    "mask",
    "rank",
    "binary search",
    "schedule",
    "moving average",
    "ema",
    "kurt",
    "kurtosis",
    "teacher",
    "student",
    "distill",
    "rouge",
    "accuracy",
    "throughput",
    "memory",
    "公式",
    "算法",
    "梯度",
    "显著性",
    "掩码",
    "秩",
    "二分",
    "调度",
    "蒸馏",
)

_FORMULA_ALGORITHM_STRONG_TERMS = (
    "equation",
    "eq.",
    "algorithm",
    "formula",
    "objective",
    "loss",
    "gradient",
    "salience",
    "mask",
    "rank",
    "binary search",
    "moving average",
    "ema",
    "kurt",
    "kurtosis",
    "teacher",
    "student",
    "distill",
    "公式",
    "算法",
    "梯度",
    "显著性",
    "掩码",
    "二分",
    "蒸馏",
)

_FORMULA_ALGORITHM_PROCEDURE_TERMS = (
    "compute",
    "calculate",
    "update",
    "search",
    "sort",
    "select",
    "sample",
    "merge",
    "prune",
    "mask",
    "schedule",
    "initialize",
    "concatenate",
    "increase",
    "decrease",
    "linearly",
    "re-compute",
    "recompute",
    "计算",
    "更新",
    "搜索",
    "排序",
    "采样",
    "合并",
    "剪枝",
    "掩码",
    "调度",
)

_FORMULA_ALGORITHM_PRIORITY_SECTION_TERMS = (
    "method",
    "methodology",
    "approach",
    "algorithm",
    "model",
    "architecture",
    "formulation",
    "objective",
    "loss",
    "optimization",
    "training",
    "implementation",
    "adapter",
    "pruning",
    "tuning",
    "distillation",
    "方法",
    "算法",
    "模型",
    "架构",
    "公式",
    "优化",
    "训练",
    "实现",
)

_FORMULA_ALGORITHM_NUMERIC_SECTION_TERMS = (
    "formulation",
    "objective",
    "algorithm",
    "adapter",
    "pruning",
    "tuning",
    "distillation",
    "implementation",
    "baseline",
    "setup",
    "hyperparameter",
    "appendix",
    "公式",
    "算法",
    "剪枝",
    "调参",
    "蒸馏",
    "实现",
    "超参数",
)

_FORMULA_ALGORITHM_LOW_PRIORITY_SECTION_TERMS = (
    "front_matter",
    "abstract",
    "introduction",
    "background",
    "related",
    "experiment",
    "experiments",
    "results",
    "analysis",
    "discussion",
    "conclusion",
    "appendix",
    "实验",
    "结果",
    "分析",
    "结论",
)

_FORMULA_ALGORITHM_EXCLUDED_SECTION_TERMS = (
    "references",
    "reference",
    "bibliography",
)

_FORMULA_SYMBOL_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "if",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "this",
    "to",
    "we",
    "with",
    "where",
}

_LATEX_GREEK_SYMBOLS = {
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "lambda",
    "mu",
    "phi",
    "sigma",
    "tau",
    "theta",
    "Delta",
    "Theta",
}

_FORMULA_SYMBOL_PATTERN = re.compile(
    r"""
    \\(?:mathcal|mathbf|mathrm|mathbb|operatorname|text)\s*\{[^{}]{1,80}\}
        (?:\s*_\s*(?:\\(?:mathcal|mathbf|mathrm|mathbb|operatorname|text)\s*\{[^{}]{1,80}\}|\{[^{}]{1,80}\}|[A-Za-z0-9()+\-]+))+
        (?:\s*\^\s*(?:\{[^{}]{1,80}\}|[A-Za-z0-9()+\-]+))*
    |
    \\(?:mathcal|mathbf|mathrm|mathbb|operatorname|text)\s*\{[^{}]{1,80}\}
        (?:\s*[_^]\s*(?:\{[^{}]{1,80}\}|[A-Za-z0-9()+\-]+))*
    |\\(?:overline|hat|bar|tilde)\s*\{[^{}]{1,80}\}
        (?:\s*[_^]\s*(?:\{[^{}]{1,80}\}|[A-Za-z0-9()+\-]+))*
    |\\(?:alpha|beta|gamma|delta|epsilon|lambda|mu|phi|sigma|tau|theta|Delta|Theta)\b
        (?:\s*[_^]\s*(?:\{[^{}]{1,80}\}|[A-Za-z0-9()+\-]+))*
    |[A-Za-z][A-Za-z0-9]*
        (?:\s*[_^]\s*(?:\{[^{}]{1,80}\}|[A-Za-z0-9()+\-]+))+
    |[A-Z][A-Za-z0-9]{1,20}\s*\([^)]{1,80}\)
    """,
    re.X,
)

_CODE_ANCHOR_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])(?:[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]+|[A-Za-z_][A-Za-z0-9_.]*\(\))(?![A-Za-z0-9_])"
)


def _formula_like_spans(text: str) -> list[str]:
    rendered = str(text or "")
    spans: list[str] = []
    for pattern in (
        r"\$\$([^$]{1,1800})\$\$",
        r"\$([^$]{1,900})\$",
        r"\\\[([\s\S]{1,2200}?)\\\]",
        r"\\\(([\s\S]{1,900}?)\\\)",
        r"\\begin\{[^{}]{1,40}\}([\s\S]{1,2200}?)\\end\{[^{}]{1,40}\}",
    ):
        spans.extend(match.strip() for match in re.findall(pattern, rendered) if str(match).strip())
    return spans


def _has_formula_signal(text: str) -> bool:
    rendered = str(text or "")
    if _formula_like_spans(rendered):
        return True
    return bool(
        re.search(r"[A-Za-z\\][A-Za-z0-9\\]*(?:\s*[_^]\s*(?:\{[^{}]{1,80}\}|[A-Za-z0-9()+\-]+))", rendered)
        or re.search(r"(?:\\gets|←|->|→|≤|>=|\\leq|\\geq|\\sum|\\prod|\\frac|\\partial)", rendered)
        or re.search(r"[A-Za-z0-9_)\]}]\s*=\s*[-+A-Za-z0-9_\\{(\[]", rendered)
    )


def _has_equation_like_formula(text: str) -> bool:
    rendered = str(text or "")
    return bool(
        re.search(r"(?:\\gets|←|->|→|≤|>=|\\leq|\\geq|\\sum|\\prod|\\frac|\\partial)", rendered)
        or re.search(r"[A-Za-z0-9_)\]}]\s*=\s*[-+A-Za-z0-9_\\{(\[]", rendered)
        or re.search(r"\\begin\{(?:array|aligned|equation|split|cases)\}", rendered)
    )


def _section_allows_formula_algorithm_contract(source_id: str, section_title: str, sentences: list[str]) -> bool:
    if str(source_id or "").startswith("addendum:"):
        return True
    lowered = str(section_title or "").lower()
    if any(term in lowered for term in _FORMULA_ALGORITHM_EXCLUDED_SECTION_TERMS):
        return False
    if any(term in lowered for term in _FORMULA_ALGORITHM_PRIORITY_SECTION_TERMS):
        return True
    if any(term in lowered for term in _FORMULA_ALGORITHM_LOW_PRIORITY_SECTION_TERMS):
        return any(_has_equation_like_formula(sentence) for sentence in sentences)
    return any(
        _has_equation_like_formula(sentence)
        or any(term in sentence.lower() for term in _FORMULA_ALGORITHM_STRONG_TERMS)
        for sentence in sentences
    )


def _normalize_formula_symbol(raw: str) -> str:
    text = str(raw or "").strip().strip("$`.,;:，。")
    if not text:
        return ""
    text = re.sub(r"\\(?:left|right)\b", "", text)
    text = re.sub(r"\\overline\s*\{\s*([^{}]{1,80})\s*\}", r"\1_bar", text)
    text = re.sub(r"\\hat\s*\{\s*([^{}]{1,80})\s*\}", r"\1_hat", text)
    text = re.sub(r"\\bar\s*\{\s*([^{}]{1,80})\s*\}", r"\1_bar", text)
    text = re.sub(r"\\tilde\s*\{\s*([^{}]{1,80})\s*\}", r"\1_tilde", text)
    for command in ("mathcal", "mathbf", "mathrm", "mathbb", "operatorname", "text"):
        text = re.sub(rf"\\{command}\s*\{{\s*([^{{}}]{{1,80}})\s*\}}", r"\1", text)
    text = re.sub(
        r"\\([A-Za-z]+)",
        lambda match: match.group(1) if match.group(1) in _LATEX_GREEK_SYMBOLS else match.group(1),
        text,
    )
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"_\{([^{}]{1,80})\}", lambda match: "_" + match.group(1).replace(" ", ""), text)
    text = re.sub(r"\^\{([^{}]{1,80})\}", lambda match: "^" + match.group(1).replace(" ", ""), text)
    text = re.sub(r"\^\(([^()]{1,40})\)", r"^\1", text)
    text = text.replace("(", "").replace(")", "")
    text = text.replace("{", "").replace("}", "").replace("\\", "")
    text = text.strip("()[],:;.")
    if text.endswith("()"):
        text = text[:-2]
    lowered = text.lower()
    if not text or len(text) > 96 or lowered in _FORMULA_SYMBOL_STOPWORDS:
        return ""
    if lowered.isdigit():
        return ""
    has_math_shape = bool(
        "_" in text
        or "^" in text
        or any(name.lower() == lowered for name in _LATEX_GREEK_SYMBOLS)
        or re.fullmatch(r"[A-Z]{2,}[A-Za-z0-9_]*", text)
        or re.fullmatch(r"[A-Z][A-Za-z0-9]{0,20}\([^)]{1,80}\)", str(raw or "").strip())
    )
    return text if has_math_shape else ""


def _extract_formula_symbols(sentences: list[str]) -> list[str]:
    symbols: list[str] = []
    for sentence in sentences:
        spans = _formula_like_spans(sentence)
        if _has_formula_signal(sentence):
            spans.append(sentence)
        for span in spans:
            for base, subscript in re.findall(
                r"([A-Za-z][A-Za-z0-9]*)\s*_\s*\{\s*\\(?:mathcal|mathbf|mathrm|mathbb|operatorname|text)\s*\{\s*([^{}]{1,80})\s*\}\s*\}",
                span,
            ):
                symbol = _normalize_formula_symbol(f"{base}_{subscript}")
                if symbol:
                    symbols.append(symbol)
            for base, subscript in re.findall(
                r"\\(?:mathcal|mathbf|mathrm|mathbb)\s*\{\s*([^{}]{1,40})\s*\}\s*_\s*\{\s*\\text\s*\{\s*([^{}]{1,80})\s*\}\s*\}",
                span,
            ):
                symbol = _normalize_formula_symbol(f"{base}_{subscript}")
                if symbol:
                    symbols.append(symbol)
            for match in _FORMULA_SYMBOL_PATTERN.findall(span):
                symbol = _normalize_formula_symbol(match)
                if symbol:
                    symbols.append(symbol)
        for match in _CODE_ANCHOR_PATTERN.findall(sentence):
            symbol = _normalize_formula_symbol(match)
            if symbol:
                symbols.append(symbol)
    return _dedupe_nonempty(symbols)


def _extract_numeric_values(sentences: list[str], *, section_title: str = "") -> list[str]:
    values: list[str] = []
    section_lower = str(section_title or "").lower()
    allow_measurement_numbers = any(term in section_lower for term in _FORMULA_ALGORITHM_NUMERIC_SECTION_TERMS)
    for sentence in sentences:
        if "\\begin{tabular}" in sentence or "\\hline" in sentence:
            continue
        has_formula = _has_formula_signal(sentence)
        lowered = sentence.lower()
        has_algorithm = any(term in lowered for term in _FORMULA_ALGORITHM_STRONG_TERMS)
        if not has_formula and not has_algorithm:
            continue
        for match in re.findall(r"(?<![A-Za-z0-9])(?:\d+\.\d+|\d+)(?:e[-+]?\d+)?%?(?![A-Za-z0-9])", sentence):
            if re.fullmatch(r"(?:19|20)\d{2}", match):
                continue
            if not allow_measurement_numbers and match.endswith("%"):
                continue
            if not allow_measurement_numbers and not re.search(r"[=<>]|\\gets|←|\\leq|\\geq|linearly|set to|initialized|default", sentence, flags=re.I):
                continue
            values.append(match)
    return _dedupe_nonempty(values)


def _algorithm_terms_for_sentences(sentences: list[str]) -> list[str]:
    lowered = "\n".join(sentences).lower()
    return _dedupe_nonempty(
        [
            term
            for term in list(_FORMULA_ALGORITHM_STRONG_TERMS) + list(_FORMULA_ALGORITHM_PROCEDURE_TERMS)
            if term in lowered
        ]
    )


def _paper_formula_algorithm_contract(state: PaperBenchReproState) -> dict[str, Any]:
    """Extract paper-derived formula and algorithm anchors for planning.

    The pipeline cannot depend on rubric content during generation, so this
    contract is intentionally derived only from paper/addendum/chunk text.  It
    keeps equations, named symbols, numeric defaults, and algorithmic verbs
    visible as implementation obligations instead of letting them collapse into
    broad method/interface labels.
    """
    chunks = _paper_chunk_dicts(state)
    design = _experiment_design_payload(state)
    paperbench = design.get("paperbench") if isinstance(design.get("paperbench"), dict) else {}
    addendum_text = str(paperbench.get("addendum_text", "") or "")
    anchor_rows: list[dict[str, Any]] = []
    symbol_inventory: list[str] = []
    numeric_inventory: list[str] = []
    obligation_inventory: list[str] = []
    source_texts: list[dict[str, str]] = []
    if addendum_text.strip():
        source_texts.append(
            {
                "source_id": "addendum:formula_algorithm_contract",
                "section_title": "addendum",
                "text": addendum_text,
            }
        )
    for chunk in chunks:
        source_texts.append(
            {
                "source_id": str(chunk.get("chunk_id", "") or chunk.get("source_path", "") or ""),
                "section_title": str(chunk.get("section_title", "") or ""),
                "text": str(chunk.get("text", "") or ""),
            }
        )

    def selected_sentences(text: str) -> list[str]:
        normalized = _prompt_safe_text(text)
        if not normalized:
            return []
        parts = [
            item.strip()
            for item in re.split(r"(?<=[.!?。；;])\s+|\n+", normalized)
            if item.strip()
        ]
        selected: list[str] = []
        for part in parts:
            if "\\begin{tabular}" in part or "\\hline" in part:
                continue
            lowered = part.lower()
            has_math = _has_formula_signal(part)
            has_equation = _has_equation_like_formula(part)
            has_strong_term = any(term in lowered for term in _FORMULA_ALGORITHM_STRONG_TERMS)
            has_procedure = any(term in lowered for term in _FORMULA_ALGORITHM_PROCEDURE_TERMS)
            if has_equation or (has_math and has_strong_term) or (has_strong_term and has_procedure):
                selected.append(part)
        return selected[:10]

    for source in source_texts:
        text = source["text"]
        if not text.strip():
            continue
        section_title = source["section_title"]
        lowered_blob = f"{section_title}\n{text}".lower()
        if not any(term in lowered_blob for term in _FORMULA_ALGORITHM_SECTION_TERMS + _FORMULA_ALGORITHM_TEXT_TERMS):
            continue
        sentences = selected_sentences(text)
        if not sentences:
            continue
        if not _section_allows_formula_algorithm_contract(source["source_id"], section_title, sentences):
            continue
        symbols = _extract_formula_symbols(sentences)[:28]
        numeric_values = _extract_numeric_values(sentences, section_title=section_title)[:20]
        algorithm_steps = _dedupe_nonempty(
            [
                sentence
                for sentence in sentences
                if any(
                    term in sentence.lower()
                    for term in (
                        "compute",
                        "update",
                        "search",
                        "sort",
                        "sample",
                        "merge",
                        "prune",
                        "mask",
                        "schedule",
                        "loss",
                        "gradient",
                        "计算",
                        "更新",
                        "搜索",
                        "排序",
                        "采样",
                        "合并",
                        "剪枝",
                        "掩码",
                    )
                )
            ]
        )[:8]
        algorithm_terms = _algorithm_terms_for_sentences(sentences)[:24]
        row = {
            "source_id": source["source_id"],
            "section_title": section_title,
            "formula_or_algorithm_excerpts": [_truncate_text(item, 520) for item in sentences[:6]],
            "required_symbols": symbols,
            "required_numeric_values": numeric_values,
            "algorithm_terms": algorithm_terms,
            "algorithm_steps": [_truncate_text(item, 360) for item in algorithm_steps],
            "implementation_obligation": _truncate_text(
                "Implement the paper-derived formulas, symbols, numeric constants/defaults, and algorithm steps above as executable code/config reached by the canonical route.",
                260,
            ),
        }
        if not any((symbols, numeric_values, algorithm_steps, algorithm_terms)):
            continue
        anchor_rows.append(row)
        symbol_inventory.extend(symbols)
        numeric_inventory.extend(numeric_values)
        obligation_inventory.extend(
            [
                f"{symbol} must be represented in executable code/config"
                for symbol in symbols[:12]
            ]
            + [
                f"numeric/formula value {value} must be represented in executable code/config"
                for value in numeric_values[:8]
            ]
            + [
                f"algorithm step: {_truncate_text(step, 220)}"
                for step in algorithm_steps[:6]
            ]
            + [
                f"algorithm term `{term}` must be represented in executable code/config"
                for term in algorithm_terms[:8]
            ]
        )
    anchor_rows = anchor_rows[:40]
    return {
        "source": "paper_and_addendum_formula_algorithm_extraction",
        "anchor_count": len(anchor_rows),
        "anchors": anchor_rows,
        "required_symbol_inventory": _dedupe_nonempty(symbol_inventory)[:160],
        "required_numeric_inventory": _dedupe_nonempty(numeric_inventory)[:120],
        "implementation_obligations": _dedupe_nonempty(obligation_inventory)[:240],
        "closure_rule": (
            "Formula/algorithm anchors are mandatory implementation obligations: "
            "they must appear as executable/importable functions, classes, constants, "
            "config defaults, or canonical-route calls, not only README/provenance text."
        ),
    }


def _paper_evidence_contract_payload(state: PaperBenchReproState) -> dict[str, Any]:
    """Build the paper-derived claim inventory every planning stage must preserve.

    The contract is derived from paper/addendum text or from the prepare gate's
    cached unit-quality payload.
    """
    gate = dict(state.temp_data.get("prepare_quality_gate", {}) or {})
    unit_quality = dict(gate.get("unit_quality", {}) or {})
    contract = dict(unit_quality.get("evidence_contract", {}) or {})
    source = "prepare_quality_gate.unit_quality"
    if not contract:
        source = "paper_text_inference"
        contract = flatten_evidence_contract(infer_evidence_contract(_paper_source_text_for_contract(state)))

    categories = (
        "named_experiments",
        "environments",
        "datasets",
        "methods",
        "metrics",
        "artifacts",
        "parameter_sweeps",
        "trend_obligations",
        "protocol_obligations",
        "fixed_hyperparameters",
        "implementation_obligations",
    )
    required_claim_inventory = {
        category: _contract_item_names(contract, category)
        for category in categories
    }
    closure_items = [
        {"category": category, "name": name}
        for category, names in required_claim_inventory.items()
        for name in names
    ]
    claim_coverage = dict(unit_quality.get("claim_inventory_coverage", {}) or {})
    formula_algorithm_contract = _paper_formula_algorithm_contract(state)
    return {
        "source": source,
        "contract": contract,
        "required_claim_inventory": required_claim_inventory,
        "formula_algorithm_contract": formula_algorithm_contract,
        "closure_items": closure_items[:240],
        "closure_item_count": len(closure_items),
        "prepare_gate_summary": {
            "schema_version": str(gate.get("schema_version", "") or ""),
            "passed": bool(gate.get("passed", False)),
            "active_unit_count": int(gate.get("active_unit_count", 0) or 0),
            "claim_coverage_ratio": claim_coverage.get("coverage_ratio"),
            "artifact_coverage_ratio": claim_coverage.get("artifact_coverage_ratio"),
        },
        "closure_rules": [
            "Every required_claim_inventory item must remain visible by exact name or clear alias from unit -> work package -> plan node -> global contract -> file plan -> active code/artifact route.",
            "Every formula_algorithm_contract anchor must remain visible from unit -> work package -> file plan -> executable code/config route.",
            "Method, dataset, metric, training, baseline, and table/figure obligations need active code owners plus reachable routes; support registries and documents are supplementary.",
            "If a paper item is intentionally bounded or represented by config, keep the exact item, owner package, code surface, artifact target, and bounded-execution rationale visible.",
        ],
    }


def _paper_evidence_contract_payload_for_generation(state: PaperBenchReproState) -> dict[str, Any]:
    """Return the generation-facing paper evidence contract for task inputs."""
    if semantic_anchor_disabled():
        return {}
    return _paper_evidence_contract_payload(state)


def _experiment_design_payload(state: PaperBenchReproState) -> dict[str, Any]:
    return paperbench_prompt_safe_experiment_design(
        state.input.experiment_design if isinstance(state.input.experiment_design, dict) else {}
    )


def _unit_implementation_contract_projection(
    state: PaperBenchReproState,
    *,
    unit_ids: set[str] | None = None,
    max_units: int = 260,
) -> dict[str, Any]:
    """Compact unit-derived code obligations so prompt truncation keeps implementation hints."""
    units = list(state.unit_extraction.units if state.unit_extraction else [])
    selected_ids = {str(item).strip() for item in unit_ids or set() if str(item).strip()}
    contracts: list[dict[str, Any]] = []
    surface_counts: dict[str, int] = {}
    module_kind_counts: dict[str, int] = {}
    artifact_inventory: list[str] = []

    for unit in units:
        unit_id = str(unit.unit_id or "").strip()
        if selected_ids and unit_id not in selected_ids:
            continue
        surfaces = _prompt_safe_positive_items(getattr(unit, "implementation_surfaces", []), limit=8)
        module_kinds = _prompt_safe_positive_items(getattr(unit, "suggested_module_kinds", []), limit=8)
        artifacts = _prompt_safe_positive_items(getattr(unit, "expected_artifacts", []), limit=8)
        obligations = _prompt_positive_obligation_items(getattr(unit, "code_obligations", []), limit=8)
        notes = _prompt_safe_inventory_notes(getattr(unit, "implementation_notes", []), limit=8)
        interfaces = _prompt_safe_positive_items(getattr(unit, "runtime_interfaces", []), limit=6)
        if not any([surfaces, module_kinds, artifacts, obligations, interfaces]):
            continue
        for surface in surfaces:
            surface_counts[surface] = surface_counts.get(surface, 0) + 1
        for module_kind in module_kinds:
            module_kind_counts[module_kind] = module_kind_counts.get(module_kind, 0) + 1
        artifact_inventory = _dedupe_nonempty(artifact_inventory + artifacts)
        contracts.append(
            {
                "unit_id": unit_id,
                "type": str(unit.type or ""),
                "statement": _truncate_text(
                    _prompt_safe_statement(unit_id, getattr(unit, "statement", ""), obligations, surfaces, artifacts),
                    220,
                ),
                "implementation_surfaces": surfaces[:6],
                "suggested_module_kinds": module_kinds[:6],
                "runtime_interfaces": interfaces[:4],
                "expected_artifacts": artifacts[:4],
                "code_obligations": obligations[:3],
                "implementation_notes": notes[:3],
                "hypothesis": _truncate_text(_unit_positive_hypothesis(unit_id, surfaces, obligations, artifacts), 180),
                "decision_value": _truncate_text(_unit_positive_decision_value(unit_id, surfaces, obligations, artifacts), 180),
            }
        )

    return {
        "unit_count": len(units),
        "selected_unit_count": len(contracts),
        "truncated": len(contracts) > max_units,
        "surface_counts": dict(sorted(surface_counts.items(), key=lambda item: (-item[1], item[0]))),
        "module_kind_counts": dict(sorted(module_kind_counts.items(), key=lambda item: (-item[1], item[0]))),
        "artifact_inventory": artifact_inventory[:80],
        "contracts": contracts[:max_units],
    }


def _path_from_file_plan(item: Any) -> str:
    if isinstance(item, dict):
        return _normalize_repo_path(str(item.get("target_file") or item.get("path") or ""))
    return _normalize_repo_path(str(getattr(item, "target_file", "") or getattr(item, "path", "") or ""))


def _file_plan_values(item: Any, field_name: str) -> list[str]:
    value = item.get(field_name) if isinstance(item, dict) else getattr(item, field_name, None)
    if isinstance(value, (list, tuple, set)):
        return _dedupe_nonempty([str(part) for part in value if str(part).strip()])
    if str(value or "").strip():
        return [str(value).strip()]
    return []


def _file_plan_text(item: Any) -> str:
    fields = (
        "target_file",
        "path",
        "purpose",
        "generation_prompt",
        "implementation_surfaces",
        "method_obligations",
        "interface_contract",
        "review_points",
        "defines_symbols",
        "calls_symbols",
        "writes_artifacts",
    )
    values: list[str] = []
    for field_name in fields:
        values.extend(_file_plan_values(item, field_name))
    return " ".join(values).lower()


def _is_support_route_path(path: str) -> bool:
    normalized = _normalize_repo_path(path).lower()
    basename = normalized.rsplit("/", 1)[-1]
    return (
        not normalized
        or basename in {"__init__.py", "config.py", "configs.py", "constants.py", "registry.py", "schema.py", "schemas.py", "settings.py", "types.py"}
        or normalized.startswith(("configs/", "config/", "tests/"))
        or normalized.endswith((".yaml", ".yml", ".toml", ".json", ".md", ".txt"))
    )


def _is_active_route_path(path: str) -> bool:
    normalized = _normalize_repo_path(path).lower()
    basename = normalized.rsplit("/", 1)[-1]
    if basename in {"main.py", "cli.py", "run.py", "run_experiments.py"}:
        return True
    return any(
        token in normalized
        for token in (
            "experiment",
            "evaluation",
            "evaluate",
            "training",
            "train",
            "runner",
            "report",
            "artifact",
            "plot",
            "figure",
        )
    )


def _work_package_unit_owner_map(state: PaperBenchReproState) -> dict[str, list[str]]:
    owners: dict[str, list[str]] = {}
    for package in list(state.work_package_planning.work_packages if state.work_package_planning else []):
        package_id = str(getattr(package, "work_package_id", "") or "").strip()
        if not package_id:
            continue
        for unit_id in list(getattr(package, "owned_unit_ids", []) or []):
            key = str(unit_id or "").strip()
            if not key:
                continue
            owners.setdefault(key, [])
            if package_id not in owners[key]:
                owners[key].append(package_id)
    return owners


def _unit_route_matrix_projection(
    state: PaperBenchReproState,
    *,
    architecture: ArchitectureOutput | None = None,
    file_plans: PackageFilePlanningOutput | None = None,
    unit_ids: set[str] | None = None,
    max_units: int = 260,
) -> dict[str, Any]:
    """Project implementation units into explicit owner/path/route obligations.

    This is derived from paper/addendum units and plan artifacts.
    """
    selected_ids = {str(item).strip() for item in unit_ids or set() if str(item).strip()}
    units = [
        unit
        for unit in list(state.unit_extraction.units if state.unit_extraction else [])
        if not selected_ids or str(getattr(unit, "unit_id", "") or "").strip() in selected_ids
    ]
    unit_owner_map = _work_package_unit_owner_map(state)
    package_layout = dict(getattr(architecture, "package_layout", {}) or {})
    execution_entrypoints = _dedupe_nonempty(
        [_normalize_repo_path(path) for path in list(getattr(architecture, "execution_entrypoints", []) or [])]
    )
    file_plan_items = list(getattr(file_plans, "file_plans", []) or [])
    rows: list[dict[str, Any]] = []
    support_only_units: list[str] = []
    missing_owner_units: list[str] = []
    missing_active_route_units: list[str] = []

    for unit in units[:max_units]:
        unit_id = str(getattr(unit, "unit_id", "") or "").strip()
        if not unit_id:
            continue
        surfaces = _prompt_safe_positive_items(getattr(unit, "implementation_surfaces", []), limit=8)
        obligations = _prompt_positive_obligation_items(getattr(unit, "code_obligations", []), limit=8)
        artifacts = _prompt_safe_positive_items(getattr(unit, "expected_artifacts", []), limit=8)
        owner_work_packages = unit_owner_map.get(unit_id, [])
        owned_paths = _dedupe_nonempty(
            [
                _normalize_repo_path(path)
                for owner in owner_work_packages
                for path in list(package_layout.get(owner, []) or [])
                if _normalize_repo_path(path)
            ]
        )
        owner_file_plans = [
            item
            for item in file_plan_items
            if (
                unit_id in _file_plan_values(item, "owned_units")
                or unit_id in _file_plan_values(item, "owned_unit_ids")
                or str((item.get("work_package_id") if isinstance(item, dict) else getattr(item, "work_package_id", "")) or "").strip() in owner_work_packages
            )
        ]
        owner_file_paths = _dedupe_nonempty([_path_from_file_plan(item) for item in owner_file_plans] + owned_paths)
        active_owner_paths = _dedupe_nonempty(
            [
                path
                for path in owner_file_paths
                if path.endswith(".py") and not _is_support_route_path(path)
            ]
        )
        owner_symbols = {
            str(symbol or "").strip().lower()
            for item in owner_file_plans
            for symbol in _file_plan_values(item, "defines_symbols")
            if str(symbol or "").strip()
        }
        route_paths = _dedupe_nonempty(
            [
                _path_from_file_plan(item)
                for item in file_plan_items
                if _path_from_file_plan(item).endswith(".py")
                and _is_active_route_path(_path_from_file_plan(item))
                and (
                    unit_id in _file_plan_values(item, "owned_units")
                    or bool(owner_symbols.intersection({symbol.lower() for symbol in _file_plan_values(item, "calls_symbols")}))
                    or any(path and path in _file_plan_values(item, "depends_on") + _file_plan_values(item, "consumes") for path in active_owner_paths)
                    or _path_from_file_plan(item) in active_owner_paths
                )
            ]
        )
        if not file_plan_items:
            route_paths = _dedupe_nonempty(
                [
                    path
                    for path in owner_file_paths + execution_entrypoints
                    if path.endswith(".py") and _is_active_route_path(path)
                ]
            )
        support_only = bool(surfaces) and set(surface.lower() for surface in surfaces).issubset(
            {"artifact_writer", "config", "tests", "entrypoint", "documentation", "packaging"}
        )
        if support_only:
            support_only_units.append(unit_id)
        elif not owner_work_packages or not active_owner_paths:
            missing_owner_units.append(unit_id)
        elif not route_paths:
            missing_active_route_units.append(unit_id)
        validation_hooks = _dedupe_nonempty(
            [
                f"unit_active_owner:{path}"
                for path in active_owner_paths[:4]
            ]
            + [
                f"unit_active_route:{path}"
                for path in route_paths[:4]
            ]
            + [
                f"artifact:{artifact}"
                for artifact in artifacts[:4]
            ]
        )
        rows.append(
            {
                "unit_id": unit_id,
                "type": str(getattr(unit, "type", "") or ""),
                "statement": _truncate_text(
                    _prompt_safe_statement(unit_id, getattr(unit, "statement", ""), obligations, surfaces, artifacts),
                    220,
                ),
                "required_surfaces": surfaces[:8],
                "required_obligations": obligations[:4],
                "expected_artifacts": artifacts[:6],
                "owner_work_package_ids": owner_work_packages,
                "owner_file_paths": owner_file_paths[:12],
                "active_owner_paths": active_owner_paths[:8],
                "called_by_or_route_paths": route_paths[:8],
                "support_only": support_only,
                "validation_hooks": validation_hooks,
                "closure_rule": (
                    "This unit is closed when at least one active Python owner implements it and a canonical "
                    "entry/training/evaluation/reporting route reaches that owner."
                ),
            }
        )

    return {
        "source": "prepare_units_and_plan_artifacts",
        "unit_count": len(units),
        "row_count": len(rows),
        "rows": rows,
        "support_only_units": support_only_units,
        "missing_owner_units": missing_owner_units,
        "missing_active_route_units": missing_active_route_units,
        "closure_rules": [
            "Every implementation-bearing unit needs owner_work_package_ids, active_owner_paths, called_by_or_route_paths, and validation_hooks before generation.",
            "Method/dataset/model/baseline/metric/training/evaluation units need active Python route ownership; documentation, config, registries, manifests, and smoke artifacts are supplementary.",
            "If full execution is expensive, keep full-mode command/API and smoke command/API separate, but both must route through the same implementation owner.",
        ],
    }


def _canonical_requirement_projection(state: PaperBenchReproState) -> list[dict[str, Any]]:
    if state.canonical_ir is not None and state.canonical_ir.requirements:
        return [item.model_dump(mode="json") for item in state.canonical_ir.requirements]
    return [
        {
            "requirement_id": str(item.requirement_id or "").strip(),
            "category": str(item.category or "experiment").strip(),
            "title": f"Cover requirement {str(item.requirement_id or '').strip()} active implementation route",
            "description": (
                f"Implement active source-unit surfaces and artifacts for requirement {str(item.requirement_id or '').strip()}."
            ),
            "source_unit_ids": [
                str(unit.unit_id or "").strip()
                for unit in list(state.unit_extraction.units if state.unit_extraction else [])
                if str(unit.unit_id or "").strip()
            ],
            "acceptance_criteria": [
                f"Active implementation route exists for requirement {str(item.requirement_id or '').strip()}."
            ],
        }
        for item in list(state.boundary_requirements.boundary_requirements if state.boundary_requirements else [])
        if str(item.requirement_id or "").strip()
    ]


def _canonical_work_package_projection(state: PaperBenchReproState) -> list[dict[str, Any]]:
    if state.canonical_ir is not None and state.canonical_ir.work_packages:
        return [item.model_dump(mode="json") for item in state.canonical_ir.work_packages]
    requirement_ids_by_wp = {
        str(item.work_package_id or "").strip(): [
            str(req_id or "").strip()
            for req_id in list(item.requirement_ids or [])
            if str(req_id or "").strip()
        ]
        for item in list(state.global_contract.work_package_contracts if state.global_contract else [])
        if str(item.work_package_id or "").strip()
    }
    return [
        {
            "work_package_id": str(item.work_package_id or "").strip(),
            "goal": str(item.goal or "").strip(),
            "requirement_ids": list(requirement_ids_by_wp.get(str(item.work_package_id or "").strip(), [])),
            "source_unit_ids": [
                str(unit_id or "").strip()
                for unit_id in list(item.owned_unit_ids or [])
                if str(unit_id or "").strip()
            ],
            "produces": [
                _normalize_repo_path(path)
                for path in list(item.produces or [])
                if _normalize_repo_path(path)
            ],
            "depends_on": [
                str(dep_id or "").strip()
                for dep_id in list(item.depends_on or [])
                if str(dep_id or "").strip()
            ],
        }
        for item in list(state.work_package_planning.work_packages if state.work_package_planning else [])
        if str(item.work_package_id or "").strip()
    ]


def _canonical_contract_stage_projection(state: PaperBenchReproState) -> list[dict[str, Any]]:
    if state.canonical_ir is not None and state.canonical_ir.contract_stages:
        return [item.model_dump(mode="json") for item in state.canonical_ir.contract_stages]
    owner_work_packages = [
        str(item.work_package_id or "").strip()
        for item in list(state.global_contract.work_package_contracts if state.global_contract else [])
        if str(item.work_package_id or "").strip()
    ]
    result_target_ids = [
        str(item.target_id or "").strip()
        for item in list(state.global_contract.result_targets if state.global_contract else [])
        if str(item.target_id or "").strip()
    ]
    return [
        {
            "stage_id": f"stage_{str(stage_name or '').strip().replace('-', '_')}",
            "label": str(stage_name or "").strip(),
            "owner_work_package_ids": list(owner_work_packages),
            "result_target_ids": list(result_target_ids),
        }
        for stage_name in list(state.global_contract.canonical_stage_sequence if state.global_contract else [])
        if str(stage_name or "").strip()
    ]


def _canonical_file_graph_projection(
    state: PaperBenchReproState,
    *,
    architecture: ArchitectureOutput | None = None,
    projected_file_plans: PackageFilePlanningOutput | None = None,
) -> dict[str, Any]:
    if state.canonical_ir is not None and state.canonical_ir.file_nodes:
        surface_registry = {
            "entrypoints": [
                item.canonical_path
                for item in state.canonical_ir.surface_nodes
                if item.surface_kind == "entrypoint" and str(item.canonical_path or "").strip()
            ],
            "config_surfaces": [
                item.canonical_path
                for item in state.canonical_ir.surface_nodes
                if item.surface_kind == "config" and str(item.canonical_path or "").strip()
            ],
            "stable_interfaces": [
                item.canonical_path
                for item in state.canonical_ir.surface_nodes
                if item.surface_kind == "stable_interface" and str(item.canonical_path or "").strip()
            ],
            "artifact_paths": [
                item.canonical_path
                for item in state.canonical_ir.surface_nodes
                if item.surface_kind == "artifact" and str(item.canonical_path or "").strip()
            ],
            "producer_surfaces": [
                item.canonical_path
                for item in state.canonical_ir.surface_nodes
                if item.surface_kind == "producer" and str(item.canonical_path or "").strip()
            ],
        }
        paths_by_work_package: dict[str, list[str]] = {}
        path_to_owner_work_package: dict[str, str] = {}
        for item in state.canonical_ir.file_nodes:
            path = _normalize_repo_path(item.canonical_path)
            if not path:
                continue
            owner = str(item.owner_work_package_id or "").strip()
            if owner:
                paths_by_work_package.setdefault(owner, [])
                if path not in paths_by_work_package[owner]:
                    paths_by_work_package[owner].append(path)
                path_to_owner_work_package[path] = owner
        return {
            "source": "canonical_ir_shadow",
            "registered_paths": list(state.canonical_ir.validation_index.get("registered_paths", [])),
            "file_nodes": [item.model_dump(mode="json") for item in state.canonical_ir.file_nodes],
            "surface_nodes": [item.model_dump(mode="json") for item in state.canonical_ir.surface_nodes],
            "surface_registry": surface_registry,
            "paths_by_work_package": paths_by_work_package,
            "path_to_owner_work_package": path_to_owner_work_package,
            "path_drift_candidates": [],
            "projection_drift": [],
            "allow_only_registered_paths": True,
        }

    architecture_obj = architecture if architecture is not None else state.architecture
    registered_paths = _dedupe_nonempty(
        [
            _normalize_repo_path(path)
            for path in list(architecture_obj.target_file_tree if architecture_obj is not None else [])
            if _normalize_repo_path(path)
        ]
    )
    path_to_owner_work_package: dict[str, str] = {}
    if architecture_obj is not None:
        for work_package_id, paths in dict(architecture_obj.package_layout or {}).items():
            owner = str(work_package_id or "").strip()
            if not owner:
                continue
            for path in list(paths or []):
                normalized = _normalize_repo_path(path)
                if normalized:
                    path_to_owner_work_package[normalized] = owner
    paths_by_work_package: dict[str, list[str]] = {}
    for path, owner in path_to_owner_work_package.items():
        paths_by_work_package.setdefault(owner, [])
        if path not in paths_by_work_package[owner]:
            paths_by_work_package[owner].append(path)
    file_nodes = [
        {
            "file_id": f"file_{path.replace('/', '_').replace('.', '_')}",
            "canonical_path": path,
            "owner_work_package_id": path_to_owner_work_package.get(path, ""),
            "related_requirement_ids": [],
            "related_plan_node_ids": [],
            "surfaces": [],
        }
        for path in registered_paths
    ]
    projected_paths = _dedupe_nonempty(
        [
            _normalize_repo_path(item.target_file)
            for item in list(projected_file_plans.file_plans if projected_file_plans is not None else [])
            if _normalize_repo_path(item.target_file)
        ]
    )
    projection_drift = sorted([path for path in projected_paths if path not in set(registered_paths)])
    surface_registry = {
        "entrypoints": _dedupe_nonempty(
            [_normalize_repo_path(path) for path in list(architecture_obj.execution_entrypoints if architecture_obj is not None else [])]
        ),
        "config_surfaces": _dedupe_nonempty(
            [_normalize_repo_path(path) for path in list(architecture_obj.config_surfaces if architecture_obj is not None else [])]
        ),
        "stable_interfaces": _dedupe_nonempty(
            [_normalize_repo_path(path) for path in list(architecture_obj.stable_interfaces if architecture_obj is not None else [])]
        ),
        "artifact_paths": _dedupe_nonempty(
            [
                _normalize_repo_path(path)
                for target in list(state.global_contract.result_targets if state.global_contract else [])
                for path in list(target.artifact_paths or [])
                if _normalize_repo_path(path)
            ]
        ),
        "producer_surfaces": _dedupe_nonempty(
            [
                _normalize_repo_path(path)
                for path in list(path_to_owner_work_package.keys())
                if _normalize_repo_path(path)
            ]
        ),
    }
    surface_paths = _dedupe_nonempty(
        list(surface_registry["entrypoints"])
        + list(surface_registry["config_surfaces"])
        + list(surface_registry["stable_interfaces"])
    )
    path_drift_candidates = sorted([path for path in surface_paths if path not in set(registered_paths)])
    return {
        "source": "state_projection",
        "registered_paths": registered_paths,
        "file_nodes": file_nodes,
        "surface_nodes": [],
        "surface_registry": surface_registry,
        "paths_by_work_package": paths_by_work_package,
        "path_to_owner_work_package": path_to_owner_work_package,
        "path_drift_candidates": path_drift_candidates,
        "projection_drift": projection_drift,
        "allow_only_registered_paths": True,
    }


def _canonical_validation_projection(state: PaperBenchReproState) -> dict[str, Any]:
    if state.canonical_ir_validation is not None:
        return state.canonical_ir_validation.model_dump(mode="json")
    return {
        "mode": "shadow",
        "passed": True,
        "planning_failure_layer": "",
        "mismatch_summary": {},
        "gate_actions": [],
        "semantic_validation_report": {},
    }

def _build_boundary_requirements_context(state: PaperBenchReproState) -> dict:
    """Build Stage-1 boundary extraction context."""
    upstream_intent = upstream_intent_payload(state)
    upstream_intent["experiment_design"] = _experiment_design_payload_for_planning(state)
    paper_context = _paper_context_payload_for_planning(state)
    return {
        "target": _target_brief_for_planning(state),
        "paper_context": paper_context,
        "paper_evidence_contract": _paper_evidence_contract_payload(state),
        "reference_source_policy": _planning_source_policy_payload(state),
        "upstream_intent": upstream_intent,
        "normalized_input": _normalized_input_for_planning(state),
        "units": _unit_payloads_for_planning(state),
        "experiment_design": _experiment_design_for_planning(state),
        "dataset_preparation": _get_dataset_preparation(state),
        "resource_manifest": _resource_manifest_for_planning(state),
    }

def _build_input_normalization_context(state: PaperBenchReproState) -> dict:
    """Build Stage-0 input-normalization context."""
    upstream_intent = upstream_intent_payload(state)
    paper_context = _paper_context_payload(state, include_chunk_text=True)
    return {
        "target": _prompt_safe_text(state.input.target),
        "language": state.input.language,
        "paper_context": paper_context,
        "upstream_intent": upstream_intent,
        "experiment_design": _experiment_design_payload(state),
        "resource_manifest": _resource_manifest_for_planning(state),
    }


def _build_unit_extraction_context(state: PaperBenchReproState) -> dict:
    """Build Stage-1 unit-extraction context."""
    upstream_intent = upstream_intent_payload(state)
    upstream_intent["experiment_design"] = _experiment_design_payload_for_planning(state)
    paper_context = _paper_context_payload_for_planning(state)
    return {
        "target": _target_brief_for_planning(state),
        "language": state.input.language,
        "paper_context": paper_context,
        "paper_evidence_contract": _paper_evidence_contract_payload(state),
        "reference_source_policy": _planning_source_policy_payload(state),
        "upstream_intent": upstream_intent,
        "normalized_input": _normalized_input_for_planning(state),
        "experiment_design": _experiment_design_for_planning(state),
        "dataset_preparation": _get_dataset_preparation(state),
        "resource_manifest": _resource_manifest_for_planning(state),
    }


def _build_reference_selection_context(
    state: PaperBenchReproState,
    boundary_output: BoundaryRequirementsOutput,
) -> dict:
    """Build Stage-2 reference selection context."""
    repo_preparation = _get_reference_repo_preparation(state)
    prepared_reference_repositories = [
        _compact_reference_repo_payload(item)
        for item in _get_reference_repo_surveys(state, boundary_output)
    ]
    upstream_intent = upstream_intent_payload(state)
    upstream_intent["experiment_design"] = _experiment_design_payload_for_planning(state)
    paper_context = _paper_context_payload_for_planning(state)
    return {
        "target": _target_brief_for_planning(state),
        "paper_context": paper_context,
        "paper_evidence_contract": _paper_evidence_contract_payload(state),
        "reference_source_policy": _planning_source_policy_payload(state),
        "upstream_intent": upstream_intent,
        "normalized_input": _normalized_input_for_planning(state),
        "units": _unit_payloads_for_planning(state),
        "experiment_design": _experiment_design_for_planning(state),
        "dataset_preparation": _get_dataset_preparation(state),
        "resource_manifest": _resource_manifest_for_planning(state),
        "boundary_requirements": _boundary_requirements_for_planning(boundary_output),
        "topic_profile": state.topic_profile.model_dump(mode="json") if state.topic_profile else {},
        "reference_repo_preparation": repo_preparation,
        "prepared_reference_repositories": prepared_reference_repositories,
    }

def _build_work_package_planning_context(state: PaperBenchReproState) -> dict:
    """Build work-package planning context aligned with reproduction middle stage."""
    repo_preparation = _get_reference_repo_preparation(state)
    prepared_reference_repositories = [
        _compact_reference_repo_payload(item)
        for item in state.reference_repo_surveys
    ]
    upstream_intent = upstream_intent_payload(state)
    upstream_intent["experiment_design"] = _experiment_design_payload_for_planning(state)
    paper_context = _paper_context_payload_for_planning(state)
    return {
        "target": _target_brief_for_planning(state),
        "paper_context": paper_context,
        "paper_evidence_contract": _paper_evidence_contract_payload(state),
        "reference_source_policy": _planning_source_policy_payload(state),
        "upstream_intent": upstream_intent,
        "normalized_input": _normalized_input_for_planning(state),
        "unit_implementation_contracts": _unit_implementation_contract_projection(state),
        "unit_route_matrix": _unit_route_matrix_projection(state),
        "units": _unit_payloads_for_planning(state),
        "boundary_requirements": _boundary_requirements_for_planning(state.boundary_requirements),
        "topic_profile": state.topic_profile.model_dump(mode="json") if state.topic_profile else {},
        "dataset_preparation": _get_dataset_preparation(state),
        "resource_manifest": _resource_manifest_for_planning(state),
        "reference_repo_preparation": repo_preparation,
        "prepared_reference_repositories": prepared_reference_repositories,
    }


def _build_work_package_local_context(
    state: PaperBenchReproState,
    work_package_id: str,
    *,
    base_context: dict | None = None,
    current_work_package: dict | None = None,
) -> dict:
    """Build a package-local planning context for fan-out refinement."""
    base = dict(base_context or _build_work_package_planning_context(state))
    selected = dict(current_work_package or {})
    if not selected:
        planning = _work_package_planning_for_prompts(state)
        work_packages = list(dict(planning).get("work_packages", []) or [])
        selected = next(
            (
                item for item in work_packages
                if isinstance(item, dict) and str(item.get("work_package_id", "") or "").strip() == str(work_package_id or "").strip()
            ),
            {},
        )
    selected_unit_ids = {
        str(item).strip()
        for item in list(dict(selected).get("owned_unit_ids", []) or [])
        if str(item).strip()
    }
    selected_requirement_ids: set[str] = set()
    for item in list(dict(base.get("boundary_requirements", {}) or {}).get("boundary_requirements", []) or []):
        if not isinstance(item, dict):
            continue
        requirement_id = str(item.get("requirement_id", "") or "").strip()
        if not requirement_id:
            continue
        source_unit_ids = {
            str(unit_id).strip()
            for unit_id in list(item.get("source_unit_ids", []) or [])
            if str(unit_id).strip()
        }
        if selected_unit_ids and source_unit_ids.intersection(selected_unit_ids):
            selected_requirement_ids.add(requirement_id)
    selected_reference_ids = {
        str(item).strip()
        for item in list(dict(selected).get("reference_ids", []) or [])
        if str(item).strip()
    }
    selected_artifact_contract = {
        "work_package_id": str(work_package_id or "").strip(),
        "produces": list(dict(selected).get("produces", []) or []),
        "inventories": dict(dict(selected).get("inventories", {}) or {}),
        "interface_contract": list(dict(selected).get("interface_contract", []) or []),
        "method_obligations": list(dict(selected).get("method_obligations", []) or []),
    }
    return {
        **base,
        "fanout_scope": "work_package",
        "fanout_work_package_id": str(work_package_id or "").strip(),
        "paper_evidence_contract": {
            **dict(base.get("paper_evidence_contract", {}) or {}),
            "fanout_selected_unit_ids": sorted(selected_unit_ids),
        },
        "unit_implementation_contracts": _unit_implementation_contract_projection(state, unit_ids=selected_unit_ids),
        "unit_route_matrix": _unit_route_matrix_projection(state, unit_ids=selected_unit_ids),
        "units": [
            item
            for item in list(base.get("units", []) or [])
            if not selected_unit_ids or str(dict(item).get("unit_id", "") or "").strip() in selected_unit_ids
        ],
        "boundary_requirements": {
            **dict(base.get("boundary_requirements", {}) or {}),
            "boundary_requirements": [
                item
                for item in list(dict(base.get("boundary_requirements", {}) or {}).get("boundary_requirements", []) or [])
                if not selected_requirement_ids or str(dict(item).get("requirement_id", "") or "").strip() in selected_requirement_ids
            ],
        },
        "prepared_reference_repositories": [
            item
            for item in list(base.get("prepared_reference_repositories", []) or [])
            if not selected_reference_ids or str(dict(item).get("ref_id", "") or "").strip() in selected_reference_ids
        ],
        "current_work_package": selected,
        "artifact_contract_slice": selected_artifact_contract,
    }


def _build_topic_profile_context(
    state: PaperBenchReproState,
    boundary_output: BoundaryRequirementsOutput,
) -> dict:
    """Build Stage-1.5 topic-profile context."""
    repo_preparation = _get_reference_repo_preparation(state)
    prepared_reference_repositories = [
        _compact_reference_repo_payload(item)
        for item in _get_reference_repo_surveys(state, boundary_output)
    ]
    upstream_intent = upstream_intent_payload(state)
    upstream_intent["experiment_design"] = _experiment_design_payload_for_planning(state)
    paper_context = _paper_context_payload_for_planning(state)
    return {
        "target": _target_brief_for_planning(state),
        "paper_context": paper_context,
        "paper_evidence_contract": _paper_evidence_contract_payload(state),
        "reference_source_policy": _planning_source_policy_payload(state),
        "upstream_intent": upstream_intent,
        "normalized_input": _normalized_input_for_planning(state),
        "unit_implementation_contracts": _unit_implementation_contract_projection(state),
        "unit_route_matrix": _unit_route_matrix_projection(state),
        "units": _unit_payloads_for_planning(state),
        "experiment_design": _experiment_design_for_planning(state),
        "dataset_preparation": _get_dataset_preparation(state),
        "resource_manifest": _resource_manifest_for_planning(state),
        "boundary_requirements": _boundary_requirements_for_planning(boundary_output),
        "reference_repo_preparation": repo_preparation,
        "prepared_reference_repositories": prepared_reference_repositories,
    }

def _build_pipeline_plan_context(
    state: PaperBenchReproState,
    boundary_output: BoundaryRequirementsOutput,
    reference_output: ReferenceSelectionOutput,
) -> dict:
    """Build Stage-3 flat planning context."""
    repo_preparation = _get_reference_repo_preparation(state)
    prepared_reference_repositories = [
        _compact_reference_repo_payload(item)
        for item in _get_reference_repo_surveys(state, boundary_output)
    ]
    upstream_intent = upstream_intent_payload(state)
    upstream_intent["experiment_design"] = _experiment_design_payload_for_planning(state)
    paper_context = _paper_context_payload_for_planning(state)
    return {
        "target": _target_brief_for_planning(state),
        "paper_context": paper_context,
        "paper_evidence_contract": _paper_evidence_contract_payload(state),
        "reference_source_policy": _planning_source_policy_payload(state),
        "upstream_intent": upstream_intent,
        "normalized_input": _normalized_input_for_planning(state),
        "unit_implementation_contracts": _unit_implementation_contract_projection(state),
        "unit_route_matrix": _unit_route_matrix_projection(state),
        "units": _unit_payloads_for_planning(state),
        "experiment_design": _experiment_design_for_planning(state),
        "dataset_preparation": _get_dataset_preparation(state),
        "resource_manifest": _resource_manifest_for_planning(state),
        "boundary_requirements": _boundary_requirements_for_planning(boundary_output),
        "work_package_planning": _work_package_planning_for_prompts(state),
        "evidence_bundles": [item.model_dump(mode="json") for item in state.evidence_bundles],
        "topic_profile": state.topic_profile.model_dump(mode="json") if state.topic_profile else {},
        "reference_selection": reference_output.model_dump(mode="json"),
        "reference_repo_preparation": repo_preparation,
        "prepared_reference_repositories": prepared_reference_repositories,
    }

def _build_global_contract_context(
    state: PaperBenchReproState,
    boundary_output: BoundaryRequirementsOutput,
    reference_output: ReferenceSelectionOutput,
    pipeline_plan: PipelinePlanOutput,
) -> dict:
    """Build Stage-3.5 global-contract context."""
    if boundary_output is None:
        boundary_output = state.boundary_requirements
    if boundary_output is None:
        raise RuntimeError(
            "global_contract context needs boundary_requirements; restore nodes/plan/boundary_requirements.json "
            "or rerun plan so active units can be synthesized before global_contract_synthesis"
        )
    upstream_intent = upstream_intent_payload(state)
    upstream_intent["experiment_design"] = _experiment_design_payload_for_planning(state)
    paper_context = _paper_context_payload_for_planning(state)
    return {
        "target": _target_brief_for_planning(state),
        "paper_context": paper_context,
        "paper_evidence_contract": _paper_evidence_contract_payload(state),
        "reference_source_policy": _planning_source_policy_payload(state),
        "upstream_intent": upstream_intent,
        "normalized_input": _normalized_input_for_planning(state),
        "unit_implementation_contracts": _unit_implementation_contract_projection(state),
        "unit_route_matrix": _unit_route_matrix_projection(state),
        "units": _unit_payloads_for_planning(state),
        "experiment_design": _experiment_design_for_planning(state),
        "dataset_preparation": _get_dataset_preparation(state),
        "resource_manifest": _resource_manifest_for_planning(state),
        "boundary_requirements": _boundary_requirements_for_planning(boundary_output),
        "work_package_planning": _work_package_planning_for_prompts(state),
        "evidence_bundles": [item.model_dump(mode="json") for item in state.evidence_bundles],
        "topic_profile": state.topic_profile.model_dump(mode="json") if state.topic_profile else {},
        "reference_selection": reference_output.model_dump(mode="json"),
        "pipeline_plan": _pipeline_plan_for_prompts(pipeline_plan),
    }

def _build_architecture_context(
    state: PaperBenchReproState,
    boundary_output: BoundaryRequirementsOutput,
    reference_output: ReferenceSelectionOutput,
    pipeline_plan: PipelinePlanOutput,
) -> dict:
    """Build Stage-4 architecture synthesis context."""
    repo_preparation = _get_reference_repo_preparation(state)
    prepared_reference_repositories = [
        _compact_reference_repo_payload(item)
        for item in _get_reference_repo_surveys(state, boundary_output)
    ]
    canonical_file_graph = _canonical_file_graph_projection(state)
    upstream_intent = upstream_intent_payload(state)
    upstream_intent["experiment_design"] = _experiment_design_payload_for_planning(state)
    paper_context = _paper_context_payload_for_planning(state)
    return {
        "target": _target_brief_for_planning(state),
        "paper_context": paper_context,
        "paper_evidence_contract": _paper_evidence_contract_payload(state),
        "reference_source_policy": _planning_source_policy_payload(state),
        "upstream_intent": upstream_intent,
        "normalized_input": _normalized_input_for_planning(state),
        "unit_implementation_contracts": _unit_implementation_contract_projection(state),
        "unit_route_matrix": _unit_route_matrix_projection(state),
        "units": _unit_payloads_for_planning(state),
        "experiment_design": _experiment_design_for_planning(state),
        "dataset_preparation": _get_dataset_preparation(state),
        "resource_manifest": _resource_manifest_for_planning(state),
        "boundary_requirements": _boundary_requirements_for_planning(boundary_output),
        "canonical_requirements": _canonical_requirement_projection(state),
        "canonical_work_packages": _canonical_work_package_projection(state),
        "canonical_contract_stages": _canonical_contract_stage_projection(state),
        "canonical_file_graph": canonical_file_graph,
        "canonical_ir_validation": _canonical_validation_projection(state),
        "work_package_planning": _work_package_planning_for_prompts(state),
        "evidence_bundles": [item.model_dump(mode="json") for item in state.evidence_bundles],
        "topic_profile": state.topic_profile.model_dump(mode="json") if state.topic_profile else {},
        "reference_selection": reference_output.model_dump(mode="json"),
        "pipeline_plan": _pipeline_plan_for_prompts(pipeline_plan),
        "global_contract": state.global_contract.model_dump(mode="json") if state.global_contract else {},
        "reference_repo_preparation": repo_preparation,
        "prepared_reference_repositories": prepared_reference_repositories,
    }


def _build_architecture_package_context(
    state: PaperBenchReproState,
    work_package_id: str,
    *,
    architecture: ArchitectureOutput | None = None,
    base_context: dict | None = None,
) -> dict:
    """Build package-local architecture context for fan-out refinement."""
    base = dict(base_context or _build_architecture_context(
        state,
        state.boundary_requirements,
        state.reference_selection,
        state.pipeline_plan,
    ))
    work_packages = list(dict(base.get("work_package_planning", {}) or {}).get("work_packages", []) or [])
    selected = next(
        (
            item for item in work_packages
            if isinstance(item, dict) and str(item.get("work_package_id", "") or "").strip() == str(work_package_id or "").strip()
        ),
        {},
    )
    selected_unit_ids = {
        str(item).strip()
        for item in list(dict(selected).get("owned_unit_ids", []) or [])
        if str(item).strip()
    }
    selected_reference_ids = {
        str(item).strip()
        for item in list(dict(selected).get("reference_ids", []) or [])
        if str(item).strip()
    }
    package_layout = dict((architecture.package_layout if architecture is not None else {}) or {})
    selected_paths = list(package_layout.get(str(work_package_id or "").strip(), []) or [])
    dependency_graph = [
        item
        for item in list(dict(base.get("architecture", {}) or {}).get("dependency_graph", []) or [])
        if isinstance(item, dict)
        and (
            str(item.get("source_path", "") or "").strip() in selected_paths
            or str(item.get("target_path", "") or "").strip() in selected_paths
        )
    ]
    return {
        **base,
        "fanout_scope": "architecture_package",
        "fanout_work_package_id": str(work_package_id or "").strip(),
        "paper_evidence_contract": {
            **dict(base.get("paper_evidence_contract", {}) or {}),
            "fanout_selected_unit_ids": sorted(selected_unit_ids),
            "fanout_selected_paths": selected_paths,
        },
        "units": [
            item
            for item in list(base.get("units", []) or [])
            if not selected_unit_ids or str(dict(item).get("unit_id", "") or "").strip() in selected_unit_ids
        ],
        "unit_implementation_contracts": _unit_implementation_contract_projection(state, unit_ids=selected_unit_ids),
        "unit_route_matrix": _unit_route_matrix_projection(
            state,
            architecture=architecture,
            unit_ids=selected_unit_ids,
        ),
        "work_package_planning": {
            **dict(base.get("work_package_planning", {}) or {}),
            "work_packages": [selected] if selected else [],
        },
        "evidence_bundles": [
            item
            for item in list(base.get("evidence_bundles", []) or [])
            if str(dict(item).get("work_package_id", "") or "").strip() == str(work_package_id or "").strip()
        ],
        "reference_selection": {
            **dict(base.get("reference_selection", {}) or {}),
            "actionable_references": [
                item
                for item in list(dict(base.get("reference_selection", {}) or {}).get("actionable_references", []) or [])
                if not selected_reference_ids or str(dict(item).get("ref_id", "") or "").strip() in selected_reference_ids
            ],
            "reference_relations": [
                item
                for item in list(dict(base.get("reference_selection", {}) or {}).get("reference_relations", []) or [])
                if not selected_reference_ids or str(dict(item).get("ref_id", "") or "").strip() in selected_reference_ids
            ],
        },
        "package_artifact_contract": {
            "work_package_id": str(work_package_id or "").strip(),
            "produces": list(dict(selected).get("produces", []) or []),
            "inventories": dict(dict(selected).get("inventories", {}) or {}),
            "interface_contract": list(dict(selected).get("interface_contract", []) or []),
            "method_obligations": list(dict(selected).get("method_obligations", []) or []),
        },
        "adjacent_dependency_files": list(
            dict.fromkeys(
                [
                    str(item.get("source_path", "") or "").strip()
                    for item in dependency_graph
                    if str(item.get("source_path", "") or "").strip()
                ] + [
                    str(item.get("target_path", "") or "").strip()
                    for item in dependency_graph
                    if str(item.get("target_path", "") or "").strip()
                ]
            )
        ),
        "package_dependency_graph": dependency_graph,
    }

def _build_package_file_planning_context(
    state: PaperBenchReproState,
    architecture: ArchitectureOutput,
    pipeline_plan: PipelinePlanOutput,
    projected_file_plans: PackageFilePlanningOutput,
) -> dict:
    """Build Stage-4.5 package-file planning context."""
    canonical_file_graph = _canonical_file_graph_projection(
        state,
        architecture=architecture,
        projected_file_plans=projected_file_plans,
    )
    critical_grounding_failures = [
        str(item.work_package_id or "").strip()
        for item in state.evidence_bundles
        if str(item.grounding_status or "").strip().lower() not in {"grounded", "self_contained"}
    ]
    path_drift_candidates = list(canonical_file_graph.get("path_drift_candidates", []) or [])
    projection_drift = list(canonical_file_graph.get("projection_drift", []) or [])
    canonical_drift = sorted(
        {
            _normalize_repo_path(path)
            for path in [*path_drift_candidates, *projection_drift]
            if _normalize_repo_path(path)
        }
    )
    upstream_intent = upstream_intent_payload(state)
    upstream_intent["experiment_design"] = _experiment_design_payload_for_planning(state)
    paper_context = _paper_context_payload_for_planning(state)
    return {
        "target": _target_brief_for_planning(state),
        "paper_context": paper_context,
        "paper_evidence_contract": _paper_evidence_contract_payload(state),
        "reference_source_policy": _planning_source_policy_payload(state),
        "upstream_intent": upstream_intent,
        "normalized_input": _normalized_input_for_planning(state),
        "unit_implementation_contracts": _unit_implementation_contract_projection(state),
        "unit_route_matrix": _unit_route_matrix_projection(
            state,
            architecture=architecture,
            file_plans=projected_file_plans,
        ),
        "units": _unit_payloads_for_planning(state),
        "boundary_requirements": _boundary_requirements_for_planning(state.boundary_requirements),
        "canonical_requirements": _canonical_requirement_projection(state),
        "canonical_work_packages": _canonical_work_package_projection(state),
        "canonical_contract_stages": _canonical_contract_stage_projection(state),
        "canonical_file_graph": canonical_file_graph,
        "canonical_ir_validation": _canonical_validation_projection(state),
        "topic_profile": state.topic_profile.model_dump(mode="json") if state.topic_profile else {},
        "work_package_planning": _work_package_planning_for_prompts(state),
        "evidence_bundles": [item.model_dump(mode="json") for item in state.evidence_bundles],
        "global_contract": state.global_contract.model_dump(mode="json") if state.global_contract else {},
        "reference_selection": state.reference_selection.model_dump(mode="json") if state.reference_selection else {},
        "architecture": _architecture_for_prompts(architecture),
        "pipeline_plan": _pipeline_plan_for_prompts(pipeline_plan),
        "projected_file_plans": projected_file_plans.model_dump(mode="json"),
        "canonical_path_drift_candidates": path_drift_candidates,
        "canonical_projection_drift": projection_drift,
        "canonical_path_drift": canonical_drift,
        "allow_only_registered_paths": bool(canonical_file_graph.get("allow_only_registered_paths", True)),
        "dataset_preparation": _get_dataset_preparation(state),
        "resource_manifest": _resource_manifest_for_planning(state),
        "critical_ungrounded_work_packages": critical_grounding_failures,
    }


def _build_package_file_planning_local_context(
    state: PaperBenchReproState,
    architecture: ArchitectureOutput,
    pipeline_plan: PipelinePlanOutput,
    projected_file_plans: PackageFilePlanningOutput,
    work_package_id: str,
    *,
    base_context: dict | None = None,
) -> dict:
    """Build file-planning context restricted to one work package/file group."""
    base = dict(base_context or _build_package_file_planning_context(
        state,
        architecture,
        pipeline_plan,
        projected_file_plans,
    ))
    package_id = str(work_package_id or "").strip()
    package_work_packages = list(dict(base.get("work_package_planning", {}) or {}).get("work_packages", []) or [])
    selected_package = next(
        (
            item for item in package_work_packages
            if isinstance(item, dict) and str(item.get("work_package_id", "") or "").strip() == package_id
        ),
        {},
    )
    package_files = list(dict(architecture.package_layout or {}).get(package_id, []) or [])
    selected_reference_ids = {
        str(item).strip()
        for item in list(dict(selected_package).get("reference_ids", []) or [])
        if str(item).strip()
    }
    selected_unit_ids = {
        str(item).strip()
        for item in list(dict(selected_package).get("owned_unit_ids", []) or [])
        if str(item).strip()
    }
    dependency_graph = [
        item
        for item in list(dict(base.get("architecture", {}) or {}).get("dependency_graph", []) or [])
        if isinstance(item, dict)
        and (
            str(item.get("source_path", "") or "").strip() in package_files
            or str(item.get("target_path", "") or "").strip() in package_files
        )
    ]
    adjacent_dependency_files = list(
        dict.fromkeys(
            [
                str(item.get("source_path", "") or "").strip()
                for item in dependency_graph
                if str(item.get("source_path", "") or "").strip()
            ] + [
                str(item.get("target_path", "") or "").strip()
                for item in dependency_graph
                if str(item.get("target_path", "") or "").strip()
            ]
        )
    )
    return {
        **base,
        "fanout_scope": "package_file_planning",
        "fanout_work_package_id": package_id,
        "paper_evidence_contract": {
            **dict(base.get("paper_evidence_contract", {}) or {}),
            "fanout_selected_unit_ids": sorted(selected_unit_ids),
            "fanout_package_files": package_files,
        },
        "unit_implementation_contracts": _unit_implementation_contract_projection(state, unit_ids=selected_unit_ids),
        "unit_route_matrix": _unit_route_matrix_projection(
            state,
            architecture=architecture,
            file_plans=projected_file_plans,
            unit_ids=selected_unit_ids,
        ),
        "units": [
            item
            for item in list(base.get("units", []) or [])
            if not selected_unit_ids or str(dict(item).get("unit_id", "") or "").strip() in selected_unit_ids
        ],
        "work_package_planning": {
            **dict(base.get("work_package_planning", {}) or {}),
            "work_packages": [selected_package] if selected_package else [],
        },
        "evidence_bundles": [
            item
            for item in list(base.get("evidence_bundles", []) or [])
            if str(dict(item).get("work_package_id", "") or "").strip() == package_id
        ],
        "architecture": _architecture_for_prompts(
            architecture,
            target_file_tree=package_files,
            file_blueprints=[
                item
                for item in list(architecture.file_blueprints or [])
                if str(item.path or "").strip() in package_files
            ],
            dependency_graph=dependency_graph,
            package_layout={package_id: package_files},
        ),
        "projected_file_plans": {
            **projected_file_plans.model_dump(mode="json"),
            "file_plans": [
                item.model_dump(mode="json")
                for item in list(projected_file_plans.file_plans or [])
                if str(item.work_package_id or "").strip() == package_id or str(item.target_file or "").strip() in package_files
            ],
        },
        "reference_selection": {
            **dict(base.get("reference_selection", {}) or {}),
            "actionable_references": [
                item
                for item in list(dict(base.get("reference_selection", {}) or {}).get("actionable_references", []) or [])
                if not selected_reference_ids or str(dict(item).get("ref_id", "") or "").strip() in selected_reference_ids
            ],
            "reference_relations": [
                item
                for item in list(dict(base.get("reference_selection", {}) or {}).get("reference_relations", []) or [])
                if not selected_reference_ids or str(dict(item).get("ref_id", "") or "").strip() in selected_reference_ids
            ],
        },
        "artifact_contract_slice": {
            "work_package_id": package_id,
            "produces": list(dict(selected_package).get("produces", []) or []),
            "inventories": dict(dict(selected_package).get("inventories", {}) or {}),
            "interface_contract": list(dict(selected_package).get("interface_contract", []) or []),
            "method_obligations": list(dict(selected_package).get("method_obligations", []) or []),
        },
        "adjacent_dependency_files": adjacent_dependency_files,
    }

def _build_generate_contract_context(
    topic_profile: TopicProfileOutput | None,
    global_contract: GlobalContractOutput | None,
    work_package_planning: WorkPackagePlanningOutput | None = None,
) -> dict:
    """Build compact generation-time contract context."""
    return {
        "topic_profile": topic_profile.model_dump(mode="json") if topic_profile else {},
        "global_contract": global_contract.model_dump(mode="json") if global_contract else {},
        "work_package_planning": _work_package_planning_output_for_prompts(work_package_planning),
    }

def _limit_json_for_prompt(data: object, max_chars: int = 16000) -> str:
    """Serialize structured context for prompting with a hard size cap."""
    text = json.dumps(data, indent=2, ensure_ascii=False, default=_json_default)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n...<truncated>..."
