"""Generation-manifest construction helpers for reproagent."""

import re
from typing import Any

from reproagent.pipeline.config import semantic_anchor_disabled
from reproagent.pipeline.schemas import (
    ArchitectureOutput,
    PaperBenchReproState,
    GenerationManifest,
    GenerationTaskInput,
    ReferenceSnippetCandidate,
    TaskItem,
)
from reproagent.pipeline.utils.contract_sanitizer import sanitize_contract_list, sanitize_scope_boundary, sanitize_task_contract
from reproagent.pipeline.utils.prompt_context_builder import _paper_evidence_contract_payload_for_generation


def _tokenize_reference_text(*parts: str) -> set[str]:
    """Extract a compact keyword set for task-to-survey matching."""
    tokens: set[str] = set()
    for part in parts:
        text = str(part or "").strip().lower()
        if not text:
            continue
        for token in re.findall(r"[a-z0-9_./-]{3,}", text):
            normalized = token.strip("._-/")
            if len(normalized) >= 3:
                tokens.add(normalized)
    return tokens


def _task_search_terms(task, plan_node_by_id: dict[str, object]) -> set[str]:
    """Build a keyword set representing one task's implementation intent."""
    related_descriptions = []
    related_names = []
    for node_id in task.related_node_ids:
        node = plan_node_by_id.get(node_id)
        if node is None:
            continue
        related_names.append(getattr(node, "name", ""))
        related_descriptions.append(getattr(node, "description", ""))
        related_descriptions.append(getattr(node, "reusable_module", ""))
    return _tokenize_reference_text(
        task.file_path,
        task.purpose,
        *list(getattr(task, "interface_contract", []) or []),
        *list(getattr(task, "implementation_surfaces", []) or []),
        *list(getattr(task, "method_obligations", []) or []),
        *list(getattr(task, "writes_artifacts", []) or []),
        *task.review_points,
        *related_names,
        *related_descriptions,
    )


def _derive_reusable_module_name(file_path: str) -> str:
    """Derive a stable reusable-module label from a matched file path."""
    normalized = str(file_path or "").strip().replace("\\", "/")
    if not normalized:
        return "reference_impl"
    stem = normalized.rsplit("/", 1)[-1]
    stem = stem.rsplit(".", 1)[0]
    return stem or "reference_impl"


def _dedupe_nonempty(values: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = str(raw or "").strip()
        if not value or value in seen:
            continue
        ordered.append(value)
        seen.add(value)
    return ordered


def _text_score(query_terms: set[str], *parts: str) -> int:
    if not query_terms:
        return 0
    tokens = _tokenize_reference_text(*parts)
    return len(query_terms.intersection(tokens))


def _shorten(value: object, max_chars: int) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)].rstrip() + "..."


def _reference_context_for_task(
    *,
    task: TaskItem,
    file_plan: Any,
    reference_ids: list[str],
    state: PaperBenchReproState,
    survey_by_ref_id: dict[str, object],
    plan_node_by_id: dict[str, object],
) -> dict[str, Any]:
    task_terms = _task_search_terms(task, plan_node_by_id)
    selected_ids = set(reference_ids)
    summaries: list[dict[str, Any]] = []
    symbol_rows: list[dict[str, Any]] = []
    for ref_id, survey in survey_by_ref_id.items():
        if selected_ids and ref_id not in selected_ids:
            continue
        survey_text = " ".join(
            [
                str(getattr(survey, "title", "") or ""),
                str(getattr(survey, "readme_summary", "") or ""),
                str(getattr(survey, "file_tree_summary", "") or ""),
                " ".join(list(getattr(survey, "likely_reusable_files", []) or [])),
                " ".join(list(getattr(survey, "protocol_clues", []) or [])),
            ]
        )
        summaries.append(
            {
                "ref_id": ref_id,
                "repository_url": str(getattr(survey, "repository_url", "") or ""),
                "local_repo_path": str(getattr(survey, "local_repo_path", "") or ""),
                "repository_origin": str(getattr(survey, "repository_origin", "") or ""),
                "top_python_files": list(getattr(survey, "top_python_files", []) or [])[:12],
                "likely_reusable_files": list(getattr(survey, "likely_reusable_files", []) or [])[:12],
                "protocol_clues": list(getattr(survey, "protocol_clues", []) or [])[:12],
                "readme_summary": _shorten(getattr(survey, "readme_summary", ""), 900),
                "file_tree_summary": _shorten(getattr(survey, "file_tree_summary", ""), 700),
                "_score": _text_score(task_terms, survey_text),
            }
        )
        for symbol in list(getattr(survey, "symbol_evidence", []) or []):
            symbol_name = str(getattr(symbol, "symbol_name", "") or "")
            file_path = str(getattr(symbol, "file_path", "") or "")
            reusable_text = " ".join(
                [
                    symbol_name,
                    file_path,
                    str(getattr(symbol, "symbol_kind", "") or ""),
                    str(getattr(symbol, "snippet", "") or ""),
                ]
            )
            score = _text_score(task_terms, reusable_text)
            if score <= 0 and selected_ids:
                score = 1
            if score <= 0:
                continue
            symbol_rows.append(
                {
                    "ref_id": ref_id,
                    "file_path": file_path,
                    "symbol_name": symbol_name,
                    "symbol_kind": str(getattr(symbol, "symbol_kind", "") or ""),
                    "snippet": _shorten(getattr(symbol, "snippet", ""), 600),
                    "_score": score,
                }
            )
    summaries = sorted(summaries, key=lambda item: (-int(item.pop("_score", 0)), item.get("ref_id", "")))[:4]
    symbol_rows = sorted(symbol_rows, key=lambda item: (-int(item.pop("_score", 0)), item.get("ref_id", ""), item.get("file_path", "")))[:12]
    return {
        "reference_surveys": summaries,
        "symbol_evidence": symbol_rows,
    }


def _paper_context_for_task(task: TaskItem, file_plan: Any, state: PaperBenchReproState, plan_node_by_id: dict[str, object]) -> dict[str, Any]:
    task_terms = _task_search_terms(task, plan_node_by_id)
    scored_chunks: list[tuple[int, int, dict[str, Any]]] = []
    for chunk in list(state.paper_chunks or []):
        text = str(getattr(chunk, "text", "") or "")
        score = _text_score(task_terms, getattr(chunk, "section_title", ""), text)
        if score <= 0:
            continue
        scored_chunks.append(
            (
                -score,
                int(getattr(chunk, "ordinal", 0) or 0),
                {
                    "chunk_id": str(getattr(chunk, "chunk_id", "") or ""),
                    "section_title": str(getattr(chunk, "section_title", "") or ""),
                    "source_path": str(getattr(chunk, "source_path", "") or ""),
                    "char_start": int(getattr(chunk, "char_start", 0) or 0),
                    "char_end": int(getattr(chunk, "char_end", 0) or 0),
                    "text_preview": _shorten(text, 1200),
                },
            )
        )
    selected = [item for _, _, item in sorted(scored_chunks)[:5]]
    if not selected:
        selected = [
            {
                "chunk_id": str(getattr(chunk, "chunk_id", "") or ""),
                "section_title": str(getattr(chunk, "section_title", "") or ""),
                "source_path": str(getattr(chunk, "source_path", "") or ""),
                "char_start": int(getattr(chunk, "char_start", 0) or 0),
                "char_end": int(getattr(chunk, "char_end", 0) or 0),
                "text_preview": _shorten(getattr(chunk, "text", ""), 900),
            }
            for chunk in list(state.paper_chunks or [])[:3]
        ]
    return {"paper_chunks": selected}


def _resource_context_for_task(task: TaskItem, file_plan: Any, state: PaperBenchReproState, plan_node_by_id: dict[str, object]) -> dict[str, Any]:
    manifest = dict(state.temp_data.get("resource_manifest", {}) or {})
    if not manifest:
        return {}
    task_terms = _task_search_terms(task, plan_node_by_id)
    selected: dict[str, Any] = {}
    for key, value in manifest.items():
        if isinstance(value, list):
            rows = []
            for item in value:
                score = _text_score(task_terms, str(key), str(item))
                if score > 0 or key in {"resources", "datasets", "external_dependencies"}:
                    rows.append(item)
            if rows:
                selected[key] = rows[:12]
        elif isinstance(value, dict):
            if _text_score(task_terms, str(key), str(value)) > 0 or key in {"resources", "datasets", "external_dependencies"}:
                selected[key] = value
        elif str(value).strip() and _text_score(task_terms, str(key), str(value)) > 0:
            selected[key] = value
    return {"resource_manifest": selected}


def _generation_context_for_task(
    *,
    task: TaskItem,
    file_plan: Any,
    reference_ids: list[str],
    state: PaperBenchReproState,
    survey_by_ref_id: dict[str, object],
    plan_node_by_id: dict[str, object],
) -> dict[str, Any]:
    return {
        **_paper_context_for_task(task, file_plan, state, plan_node_by_id),
        **_reference_context_for_task(
            task=task,
            file_plan=file_plan,
            reference_ids=reference_ids,
            state=state,
            survey_by_ref_id=survey_by_ref_id,
            plan_node_by_id=plan_node_by_id,
        ),
        **_resource_context_for_task(task, file_plan, state, plan_node_by_id),
    }


def _contract_item_names(contract: dict[str, object], category: str) -> list[str]:
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


def _paper_claim_inventory_for_state(state: PaperBenchReproState) -> dict[str, list[str]]:
    if semantic_anchor_disabled():
        return {}
    gate = dict(state.temp_data.get("prepare_quality_gate", {}) or {})
    unit_quality = dict(gate.get("unit_quality", {}) or {})
    contract = dict(unit_quality.get("evidence_contract", {}) or {})
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
    return {
        category: _contract_item_names(contract, category)[:80]
        for category in categories
    }


def _formula_algorithm_contract_for_state(state: PaperBenchReproState) -> dict[str, Any]:
    if semantic_anchor_disabled():
        return {}
    evidence_contract = _paper_evidence_contract_payload_for_generation(state)
    return dict(evidence_contract.get("formula_algorithm_contract", {}) or {})


def _paper_claim_closure_for_state(state: PaperBenchReproState) -> dict[str, Any]:
    if semantic_anchor_disabled():
        return {
            "items": [],
            "count": 0,
            "rules": [],
        }
    inventory = _paper_claim_inventory_for_state(state)
    closure_items = [
        {"category": category, "name": name}
        for category, names in inventory.items()
        for name in list(names or [])
        if str(name or "").strip()
    ]
    return {
        "items": closure_items[:240],
        "count": len(closure_items),
        "rules": [
            "Every closure item must remain visible by exact name or clear alias from unit/work package into code/config/artifact routes.",
            "README-only, registry-only, detached JSON-only, and smoke-only mentions do not close method, dataset, metric, training, baseline, or table/figure obligations.",
            "If an item is bounded or config-represented, keep the exact item, owner package, code surface, artifact target, and positive implementation-scope boundary visible.",
        ],
    }


def _prepare_quality_gate_summary_for_state(state: PaperBenchReproState) -> dict[str, Any]:
    gate = dict(state.temp_data.get("prepare_quality_gate", {}) or {})
    unit_quality = dict(gate.get("unit_quality", {}) or {})
    claim_coverage = dict(unit_quality.get("claim_inventory_coverage", {}) or {})
    return {
        "schema_version": str(gate.get("schema_version", "") or ""),
        "passed": bool(gate.get("passed", False)),
        "active_unit_count": int(gate.get("active_unit_count", 0) or 0),
        "claim_coverage_ratio": claim_coverage.get("coverage_ratio"),
        "artifact_coverage_ratio": claim_coverage.get("artifact_coverage_ratio"),
        "blocking_reasons": list(gate.get("blocking_reasons", []) or [])[:20],
        "warnings": list(gate.get("warnings", []) or [])[:16],
    }


def _reference_grounding_marker(ref_id: str, source_path: str = "") -> str:
    source = str(source_path or "reference_impl").strip().replace("\\", "/")
    return f"reference_grounding: {str(ref_id or '').strip()} {source}".strip()


def _dedupe_snippet_candidates(items: list[ReferenceSnippetCandidate], *, limit: int = 6) -> list[ReferenceSnippetCandidate]:
    selected: list[ReferenceSnippetCandidate] = []
    seen: set[tuple[str, str, str]] = set()
    for item in items:
        key = (
            str(item.ref_id or ""),
            str(item.reusable_module or ""),
            str(item.code_snippet or "")[:180],
        )
        if key in seen:
            continue
        seen.add(key)
        selected.append(item)
        if len(selected) >= limit:
            break
    return selected


def _derive_reference_candidates_for_task(
    *,
    task,
    file_plan,
    state: PaperBenchReproState,
    plan_node_by_id: dict[str, object],
) -> list[str]:
    explicit = sorted(
        {
            str(ref_id or "").strip()
            for ref_id in [
                *(list(file_plan.reference_ids) if file_plan is not None else []),
                *(list(task.reference_ids) if task is not None else []),
            ]
            if str(ref_id or "").strip()
        }
    )
    actionable_ids = {
        str(getattr(item, "ref_id", "") or "").strip()
        for item in list(getattr(getattr(state, "reference_selection", None), "actionable_references", []) or [])
        if str(getattr(item, "ref_id", "") or "").strip()
    }
    explicit_actionable = [ref_id for ref_id in explicit if ref_id in actionable_ids]
    if explicit_actionable:
        return explicit_actionable

    related_node_refs = sorted(
        {
            str(getattr(plan_node_by_id.get(node_id), "ref_id", "") or "").strip()
            for node_id in list(task.related_node_ids or [])
            if str(getattr(plan_node_by_id.get(node_id), "ref_id", "") or "").strip() in actionable_ids
        }
    )
    if related_node_refs:
        return related_node_refs

    work_package_id = str(getattr(file_plan, "work_package_id", "") or "").strip()
    if work_package_id:
        bundle = next(
            (
                item
                for item in list(getattr(state, "evidence_bundles", []) or [])
                if str(getattr(item, "work_package_id", "") or "").strip() == work_package_id
            ),
            None,
        )
        if bundle is not None:
            bundle_refs = sorted(
                {
                    str(getattr(link, "ref_id", "") or "").strip()
                    for link in list(getattr(bundle, "evidence_links", []) or [])
                    if str(getattr(link, "ref_id", "") or "").strip() in actionable_ids
                }
            )
            if bundle_refs:
                return bundle_refs
        work_package = next(
            (
                item
                for item in list(getattr(getattr(state, "repo_plan", None), "work_packages", []) or [])
                if str(getattr(item, "work_package_id", "") or "").strip() == work_package_id
            ),
            None,
        )
        if work_package is not None:
            work_package_refs = [
                str(ref_id or "").strip()
                for ref_id in list(getattr(work_package, "reference_ids", []) or [])
                if str(ref_id or "").strip() in actionable_ids
            ]
            if work_package_refs:
                return list(dict.fromkeys(work_package_refs))

    architecture_refs = [
        str(ref_id or "").strip()
        for ref_id in list(getattr(getattr(state, "repo_plan", None).architecture if getattr(state, "repo_plan", None) is not None else None, "architecture_reference_ids", []) or [])
        if str(ref_id or "").strip() in actionable_ids
    ]
    if architecture_refs:
        return list(dict.fromkeys(architecture_refs))

    return explicit


def _fallback_snippet_candidates_from_survey(
    *,
    task,
    reference_ids: list[str],
    survey_by_ref_id: dict[str, object],
    plan_node_by_id: dict[str, object],
) -> list[ReferenceSnippetCandidate]:
    """Build fallback snippet candidates from cached repo surveys when Stage-3 is ungrounded."""
    task_terms = _task_search_terms(task, plan_node_by_id)
    candidates: list[tuple[int, ReferenceSnippetCandidate]] = []

    for ref_id in reference_ids:
        survey = survey_by_ref_id.get(ref_id)
        if survey is None:
            continue
        for coverage in getattr(survey, "requirement_coverage", []):
            if not coverage.code_snippets:
                continue
            coverage_terms = _tokenize_reference_text(
                coverage.title,
                coverage.scope,
                *coverage.matched_keywords,
                *coverage.matched_files,
            )
            overlap = len(task_terms.intersection(coverage_terms))
            if overlap <= 0 and task_terms:
                continue
            matched_file = coverage.matched_files[0] if coverage.matched_files else ""
            snippet = coverage.code_snippets[0]
            candidates.append(
                (
                    overlap,
                    ReferenceSnippetCandidate(
                        ref_id=ref_id,
                        repository_url=getattr(survey, "repository_url", ""),
                        reusable_module=_derive_reusable_module_name(matched_file),
                        code_snippet=snippet,
                        insight=(
                            f"Adapt the matched reference implementation from `{matched_file or getattr(survey, 'title', ref_id)}` "
                            f"to satisfy requirement `{coverage.requirement_id}` ({coverage.title or coverage.scope}). "
                            f"Record `{_reference_grounding_marker(ref_id, matched_file)}` in the generated file near the adapted code/config."
                        ),
                        supported_task_ids=[task.task_id],
                        supported_file_paths=[task.file_path],
                    ),
                )
            )

    candidates.sort(key=lambda item: (-item[0], item[1].reusable_module, item[1].ref_id))
    return [item[1] for item in candidates[:4]]


def _snippet_candidates_from_evidence_bundle(
    *,
    task,
    reference_ids: list[str],
    state: PaperBenchReproState,
) -> list[ReferenceSnippetCandidate]:
    """Carry deterministic evidence bundle links directly into generation tasks."""
    work_package_id = str(getattr(task, "work_package_id", "") or "").strip()
    if not work_package_id:
        return []
    allowed_refs = {str(ref_id or "").strip() for ref_id in reference_ids if str(ref_id or "").strip()}
    reference_by_id = {
        str(getattr(item, "ref_id", "") or "").strip(): item
        for item in list(getattr(getattr(state, "reference_selection", None), "actionable_references", []) or [])
        if str(getattr(item, "ref_id", "") or "").strip()
    }
    bundle = next(
        (
            item
            for item in list(getattr(state, "evidence_bundles", []) or [])
            if str(getattr(item, "work_package_id", "") or "").strip() == work_package_id
        ),
        None,
    )
    if bundle is None:
        return []

    candidates: list[tuple[float, ReferenceSnippetCandidate]] = []
    for link in list(getattr(bundle, "evidence_links", []) or []):
        ref_id = str(getattr(link, "ref_id", "") or "").strip()
        if not ref_id or (allowed_refs and ref_id not in allowed_refs):
            continue
        snippet = str(getattr(link, "snippet_preview", "") or "").strip()
        file_path = str(getattr(link, "file_path", "") or "").strip()
        if not snippet and not file_path:
            continue
        reference = reference_by_id.get(ref_id)
        marker = _reference_grounding_marker(ref_id, file_path)
        candidates.append(
            (
                float(getattr(link, "confidence", 0.0) or 0.0),
                ReferenceSnippetCandidate(
                    ref_id=ref_id,
                    repository_url=getattr(reference, "repository_url", "") if reference is not None else "",
                    reusable_module=_derive_reusable_module_name(file_path),
                    code_snippet=snippet,
                    insight=(
                        f"Adapt grounded reference evidence from `{ref_id}:{file_path}` for work package `{work_package_id}`. "
                        f"Preserve the protocol/config intent, not necessarily the exact source text. "
                        f"Record `{marker}` in the generated file near the adapted implementation or registry."
                    ),
                    supported_task_ids=[task.task_id],
                    supported_file_paths=[task.file_path],
                ),
            )
        )
    candidates.sort(key=lambda item: (-item[0], item[1].ref_id, item[1].reusable_module))
    return [item[1] for item in candidates[:4]]


def _topological_file_order(architecture: ArchitectureOutput) -> list[str]:
    """Compute a stable topological order over generated files."""
    file_paths = list(dict.fromkeys(architecture.target_file_tree))
    indegree = {path: 0 for path in file_paths}
    adjacency: dict[str, list[str]] = {path: [] for path in file_paths}

    for edge in architecture.dependency_graph:
        source = edge.source_path
        target = edge.target_path
        if source not in indegree:
            indegree[source] = 0
            adjacency.setdefault(source, [])
            file_paths.append(source)
        if target not in indegree:
            indegree[target] = 0
            adjacency.setdefault(target, [])
            file_paths.append(target)
        adjacency.setdefault(target, []).append(source)
        indegree[source] += 1

    ordered: list[str] = []
    queue = [path for path in file_paths if indegree.get(path, 0) == 0]
    seen: set[str] = set()

    while queue:
        current = queue.pop(0)
        if current in seen:
            continue
        seen.add(current)
        ordered.append(current)
        for dependent in adjacency.get(current, []):
            indegree[dependent] -= 1
            if indegree[dependent] <= 0:
                queue.append(dependent)

    for path in file_paths:
        if path not in seen:
            ordered.append(path)
    return ordered

def _build_generation_manifest(state: PaperBenchReproState) -> GenerationManifest:
    """Prepare generation-time task/reference context from plan and architecture artifacts."""
    architecture = (
        state.repo_plan.architecture
        if state.repo_plan is not None and state.repo_plan.architecture.target_file_tree
        else state.architecture
    )
    if architecture is None or state.reference_selection is None or state.pipeline_plan is None or state.repo_plan is None:
        raise ValueError("generation manifest requires repo_plan, architecture, reference selection, and pipeline plan")

    task_list = [
        TaskItem(
            task_id=file_plan.task_id or file_plan.target_file,
            file_path=file_plan.target_file,
            work_package_id=file_plan.work_package_id,
            purpose=file_plan.purpose,
            hypothesis=file_plan.hypothesis,
            decision_value=file_plan.decision_value,
            scope_boundary=sanitize_scope_boundary(file_plan.scope_boundary),
            related_node_ids=list(file_plan.related_node_ids),
            reference_ids=list(file_plan.reference_ids),
            depends_on=list(file_plan.depends_on),
            blocking_dependencies=list(file_plan.blocking_dependencies),
            requires_stable_dependencies=file_plan.requires_stable_dependencies,
            interface_contract=list(file_plan.interface_contract),
            implementation_surfaces=list(file_plan.implementation_surfaces),
            method_obligations=list(file_plan.method_obligations),
            defines_symbols=list(file_plan.defines_symbols),
            calls_symbols=list(file_plan.calls_symbols),
            writes_artifacts=list(file_plan.writes_artifacts),
            reads_artifacts=list(file_plan.reads_artifacts),
            allowed_scope=dict(file_plan.allowed_scope),
            review_points=list(file_plan.review_points),
        )
        for file_plan in state.repo_plan.files
    ]

    task_by_path = {item.file_path: item for item in task_list}
    task_by_id = {item.task_id: item for item in task_list}
    file_plan_by_path = {
        item.target_file: item
        for item in state.repo_plan.files
    }
    reference_by_id = {item.ref_id: item for item in state.reference_selection.actionable_references}
    survey_by_ref_id = {item.ref_id: item for item in state.reference_repo_surveys}
    plan_node_by_id = {item.node_id: item for item in state.pipeline_plan.plan_nodes}
    snippet_index: dict[tuple[str, str], dict] = {}
    for node in state.pipeline_plan.plan_nodes:
        key = (node.ref_id, node.reusable_module)
        if not node.ref_id or not node.reusable_module or not node.traceable or not node.code_snippet.strip():
            continue
        snippet_index.setdefault(key, {
            "ref_id": node.ref_id,
            "reusable_module": node.reusable_module,
            "code_snippet": node.code_snippet,
            "insight": node.insight,
            "related_node_ids": [],
        })
        snippet_index[key]["related_node_ids"].append(node.node_id)

    # Repo generation is driven by the canonical repo_plan, not by every path that
    # appeared in an upstream architecture draft.  Architecture may still carry
    # helper paths that were later filtered out by canonical IR validation; those
    # paths should not make the generate stage fail after repo_plan has already
    # selected the authoritative task set.
    missing_task_paths = sorted(set(architecture.target_file_tree) - set(task_by_path))
    if missing_task_paths:
        backlog = state.temp_data.setdefault("degraded_backlog", [])
        payload = {
            "stage": "generation_manifest",
            "code": "architecture_paths_not_in_repo_plan",
            "message": "ignored architecture paths that are not canonical repo_plan generation tasks",
            "paths": missing_task_paths,
        }
        if isinstance(backlog, list) and payload not in backlog:
            backlog.append(payload)

    topo_paths = [item.target_file for item in state.repo_plan.files]
    ordered_tasks = [
        task_by_path[path].task_id
        for path in topo_paths
        if path in task_by_path
    ]
    if len(ordered_tasks) != len(task_list):
        unordered_task_ids = sorted(set(task_by_id) - set(ordered_tasks))
        ordered_tasks.extend(unordered_task_ids)

    task_inputs: list[GenerationTaskInput] = []
    global_review_points: list[str] = []
    paper_claim_inventory = _paper_claim_inventory_for_state(state)
    paper_claim_closure = _paper_claim_closure_for_state(state)
    paper_evidence_contract = _paper_evidence_contract_payload_for_generation(state)
    formula_algorithm_contract = _formula_algorithm_contract_for_state(state)
    prepare_gate_summary = _prepare_quality_gate_summary_for_state(state)

    for task_id in ordered_tasks:
        task = task_by_id[task_id]
        file_plan = file_plan_by_path.get(task.file_path)
        dependency_files = sorted(dict.fromkeys(file_plan.depends_on if file_plan else task.depends_on))
        related_node_ids = sorted(set(file_plan.related_node_ids if file_plan else task.related_node_ids))
        reference_ids = _derive_reference_candidates_for_task(
            task=task,
            file_plan=file_plan,
            state=state,
            plan_node_by_id=plan_node_by_id,
        )
        snippet_candidates: list[ReferenceSnippetCandidate] = []
        snippet_keys = {
            (node.ref_id, node.reusable_module)
            for node_id in related_node_ids
            for node in ([plan_node_by_id[node_id]] if node_id in plan_node_by_id else [])
            if node.traceable and node.ref_id and node.reusable_module and node.code_snippet.strip()
        }
        for ref_id, module_name in sorted(snippet_keys):
            reference = reference_by_id.get(ref_id)
            snippet_payload = snippet_index.get((ref_id, module_name))
            if snippet_payload is None:
                continue
            snippet_candidates.append(
                ReferenceSnippetCandidate(
                    ref_id=ref_id,
                    repository_url=reference.repository_url if reference else "",
                    reusable_module=module_name,
                    code_snippet=snippet_payload["code_snippet"],
                    insight=snippet_payload["insight"],
                    supported_task_ids=[task.task_id],
                    supported_file_paths=[task.file_path],
                )
            )
        snippet_candidates.extend(
            _snippet_candidates_from_evidence_bundle(
                task=task,
                reference_ids=reference_ids,
                state=state,
            )
        )
        if not snippet_candidates:
            snippet_candidates.extend(
                _fallback_snippet_candidates_from_survey(
                    task=task,
                    reference_ids=reference_ids,
                    survey_by_ref_id=survey_by_ref_id,
                    plan_node_by_id=plan_node_by_id,
                )
            )
        snippet_candidates = _dedupe_snippet_candidates(snippet_candidates, limit=6)
        reference_ids = sorted(set(reference_ids) | {item.ref_id for item in snippet_candidates})
        review_points = list(dict.fromkeys(file_plan.review_points if file_plan else task.review_points))
        global_review_points.extend(review_points)
        task_payload = sanitize_task_contract(
            {
                "task_id": task.task_id,
                "file_path": task.file_path,
                "work_package_id": (file_plan.work_package_id if file_plan else task.work_package_id),
                "dependency_files": dependency_files,
                "related_node_ids": related_node_ids,
                "reference_ids": reference_ids,
                "snippet_candidates": snippet_candidates,
                "interface_contract": list(file_plan.interface_contract if file_plan else task.interface_contract),
                "implementation_surfaces": list(file_plan.implementation_surfaces if file_plan else task.implementation_surfaces),
                "method_obligations": list(file_plan.method_obligations if file_plan else task.method_obligations),
                "defines_symbols": list(file_plan.defines_symbols if file_plan else task.defines_symbols),
                "calls_symbols": list(file_plan.calls_symbols if file_plan else task.calls_symbols),
                "writes_artifacts": list(file_plan.writes_artifacts if file_plan else task.writes_artifacts),
                "reads_artifacts": list(file_plan.reads_artifacts if file_plan else task.reads_artifacts),
                "hypothesis": str(getattr(file_plan, "hypothesis", "") or getattr(task, "hypothesis", "") or ""),
                "decision_value": str(getattr(file_plan, "decision_value", "") or getattr(task, "decision_value", "") or ""),
                "generation_prompt": str(getattr(file_plan, "generation_prompt", "") or "") if file_plan else "",
                "context_sources": list(file_plan.context_sources if file_plan else []),
                "allowed_scope": dict(file_plan.allowed_scope) if file_plan else dict(task.allowed_scope),
                "scope_boundary": sanitize_scope_boundary(
                    getattr(file_plan, "scope_boundary", {}) or getattr(task, "scope_boundary", {}) or {}
                ),
                "review_points": review_points,
                "paper_claim_inventory": paper_claim_inventory,
                "paper_claim_closure_items": paper_claim_closure["items"],
                "paper_claim_closure_rules": paper_claim_closure["rules"],
                "paper_evidence_contract": paper_evidence_contract,
                "formula_algorithm_contract": formula_algorithm_contract,
                "prepare_quality_gate_summary": prepare_gate_summary,
                "generation_context": _generation_context_for_task(
                    task=task,
                    file_plan=file_plan,
                    reference_ids=reference_ids,
                    state=state,
                    survey_by_ref_id=survey_by_ref_id,
                    plan_node_by_id=plan_node_by_id,
                ),
            }
        )
        task_payload["review_points"] = sanitize_contract_list(
            list(task_payload.get("review_points", []) or []),
            field="review_points",
        )
        task_inputs.append(
            GenerationTaskInput(
                **task_payload,
            )
        )

    return GenerationManifest(
        ordered_tasks=ordered_tasks,
        task_inputs=task_inputs,
        review_points=sanitize_contract_list(list(dict.fromkeys(global_review_points)), field="review_points"),
        tasks=task_list,
        edges=list(architecture.dependency_graph),
        topological_order=topo_paths,
    )
