"""File provenance registry helpers for reproagent."""

from __future__ import annotations

from typing import Any

from reproagent.pipeline.schemas import PaperBenchReproState, FileProvenanceRecord, ValidationCheck
from reproagent.pipeline.utils.intent_contract import upstream_intent_payload


def _normalize_repo_path(path: str) -> str:
    normalized = str(path or "").strip().replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    while "//" in normalized:
        normalized = normalized.replace("//", "/")
    return normalized.strip("/")


def _dedupe(values: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for item in values:
        value = str(item or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _validation_checks_for_path(state: PaperBenchReproState, path: str) -> list[str]:
    checks: list[ValidationCheck] = []
    if state.validation_report is not None:
        checks.extend(list(state.validation_report.artifact_checks))
        checks.extend(list(state.validation_report.implementation_checks))
        checks.extend(list(state.validation_report.semantic_checks))
        checks.extend(list(state.validation_report.trace_checks))
        checks.extend(list(state.validation_report.integration_checks))
    normalized_path = _normalize_repo_path(path)
    matched: list[str] = []
    for check in checks:
        affected_files = {
            _normalize_repo_path(item)
            for item in list(check.affected_files or [])
            if _normalize_repo_path(item)
        }
        if normalized_path in affected_files and check.name:
            matched.append(str(check.name))
    return _dedupe(matched)


def build_file_provenance_records(state: PaperBenchReproState) -> list[FileProvenanceRecord]:
    """Build a compact provenance registry for generated repository files."""
    repo_plan = state.repo_plan
    if repo_plan is None:
        return []

    file_plans = list(repo_plan.files or [])
    path_to_plan = {
        _normalize_repo_path(item.target_file): item
        for item in file_plans
        if _normalize_repo_path(item.target_file)
    }
    work_package_by_id = {
        str(item.work_package_id or "").strip(): item
        for item in list(repo_plan.work_packages or [])
        if str(item.work_package_id or "").strip()
    }
    plan_nodes = {
        str(item.node_id or "").strip(): item
        for item in list(state.pipeline_plan.plan_nodes if state.pipeline_plan else [])
        if str(item.node_id or "").strip()
    }
    expected_artifacts = {
        _normalize_repo_path(path): _normalize_repo_path(path)
        for path in list(upstream_intent_payload(state).get("expected_artifacts", []) or [])
        if _normalize_repo_path(path)
    }
    generation_refs_by_path: dict[str, list[str]] = {}
    generation_notes_by_path: dict[str, list[str]] = {}
    if state.generation_manifest is not None:
        for task in list(state.generation_manifest.task_inputs or []):
            task_path = _normalize_repo_path(task.file_path)
            if not task_path:
                continue
            refs = list(task.reference_ids or []) + [
                str(candidate.ref_id or "")
                for candidate in list(task.snippet_candidates or [])
                if str(candidate.ref_id or "").strip()
            ]
            generation_refs_by_path[task_path] = _dedupe(generation_refs_by_path.get(task_path, []) + refs)
            if list(task.snippet_candidates or []):
                generation_notes_by_path.setdefault(task_path, []).append("reference_snippets_provided_to_generation")
    generated_paths = _dedupe([_normalize_repo_path(path) for path in list(state.generated_files or []) if _normalize_repo_path(path)])
    registered_paths = _dedupe(list(path_to_plan.keys()) + generated_paths)

    records: list[FileProvenanceRecord] = []
    for path in registered_paths:
        file_plan = path_to_plan.get(path)
        owner_work_package_ids = _dedupe(
            [str(file_plan.work_package_id or "").strip()] if file_plan is not None and str(file_plan.work_package_id or "").strip() else []
        )
        owner_work_package = work_package_by_id.get(owner_work_package_ids[0], None) if owner_work_package_ids else None
        related_plan_node_ids = _dedupe(list(file_plan.related_node_ids or [])) if file_plan is not None else []
        source_requirement_ids = _dedupe(
            [
                requirement_id
                for node_id in related_plan_node_ids
                for requirement_id in list(getattr(plan_nodes.get(node_id), "requirement_ids", []) or [])
                if str(requirement_id or "").strip()
            ]
            + list(getattr(owner_work_package, "requirement_ids", []) or [])
        )
        source_claim_ids = _dedupe(
            list(getattr(file_plan, "validation_hooks", []) or [])
            + list(getattr(file_plan, "review_points", []) or [])
        ) if file_plan is not None else []
        record = FileProvenanceRecord(
            path=path,
            generated=path in set(generated_paths),
            source_requirement_ids=source_requirement_ids,
            source_claim_ids=source_claim_ids,
            source_reference_ids=_dedupe(
                (list(getattr(file_plan, "reference_ids", []) or []) if file_plan is not None else [])
                + generation_refs_by_path.get(path, [])
            ),
            owned_unit_ids=_dedupe(list(getattr(file_plan, "owned_units", []) or [])) if file_plan is not None else [],
            owner_work_package_ids=owner_work_package_ids,
            related_plan_node_ids=related_plan_node_ids,
            produced_artifacts=_dedupe(list(getattr(file_plan, "writes_artifacts", []) or [])) if file_plan is not None else [],
            expected_artifacts=[
                artifact
                for artifact in list(expected_artifacts.values())
                if artifact == path or artifact in set(_dedupe(list(getattr(file_plan, "writes_artifacts", []) or [])))
            ],
            validation_checks=_validation_checks_for_path(state, path),
            provenance_notes=_dedupe(
                [
                    "registered_in_repo_plan" if file_plan is not None else "",
                    "materialized_in_repo" if path in set(generated_paths) else "planned_only",
                    "tracked_as_expected_artifact" if path in expected_artifacts else "",
                    *generation_notes_by_path.get(path, []),
                ]
            ),
        )
        records.append(record)

    records.sort(key=lambda item: item.path)
    return records


def refresh_file_provenance(state: PaperBenchReproState) -> list[dict[str, Any]]:
    """Refresh state.temp_data cache with JSON-ready provenance entries."""
    records = [item.model_dump(mode="json") for item in build_file_provenance_records(state)]
    state.temp_data["file_provenance"] = records
    return records
