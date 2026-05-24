"""Generate-stage implementations for reproagent workflow."""

from __future__ import annotations

import logging
import json
import re
from typing import Any, Callable

from reproagent.pipeline.config import get_codegen_config, get_workflow_config, semantic_anchor_disabled
from reproagent.pipeline.schemas import (
    EvaluationDecision,
    ExecutionResult,
    GenerationCheckpoint,
    PaperBenchReproState,
)
from reproagent.sandbox import get_sandbox_provider

from reproagent.pipeline.utils.repair_helpers import materialize_selected_tasks
from reproagent.pipeline.utils.intent_contract import (
    paperbench_prompt_safe_experiment_design,
    upstream_intent_payload,
)
from reproagent.pipeline.utils.file_provenance import refresh_file_provenance
from reproagent.pipeline.utils.handoff_contract import build_stage1_repo_contract, canonical_main_entry
from reproagent.pipeline.utils.quality_status import is_validated_repo_handoff_ready

logger = logging.getLogger(__name__)


def _looks_like_repo_file_path(value: object) -> bool:
    path = str(value or "").strip().replace("\\", "/")
    if not path or path.startswith("results/") or path.endswith("/"):
        return False
    if any(char.isspace() for char in path):
        return False
    if path.startswith(("/", "-", "$")):
        return False
    if "://" in path:
        return False
    return "." in path.rsplit("/", 1)[-1]


def _planned_repo_files(state: PaperBenchReproState) -> set[str]:
    """Return planned repository files that should exist after generation."""
    planned: set[str] = set()

    def add_path(value: object) -> None:
        path = str(value or "").strip().replace("\\", "/")
        if not _looks_like_repo_file_path(path):
            return
        planned.add(path)

    if state.generation_manifest is not None:
        for item in list(state.generation_manifest.task_inputs or []):
            add_path(getattr(item, "file_path", ""))
        for item in list(state.generation_manifest.topological_order or []):
            add_path(item)
    if state.project_plan is not None:
        for item in list(state.project_plan.file_specs or []):
            if bool(getattr(item, "required", True)):
                add_path(getattr(item, "path", ""))
        for value in dict(state.project_plan.entrypoints or {}).values():
            add_path(value)
    if state.repo_plan is not None:
        for item in list(state.repo_plan.files or []):
            add_path(getattr(item, "target_file", ""))
        for value in list(state.repo_plan.entrypoints or []):
            add_path(value)
        route_entry = _explicit_main_entry(state)
        if route_entry:
            add_path(route_entry)

    return planned


def _explicit_main_entry(state: PaperBenchReproState) -> str:
    """Return a main entry only when the plan explicitly declares one."""
    if state.repo_plan is not None:
        candidate = str(getattr(state.repo_plan.canonical_route, "entry_surface", "") or "").strip()
        if candidate:
            return candidate
    if state.project_plan is not None:
        candidate = str(dict(state.project_plan.entrypoints or {}).get("main", "") or "").strip()
        if candidate:
            return candidate
    return ""


_RUNTIME_ARTIFACT_DECLARATION_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".npy",
    ".npz",
    ".pickle",
    ".pkl",
    ".pt",
    ".pth",
    ".safetensors",
}


def _file_suffix(path: str) -> str:
    name = str(path or "").strip().replace("\\", "/").rsplit("/", 1)[-1].lower()
    if "." not in name:
        return ""
    return "." + name.rsplit(".", 1)[-1]


def _is_runtime_artifact_declaration_candidate(path: str) -> bool:
    return _file_suffix(path) in _RUNTIME_ARTIFACT_DECLARATION_SUFFIXES


def _task_input_payloads_by_file(state: PaperBenchReproState) -> dict[str, dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    if state.generation_manifest is None:
        return payloads
    for item in list(state.generation_manifest.task_inputs or []):
        path = str(getattr(item, "file_path", "") or "").strip().replace("\\", "/")
        if not path:
            continue
        try:
            payload = item.model_dump(mode="json")
        except Exception:
            payload = {
                "task_id": str(getattr(item, "task_id", "") or ""),
                "file_path": path,
                "work_package_id": str(getattr(item, "work_package_id", "") or ""),
                "purpose": str(getattr(item, "purpose", "") or ""),
                "writes_artifacts": list(getattr(item, "writes_artifacts", []) or []),
            }
        payloads[path] = payload
    return payloads


def _runtime_artifact_declaration_content(
    *,
    path: str,
    task_payload: dict[str, Any],
) -> str:
    payload = {
        "artifact_type": "runtime_generated_artifact_declaration",
        "artifact_path": path,
        "producer_task_id": str(task_payload.get("task_id") or ""),
        "work_package_id": str(task_payload.get("work_package_id") or ""),
        "purpose": str(task_payload.get("purpose") or f"Materialize runtime artifact {path}"),
        "producer_obligations": [
            str(item)
            for item in list(task_payload.get("method_obligations") or [])[:24]
            if str(item).strip()
        ],
        "declared_outputs": [
            str(item)
            for item in list(task_payload.get("writes_artifacts") or [])[:40]
            if str(item).strip()
        ],
        "reproduction_note": (
            "This lightweight metadata file reserves a large or binary runtime artifact path "
            "declared by the paper reproduction plan. Full training/evaluation routes in this "
            "repository are responsible for overwriting this path with the concrete artifact."
        ),
    }
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def _materialize_runtime_artifact_declarations(
    state: PaperBenchReproState,
    project_files: dict[str, str],
    *,
    allowed_repo_files: set[str],
) -> dict[str, dict[str, Any]]:
    """Add traceable declarations for planned non-text runtime artifacts.

    LLM file generation is text-oriented. Plans can still require heavyweight
    checkpoints or arrays as repo-visible artifact paths. Materializing only
    these non-text artifact declarations prevents completeness loops without
    masking missing source/config/docs files.
    """
    task_payloads = _task_input_payloads_by_file(state)
    planned = _planned_repo_files(state)
    missing_runtime_artifacts = sorted(
        path
        for path in planned
        if path in allowed_repo_files
        and path not in project_files
        and _is_runtime_artifact_declaration_candidate(path)
    )
    declarations: dict[str, dict[str, Any]] = {}
    for path in missing_runtime_artifacts:
        task_payload = task_payloads.get(path, {"file_path": path})
        project_files[path] = _runtime_artifact_declaration_content(
            path=path,
            task_payload=task_payload,
        )
        declarations[path] = {
            "artifact_type": "runtime_generated_artifact_declaration",
            "producer_task_id": str(task_payload.get("task_id") or ""),
            "work_package_id": str(task_payload.get("work_package_id") or ""),
            "suffix": _file_suffix(path),
        }
    if declarations:
        state.temp_data["runtime_artifact_declarations"] = declarations
    return declarations


_SEMANTIC_HANDOFF_GENERIC_TOKENS = {
    "ours",
    "baseline",
    "baselines",
    "model",
    "models",
    "method",
    "methods",
    "dataset",
    "datasets",
    "experiment",
    "experiments",
    "figure",
    "fig",
    "table",
    "result",
    "results",
    "loss",
    "accuracy",
    "reward",
    "return",
    "log",
    "predictions",
    "trained_model",
    "checkpoint",
    "result_table",
    "result_figure",
}


def _semantic_token_variants(token: str) -> list[str]:
    value = str(token or "").strip().lower()
    if not value:
        return []
    value = re.sub(r"\s+", " ", value)
    variants = [
        value,
        value.replace(" ", "_"),
        value.replace(" ", "-"),
        value.replace("_", " "),
        value.replace("-", " "),
        value.replace("-", "_"),
        value.replace("_", "-"),
    ]
    return list(dict.fromkeys(item for item in variants if item))


def _semantic_handoff_tokens_from_prepare(state: PaperBenchReproState) -> list[str]:
    gate = dict(state.temp_data.get("prepare_quality_gate", {}) or {})
    unit_quality = dict(gate.get("unit_quality", {}) or {})
    evidence_contract = dict(unit_quality.get("evidence_contract", {}) or {})
    categories = (
        "named_experiments",
        "environments",
        "datasets",
        "methods",
        "metrics",
        "artifacts",
        "protocol_obligations",
        "fixed_hyperparameters",
        "implementation_obligations",
    )
    raw: list[str] = []
    for category in categories:
        values = evidence_contract.get(category, [])
        if isinstance(values, dict):
            values = list(values.values())
        for value in list(values or []):
            if isinstance(value, dict):
                raw.extend(str(item or "") for item in value.values())
            else:
                raw.append(str(value or ""))
    for sweep in list(evidence_contract.get("parameter_sweeps", []) or []):
        if isinstance(sweep, dict):
            raw.append(str(sweep.get("name", "") or ""))
            raw.extend(str(item or "") for item in list(sweep.get("values", []) or []))
    tokens: list[str] = []
    for token in raw:
        normalized = str(token or "").strip()
        lowered = normalized.lower()
        if not normalized:
            continue
        if lowered in _SEMANTIC_HANDOFF_GENERIC_TOKENS:
            continue
        if len(lowered) < 4 and not any(char.isdigit() for char in lowered):
            continue
        if lowered.startswith("figure ") or lowered.startswith("table "):
            tokens.append(normalized)
            continue
        if any(char.isdigit() for char in lowered) or "_" in lowered or "-" in lowered or len(lowered) >= 5:
            tokens.append(normalized)
    return list(dict.fromkeys(tokens))[:80]


def _semantic_handoff_contract_text(state: PaperBenchReproState) -> str:
    payload: dict[str, Any] = {}
    if state.repo_plan is not None:
        payload["repo_plan"] = state.repo_plan.model_dump(mode="json")
    if state.generation_manifest is not None:
        payload["generation_manifest"] = state.generation_manifest.model_dump(mode="json")
    if state.project_plan is not None:
        payload["project_plan"] = state.project_plan.model_dump(mode="json")
    return json.dumps(payload, ensure_ascii=False, sort_keys=True).lower()


def _semantic_handoff_checksum_issues(state: PaperBenchReproState) -> list[str]:
    if semantic_anchor_disabled():
        state.temp_data["semantic_handoff_checksum"] = {
            "disabled": True,
            "checked_tokens": [],
            "missing_tokens": [],
        }
        return []
    tokens = _semantic_handoff_tokens_from_prepare(state)
    if not tokens:
        return []
    contract_text = _semantic_handoff_contract_text(state)
    missing = [
        token
        for token in tokens
        if not any(variant in contract_text for variant in _semantic_token_variants(token))
    ]
    state.temp_data["semantic_handoff_checksum"] = {
        "checked_tokens": tokens,
        "missing_tokens": missing[:40],
    }
    if missing:
        return [
            "semantic handoff checksum lost prepare evidence tokens before generate: "
            + ", ".join(missing[:24])
        ]
    return []


def _repo_completeness_issues(state: PaperBenchReproState) -> list[str]:
    """Detect incomplete generated repos before handoff/scoring."""
    planned = _planned_repo_files(state)
    generated = {str(item or "").strip().replace("\\", "/") for item in list(state.generated_files or []) if str(item or "").strip()}
    repo_root = None
    if str(state.project_root or "").strip():
        from pathlib import Path

        repo_root = Path(str(state.project_root)).resolve()
        existing = {
            path
            for path in planned
            if (repo_root / path).exists()
            and (repo_root / path).is_file()
            and (repo_root / path).stat().st_size > 0
        }
    else:
        existing = set()

    present = generated | existing
    missing = sorted(path for path in planned if path not in present)
    issues: list[str] = []
    if missing:
        issues.append(
            f"generated repo incomplete: {len(present)}/{len(planned)} planned files present; "
            f"missing planned files={missing[:16]}"
        )
    expected_task_files = {
        str(getattr(item, "file_path", "") or "").strip().replace("\\", "/")
        for item in list(state.generation_manifest.task_inputs if state.generation_manifest is not None else [])
        if _looks_like_repo_file_path(getattr(item, "file_path", ""))
    }
    missing_task_files = sorted(path for path in expected_task_files if path not in present)
    if missing_task_files:
        issues.append(
            "generation_manifest task files were not materialized: "
            + ", ".join(missing_task_files[:16])
        )
    explicit_entry = _explicit_main_entry(state)
    if explicit_entry and explicit_entry in planned and explicit_entry not in present:
        issues.append(f"generated repo missing main entrypoint: {explicit_entry}")
    source_like = [
        path
        for path in present
        if path.endswith((".py", ".yaml", ".yml", ".toml", ".md", ".txt"))
        and not path.startswith("results/")
    ]
    if len(planned) >= 6 and len(source_like) < 5:
        issues.append(
            f"generated repo has too few non-result source/config files: {len(source_like)}"
        )
    return issues


def _task_review_issue_summary(state: PaperBenchReproState) -> dict[str, Any]:
    """Summarize task-review failures without blocking file-complete generate handoff."""
    ordered_task_ids = [
        str(item or "").strip()
        for item in list(state.generation_manifest.ordered_tasks if state.generation_manifest is not None else [])
        if str(item or "").strip()
    ]
    reviewed_task_ids = {
        str(item.get("task_id", "") or "").strip()
        for item in list(state.execution_history or [])
        if isinstance(item, dict)
        and str(item.get("task_id", "") or "").strip()
        and bool(dict(item.get("task_review", {}) or {}).get("success", False))
    }
    failed_task_ids = [
        str(item.get("task_id", "") or "").strip()
        for item in list(state.execution_history or [])
        if isinstance(item, dict)
        and str(item.get("task_id", "") or "").strip()
        and not bool(dict(item.get("task_review", {}) or {}).get("success", False))
    ]
    missing_reviewed_tasks = sorted(task_id for task_id in ordered_task_ids if task_id not in reviewed_task_ids)
    return {
        "ordered_task_count": len(ordered_task_ids),
        "passed_task_count": len(reviewed_task_ids),
        "failed_task_ids": list(dict.fromkeys(item for item in failed_task_ids if item))[:100],
        "missing_or_failed_review_task_ids": missing_reviewed_tasks[:100],
    }


def _assert_repo_generation_complete(state: PaperBenchReproState, *, context: str) -> None:
    """Hard-stop incomplete generation before any handoff or scoring artifacts are reused."""
    completeness_issues = _repo_completeness_issues(state)
    if not completeness_issues:
        return
    state.status = "failed"
    state.current_node = "generate"
    state.failed_node = "generate"
    state.terminal_outcome = "failed"
    state.terminal_outcome_reason = f"{context} left repository incomplete"
    state.error_message = "; ".join(completeness_issues[:6])
    state.temp_data["terminal_outcome"] = state.terminal_outcome
    state.temp_data["terminal_outcome_reason"] = state.terminal_outcome_reason
    state.temp_data["repo_completeness_issues"] = completeness_issues
    if isinstance(state.project_manifest, dict):
        state.project_manifest = {
            **state.project_manifest,
            "repo_completeness_issues": completeness_issues,
            "missing_planned_files": sorted(_planned_repo_files(state) - set(state.generated_files or [])),
        }
    raise RuntimeError(
        f"{context} cannot finish before all planned repo files are materialized: "
        + "; ".join(completeness_issues[:6])
    )


def _mark_generate_failed(state: PaperBenchReproState, *, reason: str, issues: list[str]) -> None:
    """Record a hard generate failure before a partial repo can be consumed downstream."""
    issue_text = "; ".join(str(item).strip() for item in issues if str(item).strip())
    state.status = "failed"
    state.current_node = "generate"
    state.failed_node = "generate"
    state.terminal_outcome = "failed"
    state.terminal_outcome_reason = reason
    state.error_message = issue_text or reason
    state.temp_data["terminal_outcome"] = state.terminal_outcome
    state.temp_data["terminal_outcome_reason"] = state.terminal_outcome_reason
    state.temp_data["generate_blocking_issues"] = list(issues)


def _classify_generate_status(state: PaperBenchReproState) -> None:
    """Keep generate-only runs honest when task review or validation did not close."""
    execution_result = state.execution_result.model_dump(mode="json") if state.execution_result else {}
    validation_report = state.validation_report
    validation_passed = bool(validation_report and validation_report.passed)
    execution_passed = bool(execution_result.get("success", False))
    completeness_issues = _repo_completeness_issues(state)
    if completeness_issues:
        _mark_generate_failed(
            state,
            reason="generate stopped with incomplete repository materialization",
            issues=completeness_issues[:12],
        )
        state.temp_data["repo_completeness_issues"] = completeness_issues
        return
    handoff_ready = is_validated_repo_handoff_ready(
        dict(state.temp_data.get("validated_repo_handoff", {}) or {})
    )
    if execution_passed and validation_passed and handoff_ready:
        state.status = "completed"
        state.terminal_outcome = "completed"
        state.terminal_outcome_reason = "generate produced a validated handoff-ready reproduction repo"
        state.failed_node = ""
        state.error_message = ""
        return

    blocking_issues: list[str] = []
    if not execution_passed:
        blocking_issues.append(str(execution_result.get("error", "") or "execution_result.success is false"))
        failed_checks = [
            str(check.get("name", "") or check.get("details", "") or "").strip()
            for check in list(execution_result.get("checks", []) or [])
            if isinstance(check, dict) and not bool(check.get("passed", False))
        ]
        blocking_issues.extend([item for item in failed_checks if item][:8])
    if validation_report is None:
        blocking_issues.append("generate completed without validation_report")
    elif not validation_passed:
        blocking_issues.extend(list(validation_report.blocked_reasons or [])[:8])
        blocking_issues.extend(
            [
                f"{check.category}:{check.name}"
                for check in (
                    list(validation_report.artifact_checks)
                    + list(validation_report.implementation_checks)
                    + list(validation_report.semantic_checks)
                    + list(validation_report.trace_checks)
                    + list(validation_report.integration_checks)
                )
                if not check.passed
            ][:12]
        )
    elif not handoff_ready:
        blocking_issues.append("validated_repo_handoff.handoff_ready is false")
        handoff = dict(state.temp_data.get("validated_repo_handoff", {}) or {})
        quality = dict(handoff.get("quality_status", {}) or {})
        if quality:
            blocking_issues.append(
                "quality_status="
                + str(quality.get("quality_status", "") or "unknown")
                + ", validation_quality_level="
                + str(quality.get("validation_quality_level", "") or "unknown")
            )

    if validation_report is None:
        state.status = "completed_unverified"
        state.terminal_outcome = "completed_unverified"
        state.terminal_outcome_reason = "generate completed without a validation report"
    else:
        state.status = "completed_with_runtime_risk"
        state.terminal_outcome = "completed_with_runtime_risk"
        state.terminal_outcome_reason = "generate completed with unresolved task review or repo validation findings"

    state.current_node = "generate"
    state.failed_node = ""
    state.error_message = ""
    state.temp_data["terminal_outcome"] = state.terminal_outcome
    state.temp_data["terminal_outcome_reason"] = state.terminal_outcome_reason
    state.temp_data["generate_validation_findings"] = blocking_issues[:24]


def _canonical_main_entry(state: PaperBenchReproState) -> str:
    return canonical_main_entry(state)


def build_repo_handoff_payload(state: PaperBenchReproState, *, start_stage: str = "rapid_validation", thread_id: str = "") -> dict[str, Any]:
    """Build a validated repository handoff payload from one completed PaperBench Repro state."""
    if not state.project_root:
        raise ValueError("repository handoff requires a generated project_root")

    start_stage_value = str(start_stage or "rapid_validation").strip() or "rapid_validation"
    validated_repo_handoff = dict(state.temp_data.get("validated_repo_handoff", {}) or {})
    rapid_validation_handoff = dict(validated_repo_handoff.get("rapid_validation", {}) or {})
    repo_materialization = dict(rapid_validation_handoff.get("repo_materialization", {}) or {})
    handoff_repo_contract = dict(repo_materialization.get("repo_contract", {}) or {})
    handoff_repo_path = str(
        handoff_repo_contract.get("repo_path", "")
        or repo_materialization.get("repo_path", "")
        or repo_materialization.get("working_root", "")
        or ""
    ).strip()

    repo_contract = build_stage1_repo_contract(state, repo_path=handoff_repo_path or state.project_root)
    for key in (
        "repo_path",
        "repo_source",
        "entrypoint_hint",
        "install_command",
        "variant_mode",
        "baseline_command",
        "idea_command",
        "variant_command",
        "smoke_command",
    ):
        value = handoff_repo_contract.get(key)
        if str(value or "").strip():
            repo_contract[key] = str(value).strip()
    if isinstance(handoff_repo_contract.get("command_contract"), dict) and handoff_repo_contract.get("command_contract"):
        repo_contract["command_contract"] = dict(handoff_repo_contract["command_contract"])
    for key in ("metric_paths", "editable_paths", "protected_paths"):
        value = handoff_repo_contract.get(key)
        if isinstance(value, list) and value:
            repo_contract[key] = list(value)

    effective_repo_path = str(repo_contract.get("repo_path", "") or handoff_repo_path or state.project_root).strip()
    main_entry = str(repo_contract.get("entrypoint_hint", "") or "").strip()
    variant_mode = str(repo_contract.get("variant_mode", "") or "unknown")
    metric_paths = list(repo_contract.get("metric_paths", []) or [])
    editable_paths = list(repo_contract.get("editable_paths", []) or [])
    protected_paths = list(repo_contract.get("protected_paths", []) or [])

    reference_repositories: list[dict[str, Any]] = []
    reference_repo_preparation = dict(state.temp_data.get("reference_repo_preparation", {}) or {})
    resource_manifest = dict(state.temp_data.get("resource_manifest", {}) or {})
    for item in reference_repo_preparation.get("prepared_repositories", []) or []:
        if not isinstance(item, dict):
            continue
        reference_repositories.append(
            {
                "ref_id": str(item.get("ref_id", "")).strip(),
                "title": str(item.get("title", "")).strip(),
                "repository_url": str(item.get("repository_url", "")).strip(),
                "local_repo_path": str(item.get("local_repo_path", "")).strip(),
                "default_branch": str(item.get("default_branch", "")).strip(),
                "repository_origin": str(item.get("repository_origin", "")).strip(),
                "paper_path": str(item.get("paper_path", "")).strip(),
                "paper_url": str(item.get("paper_url", "")).strip(),
            }
        )
    if not reference_repositories:
        for item in state.input.idea_references:
            if not isinstance(item, dict):
                continue
            reference_repositories.append(
                {
                    "ref_id": str(item.get("ref_id", "")).strip(),
                    "title": str(item.get("title", "")).strip(),
                    "repository_url": str(item.get("repository_url", "")).strip(),
                    "local_repo_path": str(item.get("local_repo_path", "") or item.get("path", "")).strip(),
                    "paper_path": str(item.get("paper_path", "")).strip(),
                    "paper_url": str(item.get("paper_url", "")).strip(),
                }
            )

    payload = {
        "target": state.input.target,
        "language": state.input.language,
        "upstream_intent": upstream_intent_payload(state),
        "thread_id": thread_id or state.input.thread_id or state.run_id,
        "start_stage": start_stage_value,
        "verified_asset_root": state.project_root if start_stage_value == "full_experiment" else "",
        "experiment_design": paperbench_prompt_safe_experiment_design(state.input.experiment_design),
        "github_upload_config": dict(state.input.github_upload_config),
        "reference_repositories": reference_repositories,
        "workspace_config": {
            "project_root": state.project_root,
            "target_repo_path": effective_repo_path,
            "baseline_command": str(repo_contract.get("baseline_command", "") or ""),
            "idea_command": str(repo_contract.get("idea_command", "") or ""),
            "variant_command": str(repo_contract.get("variant_command", "") or ""),
            "smoke_command": str(repo_contract.get("smoke_command", "") or ""),
            "metric_paths": metric_paths,
            "editable_paths": editable_paths,
            "protected_paths": protected_paths,
            "resource_manifest": resource_manifest,
        },
        "init_repo": {
            "repo_path": effective_repo_path,
            "source": str(repo_contract.get("repo_source", "") or ""),
            "variant_mode": variant_mode,
            "entrypoint_hint": main_entry,
            "install_command": str(repo_contract.get("install_command", "") or ""),
            "editable_paths": editable_paths,
            "protected_paths": protected_paths,
        },
        "validated_repo_handoff": validated_repo_handoff,
    }

    return payload


def generate_impl(
    state: PaperBenchReproState,
    *,
    build_repo_plan: Callable[[PaperBenchReproState], Any],
    build_runtime_project_plan: Callable[[PaperBenchReproState], Any],
    build_generation_manifest: Callable[[PaperBenchReproState], Any],
    ordered_runtime_task_ids: Callable[[Any, Any], list[str]],
    build_runtime_task_views: Callable[[Any, Any], list[dict[str, Any]]],
    build_task_project_plan: Callable[[Any, Any, dict[str, str]], Any],
    filter_task_generated_files: Callable[[dict[str, str], Any], dict[str, str]],
    build_generate_stage_output: Callable[[PaperBenchReproState], Any],
    get_dataset_preparation: Callable[[PaperBenchReproState], dict[str, Any]],
    run_task_review: Callable[[dict[str, str], Any, dict[str, Any]], dict[str, Any]],
    get_output_dir: Callable[[PaperBenchReproState], Any],
    write_stage_output: Callable[[PaperBenchReproState, str, object], None],
    run_or_resume_stage: Callable[..., Any],
    save_tracking_artifacts: Callable[[PaperBenchReproState], None],
    load_local_generation_bundle: Callable[[PaperBenchReproState], dict[str, Any]],
    apply_local_generation_bundle: Callable[[PaperBenchReproState, dict[str, Any]], None],
    persist_generation_checkpoints: Callable[[PaperBenchReproState], None],
    build_repo_handoff_payload: Callable[[PaperBenchReproState], dict[str, Any]],
    run_repo_validation_bundle: Callable[[PaperBenchReproState], dict[str, Any]],
    evaluate_validation_bundle: Callable[[PaperBenchReproState], tuple[Any, Any, Any]],
) -> PaperBenchReproState:
    logger.info("generate - Preparing direct generate context from repo-level plan...")
    input_payload = {
        "upstream_intent": upstream_intent_payload(state),
        "architecture": state.architecture.model_dump(mode="json") if state.architecture else {},
        "package_file_planning_output": (
            state.package_file_planning_output.model_dump(mode="json")
            if state.package_file_planning_output
            else {}
        ),
        "work_package_planning": state.work_package_planning.model_dump(mode="json") if state.work_package_planning else {},
        "global_contract": state.global_contract.model_dump(mode="json") if state.global_contract else {},
        "topic_profile": state.topic_profile.model_dump(mode="json") if state.topic_profile else {},
    }

    def _usage_totals(task_usage_summaries: list[dict[str, Any]]) -> dict[str, int]:
        keys = (
            "calls",
            "calls_with_usage",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "estimated_input_tokens",
            "estimated_output_tokens",
            "estimated_total_tokens",
        )
        return {
            key: sum(int(item.get(key, 0) or 0) for item in task_usage_summaries)
            for key in keys
        }

    def _task_contract_hash(task_view: dict[str, Any]) -> str:
        task_input = dict(task_view.get("task_input", {}) or {})
        payload = {
            "task_id": str(task_view.get("task_id", "") or ""),
            "file_path": str(task_input.get("file_path", "") or ""),
            "work_package_id": str(task_input.get("work_package_id", "") or ""),
            "dependency_files": list(task_input.get("dependency_files", []) or []),
            "review_points": list(task_input.get("review_points", []) or []),
            "interface_contract": list(task_input.get("interface_contract", []) or []),
            "implementation_surfaces": list(task_input.get("implementation_surfaces", []) or []),
            "method_obligations": list(task_input.get("method_obligations", []) or []),
            "defines_symbols": list(task_input.get("defines_symbols", []) or []),
            "calls_symbols": list(task_input.get("calls_symbols", []) or []),
            "evidence_summary": list(task_input.get("evidence_summary", []) or []),
            "writes_artifacts": list(task_input.get("writes_artifacts", []) or []),
            "reads_artifacts": list(task_input.get("reads_artifacts", []) or []),
            "canonical_route": dict(task_input.get("canonical_route", {}) or {}),
            "paper_claim_inventory": dict(task_input.get("paper_claim_inventory", {}) or {}),
            "paper_claim_closure_items": list(task_input.get("paper_claim_closure_items", []) or []),
            "paper_claim_closure_rules": list(task_input.get("paper_claim_closure_rules", []) or []),
            "paper_evidence_contract": dict(task_input.get("paper_evidence_contract", {}) or {}),
            "formula_algorithm_contract": dict(task_input.get("formula_algorithm_contract", {}) or {}),
            "prepare_quality_gate_summary": dict(task_input.get("prepare_quality_gate_summary", {}) or {}),
            "generation_context": dict(task_input.get("generation_context", {}) or {}),
            "critical_grounding_warning": bool(task_input.get("critical_grounding_warning", False)),
        }
        return json.dumps(payload, sort_keys=True, ensure_ascii=False)

    def _empty_execution_result() -> dict[str, Any]:
        return {
            "success": True,
            "output": "",
            "error": "",
            "exit_code": 0,
            "metrics": {},
            "checks": [],
            "artifacts": [],
            "artifact_summary": {},
        }

    def _recover_completed_tasks_from_partial_artifacts(
        *,
        generate_node_dir: Any,
        current_task_contract_hashes: dict[str, str],
        runtime_task_view_map: dict[str, dict[str, Any]],
    ) -> None:
        """Recover completed tasks when provider failure happened after task checkpointing."""
        local_generation_resume = (
            bool(getattr(state.input, "resume_in_place", False))
            and str(getattr(state.input, "resume_start_stage", "") or "").strip() == "local_file_generation"
        )
        if state.execution_history and not local_generation_resume:
            return
        last_attempt_path = generate_node_dir / "last_attempt.json"
        history_path = generate_node_dir / "execution_history.json"
        recovered_contract_hashes: dict[str, str] = {}
        recovered_success_by_task: dict[str, dict[str, Any]] = {}
        recovered_any_by_task: dict[str, dict[str, Any]] = {}
        history_payload: list[dict[str, Any]] = [
            dict(item) for item in list(getattr(state, "execution_history", []) or []) if isinstance(item, dict)
        ]
        if history_path.exists():
            try:
                loaded_history = json.loads(history_path.read_text(encoding="utf-8"))
            except Exception:
                loaded_history = []
            if isinstance(loaded_history, list):
                history_payload.extend(dict(item) for item in loaded_history if isinstance(item, dict))
        for item in history_payload:
            task_id = str(item.get("task_id", "") or "").strip()
            contract_hash = str(item.get("task_contract_hash", "") or "").strip()
            if not task_id:
                continue
            recovered_any_by_task[task_id] = dict(item)
            if not contract_hash:
                continue
            if contract_hash == current_task_contract_hashes.get(task_id, "") and bool(
                dict(item.get("task_review", {}) or {}).get("success", False)
            ):
                recovered_contract_hashes[task_id] = contract_hash
                recovered_success_by_task[task_id] = dict(item)
        checkpoint_path = generate_node_dir / "iteration_checkpoint.json"
        if not checkpoint_path.exists() and not recovered_contract_hashes and not local_generation_resume:
            return
        checkpoint: dict[str, Any] = {}
        if checkpoint_path.exists():
            try:
                loaded_checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            except Exception:
                loaded_checkpoint = {}
            if isinstance(loaded_checkpoint, dict):
                checkpoint = loaded_checkpoint
        if (
            checkpoint_path.exists()
            and not recovered_contract_hashes
            and not local_generation_resume
            and str(checkpoint.get("latest_status", "") or "").strip() != "passed"
        ):
            return
        checkpoint_files = {
            str(item).strip()
            for item in list(checkpoint.get("generated_files", []) or [])
            if str(item).strip()
        }
        last_attempt: dict[str, Any] = {}
        if last_attempt_path.exists():
            try:
                payload = json.loads(last_attempt_path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    last_attempt = payload
            except Exception:
                last_attempt = {}
        last_attempt_task_id = str(last_attempt.get("task_id", "") or "").strip()
        last_attempt_result = dict(last_attempt.get("execution_result", {}) or {})
        last_attempt_manifest = dict(last_attempt.get("project_manifest", {}) or {})
        recovered_contract_hashes.update(
            {
                str(item.get("task_id", "") or "").strip(): str(item.get("task_contract_hash", "") or "").strip()
                for item in list(getattr(state, "execution_history", []) or [])
                if isinstance(item, dict)
                and str(item.get("task_id", "") or "").strip()
                and str(item.get("task_contract_hash", "") or "").strip()
                and bool(dict(item.get("task_review", {}) or {}).get("success", False))
            }
        )
        if not recovered_contract_hashes:
            if history_path.exists():
                try:
                    history_payload = json.loads(history_path.read_text(encoding="utf-8"))
                    recovered_contract_hashes = {
                        str(item.get("task_id", "") or "").strip(): str(item.get("task_contract_hash", "") or "").strip()
                        for item in list(history_payload or [])
                        if isinstance(item, dict)
                        and str(item.get("task_id", "") or "").strip()
                        and str(item.get("task_contract_hash", "") or "").strip()
                        and bool(dict(item.get("task_review", {}) or {}).get("success", False))
                    }
                except Exception:
                    recovered_contract_hashes = {}
        repo_dir = get_output_dir(state) / "repo"
        planned_existing_files: set[str] = set()
        for task_id, task_view in runtime_task_view_map.items():
            normalized_task_id = str(task_id or "").strip()
            task_input = dict(task_view.get("task_input", {}) or {})
            file_path = str(task_input.get("file_path", "") or "").strip()
            if not file_path:
                continue
            candidate = repo_dir / file_path
            if candidate.exists() and candidate.is_file() and candidate.stat().st_size > 0:
                if (
                    recovered_contract_hashes.get(normalized_task_id, "")
                    == current_task_contract_hashes.get(normalized_task_id, "")
                ):
                    planned_existing_files.add(file_path)
                elif local_generation_resume:
                    planned_existing_files.add(file_path)
                    recovered_contract_hashes[normalized_task_id] = current_task_contract_hashes.get(
                        normalized_task_id,
                        "",
                    )
        changed_root = generate_node_dir / "changed_files"
        changed_files = set()
        if changed_root.exists():
            for task_view in runtime_task_view_map.values():
                task_input = dict(task_view.get("task_input", {}) or {})
                file_path = str(task_input.get("file_path", "") or "").strip()
                if file_path and (changed_root / file_path).exists():
                    changed_files.add(file_path)
        recoverable_files = checkpoint_files | planned_existing_files | changed_files
        if not recoverable_files:
            return
        recovered: list[dict[str, Any]] = []
        recovered_checkpoints: list[GenerationCheckpoint] = []
        for task_sequence, (task_id, task_view) in enumerate(runtime_task_view_map.items()):
            if task_id not in current_task_contract_hashes:
                continue
            if recovered_contract_hashes.get(task_id, "") != current_task_contract_hashes.get(task_id, ""):
                continue
            task_input = dict(task_view.get("task_input", {}) or {})
            file_path = str(task_input.get("file_path", "") or "").strip()
            if not file_path or file_path not in recoverable_files:
                continue
            repo_candidate = repo_dir / file_path
            changed_candidate = changed_root / file_path
            if not repo_candidate.exists() and changed_candidate.exists():
                try:
                    repo_candidate.parent.mkdir(parents=True, exist_ok=True)
                    repo_candidate.write_text(changed_candidate.read_text(encoding="utf-8"), encoding="utf-8")
                except Exception:
                    continue
            if not repo_candidate.exists():
                continue
            execution_result = (
                last_attempt_result
                if task_id == last_attempt_task_id and last_attempt_result
                else _empty_execution_result()
            )
            if execution_result and execution_result.get("success") is False:
                continue
            review_checks = list(execution_result.get("checks", []) or [])
            if not review_checks:
                review_checks = [
                    {
                        "name": "recovered_checkpoint_file_exists",
                        "task_id": task_id,
                        "file_path": file_path,
                        "passed": True,
                        "error": "",
                    }
                ]
            suggestions = (
                list(last_attempt.get("suggestions", []) or [])
                if task_id == last_attempt_task_id
                else []
            )
            prior_record = dict(recovered_success_by_task.get(task_id, {}) or recovered_any_by_task.get(task_id, {}) or {})
            prior_review = dict(prior_record.get("task_review", {}) or {})
            prior_review_success = bool(prior_review.get("success", False))
            if prior_review and prior_review_success:
                task_review = prior_review
                recovery_note = "recovered_execution_history"
            elif prior_review and not prior_review_success:
                continue
            else:
                task_review = {
                "task_id": task_id,
                "review_stage": "task_review",
                "success": True,
                "checks": [
                    {
                        "name": "recovered_existing_repo_file",
                        "task_id": task_id,
                        "file_path": file_path,
                        "passed": True,
                        "error": "",
                        "details": "Existing non-empty planned repo file recovered during local_file_generation resume; repo validation and repair remain authoritative.",
                    }
                ],
                "review_points": list(task_input.get("review_points", []) or []),
                "failure_summary": [],
                "suggestions": suggestions,
                "recovered_from_iteration_checkpoint": True,
                "previous_task_review_success": prior_review_success,
            }
                recovery_note = "recovered_existing_repo_file"
            recovered.append(
                {
                    "iteration": int(last_attempt.get("iteration", 0) or 0),
                    "task_sequence": (
                        int(last_attempt_manifest.get("task_sequence", task_sequence) or task_sequence)
                        if task_id == last_attempt_task_id
                        else task_sequence
                    ),
                    "task_id": task_id,
                    "task_contract_hash": current_task_contract_hashes[task_id],
                    "file_path": file_path,
                    "generated_files": sorted(recoverable_files),
                    "result": execution_result,
                    "suggestions": suggestions,
                    "repair_trace": (
                        list(last_attempt.get("repair_trace", []) or [])
                        if task_id == last_attempt_task_id
                        else [
                            {
                                "attempt": 1,
                                "success": True,
                                "changed_files": [file_path],
                                "suggestions": [],
                                "recovered_from_iteration_checkpoint": True,
                            }
                        ]
                    ),
                    "context_usage": (
                        dict(last_attempt.get("context_usage", {}) or {})
                        if task_id == last_attempt_task_id
                        else {}
                    ),
                    "task_review": task_review,
                    "task_review_attempt_count": (
                        len(list(last_attempt.get("repair_trace", []) or [])) or 1
                        if task_id == last_attempt_task_id
                        else 1
                    ),
                    "materialization_mode": "generate",
                    "recovered_from_iteration_checkpoint": True,
                }
            )
            recovered_checkpoints.append(
                GenerationCheckpoint(
                    checkpoint_id=f"generate:0:{task_sequence}:{task_id}",
                    stage="local_file_generation",
                    focus_id=task_id,
                    input_refs=[file_path],
                    output_refs=[file_path],
                    notes=[recovery_note],
                )
            )
        if recovered:
            state.execution_history = recovered
            state.generation_checkpoints = recovered_checkpoints

    def _recover_completed_task_from_last_attempt(
        *,
        generate_node_dir: Any,
        current_task_contract_hashes: dict[str, str],
        runtime_task_view_map: dict[str, dict[str, Any]],
    ) -> None:
        """Compatibility wrapper for older call sites."""
        _recover_completed_tasks_from_partial_artifacts(
            generate_node_dir=generate_node_dir,
            current_task_contract_hashes=current_task_contract_hashes,
            runtime_task_view_map=runtime_task_view_map,
        )

    def _unused_legacy_recovery_record() -> list[dict[str, Any]]:
        return [
            {
                "unused": True,
            }
        ]

    def _compute() -> dict[str, Any]:
        state.repo_plan = build_repo_plan(state)
        state.project_plan = build_runtime_project_plan(state)
        state.generation_manifest = build_generation_manifest(state)
        handoff_issues = _semantic_handoff_checksum_issues(state)
        if handoff_issues:
            state.temp_data["semantic_handoff_issues"] = handoff_issues
            raise RuntimeError("; ".join(handoff_issues[:4]))
        state.generation_checkpoints = []

        output_dir = get_output_dir(state)
        output_dir.mkdir(parents=True, exist_ok=True)
        generate_node_dir = output_dir / "nodes" / "generate"
        generate_node_dir.mkdir(parents=True, exist_ok=True)
        repo_dir = output_dir / "repo"
        repo_dir.mkdir(parents=True, exist_ok=True)
        (generate_node_dir / "plan.md").write_text(state.plan, encoding="utf-8")
        write_stage_output(state, "repo_plan.json", state.repo_plan)
        write_stage_output(state, "project_plan.json", state.project_plan)
        write_stage_output(state, "generation_manifest.json", state.generation_manifest)

        runtime_task_views = build_runtime_task_views(state.project_plan, state.generation_manifest)
        dataset_preparation = get_dataset_preparation(state)
        resource_manifest = dict(state.temp_data.get("resource_manifest", {}) or {})
        project_manifest: dict[str, Any] = {
            "entrypoints": state.project_plan.entrypoints,
            "generated_files": [],
            "dataset_preparation": dataset_preparation,
            "resource_manifest": resource_manifest,
            "ordered_tasks": [item["task_id"] for item in runtime_task_views],
        }
        runtime_task_view_map = {
            str(item["task_id"]).strip(): item
            for item in runtime_task_views
            if str(item.get("task_id", "")).strip()
        }
        current_task_contract_hashes = {
            task_id: _task_contract_hash(task_view)
            for task_id, task_view in runtime_task_view_map.items()
        }
        _recover_completed_tasks_from_partial_artifacts(
            generate_node_dir=generate_node_dir,
            current_task_contract_hashes=current_task_contract_hashes,
            runtime_task_view_map=runtime_task_view_map,
        )
        canonical_keep_files = [path for path in [_explicit_main_entry(state)] if str(path).strip()]
        allowed_repo_files = {
            str(path).strip()
            for path in (
                canonical_keep_files
                + [item.get("file_path", "") for item in (view.get("task_input", {}) for view in runtime_task_views)]
            )
            if str(path).strip()
        }
        completed_task_ids = {
            str(item.get("task_id", "")).strip()
            for item in list(state.execution_history or [])
            if isinstance(item, dict)
            and str(item.get("task_id", "")).strip()
            and bool(dict(item.get("task_review", {}) or {}).get("success"))
            and str(item.get("task_contract_hash", "") or "")
            == current_task_contract_hashes.get(str(item.get("task_id", "")).strip(), "")
        }
        selected_task_ids = [
            item["task_id"]
            for item in runtime_task_views
            if item["task_id"] not in completed_task_ids
        ]
        original_execution_history = [
            dict(item) for item in list(state.execution_history or []) if isinstance(item, dict)
        ]
        state.execution_history = [
            dict(item)
            for item in original_execution_history
            if (
                not str(item.get("task_id", "")).strip()
                or str(item.get("task_id", "")).strip() not in current_task_contract_hashes
                or str(item.get("task_contract_hash", "") or "")
                == current_task_contract_hashes.get(str(item.get("task_id", "")).strip(), "")
            )
        ]
        stale_history_dropped = len(state.execution_history) != len(original_execution_history)
        preserved_execution_result = (
            _empty_execution_result()
            if stale_history_dropped
            else (
                state.execution_result.model_dump(mode="json")
                if state.execution_result
                else _empty_execution_result()
            )
        )
        if stale_history_dropped:
            state.execution_result = ExecutionResult(**preserved_execution_result)
        materialization = materialize_selected_tasks(
            state,
            selected_task_ids=selected_task_ids,
            allowed_repo_files=allowed_repo_files,
            initial_execution_result=preserved_execution_result,
            iteration_seed=len(completed_task_ids),
            mode_label="generate",
            review_scope="generate/task_review",
            repair_round=None,
            ordered_runtime_task_ids=ordered_runtime_task_ids,
            build_runtime_task_views=build_runtime_task_views,
            build_task_project_plan=build_task_project_plan,
            filter_task_generated_files=filter_task_generated_files,
            get_codegen_config=get_codegen_config,
            get_workflow_config=get_workflow_config,
            get_sandbox_provider=get_sandbox_provider,
            get_output_dir=get_output_dir,
            run_task_review=run_task_review,
        )
        current_project_files = dict(materialization["project_files"])
        current_project_files = {
            path: content
            for path, content in current_project_files.items()
            if path in allowed_repo_files
        }
        runtime_artifact_declarations = _materialize_runtime_artifact_declarations(
            state,
            current_project_files,
            allowed_repo_files=allowed_repo_files,
        )
        from reproagent.pipeline.tools import save_project_files

        save_project_files(current_project_files, repo_dir)
        latest_execution_result = dict(materialization["latest_execution_result"])
        latest_suggestions = list(materialization["latest_suggestions"])
        execution_history = list(materialization["execution_history"])
        state.generation_checkpoints.extend(list(materialization["generation_checkpoints"]))

        state.code = current_project_files.get(_canonical_main_entry(state), "")
        state.project_root = str(repo_dir.resolve())
        state.execution_result = ExecutionResult(**latest_execution_result)
        state.execution_history = execution_history
        state.generated_files = sorted(current_project_files.keys())
        state.project_manifest = {
            **project_manifest,
            "generated_files": sorted(current_project_files.keys()),
            "task_review_issue_summary": _task_review_issue_summary(state),
            "runtime_artifact_declarations": runtime_artifact_declarations,
        }
        completeness_issues = _repo_completeness_issues(state)
        if completeness_issues:
            state.temp_data["repo_completeness_issues"] = completeness_issues
            state.project_manifest = {
                **state.project_manifest,
                "repo_completeness_issues": completeness_issues,
                "missing_planned_files": sorted(_planned_repo_files(state) - set(state.generated_files)),
            }
            write_stage_output(state, "project_manifest.json", state.project_manifest)
            raise RuntimeError(
                "local_file_generation cannot finish before all generation_manifest files are materialized: "
                + "; ".join(completeness_issues[:6])
            )
        repo_validation_bundle = run_repo_validation_bundle(state)
        state.runtime_probe, state.validation_report, state.benchmark_report = evaluate_validation_bundle(state)
        state.evaluation = EvaluationDecision(
            action="COMPLETE",
            reason=(
                "All generated files passed task review and repo-level validation gates."
                if latest_execution_result.get("success", False) and state.validation_report and state.validation_report.passed
                else "Code generation completed, but repo-level validation still has unresolved findings."
            ),
            suggestions=(
                list(state.validation_report.repair_recommendations)
                if state.validation_report and state.validation_report.repair_recommendations
                else list(latest_suggestions)
            ),
        )
        state.iteration_count = 0
        state.preflight_result = None
        state.repair_ticket = None
        state.repair_log = None
        state.experiment_results = {}
        state.checkpoint_path = str((generate_node_dir / "iteration_checkpoint.json").resolve())
        state.generate_stage_output = build_generate_stage_output(state)
        file_provenance = refresh_file_provenance(state)
        repo_handoff_payload = build_repo_handoff_payload(state)
        write_stage_output(state, "experiment_output.json", state.generate_stage_output)
        write_stage_output(state, "file_provenance.json", file_provenance)
        write_stage_output(state, "runtime_probe.json", state.runtime_probe)
        write_stage_output(state, "validation_report.json", state.validation_report)
        write_stage_output(state, "benchmark_report.json", state.benchmark_report)
        write_stage_output(state, "repo_handoff.json", repo_handoff_payload)
        state.temp_data["repo_handoff"] = dict(repo_handoff_payload)
        persist_generation_checkpoints(state)
        agent_usage_summary = _usage_totals(list(materialization["task_usage_summaries"]))
        return {
            "repo_plan": state.repo_plan.model_dump(mode="json") if state.repo_plan else {},
            "project_plan": state.project_plan.model_dump(mode="json"),
            "generation_manifest": state.generation_manifest.model_dump(mode="json") if state.generation_manifest else {},
            "generated_files": list(state.generated_files),
            "project_root": state.project_root,
            "project_manifest": dict(state.project_manifest),
            "code": state.code,
            "execution_result": state.execution_result.model_dump(mode="json") if state.execution_result else {},
            "evaluation": state.evaluation.model_dump(mode="json") if state.evaluation else {},
            "preflight_result": {},
            "experiment_results": {},
            "runtime_probe": state.runtime_probe.model_dump(mode="json") if state.runtime_probe else {},
            "validation_report": state.validation_report.model_dump(mode="json") if state.validation_report else {},
            "benchmark_report": state.benchmark_report.model_dump(mode="json") if state.benchmark_report else {},
            "repo_validation_bundle": repo_validation_bundle,
            "generate_stage_output": state.generate_stage_output.model_dump(mode="json") if state.generate_stage_output else {},
            "file_provenance": file_provenance,
            "checkpoint_path": state.checkpoint_path,
            "execution_history": state.execution_history,
            "iteration_count": state.iteration_count,
            "generation_checkpoints": [item.model_dump(mode="json") for item in state.generation_checkpoints],
            "agent_usage_summary": agent_usage_summary,
        }

    def _load() -> dict[str, Any]:
        return load_local_generation_bundle(state)

    def _write(result: dict[str, Any]) -> None:
        apply_local_generation_bundle(state, result)
        _assert_repo_generation_complete(state, context="local_file_generation")
        if state.repo_plan is not None:
            write_stage_output(state, "repo_plan.json", state.repo_plan)
        write_stage_output(state, "project_plan.json", state.project_plan)
        if state.generation_manifest is not None:
            write_stage_output(state, "generation_manifest.json", state.generation_manifest)
        if state.generate_stage_output is not None:
            write_stage_output(state, "experiment_output.json", state.generate_stage_output)
        file_provenance = refresh_file_provenance(state)
        write_stage_output(state, "file_provenance.json", file_provenance)
        if state.runtime_probe is not None:
            write_stage_output(state, "runtime_probe.json", state.runtime_probe)
        if state.validation_report is not None:
            write_stage_output(state, "validation_report.json", state.validation_report)
        if state.benchmark_report is not None:
            write_stage_output(state, "benchmark_report.json", state.benchmark_report)
        state.temp_data["repo_handoff"] = build_repo_handoff_payload(state)
        write_stage_output(state, "repo_handoff.json", build_repo_handoff_payload(state))

    bundle = run_or_resume_stage(
        state,
        "local_file_generation",
        input_payload,
        _compute,
        _load,
        _write,
    )
    apply_local_generation_bundle(state, bundle)
    _assert_repo_generation_complete(state, context="local_file_generation resume/load")
    _classify_generate_status(state)
    save_tracking_artifacts(state)
    return state
