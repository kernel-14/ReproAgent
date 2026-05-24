
"""Experiment generation workflow."""
import json
import inspect
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    from langgraph.graph import END, StateGraph
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

from reproagent.pipeline.config import build_github_repo_config, get_codegen_config, get_workflow_config, semantic_anchor_disabled
from reproagent.pipeline.nodes import generate as generate_nodes
from reproagent.pipeline.nodes import plan as plan_nodes
from reproagent.pipeline.nodes import prepare as prepare_nodes
from reproagent.pipeline.nodes import repair as repair_nodes
from reproagent.pipeline.prompts import (
    build_architecture_prompt,
    build_boundary_requirements_prompt,
    build_global_contract_prompt,
    build_input_normalization_prompt,
    build_package_file_planning_prompt,
    build_pipeline_plan_prompt,
    build_repair_plan_prompt,
    build_repair_plan_review_prompt,
    build_reference_selection_prompt,
    build_topic_profile_prompt,
    build_unit_extraction_prompt,
    build_work_package_planning_prompt,
)
from reproagent.pipeline.schemas import (
    ArchitectureOutput,
    BenchmarkReport,
    BoundaryRequirementsOutput,
    CanonicalIROutput,
    CanonicalIRValidationOutput,
    EvidenceBundleOutput,
    EvidenceLinkOutput,
    EvaluationDecision,
    ExperimentImplementationHandoff,
    ExecutionResult,
    PaperBenchReproInput,
    InputNormalizationOutput,
    PackageFilePlanningOutput,
    PaperBenchReproState,
    GenerationCheckpoint,
    GlobalContractOutput,
    PipelinePlanOutput,
    PreflightResult,
    ProjectPlan,
    RequirementAnchor,
    RepairAction,
    RepairEvalFinding,
    RepairEvalReport,
    RepairLog,
    RepairPlan,
    RepairPlanDraft,
    RepairPlanReview,
    RepairTicket,
    PreparedReferenceRepositorySurvey,
    ReferenceSelectionOutput,
    RepoFilePlan,
    RepoPlan,
    RuntimeProbe,
    TopicProfileOutput,
    UnitExtractionOutput,
    ValidationCheck,
    ValidationReport,
    WorkPackagePlanningOutput,
)
from reproagent.pipeline.utils import generation_manifest
from reproagent.pipeline.utils import plan_builder
from reproagent.pipeline.utils import prompt_context_builder
from reproagent.pipeline.utils import repo_plan_builder
from reproagent.pipeline.utils import task_review
from reproagent.pipeline.utils import canonical_ir as canonical_ir_utils
from reproagent.pipeline.utils import run_context
from reproagent.pipeline.utils import run_state_manager
from reproagent.pipeline.utils import runtime_planner
from reproagent.pipeline.utils import stage_executor
from reproagent.pipeline.utils import storage_manager
from reproagent.pipeline.utils import ref_repo_survey
from reproagent.pipeline.utils import evidence_grounding


def _stage_review_repair_budget(state: PaperBenchReproState) -> int:
    """Return the effective stage review/fix budget for planning/generation hooks."""
    configured = int(state.input.stage_review_repair_budget or 0)
    if configured > 0:
        return configured
    return max(0, int(get_workflow_config().max_stage_fix_rounds or 0))
from reproagent.pipeline.utils import local_generation_bundle
from reproagent.pipeline.utils import repair_helpers
from reproagent.pipeline.utils import memory as run_memory
from reproagent.pipeline.utils.dataset_download_tool import download_datasets
from reproagent.pipeline.utils import dataset_manager
from reproagent.pipeline.utils.ref_repo_clone import clone_reference_repository
from reproagent.pipeline.utils.ref_repo_search_tool import search_reference_repository
from reproagent.pipeline.utils.artifact_names import CANONICAL_ARTIFACTS
from reproagent.pipeline.utils import validation_helpers
from reproagent.pipeline.utils import workflow_runtime
from reproagent.pipeline.utils.artifact_writer import write_artifact
from reproagent.pipeline.utils.file_provenance import refresh_file_provenance
from reproagent.pipeline.utils.intent_contract import upstream_intent_payload
from reproagent.pipeline.utils.quality_status import is_validated_repo_handoff_ready, refresh_quality_status
from reproagent.sandbox import get_sandbox_provider


_HANDOFF_STAGE_FILENAMES = {"repo_handoff.json", "validated_repo_handoff.json"}
_PLAN_SOURCE_OF_TRUTH_FILENAMES = {
    "architecture.json",
    "boundary_requirements.json",
    "global_contract.json",
    "package_file_planning.json",
    "pipeline_plan.json",
    "reference_selection.json",
    "work_packages.json",
}


_get_output_dir = run_context._get_output_dir
_json_default = run_context._json_default
_new_run_id = run_context._new_run_id
_invoke_json_stage = stage_executor._invoke_json_stage
_normalize_dataset_requests = dataset_manager._normalize_dataset_requests
_build_dataset_preparation_payload = dataset_manager._build_dataset_preparation_payload
_update_input_dataset_status = dataset_manager._update_input_dataset_status
_update_input_benchmark_status = dataset_manager._update_input_benchmark_status
_get_dataset_preparation = dataset_manager._get_dataset_preparation
_get_resource_manifest = dataset_manager._get_resource_manifest
_prepare_benchmarks = dataset_manager.prepare_benchmarks
_prepare_baselines = dataset_manager.prepare_baselines
_build_resource_manifest = dataset_manager.build_resource_manifest
_get_reference_repo_surveys = ref_repo_survey._get_reference_repo_surveys
_limit_json_for_prompt = prompt_context_builder._limit_json_for_prompt
_build_input_normalization_context = prompt_context_builder._build_input_normalization_context
_build_unit_extraction_context = prompt_context_builder._build_unit_extraction_context
_build_boundary_requirements_context = prompt_context_builder._build_boundary_requirements_context
_build_work_package_planning_context = prompt_context_builder._build_work_package_planning_context
_build_work_package_local_context = prompt_context_builder._build_work_package_local_context
_build_topic_profile_context = prompt_context_builder._build_topic_profile_context
_build_reference_selection_context = prompt_context_builder._build_reference_selection_context
_build_pipeline_plan_context = prompt_context_builder._build_pipeline_plan_context
_build_global_contract_context = prompt_context_builder._build_global_contract_context
_build_architecture_context = prompt_context_builder._build_architecture_context
_build_architecture_package_context = prompt_context_builder._build_architecture_package_context
_build_package_file_planning_context = prompt_context_builder._build_package_file_planning_context
_build_package_file_planning_local_context = prompt_context_builder._build_package_file_planning_local_context
_build_generation_manifest = generation_manifest._build_generation_manifest
_build_runtime_task_views = runtime_planner._build_runtime_task_views
_build_task_project_plan = runtime_planner._build_task_project_plan
_filter_task_generated_files = runtime_planner._filter_task_generated_files
_build_generate_stage_output = runtime_planner._build_generate_stage_output
_project_file_plans_from_architecture = plan_builder._project_file_plans_from_architecture
_close_package_file_plans = plan_builder._close_package_file_plans
_validate_file_plans = plan_builder._validate_file_plans
_order_file_plans_for_execution_closure = plan_builder._order_file_plans_for_execution_closure
_derive_steps_from_file_plans = plan_builder._derive_steps_from_file_plans
_render_pipeline_plan_markdown = plan_builder._render_pipeline_plan_markdown
_build_runtime_project_plan = runtime_planner._build_runtime_project_plan
_ordered_runtime_task_ids = runtime_planner._ordered_runtime_task_ids
_build_repo_plan = repo_plan_builder._build_repo_plan
_build_evidence_bundles = evidence_grounding._build_evidence_bundles
_build_canonical_ir = canonical_ir_utils.build_canonical_ir
_validate_canonical_ir = canonical_ir_utils.validate_canonical_ir
_run_task_review = task_review._run_task_review
_build_terminal_evaluation = task_review._build_terminal_evaluation
_task_touches_runtime_surface = task_review._task_touches_runtime_surface
_load_iteration_checkpoint_payload = storage_manager._load_iteration_checkpoint_payload

_storage_manager_module = storage_manager
_run_state_manager_module = run_state_manager


def _invoke_json_stage_for_state(
    state: PaperBenchReproState,
    stage_name: str,
    schema_name: str,
    system: str,
    user: str,
) -> dict[str, Any]:
    try:
        parameter_count = len(inspect.signature(_invoke_json_stage).parameters)
    except (TypeError, ValueError):
        parameter_count = 5
    if parameter_count >= 5:
        return _invoke_json_stage(stage_name, schema_name, system, user, state)
    return _invoke_json_stage(stage_name, schema_name, system, user)


def _run_node(state: PaperBenchReproState, node_name: str, fn):
    """Workflow-local wrapper that keeps artifact-path helpers patchable from this module."""
    with _storage_manager_module.storage_context(
        output_dir_fn=_get_output_dir,
        json_default_fn=_json_default,
    ):
        return _run_state_manager_module._run_node(
            state,
            node_name,
            fn,
            save_tracking_artifacts=_save_tracking_artifacts,
        )


STAGE_ORDER = workflow_runtime.STAGE_ORDER
STAGE_OUTPUTS = workflow_runtime.STAGE_OUTPUTS
SUPPLEMENTARY_ARTIFACTS = workflow_runtime.SUPPLEMENTARY_ARTIFACTS
_FIXED_NODE_ONLY_ARTIFACTS = {
    "repo_plan.json": "generate",
    "project_plan.json": "generate",
    "generation_manifest.json": "generate",
}
_CURRENT_NODE_ONLY_ARTIFACTS = {
    "experiment_output.json",
    "repo_handoff.json",
    "validation_bundle.json",
    "repair_validation_bundle.json",
    "repair_review.json",
    "requirement_anchor.json",
    "repair_eval_report.json",
    "repair_findings.json",
    "preflight.json",
    "execution_result.json",
    "repair_ticket.json",
    "runtime_probe.json",
    "validation_report.json",
    "benchmark_report.json",
    "validated_repo_handoff.json",
    "repair_plan_context.json",
    "repair_plan_draft.json",
    "repair_plan_review.json",
    "repair_plan.json",
    "repair_regeneration_result.json",
    "repair_log.json",
    "generation_checkpoints.json",
    "iteration_checkpoint.json",
    "file_provenance.json",
    "last_attempt.json",
    "recovery_tickets.json",
    "upstream_intent.json",
}


def _is_terminal_blocked(state: PaperBenchReproState) -> bool:
    """Return true when an upstream stage has intentionally stopped the workflow."""
    return (
        str(getattr(state, "status", "") or "") in {"failed", "completed_with_degraded_contract"}
        or str(getattr(state, "terminal_outcome", "") or "") in {"failed", "completed_with_degraded_contract"}
        or bool(str(getattr(state, "failed_node", "") or "").strip())
    )


def _json_object_payload(payload: object) -> dict[str, Any]:
    """Return a dict payload for handoff views; malformed payloads become debug metadata."""
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump(mode="json")
    if isinstance(payload, dict):
        try:
            return json.loads(json.dumps(dict(payload), ensure_ascii=False, default=_json_default))
        except Exception:
            return {"raw_payload": str(payload), "payload_type": type(payload).__name__}
    if payload is None:
        return {}
    try:
        raw_payload = json.loads(json.dumps(payload, ensure_ascii=False, default=_json_default))
    except Exception:
        raw_payload = str(payload)
    return {"raw_payload": raw_payload, "payload_type": type(payload).__name__}


def _as_mapping(payload: object) -> dict[str, Any]:
    """Coerce handoff sub-payloads without letting malformed data break finalization."""
    return _json_object_payload(payload)


def _load_state_snapshot(output_dir: Path, snapshot_name: str) -> dict[str, Any]:
    snapshot_path = output_dir / "state_snapshots" / snapshot_name
    if not snapshot_path.exists():
        raise FileNotFoundError(f"missing state snapshot: {snapshot_path}")
    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"invalid state snapshot payload: {snapshot_path}")
    return payload


def _hydrate_state_for_in_place_resume(state: PaperBenchReproState) -> PaperBenchReproState:
    resume_stage = str(getattr(state.input, "resume_start_stage", "") or "").strip()
    if resume_stage == "generate":
        resume_stage = "local_file_generation"
        state.input.resume_start_stage = resume_stage
    if not bool(getattr(state.input, "resume_in_place", False)) or not resume_stage:
        return state
    if not state.run_id:
        state.run_id = str(getattr(state.input, "resume_from_run_id", "") or "").strip()
    output_dir = _get_output_dir(state)
    def _first_existing_snapshot(*names: str) -> str:
        for name in names:
            if (output_dir / "state_snapshots" / name).exists():
                return name
        return names[-1]

    snapshot_by_resume_stage = {
        "topic_profile_synthesis": _first_existing_snapshot("plan_failed.json", "prepare_after.json"),
        "work_package_planning": _first_existing_snapshot("plan_failed.json", "prepare_after.json"),
        "package_evidence_grounding": _first_existing_snapshot("plan_failed.json", "prepare_after.json"),
        "reference_selection": _first_existing_snapshot("plan_failed.json", "prepare_after.json"),
        "pipeline_plan": _first_existing_snapshot("plan_failed.json", "prepare_after.json"),
        "global_contract_synthesis": _first_existing_snapshot("plan_failed.json", "prepare_after.json"),
        "architecture_planning": _first_existing_snapshot("plan_failed.json", "prepare_after.json"),
        "package_file_planning": _first_existing_snapshot("plan_failed.json", "prepare_after.json"),
        "canonical_ir_synthesis": _first_existing_snapshot("plan_failed.json", "prepare_after.json"),
        "local_file_generation": "plan_after.json",
        "repair_validation": "generate_after.json",
    }
    snapshot_name = snapshot_by_resume_stage.get(resume_stage)
    if not snapshot_name:
        return state
    payload = _load_state_snapshot(output_dir, snapshot_name)
    if str(payload.get("terminal_outcome", "") or "").strip() == "failed":
        payload["terminal_outcome"] = "completed"
        payload["terminal_outcome_reason"] = ""
    if str(payload.get("status", "") or "").strip() == "failed":
        payload["status"] = "pending"
    payload["current_node"] = ""
    payload["failed_node"] = ""
    payload["error_message"] = ""
    resumed_state = PaperBenchReproState(**payload)
    resumed_state.input = state.input
    resumed_state.run_id = state.run_id
    resumed_state.temp_data = dict(resumed_state.temp_data or {})
    resumed_state.temp_data["resume_in_place"] = True
    resumed_state.temp_data["resume_start_stage"] = resume_stage
    resumed_state.current_node = ""
    resumed_state.failed_node = ""
    resumed_state.error_message = ""
    resumed_state.status = "pending"
    resumed_state.terminal_outcome = "completed"
    resumed_state.terminal_outcome_reason = ""
    plan_resume_cutoffs = {
        "topic_profile_synthesis",
        "work_package_planning",
        "package_evidence_grounding",
        "reference_selection",
        "pipeline_plan",
        "global_contract_synthesis",
        "architecture_planning",
        "package_file_planning",
        "canonical_ir_synthesis",
    }
    if resume_stage in plan_resume_cutoffs:
        if resume_stage in {"topic_profile_synthesis", "work_package_planning", "package_evidence_grounding", "reference_selection", "pipeline_plan", "global_contract_synthesis", "architecture_planning"}:
            resumed_state.architecture = None
        if resume_stage in {"topic_profile_synthesis", "work_package_planning", "package_evidence_grounding", "reference_selection", "pipeline_plan", "global_contract_synthesis", "architecture_planning", "package_file_planning"}:
            resumed_state.package_file_planning_output = None
        resumed_state.canonical_ir = None
        resumed_state.canonical_ir_validation = None
        resumed_state.planning_failure_layer = ""
    if resume_stage == "local_file_generation":
        resumed_state.project_root = ""
        resumed_state.project_manifest = {}
        resumed_state.code = ""
        resumed_state.execution_result = None
        resumed_state.preflight_result = None
        resumed_state.experiment_results = {}
        resumed_state.evaluation = None
        resumed_state.generate_stage_output = None
        resumed_state.runtime_probe = None
        resumed_state.validation_report = None
        resumed_state.benchmark_report = None
        resumed_state.repair_ticket = None
        resumed_state.requirement_anchor = None
        resumed_state.repair_eval_report = None
        resumed_state.repair_plan = None
        resumed_state.repair_log = None
        resumed_state.current_node = ""
        resumed_state.failed_node = ""
        resumed_state.error_message = ""
        resumed_state.status = "pending"
        for key in list(resumed_state.temp_data.keys()):
            if key in {
                "repo_handoff",
                "validated_repo_handoff",
                "handoff",
                "repair_ticket",
                "runtime_probe",
                "pending_repair_regeneration_attempt",
                "stage_execution_mode",
                "node_errors",
                "degraded_backlog",
                "terminal_outcome",
                "terminal_outcome_reason",
            } or key.startswith("repair_") or key.startswith("validation_"):
                resumed_state.temp_data.pop(key, None)
    elif resume_stage == "repair_validation":
        current_repo_root = output_dir / "repo"
        if current_repo_root.exists():
            from reproagent.pipeline.tools import load_project_files

            resumed_state.project_root = str(current_repo_root.resolve())
            current_repo_files = sorted(load_project_files(current_repo_root).keys())
            if current_repo_files:
                resumed_state.generated_files = current_repo_files
        resumed_state.validation_report = None
        resumed_state.benchmark_report = None
        resumed_state.repair_ticket = None
        resumed_state.requirement_anchor = None
        resumed_state.repair_eval_report = None
        resumed_state.repair_plan = None
        resumed_state.repair_log = None
        for key in list(resumed_state.temp_data.keys()):
            if key in {
                "validated_repo_handoff",
                "repair_ticket",
                "pending_repair_regeneration_attempt",
                "stage_execution_mode",
                "node_errors",
                "degraded_backlog",
                "terminal_outcome",
                "terminal_outcome_reason",
            } or key.startswith("repair_") or key.startswith("validation_"):
                resumed_state.temp_data.pop(key, None)
    return resumed_state


def _build_handoff_artifact_payload(state: PaperBenchReproState) -> dict[str, Any]:
    """Build the canonical PaperBench Repro repository handoff view."""
    repo_handoff = _json_object_payload(state.temp_data.get("repo_handoff", {}))
    validated_repo_handoff = _json_object_payload(state.temp_data.get("validated_repo_handoff", {}))
    quality_status = refresh_quality_status(state).model_dump(mode="json")
    repo_plan = state.repo_plan.model_dump(mode="json") if state.repo_plan else {}
    project_plan = state.project_plan.model_dump(mode="json") if state.project_plan else {}
    workspace_config = _as_mapping(repo_handoff.get("workspace_config", {}))
    runtime_config = _as_mapping(repo_handoff.get("runtime_config", {}))
    init_repo = _as_mapping(repo_handoff.get("init_repo", {}))
    rapid_validation_handoff = _as_mapping(validated_repo_handoff.get("rapid_validation", {}))
    repo_materialization = _as_mapping(rapid_validation_handoff.get("repo_materialization", {}))
    repo_contract = _as_mapping(repo_materialization.get("repo_contract", {}))
    if not repo_contract and repo_materialization:
        repo_contract = {
            key: repo_materialization.get(key)
            for key in (
                "repo_path",
                "repo_source",
                "entrypoint_hint",
                "install_command",
                "baseline_command",
                "idea_command",
                "variant_command",
                "smoke_command",
                "command_contract",
                "metric_paths",
                "editable_paths",
                "protected_paths",
            )
            if key in repo_materialization
        }
    repo_plan_files = [
        str(item.get("target_file", "")).strip()
        for item in list(repo_plan.get("files", []) or [])
        if isinstance(item, dict) and str(item.get("target_file", "")).strip()
    ]
    editable_scope = list(
        dict.fromkeys(
            [
                *repo_plan_files,
                *[str(item).strip() for item in list(repo_contract.get("editable_paths", []) or []) if str(item).strip()],
                *[str(item).strip() for item in list(workspace_config.get("editable_paths", []) or []) if str(item).strip()],
                *[str(item).strip() for item in list(state.generated_files) if str(item).strip()],
            ]
        )
    )
    protected_scope = list(
        dict.fromkeys(
            str(item).strip()
            for item in [
                *list(repo_contract.get("protected_paths", []) or []),
                *list(workspace_config.get("protected_paths", []) or []),
            ]
            if str(item).strip()
        )
    )
    install_command = (
        str(repo_contract.get("install_command", "") or "").strip()
        or str(runtime_config.get("install_command", "") or "").strip()
        or str(init_repo.get("install_command", "") or "").strip()
        or str(project_plan.get("install_command", "") or "").strip()
    )
    smoke_command = (
        str(repo_contract.get("smoke_command", "") or "").strip()
        or str(runtime_config.get("smoke_command", "") or "").strip()
        or str(workspace_config.get("smoke_command", "") or "").strip()
        or str(project_plan.get("smoke_command", "") or "").strip()
    )
    entrypoint = (
        str(repo_contract.get("entrypoint_hint", "") or "").strip()
        or str(dict(repo_plan.get("canonical_route", {}) or {}).get("entry_surface", "") or "").strip()
        or str(runtime_config.get("entrypoint", "") or "").strip()
        or str(init_repo.get("entrypoint_hint", "") or "").strip()
        or str(dict(project_plan.get("entrypoints", {}) or {}).get("main", "") or "").strip()
    )
    validation_evidence = {
        "validation_report": state.validation_report.model_dump(mode="json") if state.validation_report else {},
        "benchmark_report": state.benchmark_report.model_dump(mode="json") if state.benchmark_report else {},
        "runtime_probe": state.runtime_probe.model_dump(mode="json") if state.runtime_probe else {},
        "preflight_result": state.preflight_result.model_dump(mode="json") if state.preflight_result else {},
        "execution_result": state.execution_result.model_dump(mode="json") if state.execution_result else {},
    }
    repair_history = {
        "repair_log": state.repair_log.model_dump(mode="json") if state.repair_log else {},
        "repair_ticket": state.repair_ticket.model_dump(mode="json") if state.repair_ticket else {},
        "recovery_tickets": list(state.temp_data.get("recovery_tickets", []) or []),
        "stage_attempts": list(state.temp_data.get("stage_attempts", []) or []),
    }
    handoff = ExperimentImplementationHandoff(
        run_id=state.run_id,
        target=state.input.target,
        intent_contract=upstream_intent_payload(state),
        repo_root=str(
            repo_contract.get("repo_path", "")
            or repo_materialization.get("working_root", "")
            or workspace_config.get("repo_root", "")
            or workspace_config.get("target_repo_path", "")
            or state.project_root
            or ""
        ),
        project_root=state.project_root,
        entrypoint=entrypoint,
        install_command=install_command,
        smoke_command=smoke_command,
        metric_contract={
            "metric_paths": list(repo_contract.get("metric_paths", []) or workspace_config.get("metric_paths", []) or []),
            **dict(workspace_config.get("metric_contract", {}) or {}),
        },
        artifact_contract=(
            dict(repo_plan.get("artifact_contract", {}) or {})
            if isinstance(repo_plan.get("artifact_contract", {}), dict)
            else list(repo_plan.get("artifact_contract", []) or [])
        ),
        editable_scope=editable_scope,
        editable_paths=editable_scope,
        protected_scope=protected_scope,
        generated_files=list(state.generated_files),
        validation_evidence=validation_evidence,
        known_risks=list(state.temp_data.get("degraded_backlog", []) or []),
        repair_history=repair_history,
        handoff_ready=is_validated_repo_handoff_ready(validated_repo_handoff),
        quality_status=quality_status,
        file_provenance=list(state.temp_data.get("file_provenance", []) or []),
        repo_handoff=repo_handoff,
        validated_repo_handoff=validated_repo_handoff,
    )
    return handoff.model_dump(mode="json")


def _write_terminal_outcome_alias(state: PaperBenchReproState, payload: object) -> None:
    """Keep legacy terminal_outcome.json as a pointer; quality_status owns terminal quality."""
    output_dir = _get_output_dir(state)
    terminal_payload = _json_object_payload(payload)
    if terminal_payload:
        state.temp_data["terminal_outcome_payload"] = terminal_payload
    refresh_quality_status(state)
    write_artifact(
        run_dir=output_dir,
        path=output_dir / "nodes" / "repair" / "terminal_outcome.json",
        payload={},
        logical_name="terminal_outcome",
        kind="state",
        stage="repair",
        node="repair",
        authority="compatibility_alias",
        retention="keep",
        alias_of="quality_status.json",
        depends_on=["quality_status.json"],
    )


def _update_handoff_source_payload(state: PaperBenchReproState, filename: str, json_payload: object) -> None:
    if filename == "repo_handoff.json":
        state.temp_data["repo_handoff"] = _json_object_payload(json_payload)
    if filename == "validated_repo_handoff.json":
        state.temp_data["validated_repo_handoff"] = _json_object_payload(json_payload)


def _write_handoff_artifact(state: PaperBenchReproState) -> None:
    if not state.run_id:
        return
    output_dir = _get_output_dir(state)
    payload = _build_handoff_artifact_payload(state)
    state.temp_data["handoff"] = payload
    write_artifact(
        run_dir=output_dir,
        path=output_dir / "handoff.json",
        payload=payload,
        logical_name="handoff",
        kind="contract",
        stage="handoff",
        node=str(state.current_node or ""),
        authority="source_of_truth",
        retention="keep",
        schema_name="ExperimentImplementationHandoff",
        schema_version="1.0",
        depends_on=[
            "nodes/repair/repo_handoff.json",
            "nodes/repair/validated_repo_handoff.json",
            "nodes/generate/repo_handoff.json",
        ],
    )


def _write_stage_output(state: PaperBenchReproState, filename: str, payload: object) -> None:
    """Persist one stage artifact, routing node-local internals into their owning node."""
    if not state.run_id:
        return
    output_dir = _get_output_dir(state)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_payload = payload.model_dump(mode="json") if hasattr(payload, "model_dump") else payload
    if filename == "terminal_outcome.json":
        _write_terminal_outcome_alias(state, json_payload)
        return
    node_name = _FIXED_NODE_ONLY_ARTIFACTS.get(filename)
    current_node = str(state.current_node or "").strip()
    if node_name is None and current_node in {"prepare", "plan"}:
        node_name = current_node
    if node_name is None and filename in _CURRENT_NODE_ONLY_ARTIFACTS:
        if current_node in {"generate", "repair"}:
            node_name = current_node
    if node_name is not None:
        path = output_dir / "nodes" / node_name / filename
        authority = "source_of_truth" if filename in _PLAN_SOURCE_OF_TRUTH_FILENAMES else "derived"
        write_artifact(
            run_dir=output_dir,
            path=path,
            payload=json_payload,
            logical_name=filename.rsplit(".", 1)[0],
            kind="contract" if filename in _HANDOFF_STAGE_FILENAMES else "output",
            stage=node_name,
            node=node_name,
            authority=authority,
            retention="keep",
        )
        if filename in _HANDOFF_STAGE_FILENAMES:
            _update_handoff_source_payload(state, filename, json_payload)
            _write_handoff_artifact(state)
        return
    stage_executor._write_stage_output(state, filename, payload)
    if filename in _HANDOFF_STAGE_FILENAMES:
        _update_handoff_source_payload(state, filename, json_payload)
        _write_handoff_artifact(state)


def _read_stage_json(state: PaperBenchReproState, *relative_paths: str) -> Any:
    """Read the first available stage artifact, preferring node-local paths over legacy root paths."""
    output_dir = _get_output_dir(state)
    for relative_path in relative_paths:
        candidates: list[Path]
        if relative_path.startswith("nodes/"):
            candidates = [output_dir / relative_path]
        else:
            candidates = [
                output_dir / "nodes" / "prepare" / relative_path,
                output_dir / "nodes" / "plan" / relative_path,
                output_dir / "nodes" / "generate" / relative_path,
                output_dir / "nodes" / "repair" / relative_path,
                output_dir / relative_path,
            ]
        for path in candidates:
            if path.exists():
                payload = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    ref_path = str(payload.get("artifact_ref") or payload.get("canonical_path") or "").strip()
                    if ref_path:
                        ref_candidate = output_dir / ref_path
                        try:
                            if ref_candidate.exists() and ref_candidate.resolve() != path.resolve():
                                return json.loads(ref_candidate.read_text(encoding="utf-8"))
                        except Exception:
                            pass
                return payload
    raise FileNotFoundError(relative_paths[0] if relative_paths else "")


def _try_read_stage_json(state: PaperBenchReproState, *relative_paths: str) -> Any | None:
    try:
        return _read_stage_json(state, *relative_paths)
    except Exception:
        return None


def _write_repair_round_artifact(
    state: PaperBenchReproState,
    round_id: int,
    filename: str,
    payload: object,
) -> None:
    """Persist per-round repair evidence without overwriting latest node artifacts."""
    if not state.run_id:
        return
    output_dir = _get_output_dir(state)
    json_payload = payload.model_dump(mode="json") if hasattr(payload, "model_dump") else payload
    path = output_dir / "nodes" / "repair" / "rounds" / f"round_{round_id:03d}" / filename
    write_artifact(
        run_dir=output_dir,
        path=path,
        payload=json_payload,
        logical_name=f"repair_round_{round_id:03d}_{filename.rsplit('.', 1)[0]}",
        kind="output",
        stage="repair",
        node="repair",
        authority="derived",
        retention="keep",
    )


def _load_prepare_quality_gate_payload(state: PaperBenchReproState) -> dict[str, Any]:
    state_payload = state.temp_data.get("prepare_quality_gate", {})
    payload = dict(state_payload) if isinstance(state_payload, dict) else {}
    try:
        loaded = _read_stage_json(state, "nodes/prepare/prepare_quality_gate.json", "prepare_quality_gate.json")
    except FileNotFoundError:
        loaded = None
    except Exception as exc:
        raise RuntimeError("prepare quality gate artifact is unreadable; stop before plan/generate") from exc
    if loaded is not None:
        if not isinstance(loaded, dict):
            raise RuntimeError("prepare quality gate artifact is not a JSON object; stop before plan/generate")
        if payload and payload != loaded:
            logger.warning("prepare quality gate artifact differs from state.temp_data; using artifact payload")
        state.temp_data["prepare_quality_gate"] = loaded
        return loaded
    if payload:
        logger.warning("prepare quality gate artifact is missing; ignoring state.temp_data payload")
    return {}


def _ensure_prepare_quality_gate_passed(state: PaperBenchReproState) -> None:
    payload = _load_prepare_quality_gate_payload(state)
    if not payload:
        raise RuntimeError("prepare quality gate is missing; run --stage prepare before plan/generate")
    reasons = list(payload.get("blocking_reasons", []) or [])
    if str(payload.get("schema_version", "") or "") != "1.1":
        reasons.append(
            "prepare quality gate artifact is stale; rerun prepare with claim-inventory coverage checks"
        )
    unit_quality_payload = dict(payload.get("unit_quality", {}) or {})
    if not dict(unit_quality_payload.get("claim_inventory_coverage", {}) or {}):
        reasons.append(
            "prepare quality gate artifact lacks claim-inventory coverage; rerun prepare before plan/generate"
        )
    if payload.get("status") == "passed":
        prepared_reference_count = int(payload.get("prepared_reference_count", 0) or 0)
        active_unit_count = int(payload.get("active_unit_count", 0) or 0)
        grounding = dict(payload.get("reference_grounding", {}) or {})
        missing_grounding_units = list(grounding.get("missing_ref_grounding_units", []) or [])
        if prepared_reference_count > 0 and missing_grounding_units:
            allowed_missing_units = max(4, int(active_unit_count * 0.25)) if active_unit_count else 0
            if len(missing_grounding_units) > allowed_missing_units:
                reasons.append(
                    "prepare quality gate is stale or too weak: "
                    + f"{len(missing_grounding_units)}/{active_unit_count} units lack actionable reference grounding "
                    + f"despite prepared refs; allow at most {allowed_missing_units}. "
                    + "Rerun prepare/plan before generate."
                )
        if not reasons:
            return
    payload = {
        **payload,
        "degraded": True,
        "continue_with_best_effort": True,
        "next_action": "enter_plan_degraded_best_effort",
    }
    state.temp_data["prepare_quality_gate"] = payload
    backlog = state.temp_data.setdefault("degraded_backlog", [])
    degraded_issue = {
        "stage": "prepare_quality_gate",
        "code": "prepare_quality_gate_degraded_continue",
        "message": "prepare quality gate issues are carried forward for validation/repair instead of blocking plan/generate",
        "reasons": reasons[:16],
    }
    if isinstance(backlog, list) and degraded_issue not in backlog:
        backlog.append(degraded_issue)
    logger.warning(
        "prepare quality gate degraded; continuing with best available artifacts: %s",
        "; ".join(str(item) for item in reasons[:6]) or "no blocking reason recorded",
    )


def _restore_boundary_requirements_for_resume(state: PaperBenchReproState) -> BoundaryRequirementsOutput:
    if state.boundary_requirements is not None:
        return state.boundary_requirements

    payload = _try_read_stage_json(state, CANONICAL_ARTIFACTS["boundary_requirements"])
    if isinstance(payload, dict):
        try:
            state.boundary_requirements = BoundaryRequirementsOutput.model_validate(payload)
            return state.boundary_requirements
        except Exception as exc:
            logger.warning("boundary requirements artifact could not be reloaded: %s", exc)

    if state.unit_extraction and list(state.unit_extraction.units or []):
        synthesized = plan_nodes._synthesize_boundary_requirements_from_units(state)
        state.boundary_requirements = BoundaryRequirementsOutput.model_validate(synthesized)
        _write_stage_output(state, CANONICAL_ARTIFACTS["boundary_requirements"], state.boundary_requirements)
        return state.boundary_requirements

    raise RuntimeError(
        "boundary_requirements are missing for resume; restore nodes/plan/boundary_requirements.json "
        "or rerun plan so active units can be synthesized before global_contract_synthesis"
    )


def _hydrate_plan_state_from_artifacts(state: PaperBenchReproState) -> PaperBenchReproState:
    """Best-effort artifact hydration for in-place resume inside the plan node."""
    _restore_boundary_requirements_for_resume(state)
    if not state.reference_repo_surveys:
        payload = _try_read_stage_json(state, "reference_repo_surveys.json")
        if isinstance(payload, list):
            state.reference_repo_surveys = [
                item if hasattr(item, "model_dump") else PreparedReferenceRepositorySurvey.model_validate(item)
                for item in payload
            ]
    if state.topic_profile is None:
        payload = _try_read_stage_json(state, CANONICAL_ARTIFACTS["topic_profile"])
        if isinstance(payload, dict):
            state.topic_profile = TopicProfileOutput.model_validate(payload)
    if state.work_package_planning is None:
        payload = _try_read_stage_json(state, "work_packages.json")
        if isinstance(payload, dict):
            state.work_package_planning = WorkPackagePlanningOutput.model_validate(payload)
    if not state.evidence_bundles:
        payload = _try_read_stage_json(state, "evidence_bundles.json")
        if isinstance(payload, list):
            state.evidence_bundles = [
                item if hasattr(item, "model_dump") else EvidenceBundleOutput.model_validate(item)
                for item in payload
            ]
    if not state.evidence_graph:
        payload = _try_read_stage_json(state, "evidence_graph.json")
        if isinstance(payload, list):
            state.evidence_graph = [
                item if hasattr(item, "model_dump") else EvidenceLinkOutput.model_validate(item)
                for item in payload
            ]
    if state.reference_selection is None:
        payload = _try_read_stage_json(state, CANONICAL_ARTIFACTS["reference_selection"])
        if isinstance(payload, dict):
            state.reference_selection = ReferenceSelectionOutput.model_validate(payload)
    if state.pipeline_plan is None:
        payload = _try_read_stage_json(state, CANONICAL_ARTIFACTS["pipeline_plan"])
        if isinstance(payload, dict):
            state.pipeline_plan = PipelinePlanOutput.model_validate(payload)
    if state.global_contract is None:
        payload = _try_read_stage_json(state, CANONICAL_ARTIFACTS["global_contract"])
        if isinstance(payload, dict):
            state.global_contract = GlobalContractOutput.model_validate(payload)
    if state.architecture is None:
        payload = _try_read_stage_json(state, CANONICAL_ARTIFACTS["architecture"])
        if isinstance(payload, dict):
            state.architecture = ArchitectureOutput.model_validate(payload)
    if state.package_file_planning_output is None:
        payload = _try_read_stage_json(state, CANONICAL_ARTIFACTS["package_file_planning"])
        if isinstance(payload, dict):
            state.package_file_planning_output = PackageFilePlanningOutput.model_validate(payload)
    if state.canonical_ir is None:
        payload = _try_read_stage_json(state, CANONICAL_ARTIFACTS["canonical_ir"])
        if isinstance(payload, dict):
            state.canonical_ir = CanonicalIROutput.model_validate(payload)
    if state.canonical_ir_validation is None:
        payload = _try_read_stage_json(state, CANONICAL_ARTIFACTS["canonical_ir_validation"])
        if isinstance(payload, dict):
            state.canonical_ir_validation = CanonicalIRValidationOutput.model_validate(payload)
            state.planning_failure_layer = str(state.canonical_ir_validation.planning_failure_layer or "").strip()
    return state


def _pipeline_signature() -> str:
    return workflow_runtime._pipeline_signature(__file__)


def _load_stage_status(state: PaperBenchReproState) -> dict[str, Any]:
    return workflow_runtime._load_stage_status(state, get_output_dir=_get_output_dir)


def _write_stage_status(state: PaperBenchReproState, stage_status: dict[str, Any]) -> None:
    workflow_runtime._write_stage_status(
        state,
        stage_status,
        get_output_dir=_get_output_dir,
        json_default=_json_default,
    )


def _save_tracking_artifacts(state: PaperBenchReproState) -> None:
    workflow_runtime.save_tracking_artifacts(
        state,
        get_output_dir=_get_output_dir,
        json_default=_json_default,
        load_stage_status=_load_stage_status,
        structured_stage_backend_label=stage_executor._structured_stage_backend_label,
    )


def _run_or_resume_stage(
    state: PaperBenchReproState,
    stage_name: str,
    input_payload: Any,
    compute,
    load,
    write,
    *,
    invalidate_downstream_on_recompute: bool = True,
):
    return workflow_runtime.run_or_resume_stage(
        state,
        stage_name,
        input_payload,
        compute,
        load,
        write,
        get_output_dir=_get_output_dir,
        json_default=_json_default,
        load_stage_status=_load_stage_status,
        write_stage_status=_write_stage_status,
        save_tracking_artifacts=_save_tracking_artifacts,
        pipeline_signature=_pipeline_signature,
        invalidate_downstream_on_recompute=invalidate_downstream_on_recompute,
    )


def _build_runtime_probe() -> RuntimeProbe:
    return workflow_runtime.build_runtime_probe()


def _boundary_requirements_signature(state: PaperBenchReproState) -> str:
    if state.boundary_requirements is None:
        return ""
    return json.dumps(
        state.boundary_requirements.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
    )


def _refresh_reference_repo_surveys_for_requirements(state: PaperBenchReproState) -> PaperBenchReproState:
    """Rebuild reference surveys after boundary requirements are frozen."""

    if state.boundary_requirements is None:
        return state
    preparation = dict(state.temp_data.get("reference_repo_preparation", {}) or {})
    if not list(preparation.get("prepared_repositories", []) or []):
        return state

    signature = _boundary_requirements_signature(state)
    if (
        signature
        and str(state.temp_data.get("reference_repo_survey_boundary_signature", "") or "") == signature
        and state.reference_repo_surveys
    ):
        return state

    state.reference_repo_surveys = []
    state.temp_data.pop("reference_repo_surveys", None)
    refreshed = _get_reference_repo_surveys(state, state.boundary_requirements)
    state.reference_repo_surveys = [
        item if hasattr(item, "model_dump") else PreparedReferenceRepositorySurvey.model_validate(item)
        for item in list(refreshed or [])
    ]
    state.temp_data["reference_repo_surveys"] = [
        item.model_dump(mode="json")
        for item in state.reference_repo_surveys
    ]
    state.temp_data["reference_repo_survey_boundary_signature"] = signature
    _write_stage_output(
        state,
        "reference_repo_surveys.json",
        [item.model_dump(mode="json") for item in state.reference_repo_surveys],
    )
    _save_tracking_artifacts(state)
    return state


def build_repo_handoff_payload(
    state: PaperBenchReproState,
    *,
    start_stage: str = "rapid_validation",
    thread_id: str = "",
) -> dict[str, Any]:
    return generate_nodes.build_repo_handoff_payload(
        state,
        start_stage=start_stage,
        thread_id=thread_id,
    )


def _start_impl(state: PaperBenchReproState) -> PaperBenchReproState:
    return prepare_nodes.start_impl(
        state,
        new_run_id=_new_run_id,
        get_output_dir=_get_output_dir,
        json_default=_json_default,
        normalize_dataset_requests=_normalize_dataset_requests,
        build_dataset_preparation_payload=_build_dataset_preparation_payload,
        update_input_dataset_status=_update_input_dataset_status,
        update_input_benchmark_status=_update_input_benchmark_status,
        prepare_benchmarks=_prepare_benchmarks,
        prepare_baselines=_prepare_baselines,
        build_resource_manifest=_build_resource_manifest,
        build_runtime_probe=_build_runtime_probe,
        load_stage_status=_load_stage_status,
        write_stage_status=_write_stage_status,
        save_tracking_artifacts=_save_tracking_artifacts,
    )

def _input_normalization_impl(state: PaperBenchReproState) -> PaperBenchReproState:
    def _invoke(stage_name: str, schema_name: str, system: str, user: str) -> dict[str, Any]:
        return _invoke_json_stage_for_state(state, stage_name, schema_name, system, user)
    return prepare_nodes.input_normalization_impl(
        state,
        build_input_normalization_context=_build_input_normalization_context,
        limit_json_for_prompt=_limit_json_for_prompt,
        invoke_json_stage=_invoke,
        get_output_dir=_get_output_dir,
        write_stage_output=_write_stage_output,
        run_or_resume_stage=_run_or_resume_stage,
        save_tracking_artifacts=_save_tracking_artifacts,
    )


def _unit_extraction_impl(state: PaperBenchReproState) -> PaperBenchReproState:
    def _invoke(stage_name: str, schema_name: str, system: str, user: str) -> dict[str, Any]:
        return _invoke_json_stage_for_state(state, stage_name, schema_name, system, user)
    return prepare_nodes.unit_extraction_impl(
        state,
        build_unit_extraction_context=_build_unit_extraction_context,
        limit_json_for_prompt=_limit_json_for_prompt,
        invoke_json_stage=_invoke,
        get_output_dir=_get_output_dir,
        write_stage_output=_write_stage_output,
        run_or_resume_stage=_run_or_resume_stage,
        save_tracking_artifacts=_save_tracking_artifacts,
    )


def _reference_acquisition_impl(state: PaperBenchReproState) -> PaperBenchReproState:
    return prepare_nodes.reference_acquisition_impl(
        state,
        get_output_dir=_get_output_dir,
        get_reference_repo_surveys=_get_reference_repo_surveys,
        build_resource_manifest=_build_resource_manifest,
        write_stage_output=_write_stage_output,
        run_or_resume_stage=_run_or_resume_stage,
        save_tracking_artifacts=_save_tracking_artifacts,
    )


def _topic_profile_impl(state: PaperBenchReproState) -> PaperBenchReproState:
    def _invoke(stage_name: str, schema_name: str, system: str, user: str) -> dict[str, Any]:
        return _invoke_json_stage_for_state(state, stage_name, schema_name, system, user)
    return plan_nodes.topic_profile_impl(
        state,
        build_topic_profile_context=_build_topic_profile_context,
        limit_json_for_prompt=_limit_json_for_prompt,
        invoke_json_stage=_invoke,
        get_output_dir=_get_output_dir,
        write_stage_output=_write_stage_output,
        run_or_resume_stage=_run_or_resume_stage,
        save_tracking_artifacts=_save_tracking_artifacts,
    )


def _work_package_planning_impl(state: PaperBenchReproState) -> PaperBenchReproState:
    def _invoke(stage_name: str, schema_name: str, system: str, user: str) -> dict[str, Any]:
        return _invoke_json_stage_for_state(state, stage_name, schema_name, system, user)
    return plan_nodes.work_package_planning_impl(
        state,
        build_work_package_planning_context=_build_work_package_planning_context,
        build_work_package_local_context=_build_work_package_local_context,
        limit_json_for_prompt=_limit_json_for_prompt,
        invoke_json_stage=_invoke,
        get_output_dir=_get_output_dir,
        json_default=_json_default,
        write_stage_output=_write_stage_output,
        run_or_resume_stage=_run_or_resume_stage,
        save_tracking_artifacts=_save_tracking_artifacts,
    )


def _evidence_grounding_impl(state: PaperBenchReproState) -> PaperBenchReproState:
    return plan_nodes.evidence_grounding_impl(
        state,
        build_evidence_bundles=_build_evidence_bundles,
        get_output_dir=_get_output_dir,
        write_stage_output=_write_stage_output,
        run_or_resume_stage=_run_or_resume_stage,
        save_tracking_artifacts=_save_tracking_artifacts,
    )


def _scope_alignment_impl(state: PaperBenchReproState) -> PaperBenchReproState:
    def _invoke(stage_name: str, schema_name: str, system: str, user: str) -> dict[str, Any]:
        return _invoke_json_stage_for_state(state, stage_name, schema_name, system, user)
    return plan_nodes.scope_alignment_impl(
        state,
        build_boundary_requirements_context=_build_boundary_requirements_context,
        limit_json_for_prompt=_limit_json_for_prompt,
        invoke_json_stage=_invoke,
        get_output_dir=_get_output_dir,
        write_stage_output=_write_stage_output,
    )


def _contract_planning_impl(state: PaperBenchReproState) -> PaperBenchReproState:
    def _invoke(stage_name: str, schema_name: str, system: str, user: str) -> dict[str, Any]:
        return _invoke_json_stage_for_state(state, stage_name, schema_name, system, user)
    return plan_nodes.contract_planning_impl(
        state,
        build_reference_selection_context=_build_reference_selection_context,
        build_pipeline_plan_context=_build_pipeline_plan_context,
        limit_json_for_prompt=_limit_json_for_prompt,
        invoke_json_stage=_invoke,
        get_output_dir=_get_output_dir,
        write_stage_output=_write_stage_output,
        run_or_resume_stage=_run_or_resume_stage,
        save_tracking_artifacts=_save_tracking_artifacts,
    )


def _global_contract_impl(state: PaperBenchReproState) -> PaperBenchReproState:
    def _invoke(stage_name: str, schema_name: str, system: str, user: str) -> dict[str, Any]:
        return _invoke_json_stage_for_state(state, stage_name, schema_name, system, user)
    _restore_boundary_requirements_for_resume(state)
    return plan_nodes.global_contract_impl(
        state,
        build_global_contract_context=_build_global_contract_context,
        limit_json_for_prompt=_limit_json_for_prompt,
        invoke_json_stage=_invoke,
        get_output_dir=_get_output_dir,
        write_stage_output=_write_stage_output,
        run_or_resume_stage=_run_or_resume_stage,
        save_tracking_artifacts=_save_tracking_artifacts,
    )


def _architecture_planning_impl(state: PaperBenchReproState) -> PaperBenchReproState:
    def _invoke(stage_name: str, schema_name: str, system: str, user: str) -> dict[str, Any]:
        return _invoke_json_stage_for_state(state, stage_name, schema_name, system, user)
    return plan_nodes.architecture_planning_impl(
        state,
        build_architecture_context=_build_architecture_context,
        build_architecture_package_context=_build_architecture_package_context,
        limit_json_for_prompt=_limit_json_for_prompt,
        invoke_json_stage=_invoke,
        get_output_dir=_get_output_dir,
        json_default=_json_default,
        write_stage_output=_write_stage_output,
        run_or_resume_stage=_run_or_resume_stage,
        save_tracking_artifacts=_save_tracking_artifacts,
    )


def _package_file_planning_impl(state: PaperBenchReproState) -> PaperBenchReproState:
    def _invoke(stage_name: str, schema_name: str, system: str, user: str) -> dict[str, Any]:
        return _invoke_json_stage_for_state(state, stage_name, schema_name, system, user)
    return plan_nodes.package_file_planning_impl(
        state,
        project_file_plans_from_architecture=lambda architecture, pipeline_plan: _project_file_plans_from_architecture(
            architecture,
            pipeline_plan,
            state=state.model_copy(update={"canonical_ir": None, "canonical_ir_validation": None})
            if bool(getattr(state.input, "resume_in_place", False))
            and str(getattr(state.input, "resume_start_stage", "") or "").strip() in {"architecture_planning", "package_file_planning"}
            else state,
        ),
        close_package_file_plans=lambda architecture, pipeline_plan, file_planning: _close_package_file_plans(
            architecture,
            pipeline_plan,
            file_planning,
            state=state.model_copy(update={"canonical_ir": None, "canonical_ir_validation": None})
            if bool(getattr(state.input, "resume_in_place", False))
            and str(getattr(state.input, "resume_start_stage", "") or "").strip() in {"architecture_planning", "package_file_planning"}
            else state,
        ),
        validate_file_plans=lambda architecture, file_planning: _validate_file_plans(
            architecture,
            file_planning,
            state=state.model_copy(update={"canonical_ir": None, "canonical_ir_validation": None})
            if bool(getattr(state.input, "resume_in_place", False))
            and str(getattr(state.input, "resume_start_stage", "") or "").strip() in {"architecture_planning", "package_file_planning"}
            else state,
        ),
        order_file_plans_for_execution_closure=_order_file_plans_for_execution_closure,
        build_package_file_planning_context=_build_package_file_planning_context,
        build_package_file_planning_local_context=_build_package_file_planning_local_context,
        limit_json_for_prompt=_limit_json_for_prompt,
        invoke_json_stage=_invoke,
        get_output_dir=_get_output_dir,
        json_default=_json_default,
        write_stage_output=_write_stage_output,
        run_or_resume_stage=_run_or_resume_stage,
        save_tracking_artifacts=_save_tracking_artifacts,
        derive_steps_from_file_plans=_derive_steps_from_file_plans,
        render_pipeline_plan_markdown=_render_pipeline_plan_markdown,
    )


def _canonical_ir_shadow_impl(state: PaperBenchReproState) -> PaperBenchReproState:
    """Build and validate the shadow canonical IR as a proper resumable stage."""
    input_payload = {
        "boundary_requirements": (
            state.boundary_requirements.model_dump(mode="json") if state.boundary_requirements else {}
        ),
        "work_package_planning": (
            state.work_package_planning.model_dump(mode="json") if state.work_package_planning else {}
        ),
        "global_contract": (
            state.global_contract.model_dump(mode="json") if state.global_contract else {}
        ),
        "architecture": (
            state.architecture.model_dump(mode="json") if state.architecture else {}
        ),
        "package_file_planning": (
            state.package_file_planning_output.model_dump(mode="json") if state.package_file_planning_output else {}
        ),
        "unit_extraction": (
            state.unit_extraction.model_dump(mode="json") if state.unit_extraction else {}
        ),
        "pipeline_plan": (
            state.pipeline_plan.model_dump(mode="json") if state.pipeline_plan else {}
        ),
    }

    def _compute() -> dict[str, Any]:
        ir = _build_canonical_ir(state)
        ir_validation = _validate_canonical_ir(state, ir)
        return {
            "canonical_ir": ir.model_dump(mode="json"),
            "canonical_ir_validation": ir_validation.model_dump(mode="json"),
        }

    def _load() -> dict[str, Any]:
        return {
            "canonical_ir": _read_stage_json(state, CANONICAL_ARTIFACTS["canonical_ir"]),
            "canonical_ir_validation": _read_stage_json(state, CANONICAL_ARTIFACTS["canonical_ir_validation"]),
        }

    def _write(result: dict[str, Any]) -> None:
        from reproagent.pipeline.schemas import CanonicalIROutput, CanonicalIRValidationOutput
        ir = CanonicalIROutput.model_validate(result["canonical_ir"])
        ir_validation = CanonicalIRValidationOutput.model_validate(result["canonical_ir_validation"])
        state.canonical_ir = ir
        state.canonical_ir_validation = ir_validation
        state.planning_failure_layer = str(ir_validation.planning_failure_layer or "").strip()
        _write_stage_output(state, CANONICAL_ARTIFACTS["canonical_ir"], ir)
        _write_stage_output(
            state,
            CANONICAL_ARTIFACTS["semantic_assertions"],
            {
                "semantic_assertions": [
                    item.model_dump(mode="json")
                    for item in ir.semantic_assertions
                ],
                "evidence_contracts": [
                    item.model_dump(mode="json")
                    for item in ir.evidence_contracts
                ],
                "validator_expectations": [
                    item.model_dump(mode="json")
                    for item in ir.validator_expectations
                ],
            },
        )
        _write_stage_output(state, CANONICAL_ARTIFACTS["canonical_ir_validation"], ir_validation)
        _write_stage_output(
            state,
            CANONICAL_ARTIFACTS["semantic_validation_report"],
            ir_validation.semantic_validation_report,
        )

    result = _run_or_resume_stage(
        state,
        "canonical_ir_synthesis",
        input_payload,
        _compute,
        _load,
        _write,
        invalidate_downstream_on_recompute=False,
    )
    from reproagent.pipeline.schemas import CanonicalIROutput, CanonicalIRValidationOutput
    state.canonical_ir = CanonicalIROutput.model_validate(result["canonical_ir"])
    state.canonical_ir_validation = CanonicalIRValidationOutput.model_validate(result["canonical_ir_validation"])
    state.planning_failure_layer = str(state.canonical_ir_validation.planning_failure_layer or "").strip()
    _save_tracking_artifacts(state)
    return state


def _refresh_canonical_ir_for_repair_context(state: PaperBenchReproState) -> PaperBenchReproState:
    """Rebuild deterministic semantic IR before repair without rerunning generation."""
    if (
        state.boundary_requirements is None
        or state.work_package_planning is None
        or state.package_file_planning_output is None
    ):
        return state
    ir = _build_canonical_ir(state)
    ir_validation = _validate_canonical_ir(state, ir)
    state.canonical_ir = ir
    state.canonical_ir_validation = ir_validation
    state.planning_failure_layer = str(ir_validation.planning_failure_layer or "").strip()
    if state.run_id:
        output_dir = _get_output_dir(state)
        plan_node_dir = output_dir / "nodes" / "plan"
        semantic_assertions_payload = {
            "semantic_assertions": [
                item.model_dump(mode="json")
                for item in ir.semantic_assertions
            ],
            "evidence_contracts": [
                item.model_dump(mode="json")
                for item in ir.evidence_contracts
            ],
            "validator_expectations": [
                item.model_dump(mode="json")
                for item in ir.validator_expectations
            ],
        }
        for filename, payload in [
            (CANONICAL_ARTIFACTS["canonical_ir"], ir),
            (CANONICAL_ARTIFACTS["semantic_assertions"], semantic_assertions_payload),
            (CANONICAL_ARTIFACTS["canonical_ir_validation"], ir_validation),
            (CANONICAL_ARTIFACTS["semantic_validation_report"], ir_validation.semantic_validation_report),
        ]:
            write_artifact(
                run_dir=output_dir,
                path=plan_node_dir / filename,
                payload=payload,
                logical_name=filename.rsplit(".", 1)[0],
                kind="contract" if filename.endswith(".json") else "output",
                stage="plan",
                node="plan",
                authority="source_of_truth" if filename == CANONICAL_ARTIFACTS["canonical_ir"] else "derived",
                retention="keep",
            )
    return state


def normalization_gate_node(state: PaperBenchReproState) -> PaperBenchReproState:
    """Canonical planning closure gate before generate."""
    if state.canonical_ir is None or state.canonical_ir_validation is None:
        state = _canonical_ir_shadow_impl(state)
    if state.canonical_ir_validation is None:
        return state

    gate_payload = state.canonical_ir_validation.model_dump(mode="json")
    gate_actions = set(gate_payload.get("gate_actions", []) or [])
    structural_failures = []
    if state.canonical_ir is not None:
        if not list(state.canonical_ir.requirements or []):
            structural_failures.append("canonical IR has no requirements")
        if not list(state.canonical_ir.work_packages or []):
            structural_failures.append("canonical IR has no work packages")
        if not list(state.canonical_ir.file_nodes or []):
            structural_failures.append("canonical IR has no registered files")
    if structural_failures:
        state.temp_data["normalization_gate"] = {
            "status": "blocked",
            "reason": "; ".join(structural_failures),
            "planning_failure_layer": state.canonical_ir_validation.planning_failure_layer,
            "mismatch_summary": dict(state.canonical_ir_validation.mismatch_summary),
        }
        state.status = "failed"
        state.terminal_outcome = "failed"
        state.terminal_outcome_reason = "normalization gate failed; stop before generate"
        state.failed_node = "normalization_gate"
        state.error_message = "; ".join(structural_failures)
    elif "retry_generate" in gate_actions and not gate_payload.get("passed", True):
        reason = "canonical IR closure has retry_generate mismatches"
        mismatch_summary = dict(state.canonical_ir_validation.mismatch_summary)
        state.temp_data["normalization_gate"] = {
            "status": "degraded_continue",
            "reason": reason,
            "planning_failure_layer": state.canonical_ir_validation.planning_failure_layer,
            "mismatch_summary": mismatch_summary,
            "gate_actions": list(gate_actions),
        }
        backlog = state.temp_data.setdefault("degraded_backlog", [])
        degraded_issue = {
            "stage": "normalization_gate",
            "code": "canonical_ir_shadow_retry_generate_continue",
            "message": reason,
            "planning_failure_layer": state.canonical_ir_validation.planning_failure_layer,
            "mismatch_summary": mismatch_summary,
        }
        if isinstance(backlog, list) and degraded_issue not in backlog:
            backlog.append(degraded_issue)
        logger.warning(
            "normalization gate degraded; continuing to generate with canonical IR mismatches: %s",
            mismatch_summary,
        )
    else:
        state.temp_data["normalization_gate"] = {
            "status": "passed",
            "planning_failure_layer": state.canonical_ir_validation.planning_failure_layer,
            "mismatch_summary": dict(state.canonical_ir_validation.mismatch_summary),
        }
    _save_tracking_artifacts(state)
    return state


def _merge_repair_plan(state: PaperBenchReproState, draft: RepairPlanDraft, review: RepairPlanReview) -> RepairPlan:
    selected_files = (
        list(review.validated_files)
        if review.validated_files or review.approved
        else list(draft.preferred_files)
    )
    selected_work_packages = (
        list(review.validated_work_packages)
        if review.validated_work_packages or review.approved
        else list(draft.preferred_work_packages)
    )
    recommended_surfaces = list(
        dict.fromkeys([
            *list(draft.recommended_surfaces),
            *list(draft.preferred_files),
            *list(draft.preferred_work_packages),
            *list(review.accepted_surfaces),
            *list(review.validated_files),
            *list(review.validated_work_packages),
        ])
    )
    semantic_guardrails = list(dict.fromkeys([
        *draft.semantic_guardrails,
        *draft.semantic_must_keep,
        *review.semantic_risks,
    ]))
    runtime_guardrails = list(dict.fromkeys([*draft.runtime_guardrails, *review.runtime_risks]))
    requested_budget = max(1, int(review.round_budget or draft.round_budget or 1))
    return RepairPlan(
        summary=review.summary or draft.summary,
        problem_list=list(dict.fromkeys([*draft.problem_list, *draft.failure_focus])),
        semantic_guardrails=semantic_guardrails,
        runtime_guardrails=runtime_guardrails,
        recommended_surfaces=recommended_surfaces,
        selected_files=list(dict.fromkeys(selected_files)),
        selected_work_packages=list(dict.fromkeys(selected_work_packages)),
        forbidden_shortcuts=list(dict.fromkeys(draft.forbidden_shortcuts)),
        repair_guidance=list(dict.fromkeys([
            *draft.repair_guidance,
            *draft.generation_guidance,
            *draft.evaluation_guidance,
        ])),
        review_points=list(dict.fromkeys(review.required_review_points)),
        acceptance_criteria=list(dict.fromkeys([*draft.acceptance_criteria, *review.acceptance_criteria])),
        round_budget=max(1, requested_budget),
    )


def _repair_context_dict(context: object) -> dict[str, Any]:
    if isinstance(context, dict):
        return context
    if isinstance(context, str):
        try:
            parsed = json.loads(context)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _fallback_repair_plan_from_context(state: PaperBenchReproState, context: object) -> RepairPlan:
    """Build a deterministic repair plan when structured repair planning times out."""
    context = _repair_context_dict(context)
    findings = list((context.get("repair_eval_snapshot") or {}).get("top_findings") or [])
    ticket = dict(context.get("repair_ticket") or {})
    validation = dict(context.get("validation_snapshot") or {})
    runtime_blockers = [
        item for item in list(context.get("runtime_first_blockers") or []) if isinstance(item, dict)
    ]
    surfaces = [
        str(item).strip()
        for item in list(context.get("recommended_surfaces") or [])
        if str(item).strip()
    ]
    runtime_surfaces = [
        str(surface).strip()
        for blocker in runtime_blockers
        for surface in list(blocker.get("surfaces") or [])
        if str(surface).strip()
    ]
    if runtime_surfaces:
        surfaces = list(dict.fromkeys([*runtime_surfaces, *surfaces]))
    problem_list = [
        str(item.get("summary") or item.get("finding_id") or "").strip()
        for item in findings
        if isinstance(item, dict) and str(item.get("summary") or item.get("finding_id") or "").strip()
    ]
    problem_list.extend(
        str(item.get("error") or item.get("category") or "").strip()
        for item in runtime_blockers
        if str(item.get("error") or item.get("category") or "").strip()
    )
    problem_list.extend(str(item).strip() for item in list(validation.get("blocked_reasons") or []) if str(item).strip())
    guidance = [
        "Patch only the failing repo surfaces identified by validation.",
        "Preserve rubric intent and do not remove semantic coverage to pass checks.",
        "Use bounded smoke fixtures only when clearly labeled as validation artifacts, never as claimed paper results.",
    ]
    if runtime_blockers:
        guidance.insert(
            0,
            "Runtime-first mode: fix the exact startup traceback/import/syntax blocker before expanding semantic or artifact changes.",
        )
    guidance.extend(str(item).strip() for item in list(validation.get("repair_recommendations") or []) if str(item).strip())
    requirement_anchor = dict(context.get("requirement_anchor") or {})
    return RepairPlan(
        summary="Deterministic repair plan from validation findings.",
        problem_list=list(dict.fromkeys(problem_list))[:12],
        semantic_guardrails=list(dict.fromkeys(
            str(item).strip()
            for item in list(requirement_anchor.get("semantic_invariants") or [])
            if str(item).strip()
        ))[:12],
        runtime_guardrails=list(dict.fromkeys(
            str(item).strip()
            for item in list(requirement_anchor.get("runtime_invariants") or [])
            if str(item).strip()
        ))[:12],
        recommended_surfaces=list(dict.fromkeys(surfaces))[:16],
        selected_files=list(dict.fromkeys(
            str(item).strip()
            for item in list(ticket.get("required_fix_targets") or []) + surfaces
            if str(item).strip() and "." in str(item)
        ))[:12],
        selected_work_packages=list(dict.fromkeys(
            str(item).strip()
            for item in list(ticket.get("next_fix_scope") or []) + surfaces
            if str(item).strip() and str(item).startswith("wp_")
        ))[:12],
        forbidden_shortcuts=list(dict.fromkeys(
            str(item).strip()
            for item in list(ticket.get("forbidden_changes") or [])
            if str(item).strip()
        ))[:12],
        repair_guidance=list(dict.fromkeys(guidance))[:16],
        review_points=[
            "Rerun repair_validation and inspect failed semantic/integration checks before another repair.",
            "Confirm generated artifacts contain numeric validation values only when labeled as bounded smoke outputs.",
        ],
        acceptance_criteria=list(dict.fromkeys(
            str(item).strip()
            for item in list(requirement_anchor.get("acceptance_signals") or [])
            if str(item).strip()
        ))[:12],
        round_budget=2 if runtime_blockers else 1,
    )


def _build_requirement_anchor(state: PaperBenchReproState) -> RequirementAnchor:
    if semantic_anchor_disabled():
        return RequirementAnchor(source="semantic_anchor_ablation")
    requirements = list((state.boundary_requirements.boundary_requirements if state.boundary_requirements else []) or [])
    normalized_target = dict(state.temp_data.get("input_normalization", {}) or {})
    requirement_titles = [
        str(item.title or item.description or "").strip()
        for item in requirements
        if str(item.title or item.description or "").strip()
    ]
    acceptance_criteria = [
        criterion
        for requirement in requirements
        for criterion in list(requirement.acceptance_criteria or [])
        if str(criterion).strip()
    ]
    canonical_semantic_statements = [
        str(item.statement or "").strip()
        for item in list(state.canonical_ir.semantic_assertions if state.canonical_ir else [])
        if str(item.statement or "").strip()
    ]
    unit_obligations = [
        str(item).strip()
        for unit in list(state.unit_extraction.units if state.unit_extraction else [])
        for item in [
            unit.statement,
            *list(unit.code_obligations or [])[:3],
            *list(unit.expected_artifacts or [])[:2],
        ]
        if str(item).strip()
    ]
    runtime_invariants = list(dict.fromkeys([
        *(state.project_plan.entrypoints.values() if state.project_plan else []),
        *list(state.global_contract.validation_gates if state.global_contract else []),
    ]))
    required_artifacts = list(dict.fromkeys([
        target.name
        for target in list(state.global_contract.result_targets if state.global_contract else [])
        if str(target.name or "").strip()
    ]))
    semantic_invariants = list(dict.fromkeys([
        str(normalized_target.get("normalized_target", "") or "").strip(),
        str(normalized_target.get("target_summary", "") or "").strip(),
        *requirement_titles,
        *canonical_semantic_statements,
        *unit_obligations,
        *(state.global_contract.contract_notes if state.global_contract else []),
    ]))
    semantic_invariants.extend(
        str(item).strip()
        for item in list(normalized_target.get("explicit_constraints", []) or [])
        if str(item).strip()
    )
    forbidden_shortcuts = [
        "Do not replace the experiment with a toy demo or mock-only pipeline.",
        "Do not fabricate artifacts, reports, or evaluation outputs to satisfy validation.",
        "Do not delete stable interfaces or runtime entrypoints to avoid fixing the real issue.",
    ]
    return RequirementAnchor(
        source="plan_boundary_requirements",
        summary=(
            str(normalized_target.get("target_summary", "") or "").strip()
            or str(state.input.target or "").strip()
        )[:400],
        goal=str(state.input.target or "").strip(),
        semantic_invariants=[item for item in semantic_invariants if str(item).strip()][:20],
        runtime_invariants=[str(item).strip() for item in runtime_invariants if str(item).strip()][:10],
        required_artifacts=[str(item).strip() for item in required_artifacts if str(item).strip()][:10],
        forbidden_shortcuts=forbidden_shortcuts,
        acceptance_signals=[str(item).strip() for item in acceptance_criteria if str(item).strip()][:10],
    )


def _build_repair_eval_report(state: PaperBenchReproState) -> RepairEvalReport:
    report = state.validation_report or ValidationReport()
    findings: list[RepairEvalFinding] = []
    failed_checks = _all_failed_validation_checks(report)
    for index, check in enumerate(failed_checks, start=1):
        category = str(check.category or "integration").strip().lower()
        normalized_category = (
            "semantic" if category == "semantic" else
            "runtime" if category in {"implementation", "trace"} else
            "artifact" if category == "artifact" else
            "integration"
        )
        assertion_ids = [
            str(check.name).split("semantic:", 1)[1]
            for _ in [None]
            if str(check.name or "").startswith("semantic:")
        ]
        severity = "critical" if normalized_category in {"runtime", "integration"} else "high"
        findings.append(
            RepairEvalFinding(
                finding_id=f"finding_{index:03d}",
                category=normalized_category,
                severity=severity,
                summary=str(check.name or check.details or "repo validation failure").strip(),
                evidence=[str(check.details or "").strip()],
                affected_surfaces=[
                    *[str(item).strip() for item in list(check.affected_files or []) if str(item).strip()],
                    *[str(item).strip() for item in list(check.affected_work_packages or []) if str(item).strip()],
                ][:8],
                suggested_focus=[
                    *[str(item).strip() for item in list(check.affected_files or []) if str(item).strip()],
                    *[str(item).strip() for item in list(check.affected_units or []) if str(item).strip()],
                ][:8],
                assertion_ids=assertion_ids,
                related_files=[str(item).strip() for item in list(check.affected_files or []) if str(item).strip()][:8],
                failure_layer=(
                    report.planning_failure_layer
                    if normalized_category in {"semantic", "integration"}
                    else "repo_runtime"
                ),
                fix_hint=(
                    "Planning artifacts may be unreliable; patch or rewrite repo surfaces, entrypoints, and artifact wiring to satisfy the target and runtime handoff."
                    if normalized_category in {"semantic", "integration"} and report.planning_failure_layer
                    else "Patch the indicated repo surfaces and rerun the same validation checks."
                ),
            )
        )
    findings = sorted(
        findings,
        key=lambda item: (
            {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(item.severity, 9),
            item.finding_id,
        ),
    )[:8]
    anti_shortcut_status = "violated" if any(
        token in " ".join(report.blocked_reasons + report.repair_recommendations).lower()
        for token in ("mock", "placeholder", "fake", "toy demo")
    ) else "clean"
    return RepairEvalReport(
        summary=(
            "Current repo is not yet ready for handoff; keep semantic alignment to the requirement anchor "
            "while closing runtime and integration failures."
        ),
        semantic_status="aligned" if not report.failure_categories else "drift_risk",
        runtime_status="closed" if report.passed else ("partial" if report.dynamic_status == "success" else "broken"),
        anti_shortcut_status=anti_shortcut_status,
        findings=findings,
        must_keep=list((state.requirement_anchor.semantic_invariants if state.requirement_anchor else [])[:8]),
        repair_focus=list(dict.fromkeys(
            [item.summary for item in findings[:6]]
            + list(report.blocked_reasons[:4])
            + list(report.repair_recommendations[:4])
        ))[:10],
        forbidden_shortcuts=list((state.requirement_anchor.forbidden_shortcuts if state.requirement_anchor else [])),
    )


def _build_repair_findings_payload(state: PaperBenchReproState) -> dict[str, Any]:
    report = state.repair_eval_report or _build_repair_eval_report(state)
    return {
        "schema_version": "1.0",
        "failure_layer": (
            state.validation_report.planning_failure_layer
            if state.validation_report is not None
            else state.planning_failure_layer
        ),
        "findings": [item.model_dump(mode="json") for item in report.findings],
        "semantic_validation_report": (
            state.validation_report.semantic_validation_report
            if state.validation_report is not None
            else (
                state.canonical_ir_validation.semantic_validation_report
                if state.canonical_ir_validation is not None
                else {}
            )
        ),
    }


def _build_repair_plan_context(state: PaperBenchReproState) -> dict[str, Any]:
    requirement_anchor = state.requirement_anchor or _build_requirement_anchor(state)
    repair_eval_report = state.repair_eval_report or _build_repair_eval_report(state)
    validation_report = state.validation_report or ValidationReport()
    repair_ticket = state.repair_ticket or RepairTicket()
    repair_findings = _build_repair_findings_payload(state)
    evidence_contract = dict(state.temp_data.get("post_generate_evidence_contract", {}) or {})
    repo_plan = state.repo_plan
    project_plan = state.project_plan
    repo_root = str(Path(state.project_root).resolve()) if state.project_root else str((_get_output_dir(state) / "repo").resolve())
    repo_file_index = sorted(list(state.generated_files or []))
    entrypoints = {}
    if project_plan is not None:
        entrypoints = {
            key: str(value).strip()
            for key, value in dict(project_plan.entrypoints or {}).items()
            if str(value).strip()
        }
    runtime_first_blockers = _runtime_first_blockers_from_state(state, validation_report)
    runtime_first_surfaces = _runtime_first_surfaces(state, runtime_first_blockers)
    recommended_surfaces = list(dict.fromkeys([
        *runtime_first_surfaces,
        *list(repair_ticket.next_fix_scope or []),
        *[surface for finding in repair_eval_report.findings for surface in list(finding.affected_surfaces or [])],
        *list(entrypoints.values()),
    ]))[:16]
    return {
        "target": str(state.input.target or "").strip(),
        "repair_round": int(state.temp_data.get("repair_round", 0) or 0),
        "repo_root": repo_root,
        "repo_index": {
            "entrypoints": entrypoints,
            "top_level_files": repo_file_index[:40],
            "file_count": len(repo_file_index),
        },
        "requirement_anchor": requirement_anchor.model_dump(mode="json"),
        "repair_eval_snapshot": {
            "summary": repair_eval_report.summary,
            "semantic_status": repair_eval_report.semantic_status,
            "runtime_status": repair_eval_report.runtime_status,
            "anti_shortcut_status": repair_eval_report.anti_shortcut_status,
            "top_findings": [item.model_dump(mode="json") for item in repair_eval_report.findings[:6]],
            "repair_focus": list(repair_eval_report.repair_focus[:8]),
        },
        "repair_findings": repair_findings,
        "runtime_first_blockers": runtime_first_blockers,
        "runtime_first_surfaces": runtime_first_surfaces,
        "validation_snapshot": {
            "passed": validation_report.passed,
            "overall_status": validation_report.overall_status,
            "static_status": validation_report.static_status,
            "smoke_status": validation_report.smoke_status,
            "dynamic_status": validation_report.dynamic_status,
            "failure_categories": list(validation_report.failure_categories[:6]),
            "blocked_reasons": list(validation_report.blocked_reasons[:6]),
            "repair_recommendations": list(validation_report.repair_recommendations[:6]),
        },
        "repair_ticket": {
            "failure_type": repair_ticket.failure_type,
            "reason": repair_ticket.reason,
            "trigger_signals": list(repair_ticket.trigger_signals[:6]),
            "evidence": dict(repair_ticket.evidence or {}),
            "required_fix_targets": list(repair_ticket.required_fix_targets[:8]),
            "next_fix_scope": list(repair_ticket.next_fix_scope[:8]),
            "forbidden_changes": list(repair_ticket.forbidden_changes[:8]),
        },
        "post_generate_evidence_contract": evidence_contract,
        "repo_contract_index": {
            "artifact_targets": (
                [
                    item.relative_path
                    for item in list(repo_plan.artifact_contract or [])
                    if str(item.relative_path or "").strip()
                ][:10]
                if repo_plan is not None
                else []
            ),
            "entrypoints": (
                [str(item).strip() for item in list(repo_plan.entrypoints or []) if str(item).strip()][:10]
                if repo_plan is not None
                else []
            ),
            "work_packages": (
                [
                    {
                        "work_package_id": item.work_package_id,
                        "goal": item.goal,
                        "produces": list(item.produces[:4]),
                    }
                    for item in list(repo_plan.work_packages or [])[:10]
                ]
                if repo_plan is not None
                else []
            ),
        },
        "recommended_surfaces": recommended_surfaces,
        "planning_review_issues": {
            "architecture": (
                list(state.architecture.unresolved_review_failures[:6])
                if state.architecture is not None
                else []
            ),
            "package_file_planning": (
                list(state.package_file_planning_output.unresolved_review_failures[:6])
                if state.package_file_planning_output is not None
                else []
            ),
        },
        "degraded_backlog": list(state.temp_data.get("degraded_backlog", []) or [])[:16],
    }


def _repo_level_repair_review(state: PaperBenchReproState) -> dict[str, Any]:
    validation_report = state.validation_report or ValidationReport()
    requirement_anchor = state.requirement_anchor or RequirementAnchor()
    repair_plan = state.repair_plan or RepairPlan()
    blocked_reasons = [str(item).strip() for item in list(validation_report.blocked_reasons or []) if str(item).strip()]
    repair_recommendations = [
        str(item).strip()
        for item in list(validation_report.repair_recommendations or [])
        if str(item).strip()
    ]
    semantic_issues = [
        item
        for item in list(repair_plan.semantic_guardrails or []) + list(requirement_anchor.semantic_invariants or [])
        if str(item).strip()
    ]
    runtime_issues = [
        item
        for item in list(repair_plan.runtime_guardrails or []) + list(requirement_anchor.runtime_invariants or [])
        if str(item).strip()
    ]
    anti_shortcut_hits = [
        item
        for item in list(repair_plan.forbidden_shortcuts or []) + list(requirement_anchor.forbidden_shortcuts or [])
        if any(token in " ".join(blocked_reasons + repair_recommendations).lower() for token in str(item).lower().split()[:3])
    ]
    semantic_passed = bool(validation_report.passed or validation_report.overall_status in {"partial", "passed"})
    runtime_passed = bool(validation_report.passed)
    anti_shortcut_passed = state.repair_eval_report is None or state.repair_eval_report.anti_shortcut_status == "clean"
    return {
        "passed": semantic_passed and runtime_passed and anti_shortcut_passed,
        "semantic": {
            "passed": semantic_passed,
            "guardrails": semantic_issues[:24],
            "issues": blocked_reasons[:8],
        },
        "runtime": {
            "passed": runtime_passed,
            "guardrails": runtime_issues[:24],
            "issues": repair_recommendations[:8],
            "overall_status": validation_report.overall_status,
            "dynamic_status": validation_report.dynamic_status,
        },
        "anti_shortcut": {
            "passed": anti_shortcut_passed,
            "issues": anti_shortcut_hits[:12],
        },
        "acceptance_criteria": list(repair_plan.acceptance_criteria or requirement_anchor.acceptance_signals or [])[:24],
    }


def _build_repair_review_payload(
    state: PaperBenchReproState,
    *,
    semantic_review: dict[str, Any] | None = None,
    runtime_validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    semantic_payload = dict(semantic_review or state.temp_data.get("repair_plan_review", {}) or {})
    runtime_payload = dict(runtime_validation or {})
    validation_report = state.validation_report.model_dump(mode="json") if state.validation_report else {}
    preflight_result = state.preflight_result.model_dump(mode="json") if state.preflight_result else {}
    execution_result = state.execution_result.model_dump(mode="json") if state.execution_result else {}
    runtime_probe = state.runtime_probe.model_dump(mode="json") if state.runtime_probe else {}
    repair_ticket = state.repair_ticket.model_dump(mode="json") if state.repair_ticket else {}
    planning_review_issues = {
        "architecture": (
            list(state.architecture.unresolved_review_failures)
            if state.architecture is not None
            else []
        ),
        "package_file_planning": (
            list(state.package_file_planning_output.unresolved_review_failures)
            if state.package_file_planning_output is not None
            else []
        ),
    }
    semantic_present = bool(semantic_payload)
    semantic_passed = bool(semantic_payload.get("approved", False)) if semantic_present else True
    runtime_present = bool(runtime_payload or validation_report)
    runtime_passed = bool(validation_report.get("passed", False)) if runtime_present else False
    repo_level_review = _repo_level_repair_review(state)
    return {
        "passed": bool(repo_level_review.get("passed", False)),
        "requirement_anchor": (
            state.requirement_anchor.model_dump(mode="json") if state.requirement_anchor else {}
        ),
        "repair_eval_report": (
            state.repair_eval_report.model_dump(mode="json") if state.repair_eval_report else {}
        ),
        "repo_level_review": repo_level_review,
        "planning_review_issues": planning_review_issues,
        "semantic_review": {
            "present": semantic_present,
            "approved": semantic_passed,
            "summary": str(semantic_payload.get("summary", "") or ""),
            "semantic_risks": list(semantic_payload.get("semantic_risks", []) or []),
            "required_review_points": list(semantic_payload.get("required_review_points", []) or []),
            "runtime_risks": list(semantic_payload.get("runtime_risks", []) or []),
            "accepted_surfaces": list(semantic_payload.get("accepted_surfaces", []) or []),
            "acceptance_criteria": list(semantic_payload.get("acceptance_criteria", []) or []),
            "round_budget": int(semantic_payload.get("round_budget", 0) or 0),
            "raw": semantic_payload,
        },
        "runtime_validation": {
            "present": runtime_present,
            "passed": runtime_passed,
            "overall_status": str(validation_report.get("overall_status", "") or ""),
            "static_status": str(validation_report.get("static_status", "") or ""),
            "static_contract_status": str(validation_report.get("static_contract_status", "") or ""),
            "smoke_status": str(validation_report.get("smoke_status", "") or ""),
            "dynamic_status": str(validation_report.get("dynamic_status", "") or ""),
            "failure_categories": list(validation_report.get("failure_categories", []) or []),
            "blocked_reasons": list(validation_report.get("blocked_reasons", []) or []),
            "repair_recommendations": list(validation_report.get("repair_recommendations", []) or []),
            "preflight_result": preflight_result,
            "execution_result": execution_result,
            "runtime_probe": runtime_probe,
            "repair_ticket": repair_ticket,
            "raw": runtime_payload or {
                "validation_report": validation_report,
                "preflight_result": preflight_result,
                "execution_result": execution_result,
                "runtime_probe": runtime_probe,
                "repair_ticket": repair_ticket,
            },
        },
    }


def _repair_plan_impl(state: PaperBenchReproState) -> PaperBenchReproState:
    logger.info("repair_plan - Building semantic repair plan from upstream context and current repo failures...")
    if semantic_anchor_disabled():
        state.requirement_anchor = None
    elif state.requirement_anchor is None:
        state.requirement_anchor = _build_requirement_anchor(state)
    state.repair_eval_report = _build_repair_eval_report(state)
    repair_findings_payload = _build_repair_findings_payload(state)
    input_payload = {
        "target": state.input.target,
        "upstream_intent": upstream_intent_payload(state),
        "repair_round": int(state.temp_data.get("repair_round", 0) or 0),
        "requirement_anchor": state.requirement_anchor.model_dump(mode="json") if state.requirement_anchor else {},
        "repair_eval_report": state.repair_eval_report.model_dump(mode="json") if state.repair_eval_report else {},
        "repair_findings": repair_findings_payload,
        "validation_report": state.validation_report.model_dump(mode="json") if state.validation_report else {},
        "repair_ticket": state.repair_ticket.model_dump(mode="json") if state.repair_ticket else {},
        "repo_plan": state.repo_plan.model_dump(mode="json") if state.repo_plan else {},
        "planning_review_issues": {
            "architecture": (
                list(state.architecture.unresolved_review_failures)
                if state.architecture is not None
                else []
            ),
            "package_file_planning": (
                list(state.package_file_planning_output.unresolved_review_failures)
                if state.package_file_planning_output is not None
                else []
            ),
        },
    }

    def _compute() -> dict[str, Any]:
        stage_executor._reset_stage_invocation_usages()
        context = _build_repair_plan_context(state)
        limited_context = _limit_json_for_prompt(context)
        _write_stage_output(state, "repair_plan_context.json", limited_context)
        if state.requirement_anchor is not None and not semantic_anchor_disabled():
            _write_stage_output(state, "requirement_anchor.json", state.requirement_anchor)
        if state.repair_eval_report is not None:
            _write_stage_output(state, "repair_eval_report.json", state.repair_eval_report)
        _write_stage_output(state, "repair_findings.json", repair_findings_payload)
        system, user = build_repair_plan_prompt(limited_context, language=state.input.language)
        try:
            draft = RepairPlanDraft.model_validate(
                _invoke_json_stage_for_state(state, "repair_plan_gen", "RepairPlanDraft", system, user)
            )
        except (TimeoutError, Exception) as exc:
            repair_plan = _fallback_repair_plan_from_context(state, context)
            return {
                "context": limited_context,
                "draft": {"fallback": True, "reason": str(exc)},
                "review": {"approved": True, "summary": "Structured repair planning failed; used deterministic validation-driven fallback."},
                "repair_plan": repair_plan.model_dump(mode="json"),
                "agent_usage_summary": stage_executor._consume_stage_invocation_usage_summary(),
            }
        review_context = _limit_json_for_prompt(
            {
                **context,
                "repair_plan_draft": draft.model_dump(mode="json"),
            }
        )
        review_system, review_user = build_repair_plan_review_prompt(review_context, language=state.input.language)
        try:
            review = RepairPlanReview.model_validate(
                _invoke_json_stage_for_state(state, "repair_plan_eval", "RepairPlanReview", review_system, review_user)
            )
        except (TimeoutError, Exception) as exc:
            repair_plan = _fallback_repair_plan_from_context(state, context)
            return {
                "context": limited_context,
                "draft": draft.model_dump(mode="json"),
                "review": {"approved": True, "summary": f"Structured repair review failed; used deterministic validation-driven fallback: {exc}"},
                "repair_plan": repair_plan.model_dump(mode="json"),
                "agent_usage_summary": stage_executor._consume_stage_invocation_usage_summary(),
            }
        if not review.approved:
            run_memory.record_repair_plan_rejected(
                state,
                review=review.model_dump(mode="json"),
            )
            review_constraints = [
                *[str(item).strip() for item in review.semantic_risks if str(item).strip()],
                *[str(item).strip() for item in review.runtime_risks if str(item).strip()],
                *[str(item).strip() for item in review.required_review_points if str(item).strip()],
            ]
            fallback_draft = draft.model_copy(
                update={
                    "repair_guidance": list(dict.fromkeys([
                        *list(draft.repair_guidance),
                        "Respect the repair-plan evaluation constraints and preserve upstream semantic intent.",
                        *review_constraints,
                    ])),
                    "review_focus": list(dict.fromkeys([
                        *list(draft.review_focus),
                        "Re-check both semantic alignment and runtime validation after applying the repo-wide repair.",
                    ])),
                    "round_budget": max(int(review.round_budget or 0), int(draft.round_budget or 0), 1),
                }
            )
            repair_plan = _merge_repair_plan(state, fallback_draft, review)
        else:
            repair_plan = _merge_repair_plan(state, draft, review)
        repair_plan = _apply_score_feedback_repair_plan(state, repair_plan)
        repair_plan = _apply_runtime_first_repair_plan(state, repair_plan)
        return {
            "context": limited_context,
            "draft": draft.model_dump(mode="json"),
            "review": review.model_dump(mode="json"),
            "repair_plan": repair_plan.model_dump(mode="json"),
            "agent_usage_summary": stage_executor._consume_stage_invocation_usage_summary(),
        }

    def _load() -> dict[str, Any]:
        return {
            "context": _read_stage_json(state, "nodes/repair/repair_plan_context.json", "repair_plan_context.json"),
            "draft": _read_stage_json(state, "nodes/repair/repair_plan_draft.json", "repair_plan_draft.json"),
            "review": _read_stage_json(state, "nodes/repair/repair_plan_review.json", "repair_plan_review.json"),
            "repair_plan": _read_stage_json(state, "nodes/repair/repair_plan.json", "repair_plan.json"),
        }

    def _write(result: dict[str, Any]) -> None:
        if state.requirement_anchor is not None and not semantic_anchor_disabled():
            _write_stage_output(state, "requirement_anchor.json", state.requirement_anchor)
        if state.repair_eval_report is not None:
            _write_stage_output(state, "repair_eval_report.json", state.repair_eval_report)
        state.repair_plan = RepairPlan.model_validate(result.get("repair_plan") or {})
        state.temp_data["repair_plan_context"] = result.get("context", {})
        state.temp_data["repair_plan_draft"] = result.get("draft", {})
        state.temp_data["repair_plan_review"] = result.get("review", {})
        state.temp_data["repair_plan"] = state.repair_plan.model_dump(mode="json")
        _write_stage_output(
            state,
            "repair_review.json",
            _build_repair_review_payload(state, semantic_review=result.get("review", {})),
        )
        _write_stage_output(state, "repair_plan_context.json", result.get("context", {}))
        _write_stage_output(state, "repair_plan_draft.json", result.get("draft", {}))
        _write_stage_output(state, "repair_plan_review.json", result.get("review", {}))
        _write_stage_output(state, "repair_findings.json", repair_findings_payload)
        _write_stage_output(state, "repair_plan.json", state.repair_plan)

    result = _run_or_resume_stage(
        state,
        "repair_plan",
        input_payload,
        _compute,
        _load,
        _write,
    )
    if state.requirement_anchor is not None and not semantic_anchor_disabled():
        _write_stage_output(state, "requirement_anchor.json", state.requirement_anchor)
    if state.repair_eval_report is not None:
        _write_stage_output(state, "repair_eval_report.json", state.repair_eval_report)
    _write_stage_output(state, "repair_findings.json", _build_repair_findings_payload(state))
    state.repair_plan = RepairPlan.model_validate(result.get("repair_plan") or {})
    state.temp_data["repair_plan_context"] = result.get("context", {})
    state.temp_data["repair_plan_draft"] = result.get("draft", {})
    state.temp_data["repair_plan_review"] = result.get("review", {})
    state.temp_data["repair_plan"] = state.repair_plan.model_dump(mode="json")
    _save_tracking_artifacts(state)
    return state


def prepare_node(state: PaperBenchReproState) -> PaperBenchReproState:
    """Prepare normalized inputs, extracted requirements, and references."""
    def _prepare_impl(state: PaperBenchReproState) -> PaperBenchReproState:
        logger.info("prepare - Normalizing inputs and gathering references...")
        state = _start_impl(state)
        state = _input_normalization_impl(state)
        state = _unit_extraction_impl(state)
        state = _reference_acquisition_impl(state)
        logger.info("prepare - Done")
        return state
    return _run_node(state, "prepare", _prepare_impl)


def plan_node(state: PaperBenchReproState) -> PaperBenchReproState:
    """Build the stable repo contract, architecture, and file plan."""
    def _plan_impl(state: PaperBenchReproState) -> PaperBenchReproState:
        logger.info("plan - Building grounded repo contract and file plan...")
        state = _hydrate_plan_state_from_artifacts(state)
        _ensure_prepare_quality_gate_passed(state)
        resume_stage = (
            str(getattr(state.input, "resume_start_stage", "") or "").strip()
            if bool(getattr(state.input, "resume_in_place", False))
            else ""
        )
        plan_stage_order = [
            "topic_profile_synthesis",
            "work_package_planning",
            "package_evidence_grounding",
            "reference_selection",
            "pipeline_plan",
            "global_contract_synthesis",
            "architecture_planning",
            "package_file_planning",
            "canonical_ir_synthesis",
        ]
        active_index = plan_stage_order.index(resume_stage) if resume_stage in plan_stage_order else 0

        if resume_stage not in plan_stage_order or active_index <= 0:
            state = _scope_alignment_impl(state)
            state = _refresh_reference_repo_surveys_for_requirements(state)
        if active_index <= plan_stage_order.index("topic_profile_synthesis"):
            state = _topic_profile_impl(state)
        if active_index <= plan_stage_order.index("work_package_planning"):
            state = _work_package_planning_impl(state)
        if active_index <= plan_stage_order.index("package_evidence_grounding"):
            state, reference_binding_notes = plan_nodes._refresh_work_package_reference_bindings(state)
            if reference_binding_notes:
                _write_stage_output(state, "work_packages.json", state.work_package_planning)
                state.temp_data.setdefault("stage_reviews", {}).setdefault("work_package_planning", {})
                state.temp_data["stage_reviews"]["work_package_planning"] = {
                    **dict(state.temp_data["stage_reviews"].get("work_package_planning", {}) or {}),
                    "reference_binding_notes": reference_binding_notes,
                }
            state = _evidence_grounding_impl(state)
            grounding_fix_budget = _stage_review_repair_budget(state)
            grounding_attempt = 0
            critical_grounding_failures = plan_nodes._critical_grounding_gate_failures(state)
            while critical_grounding_failures and grounding_attempt < grounding_fix_budget:
                grounding_attempt += 1
                state.work_package_planning, grounding_notes = plan_nodes._augment_work_package_references_for_grounding(
                    state,
                    [
                        item.split("`", 2)[1]
                        for item in critical_grounding_failures
                        if "`" in item
                    ],
                )
                if grounding_notes:
                    _write_stage_output(state, "work_packages.json", state.work_package_planning)
                state = _evidence_grounding_impl(state)
                critical_grounding_failures = plan_nodes._critical_grounding_gate_failures(state)
            grounding_review_payload = {
                "stage_name": "package_evidence_grounding",
                "budget": grounding_fix_budget,
                "attempts": grounding_attempt + 1,
                "review_status": "passed" if not critical_grounding_failures else "budget_exhausted_continue",
                "validation_errors": list(critical_grounding_failures),
            }
            workflow_runtime.write_review_artifact(
                state,
                "package_evidence_grounding",
                grounding_review_payload,
                get_output_dir=_get_output_dir,
                json_default=_json_default,
            )
            if critical_grounding_failures:
                state.temp_data.setdefault("stage_reviews", {}).setdefault("package_evidence_grounding", {})
                state.temp_data["stage_reviews"]["package_evidence_grounding"] = grounding_review_payload
                backlog = state.temp_data.setdefault("degraded_backlog", [])
                degraded_issue = {
                    "stage": "package_evidence_grounding",
                    "code": "critical_grounding_degraded_continue",
                    "message": "critical grounding issues remain after stage repair budget; continuing to generate/repair with best available evidence",
                    "reasons": list(critical_grounding_failures)[:16],
                }
                if isinstance(backlog, list) and degraded_issue not in backlog:
                    backlog.append(degraded_issue)
                state.error_message = "; ".join(critical_grounding_failures[:8])
        if active_index <= plan_stage_order.index("pipeline_plan"):
            state = _contract_planning_impl(state)
            state, reference_binding_notes = plan_nodes._refresh_work_package_reference_bindings(state)
            if reference_binding_notes:
                _write_stage_output(state, "work_packages.json", state.work_package_planning)
                state.temp_data.setdefault("stage_reviews", {}).setdefault("work_package_planning", {})
                state.temp_data["stage_reviews"]["work_package_planning"] = {
                    **dict(state.temp_data["stage_reviews"].get("work_package_planning", {}) or {}),
                    "reference_binding_notes": reference_binding_notes,
                }
        if active_index <= plan_stage_order.index("global_contract_synthesis"):
            state = _global_contract_impl(state)
        if active_index <= plan_stage_order.index("architecture_planning"):
            state = _architecture_planning_impl(state)
        if active_index <= plan_stage_order.index("package_file_planning"):
            state = _package_file_planning_impl(state)
        if active_index <= plan_stage_order.index("canonical_ir_synthesis"):
            state = _canonical_ir_shadow_impl(state)
        logger.info("plan - Done: %d steps", len(state.temp_data.get("steps", [])))
        return state
    return _run_node(state, "plan", _plan_impl)


def _apply_local_generation_bundle(state: PaperBenchReproState, payload: dict[str, Any]) -> None:
    local_generation_bundle.apply_local_generation_bundle(state, payload)


def _load_local_generation_bundle(state: PaperBenchReproState) -> dict[str, Any]:
    return local_generation_bundle.load_local_generation_bundle(state, get_output_dir=_get_output_dir)


def _apply_repo_validation_bundle(state: PaperBenchReproState, payload: dict[str, Any]) -> None:
    execution_payload = payload.get("execution_result") or {}
    runtime_probe_payload = payload.get("runtime_probe") or {}
    validation_payload = payload.get("validation_report") or {}
    benchmark_payload = payload.get("benchmark_report") or {}
    preflight_payload = payload.get("preflight_result") or {}
    repair_ticket_payload = payload.get("repair_ticket") or {}

    state.project_root = str(payload.get("project_root", "") or state.project_root)
    state.generated_files = list(payload.get("generated_files", state.generated_files) or [])
    state.code = str(payload.get("code", "") or state.code)
    state.execution_result = ExecutionResult.model_validate(execution_payload) if execution_payload else state.execution_result
    state.runtime_probe = RuntimeProbe.model_validate(runtime_probe_payload) if runtime_probe_payload else None
    if validation_payload:
        state.validation_report = ValidationReport.model_validate(validation_payload)
    elif state.validation_report is None:
        state.validation_report = ValidationReport(
            passed=False,
            static_status="failed",
            static_contract_status="unknown",
            smoke_status="failed",
            dynamic_status="failed",
            overall_status="failed",
            quality_level="scaffold_only",
            blocked_reasons=["repair validation bundle is missing validation_report"],
            failure_categories=["validation_bundle"],
            repair_recommendations=["Re-run repo validation and rebuild the repair ticket before regeneration."],
        )
    state.benchmark_report = BenchmarkReport.model_validate(benchmark_payload) if benchmark_payload else None
    state.preflight_result = PreflightResult.model_validate(preflight_payload) if preflight_payload else state.preflight_result
    state.repair_ticket = RepairTicket.model_validate(repair_ticket_payload) if repair_ticket_payload else None
    state.temp_data["validation_bundle"] = dict(payload)
    state.temp_data["repair_ticket"] = repair_ticket_payload
    state.temp_data["validated_repo_handoff"] = dict(payload.get("validated_repo_handoff", {}) or {})
    state.temp_data["repo_handoff"] = dict(payload.get("repo_handoff", {}) or {})

    if state.validation_report is not None and state.generate_stage_output is not None:
        validation_summary = {
            "passed": state.validation_report.passed,
            "overall_status": state.validation_report.overall_status,
            "failure_categories": list(state.validation_report.failure_categories),
            "blocked_reasons": list(state.validation_report.blocked_reasons),
        }
        quality_level = str(payload.get("quality_level", "") or state.validation_report.quality_level or "scaffold_only")
        experiment_status = (
            str(payload.get("experiment_status", "") or "")
            or ("repo_validation_passed" if state.validation_report.passed else "repo_validation_failed")
        )
        state.generate_stage_output = state.generate_stage_output.model_copy(
            update={
                "validation_summary": validation_summary,
                "quality_level": quality_level,
                "experiment_status": experiment_status,
            }
        )


def _load_repo_validation_bundle(state: PaperBenchReproState) -> dict[str, Any]:
    return _read_stage_json(
        state,
        "nodes/repair/repair_validation_bundle.json",
        "nodes/repair/validation_bundle.json",
        "repair_validation_bundle.json",
        "validation_bundle.json",
    )


def _run_repo_validation_bundle(state: PaperBenchReproState) -> dict[str, Any]:
    state.preflight_result, state.execution_result = (
        validation_helpers.run_repo_execution_validation(
            state,
            get_workflow_config=get_workflow_config,
            get_output_dir=_get_output_dir,
        )
    )
    runtime_probe, validation_report, benchmark_report = _evaluate_validation_bundle(state)
    runtime_probe.execution_mode = "reproagent_docker_validation"
    if "docker" not in runtime_probe.available_commands:
        runtime_probe.available_commands.append("docker")
    state.runtime_probe = runtime_probe
    quality_level = "repo_validated" if validation_report.passed else "repo_validation_failed"
    validation_report = validation_report.model_copy(update={"quality_level": quality_level})
    experiment_status = "repo_validation_passed" if validation_report.passed else "repo_validation_failed"
    state.validation_report = validation_report
    validated_repo_handoff = (
        validation_helpers.build_validated_repo_handoff(
            state,
            build_repo_handoff_payload=build_repo_handoff_payload,
            get_output_dir=_get_output_dir,
        )
        if validation_report.passed
        else {}
    )
    state.temp_data["validated_repo_handoff"] = validated_repo_handoff
    repo_handoff = build_repo_handoff_payload(
        state,
        start_stage="rapid_validation",
        thread_id=state.input.thread_id or state.run_id,
    )
    return {
        "project_root": state.project_root,
        "generated_files": list(state.generated_files),
        "code": state.code,
        "execution_result": state.execution_result.model_dump(mode="json") if state.execution_result else {},
        "runtime_probe": runtime_probe.model_dump(mode="json"),
        "validation_report": validation_report.model_dump(mode="json"),
        "benchmark_report": benchmark_report.model_dump(mode="json"),
        "preflight_result": state.preflight_result.model_dump(mode="json") if state.preflight_result else {},
        "repair_ticket": state.repair_ticket.model_dump(mode="json") if state.repair_ticket else {},
        "quality_level": quality_level,
        "experiment_status": experiment_status,
        "validated_repo_handoff": validated_repo_handoff,
        "repo_handoff": repo_handoff,
        "validation_scope": "repo_closure_validation",
        "repo_plan_static_contract_checks": [
            item.model_dump(mode="json")
            for item in _repo_plan_static_contract_checks(state)
        ],
        "global_contract_wiring_checks": [
            item.model_dump(mode="json")
            for item in _global_contract_wiring_checks(state)
        ],
        "traceability_checks": [
            item.model_dump(mode="json")
            for item in _traceability_checks(state)
        ],
    }


def generate_node(state: PaperBenchReproState) -> PaperBenchReproState:
    """Generate the planned repository from the repo/file contract."""
    def _generate_impl(state: PaperBenchReproState) -> PaperBenchReproState:
        _ensure_prepare_quality_gate_passed(state)
        return generate_nodes.generate_impl(
            state,
            build_repo_plan=_build_repo_plan,
            build_runtime_project_plan=_build_runtime_project_plan,
            build_generation_manifest=_build_generation_manifest,
            ordered_runtime_task_ids=_ordered_runtime_task_ids,
            build_runtime_task_views=_build_runtime_task_views,
            build_task_project_plan=_build_task_project_plan,
            filter_task_generated_files=_filter_task_generated_files,
            build_generate_stage_output=_build_generate_stage_output,
            get_dataset_preparation=_get_dataset_preparation,
            run_task_review=_run_task_review,
            get_output_dir=_get_output_dir,
            write_stage_output=_write_stage_output,
            run_or_resume_stage=_run_or_resume_stage,
            save_tracking_artifacts=_save_tracking_artifacts,
            load_local_generation_bundle=_load_local_generation_bundle,
            apply_local_generation_bundle=_apply_local_generation_bundle,
            persist_generation_checkpoints=_persist_generation_checkpoints,
            build_repo_handoff_payload=build_repo_handoff_payload,
            run_repo_validation_bundle=_run_repo_validation_bundle,
            evaluate_validation_bundle=_evaluate_validation_bundle,
        )
    return _run_node(state, "generate", _generate_impl)


def _repair_validation_impl(state: PaperBenchReproState) -> PaperBenchReproState:
    state = repair_nodes.repair_validation_impl(
        state,
        run_repo_validation_bundle=_run_repo_validation_bundle,
        load_repo_validation_bundle=_load_repo_validation_bundle,
        apply_repo_validation_bundle=_apply_repo_validation_bundle,
        write_stage_output=_write_stage_output,
        run_or_resume_stage=_run_or_resume_stage,
        save_tracking_artifacts=_save_tracking_artifacts,
    )
    review_payload = _build_repair_review_payload(
        state,
        runtime_validation=dict(state.temp_data.get("validation_bundle", {}) or {}),
    )
    run_memory.refresh_memory_artifacts(state)
    stage_mode = dict(state.temp_data.get("stage_execution_mode", {}) or {}).get("repair_validation", "")
    if stage_mode == "computed":
        pending_regeneration = dict(state.temp_data.pop("pending_repair_regeneration_attempt", {}) or {})
        if pending_regeneration and state.validation_report is not None and not state.validation_report.passed:
            run_memory.record_repair_regeneration_unresolved(
                state,
                round_id=int(pending_regeneration.get("round_id", 0) or 0),
                touched_files=list(pending_regeneration.get("touched_files", []) or []),
                suggested_focus=list(pending_regeneration.get("suggested_focus", []) or []),
            )
        run_memory.record_validation_failure(state)
    state.temp_data["repair_review"] = review_payload
    _write_stage_output(state, "repair_review.json", review_payload)
    _write_stage_output(state, "repair_findings.json", _build_repair_findings_payload(state))
    _save_tracking_artifacts(state)
    return state


def _run_repair_validation_pass(state: PaperBenchReproState) -> PaperBenchReproState:
    """Run one repair-stage validation pass within the repair phase."""
    return _run_node(state, "repair", _repair_validation_impl)


def _run_repair_plan_pass(state: PaperBenchReproState) -> PaperBenchReproState:
    """Run one semantic repair-plan pass within the repair phase."""
    return _run_node(state, "repair", _repair_plan_impl)


def _normalized_repo_path(value: str) -> str:
    return validation_helpers.normalized_repo_path(value)


def _normalized_repo_key(value: str) -> str:
    return validation_helpers.normalized_repo_key(value)


def _normalized_repo_keys(values: list[str]) -> set[str]:
    return validation_helpers.normalized_repo_keys(values)


def _file_plan_artifact_keys(file_plan: RepoFilePlan | None) -> set[str]:
    return validation_helpers.file_plan_artifact_keys(file_plan)


def _repo_plan_static_contract_checks(state: PaperBenchReproState) -> list[ValidationCheck]:
    return validation_helpers.repo_plan_static_contract_checks(
        state,
        entrypoint_related_work_packages=_entrypoint_related_work_packages,
    )


def _repo_plan_artifact_owner_map(state: PaperBenchReproState) -> dict[str, list[str]]:
    return validation_helpers.repo_plan_artifact_owner_map(state)


def _global_contract_wiring_checks(state: PaperBenchReproState) -> list[ValidationCheck]:
    return validation_helpers.global_contract_wiring_checks(state)


def _repair_recommendations_from_checks(checks: list[ValidationCheck]) -> list[str]:
    return validation_helpers.repair_recommendations_from_checks(checks)


def _traceability_checks(state: PaperBenchReproState) -> list[ValidationCheck]:
    return validation_helpers.traceability_checks(state)


def _work_package_file_index(state: PaperBenchReproState) -> dict[str, list[str]]:
    return validation_helpers.work_package_file_index(state)


def _global_repair_surface_files(state: PaperBenchReproState) -> list[str]:
    return validation_helpers.global_repair_surface_files(
        state,
        work_package_file_index=_work_package_file_index,
    )


def _trace_failed_work_package_ids(report: ValidationReport) -> list[str]:
    return validation_helpers.trace_failed_work_package_ids(report)


def _refresh_work_package_evidence(state: PaperBenchReproState, work_package_ids: list[str]) -> list[str]:
    return validation_helpers.refresh_work_package_evidence(
        state,
        work_package_ids,
        build_evidence_bundles=_build_evidence_bundles,
        write_stage_output=_write_stage_output,
    )


def _evaluate_validation_bundle(state: PaperBenchReproState) -> tuple[RuntimeProbe, ValidationReport, BenchmarkReport]:
    return validation_helpers.evaluate_validation_bundle(
        state,
        build_runtime_probe=_build_runtime_probe,
        get_output_dir=_get_output_dir,
        entrypoint_related_work_packages=_entrypoint_related_work_packages,
    )


def _all_failed_validation_checks(report: ValidationReport) -> list[ValidationCheck]:
    return repair_helpers.all_failed_validation_checks(report)


_RUNTIME_FIRST_MARKERS = (
    "SyntaxError",
    "IndentationError",
    "TabError",
    "ImportError",
    "ModuleNotFoundError",
    "NameError",
    "AttributeError",
    "TypeError",
)
_TRACEBACK_FILE_RE = re.compile(r'File "([^"]+)", line \d+')
_FROM_MODULE_RE = re.compile(r"\bfrom\s+([A-Za-z_][\w.]*)\s+import\b")
_QUOTED_FROM_MODULE_RE = re.compile(r"\bfrom\s+['\"]([A-Za-z_][\w.]*)['\"]")
_NO_MODULE_RE = re.compile(r"No module named ['\"]([^'\"]+)['\"]")


def _runtime_first_text_has_blocker(text: str) -> bool:
    lowered = str(text or "").lower()
    return any(marker.lower() in lowered for marker in _RUNTIME_FIRST_MARKERS)


def _repo_relative_runtime_path(state: PaperBenchReproState, value: str) -> str:
    raw = str(value or "").strip().replace("\\", "/")
    if not raw:
        return ""
    if "/working/" in raw:
        raw = raw.rsplit("/working/", 1)[1]
    project_root = str(state.project_root or "").strip().replace("\\", "/")
    if project_root and raw.startswith(project_root.rstrip("/") + "/"):
        raw = raw[len(project_root.rstrip("/")) + 1:]
    if "/repo/" in raw:
        raw = raw.rsplit("/repo/", 1)[1]
    raw = raw.lstrip("/")
    if not raw or raw.startswith("..") or "site-packages/" in raw:
        return ""
    return _normalized_repo_path(raw)


def _module_to_repo_surface(state: PaperBenchReproState, module_name: str) -> str:
    module = str(module_name or "").strip().strip("'\"")
    if not module:
        return ""
    generated = {_normalized_repo_path(item) for item in list(state.generated_files or [])}
    candidates = [
        _normalized_repo_path(module.replace(".", "/") + ".py"),
        _normalized_repo_path(module.replace(".", "/") + "/__init__.py"),
    ]
    for candidate in candidates:
        if candidate in generated:
            return candidate
    if module == "ftrl_repro":
        return "ftrl_repro/__init__.py"
    if module.startswith(("ftrl_repro.", "src.")):
        return candidates[0]
    return ""


def _runtime_surfaces_from_text(state: PaperBenchReproState, text: str) -> list[str]:
    surfaces: list[str] = []
    for raw_path in _TRACEBACK_FILE_RE.findall(str(text or "")):
        rel_path = _repo_relative_runtime_path(state, raw_path)
        if rel_path:
            surfaces.append(rel_path)
    for regex in (_FROM_MODULE_RE, _QUOTED_FROM_MODULE_RE, _NO_MODULE_RE):
        for module_name in regex.findall(str(text or "")):
            surface = _module_to_repo_surface(state, module_name)
            if surface:
                surfaces.append(surface)
    if "No module named" in str(text or "") and "pyproject.toml" in list(state.generated_files or []):
        surfaces.append("pyproject.toml")
    if not surfaces:
        entrypoints = dict(state.project_plan.entrypoints or {}) if state.project_plan is not None else {}
        surfaces.extend(str(item).strip() for item in entrypoints.values() if str(item).strip())
    return list(dict.fromkeys(_normalized_repo_path(item) for item in surfaces if str(item).strip()))


def _compact_runtime_error_text(text: str, limit: int = 1800) -> str:
    lines = [line.rstrip() for line in str(text or "").splitlines()]
    selected: list[str] = []
    for index, line in enumerate(lines):
        if _runtime_first_text_has_blocker(line) or line.strip().startswith(("File ", "Traceback")):
            selected.extend(lines[max(0, index - 2): min(len(lines), index + 4)])
    if not selected:
        selected = lines[-16:]
    compact = "\n".join(line for line in selected if line.strip())
    return compact[:limit]


def _runtime_first_blockers_from_state(
    state: PaperBenchReproState,
    report: ValidationReport | None = None,
) -> list[dict[str, Any]]:
    report = report or state.validation_report or ValidationReport()
    fragments: list[tuple[str, str]] = []
    if state.execution_result is not None:
        execution_payload = state.execution_result.model_dump(mode="json")
        for key in ("error", "output"):
            text = str(execution_payload.get(key, "") or "").strip()
            if text:
                fragments.append((f"execution_result.{key}", text))
    for check in _all_failed_validation_checks(report):
        details = str(check.details or "").strip()
        if details:
            fragments.append((str(check.name or check.category or "validation_check"), details))
    if state.repair_ticket is not None:
        fragments.extend(
            ("repair_ticket.trigger", str(item).strip())
            for item in list(state.repair_ticket.trigger_signals or [])
            if str(item).strip()
        )

    blockers: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for source, text in fragments:
        if not _runtime_first_text_has_blocker(text):
            continue
        error = _compact_runtime_error_text(text)
        surfaces = _runtime_surfaces_from_text(state, text)
        category = next((marker for marker in _RUNTIME_FIRST_MARKERS if marker.lower() in text.lower()), "runtime")
        key = (source, error[:240])
        if key in seen:
            continue
        seen.add(key)
        blockers.append(
            {
                "source": source,
                "category": category,
                "error": error,
                "surfaces": surfaces,
            }
        )
    return blockers[:8]


def _runtime_first_surfaces(state: PaperBenchReproState, blockers: list[dict[str, Any]]) -> list[str]:
    surfaces = [
        str(surface).strip()
        for blocker in blockers
        for surface in list(blocker.get("surfaces") or [])
        if str(surface).strip()
    ]
    if not surfaces and blockers and state.project_plan is not None:
        surfaces = [
            str(item).strip()
            for item in dict(state.project_plan.entrypoints or {}).values()
            if str(item).strip()
        ]
    return list(dict.fromkeys(_normalized_repo_path(item) for item in surfaces if str(item).strip()))[:10]


def _load_repair_score_feedback(state: PaperBenchReproState) -> dict[str, Any]:
    config = get_workflow_config()
    if not bool(getattr(config, "repair_score_feedback_enabled", True)):
        return {}
    feedback = repair_helpers.load_repair_score_feedback(
        state,
        run_dir=_get_output_dir(state),
        max_items=int(getattr(config, "repair_score_feedback_max_items", 32) or 32),
        max_files=int(getattr(config, "repair_score_feedback_max_files", 8) or 8),
    )
    if feedback:
        state.temp_data["repair_score_feedback"] = feedback
        state.temp_data["repair_score_feedback_prompt"] = repair_helpers.score_feedback_for_prompt(feedback)
        _write_stage_output(state, "score_feedback.json", feedback)
    return feedback


def _score_feedback_prompt_payload(state: PaperBenchReproState) -> dict[str, Any]:
    feedback = dict(state.temp_data.get("repair_score_feedback", {}) or {})
    return repair_helpers.score_feedback_for_prompt(feedback)


def _apply_runtime_first_repair_plan(state: PaperBenchReproState, repair_plan: RepairPlan) -> RepairPlan:
    blockers = _runtime_first_blockers_from_state(state)
    if not blockers:
        return repair_plan
    surfaces = _runtime_first_surfaces(state, blockers)
    if not surfaces:
        return repair_plan
    blocker_summaries = [
        f"{item.get('category', 'runtime')}: {str(item.get('error', '')).splitlines()[-1][:240]}"
        for item in blockers[:4]
    ]
    runtime_guidance = [
        "Runtime-first repair mode is active: close the exact startup traceback before broad semantic or artifact rewrites.",
        "Do not delete imports, entrypoints, artifact APIs, or package exports to make validation pass; repair the provider/consumer mismatch.",
        "After patching, the same smoke command must import the package and reach artifact generation without SyntaxError, ImportError, or ModuleNotFoundError.",
    ]
    return repair_plan.model_copy(
        update={
            "summary": "Runtime-first repair: fix startup/import/syntax blockers before semantic expansion. "
            + str(repair_plan.summary or ""),
            "problem_list": list(dict.fromkeys([*blocker_summaries, *list(repair_plan.problem_list)]))[:12],
            "runtime_guardrails": list(dict.fromkeys([*runtime_guidance, *list(repair_plan.runtime_guardrails)]))[:16],
            "recommended_surfaces": surfaces,
            "selected_files": [item for item in surfaces if "." in item],
            "selected_work_packages": [],
            "repair_guidance": list(dict.fromkeys([*runtime_guidance, *list(repair_plan.repair_guidance)]))[:18],
            "review_points": list(dict.fromkeys([
                "Inspect traceback producer and consumer files together; repair missing symbols/functions instead of suppressing imports.",
                "Run import/compile validation mentally against the exact traceback before returning updated files.",
                *list(repair_plan.review_points),
            ]))[:16],
            "round_budget": max(2, int(repair_plan.round_budget or 1)),
        }
    )


def _apply_score_feedback_repair_plan(state: PaperBenchReproState, repair_plan: RepairPlan) -> RepairPlan:
    feedback = dict(state.temp_data.get("repair_score_feedback", {}) or {})
    if not feedback:
        return repair_plan
    prioritized_files = [
        _normalized_repo_path(path)
        for path in list(feedback.get("prioritized_files", []) or [])
        if _normalized_repo_path(str(path))
    ]
    prioritized_work_packages = [
        str(item).strip()
        for item in list(feedback.get("prioritized_work_packages", []) or [])
        if str(item).strip()
    ]
    if not prioritized_files and not prioritized_work_packages:
        return repair_plan
    leaf_notes = [
        str(item.get("requirement", "") or "")[:220]
        for item in list(feedback.get("items", []) or [])[:8]
        if (
            isinstance(item, dict)
            and bool(item.get("valid_score", True))
            and str(item.get("requirement", "") or "").strip()
        )
    ]
    score_guidance = [
        "Score-feedback repair mode is active: prioritize judge-scored low/zero rubric leaves with valid scores before broad validation-route polishing.",
        "Ignore invalid-score leaves as repair targets; they indicate judge extraction failures, not reliable implementation gaps.",
        "For each selected surface, implement the missing requirement as executable code/config/constants/call sites, not README prose or manifest declarations.",
        "Preserve runtime fixes already made; do not trade score-relevant implementation away to satisfy structural validation text.",
    ]
    return repair_plan.model_copy(
        update={
            "summary": "Score-feedback repair: close judge-failed rubric leaves. "
            + str(repair_plan.summary or ""),
            "problem_list": list(dict.fromkeys([*leaf_notes, *list(repair_plan.problem_list)]))[:16],
            "recommended_surfaces": list(
                dict.fromkeys([*prioritized_files, *list(repair_plan.recommended_surfaces)])
            )[:16],
            "selected_files": list(dict.fromkeys([*prioritized_files, *list(repair_plan.selected_files)]))[:16],
            "selected_work_packages": list(
                dict.fromkeys([*prioritized_work_packages, *list(repair_plan.selected_work_packages)])
            )[:16],
            "repair_guidance": list(dict.fromkeys([*score_guidance, *list(repair_plan.repair_guidance)]))[:18],
            "review_points": list(
                dict.fromkeys(
                    [
                        "Check the latest score_feedback.json failed_leaf_examples and patch only valid-score code-backed requirements first.",
                        "Do not stop after validation-route declarations if the judge leaf asks for a concrete formula, constant, loader, baseline, metric, or algorithm step.",
                        *list(repair_plan.review_points),
                    ]
                )
            )[:18],
            "acceptance_criteria": list(dict.fromkeys([*leaf_notes, *list(repair_plan.acceptance_criteria)]))[:24],
        }
    )


def _repair_failure_class(check: ValidationCheck) -> str:
    return repair_helpers.repair_failure_class(check)


def _entrypoint_related_work_packages(state: PaperBenchReproState) -> list[str]:
    return repair_helpers.entrypoint_related_work_packages(state)


def _select_repair_work_packages(state: PaperBenchReproState, report: ValidationReport) -> list[str]:
    return repair_helpers.select_repair_work_packages(
        state,
        report,
        entrypoint_related_work_packages=_entrypoint_related_work_packages,
    )


def _select_exact_repair_files(state: PaperBenchReproState, report: ValidationReport) -> list[str]:
    return repair_helpers.select_exact_repair_files(
        state,
        report,
        work_package_file_index=_work_package_file_index,
        global_repair_surface_files=_global_repair_surface_files,
    )


def _task_ids_for_repair_files(
    state: PaperBenchReproState,
    file_paths: list[str],
    *,
    preserve_file_order: bool = False,
) -> list[str]:
    return repair_helpers.task_ids_for_repair_files(
        state,
        file_paths,
        ordered_runtime_task_ids=_ordered_runtime_task_ids,
        preserve_file_order=preserve_file_order,
    )


def _task_ids_for_repair_work_packages(state: PaperBenchReproState, work_package_ids: list[str]) -> list[str]:
    return repair_helpers.task_ids_for_repair_work_packages(
        state,
        work_package_ids,
        ordered_runtime_task_ids=_ordered_runtime_task_ids,
        work_package_file_index=_work_package_file_index,
        task_ids_for_repair_files=_task_ids_for_repair_files,
    )


def _repair_fallback_work_packages(
    state: PaperBenchReproState,
    report: ValidationReport,
    work_package_ids: list[str],
) -> list[str]:
    return repair_helpers.repair_fallback_work_packages(
        state,
        report,
        work_package_ids,
        select_exact_repair_files=_select_exact_repair_files,
    )


def _persist_generation_checkpoints(state: PaperBenchReproState) -> None:
    workflow_runtime.persist_generation_checkpoints(
        state,
        get_output_dir=_get_output_dir,
        json_default=_json_default,
    )


def _run_repair_generation_round(
    state: PaperBenchReproState,
    *,
    selected_task_ids: list[str],
    round_id: int,
) -> dict[str, Any]:
    return repair_helpers.run_repair_generation_round(
        state,
        selected_task_ids=selected_task_ids,
        round_id=round_id,
        ordered_runtime_task_ids=_ordered_runtime_task_ids,
        build_runtime_task_views=_build_runtime_task_views,
        build_task_project_plan=_build_task_project_plan,
        filter_task_generated_files=_filter_task_generated_files,
        get_codegen_config=get_codegen_config,
        get_workflow_config=get_workflow_config,
        get_sandbox_provider=get_sandbox_provider,
        get_output_dir=_get_output_dir,
        run_task_review=_run_task_review,
        build_generate_stage_output=_build_generate_stage_output,
        write_stage_output=_write_stage_output,
        persist_generation_checkpoints=_persist_generation_checkpoints,
    )


def _run_repo_wide_repair_round(
    state: PaperBenchReproState,
    *,
    round_id: int,
) -> dict[str, Any]:
    from reproagent.pipeline.tools import load_project_files
    from reproagent.pipeline.utils.repo_agent_engine import RepoAgentEngine

    project_root = Path(state.project_root) if state.project_root else _get_output_dir(state) / "repo"
    previous_files = load_project_files(project_root) if project_root.exists() else {}
    engine = RepoAgentEngine(
        get_codegen_config(),
        get_workflow_config(),
        get_sandbox_provider(),
    )
    engine.language = state.input.language or "zh"
    engine.output_dir = _get_output_dir(state)
    engine.output_dir.mkdir(parents=True, exist_ok=True)
    node_dir = engine.output_dir / "nodes" / "repair"
    node_dir.mkdir(parents=True, exist_ok=True)
    engine.node_artifact_dir = node_dir
    engine.checkpoint_path = str((node_dir / "iteration_checkpoint.json").resolve())
    return engine.run_repo_repair_pass(
        state=state,
        target=state.input.target,
        requirement_anchor=state.requirement_anchor.model_dump(mode="json") if state.requirement_anchor else {},
        repair_eval_report=state.repair_eval_report.model_dump(mode="json") if state.repair_eval_report else {},
        repair_plan=state.repair_plan.model_dump(mode="json") if state.repair_plan else {},
        iteration=round_id,
        iteration_context={
            "previous_files": previous_files,
            "previous_review": dict(state.temp_data.get("repair_review", {}) or {}),
            "validation_report": state.validation_report.model_dump(mode="json") if state.validation_report else {},
            "execution_result": state.execution_result.model_dump(mode="json") if state.execution_result else {},
            "runtime_first_blockers": _runtime_first_blockers_from_state(state),
            "score_feedback": _score_feedback_prompt_payload(state),
            "suggestions": list(state.repair_plan.review_points if state.repair_plan else []),
        },
    )


def _repair_round_budget(state: PaperBenchReproState) -> int:
    config = get_workflow_config()
    max_budget = max(1, int(getattr(config, "repair_round_budget_max", 5) or 5))
    if state.repair_plan is not None and int(state.repair_plan.round_budget or 0) > 0:
        return min(max_budget, max(1, int(state.repair_plan.round_budget or 1)))
    return max_budget


def _select_local_repair_task_ids(
    state: PaperBenchReproState,
    report: ValidationReport,
) -> tuple[list[str], dict[str, Any]]:
    config = get_workflow_config()
    local_file_budget = max(0, int(getattr(config, "repair_local_scope_max_files", 0) or 0))
    runtime_first_blockers = _runtime_first_blockers_from_state(state, report)
    runtime_first_files = _runtime_first_surfaces(state, runtime_first_blockers)
    exact_files = _select_exact_repair_files(state, report)
    score_feedback = dict(state.temp_data.get("repair_score_feedback", {}) or {})
    score_feedback_files = [
        _normalized_repo_path(path)
        for path in list(score_feedback.get("prioritized_files", []) or [])
        if _normalized_repo_path(str(path))
    ]
    exact_files = list(dict.fromkeys([*runtime_first_files, *score_feedback_files, *exact_files]))
    focused_exact_scope = bool(state.repair_plan is not None and list(state.repair_plan.recommended_surfaces or []))
    exact_task_ids = _task_ids_for_repair_files(
        state,
        exact_files,
        preserve_file_order=focused_exact_scope,
    )
    runtime_first_task_ids = _task_ids_for_repair_files(
        state,
        runtime_first_files,
        preserve_file_order=True,
    )
    score_feedback_task_ids = [
        str(item).strip()
        for item in list(score_feedback.get("prioritized_task_ids", []) or [])
        if str(item).strip()
    ]
    exact_task_ids = list(dict.fromkeys([*runtime_first_task_ids, *score_feedback_task_ids, *exact_task_ids]))
    selected_work_packages = _select_repair_work_packages(
        state,
        report,
    )
    score_feedback_work_packages = [
        str(item).strip()
        for item in list(score_feedback.get("prioritized_work_packages", []) or [])
        if str(item).strip()
    ]
    if score_feedback_work_packages:
        selected_work_packages = list(dict.fromkeys([*score_feedback_work_packages, *selected_work_packages]))
    if focused_exact_scope and exact_task_ids:
        fallback_work_packages: list[str] = []
        package_task_ids: list[str] = []
        selected_task_ids = list(dict.fromkeys(exact_task_ids))
    else:
        fallback_work_packages = _repair_fallback_work_packages(
            state,
            report,
            selected_work_packages,
        )
        package_task_ids = _task_ids_for_repair_work_packages(state, fallback_work_packages)
        selected_task_ids = list(dict.fromkeys([*exact_task_ids, *package_task_ids]))
    if local_file_budget > 0 and len(selected_task_ids) > local_file_budget:
        selected_task_ids = selected_task_ids[:local_file_budget]
    return selected_task_ids, {
        "exact_files": exact_files,
        "runtime_first_files": runtime_first_files,
        "runtime_first_task_ids": runtime_first_task_ids,
        "score_feedback": repair_helpers.score_feedback_for_prompt(score_feedback, max_items=8),
        "selected_work_packages": selected_work_packages,
        "fallback_work_packages": fallback_work_packages,
        "focused_exact_scope": focused_exact_scope,
        "local_file_budget": local_file_budget,
        "selected_task_ids": selected_task_ids,
    }


def _should_escalate_repair_repo_wide(
    state: PaperBenchReproState,
    *,
    selected_task_ids: list[str],
    selected_work_packages: list[str],
    exact_files: list[str] | None = None,
) -> bool:
    config = get_workflow_config()
    if not selected_task_ids:
        return True
    exact_task_ids = _task_ids_for_repair_files(state, list(exact_files or []))
    exact_file_count = len(list(exact_files or []))
    local_file_budget = int(config.repair_local_scope_max_files or 0)
    runtime_first_active = bool(_runtime_first_blockers_from_state(state))
    if runtime_first_active and exact_task_ids:
        return False
    if (
        state.repair_plan is not None
        and list(state.repair_plan.recommended_surfaces or [])
        and exact_task_ids
        and (local_file_budget <= 0 or exact_file_count <= local_file_budget)
    ):
        return False
    if exact_task_ids:
        return False
    if len(selected_work_packages) > int(config.repair_local_scope_max_work_packages or 0):
        return True
    if len(selected_task_ids) > int(config.repair_local_scope_max_files or 0):
        return True
    stagnation = dict(state.temp_data.get("repair_stagnation", {}) or {})
    if int(stagnation.get("repeat_count", 0) or 0) > int(config.repair_repo_wide_retry_threshold or 0):
        return True
    return False


def _repair_regeneration_impl(state: PaperBenchReproState) -> PaperBenchReproState:
    """Run one repo-wide repair-generation round against the current failed repo."""
    current_round = max(1, int(state.temp_data.get("repair_round", 1) or 1))
    logger.info("repair - Running repo-wide regeneration round %d...", current_round)
    input_payload = {
        "repair_round": current_round,
        "iteration_count": state.iteration_count,
        "requirement_anchor": state.requirement_anchor.model_dump(mode="json") if state.requirement_anchor else {},
        "repair_eval_report": state.repair_eval_report.model_dump(mode="json") if state.repair_eval_report else {},
        "validation_report": state.validation_report.model_dump(mode="json") if state.validation_report else {},
        "repair_ticket": state.repair_ticket.model_dump(mode="json") if state.repair_ticket else {},
        "repair_plan": state.repair_plan.model_dump(mode="json") if state.repair_plan else {},
        "generated_files": list(state.generated_files),
        "project_root": state.project_root,
    }
    legacy_input_payload = {
        "repair_round": current_round,
        "iteration_count": state.iteration_count,
        "validation_report": state.validation_report.model_dump(mode="json") if state.validation_report else {},
        "repair_ticket": state.repair_ticket.model_dump(mode="json") if state.repair_ticket else {},
        "repair_plan": state.repair_plan.model_dump(mode="json") if state.repair_plan else {},
        "generated_files": list(state.generated_files),
        "work_packages": (
            state.work_package_planning.model_dump(mode="json") if state.work_package_planning else {}
        ),
    }

    def _compute() -> dict[str, Any]:
        if state.validation_report is None:
            empty_log = state.repair_log or RepairLog(converged=False, rounds_attempted=0, actions=[])
            return {
                "validation_report": ValidationReport(passed=False, overall_status="failed").model_dump(mode="json"),
                "benchmark_report": BenchmarkReport().model_dump(mode="json"),
                "repair_log": empty_log.model_dump(mode="json"),
            }
        if state.validation_report.passed:
            passed_log = state.repair_log or RepairLog(converged=True, rounds_attempted=0, actions=[])
            return {
                "validation_report": state.validation_report.model_dump(mode="json"),
                "benchmark_report": (state.benchmark_report or BenchmarkReport()).model_dump(mode="json"),
                "repair_log": passed_log.model_dump(mode="json"),
            }

        current_validation = state.validation_report
        current_benchmark = state.benchmark_report or BenchmarkReport()
        previous_log = state.repair_log or RepairLog(converged=False, rounds_attempted=0, actions=[])
        actions: list[RepairAction] = list(previous_log.actions)
        score_feedback = _load_repair_score_feedback(state)
        if score_feedback and state.repair_plan is not None:
            state.repair_plan = _apply_score_feedback_repair_plan(state, state.repair_plan)
            _write_stage_output(state, "repair_plan.json", state.repair_plan)

        def _repair_reason(report: ValidationReport, prefixes: list[str] | None = None) -> str:
            reason_parts = list(prefixes or [])
            failure_classes = list(
                dict.fromkeys(
                    _repair_failure_class(check)
                    for check in _all_failed_validation_checks(report)
                )
            )
            if failure_classes:
                reason_parts.append(f"repair_classes={','.join(failure_classes)}")
            if state.repair_ticket is not None:
                reason_parts.append(
                    f"repair_ticket={state.repair_ticket.failure_type}: {state.repair_ticket.reason}"
                )
                reason_parts.extend(
                    f"trigger={item}"
                    for item in state.repair_ticket.trigger_signals[:3]
                    if str(item).strip()
                )
            else:
                reason_parts.append("; ".join(report.repair_recommendations) or "validation failure")
            if state.repair_plan is not None and state.repair_plan.summary:
                reason_parts.append(f"repair_plan={state.repair_plan.summary}")
            return "; ".join(part for part in reason_parts if str(part).strip())

        trace_failed_packages = _trace_failed_work_package_ids(current_validation)
        refreshed_packages: list[str] = []
        if trace_failed_packages:
            refreshed_packages = _refresh_work_package_evidence(state, trace_failed_packages)
            if refreshed_packages:
                state.runtime_probe, current_validation, current_benchmark = _evaluate_validation_bundle(state)
                actions.append(
                    RepairAction(
                        round_id=current_round,
                        action_type="evidence-refresh",
                        reason="refresh package-scoped evidence bundles before regeneration when traceability is broken",
                        touched_work_packages=refreshed_packages,
                        touched_files=[],
                        outcome=f"refreshed evidence bundles for {len(refreshed_packages)} work packages",
                    )
                )
                if current_validation.passed:
                    repair_log = RepairLog(
                        converged=True,
                        rounds_attempted=previous_log.rounds_attempted,
                        actions=actions,
                    )
                    return {
                        "validation_report": current_validation.model_dump(mode="json"),
                        "benchmark_report": current_benchmark.model_dump(mode="json"),
                        "repair_log": repair_log.model_dump(mode="json"),
                    }

        touched_files: list[str] = []
        agent_usage_summary: dict[str, Any] = {}
        previous_generated_files = list(state.generated_files or [])
        local_task_ids, local_scope = _select_local_repair_task_ids(state, current_validation)
        use_repo_wide = _should_escalate_repair_repo_wide(
            state,
            selected_task_ids=local_task_ids,
            selected_work_packages=list(local_scope.get("selected_work_packages", []) or []),
            exact_files=list(local_scope.get("exact_files", []) or []),
        )
        if use_repo_wide:
            repair_generation_result = _run_repo_wide_repair_round(
                state,
                round_id=current_round,
            )
            local_scope["repair_execution_mode"] = "repo_wide"
        else:
            repair_generation_result = _run_repair_generation_round(
                state,
                selected_task_ids=local_task_ids,
                round_id=current_round,
            )
            local_scope["repair_execution_mode"] = "localized"
        touched_files = sorted(list((repair_generation_result.get("updated_files") or {}).keys()))
        if not touched_files:
            touched_files = sorted(list(repair_generation_result.get("touched_files") or []))
        safety_report = dict(repair_generation_result.get("safety_report", {}) or {})
        rejected_files = [
            str(item.get("path", "") or "").strip()
            for item in list(safety_report.get("file_reports", []) or [])
            if isinstance(item, dict) and str(item.get("path", "") or "").strip()
        ]
        agent_usage_summary = dict(repair_generation_result.get("context_usage", {}) or {})
        if not agent_usage_summary:
            agent_usage_summary = dict(repair_generation_result.get("agent_usage_summary", {}) or {})
        state.generated_files = sorted(list((repair_generation_result.get("project_files") or {}).keys()))
        if not state.generated_files:
            state.generated_files = sorted(previous_generated_files)
        file_provenance = refresh_file_provenance(state)
        state.temp_data["pending_repair_regeneration_attempt"] = {
            "round_id": current_round,
            "touched_files": touched_files,
            "suggested_focus": list(state.repair_plan.recommended_surfaces if state.repair_plan else []),
            "selected_task_ids": list(local_scope.get("selected_task_ids", []) or []),
            "selected_work_packages": list(local_scope.get("selected_work_packages", []) or []),
            "exact_files": list(local_scope.get("exact_files", []) or []),
            "repair_execution_mode": str(local_scope.get("repair_execution_mode", "") or ""),
            "rejected_files": rejected_files if safety_report and not bool(safety_report.get("passed", False)) else [],
            "repair_safety": {
                "passed": bool(safety_report.get("passed", True)),
                "issues": list(safety_report.get("issues", []) or [])[:12],
            },
        }
        history_path = _get_output_dir(state) / "nodes" / "repair" / "repair_round_history.jsonl"
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with history_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "round_id": current_round,
                "requirement_anchor_summary": (
                    state.requirement_anchor.summary if state.requirement_anchor is not None else ""
                ),
                "repair_eval_summary": (
                    state.repair_eval_report.summary if state.repair_eval_report is not None else ""
                ),
                "repair_plan_summary": (
                    state.repair_plan.summary if state.repair_plan is not None else ""
                ),
                "recommended_surfaces": list(state.repair_plan.recommended_surfaces if state.repair_plan else []),
                "repair_execution_mode": str(local_scope.get("repair_execution_mode", "") or ""),
                "selected_work_packages": list(local_scope.get("selected_work_packages", []) or []),
                "selected_task_ids": list(local_scope.get("selected_task_ids", []) or []),
                "exact_files": list(local_scope.get("exact_files", []) or []),
                "changed_files": touched_files,
                "context_usage": agent_usage_summary,
                "runtime_first_blockers": _runtime_first_blockers_from_state(state, current_validation),
                "repair_safety": {
                    "passed": bool(safety_report.get("passed", True)),
                    "issues": list(safety_report.get("issues", []) or [])[:12],
                    "file_reports": list(safety_report.get("file_reports", []) or [])[:12],
                },
            }, ensure_ascii=False) + "\n")
        if state.project_root:
            state.project_root = str((Path(state.project_root)).resolve())

        actions.append(
            RepairAction(
                round_id=current_round,
                action_type=(
                    "repair-rejected-by-safety-guard"
                    if safety_report and not bool(safety_report.get("passed", False))
                    else
                    "repo-wide-repair"
                    if str(local_scope.get("repair_execution_mode", "") or "") == "repo_wide" and touched_files
                    else "localized-repair"
                    if touched_files
                    else "repair-noop"
                ),
                reason=_repair_reason(
                    current_validation,
                    (
                        [
                            "repair safety guard rejected destructive or syntactically invalid update: "
                            + "; ".join(str(item) for item in list(safety_report.get("issues", []) or [])[:6])
                        ]
                        if safety_report and not bool(safety_report.get("passed", False))
                        else
                        [
                            "repo-wide repair pass produced no file changes"
                            if str(local_scope.get("repair_execution_mode", "") or "") == "repo_wide"
                            else "localized repair pass produced no file changes"
                        ]
                        if not touched_files
                        else None
                    ),
                ),
                touched_work_packages=list(local_scope.get("selected_work_packages", []) or []),
                touched_files=touched_files,
                outcome=(
                    "rejected_by_safety_guard"
                    if safety_report and not bool(safety_report.get("passed", False))
                    else "generated" if touched_files else "no_changes"
                ),
            )
        )

        repair_log = RepairLog(
            converged=False,
            rounds_attempted=max(previous_log.rounds_attempted, current_round),
            actions=actions,
        )
        return {
            "validation_report": current_validation.model_dump(mode="json"),
            "benchmark_report": current_benchmark.model_dump(mode="json"),
            "repair_log": repair_log.model_dump(mode="json"),
            "file_provenance": file_provenance,
            "agent_usage_summary": agent_usage_summary,
            "repair_safety": safety_report,
        }

    def _load() -> dict[str, Any]:
        output_dir = _get_output_dir(state)
        node_path = output_dir / "nodes" / "repair" / "repair_regeneration_result.json"
        path = node_path if node_path.exists() else output_dir / "repair_regeneration_result.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return {
            "validation_report": _read_stage_json(state, "nodes/repair/validation_report.json", "validation_report.json"),
            "benchmark_report": _read_stage_json(state, "nodes/repair/benchmark_report.json", "benchmark_report.json"),
            "repair_log": _read_stage_json(state, "nodes/repair/repair_log.json", "repair_log.json"),
        }

    def _write(result: dict[str, Any]) -> None:
        validation_report = ValidationReport.model_validate(result.get("validation_report") or {})
        benchmark_report = BenchmarkReport.model_validate(result.get("benchmark_report") or {})
        repair_log = RepairLog.model_validate(result.get("repair_log") or {})
        state.validation_report = validation_report
        state.benchmark_report = benchmark_report
        state.repair_log = repair_log
        file_provenance = result.get("file_provenance")
        if not isinstance(file_provenance, list):
            file_provenance = refresh_file_provenance(state)
        else:
            state.temp_data["file_provenance"] = list(file_provenance)
        review_payload = _build_repair_review_payload(state)
        state.temp_data["repair_review"] = review_payload
        _write_stage_output(state, "repair_review.json", review_payload)
        _write_stage_output(state, "repair_regeneration_result.json", result)
        _write_stage_output(state, "file_provenance.json", file_provenance)
        _write_stage_output(state, "validation_report.json", validation_report)
        _write_stage_output(state, "benchmark_report.json", benchmark_report)
        _write_stage_output(state, "repair_log.json", repair_log)

    output_dir = _get_output_dir(state)
    stage_status = _load_stage_status(state)
    existing = dict(stage_status.get("repair_regeneration", {}) or {})
    pipeline_signature_value = _pipeline_signature()
    current_input_hash = workflow_runtime._payload_hash(input_payload, json_default=_json_default)
    legacy_input_hash = workflow_runtime._payload_hash(legacy_input_payload, json_default=_json_default)
    stage_outputs_present = all((output_dir / item).exists() for item in STAGE_OUTPUTS["repair_regeneration"])
    stage_result_exists = (
        (output_dir / "nodes" / "repair" / "repair_regeneration_result.json").exists()
        or (output_dir / "repair_regeneration_result.json").exists()
    )
    reuse_completed_stage_result = (
        existing.get("status") == "completed"
        and existing.get("pipeline_signature") == pipeline_signature_value
        and stage_result_exists
        and (
            existing.get("input_hash") == legacy_input_hash
            or (existing.get("input_hash") == current_input_hash and not stage_outputs_present)
        )
    )
    if reuse_completed_stage_result:
        state.temp_data.setdefault("stage_execution_mode", {})["repair_regeneration"] = "resumed"
        stage_status["repair_regeneration"] = {
            **existing,
            "resume_source": "reused_artifacts",
        }
        _write_stage_status(state, stage_status)
        result = _load()
        _write(result)
    else:
        result = _run_or_resume_stage(
            state,
            "repair_regeneration",
            input_payload,
            _compute,
            _load,
            _write,
        )
    state.validation_report = ValidationReport.model_validate(result.get("validation_report") or {})
    state.benchmark_report = BenchmarkReport.model_validate(result.get("benchmark_report") or {})
    state.repair_log = RepairLog.model_validate(result.get("repair_log") or {})
    _save_tracking_artifacts(state)
    return state


def _run_repair_regeneration_pass(state: PaperBenchReproState) -> PaperBenchReproState:
    """Run one repair-stage regeneration pass within the repair phase."""
    return _run_node(state, "repair", _repair_regeneration_impl)


def _repair_failure_signature(
    state: PaperBenchReproState,
    *,
    round_touched_files: list[str],
) -> dict[str, Any]:
    """Build a compact deterministic signature for ineffective-iteration detection."""
    report = state.validation_report or ValidationReport()
    failure_categories = sorted(
        {
            str(item or "").strip().lower()
            for item in list(report.failure_categories or [])
            if str(item or "").strip()
        }
    )
    trigger_signals = sorted(
        {
            str(item or "").strip()
            for item in list(state.repair_ticket.trigger_signals if state.repair_ticket else [])
            if str(item or "").strip()
        }
    )
    failed_check_names = sorted(
        {
            str(item.name or "").strip()
            for item in _all_failed_validation_checks(report)
            if str(item.name or "").strip()
        }
    )
    touched_files = sorted(
        {
            str(item or "").strip()
            for item in list(round_touched_files or [])
            if str(item or "").strip()
        }
    )
    return {
        "failure_categories": failure_categories,
        "trigger_signals": trigger_signals,
        "failed_check_names": failed_check_names,
        "touched_files": touched_files,
        "had_changes": bool(touched_files),
    }


def _classify_terminal_outcome(
    state: PaperBenchReproState,
    *,
    handoff_ready: bool,
    budget_exhausted: bool,
    stagnation_detected: bool,
) -> tuple[str, str]:
    """Classify completion outcome for non-breaking strict mode rollout."""
    if handoff_ready:
        return "completed", "repo validation and repair acceptance passed"

    report = state.validation_report
    if report is None:
        reason = "repair exited without a validation report"
        if budget_exhausted:
            reason += "; repair budget exhausted"
        return "completed_unverified", reason

    failure_categories = {
        str(item or "").strip().lower()
        for item in list(report.failure_categories or [])
        if str(item or "").strip()
    }
    suffix: list[str] = []
    if budget_exhausted:
        suffix.append("repair budget exhausted")
    if stagnation_detected:
        suffix.append("stagnation detected")
    suffix_text = f"; {', '.join(suffix)}" if suffix else ""

    if bool({"semantic", "trace", "integration"}.intersection(failure_categories)):
        return (
            "completed_with_degraded_contract",
            "semantic or integration repo contract remains open" + suffix_text,
        )
    if bool({"runtime", "implementation", "artifact"}.intersection(failure_categories)) or report.overall_status in {"failed", "partial"}:
        return (
            "completed_with_runtime_risk",
            "runtime closure remains risky" + suffix_text,
        )
    return "completed_unverified", "repair finished without verifiable closure" + suffix_text


def repair_node(state: PaperBenchReproState) -> PaperBenchReproState:
    """Run validation, semantic repair planning, and eval/gen repair looping."""

    def _repair_ready_for_handoff(current_state: PaperBenchReproState) -> bool:
        if current_state.validation_report is None or not current_state.validation_report.passed:
            return False
        review_payload = _build_repair_review_payload(current_state)
        return bool(dict(review_payload.get("repo_level_review", {}) or {}).get("passed", False))

    state.temp_data.pop("repair_round", None)
    state.temp_data.pop("repair_stagnation", None)
    _ensure_prepare_quality_gate_passed(state)
    state = _refresh_canonical_ir_for_repair_context(state)
    validation_only_resume = (
        os.environ.get("PAPERBENCH_REPRO_REPAIR_VALIDATION_ONLY") == "1"
        and
        bool(getattr(state.input, "resume_in_place", False))
        and str(getattr(state.input, "resume_start_stage", "") or "").strip() == "repair_validation"
    )
    state = _run_repair_validation_pass(state)
    if state.status == "failed":
        return state
    if validation_only_resume:
        if state.validation_report is not None and state.validation_report.passed:
            state.terminal_outcome = "completed"
            state.terminal_outcome_reason = "repair validation passed in validation-only resume"
            state.status = "completed"
        else:
            state.terminal_outcome = "completed_unverified"
            state.terminal_outcome_reason = "repair validation-only resume completed with open failures"
            state.status = "completed_unverified"
        state.temp_data["terminal_outcome"] = state.terminal_outcome
        state.temp_data["terminal_outcome_reason"] = state.terminal_outcome_reason
        _write_stage_output(
            state,
            "terminal_outcome.json",
            {
                "terminal_outcome": state.terminal_outcome,
                "reason": state.terminal_outcome_reason,
                "validation_only_resume": True,
            },
        )
        _save_tracking_artifacts(state)
        return state
    if state.validation_report is None:
        state.status = "completed_unverified"
        state.terminal_outcome = "completed_unverified"
        state.terminal_outcome_reason = "repair validation did not produce a validation_report"
        state.temp_data["terminal_outcome"] = state.terminal_outcome
        state.temp_data["terminal_outcome_reason"] = state.terminal_outcome_reason
        state.failed_node = ""
        state.current_node = "repair"
        state.error_message = "repair validation did not produce a validation_report"
        recovery_ticket = {
            "ticket_type": "repair_validation_recovery",
            "node": "repair",
            "error": state.error_message,
            "next_action": "finalize_with_evidence",
            "terminal_outcome": state.terminal_outcome,
        }
        state.temp_data.setdefault("node_errors", []).append(recovery_ticket)
        state.temp_data.setdefault("recovery_tickets", []).append(recovery_ticket)
        _write_stage_output(state, "repair_review.json", _build_repair_review_payload(state))
        _write_stage_output(state, "recovery_tickets.json", state.temp_data.get("recovery_tickets", []))
        _write_stage_output(
            state,
            "terminal_outcome.json",
            {
                "terminal_outcome": state.terminal_outcome,
                "reason": state.terminal_outcome_reason,
                "handoff_ready": False,
            },
        )
        _save_tracking_artifacts(state)
        return state
    if _repair_ready_for_handoff(state):
        state.terminal_outcome = "completed"
        state.terminal_outcome_reason = "repo validation and repair acceptance passed"
        state.temp_data["terminal_outcome"] = state.terminal_outcome
        state.temp_data["terminal_outcome_reason"] = state.terminal_outcome_reason
        _write_stage_output(
            state,
            "terminal_outcome.json",
            {
                "terminal_outcome": state.terminal_outcome,
                "reason": state.terminal_outcome_reason,
            },
        )
        return state

    round_id = 1
    round_budget = _repair_round_budget(state)
    budget_exhausted = False
    stagnation_detected = False
    stagnation_repeats = 0
    previous_failure_signature: dict[str, Any] | None = None
    while state.validation_report is not None and not _repair_ready_for_handoff(state):
        if round_id > round_budget:
            budget_exhausted = True
            break
        if semantic_anchor_disabled():
            state.requirement_anchor = None
        elif state.requirement_anchor is None:
            state.requirement_anchor = _build_requirement_anchor(state)
        state = _run_repair_plan_pass(state)
        if state.status == "failed":
            return state
        round_budget = _repair_round_budget(state)
        if round_id > round_budget:
            budget_exhausted = True
            break
        if state.validation_report is not None:
            _write_repair_round_artifact(state, round_id, "validation_before.json", state.validation_report)
        if state.repair_plan is not None:
            _write_repair_round_artifact(state, round_id, "repair_plan.json", state.repair_plan)
        state.temp_data["repair_round"] = round_id
        state = _run_repair_regeneration_pass(state)
        if state.status == "failed":
            return state
        pending_attempt = dict(state.temp_data.get("pending_repair_regeneration_attempt", {}) or {})
        if pending_attempt:
            _write_repair_round_artifact(state, round_id, "repair_attempt.json", pending_attempt)
        round_touched_files = [
            str(item).strip()
            for item in list(pending_attempt.get("touched_files", []) or [])
            if str(item).strip()
        ]
        state.temp_data["repair_round"] = round_id
        state = _run_repair_validation_pass(state)
        if state.status == "failed":
            return state
        if state.validation_report is not None:
            _write_repair_round_artifact(state, round_id, "validation_after.json", state.validation_report)
        if state.execution_result is not None:
            _write_repair_round_artifact(state, round_id, "execution_after.json", state.execution_result)
        if not _repair_ready_for_handoff(state):
            current_failure_signature = _repair_failure_signature(
                state,
                round_touched_files=round_touched_files,
            )
            signatures_equal = previous_failure_signature == current_failure_signature
            if signatures_equal:
                stagnation_repeats += 1
            else:
                stagnation_repeats = 0
            previous_failure_signature = current_failure_signature
            if stagnation_repeats >= 1:
                stagnation_detected = True
                state.temp_data["repair_stagnation"] = {
                    "detected": True,
                    "repeat_count": stagnation_repeats + 1,
                    "failure_signature": current_failure_signature,
                    "round_id": round_id,
                    "reason": "repair failure signature remained unchanged for consecutive rounds",
                }
                break
        round_id += 1

    if state.repair_log is None:
        state.repair_log = RepairLog(converged=False, rounds_attempted=0, actions=[])
    if state.repair_log is not None and state.validation_report is not None:
        if stagnation_detected:
            actions = list(state.repair_log.actions)
            actions.append(
                RepairAction(
                    round_id=max(1, int(state.temp_data.get("repair_round", 0) or 0)),
                    action_type="ineffective-iteration-escalation",
                    reason="repair failure signature remained unchanged across rounds",
                    touched_work_packages=[],
                    touched_files=list(
                        dict.fromkeys(
                            state.temp_data.get("repair_stagnation", {}).get("failure_signature", {}).get("touched_files", [])
                        )
                    ),
                    outcome="degraded_continue",
                )
            )
            state.repair_log = state.repair_log.model_copy(update={"actions": actions})
        state.repair_log = state.repair_log.model_copy(
            update={
                "converged": _repair_ready_for_handoff(state),
                "rounds_attempted": max(state.repair_log.rounds_attempted, int(state.temp_data.get("repair_round", 0) or 0)),
            }
        )
        _write_stage_output(state, "repair_log.json", state.repair_log)
    final_review_payload = _build_repair_review_payload(state)
    state.temp_data["repair_review"] = final_review_payload
    _write_stage_output(state, "repair_review.json", final_review_payload)
    state.temp_data.pop("repair_round", None)
    handoff_ready = _repair_ready_for_handoff(state)
    terminal_outcome, terminal_reason = _classify_terminal_outcome(
        state,
        handoff_ready=handoff_ready,
        budget_exhausted=budget_exhausted,
        stagnation_detected=stagnation_detected,
    )
    state.status = terminal_outcome
    state.terminal_outcome = terminal_outcome
    state.terminal_outcome_reason = terminal_reason
    state.temp_data["terminal_outcome"] = terminal_outcome
    state.temp_data["terminal_outcome_reason"] = terminal_reason
    _write_stage_output(
        state,
        "terminal_outcome.json",
        {
            "terminal_outcome": terminal_outcome,
            "reason": terminal_reason,
            "handoff_ready": handoff_ready,
            "budget_exhausted": budget_exhausted,
            "stagnation_detected": stagnation_detected,
        },
    )
    if not handoff_ready:
        state.current_node = "repair"
        state.failed_node = ""
        state.error_message = terminal_reason
        state.temp_data.setdefault("node_errors", []).append(
            {
                "node": "repair",
                "error": terminal_reason,
                "terminal_outcome": terminal_outcome,
            }
        )
    else:
        state.failed_node = ""
        state.error_message = ""
    _save_tracking_artifacts(state)
    return state


def build_workflow():
    """Build reproagent workflow graph."""
    if not LANGGRAPH_AVAILABLE:
        class SimpleWorkflow:
            def invoke(self, input_data):
                payload = dict(input_data)
                run_id = str(payload.pop("run_id", "") or "").strip()
                state = PaperBenchReproState(input=PaperBenchReproInput(**payload), run_id=run_id)
                state = _hydrate_state_for_in_place_resume(state)
                resume_stage = str(getattr(state.input, "resume_start_stage", "") or "").strip()
                if bool(getattr(state.input, "resume_in_place", False)) and resume_stage == "repair_validation":
                    state = repair_node(state)
                    return state
                if bool(getattr(state.input, "resume_in_place", False)) and resume_stage == "local_file_generation":
                    state = generate_node(state)
                    if _is_terminal_blocked(state):
                        return state
                    state = repair_node(state)
                    return state
                state = prepare_node(state)
                if _is_terminal_blocked(state):
                    return state
                state = plan_node(state)
                if _is_terminal_blocked(state):
                    return state
                state = normalization_gate_node(state)
                if _is_terminal_blocked(state):
                    return state
                state = generate_node(state)
                if _is_terminal_blocked(state):
                    return state
                state = repair_node(state)
                return state
        return SimpleWorkflow()

    class WorkflowWrapper:
        def __init__(self, compiled_graph):
            self.graph = compiled_graph

        def invoke(self, input_data):
            payload = dict(input_data)
            run_id = str(payload.pop("run_id", "") or "").strip()
            state = PaperBenchReproState(input=PaperBenchReproInput(**payload), run_id=run_id)
            state = _hydrate_state_for_in_place_resume(state)
            result = self.graph.invoke(state)
            return PaperBenchReproState(**result) if isinstance(result, dict) else result

    graph = StateGraph(PaperBenchReproState)
    def resume_router(state: PaperBenchReproState) -> str:
        if bool(getattr(state.input, "resume_in_place", False)) and str(getattr(state.input, "resume_start_stage", "") or "").strip() == "repair_validation":
            return "repair"
        if bool(getattr(state.input, "resume_in_place", False)) and str(getattr(state.input, "resume_start_stage", "") or "").strip() == "local_file_generation":
            return "generate"
        return "prepare"

    def resume_router_node(state: PaperBenchReproState) -> PaperBenchReproState:
        return state

    graph.add_node("resume_router", resume_router_node)
    graph.add_node("prepare", prepare_node)
    graph.add_node("plan", plan_node)
    graph.add_node("normalization_gate", normalization_gate_node)
    graph.add_node("generate", generate_node)
    graph.add_node("repair", repair_node)

    graph.set_entry_point("resume_router")
    graph.add_conditional_edges("resume_router", resume_router)
    graph.add_edge("prepare", "plan")

    def route_after_plan(state: PaperBenchReproState) -> str:
        return END if _is_terminal_blocked(state) else "normalization_gate"

    def route_after_normalization(state: PaperBenchReproState) -> str:
        return END if _is_terminal_blocked(state) else "generate"

    def route_after_generate(state: PaperBenchReproState) -> str:
        return END if _is_terminal_blocked(state) else "repair"

    graph.add_conditional_edges("plan", route_after_plan)
    graph.add_conditional_edges("normalization_gate", route_after_normalization)
    graph.add_conditional_edges("generate", route_after_generate)
    graph.add_edge("repair", END)

    return WorkflowWrapper(graph.compile())
