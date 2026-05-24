"""Storage and artifact persistence helpers for reproagent."""

import json
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

from reproagent.pipeline.schemas import PaperBenchReproState

from . import run_context
from .artifact_names import CANONICAL_ARTIFACTS
from .artifact_writer import register_existing_file
from .dataset_manager import _get_dataset_preparation, _get_resource_manifest


_PLAN_SOURCE_OF_TRUTH_FILENAMES = {
    CANONICAL_ARTIFACTS["architecture"],
    CANONICAL_ARTIFACTS["boundary_requirements"],
    CANONICAL_ARTIFACTS["canonical_ir"],
    CANONICAL_ARTIFACTS["global_contract"],
    CANONICAL_ARTIFACTS["package_file_planning"],
    CANONICAL_ARTIFACTS["pipeline_plan"],
    CANONICAL_ARTIFACTS["reference_selection"],
    "work_packages.json",
}


_OUTPUT_DIR_OVERRIDE: ContextVar[Callable[[PaperBenchReproState], Path] | None] = ContextVar(
    "reproagent_storage_output_dir_override",
    default=None,
)
_JSON_DEFAULT_OVERRIDE: ContextVar[Callable[[Any], Any] | None] = ContextVar(
    "reproagent_storage_json_default_override",
    default=None,
)


def _get_output_dir(state: PaperBenchReproState) -> Path:
    override = _OUTPUT_DIR_OVERRIDE.get()
    if override is not None:
        return override(state)
    return run_context._get_output_dir(state)


def _json_default(value: Any) -> Any:
    override = _JSON_DEFAULT_OVERRIDE.get()
    if override is not None:
        return override(value)
    return run_context._json_default(value)


@contextmanager
def storage_context(
    *,
    output_dir_fn: Callable[[PaperBenchReproState], Path] | None = None,
    json_default_fn: Callable[[Any], Any] | None = None,
) -> Iterator[None]:
    """Temporarily override storage helpers without mutating module globals."""
    output_token = _OUTPUT_DIR_OVERRIDE.set(output_dir_fn) if output_dir_fn is not None else None
    json_token = _JSON_DEFAULT_OVERRIDE.set(json_default_fn) if json_default_fn is not None else None
    try:
        yield
    finally:
        if json_token is not None:
            _JSON_DEFAULT_OVERRIDE.reset(json_token)
        if output_token is not None:
            _OUTPUT_DIR_OVERRIDE.reset(output_token)


def _refresh_generate_artifact_index(state: PaperBenchReproState) -> None:
    """Refresh run_manifest artifact_index after generate node artifacts land on disk."""
    if not state.run_id:
        return
    run_manifest_path = _get_output_dir(state) / "run_manifest.json"
    if not run_manifest_path.exists():
        return
    try:
        payload = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return
    artifact_index = payload.get("artifact_index", {})
    if not isinstance(artifact_index, dict):
        artifact_index = {}
    output_dir = _get_output_dir(state)
    candidates = {
        "node_generate_project_manifest": output_dir / "nodes" / "generate" / "project_manifest.json",
        "node_generate_execution_result": output_dir / "nodes" / "generate" / "execution_result.json",
        "node_generate_execution_history": output_dir / "nodes" / "generate" / "execution_history.json",
        "run_summary": output_dir / "run_summary.json",
        "latest_state": output_dir / "latest_state.json",
    }
    for key, path in candidates.items():
        if path.exists():
            artifact_index[key] = str(path.resolve())
    payload["artifact_index"] = artifact_index
    run_manifest_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    register_existing_file(
        run_manifest_path,
        run_dir=output_dir,
        logical_name="run_manifest",
        kind="manifest",
        authority="source_of_truth",
    )


def _append_event(state: PaperBenchReproState, event_type: str, payload: dict) -> None:
    """Append workflow event to the run jsonl log."""
    if not state.run_id:
        return

    output_dir = _get_output_dir(state)
    output_dir.mkdir(parents=True, exist_ok=True)
    event_path = output_dir / "workflow_events.jsonl"
    with event_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event_type,
            "payload": payload,
        }, ensure_ascii=False, default=_json_default) + "\n")
    register_existing_file(
        event_path,
        run_dir=output_dir,
        logical_name="workflow_events",
        kind="log",
        authority="source_of_truth",
    )

def _save_state_snapshot(state: PaperBenchReproState, snapshot_name: str) -> None:
    """Persist a full workflow state snapshot."""
    if not state.run_id:
        return

    output_dir = _get_output_dir(state)
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshots_dir = output_dir / "state_snapshots"
    snapshots_dir.mkdir(exist_ok=True)

    payload = state.model_dump(mode="json", exclude_none=False)
    snapshot_text = json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default)
    snapshot_path = snapshots_dir / f"{snapshot_name}.json"
    latest_path = output_dir / "latest_state.json"
    snapshot_path.write_text(snapshot_text, encoding="utf-8")
    latest_path.write_text(snapshot_text, encoding="utf-8")
    register_existing_file(
        snapshot_path,
        run_dir=output_dir,
        logical_name=f"state_snapshot.{snapshot_name}",
        kind="debug",
        authority="debug_snapshot",
        retention="debug",
    )
    register_existing_file(
        latest_path,
        run_dir=output_dir,
        logical_name="state_snapshot.latest",
        kind="state",
        authority="derived",
    )


def _write_json(
    path: Path,
    payload: object,
    *,
    logical_name: str = "",
    kind: str = "output",
    stage: str = "",
    node: str = "",
    authority: str = "derived",
    retention: str = "keep",
) -> None:
    """Persist one JSON payload."""
    json_payload = payload.model_dump(mode="json") if hasattr(payload, "model_dump") else payload
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    register_existing_file(
        path,
        logical_name=logical_name or path.stem,
        kind=kind,
        stage=stage,
        node=node,
        authority=authority,
        retention=retention,
    )


def _write_node_and_root_json(
    state: PaperBenchReproState,
    node_dir: Path,
    filename: str,
    payload: object,
    *,
    write_root: bool = False,
) -> None:
    """Persist one JSON payload into the node directory and optionally the run root."""
    is_plan_source = node_dir.name == "plan" and filename in _PLAN_SOURCE_OF_TRUTH_FILENAMES
    _write_json(
        node_dir / filename,
        payload,
        kind="contract" if is_plan_source else "output",
        stage=node_dir.name,
        node=node_dir.name,
        authority="source_of_truth" if is_plan_source else "derived",
    )
    if write_root:
        _write_json(
            _get_output_dir(state) / filename,
            payload,
            kind="contract" if is_plan_source else "output",
            stage=node_dir.name if is_plan_source else "",
            node=node_dir.name if is_plan_source else "",
            authority="source_of_truth" if is_plan_source else "derived",
        )


def _write_node_text(node_dir: Path, filename: str, text: str) -> None:
    """Persist one text payload under the node directory."""
    path = node_dir / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    register_existing_file(
        path,
        logical_name=path.stem,
        kind="log" if path.suffix in {".txt", ".md"} else "output",
        authority="debug_snapshot",
        retention="debug",
    )


def _recover_generate_state_from_checkpoint(state: PaperBenchReproState, node_dir: Path) -> None:
    """Keep failed generate snapshots from overwriting per-task checkpoint progress."""
    if state.execution_history:
        return
    checkpoint_path = node_dir / "iteration_checkpoint.json"
    if not checkpoint_path.exists():
        return
    try:
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except Exception:
        return
    if not isinstance(checkpoint, dict):
        return
    generated_files = [
        str(item).strip()
        for item in list(checkpoint.get("generated_files", []) or [])
        if str(item).strip()
    ]
    if not generated_files:
        return
    repo_dir = _get_output_dir(state) / "repo"
    existing_files = [path for path in generated_files if (repo_dir / path).exists()]
    if not existing_files:
        return
    task_rows: list[dict[str, Any]] = []
    if state.generation_manifest is not None:
        for task in list(state.generation_manifest.tasks or []):
            task_rows.append(
                {
                    "task_id": str(task.task_id or "").strip(),
                    "file_path": str(task.file_path or "").strip(),
                    "review_points": list(task.review_points or []),
                }
            )
    if not task_rows and (node_dir / "generation_manifest.json").exists():
        try:
            manifest = json.loads((node_dir / "generation_manifest.json").read_text(encoding="utf-8"))
        except Exception:
            manifest = {}
        if isinstance(manifest, dict):
            for task in list(manifest.get("tasks", []) or []):
                if not isinstance(task, dict):
                    continue
                task_rows.append(
                    {
                        "task_id": str(task.get("task_id", "") or "").strip(),
                        "file_path": str(task.get("file_path", "") or "").strip(),
                        "review_points": list(task.get("review_points", []) or []),
                    }
                )
    task_by_path = {
        item["file_path"]: item
        for item in task_rows
        if item.get("file_path")
    }
    recovered_history = []
    for sequence, path in enumerate(existing_files):
        task = task_by_path.get(path, {})
        task_id = str(task.get("task_id", "") or path).strip()
        check = {
            "name": "recovered_checkpoint_file_exists",
            "task_id": task_id,
            "file_path": path,
            "passed": True,
            "error": "",
        }
        recovered_history.append(
            {
                "iteration": 0,
                "task_sequence": sequence,
                "task_id": task_id,
                "task_contract_hash": "",
                "file_path": path,
                "generated_files": existing_files,
                "result": {
                    "success": True,
                    "output": "Recovered from iteration checkpoint after generate interruption.",
                    "error": "",
                    "exit_code": 0,
                    "metrics": {},
                    "checks": [check],
                    "artifacts": [],
                    "artifact_summary": {},
                },
                "suggestions": [],
                "repair_trace": [
                    {
                        "attempt": 1,
                        "success": True,
                        "changed_files": [path],
                        "suggestions": [],
                        "recovered_from_iteration_checkpoint": True,
                    }
                ],
                "context_usage": {},
                "task_review": {
                    "task_id": task_id,
                    "review_stage": "task_review",
                    "success": True,
                    "checks": [check],
                    "review_points": list(task.get("review_points", []) or []),
                    "failure_summary": [],
                    "suggestions": [],
                    "recovered_from_iteration_checkpoint": True,
                },
                "task_review_attempt_count": 1,
                "materialization_mode": "generate",
                "recovered_from_iteration_checkpoint": True,
            }
        )
    state.execution_history = recovered_history
    state.generated_files = sorted(existing_files)
    if state.project_root == "":
        state.project_root = str(repo_dir.resolve())


def _save_node_artifacts(state: PaperBenchReproState, node_name: str) -> None:
    """Persist node-level artifacts for debugging and review."""
    if not state.run_id:
        return

    output_dir = _get_output_dir(state)
    node_dir = output_dir / "nodes" / node_name
    node_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        node_dir / "summary.json",
        {
            "run_id": state.run_id,
            "node": node_name,
            "status": state.status,
            "current_node": state.current_node,
            "failed_node": state.failed_node,
            "error_message": state.error_message,
            "iteration_count": state.iteration_count,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    )

    if node_name == "prepare":
        recovery_tickets = state.temp_data.get("recovery_tickets")
        if recovery_tickets:
            _write_json(node_dir / "recovery_tickets.json", recovery_tickets)
        _write_json(node_dir / "input.json", state.input.model_dump(mode="json"))
        if state.upstream_intent is not None:
            _write_json(node_dir / "upstream_intent.json", state.upstream_intent)
        if state.paper_chunks:
            _write_json(
                node_dir / "paper_chunks.json",
                [item.model_dump(mode="json") for item in state.paper_chunks],
            )
        _write_node_text(node_dir, "target.txt", state.input.target)
        resource_manifest = _get_resource_manifest(state)
        _write_json(node_dir / "resource_manifest.json", resource_manifest)
        reference_repo_preparation = state.temp_data.get("reference_repo_preparation", {})
        _write_json(node_dir / "reference_repo_preparation.json", reference_repo_preparation)
        prepared_repositories = reference_repo_preparation.get("prepared_repositories", [])
        _write_json(node_dir / "reference_repos.json", prepared_repositories)
        if state.normalized_input is not None:
            _write_json(node_dir / "input_normalization.json", state.normalized_input)
        if state.unit_extraction is not None:
            _write_json(node_dir / "unit_extraction.json", state.unit_extraction)
            _write_json(
                node_dir / "units.json",
                [item.model_dump(mode="json") for item in state.unit_extraction.units],
            )
        if state.reference_repo_surveys:
            _write_json(
                node_dir / "reference_repo_surveys.json",
                [item.model_dump(mode="json") for item in state.reference_repo_surveys],
            )

    if node_name == "plan":
        recovery_tickets = state.temp_data.get("recovery_tickets")
        if recovery_tickets:
            _write_node_and_root_json(state, node_dir, "recovery_tickets.json", recovery_tickets)
        if state.plan:
            _write_node_text(node_dir, "plan.md", state.plan)
            _write_json(node_dir / "plan_steps.json", state.temp_data.get("steps", []))
        if state.boundary_requirements is not None:
            _write_node_and_root_json(state, node_dir, CANONICAL_ARTIFACTS["boundary_requirements"], state.boundary_requirements)
        if state.topic_profile is not None:
            _write_node_and_root_json(state, node_dir, CANONICAL_ARTIFACTS["topic_profile"], state.topic_profile)
        if state.reference_selection is not None:
            _write_node_and_root_json(state, node_dir, CANONICAL_ARTIFACTS["reference_selection"], state.reference_selection)
        if state.pipeline_plan is not None:
            _write_node_and_root_json(state, node_dir, CANONICAL_ARTIFACTS["pipeline_plan"], state.pipeline_plan)
        if state.global_contract is not None:
            _write_node_and_root_json(state, node_dir, CANONICAL_ARTIFACTS["global_contract"], state.global_contract)
        if state.work_package_planning is not None:
            _write_node_and_root_json(state, node_dir, "work_packages.json", state.work_package_planning)
        if state.evidence_bundles:
            _write_node_and_root_json(
                state,
                node_dir,
                "evidence_bundles.json",
                [item.model_dump(mode="json") for item in state.evidence_bundles],
            )
        if state.evidence_graph:
            _write_node_and_root_json(
                state,
                node_dir,
                "evidence_graph.json",
                [item.model_dump(mode="json") for item in state.evidence_graph],
            )
        if state.architecture is not None:
            _write_node_and_root_json(state, node_dir, CANONICAL_ARTIFACTS["architecture"], state.architecture)
        if state.package_file_planning_output is not None:
            _write_node_and_root_json(
                state,
                node_dir,
                CANONICAL_ARTIFACTS["package_file_planning"],
                state.package_file_planning_output,
            )
        if state.canonical_ir is not None:
            _write_node_and_root_json(
                state,
                node_dir,
                CANONICAL_ARTIFACTS["canonical_ir"],
                state.canonical_ir,
            )
        if state.canonical_ir_validation is not None:
            _write_node_and_root_json(
                state,
                node_dir,
                CANONICAL_ARTIFACTS["canonical_ir_validation"],
                state.canonical_ir_validation,
            )
            _write_node_and_root_json(
                state,
                node_dir,
                CANONICAL_ARTIFACTS["semantic_validation_report"],
                state.canonical_ir_validation.semantic_validation_report,
            )
        if state.canonical_ir is not None:
            _write_node_and_root_json(
                state,
                node_dir,
                CANONICAL_ARTIFACTS["semantic_assertions"],
                {
                    "semantic_assertions": [
                        item.model_dump(mode="json")
                        for item in state.canonical_ir.semantic_assertions
                    ],
                    "evidence_contracts": [
                        item.model_dump(mode="json")
                        for item in state.canonical_ir.evidence_contracts
                    ],
                    "validator_expectations": [
                        item.model_dump(mode="json")
                        for item in state.canonical_ir.validator_expectations
                    ],
                },
            )
        review_payload = state.temp_data.get("stage_reviews", {}).get("work_package_planning")
        if review_payload is not None:
            _write_node_and_root_json(state, node_dir, "work_package_planning.review.json", review_payload)
        review_payload = state.temp_data.get("stage_reviews", {}).get("package_evidence_grounding")
        if review_payload is not None:
            _write_node_and_root_json(state, node_dir, "package_evidence_grounding.review.json", review_payload)
        review_payload = state.temp_data.get("stage_reviews", {}).get("architecture_planning")
        if review_payload is not None:
            _write_node_and_root_json(state, node_dir, "architecture_planning.review.json", review_payload)
        review_payload = state.temp_data.get("stage_reviews", {}).get("package_file_planning")
        if review_payload is not None:
            _write_node_and_root_json(state, node_dir, "package_file_planning.review.json", review_payload)

    if node_name == "generate":
        _recover_generate_state_from_checkpoint(state, node_dir)
        recovery_tickets = state.temp_data.get("recovery_tickets")
        if recovery_tickets:
            _write_node_and_root_json(state, node_dir, "recovery_tickets.json", recovery_tickets)
        checkpoint_payload = _load_iteration_checkpoint_payload(state)
        if state.code:
            _write_node_text(node_dir, "final_code.py", state.code)
        if state.repo_plan is not None:
            _write_json(node_dir / "repo_plan.json", state.repo_plan)
        _write_json(node_dir / "project_plan.json", state.project_plan)
        if state.generation_manifest is not None:
            _write_json(node_dir / "generation_manifest.json", state.generation_manifest)
        if state.generated_files:
            _write_json(node_dir / "generated_files.json", state.generated_files)
        file_provenance = state.temp_data.get("file_provenance")
        if file_provenance:
            _write_json(node_dir / "file_provenance.json", file_provenance)
        if state.generation_checkpoints:
            _write_json(
                node_dir / "generation_checkpoints.json",
                [item.model_dump(mode="json") for item in state.generation_checkpoints],
            )
        if state.project_manifest:
            _write_json(node_dir / "project_manifest.json", state.project_manifest)
        if state.project_root:
            _write_node_text(node_dir, "project_root.txt", state.project_root)
        if state.execution_result is not None:
            _write_json(node_dir / "execution_result.json", state.execution_result)
        if state.preflight_result is not None:
            _write_json(node_dir / "preflight.json", state.preflight_result)
        if state.experiment_results:
            _write_json(node_dir / "experiment_results.json", state.experiment_results)
        if state.evaluation is not None:
            _write_json(node_dir / "evaluation.json", state.evaluation)
        _write_json(node_dir / "execution_history.json", state.execution_history)
        if state.generate_stage_output is not None:
            _write_json(node_dir / "experiment_output.json", state.generate_stage_output)
        if checkpoint_payload:
            _write_json(node_dir / "iteration_checkpoint.json", checkpoint_payload)
        _refresh_generate_artifact_index(state)

    if node_name == "repair":
        recovery_tickets = state.temp_data.get("recovery_tickets")
        if recovery_tickets:
            _write_node_and_root_json(state, node_dir, "recovery_tickets.json", recovery_tickets)
        validation_bundle = state.temp_data.get("validation_bundle")
        if validation_bundle:
            _write_json(node_dir / "validation_bundle.json", validation_bundle)
            _write_json(node_dir / "repair_validation_bundle.json", validation_bundle)
        if state.preflight_result is not None:
            _write_json(node_dir / "preflight.json", state.preflight_result)
        if state.execution_result is not None:
            _write_json(node_dir / "execution_result.json", state.execution_result)
        if state.repair_ticket is not None:
            _write_json(node_dir / "repair_ticket.json", state.repair_ticket)
        repair_plan_context = state.temp_data.get("repair_plan_context")
        if repair_plan_context is not None:
            _write_json(node_dir / "repair_plan_context.json", repair_plan_context)
        repair_plan_draft = state.temp_data.get("repair_plan_draft")
        if repair_plan_draft is not None:
            _write_json(node_dir / "repair_plan_draft.json", repair_plan_draft)
        repair_plan_review = state.temp_data.get("repair_plan_review")
        if repair_plan_review is not None:
            _write_json(node_dir / "repair_plan_review.json", repair_plan_review)
        if state.repair_plan is not None:
            _write_json(node_dir / "repair_plan.json", state.repair_plan)
        if state.runtime_probe is not None:
            _write_json(node_dir / "runtime_probe.json", state.runtime_probe)
        if state.validation_report is not None:
            _write_json(node_dir / "validation_report.json", state.validation_report)
        if state.benchmark_report is not None:
            _write_json(node_dir / "benchmark_report.json", state.benchmark_report)
        if state.repair_log is not None:
            _write_json(node_dir / "repair_log.json", state.repair_log)
            _write_json(
                node_dir / "repair_regeneration_result.json",
                {
                    "validation_report": state.validation_report.model_dump(mode="json") if state.validation_report else {},
                    "benchmark_report": state.benchmark_report.model_dump(mode="json") if state.benchmark_report else {},
                    "repair_log": state.repair_log.model_dump(mode="json"),
                },
            )
        if state.generation_checkpoints:
            _write_json(
                node_dir / "generation_checkpoints.json",
                [item.model_dump(mode="json") for item in state.generation_checkpoints],
            )
        file_provenance = state.temp_data.get("file_provenance")
        if file_provenance:
            _write_json(node_dir / "file_provenance.json", file_provenance)

def _save_run_summary(state: PaperBenchReproState) -> None:
    """Persist a compact final summary for the run."""
    if not state.run_id:
        return

    checkpoint_payload = _load_iteration_checkpoint_payload(state)
    generate_output = state.generate_stage_output.model_dump(mode="json") if state.generate_stage_output else {}
    primary_metric_summary = checkpoint_payload.get("best_metrics", {}) if checkpoint_payload else {}
    dataset_preparation = _get_dataset_preparation(state)
    run_summary_path = _get_output_dir(state) / "run_summary.json"
    run_summary_path.write_text(
        json.dumps({
            "run_id": state.run_id,
            "status": state.status,
            "terminal_outcome": state.terminal_outcome,
            "terminal_outcome_reason": state.terminal_outcome_reason,
            "current_node": state.current_node,
            "failed_node": state.failed_node,
            "error_message": state.error_message,
            "iteration_count": state.iteration_count,
            "target": state.input.target,
            "upstream_intent": state.upstream_intent.model_dump(mode="json") if state.upstream_intent else {},
            "language": state.input.language,
            "max_iterations": state.input.max_iterations,
            "plan_length": len(state.plan),
            "code_length": len(state.code),
            "project_plan_generated": bool(state.project_plan.file_specs),
            "project_root": state.project_root,
            "prepared_datasets_count": len(dataset_preparation.get("downloaded_datasets", [])),
            "generated_files_count": len(state.generated_files),
            "execution_success": state.execution_result.success if state.execution_result else False,
            "evaluation_action": state.evaluation.action if state.evaluation else "",
            "evaluation_reason": state.evaluation.reason if state.evaluation else "",
            "experiment_status": generate_output.get("experiment_status", ""),
            "primary_metric_summary": primary_metric_summary,
            "checkpoint_path": state.checkpoint_path,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    register_existing_file(
        run_summary_path,
        run_dir=_get_output_dir(state),
        logical_name="run_summary",
        kind="report",
        authority="derived",
    )

def _load_iteration_checkpoint_payload(state: PaperBenchReproState) -> dict:
    """Load the persisted iteration checkpoint when available."""
    checkpoint_path = state.checkpoint_path.strip() if state.checkpoint_path else ""
    if not checkpoint_path:
        return {}
    path = Path(checkpoint_path)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}
