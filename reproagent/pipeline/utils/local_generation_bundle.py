"""Helpers for persisting and restoring reproagent local generation bundles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from reproagent.pipeline.schemas import (
    BenchmarkReport,
    EvaluationDecision,
    ExecutionResult,
    PaperBenchReproState,
    GenerateStageOutput,
    GenerationCheckpoint,
    GenerationManifest,
    PreflightResult,
    ProjectPlan,
    RepoPlan,
    RuntimeProbe,
    ValidationReport,
)


def apply_local_generation_bundle(state: PaperBenchReproState, payload: dict[str, Any]) -> None:
    """Hydrate workflow state from a persisted local-file-generation payload."""
    repo_plan_payload = payload.get("repo_plan", {})
    state.repo_plan = RepoPlan.model_validate(repo_plan_payload) if repo_plan_payload else None
    state.project_plan = ProjectPlan.model_validate(payload.get("project_plan", {}))
    generation_manifest_payload = payload.get("generation_manifest", {})
    state.generation_manifest = (
        GenerationManifest.model_validate(generation_manifest_payload)
        if generation_manifest_payload
        else None
    )
    state.generated_files = list(payload.get("generated_files", []))
    state.project_root = str(payload.get("project_root", ""))
    state.project_manifest = dict(payload.get("project_manifest", {}))
    state.code = str(payload.get("code", ""))
    execution_result = payload.get("execution_result") or {}
    state.execution_result = ExecutionResult.model_validate(execution_result) if execution_result else None
    evaluation_payload = payload.get("evaluation") or {}
    state.evaluation = EvaluationDecision.model_validate(evaluation_payload) if evaluation_payload else None
    preflight_payload = payload.get("preflight_result") or {}
    state.preflight_result = PreflightResult.model_validate(preflight_payload) if preflight_payload else None
    state.experiment_results = dict(payload.get("experiment_results", {}))
    runtime_probe_payload = payload.get("runtime_probe") or {}
    state.runtime_probe = RuntimeProbe.model_validate(runtime_probe_payload) if runtime_probe_payload else None
    validation_report_payload = payload.get("validation_report") or {}
    state.validation_report = (
        ValidationReport.model_validate(validation_report_payload)
        if validation_report_payload
        else None
    )
    benchmark_report_payload = payload.get("benchmark_report") or {}
    state.benchmark_report = (
        BenchmarkReport.model_validate(benchmark_report_payload)
        if benchmark_report_payload
        else None
    )
    generate_stage_output_payload = payload.get("generate_stage_output") or {}
    state.generate_stage_output = (
        GenerateStageOutput.model_validate(generate_stage_output_payload)
        if generate_stage_output_payload
        else None
    )
    state.checkpoint_path = str(payload.get("checkpoint_path", ""))
    state.execution_history = list(payload.get("execution_history", []))
    state.iteration_count = int(payload.get("iteration_count", 0) or 0)
    if isinstance(payload.get("file_provenance"), list):
        state.temp_data["file_provenance"] = list(payload.get("file_provenance") or [])
    generation_checkpoints = payload.get("generation_checkpoints", [])
    state.generation_checkpoints = [
        item if hasattr(item, "model_dump") else GenerationCheckpoint.model_validate(item)
        for item in generation_checkpoints
    ]


def load_local_generation_bundle(
    state: PaperBenchReproState,
    *,
    get_output_dir: Callable[[PaperBenchReproState], Path],
) -> dict[str, Any]:
    """Load the persisted local-file-generation payload for resume."""
    output_dir = get_output_dir(state)
    node_dir = output_dir / "nodes" / "generate"
    def _read_json(*candidates: Path) -> dict[str, Any] | list[Any]:
        for path in candidates:
            if path.exists():
                return json.loads(path.read_text(encoding="utf-8"))
        raise FileNotFoundError(candidates[0])

    repo_plan_payload = (
        _read_json(node_dir / "repo_plan.json", output_dir / "repo_plan.json")
        if (node_dir / "repo_plan.json").exists() or (output_dir / "repo_plan.json").exists()
        else {}
    )
    project_plan_payload = _read_json(node_dir / "project_plan.json", output_dir / "project_plan.json")
    experiment_output_payload = _read_json(node_dir / "experiment_output.json", output_dir / "experiment_output.json")
    generation_manifest_payload = (
        _read_json(node_dir / "generation_manifest.json", output_dir / "generation_manifest.json")
        if (node_dir / "generation_manifest.json").exists() or (output_dir / "generation_manifest.json").exists()
        else experiment_output_payload.get("generation_manifest", {})
    )
    execution_result_payload = (
        json.loads((node_dir / "execution_result.json").read_text(encoding="utf-8"))
        if (node_dir / "execution_result.json").exists()
        else {}
    )
    runtime_probe_payload = (
        json.loads((node_dir / "runtime_probe.json").read_text(encoding="utf-8"))
        if (node_dir / "runtime_probe.json").exists()
        else {}
    )
    validation_report_payload = (
        json.loads((node_dir / "validation_report.json").read_text(encoding="utf-8"))
        if (node_dir / "validation_report.json").exists()
        else {}
    )
    benchmark_report_payload = (
        json.loads((node_dir / "benchmark_report.json").read_text(encoding="utf-8"))
        if (node_dir / "benchmark_report.json").exists()
        else {}
    )
    execution_history_payload = (
        json.loads((node_dir / "execution_history.json").read_text(encoding="utf-8"))
        if (node_dir / "execution_history.json").exists()
        else []
    )
    preflight_payload = (
        json.loads((node_dir / "preflight.json").read_text(encoding="utf-8"))
        if (node_dir / "preflight.json").exists()
        else {}
    )
    generation_checkpoints_payload = (
        _read_json(node_dir / "generation_checkpoints.json", output_dir / "generation_checkpoints.json")
        if (node_dir / "generation_checkpoints.json").exists() or (output_dir / "generation_checkpoints.json").exists()
        else []
    )
    file_provenance_payload = (
        _read_json(node_dir / "file_provenance.json", output_dir / "file_provenance.json")
        if (node_dir / "file_provenance.json").exists() or (output_dir / "file_provenance.json").exists()
        else []
    )
    checkpoint_path = (
        (node_dir / "iteration_checkpoint.json")
        if (node_dir / "iteration_checkpoint.json").exists()
        else (output_dir / "iteration_checkpoint.json")
    )
    checkpoint_payload = (
        json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint_path.exists()
        else {}
    )
    last_attempt_payload = (
        json.loads((node_dir / "last_attempt.json").read_text(encoding="utf-8"))
        if (node_dir / "last_attempt.json").exists()
        else {}
    )
    if (
        not execution_history_payload
        and isinstance(last_attempt_payload, dict)
        and isinstance(checkpoint_payload, dict)
        and str(last_attempt_payload.get("task_id", "") or "").strip()
        and str(checkpoint_payload.get("latest_status", "") or "").strip() == "passed"
    ):
        task_id = str(last_attempt_payload.get("task_id", "") or "").strip()
        project_manifest_payload = dict(last_attempt_payload.get("project_manifest", {}) or {})
        generated_files_payload = [
            str(item).strip()
            for item in list(project_manifest_payload.get("generated_files", []) or [])
            if str(item).strip()
        ]
        changed_files_payload = [
            str(item).strip()
            for item in list(last_attempt_payload.get("changed_files", []) or [])
            if str(item).strip()
        ]
        file_path = changed_files_payload[0] if changed_files_payload else (
            generated_files_payload[0] if generated_files_payload else ""
        )
        execution_result_from_attempt = dict(last_attempt_payload.get("execution_result", {}) or {})
        review_checks = list(execution_result_from_attempt.get("checks", []) or [])
        task_review = {
            "task_id": task_id,
            "review_stage": "task_review",
            "success": bool(execution_result_from_attempt.get("success", True)),
            "checks": review_checks,
            "review_points": [],
            "failure_summary": [],
            "suggestions": list(last_attempt_payload.get("suggestions", []) or []),
        }
        execution_history_payload = [
            {
                "iteration": int(last_attempt_payload.get("iteration", 0) or 0),
                "task_sequence": int(project_manifest_payload.get("task_sequence", 0) or 0),
                "task_id": task_id,
                "task_contract_hash": "",
                "file_path": file_path,
                "generated_files": generated_files_payload,
                "result": execution_result_from_attempt,
                "suggestions": list(last_attempt_payload.get("suggestions", []) or []),
                "repair_trace": list(last_attempt_payload.get("repair_trace", []) or []),
                "context_usage": dict(last_attempt_payload.get("context_usage", {}) or {}),
                "task_review": task_review,
                "task_review_attempt_count": len(list(last_attempt_payload.get("repair_trace", []) or [])) or 1,
                "materialization_mode": "generate",
                "recovered_from_last_attempt": True,
            }
        ]
    entrypoints = project_plan_payload.get("entrypoints", {})
    repo_root = output_dir / "repo"
    main_path = repo_root / entrypoints.get("main", "main.py")
    return {
        "repo_plan": repo_plan_payload,
        "project_plan": project_plan_payload,
        "generation_manifest": generation_manifest_payload,
        "generated_files": experiment_output_payload.get("generated_files", []),
        "project_root": str(repo_root.resolve()),
        "project_manifest": _read_json(node_dir / "project_manifest.json") if (node_dir / "project_manifest.json").exists() else {},
        "code": main_path.read_text(encoding="utf-8") if main_path.exists() else "",
        "execution_result": execution_result_payload,
        "runtime_probe": runtime_probe_payload,
        "validation_report": validation_report_payload,
        "benchmark_report": benchmark_report_payload,
        "preflight_result": preflight_payload,
        "experiment_results": experiment_output_payload.get("experiment_results", {}),
        "generate_stage_output": experiment_output_payload,
        "file_provenance": file_provenance_payload,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "execution_history": execution_history_payload,
        "iteration_count": int(experiment_output_payload.get("iteration_state", {}).get("iteration_count", 0) or 0),
        "generation_checkpoints": generation_checkpoints_payload,
    }
