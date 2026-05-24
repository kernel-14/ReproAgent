"""Repair-stage implementations for reproagent workflow."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Callable

from reproagent.pipeline.tools import load_project_files
from reproagent.pipeline.schemas import PaperBenchReproState

logger = logging.getLogger(__name__)


def _repair_validation_repo_fingerprint(project_root: str) -> dict[str, Any]:
    """Build a stable repo-content fingerprint for repair validation caching."""
    root = Path(str(project_root or "").strip())
    if not root.exists():
        return {"project_root": str(root), "file_count": 0, "sha256": ""}
    project_files = load_project_files(root)
    digest = hashlib.sha256()
    for relative_path in sorted(project_files):
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(project_files[relative_path].encode("utf-8"))
        digest.update(b"\0")
    return {
        "project_root": str(root.resolve()),
        "file_count": len(project_files),
        "sha256": digest.hexdigest(),
    }


def repair_validation_impl(
    state: PaperBenchReproState,
    *,
    run_repo_validation_bundle: Callable[[PaperBenchReproState], dict[str, Any]],
    load_repo_validation_bundle: Callable[[PaperBenchReproState], dict[str, Any]],
    apply_repo_validation_bundle: Callable[[PaperBenchReproState, dict[str, Any]], None],
    write_stage_output: Callable[[PaperBenchReproState, str, object], None],
    run_or_resume_stage: Callable[..., Any],
    save_tracking_artifacts: Callable[[PaperBenchReproState], None],
) -> PaperBenchReproState:
    logger.info("repair_validation - Running repo-closure validation...")
    repo_fingerprint = _repair_validation_repo_fingerprint(state.project_root)
    input_payload = {
        "repair_round": int(state.temp_data.get("repair_round", 0) or 0),
        "iteration_count": state.iteration_count,
        "generate_stage_output": state.generate_stage_output.model_dump(mode="json") if state.generate_stage_output else {},
        "project_plan": state.project_plan.model_dump(mode="json"),
        "global_contract": state.global_contract.model_dump(mode="json") if state.global_contract else {},
        "repair_plan": state.repair_plan.model_dump(mode="json") if state.repair_plan else {},
        "generated_files": list(state.generated_files),
        "repo_fingerprint": repo_fingerprint,
    }

    def _compute() -> dict[str, Any]:
        return run_repo_validation_bundle(state)

    def _load() -> dict[str, Any]:
        return load_repo_validation_bundle(state)

    def _write(result: dict[str, Any]) -> None:
        apply_repo_validation_bundle(state, result)
        write_stage_output(state, "repair_validation_bundle.json", result)
        write_stage_output(state, "validation_bundle.json", result)
        if state.runtime_probe is not None:
            write_stage_output(state, "runtime_probe.json", state.runtime_probe)
        if state.validation_report is not None:
            write_stage_output(state, "validation_report.json", state.validation_report)
        if state.benchmark_report is not None:
            write_stage_output(state, "benchmark_report.json", state.benchmark_report)
        if state.preflight_result is not None:
            write_stage_output(state, "preflight.json", state.preflight_result)
        if state.execution_result is not None:
            write_stage_output(state, "execution_result.json", state.execution_result)
        if state.repair_ticket is not None:
            write_stage_output(state, "repair_ticket.json", state.repair_ticket)
        if state.generate_stage_output is not None:
            write_stage_output(state, "experiment_output.json", state.generate_stage_output)
        repo_handoff = dict(state.temp_data.get("repo_handoff", {}) or {})
        if repo_handoff:
            write_stage_output(state, "repo_handoff.json", repo_handoff)
        validated_repo_handoff = dict(state.temp_data.get("validated_repo_handoff", {}) or {})
        if validated_repo_handoff:
            write_stage_output(state, "validated_repo_handoff.json", validated_repo_handoff)

    bundle = run_or_resume_stage(
        state,
        "repair_validation",
        input_payload,
        _compute,
        _load,
        _write,
        invalidate_downstream_on_recompute=True,
    )
    apply_repo_validation_bundle(state, bundle)
    save_tracking_artifacts(state)
    return state
