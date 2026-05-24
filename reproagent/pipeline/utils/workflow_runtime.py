"""Runtime and artifact helpers for the reproagent workflow."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from reproagent.pipeline.config import get_workflow_config
from reproagent.pipeline.schemas import (
    PaperBenchReproState,
    PhaseRunSummary,
    RunManifest,
    RunSummary,
    RuntimeProbe,
    StageAttemptRecord,
    StageMetric,
    StageRunSummary,
    UsageSummary,
)
from reproagent.pipeline.utils.artifact_names import CANONICAL_ARTIFACTS
from reproagent.pipeline.utils.artifact_writer import register_existing_file
from reproagent.pipeline.utils.quality_status import refresh_quality_status

STAGE_ORDER = [
    "input_normalization",
    "unit_extraction",
    "reference_acquisition",
    "prepare_quality_gate",
    "topic_profile_synthesis",
    "work_package_planning",
    "package_evidence_grounding",
    "reference_selection",
    "pipeline_plan",
    "global_contract_synthesis",
    "architecture_planning",
    "package_file_planning",
    "canonical_ir_synthesis",
    "local_file_generation",
    "repair_validation",
    "repair_plan",
    "repair_regeneration",
]

PHASE_ORDER = ["prepare", "plan", "generate", "repair"]

PHASE_STAGE_MAP = {
    "prepare": [
        "input_normalization",
        "unit_extraction",
        "reference_acquisition",
        "prepare_quality_gate",
    ],
    "plan": [
        "topic_profile_synthesis",
        "work_package_planning",
        "package_evidence_grounding",
        "reference_selection",
        "pipeline_plan",
        "global_contract_synthesis",
        "architecture_planning",
        "package_file_planning",
    ],
    "generate": [
        "local_file_generation",
    ],
    "repair": [
        "repair_validation",
        "repair_plan",
        "repair_regeneration",
    ],
}

STAGE_OUTPUTS = {
    "input_normalization": ["nodes/prepare/input_normalization.json"],
    "unit_extraction": ["nodes/prepare/unit_extraction.json", "nodes/prepare/units.json"],
    "reference_acquisition": [
        "nodes/prepare/reference_repo_preparation.json",
        "nodes/prepare/reference_repos.json",
        "nodes/prepare/reference_repo_surveys.json",
        "nodes/prepare/resource_manifest.json",
    ],
    "prepare_quality_gate": [
        "nodes/prepare/prepare_quality_gate.json",
        "nodes/prepare/prepare_quality_gate.review.json",
    ],
    "topic_profile_synthesis": [f"nodes/plan/{CANONICAL_ARTIFACTS['topic_profile']}"],
    "work_package_planning": ["nodes/plan/work_packages.json", "nodes/plan/work_package_planning.review.json"],
    "package_evidence_grounding": [
        "nodes/plan/evidence_bundles.json",
        "nodes/plan/evidence_graph.json",
        "nodes/plan/package_evidence_grounding.review.json",
    ],
    "reference_selection": [f"nodes/plan/{CANONICAL_ARTIFACTS['reference_selection']}"],
    "pipeline_plan": [f"nodes/plan/{CANONICAL_ARTIFACTS['pipeline_plan']}"],
    "global_contract_synthesis": [f"nodes/plan/{CANONICAL_ARTIFACTS['global_contract']}"],
    "architecture_planning": [f"nodes/plan/{CANONICAL_ARTIFACTS['architecture']}", "nodes/plan/architecture_planning.review.json"],
    "package_file_planning": [
        f"nodes/plan/{CANONICAL_ARTIFACTS['package_file_planning']}",
        "nodes/plan/package_file_planning.review.json",
    ],
    "canonical_ir_synthesis": [
        f"nodes/plan/{CANONICAL_ARTIFACTS['canonical_ir']}",
        f"nodes/plan/{CANONICAL_ARTIFACTS['canonical_ir_validation']}",
        f"nodes/plan/{CANONICAL_ARTIFACTS['semantic_assertions']}",
        f"nodes/plan/{CANONICAL_ARTIFACTS['semantic_validation_report']}",
    ],
    "local_file_generation": [
        "nodes/generate/experiment_output.json",
        "nodes/generate/repo_handoff.json",
        "nodes/generate/repo_plan.json",
        "nodes/generate/project_plan.json",
        "nodes/generate/generation_manifest.json",
        "nodes/generate/iteration_checkpoint.json",
        "nodes/generate/generation_checkpoints.json",
        "nodes/generate/project_manifest.json",
        "nodes/generate/execution_result.json",
        "nodes/generate/execution_history.json",
        "nodes/generate/last_attempt.json",
    ],
    "repair_validation": [
        "nodes/repair/validation_bundle.json",
        "nodes/repair/repair_validation_bundle.json",
        "nodes/repair/repair_review.json",
        "nodes/repair/preflight.json",
        "nodes/repair/execution_result.json",
        "nodes/repair/repair_ticket.json",
        "nodes/repair/runtime_probe.json",
        "nodes/repair/validation_report.json",
        "nodes/repair/benchmark_report.json",
        "nodes/repair/repo_handoff.json",
        "nodes/repair/experiment_output.json",
        "nodes/repair/validated_repo_handoff.json",
    ],
    "repair_plan": [
        "nodes/repair/repair_review.json",
        "nodes/repair/requirement_anchor.json",
        "nodes/repair/repair_eval_report.json",
        "nodes/repair/repair_findings.json",
        "nodes/repair/repair_plan_context.json",
        "nodes/repair/repair_plan_draft.json",
        "nodes/repair/repair_plan_review.json",
        "nodes/repair/repair_plan.json",
    ],
    "repair_regeneration": [
        "nodes/repair/repair_review.json",
        "nodes/repair/repair_log.json",
        "nodes/repair/validation_report.json",
        "nodes/repair/benchmark_report.json",
        "nodes/repair/preflight.json",
        "nodes/repair/execution_result.json",
        "nodes/repair/repair_ticket.json",
        "nodes/repair/iteration_checkpoint.json",
        "nodes/repair/generation_checkpoints.json",
        "nodes/repair/last_attempt.json",
        "nodes/repair/current_prompt.md",
        "nodes/repair/repair_round_history.jsonl",
        "nodes/repair/repair_regeneration_result.json",
    ],
}

SUPPLEMENTARY_ARTIFACTS = {
    "prepare_inputs": [
        "nodes/prepare/input.json",
        "nodes/prepare/upstream_intent.json",
        "nodes/prepare/paper_chunks.json",
        "nodes/prepare/resource_manifest.json",
    ],
    "plan_requirements": [f"nodes/plan/{CANONICAL_ARTIFACTS['boundary_requirements']}"],
    "plan_contract": [
        f"nodes/plan/{CANONICAL_ARTIFACTS['reference_selection']}",
        f"nodes/plan/{CANONICAL_ARTIFACTS['pipeline_plan']}",
    ],
    "run_tracking": [
        "stage_status.json",
        "stage_attempts.json",
        "quality_status.json",
        "run_manifest.json",
        "usage_summary.json",
        "run_summary.json",
        "latest_state.json",
    ],
}


def _estimate_tokens(payload: Any, *, json_default: Callable[[Any], Any]) -> int:
    """Cheap token estimate for manifest accounting."""
    text = json.dumps(payload, ensure_ascii=False, default=json_default) if not isinstance(payload, str) else payload
    return max(1, len(text) // 4) if text else 0


def _payload_hash(payload: Any, *, json_default: Callable[[Any], Any]) -> str:
    """Stable payload hash for resume checks."""
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=json_default)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _pipeline_signature(module_file: str | Path) -> str:
    """Hash the reproagent source tree to invalidate stale stage artifacts."""
    module_root = Path(module_file).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(module_root.rglob("*.py")):
        try:
            digest.update(str(path.relative_to(module_root)).encode("utf-8"))
            digest.update(path.read_bytes())
        except Exception:
            continue
    return digest.hexdigest()


def _stage_status_path(
    state: PaperBenchReproState,
    *,
    get_output_dir: Callable[[PaperBenchReproState], Path],
) -> Path:
    return get_output_dir(state) / "stage_status.json"


def _load_stage_status(
    state: PaperBenchReproState,
    *,
    get_output_dir: Callable[[PaperBenchReproState], Path],
) -> dict[str, Any]:
    cached = state.temp_data.get("stage_status")
    if isinstance(cached, dict):
        return cached
    path = _stage_status_path(state, get_output_dir=get_output_dir)
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                state.temp_data["stage_status"] = payload
                return payload
        except Exception:
            pass
    payload: dict[str, Any] = {}
    state.temp_data["stage_status"] = payload
    return payload


def _write_stage_status(
    state: PaperBenchReproState,
    stage_status: dict[str, Any],
    *,
    get_output_dir: Callable[[PaperBenchReproState], Path],
    json_default: Callable[[Any], Any],
) -> None:
    state.temp_data["stage_status"] = stage_status
    path = _stage_status_path(state, get_output_dir=get_output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(stage_status, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )
    register_existing_file(
        path,
        run_dir=get_output_dir(state),
        logical_name="stage_status",
        kind="state",
        authority="source_of_truth",
    )


def _stage_attempts_path(
    state: PaperBenchReproState,
    *,
    get_output_dir: Callable[[PaperBenchReproState], Path],
) -> Path:
    return get_output_dir(state) / "stage_attempts.json"


def _stage_attempt_id(state: PaperBenchReproState, stage_name: str) -> str:
    counters = state.temp_data.setdefault("stage_attempt_counters", {})
    next_count = int(counters.get(stage_name, 0) or 0) + 1
    counters[stage_name] = next_count
    return f"{stage_name}:{next_count:03d}"


def _hydrate_stage_attempt_counters(
    state: PaperBenchReproState,
    *,
    get_output_dir: Callable[[PaperBenchReproState], Path],
) -> None:
    counters = state.temp_data.setdefault("stage_attempt_counters", {})
    if counters:
        return
    path = _stage_attempts_path(state, get_output_dir=get_output_dir)
    if not path.exists():
        return
    try:
        attempts = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return
    if not isinstance(attempts, list):
        return
    state.temp_data["stage_attempts"] = attempts
    for item in attempts:
        if not isinstance(item, dict):
            continue
        stage_name = str(item.get("stage_name", "") or "").strip()
        attempt_id = str(item.get("attempt_id", "") or "").strip()
        if not stage_name or ":" not in attempt_id:
            continue
        try:
            suffix = int(attempt_id.rsplit(":", 1)[1])
        except ValueError:
            continue
        counters[stage_name] = max(int(counters.get(stage_name, 0) or 0), suffix)


def _append_stage_attempt(
    state: PaperBenchReproState,
    record: StageAttemptRecord,
    *,
    get_output_dir: Callable[[PaperBenchReproState], Path],
    json_default: Callable[[Any], Any],
) -> None:
    attempts = state.temp_data.setdefault("stage_attempts", [])
    attempts.append(record.model_dump(mode="json"))
    path = _stage_attempts_path(state, get_output_dir=get_output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(attempts, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )
    register_existing_file(
        path,
        run_dir=get_output_dir(state),
        logical_name="stage_attempts",
        kind="state",
        authority="source_of_truth",
    )


def _mark_stage_started(
    state: PaperBenchReproState,
    stage_name: str,
    *,
    load_stage_status: Callable[[PaperBenchReproState], dict[str, Any]],
    write_stage_status: Callable[[PaperBenchReproState, dict[str, Any]], None],
    save_tracking_artifacts: Callable[[PaperBenchReproState], None],
    get_output_dir: Callable[[PaperBenchReproState], Path],
    json_default: Callable[[Any], Any],
) -> None:
    stage_status = load_stage_status(state)
    started_at = datetime.now().isoformat()
    attempt_id = _stage_attempt_id(state, stage_name)
    state.temp_data.setdefault("active_stage_attempts", {})[stage_name] = attempt_id
    stage_status[stage_name] = {
        "status": "running",
        "attempt_id": attempt_id,
        "started_at": started_at,
        "output_paths": STAGE_OUTPUTS.get(stage_name, []),
    }
    write_stage_status(state, stage_status)
    _append_stage_attempt(
        state,
        StageAttemptRecord(
            attempt_id=attempt_id,
            stage_name=stage_name,
            status="running",
            started_at=started_at,
            output_paths=STAGE_OUTPUTS.get(stage_name, []),
        ),
        get_output_dir=get_output_dir,
        json_default=json_default,
    )
    save_tracking_artifacts(state)


def _mark_stage_completed(
    state: PaperBenchReproState,
    stage_name: str,
    *,
    load_stage_status: Callable[[PaperBenchReproState], dict[str, Any]],
    write_stage_status: Callable[[PaperBenchReproState, dict[str, Any]], None],
    save_tracking_artifacts: Callable[[PaperBenchReproState], None],
    get_output_dir: Callable[[PaperBenchReproState], Path],
    json_default: Callable[[Any], Any],
    extra: dict[str, Any] | None = None,
) -> None:
    stage_status = load_stage_status(state)
    existing = stage_status.get(stage_name, {})
    completed_at = datetime.now().isoformat()
    attempt_id = str(existing.get("attempt_id") or state.temp_data.get("active_stage_attempts", {}).get(stage_name) or _stage_attempt_id(state, stage_name))
    payload = {
        **existing,
        "status": "completed",
        "attempt_id": attempt_id,
        "completed_at": completed_at,
        "output_paths": STAGE_OUTPUTS.get(stage_name, []),
    }
    if extra:
        payload.update(extra)
    stage_status[stage_name] = payload
    write_stage_status(state, stage_status)
    _append_stage_attempt(
        state,
        StageAttemptRecord(
            attempt_id=attempt_id,
            stage_name=stage_name,
            status="completed",
            started_at=str(existing.get("started_at", "") or ""),
            completed_at=completed_at,
            output_paths=STAGE_OUTPUTS.get(stage_name, []),
            input_hash=str(payload.get("input_hash", "") or ""),
            pipeline_signature=str(payload.get("pipeline_signature", "") or ""),
        ),
        get_output_dir=get_output_dir,
        json_default=json_default,
    )
    save_tracking_artifacts(state)


def _mark_stage_failed(
    state: PaperBenchReproState,
    stage_name: str,
    exc: Exception,
    *,
    load_stage_status: Callable[[PaperBenchReproState], dict[str, Any]],
    write_stage_status: Callable[[PaperBenchReproState, dict[str, Any]], None],
    save_tracking_artifacts: Callable[[PaperBenchReproState], None],
    get_output_dir: Callable[[PaperBenchReproState], Path],
    json_default: Callable[[Any], Any],
) -> None:
    stage_status = load_stage_status(state)
    existing = dict(stage_status.get(stage_name, {}) or {})
    failed_at = datetime.now().isoformat()
    attempt_id = str(existing.get("attempt_id") or state.temp_data.get("active_stage_attempts", {}).get(stage_name) or _stage_attempt_id(state, stage_name))
    stage_status[stage_name] = {
        **existing,
        "status": "failed",
        "attempt_id": attempt_id,
        "failed_at": failed_at,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "output_paths": STAGE_OUTPUTS.get(stage_name, []),
    }
    write_stage_status(state, stage_status)
    _append_stage_attempt(
        state,
        StageAttemptRecord(
            attempt_id=attempt_id,
            stage_name=stage_name,
            status="failed",
            started_at=str(existing.get("started_at", "") or ""),
            failed_at=failed_at,
            error_type=type(exc).__name__,
            error_message=str(exc),
            output_paths=STAGE_OUTPUTS.get(stage_name, []),
        ),
        get_output_dir=get_output_dir,
        json_default=json_default,
    )
    save_tracking_artifacts(state)


def _record_stage_recovery(
    state: PaperBenchReproState,
    stage_name: str,
    exc: Exception,
    *,
    action: str,
    load_stage_status: Callable[[PaperBenchReproState], dict[str, Any]],
    write_stage_status: Callable[[PaperBenchReproState, dict[str, Any]], None],
    save_tracking_artifacts: Callable[[PaperBenchReproState], None],
    get_output_dir: Callable[[PaperBenchReproState], Path],
    json_default: Callable[[Any], Any],
) -> None:
    recovery_ticket = {
        "ticket_type": "stage_recovery",
        "stage_name": stage_name,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "action": action,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    state.temp_data.setdefault("recovery_tickets", []).append(recovery_ticket)
    stage_status = load_stage_status(state)
    existing = dict(stage_status.get(stage_name, {}) or {})
    attempt_id = str(existing.get("attempt_id") or state.temp_data.get("active_stage_attempts", {}).get(stage_name) or _stage_attempt_id(state, stage_name))
    stage_status[stage_name] = {
        **existing,
        "status": "recovered",
        "attempt_id": attempt_id,
        "recovered_at": recovery_ticket["created_at"],
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "recovery_action": action,
        "output_paths": STAGE_OUTPUTS.get(stage_name, []),
    }
    write_stage_status(state, stage_status)
    _append_stage_attempt(
        state,
        StageAttemptRecord(
            attempt_id=attempt_id,
            stage_name=stage_name,
            status="recovered",
            started_at=str(existing.get("started_at", "") or ""),
            recovered_at=recovery_ticket["created_at"],
            recovery_action=action,
            error_type=type(exc).__name__,
            error_message=str(exc),
            output_paths=STAGE_OUTPUTS.get(stage_name, []),
        ),
        get_output_dir=get_output_dir,
        json_default=json_default,
    )
    save_tracking_artifacts(state)


def _invalidate_downstream_stages(
    state: PaperBenchReproState,
    stage_name: str,
    reason: str,
    *,
    load_stage_status: Callable[[PaperBenchReproState], dict[str, Any]],
    write_stage_status: Callable[[PaperBenchReproState, dict[str, Any]], None],
    save_tracking_artifacts: Callable[[PaperBenchReproState], None],
) -> None:
    stage_status = load_stage_status(state)
    if stage_name not in STAGE_ORDER:
        return
    start_index = STAGE_ORDER.index(stage_name) + 1
    changed = False
    for downstream_stage in STAGE_ORDER[start_index:]:
        entry = stage_status.get(downstream_stage)
        if not isinstance(entry, dict) or not entry:
            continue
        stage_status[downstream_stage] = {
            "status": "invalidated",
            "invalidated_at": datetime.now().isoformat(),
            "reason": reason,
            "output_paths": STAGE_OUTPUTS.get(downstream_stage, []),
        }
        changed = True
    if changed:
        write_stage_status(state, stage_status)
        save_tracking_artifacts(state)


def _record_stage_metric(
    state: PaperBenchReproState,
    stage_name: str,
    elapsed_seconds: float,
    input_payload: Any,
    output_payload: Any,
    *,
    json_default: Callable[[Any], Any],
    notes: list[str] | None = None,
) -> None:
    def _aggregate_execution_history_usage(items: list[dict[str, Any]]) -> dict[str, Any]:
        usage_items = [
            dict(item.get("context_usage", {}) or {})
            for item in items
            if isinstance(item, dict) and isinstance(item.get("context_usage"), dict)
        ]
        usage_sources: list[str] = []
        session_ids: list[str] = []
        for item in usage_items:
            usage_source = str(item.get("usage_source", "") or "")
            if usage_source and usage_source not in usage_sources:
                usage_sources.append(usage_source)
            session_id = str(item.get("session_id", "") or "")
            if session_id and session_id not in session_ids:
                session_ids.append(session_id)
        return {
            "calls": len(usage_items),
            "calls_with_usage": sum(1 for item in usage_items if bool(item.get("usage_found"))),
            "input_tokens": sum(int(item.get("actual_input_tokens", 0) or 0) for item in usage_items),
            "output_tokens": sum(int(item.get("actual_output_tokens", 0) or 0) for item in usage_items),
            "total_tokens": sum(int(item.get("actual_total_tokens", 0) or 0) for item in usage_items),
            "session_ids": session_ids,
            "usage_sources": usage_sources,
        }

    usage_summary = {}
    if isinstance(output_payload, dict):
        usage_summary = dict(output_payload.get("agent_usage_summary", {}) or {})
        if not usage_summary and isinstance(output_payload.get("execution_history"), list):
            usage_summary = _aggregate_execution_history_usage(list(output_payload.get("execution_history") or []))

    stage_metrics = state.temp_data.setdefault("stage_metrics", {})
    stage_metrics[stage_name] = StageMetric(
        stage_name=stage_name,
        elapsed_seconds=round(elapsed_seconds, 4),
        estimated_input_tokens=_estimate_tokens(input_payload, json_default=json_default),
        estimated_output_tokens=_estimate_tokens(output_payload, json_default=json_default),
        actual_input_tokens=int(usage_summary.get("input_tokens", 0) or 0),
        actual_output_tokens=int(usage_summary.get("output_tokens", 0) or 0),
        actual_total_tokens=int(usage_summary.get("total_tokens", 0) or 0),
        agent_calls=int(usage_summary.get("calls", 0) or 0),
        agent_calls_with_usage=int(usage_summary.get("calls_with_usage", 0) or 0),
        usage_sources=list(dict.fromkeys(str(item) for item in list(usage_summary.get("usage_sources", []) or []) if str(item).strip())),
        notes=list(notes or []),
    ).model_dump(mode="json")


def write_review_artifact(
    state: PaperBenchReproState,
    stage_name: str,
    payload: dict[str, Any],
    *,
    get_output_dir: Callable[[PaperBenchReproState], Path],
    json_default: Callable[[Any], Any],
) -> None:
    stage_reviews = state.temp_data.setdefault("stage_reviews", {})
    stage_reviews[stage_name] = payload
    output_dir = get_output_dir(state)
    output_dir.mkdir(parents=True, exist_ok=True)
    node_name = "plan" if stage_name in PHASE_STAGE_MAP.get("plan", []) else stage_name
    for phase_name, stage_names in PHASE_STAGE_MAP.items():
        if stage_name in stage_names:
            node_name = phase_name
            break
    path = output_dir / "nodes" / node_name / f"{stage_name}.review.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )
    register_existing_file(
        path,
        run_dir=output_dir,
        logical_name=f"{stage_name}.review",
        kind="report",
        stage=node_name,
        node=node_name,
        authority="derived",
    )


def _artifact_path_map(
    state: PaperBenchReproState,
    *,
    get_output_dir: Callable[[PaperBenchReproState], Path],
) -> dict[str, Path]:
    output_dir = get_output_dir(state)
    def _prefer_existing(*candidates: Path) -> Path:
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    return {
        "input_snapshot": _prefer_existing(output_dir / "nodes" / "prepare" / "input.json", output_dir / "input.json"),
        "upstream_intent": _prefer_existing(output_dir / "nodes" / "prepare" / "upstream_intent.json", output_dir / "upstream_intent.json"),
        "paper_chunks": _prefer_existing(output_dir / "nodes" / "prepare" / "paper_chunks.json", output_dir / "paper_chunks.json"),
        "resource_manifest": _prefer_existing(output_dir / "nodes" / "prepare" / "resource_manifest.json", output_dir / "resource_manifest.json"),
        "input_normalization": _prefer_existing(output_dir / "nodes" / "prepare" / "input_normalization.json", output_dir / "input_normalization.json"),
        "unit_extraction": _prefer_existing(output_dir / "nodes" / "prepare" / "unit_extraction.json", output_dir / "unit_extraction.json"),
        "units": _prefer_existing(output_dir / "nodes" / "prepare" / "units.json", output_dir / "units.json"),
        "boundary_requirements": _prefer_existing(output_dir / "nodes" / "plan" / CANONICAL_ARTIFACTS["boundary_requirements"], output_dir / CANONICAL_ARTIFACTS["boundary_requirements"]),
        "reference_repo_preparation": _prefer_existing(output_dir / "nodes" / "prepare" / "reference_repo_preparation.json", output_dir / "reference_repo_preparation.json"),
        "reference_repos": _prefer_existing(output_dir / "nodes" / "prepare" / "reference_repos.json", output_dir / "reference_repos.json"),
        "reference_repo_surveys": _prefer_existing(output_dir / "nodes" / "prepare" / "reference_repo_surveys.json", output_dir / "reference_repo_surveys.json"),
        "prepare_quality_gate": _prefer_existing(output_dir / "nodes" / "prepare" / "prepare_quality_gate.json", output_dir / "prepare_quality_gate.json"),
        "prepare_quality_gate_review": _prefer_existing(output_dir / "nodes" / "prepare" / "prepare_quality_gate.review.json", output_dir / "prepare_quality_gate.review.json"),
        "reference_selection": _prefer_existing(output_dir / "nodes" / "plan" / CANONICAL_ARTIFACTS["reference_selection"], output_dir / CANONICAL_ARTIFACTS["reference_selection"]),
        "pipeline_plan": _prefer_existing(output_dir / "nodes" / "plan" / CANONICAL_ARTIFACTS["pipeline_plan"], output_dir / CANONICAL_ARTIFACTS["pipeline_plan"]),
        "topic_profile": _prefer_existing(output_dir / "nodes" / "plan" / CANONICAL_ARTIFACTS["topic_profile"], output_dir / CANONICAL_ARTIFACTS["topic_profile"]),
        "work_packages": _prefer_existing(output_dir / "nodes" / "plan" / "work_packages.json", output_dir / "work_packages.json"),
        "work_package_review": _prefer_existing(output_dir / "nodes" / "plan" / "work_package_planning.review.json", output_dir / "work_package_planning.review.json"),
        "package_evidence_grounding_review": _prefer_existing(output_dir / "nodes" / "plan" / "package_evidence_grounding.review.json", output_dir / "package_evidence_grounding.review.json"),
        "evidence_bundles": _prefer_existing(output_dir / "nodes" / "plan" / "evidence_bundles.json", output_dir / "evidence_bundles.json"),
        "evidence_graph": _prefer_existing(output_dir / "nodes" / "plan" / "evidence_graph.json", output_dir / "evidence_graph.json"),
        "global_contract": _prefer_existing(output_dir / "nodes" / "plan" / CANONICAL_ARTIFACTS["global_contract"], output_dir / CANONICAL_ARTIFACTS["global_contract"]),
        "architecture": _prefer_existing(output_dir / "nodes" / "plan" / CANONICAL_ARTIFACTS["architecture"], output_dir / CANONICAL_ARTIFACTS["architecture"]),
        "architecture_review": _prefer_existing(output_dir / "nodes" / "plan" / "architecture_planning.review.json", output_dir / "architecture_planning.review.json"),
        "package_file_planning": _prefer_existing(output_dir / "nodes" / "plan" / CANONICAL_ARTIFACTS["package_file_planning"], output_dir / CANONICAL_ARTIFACTS["package_file_planning"]),
        "canonical_ir": _prefer_existing(output_dir / "nodes" / "plan" / CANONICAL_ARTIFACTS["canonical_ir"], output_dir / CANONICAL_ARTIFACTS["canonical_ir"]),
        "canonical_ir_validation": _prefer_existing(output_dir / "nodes" / "plan" / CANONICAL_ARTIFACTS["canonical_ir_validation"], output_dir / CANONICAL_ARTIFACTS["canonical_ir_validation"]),
        "semantic_assertions": _prefer_existing(output_dir / "nodes" / "plan" / CANONICAL_ARTIFACTS["semantic_assertions"], output_dir / CANONICAL_ARTIFACTS["semantic_assertions"]),
        "semantic_validation_report": _prefer_existing(output_dir / "nodes" / "plan" / CANONICAL_ARTIFACTS["semantic_validation_report"], output_dir / CANONICAL_ARTIFACTS["semantic_validation_report"]),
        "package_file_review": _prefer_existing(output_dir / "nodes" / "plan" / "package_file_planning.review.json", output_dir / "package_file_planning.review.json"),
        "repo_plan": _prefer_existing(
            output_dir / "nodes" / "generate" / "repo_plan.json",
            output_dir / "repo_plan.json",
        ),
        "generation_manifest": _prefer_existing(
            output_dir / "nodes" / "generate" / "generation_manifest.json",
            output_dir / "generation_manifest.json",
        ),
        "project_plan": _prefer_existing(
            output_dir / "nodes" / "generate" / "project_plan.json",
            output_dir / "project_plan.json",
        ),
        "experiment_output": _prefer_existing(
            output_dir / "nodes" / "repair" / "experiment_output.json",
            output_dir / "nodes" / "generate" / "experiment_output.json",
            output_dir / "experiment_output.json",
        ),
        "file_provenance": _prefer_existing(
            output_dir / "nodes" / "repair" / "file_provenance.json",
            output_dir / "nodes" / "generate" / "file_provenance.json",
            output_dir / "file_provenance.json",
        ),
        "repo_handoff": _prefer_existing(
            output_dir / "nodes" / "repair" / "repo_handoff.json",
            output_dir / "nodes" / "generate" / "repo_handoff.json",
            output_dir / "repo_handoff.json",
        ),
        "iteration_checkpoint": _prefer_existing(
            output_dir / "nodes" / "repair" / "iteration_checkpoint.json",
            output_dir / "nodes" / "generate" / "iteration_checkpoint.json",
            output_dir / "iteration_checkpoint.json",
        ),
        "generation_checkpoints": _prefer_existing(
            output_dir / "nodes" / "repair" / "generation_checkpoints.json",
            output_dir / "nodes" / "generate" / "generation_checkpoints.json",
            output_dir / "generation_checkpoints.json",
        ),
        "node_generate_project_manifest": output_dir / "nodes" / "generate" / "project_manifest.json",
        "node_generate_execution_result": output_dir / "nodes" / "generate" / "execution_result.json",
        "node_generate_execution_history": output_dir / "nodes" / "generate" / "execution_history.json",
        "node_generate_preflight": output_dir / "nodes" / "generate" / "preflight.json",
        "validation_bundle": _prefer_existing(
            output_dir / "nodes" / "repair" / "validation_bundle.json",
            output_dir / "validation_bundle.json",
        ),
        "repair_validation_bundle": _prefer_existing(
            output_dir / "nodes" / "repair" / "repair_validation_bundle.json",
            output_dir / "repair_validation_bundle.json",
        ),
        "preflight": _prefer_existing(output_dir / "nodes" / "repair" / "preflight.json", output_dir / "preflight.json"),
        "execution_result": _prefer_existing(
            output_dir / "nodes" / "repair" / "execution_result.json",
            output_dir / "execution_result.json",
        ),
        "repair_ticket": _prefer_existing(
            output_dir / "nodes" / "repair" / "repair_ticket.json",
            output_dir / "repair_ticket.json",
        ),
        "repair_review": _prefer_existing(
            output_dir / "nodes" / "repair" / "repair_review.json",
            output_dir / "repair_review.json",
        ),
        "repair_findings": _prefer_existing(
            output_dir / "nodes" / "repair" / "repair_findings.json",
            output_dir / "repair_findings.json",
        ),
        "repair_plan_context": _prefer_existing(
            output_dir / "nodes" / "repair" / "repair_plan_context.json",
            output_dir / "repair_plan_context.json",
        ),
        "repair_plan_draft": _prefer_existing(
            output_dir / "nodes" / "repair" / "repair_plan_draft.json",
            output_dir / "repair_plan_draft.json",
        ),
        "repair_plan": _prefer_existing(
            output_dir / "nodes" / "repair" / "repair_plan.json",
            output_dir / "repair_plan.json",
        ),
        "repair_plan_review": _prefer_existing(
            output_dir / "nodes" / "repair" / "repair_plan_review.json",
            output_dir / "repair_plan_review.json",
        ),
        "validated_repo_handoff": _prefer_existing(
            output_dir / "nodes" / "repair" / "validated_repo_handoff.json",
            output_dir / "validated_repo_handoff.json",
        ),
        "runtime_probe": _prefer_existing(
            output_dir / "nodes" / "repair" / "runtime_probe.json",
            output_dir / "runtime_probe.json",
        ),
        "validation_report": _prefer_existing(
            output_dir / "nodes" / "repair" / "validation_report.json",
            output_dir / "validation_report.json",
        ),
        "benchmark_report": _prefer_existing(
            output_dir / "nodes" / "repair" / "benchmark_report.json",
            output_dir / "benchmark_report.json",
        ),
        "repair_regeneration_result": _prefer_existing(
            output_dir / "nodes" / "repair" / "repair_regeneration_result.json",
            output_dir / "repair_regeneration_result.json",
        ),
        "recovery_tickets": _prefer_existing(
            output_dir / "nodes" / "repair" / "recovery_tickets.json",
            output_dir / "nodes" / "generate" / "recovery_tickets.json",
            output_dir / "nodes" / "plan" / "recovery_tickets.json",
            output_dir / "nodes" / "prepare" / "recovery_tickets.json",
            output_dir / "recovery_tickets.json",
        ),
        "repair_log": _prefer_existing(output_dir / "nodes" / "repair" / "repair_log.json", output_dir / "repair_log.json"),
        "memory": output_dir / "memory.md",
        "memory_events": output_dir / "memory_events.jsonl",
        "stage_status": output_dir / "stage_status.json",
        "stage_attempts": output_dir / "stage_attempts.json",
        "quality_status": output_dir / "quality_status.json",
        "run_manifest": output_dir / "run_manifest.json",
        "usage_summary": output_dir / "usage_summary.json",
        "run_summary": output_dir / "run_summary.json",
        "latest_state": output_dir / "latest_state.json",
    }


def _artifact_index(
    state: PaperBenchReproState,
    *,
    get_output_dir: Callable[[PaperBenchReproState], Path],
) -> dict[str, str]:
    artifacts = _artifact_path_map(state, get_output_dir=get_output_dir)
    return {key: str(path.resolve()) for key, path in artifacts.items() if path.exists()}


def _manifest_stage_status_summary(
    state: PaperBenchReproState,
    *,
    load_stage_status: Callable[[PaperBenchReproState], dict[str, Any]],
) -> list[StageRunSummary]:
    stage_status = load_stage_status(state)
    summaries: list[StageRunSummary] = []
    for stage_name in STAGE_ORDER:
        entry = stage_status.get(stage_name, {})
        summaries.append(
            StageRunSummary(
                stage_name=stage_name,
                status=str(entry.get("status", "pending")),
                resume_source=str(entry.get("resume_source", "")),
                skipped=bool(entry.get("skipped", False)),
                error_type=str(entry.get("error_type", "")),
                output_paths=list(entry.get("output_paths", STAGE_OUTPUTS.get(stage_name, []))),
                notes=[str(entry.get("reason", "")).strip()] if str(entry.get("reason", "")).strip() else [],
            )
        )
    return summaries


def _phase_status(
    phase_name: str,
    stage_status: dict[str, Any],
) -> str:
    stage_names = PHASE_STAGE_MAP.get(phase_name, [])
    entries = [dict(stage_status.get(stage_name, {}) or {}) for stage_name in stage_names]
    statuses = [str(entry.get("status", "pending")) for entry in entries]
    if any(status == "failed" for status in statuses):
        return "failed"
    if any(status == "running" for status in statuses):
        return "running"
    completed_like = {"completed"}
    if phase_name == "repair":
        if (
            statuses
            and statuses[0] == "completed"
            and all(status in {"pending", "invalidated"} for status in statuses[1:])
        ):
            return "completed"
        if (
            len(statuses) >= 2
            and statuses[0] == "completed"
            and statuses[1] == "completed"
            and all(status in {"pending", "invalidated"} for status in statuses[2:])
        ):
            return "completed"
    if statuses and all(status in completed_like for status in statuses):
        return "completed"
    if any(status == "invalidated" for status in statuses):
        return "invalidated"
    if any(status == "completed" for status in statuses):
        return "partial"
    return "pending"


def _manifest_phase_status_summary(
    state: PaperBenchReproState,
    *,
    load_stage_status: Callable[[PaperBenchReproState], dict[str, Any]],
) -> list[PhaseRunSummary]:
    stage_status = load_stage_status(state)
    summaries: list[PhaseRunSummary] = []
    for phase_name in PHASE_ORDER:
        stage_names = list(PHASE_STAGE_MAP.get(phase_name, []))
        output_paths: list[str] = []
        notes: list[str] = []
        seen_paths: set[str] = set()
        for stage_name in stage_names:
            entry = dict(stage_status.get(stage_name, {}) or {})
            for path in list(entry.get("output_paths", STAGE_OUTPUTS.get(stage_name, []))):
                if path not in seen_paths:
                    seen_paths.add(path)
                    output_paths.append(path)
            reason = str(entry.get("reason", "")).strip()
            if reason and reason not in notes:
                notes.append(reason)
        summaries.append(
            PhaseRunSummary(
                phase_name=phase_name,
                status=_phase_status(phase_name, stage_status),
                stage_names=stage_names,
                output_paths=output_paths,
                notes=notes,
            )
        )
    return summaries


def _manifest_run_summary(
    state: PaperBenchReproState,
    *,
    load_stage_status: Callable[[PaperBenchReproState], dict[str, Any]],
) -> RunSummary:
    stage_status = load_stage_status(state)
    phase_status_summary = _manifest_phase_status_summary(state, load_stage_status=load_stage_status)
    topic_profile = state.topic_profile
    validation_report = state.validation_report
    return RunSummary(
        completed_stage_count=sum(1 for item in stage_status.values() if item.get("status") == "completed"),
        failed_stage_count=sum(1 for item in stage_status.values() if item.get("status") == "failed"),
        invalidated_stage_count=sum(1 for item in stage_status.values() if item.get("status") == "invalidated"),
        resumed_stage_count=sum(1 for item in stage_status.values() if item.get("resume_source")),
        completed_phase_count=sum(1 for item in phase_status_summary if item.status == "completed"),
        failed_phase_count=sum(1 for item in phase_status_summary if item.status == "failed"),
        primary_topic=topic_profile.primary_topic if topic_profile else "",
        coverage_policy=topic_profile.coverage_policy if topic_profile else "",
        experiment_traits=list(topic_profile.experiment_traits) if topic_profile else [],
        work_package_count=len(state.work_package_planning.work_packages) if state.work_package_planning else 0,
        grounded_work_package_count=sum(1 for item in state.evidence_bundles if item.grounding_status == "grounded"),
        result_target_count=len(state.global_contract.result_targets) if state.global_contract else 0,
        validation_passed=validation_report.passed if validation_report else False,
        validation_static_contract_status=validation_report.static_contract_status if validation_report else "unknown",
        validation_smoke_status=validation_report.smoke_status if validation_report else "skipped",
        validation_overall_status=validation_report.overall_status if validation_report else "unknown",
        validation_quality_level=validation_report.quality_level if validation_report else "scaffold_only",
        validation_failure_categories=list(validation_report.failure_categories) if validation_report else [],
        validation_blocked_reasons=list(validation_report.blocked_reasons) if validation_report else [],
    )


def _review_artifact_path(output_dir: Path, stage_name: str) -> Path:
    node_name = ""
    for phase_name, stage_names in PHASE_STAGE_MAP.items():
        if stage_name in stage_names:
            node_name = phase_name
            break
    if not node_name:
        node_name = "debug"
    return output_dir / "nodes" / node_name / f"{stage_name}.review.json"


def _manifest_stage_review_artifacts(output_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for stage_name in (
        "work_package_planning",
        "package_evidence_grounding",
        "architecture_planning",
        "package_file_planning",
    ):
        path = _review_artifact_path(output_dir, stage_name)
        if not path.exists():
            legacy_path = output_dir / f"{stage_name}.review.json"
            if legacy_path.exists():
                path = legacy_path
        if path.exists():
            rows.append({"stage_name": stage_name, "path": str(path.resolve())})
    return rows


def save_tracking_artifacts(
    state: PaperBenchReproState,
    *,
    get_output_dir: Callable[[PaperBenchReproState], Path],
    json_default: Callable[[Any], Any],
    load_stage_status: Callable[[PaperBenchReproState], dict[str, Any]],
    structured_stage_backend_label: Callable[[], str],
) -> None:
    if not state.run_id:
        return
    output_dir = get_output_dir(state)
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_metrics_payload = state.temp_data.get("stage_metrics", {})
    ordered_metrics = [
        StageMetric.model_validate(stage_metrics_payload[name])
        for name in STAGE_ORDER
        if name in stage_metrics_payload
    ]
    if state.usage_summary is None:
        state.usage_summary = UsageSummary()
    state.usage_summary = UsageSummary(
        estimated_total_tokens=sum(item.estimated_input_tokens + item.estimated_output_tokens for item in ordered_metrics),
        actual_input_tokens=sum(item.actual_input_tokens for item in ordered_metrics),
        actual_output_tokens=sum(item.actual_output_tokens for item in ordered_metrics),
        actual_total_tokens=sum(item.actual_total_tokens for item in ordered_metrics),
        agent_calls=sum(item.agent_calls for item in ordered_metrics),
        agent_calls_with_usage=sum(item.agent_calls_with_usage for item in ordered_metrics),
        wall_clock_seconds=round(time.perf_counter() - state.temp_data.get("workflow_started_at", time.perf_counter()), 4),
        stage_breakdown=ordered_metrics,
    )
    (output_dir / "usage_summary.json").write_text(
        json.dumps(state.usage_summary.model_dump(mode="json"), indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )
    quality_status = refresh_quality_status(state)
    (output_dir / "quality_status.json").write_text(
        json.dumps(quality_status.model_dump(mode="json"), indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )
    register_existing_file(
        output_dir / "quality_status.json",
        run_dir=output_dir,
        logical_name="quality_status",
        kind="state",
        authority="source_of_truth",
    )
    state.run_manifest = RunManifest(
        run_id=state.run_id,
        target=state.input.target,
        run_dir=str(output_dir.resolve()),
        stages=[item.stage_name for item in ordered_metrics],
        stage_order=list(STAGE_ORDER),
        phases=[item.phase_name for item in _manifest_phase_status_summary(state, load_stage_status=load_stage_status)],
        phase_order=list(PHASE_ORDER),
        runtime_overrides={
            "max_iterations": state.input.max_iterations,
            "resume_from_run_id": state.input.resume_from_run_id,
            "fork_from_run_id": state.input.fork_from_run_id,
            "stage_review_repair_budget": state.input.stage_review_repair_budget,
            "structured_stage_backend": structured_stage_backend_label(),
            "terminal_outcome": state.terminal_outcome,
            "terminal_outcome_reason": state.terminal_outcome_reason,
            "quality_status": quality_status.model_dump(mode="json"),
        },
        artifact_index=_artifact_index(state, get_output_dir=get_output_dir),
        stage_review_artifacts=_manifest_stage_review_artifacts(output_dir),
        stage_status_summary=_manifest_stage_status_summary(state, load_stage_status=load_stage_status),
        phase_status_summary=_manifest_phase_status_summary(state, load_stage_status=load_stage_status),
        run_summary=_manifest_run_summary(state, load_stage_status=load_stage_status),
    )
    (output_dir / "run_manifest.json").write_text(
        json.dumps(state.run_manifest.model_dump(mode="json"), indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )


def persist_generation_checkpoints(
    state: PaperBenchReproState,
    *,
    get_output_dir: Callable[[PaperBenchReproState], Path],
    json_default: Callable[[Any], Any],
) -> None:
    if not state.run_id:
        return
    current_node = str(getattr(state, "current_node", "") or "").strip()
    node_name = "repair" if current_node == "repair" else "generate"
    node_dir = get_output_dir(state) / "nodes" / node_name
    node_dir.mkdir(parents=True, exist_ok=True)
    (node_dir / "generation_checkpoints.json").write_text(
        json.dumps([item.model_dump(mode="json") for item in state.generation_checkpoints], indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )


def run_or_resume_stage(
    state: PaperBenchReproState,
    stage_name: str,
    input_payload: Any,
    compute: Callable[[], Any],
    load: Callable[[], Any],
    write: Callable[[Any], None],
    *,
    get_output_dir: Callable[[PaperBenchReproState], Path],
    json_default: Callable[[Any], Any],
    load_stage_status: Callable[[PaperBenchReproState], dict[str, Any]],
    write_stage_status: Callable[[PaperBenchReproState, dict[str, Any]], None],
    save_tracking_artifacts: Callable[[PaperBenchReproState], None],
    pipeline_signature: Callable[[], str],
    invalidate_downstream_on_recompute: bool = True,
) -> Any:
    output_dir = get_output_dir(state)
    output_dir.mkdir(parents=True, exist_ok=True)
    _hydrate_stage_attempt_counters(state, get_output_dir=get_output_dir)
    output_paths = STAGE_OUTPUTS.get(stage_name, [])
    stage_status = load_stage_status(state)
    input_hash = _payload_hash(input_payload, json_default=json_default)
    pipeline_signature_value = pipeline_signature()
    existing = stage_status.get(stage_name, {})
    forked_stage_reuse = bool(
        existing.get("resume_source") == "forked_from_run"
        and str(existing.get("forked_from_run_id", "") or "").strip()
        and str(getattr(state.input, "fork_from_run_id", "") or "").strip()
    )
    can_reuse_existing = (
        existing.get("status") == "completed"
        and all((output_dir / item).exists() for item in output_paths)
        and (
            forked_stage_reuse
            or (
                existing.get("input_hash") == input_hash
                and existing.get("pipeline_signature") == pipeline_signature_value
            )
        )
    )
    if can_reuse_existing:
        state.temp_data.setdefault("stage_execution_mode", {})[stage_name] = "resumed"
        resumed_at = datetime.now().isoformat()
        attempt_id = _stage_attempt_id(state, stage_name)
        stage_status[stage_name] = {
            **existing,
            "attempt_id": attempt_id,
            "resume_source": "forked_from_run" if forked_stage_reuse else "reused_artifacts",
            "resumed_at": resumed_at,
        }
        write_stage_status(state, stage_status)
        result = load()
        _append_stage_attempt(
            state,
            StageAttemptRecord(
                attempt_id=attempt_id,
                stage_name=stage_name,
                status="resumed",
                completed_at=resumed_at,
                resume_source="forked_from_run" if forked_stage_reuse else "reused_artifacts",
                output_paths=output_paths,
                input_hash=input_hash,
                pipeline_signature=pipeline_signature_value,
            ),
            get_output_dir=get_output_dir,
            json_default=json_default,
        )
        _record_stage_metric(
            state,
            stage_name,
            0.0,
            input_payload,
            result,
            json_default=json_default,
            notes=["resumed"],
        )
        save_tracking_artifacts(state)
        return result

    if existing.get("status") == "completed" and invalidate_downstream_on_recompute:
        _invalidate_downstream_stages(
            state,
            stage_name,
            "upstream inputs changed",
            load_stage_status=load_stage_status,
            write_stage_status=write_stage_status,
            save_tracking_artifacts=save_tracking_artifacts,
        )

    _mark_stage_started(
        state,
        stage_name,
        load_stage_status=load_stage_status,
        write_stage_status=write_stage_status,
        save_tracking_artifacts=save_tracking_artifacts,
        get_output_dir=get_output_dir,
        json_default=json_default,
    )
    started_at = time.perf_counter()
    try:
        result = compute()
        state.temp_data.setdefault("stage_execution_mode", {})[stage_name] = "computed"
        write(result)
        _record_stage_metric(
            state,
            stage_name,
            time.perf_counter() - started_at,
            input_payload,
            result,
            json_default=json_default,
        )
        _mark_stage_completed(
            state,
            stage_name,
            load_stage_status=load_stage_status,
            write_stage_status=write_stage_status,
            save_tracking_artifacts=save_tracking_artifacts,
            get_output_dir=get_output_dir,
            json_default=json_default,
            extra={
                "input_hash": input_hash,
                "pipeline_signature": pipeline_signature_value,
                "output_paths": output_paths,
            },
        )
        return result
    except Exception as exc:
        _mark_stage_failed(
            state,
            stage_name,
            exc,
            load_stage_status=load_stage_status,
            write_stage_status=write_stage_status,
            save_tracking_artifacts=save_tracking_artifacts,
            get_output_dir=get_output_dir,
            json_default=json_default,
        )
        strict_nonrecoverable_stages = {"local_file_generation"}
        if stage_name in strict_nonrecoverable_stages:
            raise RuntimeError(
                f"{stage_name} failed; refusing to recover or continue from partial generation artifacts. "
                f"original error: {type(exc).__name__}: {exc}"
            ) from exc
        if not bool(getattr(get_workflow_config(), "allow_stage_artifact_recovery_after_failure", False)):
            raise RuntimeError(
                f"{stage_name} failed; refusing to continue with stale stage artifacts. "
                "Set PAPERBENCH_REPRO_ALLOW_STAGE_ARTIFACT_RECOVERY=1 only for forensic resume/debug runs. "
                f"original error: {type(exc).__name__}: {exc}"
            ) from exc
        try:
            result = load()
        except Exception as load_exc:
            raise RuntimeError(
                f"{stage_name} failed and no reusable stage artifact could be loaded; "
                f"original error: {type(exc).__name__}: {exc}; "
                f"artifact load error: {type(load_exc).__name__}: {load_exc}"
            ) from exc
        state.temp_data.setdefault("stage_execution_mode", {})[stage_name] = "recovered_from_existing_artifacts"
        _record_stage_metric(
            state,
            stage_name,
            time.perf_counter() - started_at,
            input_payload,
            result,
            json_default=json_default,
            notes=["recovered_from_existing_artifacts", f"{type(exc).__name__}: {exc}"],
        )
        _record_stage_recovery(
            state,
            stage_name,
            exc,
            action="reused_existing_artifacts_after_compute_failure",
            load_stage_status=load_stage_status,
            write_stage_status=write_stage_status,
            save_tracking_artifacts=save_tracking_artifacts,
            get_output_dir=get_output_dir,
            json_default=json_default,
        )
        return result


def build_runtime_probe() -> RuntimeProbe:
    """Collect a compact local runtime probe."""
    command_candidates = ["python", "python3", "git", "pytest", "conda"]
    available = [command for command in command_candidates if shutil.which(command)]
    missing = [command for command in command_candidates if command not in available]
    gpu_probe_command = next((command for command in ("mx-smi", "nvidia-smi") if shutil.which(command)), "")
    return RuntimeProbe(
        execution_mode="local",
        python_executable=sys.executable,
        gpu_available=bool(gpu_probe_command),
        gpu_probe_command=gpu_probe_command,
        available_commands=available,
        missing_commands=missing,
        notes=["reproagent local runtime probe"],
    )
