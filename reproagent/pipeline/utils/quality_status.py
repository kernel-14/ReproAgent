"""Quality status projection helpers for reproagent."""

from __future__ import annotations

from reproagent.pipeline.schemas import PaperBenchReproState, QualityStatus


def is_validated_repo_handoff_ready(handoff: dict) -> bool:
    """Return whether the generated repository handoff is ready for downstream use."""

    if not bool(handoff.get("handoff_ready", False)):
        return False
    quality = dict(handoff.get("quality_status", {}) or {})
    if not quality:
        return False
    if not bool(quality.get("validation_passed", False)):
        return False
    if str(quality.get("validation_quality_level", "") or "").strip().lower() in {"", "scaffold_only", "unverified"}:
        return False
    rapid_validation = dict(handoff.get("rapid_validation", {}) or {})
    repo_materialization = dict(rapid_validation.get("repo_materialization", {}) or {})
    latest_smoke = dict(repo_materialization.get("latest_smoke", {}) or {})
    latest_validation = dict(repo_materialization.get("latest_validation", {}) or {})
    if str(latest_smoke.get("status", "") or "").strip().lower() != "passed":
        return False
    if str(latest_validation.get("status", "") or "").strip().lower() != "passed":
        return False
    if list(latest_validation.get("failures", []) or []):
        return False
    return bool(repo_materialization and str(repo_materialization.get("status", "") or "") == "completed")


def _handoff_ready(state: PaperBenchReproState) -> bool:
    return is_validated_repo_handoff_ready(dict(state.temp_data.get("validated_repo_handoff", {}) or {}))


def _run_status(state: PaperBenchReproState) -> str:
    if state.status == "failed":
        return "completed"
    terminal_outcome = str(state.terminal_outcome or "").strip()
    if terminal_outcome:
        return "completed"
    if state.status in {"pending", "running"}:
        return state.status
    return "completed"


def _quality_status(state: PaperBenchReproState) -> str:
    if state.status == "failed":
        return "failed_with_evidence"
    terminal_outcome = str(state.terminal_outcome or state.status or "")
    validation_report = state.validation_report
    if terminal_outcome == "completed" and validation_report is not None and validation_report.passed:
        quality_level = str(validation_report.quality_level or "").strip().lower()
        if quality_level in {"", "scaffold_only", "unverified"}:
            return "unverified"
        if str(validation_report.smoke_status or "").strip().lower() not in {"success", "fixed"}:
            return "runtime_risk"
        if str(validation_report.dynamic_status or "").strip().lower() not in {"success", "fixed"}:
            return "runtime_risk"
        repair_log = state.repair_log
        if repair_log is not None and int(repair_log.rounds_attempted or 0) > 0:
            return "repaired"
        return "validated"
    if terminal_outcome == "completed_with_degraded_contract":
        return "degraded_contract"
    if terminal_outcome == "completed_with_runtime_risk":
        return "runtime_risk"
    if terminal_outcome == "completed_unverified":
        return "unverified"
    if state.status == "failed":
        return "failed_with_evidence"
    return "unverified"


def _next_action(quality_status: str, handoff_ready: bool) -> str:
    if quality_status in {"validated", "repaired"} and handoff_ready:
        return "inspect_repo_handoff"
    if quality_status == "degraded_contract":
        return "rerun_repair_with_more_budget"
    if quality_status == "runtime_risk":
        return "inspect_runtime_logs_then_repair"
    if quality_status == "failed_with_evidence":
        return "inspect_recovery_tickets"
    return "review_partial_artifacts_or_rerun"


def build_quality_status(state: PaperBenchReproState) -> QualityStatus:
    """Project internal reproagent state into separated run and quality status."""
    validation_report = state.validation_report
    handoff_ready = _handoff_ready(state)
    quality_status = _quality_status(state)
    terminal_outcome = str(state.terminal_outcome or state.status or "")
    if state.status == "failed":
        terminal_outcome = "failed"
    return QualityStatus(
        run_status=_run_status(state),
        quality_status=quality_status,
        handoff_ready=handoff_ready,
        terminal_outcome=terminal_outcome,
        terminal_outcome_reason=str(state.terminal_outcome_reason or state.error_message or ""),
        validation_passed=bool(validation_report.passed) if validation_report else False,
        validation_quality_level=str(validation_report.quality_level if validation_report else "scaffold_only"),
        failure_categories=list(validation_report.failure_categories if validation_report else []),
        blocked_reasons=list(validation_report.blocked_reasons if validation_report else []),
        next_recommended_action=_next_action(quality_status, handoff_ready),
    )


def refresh_quality_status(state: PaperBenchReproState) -> QualityStatus:
    """Refresh state and temp cache with the current quality status payload."""
    state.quality_status = build_quality_status(state)
    state.temp_data["quality_status"] = state.quality_status.model_dump(mode="json")
    return state.quality_status
