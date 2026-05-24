"""Run-state transition helpers for reproagent."""
import logging
from datetime import datetime, timezone
from typing import Callable

from reproagent.pipeline.schemas import ExecutionResult, PaperBenchReproState, RepairTicket

logger = logging.getLogger(__name__)

from .storage_manager import (
    _append_event,
    _save_node_artifacts,
    _save_run_summary,
    _save_state_snapshot,
)

def _mark_failed(state: PaperBenchReproState, node_name: str, exc: Exception) -> PaperBenchReproState:
    """Mark node as failed and record error."""
    hard_stop_nodes = {"prepare", "plan", "generate"}
    if node_name in hard_stop_nodes:
        state.status = "failed"
        state.terminal_outcome = "failed"
        state.terminal_outcome_reason = f"{node_name} failed; stop before downstream stages consume partial artifacts"
    else:
        state.status = "completed_unverified"
        state.terminal_outcome = "completed_unverified"
        state.terminal_outcome_reason = f"{node_name} hit a recoverable error; workflow preserved partial artifacts"
    state.current_node = node_name
    state.failed_node = node_name
    state.error_message = str(exc)

    recovery_ticket = {
        "ticket_type": "node_recovery",
        "node": node_name,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "next_action": "stop_before_downstream" if node_name in hard_stop_nodes else "continue_with_partial_artifacts",
        "terminal_outcome": state.terminal_outcome,
        "terminal_outcome_reason": state.terminal_outcome_reason,
    }
    state.temp_data.setdefault("node_errors", []).append(recovery_ticket)
    state.temp_data.setdefault("recovery_tickets", []).append(recovery_ticket)
    if state.repair_ticket is None:
        state.repair_ticket = RepairTicket(
            failure_type=f"{node_name}_recoverable_error",
            reason=str(exc),
            trigger_signals=[type(exc).__name__, node_name],
            evidence={"node": node_name, "error": str(exc)},
            next_fix_scope=[node_name],
        )

    if node_name == "generate" and state.execution_result is None:
        state.execution_result = ExecutionResult(
            success=False,
            error=str(exc),
            exit_code=-1,
        )

    return state

def _save_run_tracking(
    state: PaperBenchReproState,
    save_tracking_artifacts: Callable[[PaperBenchReproState], None] | None,
) -> None:
    """Persist terminal run metadata, including quality_status when available."""
    if save_tracking_artifacts is None:
        _save_run_summary(state)
        return
    try:
        save_tracking_artifacts(state)
    except Exception as exc:  # pragma: no cover - defensive fallback for terminal metadata writes.
        logger.warning("failed to save tracking artifacts; falling back to run_summary: %s", exc, exc_info=True)
        _save_run_summary(state)


def _run_node(
    state: PaperBenchReproState,
    node_name: str,
    fn,
    *,
    save_tracking_artifacts: Callable[[PaperBenchReproState], None] | None = None,
):
    """Node execution wrapper with exception handling and short-circuit."""
    if state.status == "failed" and node_name not in {"repair"}:
        return state
    if state.status == "failed":
        state.status = "completed_unverified"
        state.terminal_outcome = "completed_unverified"
        state.terminal_outcome_reason = "workflow resumed from previous recoverable failure"
    if state.status in {
        "completed_with_degraded_contract",
        "completed_with_runtime_risk",
        "completed_unverified",
    } and node_name not in {"repair"}:
        return state

    state.status = "running"
    state.current_node = node_name

    try:
        _append_event(state, "node_started", {"node": node_name})
        next_state = fn(state)
        if next_state.status not in {
            "failed",
            "completed_with_degraded_contract",
            "completed_with_runtime_risk",
            "completed_unverified",
        }:
            next_state.status = "completed"
        _save_node_artifacts(next_state, node_name)
        _save_state_snapshot(next_state, f"{node_name}_after")
        _append_event(
            next_state,
            "node_completed",
            {"node": node_name, "status": next_state.status, "iteration_count": next_state.iteration_count},
        )
        if node_name in {"generate", "repair"}:
            _save_run_tracking(next_state, save_tracking_artifacts)
        return next_state
    except Exception as exc:
        logger.error("%s - Failed: %s", node_name, exc, exc_info=True)
        failed_state = _mark_failed(state, node_name, exc)
        _save_node_artifacts(failed_state, node_name)
        _save_state_snapshot(failed_state, f"{node_name}_failed")
        _append_event(
            failed_state,
            "node_recovered",
            {
                "node": node_name,
                "error": str(exc),
                "status": failed_state.status,
                "next_action": (
                    "stop_before_downstream"
                    if node_name in {"prepare", "plan", "generate"}
                    else "continue_with_partial_artifacts"
                ),
            },
        )
        _save_run_tracking(failed_state, save_tracking_artifacts)
        return failed_state
