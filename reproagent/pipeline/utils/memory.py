"""Run-scoped deterministic memory helpers for reproagent."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from reproagent.pipeline.config import get_workflow_config
from reproagent.pipeline.schemas import PaperBenchReproState, ValidationCheck, ValidationReport

from .run_context import _get_output_dir

_EVENTS_FILENAME = "memory_events.jsonl"
_MEMORY_FILENAME = "memory.md"
_MAX_SUMMARY_CHARS = 4000


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _output_dir(state: PaperBenchReproState) -> Path:
    path = _get_output_dir(state)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _memory_paths(state: PaperBenchReproState) -> tuple[Path, Path]:
    output_dir = _output_dir(state)
    return output_dir / _MEMORY_FILENAME, output_dir / _EVENTS_FILENAME


def _workflow_memory_config() -> tuple[int, int, int]:
    config = get_workflow_config()
    active_lessons = max(1, int(getattr(config, "memory_max_active_lessons", 30) or 30))
    recent_mistakes = max(1, int(getattr(config, "memory_max_recent_mistakes", 12) or 12))
    compact_threshold = max(1000, int(getattr(config, "memory_compact_threshold_chars", 12000) or 12000))
    return active_lessons, recent_mistakes, compact_threshold


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _trim_text(value: Any, *, limit: int = 400) -> str:
    text = _normalize_text(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _signature_from_event(event: dict[str, Any]) -> str:
    return "|".join(
        [
            _normalize_text(event.get("stage")),
            _normalize_text(event.get("scope")),
            _normalize_text(event.get("task_id")),
            _normalize_text(event.get("check_name")),
            _normalize_text(event.get("file_path")),
            _normalize_text(event.get("error_category")),
        ]
    )


def _error_category(event: dict[str, Any]) -> str:
    explicit = _normalize_text(event.get("error_category"))
    if explicit:
        return explicit
    check_name = _normalize_text(event.get("check_name"))
    failure = _normalize_text(event.get("failure")).lower()
    if check_name:
        return check_name
    if "json" in failure or "schema" in failure:
        return "structured_output"
    if "syntax" in failure:
        return "python_syntax"
    if "smoke" in failure:
        return "runtime_smoke"
    if "docker" in failure:
        return "docker_validation"
    if "preflight" in failure:
        return "preflight"
    return "general_failure"


def _avoid_text_for_event(event: dict[str, Any]) -> str:
    explicit = _trim_text(event.get("avoid_next_time"), limit=220)
    if explicit:
        return explicit

    category = _error_category(event)
    stage = _normalize_text(event.get("stage"))
    if category == "python_syntax":
        return "Always produce parseable Python before marking the task complete."
    if category == "file_exists":
        return "Only claim a task complete after writing every required target file under the current task boundary."
    if category == "workpackage_dynamic_smoke":
        return "Do not run work-package smoke until the work package is closed and the declared smoke surface is stable."
    if category == "structured_output":
        return "Return strict JSON only and match the expected schema exactly."
    if category == "repair_plan_eval":
        return "Keep the repair plan aligned with the current validation failures and do not broaden scope."
    if category in {"preflight", "runtime_smoke", "docker_validation"} or stage == "repair_validation":
        return "Preserve validation contracts and runtime entrypoints while repairing only the failing surface."
    return "Use the latest failure evidence from this run to avoid repeating the same mistake."


def _lesson_from_event(event: dict[str, Any]) -> str:
    explicit = _trim_text(event.get("lesson"), limit=220)
    if explicit:
        return explicit
    return _avoid_text_for_event(event)


def _load_memory_events(state: PaperBenchReproState) -> list[dict[str, Any]]:
    _, events_path = _memory_paths(state)
    if not events_path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in events_path.read_text(encoding="utf-8").splitlines():
        rendered = line.strip()
        if not rendered:
            continue
        try:
            payload = json.loads(rendered)
        except Exception:
            continue
        if isinstance(payload, dict):
            events.append(payload)
    return events


def _write_memory_events(state: PaperBenchReproState, events: list[dict[str, Any]]) -> None:
    _, events_path = _memory_paths(state)
    events_path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False, sort_keys=True) for item in events) + ("\n" if events else ""),
        encoding="utf-8",
    )


def _build_active_lessons(events: list[dict[str, Any]], *, max_items: int) -> list[str]:
    lesson_weights: Counter[str] = Counter()
    latest_index: dict[str, int] = {}
    for index, event in enumerate(events):
        lesson = _lesson_from_event(event)
        if not lesson:
            continue
        weight = 1
        if str(event.get("kind", "")).strip() == "failure":
            weight += 2
        if str(event.get("event_name", "")).strip() == "repair_plan_eval_rejected":
            weight += 1
        lesson_weights[lesson] += weight
        latest_index[lesson] = index
    ranked = sorted(
        lesson_weights,
        key=lambda item: (-lesson_weights[item], -latest_index.get(item, -1), item),
    )
    return ranked[:max_items]


def _recent_failure_events(events: list[dict[str, Any]], *, max_items: int) -> list[dict[str, Any]]:
    failures = [event for event in events if str(event.get("kind", "")).strip() == "failure"]
    return failures[-max_items:]


def _render_recent_mistake(event: dict[str, Any]) -> str:
    scope = _normalize_text(event.get("scope")) or "event"
    task_id = _normalize_text(event.get("task_id"))
    header = f"### {_normalize_text(event.get('timestamp'))} {scope}"
    if task_id:
        header += f" {task_id}"
    lines = [header]
    failure = _trim_text(event.get("failure"), limit=500)
    cause = _trim_text(event.get("cause"), limit=500)
    fix_applied = _trim_text(event.get("fix_applied"), limit=500)
    avoid_next_time = _avoid_text_for_event(event)
    if failure:
        lines.append(f"- Failure: {failure}")
    if cause:
        lines.append(f"- Cause: {cause}")
    if fix_applied:
        lines.append(f"- Fix Applied: {fix_applied}")
    if avoid_next_time:
        lines.append(f"- Avoid Next Time: {avoid_next_time}")
    return "\n".join(lines)


def _render_memory_markdown(events: list[dict[str, Any]]) -> str:
    max_lessons, max_recent, _ = _workflow_memory_config()
    active_lessons = _build_active_lessons(events, max_items=max_lessons)
    recent_failures = _recent_failure_events(events, max_items=max_recent)
    lines = ["# PaperBench Repro Memory", "", "## Active Lessons", ""]
    if active_lessons:
        lines.extend(f"- {item}" for item in active_lessons)
    else:
        lines.append("- No active lessons recorded yet.")
    lines.extend(["", "## Recent Mistakes", ""])
    if recent_failures:
        blocks = [_render_recent_mistake(item) for item in recent_failures]
        for index, block in enumerate(blocks):
            if index:
                lines.append("")
            lines.append(block)
    else:
        lines.append("No recent mistakes recorded yet.")
    return "\n".join(lines).strip() + "\n"


def _compact_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    max_lessons, max_recent, _ = _workflow_memory_config()
    recent_failures = _recent_failure_events(events, max_items=max_recent)
    recent_signatures = {_signature_from_event(item) for item in recent_failures}
    compacted: list[dict[str, Any]] = []
    seen_success_signatures: set[str] = set()
    for event in events:
        signature = _signature_from_event(event)
        kind = str(event.get("kind", "")).strip()
        if signature in recent_signatures:
            compacted.append(event)
            continue
        if kind == "success":
            if signature in seen_success_signatures:
                continue
            seen_success_signatures.add(signature)
            compacted.append(event)
            continue
    active_lessons = _build_active_lessons(events, max_items=max_lessons)
    seen_lessons: set[str] = set()
    for lesson in active_lessons:
        if lesson in seen_lessons:
            continue
        seen_lessons.add(lesson)
        compacted.insert(
            0,
            {
                "timestamp": _utc_now(),
                "kind": "lesson",
                "stage": "memory_compact",
                "scope": "memory/lesson",
                "event_name": "compacted_lesson",
                "lesson": lesson,
                "avoid_next_time": lesson,
                "error_category": "lesson",
            },
        )
    deduped: list[dict[str, Any]] = []
    seen_compacted: set[str] = set()
    for event in compacted:
        rendered = json.dumps(event, ensure_ascii=False, sort_keys=True)
        if rendered in seen_compacted:
            continue
        seen_compacted.add(rendered)
        deduped.append(event)
    return deduped


def _write_memory_markdown(state: PaperBenchReproState, events: list[dict[str, Any]]) -> str:
    memory_path, _ = _memory_paths(state)
    markdown = _render_memory_markdown(events)
    _, _, threshold = _workflow_memory_config()
    if len(markdown) > threshold:
        markdown = _render_memory_markdown(_compact_events(events))
    memory_path.write_text(markdown, encoding="utf-8")
    return markdown


def refresh_memory_artifacts(state: PaperBenchReproState) -> None:
    if not state.run_id:
        return
    events = _load_memory_events(state)
    _write_memory_markdown(state, events)


def get_run_memory_prompt(state: PaperBenchReproState, *, max_chars: int | None = None) -> str:
    if not state.run_id:
        return ""
    refresh_memory_artifacts(state)
    memory_path, _ = _memory_paths(state)
    if not memory_path.exists():
        return ""
    text = memory_path.read_text(encoding="utf-8").strip()
    if not text:
        return ""
    limit = max(200, int(max_chars or _MAX_SUMMARY_CHARS))
    if len(text) > limit:
        text = text[: limit - 3].rstrip() + "..."
    return (
        "Run Memory:\n"
        "The following lessons are known from previous failed review/fix attempts in this run.\n"
        "Use them to avoid repeating mistakes, but do not invent new requirements or override the task plan, global contract, or work package contract.\n\n"
        f"{text}"
    )


def append_memory_event(state: PaperBenchReproState, event: dict[str, Any]) -> None:
    if not state.run_id:
        return
    normalized = {
        "timestamp": _normalize_text(event.get("timestamp")) or _utc_now(),
        "kind": _normalize_text(event.get("kind")) or "failure",
        "stage": _normalize_text(event.get("stage")),
        "scope": _normalize_text(event.get("scope")),
        "event_name": _normalize_text(event.get("event_name")),
        "task_id": _normalize_text(event.get("task_id")),
        "check_name": _normalize_text(event.get("check_name")),
        "file_path": _normalize_text(event.get("file_path")),
        "failure": _trim_text(event.get("failure"), limit=600),
        "cause": _trim_text(event.get("cause"), limit=600),
        "fix_applied": _trim_text(event.get("fix_applied"), limit=600),
        "avoid_next_time": _trim_text(event.get("avoid_next_time"), limit=240),
        "lesson": _trim_text(event.get("lesson"), limit=240),
        "error_category": _normalize_text(event.get("error_category")) or _error_category(event),
        "metadata": event.get("metadata") if isinstance(event.get("metadata"), dict) else {},
    }
    events = _load_memory_events(state)
    events.append(normalized)
    _write_memory_events(state, events)
    _write_memory_markdown(state, events)


def record_task_review_failure(
    state: PaperBenchReproState,
    *,
    task_id: str,
    file_path: str,
    scope: str,
    check: dict[str, Any],
) -> None:
    check_name = _normalize_text(check.get("name"))
    append_memory_event(
        state,
        {
            "kind": "failure",
            "stage": "local_file_generation",
            "scope": scope,
            "event_name": "task_review_failed",
            "task_id": task_id,
            "check_name": check_name,
            "file_path": _normalize_text(check.get("file_path")) or file_path,
            "failure": f"{check_name} failed for {file_path}" if check_name else f"task review failed for {file_path}",
            "cause": _trim_text(check.get("error") or check.get("details")),
            "avoid_next_time": _avoid_text_for_event(
                {
                    "check_name": check_name,
                    "stage": "local_file_generation",
                    "failure": check.get("error") or check.get("details") or "",
                }
            ),
        },
    )


def record_task_review_fix(
    state: PaperBenchReproState,
    *,
    task_id: str,
    file_path: str,
    checks: list[dict[str, Any]],
) -> None:
    failed_checks = [item for item in checks if not item.get("passed", False)]
    if not failed_checks:
        return
    first = failed_checks[0]
    append_memory_event(
        state,
        {
            "kind": "success",
            "stage": "local_file_generation",
            "scope": "generate/task_review_fix",
            "event_name": "task_review_fixed_after_retry",
            "task_id": task_id,
            "check_name": _normalize_text(first.get("name")),
            "file_path": _normalize_text(first.get("file_path")) or file_path,
            "failure": f"resolved {_normalize_text(first.get('name'))} after retry",
            "cause": _trim_text(first.get("error") or first.get("details")),
            "fix_applied": f"regenerated {file_path}",
            "avoid_next_time": _avoid_text_for_event(first),
        },
    )


def _all_failed_validation_checks(report: ValidationReport) -> list[ValidationCheck]:
    return [
        item
        for item in [
            *list(report.artifact_checks),
            *list(report.implementation_checks),
            *list(report.trace_checks),
            *list(report.integration_checks),
        ]
        if not item.passed
    ]


def record_validation_failure(state: PaperBenchReproState, *, scope: str = "repair/runtime_validation") -> None:
    report = state.validation_report
    if report is None or report.passed:
        return
    failed_checks = _all_failed_validation_checks(report)
    if not failed_checks:
        append_memory_event(
            state,
            {
                "kind": "failure",
                "stage": "repair_validation",
                "scope": scope,
                "event_name": "repair_validation_failed",
                "failure": f"repair validation failed: {report.overall_status or 'failed'}",
                "cause": "; ".join(list(report.failure_categories) + list(report.blocked_reasons) + list(report.repair_recommendations[:3])),
                "error_category": "repair_validation",
            },
        )
        return
    for check in failed_checks:
        append_memory_event(
            state,
            {
                "kind": "failure",
                "stage": "repair_validation",
                "scope": scope,
                "event_name": "repair_validation_failed",
                "check_name": check.name,
                "file_path": ",".join(check.affected_files[:4]),
                "failure": f"{check.category}:{check.name} failed",
                "cause": check.details,
                "error_category": check.name or check.category or "repair_validation",
                "metadata": {
                    "affected_work_packages": list(check.affected_work_packages),
                    "affected_units": list(check.affected_units),
                },
            },
        )


def record_repair_plan_rejected(state: PaperBenchReproState, *, review: dict[str, Any]) -> None:
    semantic_risks = [str(item).strip() for item in list(review.get("semantic_risks", []) or []) if str(item).strip()]
    append_memory_event(
        state,
        {
            "kind": "failure",
            "stage": "repair_plan_eval",
            "scope": "repair/repair_plan_eval",
            "event_name": "repair_plan_eval_rejected",
            "failure": "repair plan review rejected the proposed plan",
            "cause": "; ".join(semantic_risks) or _trim_text(review.get("summary")),
            "avoid_next_time": "Keep the repair plan tightly aligned with the active validation failures and semantic must-keep constraints.",
            "error_category": "repair_plan_eval",
        },
    )


def record_repair_regeneration_unresolved(
    state: PaperBenchReproState,
    *,
    round_id: int,
    touched_files: list[str],
    suggested_focus: list[str],
) -> None:
    report = state.validation_report
    if report is None or report.passed:
        return
    append_memory_event(
        state,
        {
            "kind": "failure",
            "stage": "repair_regeneration",
            "scope": "repair/repair_regeneration",
            "event_name": "repair_regeneration_unresolved",
            "task_id": ",".join(suggested_focus[:4]),
            "file_path": ",".join(touched_files[:6]),
            "failure": f"repair regeneration round {round_id} did not resolve validation failures",
            "cause": "; ".join(list(report.failure_categories) + list(report.blocked_reasons) + list(report.repair_recommendations[:3])),
            "avoid_next_time": "When a repair round fails, preserve the failing validation contract and target the exact remaining broken surface in the next round.",
            "error_category": "repair_regeneration",
        },
    )


def record_structured_stage_parse_failure(
    state: PaperBenchReproState,
    *,
    stage_name: str,
    schema_name: str,
    error: Exception,
    raw_response: Any = "",
    attempt: int = 1,
) -> None:
    metadata: dict[str, Any] = {"attempt": max(1, int(attempt or 1))}
    excerpt = _trim_text(raw_response, limit=1200)
    if excerpt:
        metadata["raw_response_excerpt"] = excerpt
    append_memory_event(
        state,
        {
            "kind": "failure",
            "stage": stage_name,
            "scope": "structured_json_stage",
            "event_name": "structured_stage_parse_failed",
            "failure": f"{stage_name} failed to parse structured output for schema {schema_name}",
            "cause": str(error),
            "avoid_next_time": "Return strict JSON only and match the expected schema exactly. Do not add markdown fences or extra prose.",
            "error_category": "structured_output",
            "metadata": metadata,
        },
    )
