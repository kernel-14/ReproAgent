"""Repair-selection and repair-round helpers for reproagent workflow execution."""

from __future__ import annotations

import json
import ast
import re
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Callable

from reproagent.pipeline.schemas import (
    ExecutionResult,
    PaperBenchReproState,
    GenerationCheckpoint,
    PreflightResult,
    RepoPlan,
    ValidationCheck,
    ValidationReport,
)
from reproagent.pipeline.utils import memory as run_memory


def _canonical_entry_surface(state: PaperBenchReproState) -> str:
    if state.repo_plan is not None:
        candidate = str(state.repo_plan.canonical_route.entry_surface or "").strip()
        if candidate:
            return candidate
        if state.repo_plan.entrypoints:
            fallback = str(state.repo_plan.entrypoints[0] or "").strip()
            if fallback:
                return fallback
    return str(state.project_plan.entrypoints.get("main", "main.py") or "main.py").strip()


def _canonical_surface_paths(state: PaperBenchReproState, *surface_kinds: str) -> list[str]:
    if state.repo_plan is None:
        return []
    payload = dict(state.repo_plan.canonical_ir or {})
    allowed_kinds = {str(kind).strip() for kind in surface_kinds if str(kind).strip()}
    ordered: list[str] = []
    seen: set[str] = set()
    for item in list(payload.get("surface_nodes", []) or []):
        if not isinstance(item, dict):
            continue
        surface_kind = str(item.get("surface_kind") or "").strip()
        if allowed_kinds and surface_kind not in allowed_kinds:
            continue
        normalized = normalize_repo_path(item.get("canonical_path", ""))
        key = normalize_repo_key(normalized)
        if not normalized or not key or key in seen:
            continue
        seen.add(key)
        ordered.append(normalized)
    return ordered


def _runtime_first_blockers_from_execution_result(execution_result: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract exact runtime blockers that should outrank broad semantic repair."""
    if not execution_result or bool(execution_result.get("success", False)):
        return []

    text_parts: list[str] = []
    for key in ("error", "output"):
        value = str(execution_result.get(key, "") or "").strip()
        if value:
            text_parts.append(value)
    for check in list(execution_result.get("checks", []) or []):
        if not isinstance(check, dict) or bool(check.get("passed", False)):
            continue
        details = str(check.get("error") or check.get("details") or "").strip()
        if details:
            text_parts.append(f"{check.get('name', 'runtime_check')}: {details}")
    raw_text = "\n".join(text_parts).strip()
    if not raw_text:
        return []

    lines = [line.rstrip() for line in raw_text.splitlines()]
    traceback_start = next((idx for idx, line in enumerate(lines) if "Traceback (most recent call last)" in line), -1)
    if traceback_start >= 0:
        tail_lines = lines[traceback_start:]
        next_traceback = next(
            (
                idx
                for idx, line in enumerate(tail_lines[1:], start=1)
                if "Traceback (most recent call last)" in line
            ),
            len(tail_lines),
        )
        trace_lines = tail_lines[:next_traceback]
    else:
        trace_lines = lines
    trace_excerpt = "\n".join(trace_lines[:80]).strip()

    exception_line = ""
    for line in reversed(trace_lines):
        stripped = line.strip()
        if re.match(r"^[A-Za-z_][A-Za-z0-9_.]*(Error|Exception|Warning)?\s*:", stripped):
            exception_line = stripped
            break
    if not exception_line:
        exception_line = (trace_lines[-1].strip() if trace_lines else raw_text[:240]).strip()

    file_refs: list[dict[str, Any]] = []
    seen_refs: set[tuple[str, int]] = set()
    for match in re.finditer(r'File "([^"]+)", line (\d+)(?:, in ([^\n]+))?', raw_text):
        raw_path = match.group(1)
        rel_path = raw_path
        for marker in ("/repo_validation_runtime/working/", "/repo/"):
            if marker in rel_path:
                rel_path = rel_path.rsplit(marker, 1)[1]
                break
        rel_path = rel_path.strip()
        line_no = int(match.group(2))
        key = (rel_path, line_no)
        if key in seen_refs:
            continue
        seen_refs.add(key)
        file_refs.append(
            {
                "path": rel_path,
                "line": line_no,
                "function": str(match.group(3) or "").strip(),
            }
        )

    return [
        {
            "kind": "runtime_traceback" if traceback_start >= 0 else "runtime_execution_failure",
            "summary": exception_line[:500],
            "file_refs": file_refs[:12],
            "traceback": trace_excerpt[:4000],
            "repair_priority": (
                "Fix this exact startup/import/runtime blocker before broad semantic expansion, "
                "artifact backfilling, or contract polishing."
            ),
        }
    ]


def materialize_selected_tasks(
    state: PaperBenchReproState,
    *,
    selected_task_ids: list[str],
    allowed_repo_files: set[str] | None = None,
    initial_execution_result: dict[str, Any] | None = None,
    iteration_seed: int,
    mode_label: str,
    review_scope: str,
    repair_round: int | None,
    ordered_runtime_task_ids: Callable[[Any, Any], list[str]],
    build_runtime_task_views: Callable[[Any, Any], list[dict[str, Any]]],
    build_task_project_plan: Callable[[Any, Any, dict[str, str]], Any],
    filter_task_generated_files: Callable[[dict[str, str], Any], dict[str, str]],
    get_codegen_config: Callable[[], Any],
    get_workflow_config: Callable[[], Any],
    get_sandbox_provider: Callable[[], Any],
    get_output_dir: Callable[[PaperBenchReproState], Path],
    run_task_review: Callable[[dict[str, str], Any, dict[str, Any]], dict[str, Any]],
    repo_agent_engine_cls: Any | None = None,
) -> dict[str, Any]:
    """Materialize a selected ordered task set into the single canonical repo with shared generate/repair semantics."""
    from reproagent.pipeline.tools import load_project_files
    from reproagent.pipeline.utils.repo_agent_engine import RepoAgentEngine

    phase_iteration = max(0, int(repair_round if repair_round is not None else iteration_seed or 0))

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

    def _filter_project_files(project_files: dict[str, str]) -> dict[str, str]:
        if not allowed_repo_files:
            return dict(project_files)
        allowed_keys = {
            str(path).strip()
            for path in allowed_repo_files
            if str(path).strip()
        }
        return {
            path: content
            for path, content in project_files.items()
            if str(path).strip() in allowed_keys
        }

    project_root = Path(state.project_root) if state.project_root else get_output_dir(state) / "repo"
    current_project_files = _filter_project_files(
        load_project_files(project_root) if project_root.exists() else {}
    )
    node_name = "repair" if mode_label == "repair" else "generate"
    node_dir = get_output_dir(state) / "nodes" / node_name
    candidate_root = node_dir / "changed_files"
    if mode_label == "generate" and candidate_root.exists():
        candidate_files = _filter_project_files(load_project_files(candidate_root))
        current_project_files = {
            **candidate_files,
            **current_project_files,
        }
    selected_task_set = {str(task_id).strip() for task_id in list(selected_task_ids or []) if str(task_id).strip()}
    engine_cls = repo_agent_engine_cls or RepoAgentEngine
    engine = engine_cls(
        get_codegen_config(),
        get_workflow_config(),
        get_sandbox_provider(),
    )
    engine.language = state.input.language or "zh"
    engine.output_dir = get_output_dir(state)
    engine.output_dir.mkdir(parents=True, exist_ok=True)
    node_dir.mkdir(parents=True, exist_ok=True)
    engine.node_artifact_dir = node_dir
    engine.checkpoint_path = str((node_dir / "iteration_checkpoint.json").resolve())

    if not selected_task_ids:
        if project_root.exists() or current_project_files:
            engine._persist_project_files(current_project_files)
        return {
            "project_files": current_project_files,
            "execution_history": list(state.execution_history or []),
            "latest_execution_result": dict(initial_execution_result or _empty_execution_result()),
            "latest_suggestions": [],
            "generated_task_ids": [],
            "generation_checkpoints": [],
            "task_usage_summaries": [],
            "touched_files": [],
        }

    runtime_task_views = {item["task_id"]: item for item in build_runtime_task_views(state.project_plan, state.generation_manifest)}
    runtime_order = ordered_runtime_task_ids(state.project_plan, state.generation_manifest)
    runtime_repo_review_points = {
        str(item.get("task_id") or item.get("target_file") or "").strip(): list(item.get("review_points", []) or [])
        for item in (
            state.project_plan.runtime_contract.get("repo_plan_files", [])
            if isinstance(state.project_plan.runtime_contract, dict)
            else []
        )
        if isinstance(item, dict)
    }
    runtime_generation_review_points = {
        str(item.get("task_id") or item.get("file_path") or "").strip(): list(item.get("review_points", []) or [])
        for item in (
            state.project_plan.runtime_contract.get("generation_tasks", [])
            if isinstance(state.project_plan.runtime_contract, dict)
            else []
        )
        if isinstance(item, dict)
    }
    manifest_review_points = {
        str(item.task_id or "").strip(): list(item.review_points or [])
        for item in (list(state.generation_manifest.task_inputs) if state.generation_manifest is not None else [])
        if str(item.task_id or "").strip()
    }
    if mode_label == "repair":
        ordered_selected_task_ids = list(dict.fromkeys(selected_task_ids))
    else:
        ordered_selected_task_ids = [
            task_id for task_id in runtime_order if task_id in set(selected_task_ids)
        ] or list(dict.fromkeys(selected_task_ids))

    execution_history: list[dict[str, Any]] = [
        dict(item) for item in list(state.execution_history or []) if isinstance(item, dict)
    ]
    latest_execution_result = dict(
        initial_execution_result
        or (state.execution_result.model_dump(mode="json") if state.execution_result else _empty_execution_result())
    )
    initial_runtime_execution_result = dict(latest_execution_result)
    runtime_first_blockers = (
        _runtime_first_blockers_from_execution_result(initial_runtime_execution_result)
        if mode_label == "repair"
        else []
    )
    latest_suggestions: list[str] = list(execution_history[-1].get("suggestions", []) or []) if execution_history else []
    workflow_config = get_workflow_config()
    task_review_disabled = bool(getattr(workflow_config, "disable_task_review", False))
    review_fix_budget = 30 if mode_label == "repair" else max(
        0,
        int(
            state.input.stage_review_repair_budget
            or workflow_config.max_stage_fix_rounds
            or 0
        ),
    )
    configured_max_attempts = max(1, int(getattr(workflow_config, "task_review_max_attempts", 3) or 3))
    # Disabling full task review should skip expensive semantic review, not the
    # cheap materialization/syntax retry needed to ensure every planned file lands.
    if task_review_disabled and mode_label == "generate":
        max_task_attempts = configured_max_attempts
    elif task_review_disabled:
        max_task_attempts = 1
    else:
        max_task_attempts = min(configured_max_attempts, 1 + review_fix_budget)
    emitted_checkpoints: list[GenerationCheckpoint] = []
    generated_task_ids: list[str] = []
    task_usage_summaries: list[dict[str, Any]] = []
    touched_files: list[str] = []
    generate_workpackage_review = mode_label == "generate" and not task_review_disabled

    def _append_repair_round_history(
        *,
        task_id: str,
        task_sequence: int,
        task_input: dict[str, Any],
        changed_files: dict[str, str],
        review_result: dict[str, Any],
        repair_trace: list[dict[str, Any]],
        context_usage: dict[str, Any],
        safety_reports: list[dict[str, Any]] | None = None,
    ) -> None:
        if mode_label != "repair":
            return
        failed_checks = [
            dict(check)
            for check in list(review_result.get("checks", []) or [])
            if not bool(check.get("passed", False))
        ]
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "round_id": int(repair_round or phase_iteration or 0),
            "iteration": phase_iteration,
            "task_sequence": task_sequence,
            "task_id": task_id,
            "file_path": str(task_input.get("file_path", "") or ""),
            "work_package_id": str(task_input.get("work_package_id", "") or ""),
            "materialization_mode": mode_label,
            "changed_files": sorted(changed_files.keys()),
            "review_passed": bool(review_result.get("success", False)),
            "failed_checks": failed_checks,
            "suggestions": list(review_result.get("suggestions", []) or []),
            "attempt_count": len(repair_trace),
            "repair_trace": list(repair_trace),
            "context_usage": dict(context_usage),
            "repair_safety": {
                "passed": all(bool(item.get("passed", True)) for item in list(safety_reports or [])),
                "reports": list(safety_reports or [])[-3:],
            } if safety_reports else {},
        }
        history_path = node_dir / "repair_round_history.jsonl"
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with history_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _persist_partial_materialization_state(
        *,
        current_task_id: str,
        task_sequence: int,
        current_project_files: dict[str, str],
        execution_history: list[dict[str, Any]],
        latest_execution_result: dict[str, Any],
    ) -> None:
        """Persist per-task progress before the enclosing stage finishes."""
        generated_files = sorted(current_project_files.keys())
        project_manifest = {
            "entrypoints": {
                **state.project_plan.entrypoints,
                "main": _canonical_entry_surface(state),
            },
            "generated_files": generated_files,
            "current_task": current_task_id,
            "ordered_tasks": list(ordered_selected_task_ids),
            "materialization_mode": mode_label,
            "task_sequence": task_sequence,
        }
        state.execution_history = list(execution_history)
        state.generated_files = generated_files
        state.project_manifest = dict(project_manifest)
        state.generation_checkpoints = list(emitted_checkpoints)
        state.project_root = str(engine._repo_dir().resolve())
        state.execution_result = ExecutionResult(**latest_execution_result)
        node_dir.mkdir(parents=True, exist_ok=True)
        (node_dir / "execution_history.json").write_text(
            json.dumps(execution_history, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        (node_dir / "execution_result.json").write_text(
            json.dumps(latest_execution_result, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        (node_dir / "project_manifest.json").write_text(
            json.dumps(project_manifest, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        (node_dir / "generation_checkpoints.json").write_text(
            json.dumps([item.model_dump(mode="json") for item in emitted_checkpoints], indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    def _materialization_review(
        project_files: dict[str, str],
        task_input: dict[str, Any],
        changed_files: dict[str, str],
    ) -> dict[str, Any]:
        """Cheap review for no-review mode: enforce target materialization and Python syntax only."""
        task_id = str(task_input.get("task_id", "") or "").strip()
        target_file = str(task_input.get("file_path", "") or "").strip()
        checks: list[dict[str, Any]] = []
        exists = bool(target_file and target_file in project_files)
        checks.append(
            {
                "name": "file_exists",
                "task_id": task_id,
                "file_path": target_file,
                "passed": exists,
                "error": "" if exists else f"Missing generated file: {target_file}",
                "details": "Target file materialized." if exists else "Target file was not produced.",
            }
        )
        if exists and target_file.endswith(".py"):
            try:
                ast.parse(project_files.get(target_file, ""))
                syntax_ok = True
                syntax_error = ""
            except SyntaxError as exc:
                syntax_ok = False
                syntax_error = f"{exc.__class__.__name__}: {exc}"
            checks.append(
                {
                    "name": "python_syntax",
                    "task_id": task_id,
                    "file_path": target_file,
                    "passed": syntax_ok,
                    "error": syntax_error,
                    "details": "Python syntax parses." if syntax_ok else syntax_error,
                }
            )
        generated_target = bool(target_file and target_file in changed_files)
        if not generated_target and not exists:
            suggestions = ["Create the planned target file before continuing."]
        elif not all(bool(check.get("passed", False)) for check in checks):
            suggestions = ["Fix the target file syntax/materialization issue."]
        else:
            suggestions = []
        return {
            "success": all(bool(check.get("passed", False)) for check in checks),
            "checks": checks,
            "review_points": [],
            "suggestions": suggestions,
            "disabled_full_task_review": True,
            "review_stage": "materialization_review",
        }

    def _work_package_key(task_id: str, task_input: dict[str, Any]) -> str:
        work_package_id = str(task_input.get("work_package_id", "") or "").strip()
        return work_package_id or f"task::{task_id}"

    def _work_package_owner_files(task_input: dict[str, Any]) -> list[str]:
        files = [
            str(path or "").strip()
            for path in list(task_input.get("work_package_files", []) or [])
            if str(path or "").strip()
        ]
        if not files:
            target_file = str(task_input.get("file_path", "") or "").strip()
            if target_file:
                files = [target_file]
        return list(dict.fromkeys(files))

    def _task_input_with_manifest_points(task_id: str) -> tuple[dict[str, Any], list[str]]:
        task_view = runtime_task_views.get(task_id)
        if task_view is None:
            return {}, []
        task_input = dict(task_view["task_input"])
        task_key = str(task_input.get("task_id") or task_id).strip()
        repo_points = list(runtime_repo_review_points.get(task_key, []))
        runtime_points = (
            runtime_generation_review_points.get(task_key, [])
            if mode_label == "repair"
            else []
        )
        manifest_points = list(manifest_review_points.get(task_key, []))
        merged_review_points = list(
            dict.fromkeys(
                [
                    *repo_points,
                    *list(runtime_points),
                    *manifest_points,
                    *list(task_input.get("review_points", [])),
                    *list(state.repair_plan.review_points if state.repair_plan else []),
                ]
            )
        )
        task_input["review_points"] = merged_review_points
        return task_input, merged_review_points

    selected_work_package_key_by_task: dict[str, str] = {}
    for selected_task_id in ordered_selected_task_ids:
        selected_input, _ = _task_input_with_manifest_points(selected_task_id)
        if not selected_input:
            continue
        selected_work_package_key_by_task[selected_task_id] = _work_package_key(selected_task_id, selected_input)
    work_package_files_by_key: dict[str, list[str]] = {}
    work_package_required_files_by_key: dict[str, list[str]] = {}
    for candidate_task_id in runtime_order:
        candidate_input = dict(runtime_task_views.get(candidate_task_id, {}).get("task_input", {}) or {})
        if not candidate_input:
            continue
        work_package_key = _work_package_key(candidate_task_id, candidate_input)
        target_file = str(candidate_input.get("file_path", "") or "").strip()
        owner_files = [
            str(item or "").strip()
            for item in list(candidate_input.get("work_package_files", []) or [])
            if str(item or "").strip()
        ] or ([target_file] if target_file else [])
        required_files = [
            str(item or "").strip()
            for item in list(candidate_input.get("work_package_required_files", []) or [])
            if str(item or "").strip()
        ] or [
            str(item or "").strip()
            for item in [
                *owner_files,
                *list(candidate_input.get("dependency_files", []) or []),
            ]
            if str(item or "").strip()
        ]
        work_package_files_by_key[work_package_key] = list(
            dict.fromkeys([*work_package_files_by_key.get(work_package_key, []), *owner_files])
        )
        work_package_required_files_by_key[work_package_key] = list(
            dict.fromkeys([*work_package_required_files_by_key.get(work_package_key, []), *required_files])
        )
    last_selected_task_by_work_package: dict[str, str] = {}
    for selected_task_id in ordered_selected_task_ids:
        work_package_key = selected_work_package_key_by_task.get(selected_task_id, "")
        if work_package_key:
            last_selected_task_by_work_package[work_package_key] = selected_task_id

    def _work_package_review_due(
        *,
        task_id: str,
        task_input: dict[str, Any],
        project_files: dict[str, str],
    ) -> tuple[bool, list[str]]:
        if not generate_workpackage_review:
            return False, []
        work_package_key = _work_package_key(task_id, task_input)
        if last_selected_task_by_work_package.get(work_package_key, "") != task_id:
            return False, []
        owner_files = list(
            dict.fromkeys(
                [
                    *_work_package_owner_files(task_input),
                    *work_package_files_by_key.get(work_package_key, []),
                ]
            )
        )
        missing_files = [path for path in owner_files if path not in project_files]
        return not missing_files, missing_files

    def _run_work_package_review(
        *,
        project_files: dict[str, str],
        task_id: str,
        task_input: dict[str, Any],
    ) -> dict[str, Any]:
        work_package_key = _work_package_key(task_id, task_input)
        work_package_id = str(task_input.get("work_package_id", "") or "").strip()
        package_task_ids = [
            candidate_task_id
            for candidate_task_id in runtime_order
            if _work_package_key(
                candidate_task_id,
                dict(runtime_task_views.get(candidate_task_id, {}).get("task_input", {}) or {}),
            ) == work_package_key
        ] or [task_id]
        reviews: list[dict[str, Any]] = []
        checks: list[dict[str, Any]] = []
        suggestions: list[str] = []
        package_owner_files = work_package_files_by_key.get(work_package_key, [])
        package_required_files = work_package_required_files_by_key.get(work_package_key, []) or package_owner_files
        for package_task_id in package_task_ids:
            package_task_input, package_review_points = _task_input_with_manifest_points(package_task_id)
            if not package_task_input:
                continue
            merged_owner_files = list(
                dict.fromkeys(
                    [
                        *list(package_task_input.get("work_package_files", []) or []),
                        *package_owner_files,
                    ]
                )
            )
            merged_required_files = list(
                dict.fromkeys(
                    [
                        *list(package_task_input.get("work_package_required_files", []) or []),
                        *package_required_files,
                    ]
                )
            ) or merged_owner_files
            package_task_input = {
                **package_task_input,
                "work_package_files": merged_owner_files,
                "work_package_required_files": merged_required_files,
            }
            run_package_smoke = package_task_id == task_id
            package_result = run_task_review(
                project_files,
                package_task_input,
                {
                    "task_id": str(package_task_input.get("task_id", "") or package_task_id),
                    "file_path": str(package_task_input.get("file_path", "") or ""),
                    "route": "work_package_review",
                    "review_stage": "work_package_review",
                    "work_package_review": True,
                    "run_workpackage_smoke": run_package_smoke,
                    "work_package_files": merged_owner_files,
                    "work_package_required_files": merged_required_files,
                    "review_points": package_review_points,
                },
            )
            reviews.append(package_result)
            checks.extend(list(package_result.get("checks", []) or []))
            suggestions.extend(list(package_result.get("suggestions", []) or []))
        failed_checks = [check for check in checks if not bool(check.get("passed", False))]
        unique_suggestions = list(dict.fromkeys(str(item) for item in suggestions if str(item).strip()))[:10]
        return {
            "task_id": task_id,
            "review_stage": "work_package_review",
            "success": not failed_checks,
            "checks": checks,
            "review_points": list(task_input.get("review_points", []) or []),
            "failure_summary": unique_suggestions[:6],
            "suggestions": unique_suggestions,
            "work_package_id": work_package_id,
            "work_package_key": work_package_key,
            "reviewed_task_ids": package_task_ids,
            "subreviews": reviews,
        }

    for offset, task_id in enumerate(ordered_selected_task_ids):
        task_sequence = iteration_seed + offset
        task_view = runtime_task_views.get(task_id)
        if task_view is None:
            continue
        generated_task_ids.append(task_id)
        task_input, merged_review_points = _task_input_with_manifest_points(task_id)
        task_manifest = {
            **task_view["task_manifest"],
            "current_task_input": task_input,
            "task_inputs": [task_input],
            "review_points": merged_review_points,
        }
        if state.repair_plan is not None:
            task_manifest["repair_plan"] = state.repair_plan.model_dump(mode="json")

        task_project_files = dict(current_project_files)
        task_suggestions = list(latest_suggestions)
        task_attempt_usages: list[dict[str, Any]] = []
        task_repair_trace: list[dict[str, Any]] = []
        task_generated_files: dict[str, str] = {}
        task_review_result: dict[str, Any] | None = None
        first_failed_checks: list[dict[str, Any]] = []
        generated: dict[str, Any] = {}
        task_safety_reports: list[dict[str, Any]] = []

        for attempt_index in range(max_task_attempts):
            task_project_plan = build_task_project_plan(state.project_plan, task_input, task_project_files)
            iteration_context = {
                "previous_files": task_project_files,
                "execution_result": latest_execution_result,
                "initial_runtime_execution_result": initial_runtime_execution_result,
                "runtime_first_blockers": runtime_first_blockers,
                "score_feedback": dict(state.temp_data.get("repair_score_feedback_prompt", {}) or {}),
                "suggestions": task_suggestions,
                "iteration_count": phase_iteration,
                "task_sequence": task_sequence,
                "review_fix_round": attempt_index,
                "state": state,
            }
            if repair_round is not None:
                iteration_context["repair_round"] = repair_round
                iteration_context["repair_plan"] = state.repair_plan.model_dump(mode="json") if state.repair_plan else {}
                iteration_context["validation_report"] = state.validation_report.model_dump(mode="json") if state.validation_report else {}
                iteration_context["repair_ticket"] = state.repair_ticket.model_dump(mode="json") if state.repair_ticket else {}

            attempt_previous_files = dict(task_project_files)
            generated = engine._generate_project_from_plan(
                state.plan,
                state.input.target,
                task_project_plan,
                task_manifest,
                iteration_context,
                iteration=task_sequence,
                attempt_index=attempt_index,
            )
            generated_project_files = generated.get("project_files")
            candidate_project_files = dict(attempt_previous_files)
            if isinstance(generated_project_files, dict):
                candidate_project_files = dict(generated_project_files)
            else:
                if isinstance(generated.get("updated_files"), dict):
                    candidate_project_files.update(generated["updated_files"])
            generated_updated_files = generated.get("updated_files")
            if not isinstance(generated_updated_files, dict):
                generated_updated_files = {
                    path: content
                    for path, content in candidate_project_files.items()
                    if attempt_previous_files.get(path) != content
                }
            if mode_label == "repair":
                recommended_surfaces = [
                    str(task_input.get("file_path", "") or "").strip(),
                    *list(state.repair_plan.recommended_surfaces if state.repair_plan else []),
                ]
                safety_report = engine._guard_repo_repair_update(
                    previous_files=attempt_previous_files,
                    project_files=candidate_project_files,
                    updated_files=dict(generated_updated_files),
                    recommended_surfaces=[
                        str(item).strip()
                        for item in recommended_surfaces
                        if str(item).strip()
                    ],
                )
                task_safety_reports.append(safety_report)
                engine._write_repair_safety_report(safety_report)
                if not bool(safety_report.get("passed", False)):
                    task_project_files = dict(attempt_previous_files)
                    engine._persist_project_files(task_project_files)
                    guard_issues = [
                        str(item)
                        for item in list(safety_report.get("issues", []) or [])
                        if str(item).strip()
                    ]
                    failed_check = {
                        "name": "repair_safety_guard",
                        "passed": False,
                        "severity": "critical",
                        "details": "; ".join(guard_issues[:8]),
                    }
                    if not first_failed_checks:
                        first_failed_checks = [failed_check]
                    task_suggestions = [
                        "Repair safety guard rejected the previous update; preserve existing implementation and patch only the failing symbols/imports/call sites.",
                        *guard_issues[:6],
                    ]
                    task_repair_trace.append(
                        {
                            "attempt": attempt_index + 1,
                            "success": False,
                            "changed_files": sorted(dict(generated_updated_files).keys()),
                            "suggestions": list(task_suggestions),
                            "safety_guard": {
                                "passed": False,
                                "issues": guard_issues[:12],
                                "file_reports": list(safety_report.get("file_reports", []) or [])[:8],
                            },
                        }
                    )
                    task_attempt_usages.append(dict(generated.get("context_usage", {})))
                    if attempt_index + 1 < max_task_attempts:
                        continue
                    task_review_result = {
                        "success": False,
                        "checks": [failed_check],
                        "review_points": merged_review_points,
                        "suggestions": list(task_suggestions),
                    }
                    break

            task_project_files = candidate_project_files
            attempt_generated_files = filter_task_generated_files(
                generated_updated_files,
                task_input,
            )
            for changed_path in attempt_generated_files:
                if changed_path in task_project_files:
                    task_generated_files[changed_path] = task_project_files[changed_path]
            task_attempt_usages.append(dict(generated.get("context_usage", {})))

            if task_review_disabled:
                task_review_result = _materialization_review(
                    task_project_files,
                    task_input,
                    attempt_generated_files,
                )
            elif mode_label == "generate":
                task_review_result = _materialization_review(
                    task_project_files,
                    task_input,
                    attempt_generated_files,
                )
                if task_review_result["success"]:
                    review_due, missing_owner_files = _work_package_review_due(
                        task_id=task_id,
                        task_input=task_input,
                        project_files=task_project_files,
                    )
                    if review_due:
                        task_review_result = _run_work_package_review(
                            project_files=task_project_files,
                            task_id=task_id,
                            task_input=task_input,
                        )
                    else:
                        task_review_result = {
                            **task_review_result,
                            "review_stage": "materialization_review",
                            "deferred_work_package_review": True,
                            "work_package_id": str(task_input.get("work_package_id", "") or ""),
                            "work_package_review_missing_files": missing_owner_files,
                        }
            else:
                task_review_result = run_task_review(
                    task_project_files,
                    task_input,
                    {
                        "task_id": str(task_input.get("task_id", "") or task_id),
                        "file_path": str(task_input.get("file_path", "") or ""),
                        "route": "task_review",
                        "review_points": merged_review_points,
                    },
                )
            task_suggestions = list(task_review_result.get("suggestions", []) or [])
            task_repair_trace.append(
                {
                    "attempt": attempt_index + 1,
                    "success": task_review_result["success"],
                    "changed_files": sorted(attempt_generated_files.keys()),
                    "suggestions": list(task_suggestions),
                    "agent_success": bool(dict(generated.get("agent_result", {}) or {}).get("success", False)),
                    "agent_exit_code": int(dict(generated.get("agent_result", {}) or {}).get("exit_code", 0) or 0),
                    "agent_error": str(dict(generated.get("agent_result", {}) or {}).get("error", "") or "")[:800],
                }
            )
            if (
                mode_label == "generate"
                and not task_review_result["success"]
                and not attempt_generated_files
                and not bool(dict(generated.get("agent_result", {}) or {}).get("success", False))
            ):
                task_suggestions = [
                    "Previous generation agent call did not produce the target file. Create or update the target file on disk before summarizing.",
                    *task_suggestions,
                ][:10]
                latest_suggestions = list(task_suggestions)
                break
            if not task_review_result["success"]:
                if not first_failed_checks:
                    first_failed_checks = [
                        dict(check)
                        for check in task_review_result["checks"]
                        if not check.get("passed", False)
                    ]
                if not task_review_disabled:
                    for check in task_review_result["checks"]:
                        if check.get("passed", False):
                            continue
                        run_memory.record_task_review_failure(
                            state,
                            task_id=str(task_input.get("task_id", "") or task_id),
                            file_path=str(task_input.get("file_path", "") or ""),
                            scope=review_scope,
                            check=check,
                        )
                if (
                    mode_label == "generate"
                    and str(task_review_result.get("review_stage", "") or "") == "work_package_review"
                ):
                    break
            if task_review_result["success"]:
                if attempt_index > 0 and not task_review_disabled:
                    run_memory.record_task_review_fix(
                        state,
                        task_id=str(task_input.get("task_id", "") or task_id),
                        file_path=str(task_input.get("file_path", "") or ""),
                        checks=first_failed_checks,
                    )
                break

        current_project_files = _filter_project_files(task_project_files)
        touched_files.extend(task_generated_files.keys())
        final_task_review = task_review_result or {
            "success": False,
            "checks": [],
            "review_points": merged_review_points,
            "suggestions": [],
        }
        checks = list(final_task_review["checks"])
        review_stage = str(final_task_review.get("review_stage", "") or "task_review")
        task_success_for_execution = bool(final_task_review["success"])
        if mode_label == "generate" and review_stage == "work_package_review":
            task_success_for_execution = True
        latest_execution_result = {
            "success": latest_execution_result.get("success", True) and task_success_for_execution,
            "output": (
                latest_execution_result.get("output", "")
                + f"\n{mode_label} {review_stage} completed for {task_id}"
                + (
                    f" after {len(task_repair_trace)} attempt(s)"
                    if len(task_repair_trace) > 1
                    else ""
                )
            ).strip(),
            "error": "" if task_success_for_execution else f"{mode_label} {review_stage} detected structural issues.",
            "exit_code": 0 if task_success_for_execution else 1,
            "metrics": dict(latest_execution_result.get("metrics", {})),
            "checks": list(latest_execution_result.get("checks", [])) + checks,
            "artifacts": list(latest_execution_result.get("artifacts", [])),
            "artifact_summary": dict(latest_execution_result.get("artifact_summary", {})),
        }
        if (
            mode_label == "generate"
            and str(final_task_review.get("review_stage", "") or "") == "work_package_review"
            and not bool(final_task_review.get("success", False))
        ):
            latest_suggestions = []
        else:
            latest_suggestions = list(final_task_review.get("suggestions", []) or task_suggestions)
        aggregated_task_usage = engine._summarize_task_usages(task_attempt_usages)
        task_usage_summaries.append(aggregated_task_usage)

        execution_history.append(
            {
                "iteration": phase_iteration,
                "task_sequence": task_sequence,
                "task_id": task_id,
                "task_contract_hash": json.dumps(
                    {
                        "task_id": str(task_input.get("task_id", "") or task_id),
                        "file_path": str(task_input.get("file_path", "") or ""),
                        "work_package_id": str(task_input.get("work_package_id", "") or ""),
                        "dependency_files": list(task_input.get("dependency_files", []) or []),
                        "review_points": list(merged_review_points),
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
                    },
                    sort_keys=True,
                    ensure_ascii=False,
                ),
                "file_path": str(task_input.get("file_path", "") or ""),
                "generated_files": sorted(current_project_files.keys()),
                "result": latest_execution_result,
                "suggestions": list(latest_suggestions),
                "repair_trace": task_repair_trace,
                "context_usage": aggregated_task_usage,
                "task_review": final_task_review,
                "task_review_attempt_count": len(task_repair_trace),
                "repair_safety": {
                    "passed": all(bool(item.get("passed", True)) for item in task_safety_reports),
                    "reports": task_safety_reports[-3:],
                } if mode_label == "repair" and task_safety_reports else {},
                "materialization_mode": mode_label,
            }
        )
        emitted_checkpoints.append(
            GenerationCheckpoint(
                checkpoint_id=f"{mode_label}:{phase_iteration}:{task_sequence}:{task_id}",
                stage="local_file_generation" if mode_label == "generate" else "repair_round",
                focus_id=task_id,
                input_refs=[str(task_input.get("file_path", "") or "")],
                output_refs=sorted(task_generated_files.keys()),
                notes=(
                    [review_stage]
                    if len(task_repair_trace) <= 1
                    else [review_stage, f"attempts={len(task_repair_trace)}"]
                ),
            )
        )
        engine._save_iteration_checkpoint(
            iteration=phase_iteration,
            execution_history=execution_history,
            generated_files=current_project_files,
            termination_reason=f"Completed task {task_id}",
            latest_stage=review_stage,
            latest_status="passed" if final_task_review["success"] else "failed",
        )
        engine._save_attempt(
            phase_iteration,
            current_project_files.get(_canonical_entry_surface(state), ""),
            latest_execution_result,
            list(latest_suggestions),
            execution_history[-1]["repair_trace"],
            aggregated_task_usage,
            current_project_files,
            str(engine._repo_dir().resolve()),
            {
                "entrypoints": {
                    **state.project_plan.entrypoints,
                    "main": _canonical_entry_surface(state),
                },
                "generated_files": sorted(current_project_files.keys()),
                "current_task": task_id,
                "ordered_tasks": list(ordered_selected_task_ids),
                "materialization_mode": mode_label,
                "task_sequence": task_sequence,
            },
            task_id=generated.get("task_id", task_id),
            changed_files=task_generated_files,
        )
        _persist_partial_materialization_state(
            current_task_id=task_id,
            task_sequence=task_sequence,
            current_project_files=current_project_files,
            execution_history=execution_history,
            latest_execution_result=latest_execution_result,
        )
        _append_repair_round_history(
            task_id=task_id,
            task_sequence=task_sequence,
            task_input=task_input,
            changed_files=task_generated_files,
            review_result=final_task_review,
            repair_trace=task_repair_trace,
            context_usage=aggregated_task_usage,
            safety_reports=task_safety_reports,
        )

    if project_root.exists() or current_project_files:
        engine._persist_project_files(current_project_files)
    return {
        "project_files": current_project_files,
        "execution_history": execution_history,
        "latest_execution_result": latest_execution_result,
        "latest_suggestions": latest_suggestions,
        "generated_task_ids": generated_task_ids,
        "generation_checkpoints": emitted_checkpoints,
        "task_usage_summaries": task_usage_summaries,
        "touched_files": list(dict.fromkeys(touched_files)),
    }


def all_failed_validation_checks(report: ValidationReport) -> list[ValidationCheck]:
    """Return all failed checks from a validation report."""
    return [
        item
        for item in [
            *report.artifact_checks,
            *report.implementation_checks,
            *report.trace_checks,
            *report.integration_checks,
        ]
        if not item.passed
    ]


def repair_failure_class(check: ValidationCheck) -> str:
    """Classify one failed validation check into a compact repair class."""
    name = str(check.name or "").lower()
    details = str(check.details or "").lower()
    if "py_compile" in name or any(token in details for token in ("syntaxerror", "indentationerror", "unexpected indent")):
        return "syntax"
    if any(token in details for token in ("cannot import", "importerror", "module not found", "unsupported keyword")):
        return "import"
    if any(token in details for token in ("placeholder", "stub", "todo")):
        return "placeholder"
    if check.category == "artifact" or "artifact missing" in details:
        return "artifact"
    return "contract"


def strengthen_repo_plan_for_repair(
    state: PaperBenchReproState,
    report: ValidationReport,
    work_package_ids: list[str],
) -> RepoPlan | None:
    """Append repair intent to repo-plan file prompts before regeneration."""
    if state.repo_plan is None:
        return None
    selected_work_packages = set(work_package_ids)
    if not selected_work_packages:
        return state.repo_plan

    work_package_by_id = {item.work_package_id: item for item in state.repo_plan.work_packages}
    failed_checks = all_failed_validation_checks(report)
    issue_map: dict[str, list[str]] = {}
    for check in failed_checks:
        failure_class = repair_failure_class(check)
        target_work_packages = list(check.affected_work_packages) or work_package_ids
        for work_package_id in target_work_packages:
            if work_package_id not in selected_work_packages:
                continue
            issue_map.setdefault(work_package_id, []).append(f"[{failure_class}] {check.details}")

    updated_files = []
    for file_plan in state.repo_plan.files:
        if file_plan.work_package_id not in selected_work_packages:
            updated_files.append(file_plan)
            continue
        work_package = work_package_by_id.get(file_plan.work_package_id)
        prompt_suffix_parts: list[str] = []
        package_issues = issue_map.get(file_plan.work_package_id, [])
        if package_issues:
            prompt_suffix_parts.append(
                "Repair focus: "
                + " ".join(package_issues[:3])
                + " Fix repository contract and runnable execution closure before optional additions."
            )
        if work_package is not None:
            package_contract_bits = [*work_package.interface_contract[:4], *work_package.method_obligations[:4]]
            inventory_bits: list[str] = []
            for values in work_package.inventories.values():
                inventory_bits.extend(values[:2])
            if package_contract_bits:
                prompt_suffix_parts.append(
                    "Keep package interface obligations aligned: " + " | ".join(package_contract_bits[:8]) + "."
                )
            if inventory_bits:
                prompt_suffix_parts.append(
                    "Preserve package-owned inventory and artifact surfaces: " + " | ".join(inventory_bits[:8]) + "."
                )
        if file_plan.writes_artifacts:
            prompt_suffix_parts.append(
                "Ensure declared artifact paths are produced: " + " | ".join(file_plan.writes_artifacts[:6]) + "."
            )
        if file_plan.validation_hooks:
            prompt_suffix_parts.append(
                "Satisfy validation hooks first: " + ", ".join(file_plan.validation_hooks[:6]) + "."
            )
        canonical_route = state.repo_plan.canonical_route
        if canonical_route.stage_sequence:
            prompt_suffix_parts.append(
                "Preserve canonical route stages: " + " -> ".join(canonical_route.stage_sequence[:8]) + "."
            )
        if canonical_route.entry_surface:
            prompt_suffix_parts.append(f"Keep canonical entry surface stable: {canonical_route.entry_surface}.")
        artifact_contract_bits = [
            item.relative_path
            for item in state.repo_plan.artifact_contract
            if item.owner_work_package == file_plan.work_package_id and item.relative_path
        ]
        if artifact_contract_bits:
            prompt_suffix_parts.append(
                "Honor artifact contract outputs: " + " | ".join(artifact_contract_bits[:8]) + "."
            )
        unresolved_architecture_failures = list(state.repo_plan.architecture.unresolved_review_failures or [])
        if unresolved_architecture_failures:
            prompt_suffix_parts.append(
                "Close unresolved architecture review failures first: "
                + " | ".join(unresolved_architecture_failures[:6])
                + "."
            )
        canonical_surface_bits = _canonical_surface_paths(
            state,
            "entrypoint",
            "config",
            "stable_interface",
        )
        if canonical_surface_bits:
            prompt_suffix_parts.append(
                "Preserve canonical contract surfaces: "
                + " | ".join(dict.fromkeys(canonical_surface_bits[:12]))
                + "."
            )
        stage_surface_bits = [
            item.path
            for item in state.repo_plan.stage_public_surfaces
            if item.path and item.path == file_plan.target_file
        ]
        if stage_surface_bits:
            prompt_suffix_parts.append(
                "This file is a declared stage public surface and must remain callable/stable."
            )
        if state.repair_plan is not None:
            if state.repair_plan.semantic_must_keep:
                prompt_suffix_parts.append(
                    "Semantic must-keep constraints: "
                    + " | ".join(state.repair_plan.semantic_must_keep[:8])
                    + "."
                )
            if state.repair_plan.failure_focus:
                prompt_suffix_parts.append(
                    "Repair-plan failure focus: " + " | ".join(state.repair_plan.failure_focus[:8]) + "."
                )
            if state.repair_plan.generation_guidance:
                prompt_suffix_parts.append(
                    "Repair-plan generation guidance: "
                    + " | ".join(state.repair_plan.generation_guidance[:8])
                    + "."
                )
            if state.repair_plan.review_points:
                prompt_suffix_parts.append(
                    "Repair-plan review points: " + " | ".join(state.repair_plan.review_points[:8]) + "."
                )
        prompt_suffix = (" " + " ".join(prompt_suffix_parts)) if prompt_suffix_parts else ""
        updated_files.append(
            file_plan.model_copy(
                update={"generation_prompt": (file_plan.generation_prompt + prompt_suffix).strip()}
            )
        )

    return state.repo_plan.model_copy(update={"files": updated_files})


def refresh_generation_views_from_repo_plan(
    state: PaperBenchReproState,
    *,
    build_runtime_project_plan: Callable[[PaperBenchReproState], Any],
    build_generation_manifest: Callable[[PaperBenchReproState], Any],
    write_stage_output: Callable[[PaperBenchReproState, str, object], None],
) -> None:
    """Rebuild runtime project views after repo-plan repair strengthening."""
    if state.repo_plan is None:
        return
    state.project_plan = build_runtime_project_plan(state)
    state.generation_manifest = build_generation_manifest(state)
    write_stage_output(state, "repo_plan.json", state.repo_plan)
    write_stage_output(state, "project_plan.json", state.project_plan)
    write_stage_output(state, "generation_manifest.json", state.generation_manifest)


_OBLIGATION_MATCH_HINTS = (
    "obligation",
    "inventory",
    "prerequisite",
    "measurement",
    "baseline",
    "ablation",
    "experiment",
    "artifact",
    "contract",
)

_OBLIGATION_MATCH_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "into",
    "that",
    "this",
    "these",
    "those",
    "using",
    "used",
    "paper",
    "items",
    "item",
    "owned",
    "files",
    "missing",
    "surface",
    "surfaces",
    "asset",
    "assets",
    "required",
    "details",
    "path",
    "paths",
    "repair",
    "contract",
}

_SCORE_SUMMARY_EXCLUDE_DIR_PARTS = {
    "repo_score_snapshot",
    "adaptive_pruning_score_snapshots",
}

_SCORE_FEEDBACK_STOPWORDS = {
    *_OBLIGATION_MATCH_STOPWORDS,
    "code",
    "implemented",
    "implementation",
    "function",
    "following",
    "correctly",
    "given",
    "computed",
    "using",
    "model",
    "models",
    "dataset",
    "datasets",
    "training",
    "evaluation",
    "score",
    "scores",
    "method",
    "methods",
    "section",
    "table",
    "figure",
    "expect",
    "expects",
    "expected",
    "criterion",
    "criteria",
    "concretely",
    "explicit",
    "explicitly",
    "should",
    "would",
    "could",
    "must",
    "able",
    "memory",
    "family",
    "families",
    "factory",
    "mechanism",
}


def normalize_repo_path(value: str) -> str:
    """Normalize a repo-relative path for repair targeting."""
    normalized = str(value or "").strip().replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    while "//" in normalized:
        normalized = normalized.replace("//", "/")
    return normalized.strip("/")


def normalize_repo_key(value: str) -> str:
    """Return a case-insensitive comparable repo path key."""
    return normalize_repo_path(value).lower()


def _append_unique_path(targets: list[str], seen: set[str], path: str) -> None:
    normalized = normalize_repo_path(path)
    key = normalize_repo_key(normalized)
    if not normalized or not key or key in seen:
        return
    seen.add(key)
    targets.append(normalized)


def _repair_ticket_targets(state: PaperBenchReproState) -> list[str]:
    if state.repair_ticket is None:
        return []
    return [normalize_repo_path(path) for path in state.repair_ticket.required_fix_targets if normalize_repo_path(path)]


def _repair_plan_selected_files(state: PaperBenchReproState) -> list[str]:
    if state.repair_plan is None:
        return []
    selected = [
        *list(state.repair_plan.selected_files),
        *list(state.repair_plan.recommended_surfaces),
    ]
    return [normalize_repo_path(path) for path in selected if normalize_repo_path(path)]


def _repair_plan_selected_work_packages(state: PaperBenchReproState) -> list[str]:
    if state.repair_plan is None:
        return []
    return [str(item).strip() for item in state.repair_plan.selected_work_packages if str(item).strip()]


def _artifact_owner_surface_files(
    state: PaperBenchReproState,
    artifact_paths: list[str],
) -> list[str]:
    if state.repo_plan is None:
        return []
    artifact_keys = {normalize_repo_key(path) for path in artifact_paths if normalize_repo_key(path)}
    if not artifact_keys:
        return []

    owner_work_packages: list[str] = []
    producer_surfaces: list[str] = []
    for item in state.repo_plan.artifact_contract:
        relative_key = normalize_repo_key(item.relative_path)
        if relative_key not in artifact_keys:
            continue
        if item.producer_surface:
            producer_surfaces.append(item.producer_surface)
        if item.owner_work_package:
            owner_work_packages.append(item.owner_work_package)
    if state.global_contract is not None:
        for target in state.global_contract.result_targets:
            target_keys = {normalize_repo_key(path) for path in target.artifact_paths if normalize_repo_key(path)}
            if not artifact_keys.intersection(target_keys):
                continue
            owner_work_packages.extend(target.owner_work_packages)

    selected: list[str] = []
    seen: set[str] = set()
    for path in producer_surfaces:
        _append_unique_path(selected, seen, path)
    for file_plan in state.repo_plan.files:
        if file_plan.work_package_id in owner_work_packages:
            _append_unique_path(selected, seen, file_plan.target_file)
            for path in file_plan.writes_artifacts:
                if normalize_repo_key(path) in artifact_keys:
                    _append_unique_path(selected, seen, file_plan.target_file)
    return selected


def _failed_repair_checks(report: ValidationReport) -> list[Any]:
    """Return failed checks that can identify concrete repair scope."""
    return [
        check
        for check in [
            *report.artifact_checks,
            *report.implementation_checks,
            *report.semantic_checks,
            *report.trace_checks,
            *report.integration_checks,
        ]
        if not check.passed
    ]


def _exact_repair_priority_map(
    state: PaperBenchReproState,
    report: ValidationReport,
) -> dict[str, int]:
    direct_ticket_targets = {
        normalize_repo_key(path)
        for path in _repair_ticket_targets(state)
        if normalize_repo_key(path)
    }
    ticket_owner_surfaces = {
        normalize_repo_key(path)
        for path in _artifact_owner_surface_files(state, _repair_ticket_targets(state))
        if normalize_repo_key(path)
    }
    direct_check_files = {
        normalize_repo_key(path)
        for check in _failed_repair_checks(report)
        for path in check.affected_files
        if normalize_repo_key(path)
    }
    check_owner_surfaces = {
        normalize_repo_key(path)
        for check in _failed_repair_checks(report)
        for path in _artifact_owner_surface_files(state, list(check.affected_files))
        if normalize_repo_key(path)
    }
    entrypoint_surfaces: set[str] = set()
    if state.repo_plan is not None:
        if str(state.repo_plan.canonical_route.entry_surface or "").strip():
            entrypoint_surfaces.add(normalize_repo_key(state.repo_plan.canonical_route.entry_surface))
        entrypoint_surfaces.update(
            normalize_repo_key(path)
            for path in state.repo_plan.entrypoints
            if normalize_repo_key(path)
        )
        entrypoint_surfaces.update(
            normalize_repo_key(item.path)
            for item in state.repo_plan.stage_public_surfaces
            if normalize_repo_key(item.path)
        )
        entrypoint_surfaces.update(
            normalize_repo_key(item.target_file)
            for item in state.repo_plan.files
            if item.writes_artifacts and normalize_repo_key(item.target_file)
        )

    priorities: dict[str, int] = {}
    for key in ticket_owner_surfaces:
        priorities[key] = min(priorities.get(key, 999), 0)
    for key in direct_ticket_targets:
        priorities[key] = min(priorities.get(key, 999), 1)
    for key in direct_check_files:
        priorities[key] = min(priorities.get(key, 999), 2)
    for key in check_owner_surfaces:
        priorities[key] = min(priorities.get(key, 999), 3)
    for key in entrypoint_surfaces:
        priorities[key] = min(priorities.get(key, 999), 4)
    return priorities


def _direct_task_paths(state: PaperBenchReproState) -> set[str]:
    task_paths = {
        normalize_repo_path(item.target_file)
        for item in (state.package_file_planning_output.file_plans if state.package_file_planning_output else [])
    }
    if state.repo_plan is not None:
        task_paths.update(normalize_repo_path(item.target_file) for item in state.repo_plan.files)
    return {path for path in task_paths if path}


def work_package_has_tag(work_package: Any, *tags: str) -> bool:
    """Return True when a work package contains any of the requested tags."""
    package_tags = {str(item).strip().lower() for item in getattr(work_package, "tags", [])}
    return any(str(tag).strip().lower() in package_tags for tag in tags)


def meaningful_obligation_tokens(text: str) -> set[str]:
    """Extract meaningful obligation tokens from free-form repair details."""
    tokens: set[str] = set()
    for raw in text.lower().replace("/", " ").replace("-", " ").split():
        cleaned = "".join(char for char in raw if char.isalnum() or char == "_").strip("_")
        if len(cleaned) < 4 or cleaned in _OBLIGATION_MATCH_STOPWORDS or cleaned.isdigit():
            continue
        tokens.add(cleaned)
    return tokens


def score_feedback_tokens(text: str) -> set[str]:
    """Extract comparable tokens from judge requirements and local planning text."""
    tokens: set[str] = set()
    normalized = str(text or "").lower()
    normalized = normalized.replace("$", " ")
    for raw in re.split(r"[^a-z0-9_]+", normalized):
        cleaned = raw.strip("_")
        if len(cleaned) < 3 or cleaned in _SCORE_FEEDBACK_STOPWORDS or cleaned.isdigit():
            continue
        tokens.add(cleaned)
    return tokens


def _latest_score_summary_path(run_dir: Path) -> Path | None:
    if not run_dir.exists():
        return None
    candidates: list[Path] = []
    for path in run_dir.glob("**/summary.json"):
        relative_parts = set(path.relative_to(run_dir).parts)
        if "nodes" in relative_parts:
            continue
        if any(part.startswith("repo_score_snapshot") for part in relative_parts):
            continue
        if not any(part.startswith("score") for part in relative_parts):
            continue
        if any(part in _SCORE_SUMMARY_EXCLUDE_DIR_PARTS for part in relative_parts):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict) and isinstance(payload.get("graded_task_tree"), dict):
            candidates.append(path)
    if not candidates:
        return None
    candidates.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return candidates[0]


def _score_feedback_leaf_rows(
    node: dict[str, Any],
    path: list[str] | None = None,
    *,
    include_invalid: bool = False,
) -> list[dict[str, Any]]:
    path = list(path or [])
    requirement = str(
        node.get("requirements")
        or node.get("requirement")
        or node.get("name")
        or node.get("id")
        or ""
    ).strip()
    next_path = [*path, requirement] if requirement else list(path)
    children = [item for item in list(node.get("sub_tasks") or []) if isinstance(item, dict)]
    if children:
        rows: list[dict[str, Any]] = []
        for child in children:
            rows.extend(_score_feedback_leaf_rows(child, next_path, include_invalid=include_invalid))
        return rows
    score = node.get("score")
    valid_score = node.get("valid_score", True)
    try:
        numeric_score = float(score)
    except Exception:
        numeric_score = 0.0
    is_valid_score = bool(valid_score)
    if not is_valid_score and not include_invalid:
        return []
    if is_valid_score and numeric_score >= 1.0:
        return []
    explanation = str(node.get("explanation") or "").strip()
    judge_metadata = dict(node.get("judge_metadata", {}) or {})
    judge_response = str(judge_metadata.get("full_judge_response", "") or "")
    return [
        {
            "requirement": requirement,
            "path": next_path[-4:],
            "score": numeric_score,
            "valid_score": is_valid_score,
            "weight": node.get("weight", 1),
            "explanation": explanation[:900],
            "judge_response": judge_response[:3000],
            "tokens": sorted(score_feedback_tokens(" ".join(next_path) + " " + explanation + " " + judge_response)),
        }
    ]


def _repo_paths_from_score_leaf(leaf: dict[str, Any], known_paths: set[str]) -> list[str]:
    text = "\n".join(
        [
            str(leaf.get("requirement", "") or ""),
            " ".join(str(item) for item in list(leaf.get("path", []) or [])),
            str(leaf.get("explanation", "") or ""),
            str(leaf.get("judge_response", "") or ""),
        ]
    )
    hits: list[str] = []
    for match in re.finditer(r"(?:^|[\"'`(\\s])((?:src|tests|configs|scripts)/[A-Za-z0-9_./-]+|[A-Za-z0-9_.-]+\\.py)", text):
        path = normalize_repo_path(match.group(1).rstrip(".,;:)\"'`"))
        if path in known_paths:
            hits.append(path)
    for path in known_paths:
        if path and path in text:
            hits.append(path)
    return list(dict.fromkeys(hits))


def _score_feedback_strong_tokens(tokens: set[str]) -> set[str]:
    strong = set()
    for token in tokens:
        if "_" in token or any(char.isdigit() for char in token) or len(token) >= 7:
            strong.add(token)
    return strong or set(tokens)


def _score_feedback_surface_index(state: PaperBenchReproState) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if state.repo_plan is not None:
        work_packages = {item.work_package_id: item for item in state.repo_plan.work_packages}
        for file_plan in state.repo_plan.files:
            work_package = work_packages.get(file_plan.work_package_id)
            strong_parts = [
                file_plan.target_file,
                file_plan.purpose,
                file_plan.hypothesis,
                file_plan.decision_value,
                file_plan.stop_rule_or_pruning_rationale,
                " ".join(file_plan.interface_contract),
                " ".join(file_plan.implementation_surfaces),
                " ".join(file_plan.method_obligations),
                " ".join(file_plan.defines_symbols),
                " ".join(file_plan.calls_symbols),
                " ".join(file_plan.writes_artifacts),
                " ".join(file_plan.validation_hooks),
                " ".join(file_plan.review_points),
            ]
            weak_parts = [file_plan.generation_prompt]
            if work_package is not None:
                strong_parts.extend(
                    [
                        str(getattr(work_package, "work_package_id", "") or ""),
                        str(getattr(work_package, "goal", "") or ""),
                        " ".join(list(getattr(work_package, "tags", []) or [])),
                        " ".join(list(getattr(work_package, "interface_contract", []) or [])),
                        " ".join(list(getattr(work_package, "method_obligations", []) or [])),
                        " ".join(
                            item
                            for values in dict(getattr(work_package, "inventories", {}) or {}).values()
                            for item in list(values or [])
                        ),
                    ]
                )
            rows.append(
                {
                    "file_path": normalize_repo_path(file_plan.target_file),
                    "task_id": str(file_plan.task_id or "").strip(),
                    "work_package_id": str(file_plan.work_package_id or "").strip(),
                    "strong_tokens": score_feedback_tokens(" ".join(str(item) for item in strong_parts)),
                    "weak_tokens": score_feedback_tokens(" ".join(str(item) for item in weak_parts)),
                }
            )
    elif state.generation_manifest is not None:
        for item in state.generation_manifest.task_inputs:
            strong_parts = [
                item.task_id,
                item.file_path,
                " ".join(item.review_points),
                " ".join(item.method_obligations),
                " ".join(item.implementation_surfaces),
                " ".join(item.defines_symbols),
                " ".join(item.calls_symbols),
                item.hypothesis,
                item.decision_value,
            ]
            weak_parts = [item.generation_prompt]
            rows.append(
                {
                    "file_path": normalize_repo_path(item.file_path),
                    "task_id": str(item.task_id or "").strip(),
                    "work_package_id": str(getattr(item, "work_package_id", "") or "").strip(),
                    "strong_tokens": score_feedback_tokens(" ".join(str(item) for item in strong_parts)),
                    "weak_tokens": score_feedback_tokens(" ".join(str(item) for item in weak_parts)),
                }
            )
    return [row for row in rows if row.get("file_path")]


def load_repair_score_feedback(
    state: PaperBenchReproState,
    *,
    run_dir: Path,
    max_items: int = 32,
    max_files: int = 8,
) -> dict[str, Any]:
    """Load latest judge score failures and map them to local repair surfaces."""
    summary_path = _latest_score_summary_path(run_dir)
    if summary_path is None:
        return {}
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    graded_tree = payload.get("graded_task_tree")
    if not isinstance(graded_tree, dict):
        return {}
    all_problem_leaves = _score_feedback_leaf_rows(graded_tree, include_invalid=True)
    leaves = [leaf for leaf in all_problem_leaves if bool(leaf.get("valid_score", True))]
    invalid_leaves = [leaf for leaf in all_problem_leaves if not bool(leaf.get("valid_score", True))]
    if not leaves and not invalid_leaves:
        return {}

    surface_index = _score_feedback_surface_index(state)
    surface_by_path = {str(item.get("file_path", "") or ""): item for item in surface_index}
    known_paths = set(surface_by_path)
    selected_items: list[dict[str, Any]] = []
    file_scores: dict[str, int] = {}
    file_task_ids: dict[str, str] = {}
    file_work_packages: dict[str, str] = {}
    package_scores: dict[str, int] = {}

    def _leaf_priority(row: dict[str, Any]) -> tuple[float, float]:
        try:
            numeric_score = float(row.get("score", 0) or 0)
        except Exception:
            numeric_score = 0.0
        try:
            weight = float(row.get("weight", 1) or 1)
        except Exception:
            weight = 1.0
        return numeric_score, -weight

    for leaf in sorted(leaves, key=_leaf_priority)[: max(1, int(max_items or 1))]:
        leaf_tokens = set(leaf.get("tokens", []) or [])
        explicit_paths = _repo_paths_from_score_leaf(leaf, known_paths)
        matches: list[tuple[int, dict[str, Any]]] = []
        for path in explicit_paths:
            surface = surface_by_path.get(path)
            if surface:
                matches.append((1000, surface))
        if leaf_tokens:
            strong_leaf_tokens = _score_feedback_strong_tokens(leaf_tokens)
            for surface in surface_index:
                if str(surface.get("file_path", "") or "") in explicit_paths:
                    continue
                strong_surface_tokens = set(surface.get("strong_tokens", set()) or set())
                weak_surface_tokens = set(surface.get("weak_tokens", set()) or set())
                strong_overlap = len(strong_leaf_tokens.intersection(strong_surface_tokens))
                weak_overlap = len(leaf_tokens.intersection(weak_surface_tokens))
                overlap = strong_overlap * 6 + min(weak_overlap, 4)
                if overlap > 0:
                    matches.append((overlap, surface))
        matches.sort(key=lambda item: (-item[0], str(item[1].get("file_path", ""))))
        top_surfaces = [
            {
                "file_path": str(surface.get("file_path", "") or ""),
                "task_id": str(surface.get("task_id", "") or ""),
                "work_package_id": str(surface.get("work_package_id", "") or ""),
                "overlap": overlap,
            }
            for overlap, surface in matches[:3]
        ]
        if not top_surfaces and state.project_plan.entrypoints:
            entry = normalize_repo_path(str(next(iter(state.project_plan.entrypoints.values())) or ""))
            if entry:
                top_surfaces = [{"file_path": entry, "task_id": "", "work_package_id": "", "overlap": 0}]
        for surface in top_surfaces:
            path = normalize_repo_path(surface.get("file_path", ""))
            if not path:
                continue
            file_scores[path] = file_scores.get(path, 0) + int(surface.get("overlap", 0) or 1)
            if surface.get("task_id"):
                file_task_ids[path] = str(surface.get("task_id", "") or "")
            if surface.get("work_package_id"):
                file_work_packages[path] = str(surface.get("work_package_id", "") or "")
                package_scores[file_work_packages[path]] = package_scores.get(file_work_packages[path], 0) + int(surface.get("overlap", 0) or 1)
        selected_items.append(
            {
                "requirement": str(leaf.get("requirement", "") or "")[:260],
                "path": list(leaf.get("path", []) or [])[-4:],
                "score": leaf.get("score", 0),
                "valid_score": bool(leaf.get("valid_score", True)),
                "weight": leaf.get("weight", 1),
                "suggested_surfaces": top_surfaces,
                "explanation": str(leaf.get("explanation", "") or "")[:500],
            }
        )

    prioritized_files = [
        path
        for path, _score in sorted(file_scores.items(), key=lambda item: (-item[1], item[0]))
    ][: max(1, int(max_files or 1))]
    prioritized_task_ids = [
        file_task_ids[path]
        for path in prioritized_files
        if file_task_ids.get(path)
    ]
    prioritized_work_packages = [
        package_id
        for package_id, _score in sorted(package_scores.items(), key=lambda item: (-item[1], item[0]))
    ]
    return {
        "source": str(summary_path),
        "score": payload.get("score"),
        "model": payload.get("model"),
        "max_depth": payload.get("max_depth"),
        "num_leaf_nodes": payload.get("num_leaf_nodes"),
        "num_invalid_leaf_nodes": payload.get("num_invalid_leaf_nodes"),
        "failed_leaf_count": len(leaves),
        "invalid_leaf_count": len(invalid_leaves),
        "invalid_leaf_examples": [
            {
                "requirement": str(leaf.get("requirement", "") or "")[:260],
                "path": list(leaf.get("path", []) or [])[-4:],
                "explanation": str(leaf.get("explanation", "") or "")[:360],
            }
            for leaf in invalid_leaves[:8]
        ],
        "items": selected_items,
        "prioritized_files": prioritized_files,
        "prioritized_task_ids": list(dict.fromkeys(prioritized_task_ids)),
        "prioritized_work_packages": list(dict.fromkeys(prioritized_work_packages)),
    }


def score_feedback_for_prompt(feedback: dict[str, Any], *, max_items: int = 12) -> dict[str, Any]:
    """Compact score feedback for LLM prompts."""
    if not feedback:
        return {}
    items = []
    for item in list(feedback.get("items", []) or [])[:max_items]:
        if not bool(item.get("valid_score", True)):
            continue
        items.append(
            {
                "requirement": str(item.get("requirement", "") or "")[:240],
                "score": item.get("score", 0),
                "valid_score": bool(item.get("valid_score", True)),
                "suggested_surfaces": list(item.get("suggested_surfaces", []) or [])[:3],
                "explanation": str(item.get("explanation", "") or "")[:360],
            }
        )
    prioritized_files = list(feedback.get("prioritized_files", []) or [])[:12]
    prioritized_work_packages = list(feedback.get("prioritized_work_packages", []) or [])[:12]
    if not items and not prioritized_files and not prioritized_work_packages:
        return {}
    return {
        "source": str(feedback.get("source", "") or ""),
        "score": feedback.get("score"),
        "model": feedback.get("model"),
        "num_leaf_nodes": feedback.get("num_leaf_nodes"),
        "num_invalid_leaf_nodes": feedback.get("num_invalid_leaf_nodes"),
        "invalid_leaf_count_excluded_from_repair": feedback.get("invalid_leaf_count", 0),
        "prioritized_files": prioritized_files,
        "prioritized_work_packages": prioritized_work_packages,
        "failed_leaf_examples": items,
    }


def repo_plan_work_package_order(state: PaperBenchReproState) -> list[str]:
    """Return a stable work-package order preferring repo-plan ownership when available."""
    if state.repo_plan is not None:
        return [item.work_package_id for item in state.repo_plan.work_packages]
    if state.work_package_planning is not None:
        return [item.work_package_id for item in state.work_package_planning.work_packages]
    return []


def obligation_related_work_packages(state: PaperBenchReproState, details: str) -> list[str]:
    """Infer work packages from obligation-like repair details."""
    if state.repo_plan is None:
        return []
    lowered = details.lower()
    if not any(hint in lowered for hint in _OBLIGATION_MATCH_HINTS):
        return []
    detail_tokens = meaningful_obligation_tokens(details)
    if not detail_tokens:
        return []
    related: list[str] = []
    for work_package in state.repo_plan.work_packages:
        contract_tokens = meaningful_obligation_tokens(
            " ".join(
                [
                    *work_package.interface_contract,
                    *work_package.method_obligations,
                    *work_package.tags,
                    *[item for values in work_package.inventories.values() for item in values],
                ]
            )
        )
        if contract_tokens and detail_tokens.intersection(contract_tokens):
            related.append(work_package.work_package_id)
    return related


def entrypoint_related_work_packages(state: PaperBenchReproState) -> list[str]:
    """Return work packages that own entrypoint-like surfaces."""
    selected: list[str] = []
    entrypoint_paths: set[str] = set()
    if state.repo_plan is not None:
        if str(state.repo_plan.canonical_route.entry_surface or "").strip():
            entrypoint_paths.add(str(state.repo_plan.canonical_route.entry_surface or "").strip())
        entrypoint_paths.update(state.repo_plan.entrypoints)
        for work_package in state.repo_plan.work_packages:
            if work_package_has_tag(work_package, "entrypoint", "config", "orchestration"):
                selected.append(work_package.work_package_id)
        for file_plan in state.repo_plan.files:
            if file_plan.target_file in entrypoint_paths and file_plan.work_package_id:
                selected.append(file_plan.work_package_id)
    return list(dict.fromkeys(selected))


def close_work_package_dependencies(
    state: PaperBenchReproState,
    work_package_ids: list[str],
    *,
    entrypoint_related_work_packages: Callable[[PaperBenchReproState], list[str]],
) -> list[str]:
    """Expand selected work packages through dependency closure while preserving stable order."""
    dependency_map: dict[str, set[str]] = {}
    if state.repo_plan is not None:
        dependency_map = {item.work_package_id: set(item.depends_on) for item in state.repo_plan.work_packages}
    elif state.work_package_planning is not None:
        dependency_map = {item.work_package_id: set(item.depends_on) for item in state.work_package_planning.work_packages}
    closure = set(work_package_ids)
    changed = True
    while changed:
        changed = False
        for package_id in list(closure):
            for dependency in dependency_map.get(package_id, set()):
                if dependency and dependency not in closure:
                    closure.add(dependency)
                    changed = True

    preferred = set(entrypoint_related_work_packages(state))
    ordered_ids = repo_plan_work_package_order(state)
    prioritized = [package_id for package_id in ordered_ids if package_id in closure and package_id in preferred]
    prioritized.extend(
        package_id
        for package_id in ordered_ids
        if package_id in closure and package_id not in prioritized
    )
    return prioritized or list(dict.fromkeys(work_package_ids))


def select_repair_work_packages(
    state: PaperBenchReproState,
    report: ValidationReport,
    *,
    entrypoint_related_work_packages: Callable[[PaperBenchReproState], list[str]],
) -> list[str]:
    """Select work-package ids implicated by a validation report."""
    affected: list[str] = list(_repair_plan_selected_work_packages(state))
    if state.repo_plan is not None and state.repair_ticket is not None:
        target_keys = {
            normalize_repo_key(path)
            for path in state.repair_ticket.required_fix_targets
            if normalize_repo_key(path)
        }
        for file_plan in state.repo_plan.files:
            if normalize_repo_key(file_plan.target_file) in target_keys and file_plan.work_package_id:
                affected.append(file_plan.work_package_id)
            if any(normalize_repo_key(path) in target_keys for path in file_plan.writes_artifacts):
                if file_plan.work_package_id:
                    affected.append(file_plan.work_package_id)
    for check in _failed_repair_checks(report):
        affected.extend(check.affected_work_packages)
        affected.extend(obligation_related_work_packages(state, check.details))
        if check.affected_files and state.global_contract is not None:
            affected_file_keys = {
                normalize_repo_key(path)
                for path in check.affected_files
                if normalize_repo_key(path)
            }
            for target in state.global_contract.result_targets:
                if any(normalize_repo_key(path) in affected_file_keys for path in target.artifact_paths):
                    affected.extend(target.owner_work_packages)
    if any(not item.passed for item in report.integration_checks):
        affected.extend(entrypoint_related_work_packages(state))
    if affected:
        return close_work_package_dependencies(
            state,
            list(dict.fromkeys(affected)),
            entrypoint_related_work_packages=entrypoint_related_work_packages,
        )
    if state.work_package_planning is None:
        return []
    return close_work_package_dependencies(
        state,
        [item.work_package_id for item in state.work_package_planning.work_packages],
        entrypoint_related_work_packages=entrypoint_related_work_packages,
    )


def select_exact_repair_files(
    state: PaperBenchReproState,
    report: ValidationReport,
    *,
    work_package_file_index: Callable[[PaperBenchReproState], dict[str, list[str]]],
    global_repair_surface_files: Callable[[PaperBenchReproState], list[str]],
) -> list[str]:
    """Select directly repairable task files from the validation report."""
    task_paths = _direct_task_paths(state)
    selected: list[str] = []
    seen: set[str] = set()
    failed_categories: set[str] = set()
    file_index = work_package_file_index(state)
    focused_plan_scope = bool(_repair_plan_selected_files(state))

    for path in _repair_plan_selected_files(state):
        if path in task_paths:
            _append_unique_path(selected, seen, path)
            continue
        for owner_surface in _artifact_owner_surface_files(state, [path]):
            if owner_surface in task_paths:
                _append_unique_path(selected, seen, owner_surface)

    for path in _repair_ticket_targets(state):
        if path in task_paths:
            _append_unique_path(selected, seen, path)
            continue
        for owner_surface in _artifact_owner_surface_files(state, [path]):
            if owner_surface in task_paths:
                _append_unique_path(selected, seen, owner_surface)

    if focused_plan_scope and selected:
        return list(selected)

    for check in _failed_repair_checks(report):
        failed_categories.add(check.category)
        for path in check.affected_files:
            normalized = normalize_repo_path(path)
            if normalized in task_paths:
                _append_unique_path(selected, seen, normalized)
            for owner_surface in _artifact_owner_surface_files(state, [normalized]):
                if owner_surface in task_paths:
                    _append_unique_path(selected, seen, owner_surface)
        if not check.affected_files and check.affected_work_packages:
            for work_package_id in check.affected_work_packages:
                for path in file_index.get(work_package_id, []):
                    _append_unique_path(selected, seen, path)
        elif check.affected_work_packages:
            for work_package_id in check.affected_work_packages:
                if any(
                    normalize_repo_path(path) in task_paths
                    for path in check.affected_files
                ):
                    continue
                for path in file_index.get(work_package_id, []):
                    _append_unique_path(selected, seen, path)
    if not selected and failed_categories.intersection({"artifact", "semantic", "trace", "integration"}):
        for path in global_repair_surface_files(state):
            _append_unique_path(selected, seen, path)
    priority_map = _exact_repair_priority_map(state, report)
    indexed = list(enumerate(selected))
    indexed.sort(key=lambda item: (priority_map.get(normalize_repo_key(item[1]), 999), item[0]))
    return [path for _, path in indexed]


def task_ids_for_repair_files(
    state: PaperBenchReproState,
    file_paths: list[str],
    *,
    ordered_runtime_task_ids: Callable[[Any, Any], list[str]],
    preserve_file_order: bool = False,
) -> list[str]:
    """Map file paths to ordered task ids."""
    runtime_order = ordered_runtime_task_ids(state.project_plan, state.generation_manifest)
    if state.repo_plan is not None:
        task_inputs = {item.target_file: item.task_id for item in state.repo_plan.files if item.task_id}
        selected_ids = [task_inputs[path] for path in file_paths if path in task_inputs]
        if preserve_file_order and selected_ids:
            return list(dict.fromkeys(selected_ids))
        if runtime_order:
            ordered = [task_id for task_id in runtime_order if task_id in selected_ids]
            if ordered:
                return ordered
        if selected_ids:
            return list(dict.fromkeys(selected_ids))
    if state.generation_manifest is None:
        return []
    task_inputs = {item.file_path: item.task_id for item in state.generation_manifest.task_inputs}
    selected_ids = [task_inputs[path] for path in file_paths if path in task_inputs]
    if preserve_file_order and selected_ids:
        return list(dict.fromkeys(selected_ids))
    return [task_id for task_id in runtime_order if task_id in selected_ids] if runtime_order else []


def task_ids_for_repair_work_packages(
    state: PaperBenchReproState,
    work_package_ids: list[str],
    *,
    ordered_runtime_task_ids: Callable[[Any, Any], list[str]],
    work_package_file_index: Callable[[PaperBenchReproState], dict[str, list[str]]],
    task_ids_for_repair_files: Callable[[PaperBenchReproState, list[str]], list[str]],
) -> list[str]:
    """Map work-package ids to ordered task ids via produced file paths."""
    runtime_order = ordered_runtime_task_ids(state.project_plan, state.generation_manifest)
    if state.repo_plan is not None:
        selected_ids = [
            item.task_id
            for item in state.repo_plan.files
            if item.work_package_id in work_package_ids and item.task_id
        ]
        if runtime_order:
            ordered = [task_id for task_id in runtime_order if task_id in selected_ids]
            if ordered:
                return ordered
        if selected_ids:
            return list(dict.fromkeys(selected_ids))
    file_index = work_package_file_index(state)
    selected_files: list[str] = []
    for work_package_id in work_package_ids:
        selected_files.extend(file_index.get(work_package_id, []))
    if not selected_files and state.package_file_planning_output is not None:
        selected_files = [item.target_file for item in state.package_file_planning_output.file_plans]
    return task_ids_for_repair_files(state, list(dict.fromkeys(selected_files)))


def repair_fallback_work_packages(
    state: PaperBenchReproState,
    report: ValidationReport,
    work_package_ids: list[str],
    *,
    select_exact_repair_files: Callable[[PaperBenchReproState, ValidationReport], list[str]],
) -> list[str]:
    """Prefer package fallback only for packages not already covered by exact-file repair."""
    selected_packages = set(work_package_ids)
    exact_targets = set(select_exact_repair_files(state, report))
    if not exact_targets:
        return list(dict.fromkeys(work_package_ids))
    covered_packages: set[str] = set()
    if state.repo_plan is not None:
        for file_plan in state.repo_plan.files:
            if file_plan.target_file in exact_targets and file_plan.work_package_id:
                covered_packages.add(file_plan.work_package_id)
    fallback = [
        package_id
        for package_id in work_package_ids
        if package_id in selected_packages and package_id not in covered_packages
    ]
    return fallback or list(dict.fromkeys(work_package_ids))


def run_repair_generation_round(
    state: PaperBenchReproState,
    *,
    selected_task_ids: list[str],
    round_id: int,
    ordered_runtime_task_ids: Callable[[Any, Any], list[str]],
    build_runtime_task_views: Callable[[Any, Any], list[dict[str, Any]]],
    build_task_project_plan: Callable[[Any, Any, dict[str, str]], Any],
    filter_task_generated_files: Callable[[dict[str, str], Any], dict[str, str]],
    get_codegen_config: Callable[[], Any],
    get_workflow_config: Callable[[], Any],
    get_sandbox_provider: Callable[[], Any],
    get_output_dir: Callable[[PaperBenchReproState], Path],
    run_task_review: Callable[[dict[str, str], Any, dict[str, Any]], dict[str, Any]],
    build_generate_stage_output: Callable[[PaperBenchReproState], Any],
    write_stage_output: Callable[[PaperBenchReproState, str, object], None],
    persist_generation_checkpoints: Callable[[PaperBenchReproState], None],
) -> dict[str, Any]:
    """Re-generate a selected set of tasks against the current repo snapshot."""
    if not selected_task_ids or state.project_root == "":
        return {"touched_files": [], "agent_usage_summary": {}}
    materialization = materialize_selected_tasks(
        state,
        selected_task_ids=selected_task_ids,
        iteration_seed=state.iteration_count,
        mode_label="repair",
        review_scope="repair/task_review",
        repair_round=round_id,
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
    latest_execution_result = dict(materialization["latest_execution_result"])
    latest_preflight = {
        "status": "passed",
        "checks": list(latest_execution_result.get("checks", [])),
        "blocking_failures": [],
        "warning_messages": [],
        "suggested_fixes": [],
    }
    latest_experiment_results = dict(state.experiment_results)
    latest_suggestions = list(materialization["latest_suggestions"])
    touched_files = list(materialization["touched_files"])
    state.temp_data["pending_repair_regeneration_attempt"] = {
        "round_id": round_id,
        "touched_files": list(dict.fromkeys(touched_files)),
        "selected_task_ids": list(selected_task_ids),
    }

    state.code = current_project_files.get(_canonical_entry_surface(state), "")
    state.execution_result = ExecutionResult.model_validate(latest_execution_result)
    state.preflight_result = PreflightResult.model_validate(latest_preflight) if latest_preflight else None
    state.experiment_results = latest_experiment_results
    state.generated_files = sorted(current_project_files.keys())
    state.execution_history = list(materialization["execution_history"])
    state.iteration_count = max(0, int(round_id))
    state.generate_stage_output = build_generate_stage_output(state)
    write_stage_output(state, "project_plan.json", state.project_plan)
    write_stage_output(state, "generation_manifest.json", state.generation_manifest)
    write_stage_output(state, "experiment_output.json", state.generate_stage_output)
    state.generation_checkpoints.extend(list(materialization["generation_checkpoints"]))
    persist_generation_checkpoints(state)
    return {
        "touched_files": list(dict.fromkeys(touched_files)),
        "agent_usage_summary": {
            "calls": sum(int(item.get("calls", 0) or 0) for item in materialization["task_usage_summaries"]),
            "calls_with_usage": sum(int(item.get("calls_with_usage", 0) or 0) for item in materialization["task_usage_summaries"]),
            "input_tokens": sum(int(item.get("input_tokens", 0) or 0) for item in materialization["task_usage_summaries"]),
            "output_tokens": sum(int(item.get("output_tokens", 0) or 0) for item in materialization["task_usage_summaries"]),
            "total_tokens": sum(int(item.get("total_tokens", 0) or 0) for item in materialization["task_usage_summaries"]),
            "estimated_input_tokens": sum(int(item.get("estimated_input_tokens", 0) or 0) for item in materialization["task_usage_summaries"]),
            "estimated_output_tokens": sum(int(item.get("estimated_output_tokens", 0) or 0) for item in materialization["task_usage_summaries"]),
            "estimated_total_tokens": sum(int(item.get("estimated_total_tokens", 0) or 0) for item in materialization["task_usage_summaries"]),
        },
    }


def repair_score_feedback_enabled(workflow_config: Any) -> bool:
    return bool(getattr(workflow_config, "repair_score_feedback_enabled", True))
