"""Runtime project-plan and per-task contract helpers for reproagent."""

from reproagent.pipeline.schemas import (
    ArtifactContract,
    PaperBenchReproState,
    FileSpec,
    GenerateStageOutput,
    GenerationManifest,
    ProjectPlan,
    RepoPlan,
)

from .dataset_manager import _get_dataset_preparation, _get_resource_manifest
from .storage_manager import _load_iteration_checkpoint_payload
from .contract_sanitizer import sanitize_contract_list, sanitize_contract_text, sanitize_scope_boundary, sanitize_task_contract


def _dedupe(items: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for item in items:
        value = str(item or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _looks_like_artifact_path(path: str) -> bool:
    normalized = str(path or "").strip().replace("\\", "/").strip("/")
    if not normalized or normalized in {".", ".."}:
        return False
    lowered = normalized.lower()
    if lowered.startswith(("/", "~")) or ".." in lowered.split("/"):
        return False
    first = lowered.split("/", 1)[0]
    if "/" in lowered or "." in lowered:
        return True
    return first in {
        "results",
        "outputs",
        "artifacts",
        "reports",
        "figures",
        "plots",
        "tables",
        "checkpoints",
        "runs",
        "logs",
    }


def _artifact_paths(paths: list[str]) -> list[str]:
    return _dedupe(
        [
            str(path or "").strip().replace("\\", "/").strip("/")
            for path in paths
            if _looks_like_artifact_path(str(path or ""))
        ]
    )


def _global_contract_artifact_paths(state: PaperBenchReproState) -> list[str]:
    """Extract declared artifact paths from the global contract."""
    if state.global_contract is None:
        return []
    paths: list[str] = []
    for item in state.global_contract.result_targets:
        for path in item.artifact_paths:
            normalized = str(path).strip()
            if _looks_like_artifact_path(normalized) and normalized not in paths:
                paths.append(normalized)
    return paths


def _global_contract_metrics_path(state: PaperBenchReproState) -> str:
    """Resolve the preferred metrics path from the global contract."""
    for path in _global_contract_artifact_paths(state):
        if path.endswith("metrics.json"):
            return path
    return "results/metrics.json"


def _build_runtime_project_plan(state: PaperBenchReproState) -> ProjectPlan:
    """Build the generate-stage runtime contract directly from architecture and task artifacts."""
    if state.repo_plan is not None:
        return _build_runtime_project_plan_from_repo_plan(state.repo_plan, state)
    raise ValueError("runtime project plan requires repo_plan; task-list-only fallback has been removed")


def _build_runtime_project_plan_from_repo_plan(repo_plan: RepoPlan, state: PaperBenchReproState) -> ProjectPlan:
    """Build runtime contract from the unified repo_plan."""
    main_path = (
        str(repo_plan.canonical_route.entry_surface or "").strip()
        or (repo_plan.entrypoints[0] if repo_plan.entrypoints else "")
        or "main.py"
    )
    file_specs = [
        FileSpec(
            path=item.target_file,
            purpose=item.purpose or f"Implement {item.target_file}",
            dependencies=list(item.depends_on),
            required=True,
        )
        for item in repo_plan.files
    ]
    dataset_preparation = _get_dataset_preparation(state)
    resource_manifest = _get_resource_manifest(state)
    metrics_path = _global_contract_metrics_path(state)
    required_artifacts = _artifact_paths([*list(repo_plan.artifact_paths), *list(repo_plan.canonical_route.expected_outputs)])
    optional_artifacts = [path for path in required_artifacts if path != metrics_path] or ["experiment_summary.json"]
    return ProjectPlan(
        project_type="single_experiment_project",
        summary=repo_plan.summary or state.input.target[:500],
        entrypoints={
            "main": main_path,
            "runtime_smoke": f"python {main_path} --mode runtime_smoke",
            "docker_validate": f"python {main_path} --mode docker_validate",
        },
        runtime_contract={
            "output_dir": "results",
            "artifacts_dir": "results",
            "notes": list((repo_plan.structure_decisions or repo_plan.architecture.rationale.splitlines())[:8]),
            "dataset_root": dataset_preparation.get("download_root", ""),
            "requested_datasets": dataset_preparation.get("requested_datasets", []),
            "prepared_datasets": dataset_preparation.get("downloaded_datasets", []),
            "resource_manifest": resource_manifest,
            "topic_profile": repo_plan.topic_profile,
            "global_contract": repo_plan.global_contract,
            "canonical_route": repo_plan.canonical_route.model_dump(mode="json"),
            "stage_public_surfaces": [item.model_dump(mode="json") for item in repo_plan.stage_public_surfaces],
            "artifact_contract": [
                item.model_dump(mode="json")
                for item in repo_plan.artifact_contract
                if _looks_like_artifact_path(item.relative_path)
            ],
            "declared_artifact_paths": required_artifacts,
            "result_targets": list(repo_plan.global_contract.get("result_targets", [])) if isinstance(repo_plan.global_contract, dict) else [],
            "generation_tasks": [
                {
                    "task_id": item.task_id or item.target_file,
                    "file_path": item.target_file,
                    "work_package_id": item.work_package_id,
                    "purpose": item.purpose,
                    "depends_on": list(item.depends_on),
                    "blocking_dependencies": list(item.blocking_dependencies),
                    "interface_contract": list(item.interface_contract),
                    "implementation_surfaces": list(item.implementation_surfaces),
                    "method_obligations": list(item.method_obligations),
                    "defines_symbols": list(item.defines_symbols),
                    "calls_symbols": list(item.calls_symbols),
                    "writes_artifacts": list(item.writes_artifacts),
                    "reads_artifacts": list(item.reads_artifacts),
                    "context_sources": list(item.context_sources),
                    "allowed_scope": dict(item.allowed_scope),
                    "review_points": list(item.review_points),
                    "generation_prompt": item.generation_prompt,
                    "hypothesis": item.hypothesis,
                    "decision_value": item.decision_value,
                    "scope_boundary": dict(getattr(item, "scope_boundary", {}) or {}),
                }
                for item in repo_plan.files
            ],
            "repo_plan_files": [item.model_dump(mode="json") for item in repo_plan.files],
        },
        file_specs=file_specs,
        artifact_contract=ArtifactContract(
            metrics_path=metrics_path,
            required_files=[metrics_path],
            optional_files=optional_artifacts,
        ),
    )

def _runtime_contract_payload(project_plan: ProjectPlan) -> dict:
    """Return the normalized runtime contract payload."""
    return project_plan.runtime_contract if isinstance(project_plan.runtime_contract, dict) else {}


def _runtime_generation_tasks(project_plan: ProjectPlan) -> list[dict]:
    """Return normalized generation-task records from the runtime contract."""
    runtime_contract = _runtime_contract_payload(project_plan)
    tasks = runtime_contract.get("generation_tasks", [])
    return [item for item in tasks if isinstance(item, dict)]


def _runtime_repo_plan_files(project_plan: ProjectPlan) -> list[dict]:
    """Return normalized repo-plan file records from the runtime contract."""
    runtime_contract = _runtime_contract_payload(project_plan)
    repo_plan_files = runtime_contract.get("repo_plan_files", [])
    return [item for item in repo_plan_files if isinstance(item, dict)]


def _runtime_canonical_entry_surface(project_plan: ProjectPlan) -> str:
    canonical_route = dict(_runtime_contract_payload(project_plan).get("canonical_route", {}) or {})
    candidate = str(canonical_route.get("entry_surface", "") or "").strip()
    if candidate:
        return candidate
    fallback = str(project_plan.entrypoints.get("main", "") or "").strip()
    return fallback or "main.py"


def _path_generation_priority(path: str) -> tuple[int, str]:
    normalized = str(path or "").strip().replace("\\", "/").lower()
    basename = normalized.rsplit("/", 1)[-1]
    if basename in {"__init__.py", "pyproject.toml", "requirements.txt", "setup.py", "setup.cfg"}:
        return (0, normalized)
    if normalized.startswith(("configs/", "config/")) or normalized.endswith((".yaml", ".yml", ".toml", ".json")):
        return (1, normalized)
    if any(token in normalized for token in ("data", "dataset", "environment", "env", "config", "constant", "registry", "schema", "types")):
        return (2, normalized)
    if any(token in normalized for token in ("method", "model", "policy", "agent", "baseline", "refinement", "mask", "score", "sampler", "optimizer")):
        return (3, normalized)
    if any(token in normalized for token in ("metric", "evaluate", "evaluation", "artifact", "report", "plot", "figure", "table")):
        return (4, normalized)
    if any(token in normalized for token in ("train", "training", "experiment", "runner", "run_")):
        return (5, normalized)
    if basename in {"main.py", "cli.py", "run.py", "run_experiments.py"} or normalized.endswith(("/main.py", "/cli.py")):
        return (8, normalized)
    if normalized.endswith(".md") or normalized.startswith("tests/"):
        return (9, normalized)
    return (6, normalized)


def _ordered_runtime_task_ids(project_plan: ProjectPlan, generation_manifest: GenerationManifest | None = None) -> list[str]:
    """Return repo-plan-first ordered task ids for generation or repair execution."""
    ordered: list[str] = []
    for item in _runtime_repo_plan_files(project_plan):
        task_id = str(item.get("task_id") or item.get("target_file") or "").strip()
        if task_id and task_id not in ordered:
            ordered.append(task_id)
    if ordered:
        path_by_id = {
            str(item.get("task_id") or item.get("target_file") or "").strip(): str(item.get("target_file") or "").strip()
            for item in _runtime_repo_plan_files(project_plan)
            if str(item.get("task_id") or item.get("target_file") or "").strip()
        }
        original_index = {task_id: index for index, task_id in enumerate(ordered)}
        return sorted(
            ordered,
            key=lambda task_id: (
                _path_generation_priority(path_by_id.get(task_id, "")),
                original_index.get(task_id, 9999),
            ),
        )

    if generation_manifest is not None:
        ordered = [task_id for task_id in generation_manifest.ordered_tasks if str(task_id).strip()]
        path_by_id = {
            str(item.task_id or "").strip(): str(item.file_path or "").strip()
            for item in list(generation_manifest.task_inputs or [])
            if str(item.task_id or "").strip()
        }
        original_index = {task_id: index for index, task_id in enumerate(ordered)}
        return sorted(
            ordered,
            key=lambda task_id: (
                _path_generation_priority(path_by_id.get(task_id, "")),
                original_index.get(task_id, 9999),
            ),
        )
    return []


def _build_runtime_task_views(project_plan: ProjectPlan, generation_manifest: GenerationManifest | None = None) -> list[dict]:
    """Build repo-plan-first per-task runtime views."""
    repo_plan_files = _runtime_repo_plan_files(project_plan)
    file_plan_by_task_id = {
        str(item.get("task_id") or item.get("target_file") or "").strip(): item
        for item in repo_plan_files
        if str(item.get("task_id") or item.get("target_file") or "").strip()
    }
    file_plan_by_path = {
        str(item.get("target_file") or "").strip(): item
        for item in repo_plan_files
        if str(item.get("target_file") or "").strip()
    }
    stage_public_surfaces = [
        item for item in list(_runtime_contract_payload(project_plan).get("stage_public_surfaces", []) or [])
        if isinstance(item, dict)
    ]
    canonical_route = dict(_runtime_contract_payload(project_plan).get("canonical_route", {}) or {})
    entrypoint_paths = _dedupe(
        [
            str(canonical_route.get("entry_surface", "") or "").strip(),
            *[
                str(item.get("path") or "").strip()
                for item in stage_public_surfaces
                if str(item.get("surface_kind") or "").strip() == "entrypoint"
            ],
        ]
    )
    manifest_input_by_task_id = {
        item.task_id: item.model_dump(mode="json")
        for item in generation_manifest.task_inputs
    } if generation_manifest is not None else {}

    ordered_ids = _ordered_runtime_task_ids(project_plan, generation_manifest)
    views: list[dict] = []
    for task_id in ordered_ids:
        file_plan = file_plan_by_task_id.get(task_id, {})
        manifest_input = manifest_input_by_task_id.get(task_id, {})
        file_path = (
            str(file_plan.get("target_file") or manifest_input.get("file_path") or "").strip()
        )
        if not file_path:
            continue

        review_points = list(
            dict.fromkeys(
                [
                    *list(file_plan.get("review_points", [])),
                    *list(manifest_input.get("review_points", [])),
                ]
            )
        )
        dependency_files = list(
            dict.fromkeys(
                [
                    *list(file_plan.get("depends_on", [])),
                    *list(manifest_input.get("dependency_files", [])),
                ]
            )
        )
        related_node_ids = list(
            dict.fromkeys(
                [
                    *list(file_plan.get("related_node_ids", [])),
                    *list(manifest_input.get("related_node_ids", [])),
                ]
            )
        )
        reference_ids = list(
            dict.fromkeys(
                [
                    *list(file_plan.get("reference_ids", [])),
                    *list(manifest_input.get("reference_ids", [])),
                ]
            )
        )
        current_work_package_id = str(file_plan.get("work_package_id") or "").strip()
        current_work_package_contract = next(
            (
                item for item in list(dict(_runtime_contract_payload(project_plan).get("global_contract", {}) or {}).get("work_package_contracts", []) or [])
                if isinstance(item, dict) and str(item.get("work_package_id") or "").strip() == current_work_package_id
            ),
            {},
        )
        current_work_package_inventories = dict(current_work_package_contract.get("inventories", {}) or {})
        implementation_surfaces = _dedupe(
            list(file_plan.get("implementation_surfaces", []) or [])
            + list(manifest_input.get("implementation_surfaces", []) or [])
            + list(current_work_package_inventories.get("implementation_surface_inventory", []) or [])
        )
        method_obligations = sanitize_contract_list(
            _dedupe(
                list(file_plan.get("method_obligations", []) or [])
                + list(manifest_input.get("method_obligations", []) or [])
                + list(current_work_package_contract.get("method_obligations", []) or [])
            ),
            field="method_obligations",
        )
        interface_contract = sanitize_contract_list(
            _dedupe(
                list(file_plan.get("interface_contract", []) or [])
                + list(manifest_input.get("interface_contract", []) or [])
                + list(current_work_package_contract.get("interface_contract", []) or [])
            ),
            field="interface_contract",
        )
        defines_symbols = _dedupe(
            list(file_plan.get("defines_symbols", []) or [])
            + list(manifest_input.get("defines_symbols", []) or [])
        )
        calls_symbols = _dedupe(
            list(file_plan.get("calls_symbols", []) or [])
            + list(manifest_input.get("calls_symbols", []) or [])
        )
        work_package_file_plans = [
            item for item in repo_plan_files
            if isinstance(item, dict) and str(item.get("work_package_id") or "").strip() == current_work_package_id
        ] if current_work_package_id else []
        work_package_files = [
            str(item.get("target_file") or "").strip()
            for item in work_package_file_plans
            if str(item.get("target_file") or "").strip()
        ]
        work_package_dependency_files = list(
            dict.fromkeys(
                path
                for item in work_package_file_plans
                for path in [
                    *list(item.get("depends_on", []) or []),
                    *list(item.get("blocking_dependencies", []) or []),
                ]
                if str(path).strip() and str(path).strip() in file_plan_by_path
            )
        )
        work_package_required_files = list(
            dict.fromkeys([*work_package_files, *work_package_dependency_files])
        )
        smoke_surface_candidates = [
            path for path in work_package_files
            if path in entrypoint_paths
        ]
        smoke_surface_candidates.extend(
            str(item.get("target_file") or "").strip()
            for item in work_package_file_plans
            if list(item.get("writes_artifacts", []) or [])
        )
        smoke_surface_candidates.extend(
            str(item.get("path") or "").strip()
            for item in stage_public_surfaces
            if str(item.get("path") or "").strip() in work_package_files
        )
        smoke_surface_file = next(
            (
                path for path in dict.fromkeys(smoke_surface_candidates)
                if path
            ),
            "",
        )
        smoke_mode = ""
        smoke_command = ""
        if smoke_surface_file and smoke_surface_file in entrypoint_paths:
            smoke_mode = "command"
            smoke_command = (
                str(project_plan.entrypoints.get("runtime_smoke", "")).strip()
                or f"python {_runtime_canonical_entry_surface(project_plan)}"
            )
        elif smoke_surface_file.endswith(".py"):
            smoke_mode = "import"

        task_input = sanitize_task_contract({
            "task_id": task_id,
            "file_path": file_path,
            "dependency_files": dependency_files,
            "related_node_ids": related_node_ids,
            "reference_ids": reference_ids,
            "snippet_candidates": list(manifest_input.get("snippet_candidates", [])),
            "allowed_scope": dict(
                file_plan.get("allowed_scope")
                or manifest_input.get("allowed_scope", {})
            ),
            "scope_boundary": sanitize_scope_boundary(
                file_plan.get("scope_boundary")
                or manifest_input.get("scope_boundary", {})
                or current_work_package_contract.get("scope_boundary", {})
            ),
            "review_points": sanitize_contract_list(review_points, field="review_points"),
            "work_package_id": current_work_package_id,
            "interface_contract": interface_contract,
            "implementation_surfaces": implementation_surfaces,
            "context_sources": _dedupe(
                list(file_plan.get("context_sources", []) or [])
                + list(manifest_input.get("context_sources", []) or [])
            ),
            "defines_symbols": defines_symbols,
            "calls_symbols": calls_symbols,
            "method_obligations": method_obligations,
            "generation_prompt": sanitize_contract_text("\n".join(
                part
                for part in [
                    str(file_plan.get("generation_prompt", "") or "").strip(),
                    str(manifest_input.get("generation_prompt", "") or "").strip(),
                ]
                if part
            )),
            "evidence_summary": list(current_work_package_contract.get("evidence_summary", []) or []),
            "writes_artifacts": _dedupe(
                list(file_plan.get("writes_artifacts", []) or [])
                + list(manifest_input.get("writes_artifacts", []) or [])
            ),
            "reads_artifacts": _dedupe(
                list(file_plan.get("reads_artifacts", []) or [])
                + list(manifest_input.get("reads_artifacts", []) or [])
            ),
            "work_package_files": work_package_files,
            "work_package_dependency_files": work_package_dependency_files,
            "work_package_required_files": work_package_required_files,
            "work_package_smoke": {
                "mode": smoke_mode,
                "command": smoke_command,
                "surface_file": smoke_surface_file,
                "timeout_seconds": 10,
            },
            "canonical_route": canonical_route,
            "stage_public_surfaces": stage_public_surfaces,
            "artifact_contract": list(_runtime_contract_payload(project_plan).get("artifact_contract", []) or []),
            "paper_claim_inventory": dict(manifest_input.get("paper_claim_inventory", {}) or {}),
            "paper_claim_closure_items": list(manifest_input.get("paper_claim_closure_items", []) or []),
            "paper_claim_closure_rules": list(manifest_input.get("paper_claim_closure_rules", []) or []),
            "paper_evidence_contract": dict(manifest_input.get("paper_evidence_contract", {}) or {}),
            "prepare_quality_gate_summary": dict(manifest_input.get("prepare_quality_gate_summary", {}) or {}),
            "generation_context": dict(manifest_input.get("generation_context", {}) or {}),
            "critical_grounding_warning": bool(
                current_work_package_contract
                and str(current_work_package_contract.get("grounding_status", "") or "").strip().lower()
                not in {"", "grounded", "self_contained"}
            ),
        })
        task_input["review_points"] = sanitize_contract_list(
            list(task_input.get("review_points", []) or []),
            field="review_points",
        )

        views.append(
            {
                "task_id": task_id,
                "task_input": task_input,
                "task_manifest": {
                    "ordered_tasks": [task_id],
                    "task_inputs": [task_input],
                    "current_task_input": task_input,
                    "review_points": review_points,
                    "topological_order": [file_path],
                },
            }
        )
    return views


def _build_task_project_plan(project_plan: ProjectPlan, task_input: dict, existing_files: dict[str, str]) -> dict:
    """Build a narrow project contract for one ordered task generation step."""
    keep_paths = {task_input.get("file_path", "")}
    keep_paths.update(task_input.get("dependency_files", []))
    runtime_contract = _runtime_contract_payload(project_plan)
    repo_plan_files = _runtime_repo_plan_files(project_plan)
    task_id = str(task_input.get("task_id", "")).strip()
    current_file_plan = next(
        (
            item for item in repo_plan_files
            if (task_id and item.get("task_id") == task_id)
            or item.get("target_file") == task_input.get("file_path", "")
        ),
        {},
    )
    current_work_package_id = current_file_plan.get("work_package_id", "")
    if current_work_package_id:
        keep_paths.update(
            item.get("target_file", "")
            for item in repo_plan_files
            if isinstance(item, dict) and item.get("work_package_id") == current_work_package_id
        )
    canonical_entry = _runtime_canonical_entry_surface(project_plan)
    if canonical_entry and (
        canonical_entry in existing_files
        or canonical_entry == task_input.get("file_path", "")
    ):
        keep_paths.add(canonical_entry)
    keep_paths.update(path for path in existing_files if path in {"README.md", "requirements.txt", "pyproject.toml"})

    file_specs = [
        spec.model_dump(mode="json")
        for spec in project_plan.file_specs
        if spec.path in keep_paths
    ]
    return {
        "project_type": project_plan.project_type,
        "summary": project_plan.summary,
        "entrypoints": project_plan.entrypoints,
        "runtime_contract": {
            **runtime_contract,
            "current_task_file_plan": current_file_plan,
            "current_task_work_package_id": current_work_package_id,
            "current_task_neighbor_file_plans": [
                item
                for item in repo_plan_files
                if isinstance(item, dict)
                and item.get("work_package_id") == current_work_package_id
                and item.get("target_file") != task_input.get("file_path", "")
            ][:6],
        },
        "file_specs": file_specs,
        "artifact_contract": project_plan.artifact_contract.model_dump(mode="json"),
    }

def _filter_task_generated_files(project_files: dict[str, str], task_input: dict) -> dict[str, str]:
    """Keep only files that belong to the current task update surface."""
    allowed_paths = {task_input.get("file_path", "")}
    return {
        path: content
        for path, content in project_files.items()
        if path in allowed_paths
    }

def _build_generate_stage_output(state: PaperBenchReproState) -> GenerateStageOutput:
    """Build the structured generate-stage output for downstream consumers."""
    evaluation_payload = state.evaluation.model_dump(mode="json") if state.evaluation else {}
    checks_passed = not state.execution_result or all(check.get("passed", False) for check in state.execution_result.checks)
    checkpoint_payload = _load_iteration_checkpoint_payload(state)
    experiment_status = "task_review_failed"
    if state.generated_files and checks_passed and (state.execution_result is None or state.execution_result.success):
        experiment_status = "task_review_passed"
    elif state.generated_files:
        experiment_status = "code_generated"
    quality_level = "generated_only"
    if checks_passed:
        quality_level = "task_review_passed"
    iteration_state = {
        "iteration_count": state.iteration_count,
        "evaluation": evaluation_payload,
        "execution_history_count": len(state.execution_history),
    }
    return GenerateStageOutput(
        project_plan=state.project_plan.model_dump(mode="json"),
        generation_manifest=state.generation_manifest.model_dump(mode="json") if state.generation_manifest else {},
        topic_profile=state.topic_profile.model_dump(mode="json") if state.topic_profile else {},
        global_contract=state.global_contract.model_dump(mode="json") if state.global_contract else {},
        generated_files=list(state.generated_files),
        file_count=len(state.generated_files),
        iteration_checkpoint=checkpoint_payload,
        experiment_status=experiment_status,
        validation_summary={
            "task_review_gate": "passed" if checks_passed else "failed",
            "runtime_validation": "deferred_to_downstream",
        },
        quality_level=quality_level,
        iteration_state=iteration_state,
        checkpoint_path=state.checkpoint_path,
    )
