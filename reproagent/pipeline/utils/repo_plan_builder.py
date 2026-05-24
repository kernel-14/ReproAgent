"""Repo-plan construction helpers for reproagent."""

from reproagent.pipeline.schemas import (
    PaperBenchReproState,
    RepoArtifactContract,
    RepoCanonicalRoute,
    RepoFilePlan,
    RepoPlan,
    RepoStagePublicSurface,
)
from reproagent.pipeline.utils.contract_sanitizer import sanitize_contract_text, sanitize_scope_boundary


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


def _normalize_repo_path(path: str) -> str:
    normalized = str(path or "").strip().replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    while "//" in normalized:
        normalized = normalized.replace("//", "/")
    return normalized.strip("/")


def _looks_like_artifact_path(path: str) -> bool:
    """Keep result artifacts as paths, not prose labels such as `metric_formula`."""
    normalized = _normalize_repo_path(path)
    if not normalized or normalized in {".", ".."}:
        return False
    lowered = normalized.lower()
    if lowered.startswith(("/", "~")) or ".." in lowered.split("/"):
        return False
    first = lowered.split("/", 1)[0]
    if "/" in lowered:
        return True
    if "." in lowered:
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


def _artifact_path_candidates(paths: list[str]) -> list[str]:
    return _dedupe([_normalize_repo_path(path) for path in paths if _looks_like_artifact_path(path)])


def _canonical_registered_paths(state: PaperBenchReproState) -> list[str]:
    if state.canonical_ir is None:
        raise ValueError("repo_plan requires canonical_ir")
    return [
        _normalize_repo_path(path)
        for path in list(state.canonical_ir.validation_index.get("registered_paths", []) or [])
        if _normalize_repo_path(path)
    ]


def _canonical_file_node_by_path(state: PaperBenchReproState) -> dict[str, object]:
    if state.canonical_ir is None:
        raise ValueError("repo_plan requires canonical_ir")
    return {
        _normalize_repo_path(item.canonical_path): item
        for item in state.canonical_ir.file_nodes
        if _normalize_repo_path(item.canonical_path)
    }


def _canonical_surface_nodes_by_kind(state: PaperBenchReproState, surface_kind: str) -> list[object]:
    if state.canonical_ir is None:
        raise ValueError("repo_plan requires canonical_ir")
    return [
        item
        for item in state.canonical_ir.surface_nodes
        if str(item.surface_kind or "").strip() == surface_kind
        and _normalize_repo_path(item.canonical_path)
    ]


def _file_plan_by_path(state: PaperBenchReproState) -> dict[str, RepoFilePlan]:
    if state.package_file_planning_output is None:
        raise ValueError("repo_plan requires package_file_planning_output")
    return {
        _normalize_repo_path(item.target_file): item
        for item in state.package_file_planning_output.file_plans
        if _normalize_repo_path(item.target_file)
    }


def _ordered_registered_paths(state: PaperBenchReproState) -> list[str]:
    registered_paths = _canonical_registered_paths(state)
    architecture_paths = [
        _normalize_repo_path(path)
        for path in list(state.architecture.target_file_tree if state.architecture is not None else [])
        if _normalize_repo_path(path)
    ]
    file_plan_order = [
        _normalize_repo_path(item.target_file)
        for item in state.package_file_planning_output.file_plans
        if _normalize_repo_path(item.target_file)
    ] if state.package_file_planning_output is not None else []
    entrypoints = _dedupe(
        [
            _normalize_repo_path(item.canonical_path)
            for item in _canonical_surface_nodes_by_kind(state, "entrypoint")
            if _normalize_repo_path(item.canonical_path) in set(registered_paths + architecture_paths)
        ]
    )
    ordered: list[str] = []
    seen: set[str] = set()
    allowed_paths = set(registered_paths + architecture_paths)
    raw_order: list[str] = []
    for path in [*file_plan_order, *architecture_paths, *registered_paths, *entrypoints]:
        if path not in allowed_paths or path in seen:
            continue
        seen.add(path)
        raw_order.append(path)
    original_index = {path: index for index, path in enumerate(raw_order)}
    ordered.extend(
        sorted(
            raw_order,
            key=lambda path: (_path_generation_priority(path), original_index.get(path, 9999)),
        )
    )
    return ordered


def _work_package_by_id(state: PaperBenchReproState) -> dict[str, object]:
    if state.work_package_planning is None:
        raise ValueError("repo_plan requires work_package_planning")
    return {
        str(item.work_package_id or "").strip(): item
        for item in state.work_package_planning.work_packages
        if str(item.work_package_id or "").strip()
    }


def _canonical_work_package_ids(state: PaperBenchReproState) -> list[str]:
    return _dedupe(
        [
            str(item.work_package_id or "").strip()
            for item in list(state.canonical_ir.work_packages if state.canonical_ir else [])
            if str(item.work_package_id or "").strip()
        ]
    )


def _architecture_blueprint_by_path(state: PaperBenchReproState) -> dict[str, object]:
    if state.architecture is None:
        return {}
    return {
        _normalize_repo_path(item.path): item
        for item in state.architecture.file_blueprints
        if _normalize_repo_path(item.path)
    }


def _artifact_paths_for_file(state: PaperBenchReproState, file_path: str) -> list[str]:
    artifact_surfaces = _canonical_surface_nodes_by_kind(state, "artifact")
    file_node_by_path = _canonical_file_node_by_path(state)
    file_node = file_node_by_path.get(_normalize_repo_path(file_path))
    if file_node is None:
        return []
    owner_file_id = str(file_node.file_id or "").strip()
    owner_work_package_id = str(file_node.owner_work_package_id or "").strip()
    return _artifact_path_candidates(
        [
            _normalize_repo_path(item.canonical_path)
            for item in artifact_surfaces
            if _normalize_repo_path(item.canonical_path)
            and (
                str(item.owner_file_id or "").strip() == owner_file_id
                or (
                    owner_work_package_id
                    and str(item.owner_work_package_id or "").strip() == owner_work_package_id
                )
            )
        ]
    )


def _bench_visible_contract_obligations(file_path: str) -> list[str]:
    """Generic benchmark-facing implementation obligations for generated files."""
    lowered = _normalize_repo_path(file_path).lower()
    obligations: list[str] = []
    if lowered == "readme.md":
        obligations.append(
            "Document the canonical run path, environment/readiness expectations, configuration surfaces, named protocol coverage, and declared output artifacts."
        )
    if "requirement" in lowered or lowered in {"requirements.txt", "pyproject.toml", "environment.yml"}:
        obligations.append(
            "Declare lightweight core dependencies separately from optional heavy simulator/training dependencies when possible."
        )
    if any(token in lowered for token in ("config", "default", ".yaml", ".yml", ".json", "main.py", "experiment")):
        obligations.append(
            "Expose a paper-derived configuration contract with task/environment entries, method/baseline selectors, seeds, hyperparameters, output paths, and active reproduction scope notes."
        )
    if any(token in lowered for token in ("main.py", "cli", "run", "experiment", "train", "evaluation", "report", "artifact")):
        obligations.append(
            "Provide a dry-run or runtime-smoke mode that validates configuration and writes auxiliary readiness/manifest artifacts, while paper-visible tables, figures, metrics, predictions, and reports are backed by measured implementation routes."
        )
    if any(token in lowered for token in ("experiment", "evaluation", "report", "artifact", "main.py")):
        obligations.append(
            "Make named experiment protocols statically discoverable as registries or matrices connecting tasks, methods, measurements, and artifact paths."
        )
    if any(token in lowered for token in ("artifact", "report", "evaluation", "main.py")):
        obligations.append(
            "Expose artifact layout helpers or constants for metrics, tables, figures, config snapshots, run manifests, and reports so static review can find output contracts."
        )
    if any(token in lowered for token in ("environment", "data", "task", "main.py")):
        obligations.append(
            "Represent external environments or datasets through import-light descriptors/factories with clear availability checks and faithful fallback errors."
        )
    return obligations


def _surface_kind_by_path(state: PaperBenchReproState) -> dict[str, list[str]]:
    if state.canonical_ir is None:
        raise ValueError("repo_plan requires canonical_ir")
    mapping: dict[str, list[str]] = {}
    for item in state.canonical_ir.surface_nodes:
        path = _normalize_repo_path(item.canonical_path)
        kind = str(item.surface_kind or "").strip()
        if not path or not kind:
            continue
        mapping.setdefault(path, [])
        if kind not in mapping[path]:
            mapping[path].append(kind)
    return mapping


def _producer_surface_by_owner(state: PaperBenchReproState) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in _canonical_surface_nodes_by_kind(state, "producer"):
        owner = str(item.owner_work_package_id or "").strip()
        path = _normalize_repo_path(item.canonical_path)
        if owner and path and owner not in mapping:
            mapping[owner] = path
    return mapping


def _record_degraded_repo_plan_issue(state: PaperBenchReproState, *, code: str, message: str, paths: list[str] | None = None) -> None:
    payload = {
        "stage": "repo_plan",
        "code": str(code or "").strip(),
        "message": str(message or "").strip(),
        "paths": [item for item in list(paths or []) if str(item).strip()],
    }
    backlog = state.temp_data.setdefault("degraded_backlog", [])
    if isinstance(backlog, list) and payload not in backlog:
        backlog.append(payload)


def _fallback_file_plan_for_path(state: PaperBenchReproState, path: str) -> RepoFilePlan:
    normalized_path = _normalize_repo_path(path)
    blueprint = _architecture_blueprint_by_path(state).get(normalized_path)
    file_node = _canonical_file_node_by_path(state).get(normalized_path)
    owner_notes: list[str] = []
    work_package_id = _repo_plan_owner_fallback(
        state,
        normalized_path,
        file_node=file_node,
        file_plan=RepoFilePlan(target_file=normalized_path),
        canonical_work_package_ids=_canonical_work_package_ids(state),
        owner_notes=owner_notes,
    )
    purpose = (
        str(getattr(blueprint, "purpose", "") or "").strip()
        or f"Implement {normalized_path}."
    )
    writes_artifacts = _artifact_paths_for_file(state, normalized_path)
    validation_hooks = ["python_syntax"] if normalized_path.endswith(".py") else ["file_exists"]
    _record_degraded_repo_plan_issue(
        state,
        code="missing_file_plan_projected",
        message=f"projected fallback file plan for `{normalized_path}` because package file planning coverage was incomplete",
        paths=[normalized_path],
    )
    return RepoFilePlan(
        target_file=normalized_path,
        task_id=normalized_path,
        work_package_id=work_package_id,
        purpose=purpose,
        related_node_ids=list(getattr(file_node, "related_plan_node_ids", []) or []),
        interface_contract=[],
        context_sources=[f"node:{item}" for item in list(getattr(file_node, "related_plan_node_ids", []) or [])],
        writes_artifacts=writes_artifacts,
        validation_hooks=validation_hooks,
        review_points=["projected fallback file plan"],
    )


def _required_package_file_plan_paths(state: PaperBenchReproState) -> list[str]:
    if state.package_file_planning_output is None:
        return []
    return _dedupe(
        [
            _normalize_repo_path(item.target_file)
            for item in list(state.package_file_planning_output.file_plans or [])
            if _normalize_repo_path(item.target_file)
        ]
    )


def _fallback_entrypoints(state: PaperBenchReproState, file_plan_by_path: dict[str, RepoFilePlan]) -> list[str]:
    ranked: list[str] = []
    candidates = [
        *list(file_plan_by_path),
        *[
            _normalize_repo_path(item.path)
            for item in list(state.architecture.file_blueprints if state.architecture is not None else [])
            if _normalize_repo_path(item.path)
        ],
    ]
    for path in candidates:
        normalized = _normalize_repo_path(path)
        if not normalized or normalized in ranked:
            continue
        lower = normalized.lower()
        if lower == "main.py":
            ranked.insert(0, normalized)
            continue
        if lower.endswith("/main.py") or lower.endswith("main.py"):
            ranked.append(normalized)
            continue
        if any(token in lower for token in ("run", "train", "eval", "cli")) and normalized.endswith(".py"):
            ranked.append(normalized)
    return ranked[:3]


def _path_generation_priority(path: str) -> tuple[int, str]:
    """Order dependency/helper files before route and entry files."""
    normalized = _normalize_repo_path(path).lower()
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


def _ordered_repo_work_packages(state: PaperBenchReproState) -> list[object]:
    canonical_work_package_ids = [
        str(item.work_package_id or "").strip()
        for item in list(state.canonical_ir.work_packages if state.canonical_ir else [])
        if str(item.work_package_id or "").strip()
    ]
    work_package_by_id = _work_package_by_id(state)
    ordered = [work_package_by_id[item] for item in canonical_work_package_ids if item in work_package_by_id]
    return ordered or list(work_package_by_id.values())


def _repo_plan_owner_fallback(
    state: PaperBenchReproState,
    path: str,
    *,
    file_node: object,
    file_plan: RepoFilePlan,
    canonical_work_package_ids: list[str],
    owner_notes: list[str],
) -> str:
    canonical_owners = set(canonical_work_package_ids)
    canonical_owner = str(getattr(file_node, "owner_work_package_id", "") or "").strip()
    file_plan_owner = str(file_plan.work_package_id or "").strip()
    if canonical_owner in canonical_owners:
        if file_plan_owner and file_plan_owner != canonical_owner:
            owner_notes.append(
                f"file plan owner `{file_plan_owner}` for `{path}` was ignored; canonical owner `{canonical_owner}` is authoritative"
            )
        return canonical_owner

    if canonical_owner:
        owner_notes.append(
            f"canonical owner `{canonical_owner}` for `{path}` is not a registered canonical work package"
        )
    else:
        owner_notes.append(f"canonical owner missing for `{path}`")

    if file_plan_owner in canonical_owners:
        owner_notes.append(
            f"using file plan owner `{file_plan_owner}` for `{path}` because canonical file node owner was unresolved"
        )
        return file_plan_owner
    if file_plan_owner:
        owner_notes.append(
            f"file plan owner `{file_plan_owner}` for `{path}` is not a registered canonical work package"
        )

    if state.architecture is not None:
        for work_package_id, paths in dict(state.architecture.package_layout or {}).items():
            normalized_owner = str(work_package_id or "").strip()
            if normalized_owner not in canonical_owners:
                continue
            if path in {
                _normalize_repo_path(item)
                for item in list(paths or [])
                if _normalize_repo_path(item)
            }:
                owner_notes.append(
                    f"using architecture provenance owner `{normalized_owner}` for `{path}` within canonical work package set"
                )
                return normalized_owner

    message = f"canonical registered path `{path}` has no available canonical work package owner"
    owner_notes.append(message)
    _record_degraded_repo_plan_issue(
        state,
        code="unresolved_canonical_file_owner",
        message=message,
        paths=[path],
    )
    raise ValueError(message)


def _build_repo_plan(state: PaperBenchReproState) -> RepoPlan:
    """Build canonical-only repo_plan consumed downstream by generation, validation, and repair."""
    if (
        state.architecture is None
        or state.package_file_planning_output is None
        or state.work_package_planning is None
        or state.canonical_ir is None
    ):
        raise ValueError(
            "repo_plan requires architecture, package-scoped file plans, work package planning, and canonical_ir"
        )

    registered_paths = _ordered_registered_paths(state)
    file_node_by_path = _canonical_file_node_by_path(state)
    file_plan_by_path = _file_plan_by_path(state)
    blueprint_by_path = _architecture_blueprint_by_path(state)
    work_package_by_id = _work_package_by_id(state)
    evidence_bundle_by_package = {item.work_package_id: item for item in state.evidence_bundles}
    surface_kind_by_path = _surface_kind_by_path(state)
    producer_surface_by_owner = _producer_surface_by_owner(state)
    canonical_work_package_ids = _canonical_work_package_ids(state)
    critical_ungrounded_work_packages = {
        str(item.work_package_id or "").strip()
        for item in state.evidence_bundles
        if str(item.grounding_status or "").strip().lower() not in {"grounded", "self_contained"}
    }

    files: list[RepoFilePlan] = []
    structure_decisions: list[str] = []
    owner_notes: list[str] = []
    artifact_paths: list[str] = []

    missing_file_plans = [path for path in registered_paths if path not in file_plan_by_path]
    for path in missing_file_plans:
        file_plan_by_path[path] = _fallback_file_plan_for_path(state, path)

    required_plan_paths = _dedupe([*registered_paths, *_required_package_file_plan_paths(state)])
    missing_from_canonical = [path for path in required_plan_paths if path not in registered_paths]
    if missing_from_canonical:
        message = (
            "repo_plan would drop package file plans that are not registered in canonical_ir: "
            + ", ".join(missing_from_canonical[:16])
        )
        _record_degraded_repo_plan_issue(
            state,
            code="package_file_plan_not_in_canonical_ir",
            message=message,
            paths=missing_from_canonical[:32],
        )
        raise ValueError(message)

    for path in registered_paths:
        file_node = file_node_by_path.get(path)
        if file_node is None:
            _record_degraded_repo_plan_issue(
                state,
                code="missing_file_node_skipped",
                message=f"skipped canonical registered path `{path}` because no canonical file node was available",
                paths=[path],
            )
            raise ValueError(f"canonical registered path `{path}` has no canonical file node")
        file_plan = file_plan_by_path[path]
        blueprint = blueprint_by_path.get(path)
        work_package_id = _repo_plan_owner_fallback(
            state,
            path,
            file_node=file_node,
            file_plan=file_plan,
            canonical_work_package_ids=canonical_work_package_ids,
            owner_notes=owner_notes,
        )
        work_package = work_package_by_id.get(work_package_id)
        evidence_bundle = evidence_bundle_by_package.get(work_package_id)
        writes_artifacts = _artifact_path_candidates(list(file_plan.writes_artifacts) + _artifact_paths_for_file(state, path))
        artifact_paths.extend(item for item in writes_artifacts if item not in artifact_paths)
        validation_hooks = _dedupe(
            list(file_plan.validation_hooks)
            or (["python_syntax"] if path.endswith(".py") else ["file_exists"])
        )
        context_sources = _dedupe(
            list(file_plan.context_sources)
            + [f"node:{item}" for item in list(file_node.related_plan_node_ids or [])]
        )
        purpose = (
            str(file_plan.purpose or "").strip()
            or str(getattr(blueprint, "purpose", "") or "").strip()
            or f"Implement {path}."
        )
        work_package_inventories = dict(getattr(work_package, "inventories", {}) or {}) if work_package is not None else {}
        implementation_surfaces = _dedupe(
            list(file_plan.implementation_surfaces)
            + list(work_package_inventories.get("implementation_surface_inventory", []) or [])
        )
        method_obligations = _dedupe(
            list(file_plan.method_obligations)
            + (list(work_package.method_obligations) if work_package is not None else [])
            + _bench_visible_contract_obligations(path)
        )
        hypothesis = str(getattr(file_plan, "hypothesis", "") or "").strip() or (
            str(getattr(work_package, "hypothesis", "") or "").strip() if work_package is not None else ""
        )
        decision_value = str(getattr(file_plan, "decision_value", "") or "").strip() or (
            str(getattr(work_package, "decision_value", "") or "").strip() if work_package is not None else ""
        )
        stop_rule_or_pruning_rationale = str(
            getattr(file_plan, "stop_rule_or_pruning_rationale", "") or ""
        ).strip() or (
            str(getattr(work_package, "stop_rule_or_pruning_rationale", "") or "").strip()
            if work_package is not None
            else ""
        )
        generation_prompt = "\n".join(
            part
            for part in [
                str(file_plan.generation_prompt or "").strip(),
                f"Hypothesis: {hypothesis}" if hypothesis else "",
                f"Decision value: {decision_value}" if decision_value else "",
                (
                    "Implementation surfaces: " + " | ".join(implementation_surfaces[:8])
                    if implementation_surfaces
                    else ""
                ),
                (
                    "Evidence bundle: " + " | ".join(list(evidence_bundle.context_summary)[:6])
                    if evidence_bundle is not None and evidence_bundle.context_summary
                    else ""
                ),
                (
                    "Method obligations: " + " | ".join(method_obligations[:8])
                    if method_obligations
                    else ""
                ),
                (
                    "Canonical artifacts: " + " | ".join(writes_artifacts[:6])
                    if writes_artifacts
                    else ""
                ),
                (
                    "Critical grounding warning: this work package is still ungrounded, so preserve canonical task alignment strictly and avoid unsupported expansion."
                    if work_package_id in critical_ungrounded_work_packages
                    else ""
                ),
            ]
            if part
        )
        scope_boundary = sanitize_scope_boundary(getattr(file_plan, "scope_boundary", {}) or {})
        if not scope_boundary and work_package is not None:
            scope_boundary = sanitize_scope_boundary(getattr(work_package, "scope_boundary", {}) or {})
        files.append(
            file_plan.model_copy(
                update={
                    "target_file": path,
                    "work_package_id": work_package_id,
                    "purpose": purpose,
                    "hypothesis": hypothesis,
                    "decision_value": decision_value,
                    "stop_rule_or_pruning_rationale": stop_rule_or_pruning_rationale,
                    "scope_boundary": scope_boundary,
                    "related_node_ids": _dedupe(
                        list(file_plan.related_node_ids) + list(file_node.related_plan_node_ids or [])
                    ),
                    "interface_contract": _dedupe(
                        list(file_plan.interface_contract)
                        + (list(work_package.interface_contract) if work_package is not None else [])
                    ),
                    "implementation_surfaces": implementation_surfaces,
                    "method_obligations": method_obligations,
                    "context_sources": context_sources,
                    "writes_artifacts": writes_artifacts,
                    "validation_hooks": validation_hooks,
                    "generation_prompt": sanitize_contract_text(generation_prompt),
                }
            )
        )
        structure_decisions.append(
            f"{path} owned by {work_package_id or 'unassigned'} with canonical surfaces "
            f"{', '.join(surface_kind_by_path.get(path, [])) or 'internal'}"
        )

    structure_decisions.extend(_dedupe(owner_notes))

    entrypoints = _dedupe(
        [
            _normalize_repo_path(item.canonical_path)
            for item in _canonical_surface_nodes_by_kind(state, "entrypoint")
            if _normalize_repo_path(item.canonical_path) in file_plan_by_path
        ]
    )
    if not entrypoints:
        entrypoints = _fallback_entrypoints(state, file_plan_by_path)
        if entrypoints:
            _record_degraded_repo_plan_issue(
                state,
                code="missing_canonical_entrypoint_projected",
                message=f"projected fallback entrypoint `{entrypoints[0]}` because canonical entrypoint surfaces were unresolved",
                paths=[entrypoints[0]],
            )
            structure_decisions.append(
                f"fallback entrypoint `{entrypoints[0]}` was projected because canonical entrypoint surfaces were unresolved"
            )
        else:
            entrypoints = ["main.py"]
            _record_degraded_repo_plan_issue(
                state,
                code="missing_canonical_entrypoint_defaulted",
                message="defaulted repo entrypoint to `main.py` because no canonical or heuristic entrypoint could be projected",
                paths=["main.py"],
            )
            structure_decisions.append(
                "fallback entrypoint `main.py` was defaulted because canonical entrypoint surfaces were unresolved"
            )

    stage_public_surfaces: list[RepoStagePublicSurface] = []
    seen_stage_surfaces: set[tuple[str, str]] = set()
    for file_plan in files:
        for surface_kind in list(surface_kind_by_path.get(file_plan.target_file, []) or []):
            if surface_kind == "artifact":
                continue
            route_kind = "artifact_producer" if surface_kind == "producer" else surface_kind
            key = (file_plan.target_file, route_kind)
            if key in seen_stage_surfaces:
                continue
            seen_stage_surfaces.add(key)
            stage_public_surfaces.append(
                RepoStagePublicSurface(
                    stage_name="generate",
                    path=file_plan.target_file,
                    surface_kind=route_kind,
                    purpose=file_plan.purpose,
                    inputs=list(file_plan.depends_on),
                    outputs=_dedupe([*list(file_plan.writes_artifacts), file_plan.target_file]),
                )
            )
    if entrypoints:
        seen_entrypoint_paths = {
            item.path
            for item in stage_public_surfaces
            if item.surface_kind == "entrypoint"
        }
        for entrypoint in entrypoints:
            if entrypoint in seen_entrypoint_paths:
                continue
            fallback_file_plan = file_plan_by_path.get(entrypoint, RepoFilePlan(target_file=entrypoint))
            stage_public_surfaces.append(
                RepoStagePublicSurface(
                    stage_name="generate",
                    path=entrypoint,
                    surface_kind="entrypoint",
                    purpose=fallback_file_plan.purpose or f"Executable entrypoint for {entrypoint}",
                    inputs=list(fallback_file_plan.depends_on),
                    outputs=_dedupe([*list(fallback_file_plan.writes_artifacts), entrypoint]),
                )
            )

    artifact_contract: list[RepoArtifactContract] = []
    seen_artifact_contract_keys: set[str] = set()
    for surface in _canonical_surface_nodes_by_kind(state, "artifact"):
        artifact_path = _normalize_repo_path(surface.canonical_path)
        if not _looks_like_artifact_path(artifact_path):
            continue
        if artifact_path and artifact_path not in artifact_paths:
            artifact_paths.append(artifact_path)
        owner_work_package = str(surface.owner_work_package_id or "").strip()
        contract_key = f"{artifact_path.lower()}::{owner_work_package}"
        if contract_key in seen_artifact_contract_keys:
            continue
        seen_artifact_contract_keys.add(contract_key)
        artifact_contract.append(
            RepoArtifactContract(
                artifact_key=str(surface.surface_id or artifact_path or "artifact").strip(),
                relative_path=artifact_path,
                owner_work_package=owner_work_package,
                producer_surface=producer_surface_by_owner.get(owner_work_package, ""),
                stage_name="generate",
                description="projected from canonical IR artifact surface",
                required=True,
            )
        )

    contract_stage_labels = [
        str(item.label or "").strip()
        for item in list(state.canonical_ir.contract_stages or [])
        if str(item.label or "").strip()
    ]
    canonical_route = RepoCanonicalRoute(
        summary=str(state.architecture.rationale or state.input.target[:500]).strip(),
        entry_surface=entrypoints[0],
        required_inputs=[],
        stage_sequence=contract_stage_labels,
        expected_outputs=list(artifact_paths),
        example_invocation=f"python {entrypoints[0]}",
    )

    planned_file_paths = _dedupe([item.target_file for item in files if _normalize_repo_path(item.target_file)])
    missing_after_build = [path for path in required_plan_paths if path not in planned_file_paths]
    if missing_after_build:
        message = (
            "repo_plan failed to preserve required canonical/package file paths: "
            + ", ".join(missing_after_build[:16])
        )
        _record_degraded_repo_plan_issue(
            state,
            code="repo_plan_missing_required_paths",
            message=message,
            paths=missing_after_build[:32],
        )
        raise ValueError(message)

    return RepoPlan(
        package_name="generated_experiment",
        summary=str(state.architecture.rationale or state.input.target[:500]).strip(),
        architecture=state.architecture,
        work_packages=list(_ordered_repo_work_packages(state)),
        evidence_bundles=list(state.evidence_bundles),
        files=files,
        entrypoints=entrypoints,
        canonical_route=canonical_route,
        stage_public_surfaces=stage_public_surfaces,
        artifact_contract=artifact_contract,
        structure_decisions=structure_decisions,
        artifact_paths=artifact_paths,
        global_contract=state.global_contract.model_dump(mode="json") if state.global_contract else {},
        topic_profile=state.topic_profile.model_dump(mode="json") if state.topic_profile else {},
        canonical_ir=state.canonical_ir.model_dump(mode="json"),
        canonical_ir_validation=(
            state.canonical_ir_validation.model_dump(mode="json")
            if state.canonical_ir_validation
            else {}
        ),
    )
