"""Plan-stage implementations for reproagent workflow."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable

from pydantic import ValidationError

from reproagent.pipeline.utils.artifact_writer import register_existing_file
from reproagent.pipeline.prompts import (
    build_architecture_prompt,
    build_boundary_requirements_prompt,
    build_global_contract_prompt,
    build_package_file_planning_prompt,
    build_package_file_planning_repair_prompt,
    build_package_file_planning_schema_fix_prompt,
    build_pipeline_plan_prompt,
    build_reference_selection_prompt,
    build_topic_profile_prompt,
    build_work_package_planning_prompt,
    build_work_package_planning_repair_prompt,
)
from reproagent.pipeline.schemas import (
    ActionableReference,
    ArchitectureOutput,
    ArchitectureDependency,
    ArchitectureFileBlueprint,
    ArchitectureTaskModelOutput,
    BoundaryRequirementsOutput,
    EvidenceBundleOutput,
    EvidenceLinkOutput,
    PaperBenchReproState,
    GlobalContractResultTarget,
    GlobalContractOutput,
    PackageFilePlanningOutput,
    PipelinePlanOutput,
    ReferenceRelation,
    ReferenceSelectionOutput,
    RepoFilePlan,
    TopicProfileOutput,
    WorkPackagePlanningOutput,
)
from reproagent.pipeline.utils import workflow_runtime
from reproagent.pipeline.utils.artifact_names import CANONICAL_ARTIFACTS
from reproagent.pipeline.utils.early_quality import (
    claim_inventory_quality_issues,
    file_plan_quality_issues,
    file_plan_quality_report,
    work_package_quality_report,
)
from reproagent.pipeline.config import get_workflow_config

logger = logging.getLogger(__name__)


def _should_force_empty_reference_selection(state: PaperBenchReproState) -> bool:
    preparation = dict(state.temp_data.get("reference_repo_preparation", {}) or {})
    prepared = preparation.get("prepared_repositories")
    return not isinstance(prepared, list) or len(prepared) == 0


def _fork_resume_enabled(state: PaperBenchReproState) -> bool:
    return bool(str(getattr(state.input, "fork_from_run_id", "") or "").strip()) and not bool(
        str(getattr(state.input, "resume_from_run_id", "") or "").strip()
    )


def _load_forked_json_artifact(
    state: PaperBenchReproState,
    *,
    get_output_dir: Callable[[PaperBenchReproState], Any] | None,
    relative_path: str,
) -> dict[str, Any] | None:
    if not _fork_resume_enabled(state) or get_output_dir is None:
        return None
    try:
        payload = _read_plan_json_artifact(get_output_dir(state), relative_path)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _load_in_place_resume_json_artifact(
    state: PaperBenchReproState,
    *,
    get_output_dir: Callable[[PaperBenchReproState], Any] | None,
    relative_path: str,
) -> dict[str, Any] | None:
    if not bool(getattr(state.input, "resume_in_place", False)) or get_output_dir is None:
        return None
    try:
        payload = _read_plan_json_artifact(get_output_dir(state), relative_path)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _read_plan_json_artifact(output_dir: Any, relative_path: str) -> Any:
    """Read a plan artifact from current node layout or legacy root layout."""
    base = output_dir
    candidates = [
        base / relative_path,
        base / "nodes" / "plan" / relative_path,
        base / "nodes" / "plan" / str(relative_path).rsplit("/", 1)[-1],
    ]
    for path in candidates:
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                ref_path = str(payload.get("artifact_ref") or payload.get("canonical_path") or "").strip()
                if ref_path and ref_path != str(relative_path):
                    ref_candidate = base / ref_path
                    try:
                        if ref_candidate.exists() and ref_candidate.resolve() != path.resolve():
                            return json.loads(ref_candidate.read_text(encoding="utf-8"))
                    except Exception:
                        pass
            return payload
    raise FileNotFoundError(str(candidates[0]))


def _read_architecture_debug_json(output_dir: Any, filename: str) -> Any | None:
    path = output_dir / "nodes" / "plan" / "debug" / filename
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

_METHOD_SPINE_HINTS: dict[str, tuple[str, ...]] = {
    "entrypoint": ("entry", "entrypoint", "main", "cli", "run", "启动", "入口", "主入口"),
    "artifact": ("artifact", "result", "metric", "report", "prediction", "output", "产物", "结果", "指标"),
    "config": ("config", "configs", "yaml", "yml", "json", "toml", "配置"),
    "data": ("data", "dataset", "loader", "preprocess", "数据"),
    "evaluation": ("eval", "metric", "benchmark", "score", "validation", "评估"),
}

_CONTRACT_TOKEN_STOPWORDS = {
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
    "method",
    "contract",
    "wiring",
    "module",
    "modules",
    "surface",
    "surfaces",
    "file",
    "files",
    "path",
    "paths",
    "are",
    "can",
    "how",
    "its",
    "not",
    "should",
    "than",
    "then",
    "when",
    "will",
    "your",
    "about",
    "above",
    "after",
    "alone",
    "because",
    "before",
    "between",
    "each",
    "expected",
    "fine",
    "into",
    "names",
    "other",
    "these",
    "those",
    "values",
}

_REFERENCE_BINDING_EXCLUDED_ROLES = {
    "dataset_benchmark",
    "supporting_repo",
    "external_dataset",
    "evaluation_dataset",
    "paper_dataset",
}

_GENERIC_WORK_PACKAGE_OWNER_HINTS: dict[str, str] = {
    "paper_addendum_constraints": "setup",
    "paper_task_environment_setup": "setup",
    "paper_dataset_inventory": "data",
    "paper_contract_environment_protocol": "environment",
    "paper_contract_method_baseline_protocol": "method",
    "paper_method_core": "method",
    "paper_training_or_optimization_loop": "training",
    "paper_evaluation_protocol": "evaluation",
    "paper_contract_dataset_metric_protocol": "evaluation",
    "paper_contract_sweep_hyperparameter_protocol": "configuration",
    "paper_contract_experiment_artifact_protocol": "artifact",
    "paper_evidence_matrix": "experiment",
    "paper_named_experiment_protocols": "experiment",
}

_CRITICAL_GROUNDING_TAGS = {
    "method",
    "dataset",
    "evaluation",
    "artifact",
    "entrypoint",
    "protocol",
}

_ARTIFACT_ROOT_PREFIXES = (
    "result/",
    "results/",
    "output/",
    "outputs/",
    "artifact/",
    "artifacts/",
    "report/",
    "reports/",
    "figure/",
    "figures/",
    "plot/",
    "plots/",
    "checkpoint/",
    "checkpoints/",
    "submission/",
    "submissions/",
    "prediction/",
    "predictions/",
    "logs/",
)

_ARTIFACT_FILE_SUFFIXES = (
    ".csv",
    ".tsv",
    ".jsonl",
    ".parquet",
    ".npy",
    ".npz",
    ".pkl",
    ".pickle",
    ".png",
    ".jpg",
    ".jpeg",
    ".pdf",
    ".html",
)

_ARTIFACT_ENCODED_SUFFIXES = (
    "json",
    "jsonl",
    "csv",
    "tsv",
    "parquet",
    "npy",
    "npz",
    "pkl",
    "pickle",
    "png",
    "jpg",
    "jpeg",
    "pdf",
    "html",
)

_IMPLEMENTATION_SURFACE_PATHS: dict[str, str] = {
    token: token
    for token in (
        "entry",
        "entrypoint",
        "cli",
        "main",
        "config",
        "configuration",
        "settings",
        "environment",
        "environments",
        "env",
        "simulator",
        "simulation",
        "agent",
        "agents",
        "policy",
        "policies",
        "trajectory",
        "trajectories",
        "collector",
        "data",
        "dataset",
        "datasets",
        "loader",
        "preprocess",
        "preprocessing",
        "model",
        "models",
        "network",
        "encoder",
        "decoder",
        "method",
        "methods",
        "algorithm",
        "algorithms",
        "explain",
        "explainer",
        "explainers",
        "explanation",
        "mask",
        "evaluation",
        "evaluate",
        "metric",
        "metrics",
        "score",
        "scores",
        "baseline",
        "baselines",
        "ablation",
        "ablations",
        "train",
        "training",
        "trainer",
        "artifact",
        "artifacts",
        "artifact_writer",
        "writer",
        "serialize",
        "logging",
        "report",
        "reporting",
        "plot",
        "plots",
        "figure",
        "figures",
    )
}

_SURFACE_HINT_PRIORITY: tuple[tuple[tuple[str, ...], str], ...] = (
    (("entrypoint", "entry", "cli", "main", "command"), "entrypoint"),
    (("policy", "policies", "agent", "agents"), "agent"),
    (("trajectory", "trajectories", "collector", "rollout", "episode"), "trajectory"),
    (("explainer", "explainers", "explain", "explanation", "mask"), "explainer"),
    (("baseline", "baselines", "ablation", "ablations"), "baseline"),
    (("training_loop", "trainer", "training", "train", "finetune", "optimization", "optimize"), "training"),
    (("experiment", "planner", "matrix", "protocol"), "experiment"),
    (("evaluation", "evaluate", "metric", "metrics", "score", "scores"), "evaluation"),
    (("reporting", "report", "artifact", "artifacts", "writer", "serialize", "logger", "logging"), "artifact"),
    (("plot", "plots", "plotting", "figure", "figures"), "plotting"),
    (("datapipeline", "data_pipeline", "dataset", "datasets", "data", "loader", "preprocess", "preprocessing", "sampling"), "data"),
    (("model_or_method", "model", "models", "network", "encoder", "decoder"), "model"),
    (("method", "methods", "algorithm", "algorithms"), "method"),
    (("config", "configs", "configuration", "settings", "hyperparameter", "parameter", "yaml", "yml", "toml"), "config"),
    (("environment", "environments", "env", "simulator", "simulation"), "environment"),
)


_GENERIC_SURFACE_TERMS: set[str] = {
    "entry",
    "entrypoint",
    "cli",
    "main",
    "command",
    "config",
    "configs",
    "configuration",
    "settings",
    "environment",
    "environments",
    "env",
    "simulator",
    "simulation",
    "dataset",
    "datasets",
    "data",
    "loader",
    "preprocess",
    "preprocessing",
    "model",
    "models",
    "method",
    "methods",
    "algorithm",
    "algorithms",
    "baseline",
    "baselines",
    "ablation",
    "ablations",
    "train",
    "training",
    "trainer",
    "evaluation",
    "evaluate",
    "metric",
    "metrics",
    "artifact",
    "artifacts",
    "report",
    "reporting",
    "plot",
    "plots",
    "figure",
    "figures",
    "protocol",
    "matrix",
    "surface",
    "implementation",
}


def _semantic_fallback_source_path(value: str, *, package_id: str = "") -> str:
    """Derive a task-specific source path without using a fixed scaffold."""

    terms = list(_tokenize_text(value, package_id))
    term_set = set(terms)
    lowered_value = str(value or "").lower()
    if term_set.intersection({"entrypoint", "entry", "cli", "run", "command"}) or re.search(
        r"\bmain(?:\.py|\s+entry|\s+script|\s+cli)\b",
        lowered_value,
    ):
        return "main.py"
    meaningful = [
        term
        for term in terms
        if term not in _GENERIC_SURFACE_TERMS and len(term) > 1 and not term.isdigit()
    ]
    slug_source = " ".join(meaningful[:4]) or str(package_id or value or "repo_surface")
    slug = _slugify_contract_id(slug_source)
    if term_set.intersection({"config", "configs", "configuration", "settings", "hyperparameter", "parameter"}):
        return f"configs/{slug}.yaml"
    return f"src/{slug}.py"


def _stage_review_repair_budget(state: PaperBenchReproState) -> int:
    configured = int(state.input.stage_review_repair_budget or 0)
    if configured > 0:
        return configured
    return max(0, int(get_workflow_config().max_stage_fix_rounds or 0))


def _estimated_prompt_chars(payload: Any) -> int:
    try:
        return len(json.dumps(payload, ensure_ascii=False, default=str))
    except Exception:
        return len(str(payload))


def _fanout_retry_count(stage_debug: dict[str, Any]) -> int:
    review_status = str(stage_debug.get("review_status", "") or "").strip().lower()
    attempts = int(stage_debug.get("attempts", 0) or stage_debug.get("attempts_used", 0) or 0)
    if review_status in {"accepted", "passed"}:
        return max(0, attempts - 1)
    return max(1, attempts - 1)


def _should_fan_out_stage(
    state: PaperBenchReproState,
    *,
    stage_name: str,
    work_package_count: int,
    prompt_payload: Any,
    reference_count: int,
    retry_count: int = 0,
) -> bool:
    config = get_workflow_config()
    if stage_name == "work_package_planning" and not bool(getattr(config, "work_package_planning_fanout_enabled", False)):
        return False
    if work_package_count > int(config.planning_fanout_min_work_packages or 0):
        return True
    if _estimated_prompt_chars(prompt_payload) > int(config.planning_fanout_prompt_chars or 0):
        return True
    if reference_count > int(config.planning_fanout_reference_threshold or 0):
        return True
    if retry_count > int(config.planning_fanout_retry_threshold or 0):
        return True
    return False


def _work_package_items(state: PaperBenchReproState) -> list[dict[str, Any]]:
    planning = state.work_package_planning.model_dump(mode="json") if state.work_package_planning else {}
    return [
        item
        for item in list(dict(planning).get("work_packages", []) or [])
        if isinstance(item, dict) and str(item.get("work_package_id", "") or "").strip()
    ]


def _fanout_work_package_ids(state: PaperBenchReproState) -> list[str]:
    return [
        str(item.get("work_package_id", "") or "").strip()
        for item in _work_package_items(state)
        if str(item.get("work_package_id", "") or "").strip()
    ]


def _merge_work_package_results(
    coarse: WorkPackagePlanningOutput,
    refined_outputs: list[WorkPackagePlanningOutput],
) -> WorkPackagePlanningOutput:
    refined_by_id: dict[str, dict[str, Any]] = {}
    notes: list[str] = list(coarse.planning_notes)
    for output in refined_outputs:
        notes.extend(list(output.planning_notes))
        for item in output.work_packages:
            work_package_id = str(item.work_package_id or "").strip()
            if work_package_id:
                refined_by_id[work_package_id] = item.model_dump(mode="json")
    merged_packages: list[dict[str, Any]] = []
    for item in coarse.work_packages:
        work_package_id = str(item.work_package_id or "").strip()
        merged_packages.append(refined_by_id.get(work_package_id, item.model_dump(mode="json")))
    merged = WorkPackagePlanningOutput.model_validate(
        {
            "work_packages": merged_packages,
            "coverage_summary": coarse.coverage_summary.model_dump(mode="json"),
            "planning_notes": _dedupe_nonempty(notes),
        }
    )
    return merged


def _merge_architecture_outputs(
    coarse: ArchitectureOutput,
    refined_outputs: list[ArchitectureOutput],
) -> ArchitectureOutput:
    if not refined_outputs:
        return coarse
    file_blueprints: dict[str, dict[str, Any]] = {
        str(item.path or "").strip(): item.model_dump(mode="json")
        for item in coarse.file_blueprints
        if str(item.path or "").strip()
    }
    dependency_edges: dict[tuple[str, str, str], dict[str, Any]] = {
        (
            str(item.source_path or "").strip(),
            str(item.target_path or "").strip(),
            str(item.dependency_type or "").strip(),
        ): item.model_dump(mode="json")
        for item in coarse.dependency_graph
    }
    package_layout = {
        str(key): list(value)
        for key, value in dict(coarse.package_layout or {}).items()
        if str(key).strip()
    }
    target_file_tree = list(coarse.target_file_tree)
    stable_interfaces = list(coarse.stable_interfaces)
    execution_entrypoints = list(coarse.execution_entrypoints)
    config_surfaces = list(coarse.config_surfaces)
    dependency_rules = list(coarse.dependency_rules)
    result_targets = list(coarse.result_targets)
    architecture_reference_ids = list(coarse.architecture_reference_ids)
    rationale_parts = [str(coarse.rationale or "").strip()] if str(coarse.rationale or "").strip() else []
    unresolved_review_failures = list(coarse.unresolved_review_failures)
    for output in refined_outputs:
        for item in output.file_blueprints:
            path = str(item.path or "").strip()
            if path:
                file_blueprints[path] = item.model_dump(mode="json")
        for item in output.dependency_graph:
            key = (
                str(item.source_path or "").strip(),
                str(item.target_path or "").strip(),
                str(item.dependency_type or "").strip(),
            )
            if key[0] and key[1]:
                dependency_edges[key] = item.model_dump(mode="json")
        target_file_tree = _dedupe_nonempty(target_file_tree + list(output.target_file_tree))
        stable_interfaces = _dedupe_nonempty(stable_interfaces + list(output.stable_interfaces))
        execution_entrypoints = _dedupe_nonempty(execution_entrypoints + list(output.execution_entrypoints))
        config_surfaces = _dedupe_nonempty(config_surfaces + list(output.config_surfaces))
        dependency_rules = _dedupe_nonempty(dependency_rules + list(output.dependency_rules))
        result_targets = _dedupe_nonempty(result_targets + list(output.result_targets))
        architecture_reference_ids = _dedupe_nonempty(architecture_reference_ids + list(output.architecture_reference_ids))
        unresolved_review_failures = _dedupe_nonempty(unresolved_review_failures + list(output.unresolved_review_failures))
        for package_id, paths in dict(output.package_layout or {}).items():
            normalized_id = str(package_id or "").strip()
            if not normalized_id:
                continue
            package_layout[normalized_id] = _dedupe_nonempty(list(package_layout.get(normalized_id, [])) + list(paths or []))
        if str(output.rationale or "").strip():
            rationale_parts.append(str(output.rationale or "").strip())
    return coarse.model_copy(
        update={
            "target_file_tree": target_file_tree,
            "file_blueprints": [ArchitectureFileBlueprint.model_validate(item) for item in file_blueprints.values()],
            "dependency_graph": [ArchitectureDependency.model_validate(item) for item in dependency_edges.values()],
            "stable_interfaces": stable_interfaces,
            "execution_entrypoints": execution_entrypoints,
            "config_surfaces": config_surfaces,
            "package_layout": package_layout,
            "dependency_rules": dependency_rules,
            "result_targets": result_targets,
            "architecture_reference_ids": architecture_reference_ids,
            "unresolved_review_failures": unresolved_review_failures,
            "rationale": " ".join(part for part in rationale_parts if part).strip(),
        }
    )


def _merge_package_file_plan_outputs(
    coarse: PackageFilePlanningOutput,
    refined_outputs: list[PackageFilePlanningOutput],
) -> PackageFilePlanningOutput:
    refined_by_path: dict[str, dict[str, Any]] = {}
    notes: list[str] = list(coarse.planning_notes)
    unresolved: list[str] = list(coarse.unresolved_review_failures)
    for output in refined_outputs:
        notes.extend(list(output.planning_notes))
        unresolved.extend(list(output.unresolved_review_failures))
        for item in output.file_plans:
            path = str(item.target_file or "").strip()
            if path:
                refined_by_path[path] = item.model_dump(mode="json")
    merged_file_plans = [
        RepoFilePlan.model_validate(refined_by_path.get(str(item.target_file or "").strip(), item.model_dump(mode="json")))
        for item in coarse.file_plans
    ]
    return PackageFilePlanningOutput.model_validate(
        {
            "file_plans": [item.model_dump(mode="json") for item in merged_file_plans],
            "planning_notes": _dedupe_nonempty(notes),
            "unresolved_review_failures": _dedupe_nonempty(unresolved),
        }
    )


def _positive_architecture_dependency_rules(
    dependency_graph: list[dict[str, Any]],
    *,
    stable_interfaces: list[str],
    execution_entrypoints: list[str],
    config_surfaces: list[str],
    package_layout: dict[str, list[str]],
) -> list[str]:
    """Project architecture dependency policy as allowed implementation routes."""
    rules: list[str] = []
    for edge in dependency_graph[:40]:
        source_path = _normalize_repo_path(str(edge.get("source_path", "") or ""))
        target_path = _normalize_repo_path(str(edge.get("target_path", "") or ""))
        dependency_type = str(edge.get("dependency_type", "imports") or "imports").strip() or "imports"
        if source_path and target_path:
            rules.append(f"{source_path} may use {target_path} via {dependency_type}.")
    if stable_interfaces:
        rules.append("Stable interface surfaces: " + ", ".join(_dedupe_nonempty(stable_interfaces)[:12]) + ".")
    if execution_entrypoints:
        rules.append("Execution entrypoints call package-owned data, method, training, evaluation, and artifact routes.")
    if config_surfaces:
        rules.append("Configuration surfaces provide bounded execution defaults and named experiment selectors.")
    for work_package_id, paths in list(dict(package_layout or {}).items())[:12]:
        owned_paths = _dedupe_nonempty([_normalize_repo_path(path) for path in list(paths or []) if _normalize_repo_path(path)])
        if owned_paths:
            rules.append(f"{work_package_id} owns package paths: {', '.join(owned_paths[:12])}.")
    if not rules:
        rules.append("Generated files use package-local stable interfaces and explicit entrypoint-to-artifact routes.")
    return _dedupe_nonempty(rules)[:64]


def _formula_algorithm_contract_from_input(input_payload: dict[str, Any]) -> dict[str, Any]:
    evidence_contract = dict(input_payload.get("paper_evidence_contract", {}) or {})
    return dict(evidence_contract.get("formula_algorithm_contract", {}) or {})


def _formula_algorithm_contract_for_state(state: PaperBenchReproState) -> dict[str, Any]:
    cache_key = "plan_node_formula_algorithm_contract_v1"
    cached = state.temp_data.get(cache_key)
    if isinstance(cached, dict):
        return cached
    try:
        from reproagent.pipeline.utils.prompt_context_builder import _paper_evidence_contract_payload

        payload = _paper_evidence_contract_payload(state)
        contract = dict(dict(payload.get("formula_algorithm_contract", {}) or {}))
    except Exception:
        contract = {}
    state.temp_data[cache_key] = contract
    return contract


def _input_payload_with_formula_contract(
    input_payload: dict[str, Any],
    state: PaperBenchReproState,
) -> dict[str, Any]:
    if _formula_algorithm_contract_from_input(input_payload):
        return input_payload
    contract = _formula_algorithm_contract_for_state(state)
    if not contract:
        return input_payload
    evidence_contract = dict(input_payload.get("paper_evidence_contract", {}) or {})
    evidence_contract["formula_algorithm_contract"] = contract
    return {**input_payload, "paper_evidence_contract": evidence_contract}


def _formula_anchor_text_terms(value: Any) -> set[str]:
    return {
        token
        for token in _tokenize_text(value)
        if token not in _CONTRACT_TOKEN_STOPWORDS
        and token not in _GENERIC_SURFACE_TERMS
        and len(token) >= 2
    }


def _formula_contract_anchor_obligations(
    input_payload: dict[str, Any],
    *,
    text_scope: str = "",
    limit: int = 8,
) -> list[str]:
    """Project paper-derived formula anchors into positive implementation obligations."""
    contract = _formula_algorithm_contract_from_input(input_payload)
    if not contract:
        return []
    scope_terms = _formula_anchor_text_terms(text_scope)
    anchor_rows = [item for item in list(contract.get("anchors", []) or []) if isinstance(item, dict)]
    scored_rows: list[tuple[int, int, dict[str, Any]]] = []
    for index, anchor in enumerate(anchor_rows):
        anchor_terms = _formula_anchor_text_terms(
            " ".join(
                [
                    str(anchor.get("section_title", "") or ""),
                    str(anchor.get("source_id", "") or ""),
                    *[str(item) for item in list(anchor.get("required_symbols", []) or [])],
                    *[str(item) for item in list(anchor.get("required_numeric_values", []) or [])],
                    *[str(item) for item in list(anchor.get("algorithm_terms", []) or [])],
                    *[str(item) for item in list(anchor.get("algorithm_steps", []) or [])],
                    *[str(item) for item in list(anchor.get("formula_or_algorithm_excerpts", []) or [])[:2]],
                ]
            )
        )
        score = len(scope_terms.intersection(anchor_terms)) if scope_terms else 0
        scored_rows.append((score, index, anchor))
    if scope_terms and any(score > 0 for score, _index, _anchor in scored_rows):
        selected_rows = [anchor for score, _index, anchor in sorted(scored_rows, key=lambda item: (-item[0], item[1])) if score > 0]
    else:
        selected_rows = [anchor for _score, _index, anchor in scored_rows]

    obligations: list[str] = []
    for anchor in selected_rows[: max(1, limit)]:
        section = str(anchor.get("section_title", "") or anchor.get("source_id", "") or "").strip()
        symbols = _dedupe_nonempty([str(item) for item in list(anchor.get("required_symbols", []) or [])])
        numeric_values = _dedupe_nonempty([str(item) for item in list(anchor.get("required_numeric_values", []) or [])])
        algorithm_terms = _dedupe_nonempty([str(item) for item in list(anchor.get("algorithm_terms", []) or [])])[:8]
        algorithm_steps = _dedupe_nonempty([str(item) for item in list(anchor.get("algorithm_steps", []) or [])])[:2]
        excerpts = _dedupe_nonempty([str(item) for item in list(anchor.get("formula_or_algorithm_excerpts", []) or [])])[:1]
        priority_symbols = [
            value
            for value in symbols
            if _formula_anchor_text_terms(value).intersection(scope_terms)
        ]
        if not priority_symbols and not scope_terms:
            priority_symbols = [
                value
                for value in symbols
                if any(token in str(value).lower() for token in ("tau", "mu", "apt", "mask", "rank"))
            ]
        symbols = _dedupe_nonempty(priority_symbols + symbols)[:14]
        priority_numeric = [
            value
            for value in numeric_values
            if str(value).strip() in {"0.85", "0.15", "0.9", "0.1", "4", "8"}
        ]
        numeric_values = _dedupe_nonempty(priority_numeric + numeric_values)[:8]
        parts = []
        if section:
            parts.append(section)
        if symbols:
            parts.append("symbols " + ", ".join(symbols))
        if numeric_values:
            parts.append("numeric/defaults " + ", ".join(numeric_values))
        if algorithm_terms:
            parts.append("algorithm terms " + ", ".join(algorithm_terms))
        if algorithm_steps:
            parts.append("steps " + " ; ".join(algorithm_steps))
        elif excerpts:
            parts.append("formula " + excerpts[0])
        if parts:
            obligations.append("Implement paper formula/algorithm anchor as executable code/config: " + " | ".join(parts))

    inventory_symbols = _dedupe_nonempty([str(item) for item in list(contract.get("required_symbol_inventory", []) or [])])[:24]
    inventory_values = _dedupe_nonempty([str(item) for item in list(contract.get("required_numeric_inventory", []) or [])])[:16]
    if inventory_symbols or inventory_values:
        obligations.append(
            "Keep formula/algorithm inventory code-visible: "
            + (("symbols " + ", ".join(inventory_symbols)) if inventory_symbols else "")
            + (("; numeric/defaults " + ", ".join(inventory_values)) if inventory_values else "")
        )
    implementation_obligations = _dedupe_nonempty(
        [str(item) for item in list(contract.get("implementation_obligations", []) or [])]
    )[: max(4, limit)]
    obligations.extend("Satisfy formula/algorithm implementation obligation: " + item for item in implementation_obligations)
    return _dedupe_nonempty(obligations)[:limit]


def _formula_anchor_symbols_for_scope(
    input_payload: dict[str, Any],
    *,
    text_scope: str,
    limit: int = 16,
) -> list[str]:
    contract = _formula_algorithm_contract_from_input(input_payload)
    if not contract:
        return []
    obligations = _formula_contract_anchor_obligations(input_payload, text_scope=text_scope, limit=limit)
    selected_terms = _formula_anchor_text_terms(" ".join(obligations))
    symbols: list[str] = []
    for anchor in list(contract.get("anchors", []) or []):
        if not isinstance(anchor, dict):
            continue
        anchor_terms = _formula_anchor_text_terms(
            " ".join(
                [
                    str(anchor.get("section_title", "") or ""),
                    *[str(item) for item in list(anchor.get("required_symbols", []) or [])],
                    *[str(item) for item in list(anchor.get("algorithm_terms", []) or [])],
                    *[str(item) for item in list(anchor.get("formula_or_algorithm_excerpts", []) or [])[:1]],
                ]
            )
        )
        if selected_terms and not selected_terms.intersection(anchor_terms):
            continue
        symbols.extend(str(item) for item in list(anchor.get("required_symbols", []) or []))
        symbols.extend(str(item) for item in list(anchor.get("algorithm_terms", []) or []))
    if not symbols:
        symbols.extend(str(item) for item in list(contract.get("required_symbol_inventory", []) or []))
    return _dedupe_nonempty(symbols)[:limit]


def _append_text_with_obligations(base: str, obligations: list[str], *, limit: int) -> str:
    parts = _dedupe_nonempty([str(base or "").strip()] + [str(item) for item in obligations])
    return "; ".join(parts)[:limit]


def _grounding_terms_for_work_package(work_package: Any) -> set[str]:
    tokens: set[str] = set()
    parts: list[str] = [
        str(getattr(work_package, "goal", "") or ""),
        *[str(item) for item in list(getattr(work_package, "tags", []) or [])],
        *[str(item) for item in list(getattr(work_package, "evidence_needs", []) or [])],
        *[str(item) for item in list(getattr(work_package, "method_obligations", []) or [])],
    ]
    inventories = dict(getattr(work_package, "inventories", {}) or {})
    for values in inventories.values():
        parts.extend(str(item) for item in list(values or []))
    for part in parts:
        for raw in str(part or "").lower().replace("/", " ").replace("_", " ").replace("-", " ").split():
            token = raw.strip(" .,:;()[]{}")
            if len(token) >= 3 and token not in _CONTRACT_TOKEN_STOPWORDS:
                tokens.add(token)
    return tokens


def _clean_reference_token(raw: object) -> str:
    token = str(raw or "").lower().strip(" .,:;()[]{}'\"`")
    if len(token) < 3 or token in _CONTRACT_TOKEN_STOPWORDS or token in _GENERIC_SURFACE_TERMS:
        return ""
    return token


def _coverage_signal_terms(coverage: Any) -> set[str]:
    terms: set[str] = set()
    for raw in [
        str(getattr(coverage, "title", "") or ""),
        str(getattr(coverage, "scope", "") or ""),
        *[str(item) for item in list(getattr(coverage, "matched_keywords", []) or [])],
    ]:
        for token in str(raw).replace("/", " ").replace("_", " ").replace("-", " ").split():
            cleaned = _clean_reference_token(token)
            if cleaned:
                terms.add(cleaned)
    for path in list(getattr(coverage, "matched_files", []) or []):
        for token in _normalize_repo_path(str(path or "")).replace("/", " ").replace("_", " ").replace("-", " ").split():
            cleaned = _clean_reference_token(token.rsplit(".", 1)[0])
            if cleaned:
                terms.add(cleaned)
    return terms


def _survey_symbol_signal_terms(survey: Any) -> set[str]:
    terms: set[str] = set()
    for evidence in list(getattr(survey, "symbol_evidence", []) or []):
        for raw in [
            str(getattr(evidence, "symbol_name", "") or ""),
            str(getattr(evidence, "symbol_kind", "") or ""),
            *[str(item) for item in list(getattr(evidence, "matched_keywords", []) or [])],
            *[str(item) for item in list(getattr(evidence, "matched_surfaces", []) or [])],
        ]:
            for token in str(raw).replace("/", " ").replace("_", " ").replace("-", " ").split():
                cleaned = _clean_reference_token(token)
                if cleaned:
                    terms.add(cleaned)
    return terms


def _survey_relevance_signal(state: PaperBenchReproState, survey: Any) -> tuple[int, list[str]]:
    paper_terms: set[str] = set()
    for unit in list(state.unit_extraction.units if state.unit_extraction else []):
        paper_terms.update(
            _clean_reference_token(token)
            for token in _tokenize_text(
                str(getattr(unit, "unit_id", "") or ""),
                str(getattr(unit, "statement", "") or ""),
                *[str(item) for item in list(getattr(unit, "implementation_surfaces", []) or [])],
                *[str(item) for item in list(getattr(unit, "runtime_interfaces", []) or [])],
                *[str(item) for item in list(getattr(unit, "expected_artifacts", []) or [])],
            )
        )
    paper_terms.discard("")
    survey_terms: set[str] = set()
    for coverage in list(getattr(survey, "requirement_coverage", []) or []):
        survey_terms.update(_coverage_signal_terms(coverage))
    survey_terms.update(_survey_symbol_signal_terms(survey))
    overlap = sorted(paper_terms.intersection(survey_terms))
    evidence_count = 0
    for evidence in list(getattr(survey, "symbol_evidence", []) or []):
        symbol_terms = set()
        for raw in [
            str(getattr(evidence, "symbol_name", "") or ""),
            *[str(item) for item in list(getattr(evidence, "matched_keywords", []) or [])],
        ]:
            for token in str(raw).replace("/", " ").replace("_", " ").replace("-", " ").split():
                cleaned = _clean_reference_token(token)
                if cleaned:
                    symbol_terms.add(cleaned)
        if symbol_terms.intersection(paper_terms):
            evidence_count += 1
    return len(overlap) + evidence_count, overlap[:12]


def _survey_is_actionable_for_main_plan(state: PaperBenchReproState, survey: Any) -> tuple[bool, str]:
    role = str(getattr(survey, "reference_role", "") or "").strip().lower()
    if role in _REFERENCE_BINDING_EXCLUDED_ROLES:
        return False, f"reference role `{role}` is auxiliary, not an implementation source"
    signal_score, overlap = _survey_relevance_signal(state, survey)
    if signal_score < 3:
        return False, "insufficient paper-specific overlap: " + (", ".join(overlap) if overlap else "none")
    return True, "paper-specific overlap: " + ", ".join(overlap[:8])


def _survey_grounding_score(work_package: Any, survey: Any) -> int:
    package_terms = _grounding_terms_for_work_package(work_package)
    if not package_terms:
        return 0
    survey_terms: set[str] = set()
    survey_terms.update(
        token
        for raw in [
            str(getattr(survey, "title", "") or ""),
            str(getattr(survey, "readme_summary", "") or ""),
            str(getattr(survey, "file_tree_summary", "") or ""),
            *[str(item) for item in list(getattr(survey, "protocol_clues", []) or [])],
            *[str(item) for item in list(getattr(survey, "top_python_files", []) or [])],
            *[str(item) for item in list(getattr(survey, "likely_reusable_files", []) or [])],
        ]
        for token in str(raw).lower().replace("/", " ").replace("_", " ").replace("-", " ").split()
        if len(token.strip(" .,:;()[]{}")) >= 3
    )
    for coverage in list(getattr(survey, "requirement_coverage", []) or []):
        survey_terms.update(_coverage_signal_terms(coverage))
    survey_terms.update(_survey_symbol_signal_terms(survey))
    overlap = package_terms.intersection(survey_terms)
    return len(overlap)


def _augment_work_package_references_for_grounding(
    state: PaperBenchReproState,
    failed_work_package_ids: list[str],
) -> tuple[WorkPackagePlanningOutput, list[str]]:
    """Best-effort review-fix for critical grounding by widening reference bindings."""
    if state.work_package_planning is None:
        raise ValueError("grounding repair requires work_package_planning")
    failed_ids = set(failed_work_package_ids)
    survey_candidates = list(state.reference_repo_surveys)
    notes: list[str] = []
    patched_packages: list[dict[str, Any]] = []
    for work_package in state.work_package_planning.work_packages:
        if work_package.work_package_id not in failed_ids:
            patched_packages.append(work_package.model_dump(mode="json"))
            continue
        ranked = sorted(
            (
                (_survey_grounding_score(work_package, survey), survey.ref_id)
                for survey in survey_candidates
            ),
            key=lambda item: (-item[0], item[1]),
        )
        extra_refs = [
            ref_id
            for score, ref_id in ranked
            if score > 0 and ref_id not in work_package.reference_ids
        ][:2]
        patched_packages.append(
            work_package.model_copy(
                update={
                    "reference_ids": _dedupe_nonempty(list(work_package.reference_ids) + extra_refs)
                }
            ).model_dump(mode="json")
        )
        if extra_refs:
            notes.append(
                f"grounding review-fix widened `{work_package.work_package_id}` reference_ids with: {', '.join(extra_refs)}"
            )
    return (
        WorkPackagePlanningOutput.model_validate(
            {
                "work_packages": patched_packages,
                "coverage_summary": state.work_package_planning.coverage_summary.model_dump(mode="json"),
                "planning_notes": _dedupe_nonempty(list(state.work_package_planning.planning_notes) + notes),
            }
        ),
        notes,
    )


def _supported_requirement_ids_from_survey(survey: Any) -> list[str]:
    supported: list[str] = []
    for coverage in list(getattr(survey, "requirement_coverage", []) or []):
        requirement_id = str(getattr(coverage, "requirement_id", "") or "").strip()
        if not requirement_id:
            continue
        if int(getattr(coverage, "keyword_hits", 0) or 0) > 0:
            supported.append(requirement_id)
            continue
        if list(getattr(coverage, "matched_files", []) or []) or list(getattr(coverage, "code_snippets", []) or []):
            supported.append(requirement_id)
    return _dedupe_nonempty(supported)


def _survey_reusable_modules(survey: Any) -> list[str]:
    modules: list[str] = []
    for evidence in list(getattr(survey, "symbol_evidence", []) or []):
        symbol_name = str(getattr(evidence, "symbol_name", "") or "").strip()
        if symbol_name and symbol_name != "__module__":
            modules.append(symbol_name)
    for raw_path in [
        *list(getattr(survey, "likely_reusable_files", []) or []),
        *list(getattr(survey, "top_python_files", []) or []),
    ]:
        normalized = _normalize_repo_path(str(raw_path or ""))
        if not normalized:
            continue
        stem = normalized.rsplit("/", 1)[-1].rsplit(".", 1)[0].strip()
        if stem:
            modules.append(stem)
    return _dedupe_nonempty(modules)[:8]


def _survey_insights(survey: Any) -> list[str]:
    insights: list[str] = []
    supported_requirement_ids = _supported_requirement_ids_from_survey(survey)
    if supported_requirement_ids:
        insights.append(
            "Supports requirements: " + ", ".join(supported_requirement_ids[:6])
        )
    likely_reusable_files = _dedupe_nonempty([str(item) for item in list(getattr(survey, "likely_reusable_files", []) or [])])
    if likely_reusable_files:
        insights.append(
            "Likely reusable files: " + ", ".join(likely_reusable_files[:4])
        )
    protocol_clues = _dedupe_nonempty([str(item) for item in list(getattr(survey, "protocol_clues", []) or [])])
    if protocol_clues:
        insights.append("Protocol clues available from local repo survey.")
    symbol_evidence = list(getattr(survey, "symbol_evidence", []) or [])
    if symbol_evidence:
        insights.append(
            "Symbol-level evidence available from local repo survey."
        )
    return _dedupe_nonempty(insights)[:4]


def _survey_supported_scope_items(survey: Any) -> list[str]:
    supported_scope_items = _dedupe_nonempty(
        [
            str(getattr(item, "scope", "") or "").strip()
            for item in list(getattr(survey, "requirement_coverage", []) or [])
            if str(getattr(item, "scope", "") or "").strip()
            and (
                int(getattr(item, "keyword_hits", 0) or 0) > 0
                or list(getattr(item, "matched_files", []) or [])
                or list(getattr(item, "code_snippets", []) or [])
            )
        ]
    )
    if supported_scope_items:
        return supported_scope_items
    derived = []
    for evidence in list(getattr(survey, "symbol_evidence", []) or []):
        derived.extend(str(item or "").strip() for item in list(getattr(evidence, "matched_surfaces", []) or []))
    return _dedupe_nonempty(derived)


def _valid_ref_ids_from_state(state: PaperBenchReproState) -> set[str]:
    return {
        str(getattr(survey, "ref_id", "") or "").strip()
        for survey in list(state.reference_repo_surveys or [])
        if str(getattr(survey, "ref_id", "") or "").strip()
    }


def _survey_snippet_candidates(survey: Any) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    for evidence in list(getattr(survey, "symbol_evidence", []) or []):
        snippet = str(getattr(evidence, "snippet", "") or "").strip()
        module = str(getattr(evidence, "symbol_name", "") or "").strip()
        if snippet and module:
            candidates.append((module, snippet))
    for coverage in list(getattr(survey, "requirement_coverage", []) or []):
        for raw_path, snippet in zip(
            list(getattr(coverage, "matched_files", []) or []),
            list(getattr(coverage, "code_snippets", []) or []),
        ):
            normalized = _normalize_repo_path(str(raw_path or ""))
            module = normalized.rsplit("/", 1)[-1].rsplit(".", 1)[0].strip() if normalized else ""
            rendered = str(snippet or "").strip()
            if module and rendered:
                candidates.append((module, rendered))
    return candidates


def _sanitize_pipeline_plan_with_surveys(
    state: PaperBenchReproState,
    pipeline_plan: PipelinePlanOutput,
) -> PipelinePlanOutput:
    formula_input_payload = _input_payload_with_formula_contract({}, state)
    work_packages_by_id = {
        str(getattr(item, "work_package_id", "") or "").strip(): item
        for item in list(state.work_package_planning.work_packages if state.work_package_planning else [])
        if str(getattr(item, "work_package_id", "") or "").strip()
    }
    units_by_id = {
        str(getattr(unit, "unit_id", "") or "").strip(): unit.model_dump(mode="json")
        for unit in list(state.unit_extraction.units if state.unit_extraction else [])
        if str(getattr(unit, "unit_id", "") or "").strip()
    }
    survey_by_ref_id = {
        str(getattr(item, "ref_id", "") or "").strip(): item
        for item in list(state.reference_repo_surveys or [])
        if str(getattr(item, "ref_id", "") or "").strip()
    }
    valid_ref_ids = set(survey_by_ref_id)
    patched_nodes: list[dict[str, Any]] = []
    for node in list(pipeline_plan.plan_nodes or []):
        payload = node.model_dump(mode="json")
        node_slug = _slugify_contract_id(str(payload.get("node_id", "") or ""))
        package_match = next(
            (
                work_package
                for package_id, work_package in work_packages_by_id.items()
                if package_id and _slugify_contract_id(package_id) in node_slug
            ),
            None,
        )
        owned_units: list[dict[str, Any]] = []
        obligations: list[str] = []
        artifacts: list[str] = []
        work_package_id = str(payload.get("node_id", "") or "plan_node").strip()
        goal = str(payload.get("name", "") or payload.get("description", "") or "").strip()
        if package_match is not None:
            work_package_id = str(package_match.work_package_id or work_package_id).strip()
            goal = str(package_match.goal or goal).strip()
            owned_units = [
                units_by_id[unit_id]
                for unit_id in _dedupe_nonempty([str(item) for item in list(package_match.owned_unit_ids or [])])
                if unit_id in units_by_id
            ]
            obligations = _dedupe_nonempty(
                [str(item) for item in list(package_match.method_obligations or [])]
                + [
                    str(obligation)
                    for unit in owned_units
                    for obligation in _payload_positive_obligations(unit, limit=8)
                ]
            )
            artifacts = _dedupe_nonempty(
                [str(item) for item in list(package_match.produces or [])]
                + [
                    str(artifact)
                    for unit in owned_units
                    for artifact in list(unit.get("expected_artifacts", []) or [])
                ]
            )
            formula_obligations = _formula_contract_anchor_obligations(
                formula_input_payload,
                text_scope=" ".join(
                    _dedupe_nonempty(
                        [
                            work_package_id,
                            goal,
                            *obligations,
                            *artifacts,
                            *[str(item) for item in list(package_match.tags or [])],
                            *[
                                str(item)
                                for values in dict(package_match.inventories or {}).values()
                                for item in list(values or [])
                            ],
                        ]
                    )
                ),
                limit=6,
            )
            obligations = _dedupe_nonempty(obligations + formula_obligations)
            payload["name"] = goal[:120] if goal else payload.get("name", "")
            payload["description"] = _positive_plan_focus(
                work_package_id=work_package_id,
                goal=goal,
                owned_units=owned_units,
                obligations=obligations,
                artifacts=artifacts,
            )[:1200]
        else:
            formula_obligations = _formula_contract_anchor_obligations(
                formula_input_payload,
                text_scope=" ".join(
                    _dedupe_nonempty(
                        [
                            work_package_id,
                            goal,
                            str(payload.get("description", "") or ""),
                            str(payload.get("hypothesis", "") or ""),
                            str(payload.get("decision_value", "") or ""),
                        ]
                    )
                ),
                limit=4,
            )
            obligations = _dedupe_nonempty(obligations + formula_obligations)
            payload["description"] = _positive_plan_focus(
                work_package_id=work_package_id,
                goal=goal,
                owned_units=owned_units,
                obligations=obligations,
                artifacts=artifacts,
            )[:1200]
        payload["hypothesis"] = _positive_hypothesis_for_plan_node(
            work_package_id=work_package_id,
            goal=goal,
            owned_units=owned_units,
            obligations=obligations,
            artifacts=artifacts,
        )
        payload["decision_value"] = _positive_decision_value_for_plan_node(
            work_package_id=work_package_id,
            goal=goal,
            owned_units=owned_units,
            obligations=obligations,
            artifacts=artifacts,
        )
        payload["stop_rule_or_pruning_rationale"] = _positive_scope_for_plan_node(work_package_id)
        ref_id = str(payload.get("ref_id", "") or "").strip()
        traceable = bool(payload.get("traceable"))
        if ref_id and ref_id not in valid_ref_ids:
            payload["ref_id"] = ""
            payload["reusable_module"] = ""
            payload["traceable"] = False
            payload["code_snippet"] = ""
            payload["insight"] = ""
            patched_nodes.append(payload)
            continue
        if not ref_id:
            payload["traceable"] = False
            payload["code_snippet"] = ""
            payload["insight"] = ""
            patched_nodes.append(payload)
            continue
        survey = survey_by_ref_id.get(ref_id)
        candidates = _survey_snippet_candidates(survey)
        reusable_module = str(payload.get("reusable_module", "") or "").strip()
        snippet = str(payload.get("code_snippet", "") or "").strip()
        has_grounded_candidate = False
        for candidate_module, candidate_snippet in candidates:
            if reusable_module and reusable_module == candidate_module:
                if snippet and snippet in candidate_snippet:
                    has_grounded_candidate = True
                    break
                if not snippet:
                    payload["code_snippet"] = candidate_snippet
                    has_grounded_candidate = True
                    break
            if snippet and snippet in candidate_snippet:
                payload["reusable_module"] = candidate_module
                has_grounded_candidate = True
                break
        if traceable and not has_grounded_candidate:
            payload["traceable"] = False
            payload["code_snippet"] = ""
            payload["insight"] = ""
        patched_nodes.append(payload)
    return PipelinePlanOutput.model_validate(
        {
            "plan_nodes": patched_nodes,
            "coverage_summary": pipeline_plan.coverage_summary.model_dump(mode="json"),
        }
    )


def _normalize_pipeline_plan_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Coerce nullable LLM fields before schema validation."""
    normalized = dict(payload or {})
    nodes: list[dict[str, Any]] = []
    for index, raw_node in enumerate(list(normalized.get("plan_nodes", []) or []), start=1):
        if not isinstance(raw_node, dict):
            continue
        node = dict(raw_node)
        for field_name in (
            "node_id",
            "parent_node_id",
            "name",
            "description",
            "hypothesis",
            "decision_value",
            "stop_rule_or_pruning_rationale",
            "ref_id",
            "reusable_module",
            "code_snippet",
            "insight",
        ):
            if node.get(field_name) is None:
                node[field_name] = ""
        if not str(node.get("node_id", "") or "").strip():
            node["node_id"] = f"plan_node_{index:03d}"
        if not str(node.get("name", "") or "").strip():
            node["name"] = str(node.get("description", "") or node.get("node_id", "") or f"plan node {index}").strip()
        if not str(node.get("level", "") or "").strip():
            node["level"] = "module"
        for field_name in ("requirement_ids", "depends_on"):
            values = node.get(field_name)
            node[field_name] = [
                str(item).strip()
                for item in (values if isinstance(values, list) else [])
                if str(item).strip()
            ]
        nodes.append(node)
    normalized["plan_nodes"] = nodes
    coverage = normalized.get("coverage_summary")
    if not isinstance(coverage, dict):
        normalized["coverage_summary"] = {}
    return normalized


def _work_package_unit_lookup(input_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item.get("unit_id", "") or "").strip(): dict(item)
        for item in list(input_payload.get("units", []) or [])
        if isinstance(item, dict) and str(item.get("unit_id", "") or "").strip()
    }


def _unit_source_anchor_ids(unit: dict[str, Any]) -> list[str]:
    anchor_ids: list[str] = []
    for field_name in (
        "source_unit_ids",
        "source_paragraph_ids",
        "source_chunk_ids",
        "source_section_ids",
        "source_ids",
    ):
        anchor_ids.extend(
            str(item).strip()
            for item in list(unit.get(field_name, []) or [])
            if str(item).strip()
        )
    return _dedupe_nonempty(anchor_ids)


def _requirement_ids_for_unit_ids(input_payload: dict[str, Any], unit_ids: list[str]) -> list[str]:
    selected_unit_ids = {str(item or "").strip() for item in unit_ids if str(item or "").strip()}
    if not selected_unit_ids:
        return []
    selected_source_ids = set(selected_unit_ids)
    units_by_id = _work_package_unit_lookup(input_payload)
    for unit_id in selected_unit_ids:
        unit = units_by_id.get(unit_id)
        if not unit:
            continue
        selected_source_ids.update(_unit_source_anchor_ids(unit))
    requirement_ids: list[str] = []
    for requirement in list(dict(input_payload.get("boundary_requirements", {}) or {}).get("boundary_requirements", []) or []):
        if not isinstance(requirement, dict):
            continue
        requirement_id = str(requirement.get("requirement_id", "") or "").strip()
        source_unit_ids = {
            str(item or "").strip()
            for item in list(requirement.get("source_unit_ids", []) or [])
            if str(item or "").strip()
        }
        if requirement_id and source_unit_ids.intersection(selected_source_ids):
            requirement_ids.append(requirement_id)
    return _dedupe_nonempty(requirement_ids)


def _traceable_reference_hint(
    input_payload: dict[str, Any],
    reference_ids: list[str],
) -> tuple[str, str, str, str]:
    selected_ids = [str(item or "").strip() for item in reference_ids if str(item or "").strip()]
    if not selected_ids:
        return "", "", "", ""
    selected_id_set = set(selected_ids)
    for reference in list(dict(input_payload.get("reference_selection", {}) or {}).get("actionable_references", []) or []):
        if not isinstance(reference, dict):
            continue
        ref_id = str(reference.get("ref_id", "") or "").strip()
        if ref_id not in selected_id_set:
            continue
        reusable_modules = _dedupe_nonempty([str(item) for item in list(reference.get("reusable_modules", []) or [])])
        insights = _dedupe_nonempty([str(item) for item in list(reference.get("insights", []) or [])])
        return (
            ref_id,
            reusable_modules[0] if reusable_modules else "",
            "",
            insights[0] if insights else "",
        )
    for reference in list(input_payload.get("prepared_reference_repositories", []) or []):
        if not isinstance(reference, dict):
            continue
        ref_id = str(reference.get("ref_id", "") or "").strip()
        if ref_id not in selected_id_set:
            continue
        reusable_files = _dedupe_nonempty([str(item) for item in list(reference.get("likely_reusable_files", []) or [])])
        protocol_clues = _dedupe_nonempty([str(item) for item in list(reference.get("protocol_clues", []) or [])])
        return (
            ref_id,
            reusable_files[0] if reusable_files else "",
            "",
            protocol_clues[0] if protocol_clues else "",
        )
    return selected_ids[0], "", "", ""


def _synthesize_pipeline_plan_output(
    input_payload: dict[str, Any],
    *,
    reason: str = "",
) -> PipelinePlanOutput:
    """Project verified work packages into flat traceable plan nodes."""
    work_packages = [
        dict(item)
        for item in list(dict(input_payload.get("work_package_planning", {}) or {}).get("work_packages", []) or [])
        if isinstance(item, dict)
    ]
    formula_obligations_global = _formula_contract_anchor_obligations(input_payload, limit=12)
    units_by_id = _work_package_unit_lookup(input_payload)
    plan_nodes: list[dict[str, Any]] = []
    covered_requirement_ids: set[str] = set()
    for index, work_package in enumerate(work_packages, start=1):
        work_package_id = str(work_package.get("work_package_id", "") or "").strip() or f"wp_{index:03d}"
        owned_unit_ids = _dedupe_nonempty([str(item) for item in list(work_package.get("owned_unit_ids", []) or [])])
        requirement_ids = _dedupe_nonempty(
            [str(item) for item in list(work_package.get("requirement_ids", []) or [])]
            + _requirement_ids_for_unit_ids(input_payload, owned_unit_ids)
        )
        covered_requirement_ids.update(requirement_ids)
        owned_units = [units_by_id[unit_id] for unit_id in owned_unit_ids if unit_id in units_by_id]
        goal = str(work_package.get("goal", "") or "").strip()
        if not goal and owned_units:
            goal = str(owned_units[0].get("statement", "") or "").strip()
        obligation_text = _dedupe_nonempty(
            [str(item) for item in list(work_package.get("method_obligations", []) or [])]
            + [
                str(obligation)
                for unit in owned_units
                for obligation in _payload_positive_obligations(unit, limit=8)
            ]
        )
        obligation_text = _dedupe_nonempty(
            obligation_text
            + _formula_contract_anchor_obligations(
                input_payload,
                text_scope=" ".join(
                    _dedupe_nonempty(
                        [
                            work_package_id,
                            goal,
                            *obligation_text,
                            *[str(item) for item in list(work_package.get("tags", []) or [])],
                            *[
                                str(item)
                                for values in dict(work_package.get("inventories", {}) or {}).values()
                                for item in list(values or [])
                            ],
                        ]
                    )
                ),
                limit=6,
            )
        )
        artifact_text = _dedupe_nonempty(
            [str(item) for item in list(work_package.get("produces", []) or [])]
            + [
                str(artifact)
                for unit in owned_units
                for artifact in list(unit.get("expected_artifacts", []) or [])
            ]
        )
        ref_id, reusable_module, code_snippet, insight = _traceable_reference_hint(
            input_payload,
            [str(item) for item in list(work_package.get("reference_ids", []) or [])],
        )
        description_parts = _dedupe_nonempty(
            [goal]
            + obligation_text[:6]
            + ([f"Expected artifacts: {', '.join(artifact_text[:6])}"] if artifact_text else [])
        )
        plan_nodes.append(
            {
                "node_id": f"node_{index:03d}_{_slugify_contract_id(work_package_id)}",
                "parent_node_id": "",
                "name": goal[:120] if goal else f"Implement {work_package_id}",
                "level": "module",
                "description": "; ".join(description_parts)[:1200],
                "hypothesis": str(work_package.get("hypothesis", "") or "").strip()
                or next((str(unit.get("hypothesis", "") or "").strip() for unit in owned_units if str(unit.get("hypothesis", "") or "").strip()), ""),
                "decision_value": str(work_package.get("decision_value", "") or "").strip()
                or next((str(unit.get("decision_value", "") or "").strip() for unit in owned_units if str(unit.get("decision_value", "") or "").strip()), ""),
                "stop_rule_or_pruning_rationale": "Implementation scope: preserve the work-package owned routes, artifacts, and bounded execution defaults.",
                "requirement_ids": requirement_ids,
                "ref_id": ref_id,
                "reusable_module": reusable_module,
                "depends_on": _dedupe_nonempty([str(item) for item in list(work_package.get("depends_on", []) or [])]),
                "traceable": bool(ref_id and (reusable_module or insight or code_snippet)),
                "code_snippet": code_snippet,
                "insight": insight or (f"Deterministic plan node synthesized from {work_package_id}." if reason else ""),
            }
            )
    all_requirement_ids = _dedupe_nonempty(
        [
            str(item.get("requirement_id", "") or "")
            for item in list(dict(input_payload.get("boundary_requirements", {}) or {}).get("boundary_requirements", []) or [])
            if isinstance(item, dict)
        ]
    )
    if not plan_nodes:
        for index, unit in enumerate(list(input_payload.get("units", []) or []), start=1):
            if not isinstance(unit, dict):
                continue
            unit_id = str(unit.get("unit_id", "") or "").strip() or f"unit_{index:03d}"
            requirement_ids = _requirement_ids_for_unit_ids(input_payload, [unit_id])
            covered_requirement_ids.update(requirement_ids)
            statement = str(unit.get("statement", "") or "").strip() or f"Implement {unit_id}"
            plan_nodes.append(
                {
                    "node_id": f"node_{index:03d}_{_slugify_contract_id(unit_id)}",
                    "parent_node_id": "",
                    "name": statement[:120],
                    "level": "module",
                    "description": "; ".join(
                        _dedupe_nonempty(
                            [statement]
                            + _payload_positive_obligations(unit, limit=6)
                            + [str(item) for item in list(unit.get("expected_artifacts", []) or [])][:4]
                        )
                    )[:1200],
                    "hypothesis": str(unit.get("hypothesis", "") or "").strip(),
                    "decision_value": str(unit.get("decision_value", "") or "").strip(),
                    "stop_rule_or_pruning_rationale": "Implementation scope: preserve the unit-owned routes, artifacts, and bounded execution defaults.",
                    "requirement_ids": requirement_ids,
                    "ref_id": "",
                    "reusable_module": "",
                    "depends_on": [],
                    "traceable": False,
                    "code_snippet": "",
                    "insight": f"Deterministic plan node synthesized from {unit_id}.",
                }
            )
    if formula_obligations_global and plan_nodes:
        core_index = 0
        for index, node in enumerate(plan_nodes):
            text = " ".join(str(node.get(field, "") or "") for field in ("node_id", "name", "description", "hypothesis", "decision_value"))
            if _formula_anchor_text_terms(text).intersection(_formula_anchor_text_terms(" ".join(formula_obligations_global))):
                core_index = index
                break
        node = dict(plan_nodes[core_index])
        node["description"] = _append_text_with_obligations(
            str(node.get("description", "") or ""),
            formula_obligations_global[:6],
            limit=1200,
        )
        node["insight"] = _append_text_with_obligations(
            str(node.get("insight", "") or ""),
            ["Formula/algorithm anchors are mandatory implementation obligations for the executable route."],
            limit=900,
        )
        plan_nodes[core_index] = node
    return PipelinePlanOutput.model_validate(
        {
            "plan_nodes": plan_nodes,
            "coverage_summary": {
                "total_requirements": len(all_requirement_ids),
                "covered_requirements": len(covered_requirement_ids),
                "uncovered_requirement_ids": [
                    requirement_id
                    for requirement_id in all_requirement_ids
                    if requirement_id not in covered_requirement_ids
                ],
            },
        }
    )


def _infer_reference_role(
    supported_scope_items: list[str],
    reusable_modules: list[str],
    insights: list[str],
) -> str:
    haystack = " ".join([*supported_scope_items, *reusable_modules, *insights]).lower()
    for role, hints in _METHOD_SPINE_HINTS.items():
        if any(token in haystack for token in hints):
            return role
    return "supporting_repo"


def _merge_reference_selection_with_surveys(
    state: PaperBenchReproState,
    selection: ReferenceSelectionOutput,
) -> ReferenceSelectionOutput:
    """Preserve prepared repo surveys as actionable references and drop hallucinated refs."""
    if not state.reference_repo_surveys:
        return selection

    valid_ref_ids = _valid_ref_ids_from_state(state)

    relation_by_ref_id: dict[str, ReferenceRelation] = {
        str(item.ref_id or "").strip(): item
        for item in list(selection.reference_relations or [])
        if str(item.ref_id or "").strip() in valid_ref_ids
    }
    relation_order = [
        str(item.ref_id or "").strip()
        for item in list(selection.reference_relations or [])
        if str(item.ref_id or "").strip() in valid_ref_ids
    ]
    actionable_by_id: dict[str, ActionableReference] = {}
    actionable_order: list[str] = []
    survey_by_id = {
        str(getattr(survey, "ref_id", "") or "").strip(): survey
        for survey in list(state.reference_repo_surveys)
        if str(getattr(survey, "ref_id", "") or "").strip()
    }
    for item in list(selection.actionable_references or []):
        ref_id = str(item.ref_id or "").strip()
        if ref_id not in valid_ref_ids:
            continue
        role = str(getattr(relation_by_ref_id.get(ref_id), "reference_role", "") or "").strip().lower()
        survey = survey_by_id.get(ref_id)
        is_actionable = True
        if survey is not None:
            is_actionable, _ = _survey_is_actionable_for_main_plan(state, survey)
        if role in _REFERENCE_BINDING_EXCLUDED_ROLES or not is_actionable:
            continue
        actionable_by_id[ref_id] = item
        actionable_order.append(ref_id)

    for survey in state.reference_repo_surveys:
        ref_id = str(getattr(survey, "ref_id", "") or "").strip()
        if not ref_id:
            continue
        is_actionable, relevance_reason = _survey_is_actionable_for_main_plan(state, survey)
        supported_requirement_ids = _supported_requirement_ids_from_survey(survey)
        reusable_modules = _survey_reusable_modules(survey)
        insights = _dedupe_nonempty(_survey_insights(survey) + [relevance_reason])[:4]
        existing = actionable_by_id.get(ref_id)
        if existing is None and not is_actionable:
            continue
        raw_repository_origin = str(getattr(survey, "repository_origin", "") or "").strip()
        repository_origin = raw_repository_origin.lower().replace("-", "_").replace(" ", "_")
        if repository_origin == "thirdparty":
            repository_origin = "third_party"
        if repository_origin not in {"official", "community", "third_party", "library", "external", "unofficial", "unknown"}:
            repository_origin = "unknown"
        merged_payload = {
            "ref_id": ref_id,
            "title": str((existing.title if existing is not None else "") or getattr(survey, "title", "") or ref_id).strip(),
            "paper_url": str((existing.paper_url if existing is not None else "") or getattr(survey, "paper_url", "") or "").strip(),
            "repository_url": str((existing.repository_url if existing is not None else "") or getattr(survey, "repository_url", "") or "").strip(),
            "repository_origin": (
                str(existing.repository_origin or "").strip().lower()
                if existing is not None
                and str(existing.repository_origin or "").strip().lower() in {"official", "community", "third_party", "library", "external", "unofficial", "unknown"}
                else repository_origin
            ) or "unknown",
            "raw_repository_origin": (
                str(getattr(existing, "raw_repository_origin", "") or "").strip()
                if existing is not None and str(getattr(existing, "raw_repository_origin", "") or "").strip()
                else raw_repository_origin
            ),
            "source_kind": (
                str(getattr(existing, "source_kind", "") or "").strip()
                if existing is not None and str(getattr(existing, "source_kind", "") or "").strip()
                else repository_origin
            ),
            "local_repo_path": str((existing.local_repo_path if existing is not None else "") or getattr(survey, "local_repo_path", "") or "").strip(),
            "default_branch": str((existing.default_branch if existing is not None else "") or getattr(survey, "default_branch", "") or "").strip(),
            "supported_requirement_ids": _dedupe_nonempty(
                [
                    *(list(existing.supported_requirement_ids) if existing is not None else []),
                    *supported_requirement_ids,
                ]
            ),
            "reusable_modules": _dedupe_nonempty(
                [
                    *(list(existing.reusable_modules) if existing is not None else []),
                    *reusable_modules,
                ]
            )[:8],
            "insights": _dedupe_nonempty(
                [
                    *(list(existing.insights) if existing is not None else []),
                    *insights,
                ]
            )[:4],
            "file_tree": str((existing.file_tree if existing is not None else "") or getattr(survey, "file_tree_summary", "") or "").strip(),
            "readme_summary": str((existing.readme_summary if existing is not None else "") or getattr(survey, "readme_summary", "") or "").strip(),
            "top_python_files": _dedupe_nonempty(
                [
                    *(list(existing.top_python_files) if existing is not None else []),
                    *list(getattr(survey, "top_python_files", []) or []),
                ]
            )[:10],
            "likely_reusable_files": _dedupe_nonempty(
                [
                    *(list(existing.likely_reusable_files) if existing is not None else []),
                    *list(getattr(survey, "likely_reusable_files", []) or []),
                ]
            )[:10],
            "protocol_clues": _dedupe_nonempty(
                [
                    *(list(existing.protocol_clues) if existing is not None else []),
                    *list(getattr(survey, "protocol_clues", []) or []),
                ]
            )[:10],
            "requirement_coverage": (
                list(existing.requirement_coverage)
                if existing is not None and list(existing.requirement_coverage)
                else list(getattr(survey, "requirement_coverage", []) or [])
            ),
            "symbol_evidence": (
                list(existing.symbol_evidence)
                if existing is not None and list(getattr(existing, "symbol_evidence", []) or [])
                else list(getattr(survey, "symbol_evidence", []) or [])
            )[:12],
        }
        actionable_by_id[ref_id] = ActionableReference.model_validate(merged_payload)
        if ref_id not in actionable_order:
            actionable_order.append(ref_id)

        supported_scope_items = _dedupe_nonempty(
            [
                *(list(relation_by_ref_id.get(ref_id).supported_scope_items) if ref_id in relation_by_ref_id else []),
                *_survey_supported_scope_items(survey),
            ]
        )
        existing_relation = relation_by_ref_id.get(ref_id)
        survey_reference_role = str(getattr(survey, "reference_role", "") or "").strip()
        inferred_role = _infer_reference_role(
            supported_scope_items,
            list(actionable_by_id[ref_id].reusable_modules),
            list(actionable_by_id[ref_id].insights),
        )
        relation_by_ref_id[ref_id] = ReferenceRelation.model_validate(
            {
                "ref_id": ref_id,
                "supported_scope_items": supported_scope_items,
                "reference_role": (
                    str(existing_relation.reference_role or "").strip()
                    if existing_relation is not None and str(existing_relation.reference_role or "").strip()
                    else survey_reference_role
                    if survey_reference_role
                    else inferred_role
                ),
            }
        )
        if ref_id not in relation_order:
            relation_order.append(ref_id)

    return ReferenceSelectionOutput.model_validate(
        {
            "actionable_references": [
                actionable_by_id[ref_id].model_dump(mode="json")
                for ref_id in actionable_order
                if ref_id in actionable_by_id
            ],
            "reference_relations": [
                relation_by_ref_id[ref_id].model_dump(mode="json")
                for ref_id in relation_order
                if ref_id in relation_by_ref_id and ref_id in actionable_by_id
            ],
        }
    )


def _normalize_reference_selection_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload or {})
    string_fields = (
        "ref_id",
        "title",
        "paper_url",
        "repository_url",
        "repository_origin",
        "raw_repository_origin",
        "source_kind",
        "local_repo_path",
        "default_branch",
        "file_tree",
        "readme_summary",
    )
    list_fields = (
        "supported_requirement_ids",
        "reusable_modules",
        "insights",
        "top_python_files",
        "likely_reusable_files",
        "protocol_clues",
        "requirement_coverage",
        "symbol_evidence",
    )
    references: list[dict[str, Any]] = []
    for item in list(normalized.get("actionable_references", []) or []):
        if not isinstance(item, dict):
            continue
        reference = dict(item)
        for field_name in string_fields:
            if field_name in reference and reference[field_name] is None:
                reference[field_name] = ""
        for field_name in list_fields:
            if reference.get(field_name) is None:
                reference[field_name] = []
        references.append(reference)
    normalized["actionable_references"] = references

    relations: list[dict[str, Any]] = []
    for item in list(normalized.get("reference_relations", []) or []):
        if not isinstance(item, dict):
            continue
        relation = dict(item)
        for field_name in ("ref_id", "reference_role"):
            if field_name in relation and relation[field_name] is None:
                relation[field_name] = ""
        if relation.get("supported_scope_items") is None:
            relation["supported_scope_items"] = []
        relations.append(relation)
    normalized["reference_relations"] = relations
    return normalized


def _autobind_work_package_references(
    state: PaperBenchReproState,
    planning: WorkPackagePlanningOutput,
) -> tuple[WorkPackagePlanningOutput, list[str]]:
    """Best-effort deterministic binding from work packages to prepared reference repos."""
    if not state.reference_repo_surveys or state.reference_selection is None:
        return planning, []
    actionable_ref_ids = {
        str(getattr(reference, "ref_id", "") or "").strip()
        for reference in list(state.reference_selection.actionable_references or [])
        if str(getattr(reference, "ref_id", "") or "").strip()
    }
    relation_by_ref_id = {
        str(getattr(relation, "ref_id", "") or "").strip(): str(getattr(relation, "reference_role", "") or "").strip().lower()
        for relation in list(state.reference_selection.reference_relations or [])
        if str(getattr(relation, "ref_id", "") or "").strip()
    }
    survey_by_id = {
        str(getattr(survey, "ref_id", "") or "").strip(): survey
        for survey in list(state.reference_repo_surveys)
        if str(getattr(survey, "ref_id", "") or "").strip() in actionable_ref_ids
        and relation_by_ref_id.get(str(getattr(survey, "ref_id", "") or "").strip(), "") not in _REFERENCE_BINDING_EXCLUDED_ROLES
    }
    if not survey_by_id:
        patched_packages = [
            work_package.model_copy(update={"reference_ids": []}).model_dump(mode="json")
            for work_package in planning.work_packages
        ]
        stale_count = sum(1 for work_package in planning.work_packages if list(work_package.reference_ids or []))
        original_notes = [str(note) for note in list(planning.planning_notes or [])]
        planning_notes = [
            note
            for note in original_notes
            if "reference_ids from local surveys" not in note
        ]
        removed_note_count = len(original_notes) - len(planning_notes)
        notes = []
        if stale_count:
            notes.append(
                "removed stale work-package reference bindings because no actionable implementation "
                f"references survived relevance filtering: {stale_count} packages"
            )
        if removed_note_count:
            notes.append(
                "removed stale work-package reference-binding notes because no actionable implementation "
                f"references survived relevance filtering: {removed_note_count} notes"
            )
        return (
            WorkPackagePlanningOutput.model_validate(
                {
                    "work_packages": patched_packages,
                    "coverage_summary": planning.coverage_summary.model_dump(mode="json"),
                    "planning_notes": _dedupe_nonempty(planning_notes + notes),
                }
            ),
            notes,
        )
    notes: list[str] = []
    patched_packages: list[dict[str, Any]] = []
    for work_package in planning.work_packages:
        raw_existing = _dedupe_nonempty([str(ref_id) for ref_id in list(work_package.reference_ids or [])])
        existing = [
            ref_id
            for ref_id in raw_existing
            if ref_id in survey_by_id and _survey_grounding_score(work_package, survey_by_id[ref_id]) >= 3
        ]
        dropped = [ref_id for ref_id in raw_existing if ref_id not in existing]
        if existing:
            if dropped:
                patched_packages.append(
                    work_package.model_copy(update={"reference_ids": existing}).model_dump(mode="json")
                )
                notes.append(
                    f"normalized `{work_package.work_package_id}` reference_ids to repo refs only: "
                    + ", ".join(existing[:4])
                )
            else:
                patched_packages.append(work_package.model_dump(mode="json"))
            continue
        ranked = sorted(
            (
                (_survey_grounding_score(work_package, survey), str(survey.ref_id or "").strip())
                for survey in list(survey_by_id.values())
            ),
            key=lambda item: (-item[0], item[1]),
        )
        derived = [ref_id for score, ref_id in ranked if score >= 3 and ref_id][:3]
        patched_packages.append(
            work_package.model_copy(update={"reference_ids": derived}).model_dump(mode="json")
        )
        if derived:
            notes.append(
                f"autobound `{work_package.work_package_id}` reference_ids from local surveys: {', '.join(derived)}"
            )
        elif dropped:
            notes.append(
                f"removed non-repo reference_ids from `{work_package.work_package_id}` before grounding: "
                + ", ".join(dropped[:4])
            )
    return (
        WorkPackagePlanningOutput.model_validate(
            {
                "work_packages": patched_packages,
                "coverage_summary": planning.coverage_summary.model_dump(mode="json"),
                "planning_notes": _dedupe_nonempty(list(planning.planning_notes) + notes),
            }
        ),
        notes,
    )


def _refresh_work_package_reference_bindings(state: PaperBenchReproState) -> tuple[PaperBenchReproState, list[str]]:
    """Re-apply deterministic reference bindings after reference selection/resume refreshes."""
    if state.work_package_planning is None or not state.reference_repo_surveys:
        return state, []
    patched, notes = _autobind_work_package_references(state, state.work_package_planning)
    if not notes:
        return state, []
    state.work_package_planning = patched
    state.temp_data["work_package_planning"] = patched.model_dump(mode="json")
    return state, notes


_WEAK_EVIDENCE_PATH_TOKENS = (
    ".github/",
    "code_of_conduct",
    "contributing",
    "issue_template",
    "pull_request_template",
    "security.md",
    "license",
    "readme",
)


def _evidence_link_is_actionable(link: Any) -> bool:
    path = str(getattr(link, "file_path", "") or "").strip().lower()
    snippet = str(getattr(link, "snippet_preview", "") or "").strip().lower()
    if not path and not snippet:
        return False
    if any(token in path for token in _WEAK_EVIDENCE_PATH_TOKENS):
        return False
    return True


def _critical_grounding_gate_failures(state: PaperBenchReproState) -> list[str]:
    """Return blocking critical-grounding failures after evidence grounding."""
    if state.work_package_planning is None:
        return []
    valid_reference_ids = {
        str(getattr(survey, "ref_id", "") or "").strip()
        for survey in list(state.reference_repo_surveys or [])
        if str(getattr(survey, "ref_id", "") or "").strip()
    }
    bundle_by_package = {item.work_package_id: item for item in state.evidence_bundles}
    failures: list[str] = []
    for work_package in state.work_package_planning.work_packages:
        work_package_id = str(work_package.work_package_id or "").strip()
        if not work_package_id:
            continue
        tags = {str(item or "").strip().lower() for item in list(work_package.tags or []) if str(item or "").strip()}
        goal = str(work_package.goal or "").lower()
        has_reference_scope = any(
            str(ref_id or "").strip() in valid_reference_ids
            for ref_id in list(work_package.reference_ids or [])
        )
        is_critical = bool(tags.intersection(_CRITICAL_GROUNDING_TAGS)) or any(
            token in goal for token in ("method", "dataset", "evaluation", "artifact", "entry", "protocol")
        )
        if not (has_reference_scope and is_critical):
            continue
        bundle = bundle_by_package.get(work_package_id)
        grounding_status = str(bundle.grounding_status if bundle is not None else "").strip().lower()
        actionable_links = [
            link
            for link in list(getattr(bundle, "evidence_links", []) if bundle is not None else [])
            if _evidence_link_is_actionable(link)
        ]
        if grounding_status == "self_contained":
            continue
        if grounding_status == "grounded" and actionable_links:
            continue
        if grounding_status == "grounded":
            failures.append(
                f"critical work package `{work_package_id}` is grounded only by weak doc/community evidence"
            )
            continue
        failures.append(
            f"critical work package `{work_package_id}` is still ungrounded after evidence_grounding"
        )
    return failures


def _pipeline_plan_quality_failures(state: PaperBenchReproState, *, fallback_used: bool) -> list[str]:
    if state.pipeline_plan is None:
        return ["pipeline_plan is missing"]
    failures: list[str] = []
    if fallback_used and not bool(getattr(get_workflow_config(), "allow_plan_fallback_continue", False)):
        failures.append(
            "pipeline_plan used deterministic/provider fallback; rerun this planning stage instead of continuing to generate"
        )
    summary = state.pipeline_plan.coverage_summary
    total = int(summary.total_requirements or 0)
    covered = int(summary.covered_requirements or 0)
    work_package_count = len(list(state.work_package_planning.work_packages if state.work_package_planning else []))
    unit_count = len(list(state.unit_extraction.units if state.unit_extraction else []))
    if total <= 0 and (work_package_count > 0 or unit_count > 0):
        failures.append(
            "pipeline_plan has no boundary requirement coverage model despite existing units/work packages; "
            + f"requirements={total}, units={unit_count}, work_packages={work_package_count}"
        )
    if total > 0 and covered <= 0:
        failures.append(
            f"pipeline_plan covers 0/{total} boundary requirements; continue degraded and repair downstream"
        )
    elif total > 0 and covered < total:
        failures.append(
            f"pipeline_plan leaves boundary requirements uncovered: {covered}/{total}"
        )
    if fallback_used and total > 0 and covered < max(1, int(total * 0.5)):
        failures.append(
            "deterministic fallback pipeline_plan has insufficient requirement coverage: "
            + f"{covered}/{total}"
        )
    if list(state.reference_repo_surveys or []):
        ungrounded_package_ids = [
            str(getattr(bundle, "work_package_id", "") or "").strip()
            for bundle in list(state.evidence_bundles or [])
            if str(getattr(bundle, "grounding_status", "") or "").strip().lower() == "ungrounded"
        ]
        if len(ungrounded_package_ids) > max(3, int(len(list(state.evidence_bundles or [])) * 0.25)):
            failures.append(
                "too many work packages remain ungrounded before pipeline planning: "
                + ", ".join(ungrounded_package_ids[:8])
            )
    failures.extend(_critical_grounding_gate_failures(state))
    failures.extend(_paired_mechanism_pipeline_plan_failures(state))
    return _dedupe_nonempty(failures)


def _paired_mechanism_pipeline_plan_failures(state: PaperBenchReproState) -> list[str]:
    """Require positive coverage for paper-named paired mechanisms."""

    if state.pipeline_plan is None:
        return []
    work_package_planning = getattr(state, "work_package_planning", None)
    source_text = _json_text(
        [
            item.model_dump(mode="json") if hasattr(item, "model_dump") else item
            for item in list(getattr(work_package_planning, "work_packages", []) or [])
        ]
    )
    plan_text = _json_text(state.pipeline_plan.model_dump(mode="json"))
    failures: list[str] = []
    paired_terms = [
        ("ode", "sde"),
        ("deterministic", "stochastic"),
        ("teacher", "student"),
    ]
    for first, second in paired_terms:
        source_has_pair = _contains_token(source_text, first) and _contains_token(source_text, second)
        if not source_has_pair:
            continue
        for term in (first, second):
            if not _contains_token(plan_text, term):
                failures.append(
                    f"pipeline_plan dropped paired mechanism `{term.upper()}` despite work-package coverage"
                )
    return failures


def _json_text(value: Any) -> str:
    strings: list[str] = []

    def _collect(item: Any) -> None:
        if isinstance(item, str):
            strings.append(item)
            return
        if isinstance(item, dict):
            for child in item.values():
                _collect(child)
            return
        if isinstance(item, (list, tuple, set)):
            for child in item:
                _collect(child)

    _collect(value)
    if not strings:
        return str(value or "").lower()
    return "\n".join(strings).lower().replace("\\n", "\n").replace("\\r", "\n")


def _contains_token(text: str, token: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(token.lower())}(?![a-z0-9])", text) is not None


def _mark_pipeline_plan_blocked(state: PaperBenchReproState, failures: list[str]) -> None:
    backlog = state.temp_data.setdefault("degraded_backlog", [])
    payload = {
        "stage": "pipeline_plan",
        "code": "pipeline_plan_quality_degraded_continue",
        "message": "pipeline plan quality gate reported issues; continuing so validation/repair can address them",
        "reasons": list(failures or [])[:16],
    }
    if isinstance(backlog, list) and payload not in backlog:
        backlog.append(payload)
    state.temp_data.setdefault("stage_reviews", {})["pipeline_plan"] = {
        "stage_name": "pipeline_plan",
        "review_status": "degraded_best_effort",
        "validation_errors": list(failures or []),
        "next_action": "continue_to_architecture_and_repair",
    }
    state.error_message = "; ".join(failures[:8])


def _dedupe_nonempty(items: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for item in items:
        rendered = str(item or "").strip()
        if not rendered or rendered in seen:
            continue
        seen.add(rendered)
        ordered.append(rendered)
    return ordered


_IMPLEMENTATION_INVENTORY_KEYS = {
    "obligation_matrix",
    "experiment_inventory",
    "environment_inventory",
    "dataset_inventory",
    "method_inventory",
    "policy_inventory",
    "model_inventory",
    "baseline_inventory",
    "metric_inventory",
    "measurement_inventory",
    "parameter_inventory",
    "result_trend_inventory",
    "artifact_inventory",
    "result_artifact_inventory",
    "implementation_obligation_inventory",
    "implementation_surface_inventory",
    "closure_inventory",
}


def _implementation_inventory_map(payload: dict[str, list[str]]) -> dict[str, list[str]]:
    """Keep only inventories that describe positive implementation coverage."""
    return {
        key: values
        for key, values in dict(payload or {}).items()
        if str(key or "").strip() in _IMPLEMENTATION_INVENTORY_KEYS
    }


def _unit_positive_obligations(unit: Any, *, limit: int = 8) -> list[str]:
    """Prefer active routes, interfaces, artifacts, and implementation actions."""
    items: list[str] = []
    for field_name in ("runtime_interfaces", "expected_artifacts", "implementation_surfaces"):
        for item in list(getattr(unit, field_name, []) or []):
            value = str(item or "").strip()
            if value:
                items.append(value)
    for item in list(getattr(unit, "code_obligations", []) or []):
        value = str(item or "").strip()
        if value and any(
            token in value.lower()
            for token in (
                "implement",
                "实现",
                "expose",
                "create",
                "write",
                "support",
                "maintain",
                "compute",
                "record",
                "save",
                "提供",
                "创建",
                "写出",
                "支持",
                "记录",
                "保存",
            )
        ):
            items.append(value)
    return _dedupe_nonempty(items)[:limit]


def _unit_scope_label(unit: Any) -> str:
    unit_id = str(getattr(unit, "unit_id", "") or "").strip()
    statement = str(getattr(unit, "statement", "") or "").strip()
    pieces = _unit_positive_obligations(unit, limit=5)
    if not pieces and statement:
        pieces = [statement]
    prefix = f"{unit_id}: " if unit_id else ""
    return prefix + "; ".join(pieces[:5])


def _payload_positive_obligations(unit: dict[str, Any], *, limit: int = 8) -> list[str]:
    """Dict variant used by deterministic pipeline-plan projection."""
    items: list[str] = []
    for field_name in ("runtime_interfaces", "expected_artifacts", "implementation_surfaces"):
        for item in list(unit.get(field_name, []) or []):
            value = str(item or "").strip()
            if value:
                items.append(value)
    for item in list(unit.get("code_obligations", []) or []):
        value = str(item or "").strip()
        if value and any(
            token in value.lower()
            for token in (
                "implement",
                "实现",
                "expose",
                "create",
                "write",
                "support",
                "maintain",
                "compute",
                "record",
                "save",
                "提供",
                "创建",
                "写出",
                "支持",
                "记录",
                "保存",
            )
        ):
            items.append(value)
    return _dedupe_nonempty(items)[:limit]


def _positive_plan_focus(
    *,
    work_package_id: str,
    goal: str,
    owned_units: list[dict[str, Any]],
    obligations: list[str],
    artifacts: list[str],
) -> str:
    """Summarize what the plan node owns using implementation-facing fields only."""
    focus_items = _dedupe_nonempty(
        [goal]
        + obligations[:5]
        + artifacts[:4]
        + [
            str(surface)
            for unit in owned_units
            for surface in list(unit.get("implementation_surfaces", []) or [])
        ][:4]
    )
    if focus_items:
        return "; ".join(focus_items[:6])
    unit_ids = _dedupe_nonempty([str(unit.get("unit_id", "") or "") for unit in owned_units])
    if unit_ids:
        return f"Implement the paper-derived contract for {', '.join(unit_ids[:4])}."
    return f"Implement the paper-derived contract for {work_package_id}."


def _positive_hypothesis_for_plan_node(
    *,
    work_package_id: str,
    goal: str,
    owned_units: list[dict[str, Any]],
    obligations: list[str],
    artifacts: list[str],
) -> str:
    focus = _positive_plan_focus(
        work_package_id=work_package_id,
        goal=goal,
        owned_units=owned_units,
        obligations=obligations,
        artifacts=artifacts,
    )
    return f"The {work_package_id} route exercises the paper-derived implementation contract: {focus}"


def _positive_decision_value_for_plan_node(
    *,
    work_package_id: str,
    goal: str,
    owned_units: list[dict[str, Any]],
    obligations: list[str],
    artifacts: list[str],
) -> str:
    focus_items = _dedupe_nonempty(
        artifacts[:4]
        + obligations[:4]
        + [goal]
        + [
            str(interface)
            for unit in owned_units
            for interface in list(unit.get("runtime_interfaces", []) or [])
        ][:4]
    )
    focus = ", ".join(focus_items[:5]) if focus_items else work_package_id
    return f"Determines whether the repository exposes runnable, reviewable coverage for {focus}."


def _positive_scope_for_plan_node(work_package_id: str) -> str:
    return (
        "Implementation scope: own the paper-derived runnable route, interfaces, artifacts, "
        f"and bounded execution defaults for {work_package_id}."
    )


def _record_degraded_planning_issue(
    state: PaperBenchReproState,
    *,
    stage: str,
    code: str,
    message: str,
    reasons: list[str],
) -> None:
    backlog = state.temp_data.setdefault("degraded_backlog", [])
    payload = {
        "stage": stage,
        "code": code,
        "message": message,
        "reasons": list(reasons or [])[:16],
    }
    if isinstance(backlog, list) and payload not in backlog:
        backlog.append(payload)


def _has_refinement_semantic_signal(*values: Any) -> bool:
    lowered = " ".join(str(value or "") for value in values).lower().replace("_", " ").replace("-", " ")
    weak_generic_patterns = (
        "training or refinement",
        "training/refinement",
        "optimization or refinement",
        "experiment/config/evaluation/refinement",
        "train/evaluate/refine",
        "pretraining, refinement",
        "training, pretraining, refinement",
        "training, refinement, or evaluation",
    )
    for pattern in weak_generic_patterns:
        lowered = lowered.replace(pattern, " ")
    strong_patterns = (
        "test time adaptation",
        "test time training",
        "iterative refinement",
        "refinement stage",
        "refinement module",
        "refinement network",
        "refinement policy",
        "refinement step",
        "refinement loss",
        "refining states",
        "refine states",
        "roll in",
        "rollin",
    )
    if any(pattern in lowered for pattern in strong_patterns):
        return True
    return bool(re.search(r"\b(adapt|adaptive|adaptation|refiner|refining)\b", lowered))


def _tokenize_text(*values: Any) -> set[str]:
    tokens: set[str] = set()
    for value in values:
        text = str(value or "").lower().replace("-", "_")
        tokens.update(
            span
            for span in re.findall(r"[a-z0-9]+", text)
            if len(span) >= 2
        )
        for cjk_span in re.findall(r"[\u4e00-\u9fff]+", text):
            if len(cjk_span) >= 2:
                tokens.add(cjk_span)
                for ngram_size in (2, 3, 4):
                    if len(cjk_span) >= ngram_size:
                        tokens.update(
                            cjk_span[index : index + ngram_size]
                            for index in range(0, len(cjk_span) - ngram_size + 1)
                        )
        for token in re.split(r"[^a-z0-9_\u4e00-\u9fff]+", text):
            parts = [part.strip(".,;:()[]{}'\"`") for part in token.replace("/", "_").split("_")]
            for part in parts:
                if len(part) >= 2 and not re.fullmatch(r"[\u4e00-\u9fff]+", part):
                    tokens.add(part)
    return tokens


def _boundary_requirement_unit_links(state: PaperBenchReproState, requirement_payload: dict[str, Any]) -> list[str]:
    units = list(state.unit_extraction.units if state.unit_extraction else [])
    if not units:
        return []
    raw_source_ids = _dedupe_nonempty(
        [str(value) for value in list(requirement_payload.get("source_unit_ids", []) or [])]
    )
    source_id_set = set(raw_source_ids)
    linked: list[str] = []
    for unit in units:
        unit_id = str(unit.unit_id or "").strip()
        if not unit_id:
            continue
        unit_source_ids = {
            unit_id,
            *[str(item).strip() for item in list(unit.source_paragraph_ids or []) if str(item).strip()],
            *[str(item).strip() for item in list(unit.citation_refs or []) if str(item).strip()],
        }
        if source_id_set.intersection(unit_source_ids):
            linked.append(unit_id)
    requirement_terms = _tokenize_text(
        requirement_payload.get("title", ""),
        requirement_payload.get("scope", ""),
        requirement_payload.get("description", ""),
        *list(requirement_payload.get("acceptance_criteria", []) or []),
        *raw_source_ids,
    )
    ranked: list[tuple[int, int, str]] = []
    for unit in units:
        unit_id = str(unit.unit_id or "").strip()
        if not unit_id or unit_id in linked:
            continue
        unit_terms = _tokenize_text(
            unit_id,
            unit.statement,
            *list(unit.source_paragraph_ids or []),
            *list(unit.implementation_surfaces or []),
            *list(unit.code_obligations or []),
            *list(unit.runtime_interfaces or []),
            *list(unit.expected_artifacts or []),
        )
        overlap = len(requirement_terms.intersection(unit_terms))
        if overlap > 0:
            priority = 0 if unit_id.startswith("unit_") else 1 if unit_id.startswith("paper_semantic_chunk_") else 2
            ranked.append((overlap, priority, unit_id))
    linked.sort(key=lambda unit_id: (2 if unit_id.startswith("paper_contract_") or unit_id.startswith("paper_") and not unit_id.startswith("paper_semantic_chunk_") else 0, unit_id))
    ranked.sort(key=lambda item: (item[1], -item[0], item[2]))
    return _dedupe_nonempty([*linked, *[unit_id for _score, _priority, unit_id in ranked[:8]]])[:8]


def _synthesize_boundary_requirements_from_units(state: PaperBenchReproState) -> dict[str, Any]:
    """Derive paper-specific boundary requirements when the LLM returns none."""

    def _verification_description(item: Any) -> str:
        if isinstance(item, dict):
            return str(item.get("description", "") or "").strip()
        description = getattr(item, "description", "")
        if description:
            return str(description).strip()
        if hasattr(item, "model_dump"):
            payload = item.model_dump(mode="json")
            if isinstance(payload, dict):
                return str(payload.get("description", "") or "").strip()
        return ""

    units = [
        unit
        for unit in list(state.unit_extraction.units if state.unit_extraction else [])
        if str(getattr(unit, "unit_id", "") or "").strip()
        and str(getattr(unit, "status", "active") or "active").strip().lower() == "active"
    ]
    requirements: list[dict[str, Any]] = []
    scope_items: list[str] = []
    for index, unit in enumerate(units[:18], start=1):
        unit_id = str(unit.unit_id or "").strip()
        statement = str(unit.statement or "").strip()
        if not statement:
            continue
        unit_type = str(unit.type or "experiment").strip().lower()
        surfaces = _dedupe_nonempty(
            [
                *[str(item) for item in list(unit.implementation_surfaces or [])],
                *[str(item) for item in list(unit.suggested_module_kinds or [])],
            ]
        )
        criteria = _dedupe_nonempty(
            [
                *_unit_positive_obligations(unit, limit=3),
                *[_verification_description(item) for item in list(unit.verification_targets or [])[:2]],
            ]
        )[:4]
        if not criteria:
            criteria = [statement]
        scope = surfaces[0] if surfaces else unit_type
        category = "experiment"
        if unit_type in {"method", "metric", "protocol", "task", "baseline", "data", "dataset"}:
            category = unit_type
        requirements.append(
            {
                "requirement_id": f"req_unit_{index:03d}",
                "title": statement[:120],
                "category": category,
                "scope": scope,
                "description": statement,
                "source_unit_ids": [unit_id],
                "acceptance_criteria": criteria,
            }
        )
        scope_items.extend([statement, *surfaces[:3]])
    return {
        "boundary_requirements": requirements,
        "requirement_scope_items": _dedupe_nonempty(scope_items)[:40],
    }


def _normalize_repo_path(path: str) -> str:
    normalized = str(path or "").strip().replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized.strip("/")


def _semantic_repo_path_key(path: str) -> str:
    normalized = _normalize_repo_path(path)
    if not normalized:
        return ""
    lowered = normalized.lower()
    if lowered in {"main.py", "readme.md", "requirements.txt", "pyproject.toml"}:
        return lowered
    if lowered.startswith("tests/") or "/tests/" in lowered:
        return lowered
    if lowered.startswith("scripts/"):
        return lowered
    parts = lowered.split("/")
    stem = parts[-1].rsplit(".", 1)[0] if "." in parts[-1] else parts[-1]
    stem = stem.replace("-", "_")
    parent = parts[-2] if len(parts) > 1 else ""
    singular_parent = parent[:-1] if parent.endswith("s") else parent
    if len(parts) == 2 and parts[0] == "src" and stem.endswith("_algorithm"):
        return f"src/algorithm/{stem.removesuffix('_algorithm')}"
    if parent in {"algorithm", "algorithms", "method", "methods", "model", "models", "module", "modules"}:
        return f"src/{singular_parent}/{stem}"
    if stem.endswith("_algorithm"):
        return f"src/algorithm/{stem.removesuffix('_algorithm')}"
    if stem.startswith("algorithm_"):
        return f"src/algorithm/{stem.removeprefix('algorithm_')}"
    return lowered


_PACKAGE_MODULE_ROLE_ALIASES: dict[str, str] = {
    "ablation": "baselines",
    "ablations": "baselines",
    "agent": "agents",
    "agents": "agents",
    "artifact": "reporting",
    "artifacts": "reporting",
    "baseline": "baselines",
    "baselines": "baselines",
    "env": "environments",
    "envs": "environments",
    "environment": "environments",
    "environments": "environments",
    "evaluate": "metrics",
    "evaluation": "metrics",
    "experiment": "experiments",
    "experiments": "experiments",
    "explain": "explainers",
    "explainer": "explainers",
    "explainers": "explainers",
    "explanation": "explainers",
    "metric": "metrics",
    "metrics": "metrics",
    "method": "methods",
    "methods": "methods",
    "model": "models",
    "models": "models",
    "policies": "agents",
    "policy": "agents",
    "protocol": "experiments",
    "protocols": "experiments",
    "report": "reporting",
    "reporting": "reporting",
    "reports": "reporting",
    "rollout": "rollout",
    "rollouts": "rollout",
    "train": "training",
    "trainer": "training",
    "training": "training",
    "trajectories": "rollout",
    "trajectory": "rollout",
}

_NON_PACKAGE_SRC_DIRS = {
    "algorithm",
    "algorithms",
    "core",
    "data",
    "datasets",
    "experiments",
    "method",
    "methods",
    "model",
    "models",
    "module",
    "modules",
    "reporting",
    "tests",
}

_GROUPED_SRC_LAYOUT: dict[str, str] = {
    "agent": "methods",
    "agents": "methods",
    "artifact": "core",
    "artifact_contract": "core",
    "artifacts": "reporting",
    "baseline": "methods",
    "baselines": "methods",
    "contract": "core",
    "contracts": "core",
    "data": "data",
    "dataset": "data",
    "dataset_registry": "data",
    "environment": "data",
    "environment_registry": "data",
    "environments": "data",
    "evaluation": "experiments",
    "experiment": "experiments",
    "experiment_registry": "experiments",
    "experiments": "experiments",
    "explainer": "methods",
    "explainers": "methods",
    "method": "methods",
    "method_registry": "methods",
    "methods": "methods",
    "model": "methods",
    "models": "methods",
    "plotting": "reporting",
    "refinement": "methods",
    "report": "reporting",
    "reporting": "reporting",
    "sweep": "experiments",
    "sweep_registry": "experiments",
    "training": "experiments",
    "trajectories": "data",
    "trend": "reporting",
    "trend_assertions": "reporting",
}


def _default_grouped_src_path(path: str) -> str:
    """Map flat src modules to the default grouped source layout."""
    normalized = _normalize_repo_path(path)
    parts = normalized.split("/")
    if len(parts) != 2 or parts[0].lower() != "src" or not parts[1].endswith(".py"):
        return normalized
    stem = parts[1].rsplit(".", 1)[0].lower().replace("-", "_")
    if stem == "__init__":
        return normalized
    group = _GROUPED_SRC_LAYOUT.get(stem)
    if not group:
        if any(token in stem for token in ("data", "dataset", "environment", "trajectory")):
            group = "data"
        elif any(token in stem for token in ("eval", "experiment", "sweep", "train")):
            group = "experiments"
        elif any(token in stem for token in ("artifact", "plot", "report", "trend", "metric")):
            group = "reporting"
        elif any(token in stem for token in ("contract", "schema", "config")):
            group = "core"
        else:
            group = "methods"
    return f"src/{group}/{stem}.py"


def _module_role_alias(stem: str) -> str:
    normalized = str(stem or "").strip().lower().replace("-", "_")
    return _PACKAGE_MODULE_ROLE_ALIASES.get(normalized, normalized)


def _detect_src_package_root(paths: list[str]) -> str:
    counts: dict[str, int] = {}
    has_init: set[str] = set()
    for path in paths:
        normalized = _normalize_repo_path(path).lower()
        parts = normalized.split("/")
        if len(parts) >= 3 and parts[0] == "src" and parts[1] not in _NON_PACKAGE_SRC_DIRS:
            package = parts[1]
            root = f"src/{package}"
        elif len(parts) >= 2 and parts[0] not in _NON_PACKAGE_SRC_DIRS.union({"configs", "scripts"}):
            package = parts[0]
            root = package
        else:
            continue
        if not package.replace("_", "").isalnum() or package[0].isdigit():
            continue
        if normalized.endswith(".py"):
            counts[root] = counts.get(root, 0) + 1
        if parts[-1] == "__init__.py":
            has_init.add(root)
    candidates = [
        root
        for root, count in counts.items()
        if root in has_init or count >= 4
    ]
    if not candidates:
        return ""
    return sorted(candidates, key=lambda root: (root not in has_init, -counts.get(root, 0), root))[0]


def _package_module_alias_map(paths: list[str], package_root: str) -> dict[str, str]:
    package_root = _normalize_repo_path(package_root).lower()
    if not package_root:
        return {}
    aliases: dict[str, str] = {}
    for path in paths:
        normalized = _normalize_repo_path(path)
        lowered = normalized.lower()
        if not lowered.startswith(package_root + "/") or not lowered.endswith(".py"):
            continue
        stem = lowered.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        if stem == "__init__":
            continue
        aliases.setdefault(stem, normalized)
        aliases.setdefault(_module_role_alias(stem), normalized)
    return aliases


def _align_repo_path_to_package_layout(
    path: str,
    *,
    package_root: str = "",
    package_alias_map: dict[str, str] | None = None,
) -> str:
    normalized = _normalize_repo_path(path)
    package_root = _normalize_repo_path(package_root)
    if not normalized or not package_root:
        return normalized
    parts = normalized.split("/")
    if len(parts) == 2 and parts[0].lower() == "src" and parts[1].lower().endswith(".py"):
        stem = parts[1].rsplit(".", 1)[0].lower().replace("-", "_")
        if stem == "__init__":
            return normalized
        aliases = dict(package_alias_map or {})
        return aliases.get(_module_role_alias(stem)) or aliases.get(stem) or normalized
    return normalized


def _canonicalize_semantic_repo_paths(
    paths: list[str],
    *,
    package_root: str = "",
    package_alias_map: dict[str, str] | None = None,
) -> list[str]:
    ordered: list[str] = []
    seen_keys: set[str] = set()
    for raw_path in paths:
        normalized = _align_repo_path_to_package_layout(
            raw_path,
            package_root=package_root,
            package_alias_map=package_alias_map,
        )
        if not _is_valid_source_repo_path(normalized):
            continue
        key = _semantic_repo_path_key(normalized)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        ordered.append(
            _canonical_repo_path(
                normalized,
                package_root=package_root,
                package_alias_map=package_alias_map,
            )
        )
    return _dedupe_nonempty(ordered)


def _canonical_repo_path(
    path: str,
    *,
    package_root: str = "",
    package_alias_map: dict[str, str] | None = None,
) -> str:
    normalized = _align_repo_path_to_package_layout(
        path,
        package_root=package_root,
        package_alias_map=package_alias_map,
    )
    if not normalized:
        return ""
    key = _semantic_repo_path_key(normalized)
    for singular, plural in {
        "algorithm": "algorithms",
        "method": "methods",
        "model": "models",
        "module": "modules",
    }.items():
        prefix = f"src/{singular}/"
        if key.startswith(prefix):
            stem = key.removeprefix(prefix).strip("/")
            if stem:
                return f"src/{plural}/{stem}.py"
    return normalized


def _looks_like_repo_relative_path(path: str) -> bool:
    normalized = _normalize_repo_path(path)
    if not normalized or normalized.endswith("/") or " " in normalized:
        return False
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._/-")
    return all(char in allowed for char in normalized)


def _is_valid_source_repo_path(path: str) -> bool:
    normalized = _normalize_repo_path(path)
    if not _looks_like_repo_relative_path(normalized):
        return False
    if _looks_like_contract_output_path(normalized):
        return False
    if not _looks_like_implementation_path(normalized):
        return False
    lowered = normalized.lower()
    if any(lowered.startswith(prefix) for prefix in _ARTIFACT_ROOT_PREFIXES):
        return False
    if "result" in lowered and lowered.endswith(".py") and _looks_like_encoded_artifact_name(lowered.rsplit(".", 1)[0]):
        return False
    return True


def _looks_like_implementation_path(path: str) -> bool:
    normalized = _normalize_repo_path(path)
    if not normalized or normalized.endswith("/"):
        return False
    name = normalized.rsplit("/", 1)[-1].lower()
    if name in {"readme.md", "requirements.txt", "pyproject.toml", "setup.py", "setup.cfg"}:
        return True
    if "." not in name:
        return normalized in {"main", "src", "configs"}
    suffix = name.rsplit(".", 1)[-1]
    return suffix in {"py", "md", "txt", "toml", "yaml", "yml", "json", "ini", "cfg", "sh"}


def _infer_file_kind(path: str) -> str:
    normalized = _normalize_repo_path(path).lower()
    if normalized.endswith(".md"):
        return "doc"
    if normalized.endswith((".toml", ".yaml", ".yml", ".json", ".ini", ".cfg", ".txt")):
        return "config"
    if normalized.endswith(".sh"):
        return "script"
    if normalized.startswith("tests/") or normalized.endswith("_test.py") or normalized.startswith("test_"):
        return "test"
    return "source"


def _normalize_architecture_blueprint_kind(kind: Any, path: str = "") -> str:
    """Coerce loose model-emitted blueprint kinds into schema-supported literals."""
    rendered = str(kind or "").strip().lower()
    if rendered in {"source", "test", "config", "doc", "script"}:
        return rendered
    if rendered in {"documentation", "readme", "guide", "notes"}:
        return "doc"
    if rendered in {"artifact", "output", "result", "report"}:
        inferred = _infer_file_kind(path)
        return "doc" if inferred == "doc" else "config" if inferred == "config" else "source"
    if rendered in {"dependency_manifest", "manifest", "requirements", "lockfile"}:
        return "config"
    if rendered in {"prompt", "template"}:
        inferred = _infer_file_kind(path)
        return "doc" if inferred == "doc" else "config" if inferred == "config" else "source"
    return _infer_file_kind(path)


def _normalize_architecture_implementation_strategy(strategy: Any) -> str:
    """Coerce loose strategy labels into ArchitectureFileBlueprint literals."""
    rendered = str(strategy or "").strip().lower()
    if rendered in {"new", "adapted", "reused"}:
        return rendered
    normalized = rendered.replace("_", "-").replace(" ", "-")
    if normalized in {"new", "new-code", "from-scratch", "scratch", "fresh", "implement"}:
        return "new"
    if normalized in {"reused", "reuse", "re-use", "existing", "copied", "copy", "unchanged"}:
        return "reused"
    if normalized in {
        "adapt",
        "adapted",
        "paper",
        "grounded",
        "paper-grounded",
        "paper-based",
        "paper-derived",
        "from-paper",
        "reference-grounded",
        "ref-grounded",
    }:
        return "adapted"
    return "adapted"


def _infer_file_purpose(path: str) -> str:
    normalized = _normalize_repo_path(path).lower()
    if normalized == "main.py":
        return "Canonical experiment entrypoint."
    if normalized.endswith("readme.md"):
        return "Repository usage and experiment instructions."
    if normalized.endswith(("requirements.txt", "pyproject.toml")):
        return "Dependency and environment declaration."
    if normalized.endswith((".yaml", ".yml", ".json", ".toml")):
        return "Runtime or experiment configuration surface."
    if "refinement" in normalized:
        return "Refinement / adaptation algorithm surface."
    if "training" in normalized or "trainer" in normalized:
        return "Training loop and optimization protocol surface."
    if "evaluation" in normalized or "metrics" in normalized or "metric" in normalized:
        return "Evaluation metrics and reporting surface."
    if "dataset" in normalized or "data" in normalized or "loader" in normalized:
        return "Dataset loading and preprocessing surface."
    if "baseline" in normalized or "methods" in normalized:
        return "Method registry and baseline comparison surface."
    if "model" in normalized or "adapter" in normalized or "unet" in normalized or "ldm" in normalized or "ddpm" in normalized:
        return "Core model and adaptor implementation surface."
    if "artifact" in normalized or "report" in normalized or "plot" in normalized:
        return "Artifact writing and result reporting surface."
    return f"Implement {path}."


def _source_path_for_surface_hint(value: str) -> str:
    normalized = _normalize_repo_path(value)
    lowered = normalized.lower().replace("-", "_").replace("/", "_").strip("_")
    if _looks_like_contract_output_path(normalized):
        return ""
    if _is_valid_source_repo_path(normalized) and (
        "/" in normalized
        or "." in normalized.rsplit("/", 1)[-1]
        or normalized in {"main", "src", "configs"}
    ):
        return normalized
    explicit_prefixes = ("environment_registry", "experiment_registry", "metrics_logger")
    for prefix in explicit_prefixes:
        if lowered.startswith(prefix) or prefix in lowered:
            return _semantic_fallback_source_path(prefix)
    if lowered in _IMPLEMENTATION_SURFACE_PATHS:
        return _semantic_fallback_source_path(value)
    for tokens, _path in _SURFACE_HINT_PRIORITY:
        if any(token in lowered for token in tokens):
            return _semantic_fallback_source_path(value)
    parent = normalized.rsplit("/", 1)[0].lower()
    basename = normalized.rsplit("/", 1)[-1]
    stem = basename.rsplit(".", 1)[0].lower().replace("-", "_")
    if parent in {"configs", "config"}:
        return _semantic_fallback_source_path(stem or normalized)
    if parent.startswith("src/") and len(stem) > 28:
        for tokens, _path in _SURFACE_HINT_PRIORITY:
            if any(token in stem for token in tokens):
                return _semantic_fallback_source_path(stem)
    if _is_valid_source_repo_path(normalized):
        return normalized
    for token, _path in _IMPLEMENTATION_SURFACE_PATHS.items():
        if token and token in lowered:
            return _semantic_fallback_source_path(value)
    return ""


def _sanitize_architecture_source_paths(paths: list[str]) -> list[str]:
    sanitized: list[str] = []
    for path in paths:
        normalized = _normalize_repo_path(path)
        hinted = _source_path_for_surface_hint(normalized)
        candidate = hinted or normalized
        if _is_valid_source_repo_path(candidate):
            sanitized.append(candidate)
    return _canonicalize_semantic_repo_paths(sanitized)


def _normalize_contract_list(items: list[Any] | tuple[Any, ...] | None) -> list[str]:
    return _dedupe_nonempty([str(item) for item in list(items or [])])


def _normalize_inventory_map(payload: dict[str, Any] | None) -> dict[str, list[str]]:
    normalized: dict[str, list[str]] = {}
    for key, values in dict(payload or {}).items():
        normalized_key = str(key or "").strip()
        if normalized_key not in _IMPLEMENTATION_INVENTORY_KEYS:
            continue
        normalized[normalized_key] = _normalize_contract_list(values if isinstance(values, list) else [values])
    return normalized


_EXPERIMENT_NAME_RE = re.compile(
    r"\b(?:Experiment|Exp\.?|Study|Evaluation|Ablation)\s+"
    r"(?:[IVX]{1,6}(?![A-Za-z])(?:\s*-\s*[IVX]{1,6}(?![A-Za-z]))?|\d+[A-Za-z]?)"
    r"(?:(?:\s*:\s*|\s+[–—-]\s+)[A-Za-z0-9][^.;\n]{0,80})?",
    re.I,
)
_TABLE_FIGURE_RE = re.compile(r"\b(?:Table|Figure|Fig\.?)\s+\d+[A-Za-z]?\b", re.I)
_QUOTED_VARIANT_RE = re.compile(r"['\"]([^'\"]{2,60})['\"]")
_CAMEL_OR_VERSIONED_NAME_RE = re.compile(
    r"\b[A-Z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)*(?:-v\d+)?\b"
)
_ACRONYM_NAME_RE = re.compile(r"\b[A-Z][A-Z0-9]{1,8}(?:[-_][A-Z0-9]+)?\b")
_TITLE_PHRASE_RE = re.compile(
    r"\b[A-Z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)*(?:\s+(?:[A-Z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)*|\d+)){1,4}\b"
)
_TASK_NOUN_PHRASE_RE = re.compile(
    r"\b([A-Za-z][A-Za-z0-9+_.-]*(?:\s+[A-Za-z][A-Za-z0-9+_.-]*){0,4})\s+"
    r"(?:environment|environments|env|task|tasks|game|games|benchmark|benchmarks|application|applications)\b",
    re.I,
)
_PAREN_LIST_RE = re.compile(r"\(([^()]{3,180})\)")

_BASELINE_HINTS = {
    "baseline",
    "baselines",
    "comparison",
    "compare",
    "compares",
    "compared",
    "comparing",
    "versus",
    "vs",
    "against",
    "variant",
    "variants",
    "ablation",
    "method",
    "methods",
}
_MEASUREMENT_HINTS = {
    "metric",
    "metrics",
    "measure",
    "measurement",
    "score",
    "scores",
    "reward",
    "return",
    "accuracy",
    "f1",
    "auc",
    "loss",
    "error",
    "curve",
    "curves",
    "fidelity",
    "runtime",
    "time",
    "cost",
    "sample",
    "samples",
    "throughput",
}
_ENVIRONMENT_HINTS = {
    "environment",
    "environments",
    "env",
    "task",
    "tasks",
    "dataset",
    "datasets",
    "benchmark",
    "benchmarks",
    "simulator",
    "simulation",
}
_GENERIC_INVENTORY_NAMES = {
    "Experiment",
    "Exp",
    "Study",
    "Evaluation",
    "Ablation",
    "Table",
    "Figure",
    "Fig",
    "Section",
    "Appendix",
    "Algorithm",
    "Result",
    "Results",
    "Method",
    "Methods",
    "Baseline",
    "Baselines",
    "Dataset",
    "Task",
    "Tasks",
    "Environment",
    "Environments",
    "The",
    "This",
    "That",
    "We",
    "Our",
    "our",
    "AI",
    "TTCP",
    "B-line",
    "For",
    "From",
    "Given",
    "Input",
    "Output",
    "Implementation",
    "Implement",
    "Represent",
    "Expose",
    "Preserve",
    "Create",
    "Generated",
    "Code",
}
_INVENTORY_PHRASE_STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "by",
    "for",
    "from",
    "in",
    "into",
    "of",
    "on",
    "or",
    "the",
    "this",
    "to",
    "using",
    "with",
}
_GENERIC_PHRASE_HEADS = {
    "all",
    "any",
    "baseline",
    "benchmarks",
    "code",
    "comparison",
    "default",
    "different",
    "each",
    "experiment",
    "experiments",
    "generated",
    "generic",
    "implementation",
    "input",
    "method",
    "methods",
    "output",
    "paper",
    "result",
    "results",
    "same",
    "selected",
    "task",
    "tasks",
    "we",
    "preserve",
    "expose",
    "representative",
    "simulated",
    "realworld",
    "agent",
    "policy",
    "environment",
    "environments",
    "first",
    "second",
    "third",
    "additionally",
    "since",
    "technique",
    "detail",
    "details",
    "both",
    "section",
    "these",
    "they",
    "our",
    "design",
    "discovery",
    "flaws",
    "step",
    "calculate",
    "cheng",
    "raff",
    "drl",
    "rl",
    "s",
    "algorithm",
    "algorithms",
}
_VERSIONED_ENV_PHRASE_RE = re.compile(
    r"\b[A-Z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)*-v\d+\b"
)
_COMMON_ENV_PHRASE_RE = re.compile(
    r"\b(?:Hopper|Walker2d|Reacher|HalfCheetah|MountainCar(?:Continuous)?|Ant|Humanoid|Swimmer|Selfish Mining|"
    r"CAGE(?: Challenge)?(?: 2)?|CybORG|Autonomous Driving|MetaDrive|Malware Mutation|MalConv)\b",
    re.IGNORECASE,
)


def _bounded_inventory_items(items: list[str], limit: int = 18) -> list[str]:
    """Keep inventories useful for prompts without turning them into paper summaries."""
    return _dedupe_nonempty(
        [
            str(item or "").strip().strip(" ,.;:")
            for item in list(items or [])
            if 2 <= len(str(item or "").strip()) <= 140
        ]
    )[:limit]


def _normalize_environment_item(value: str) -> str:
    cleaned = _clean_inventory_phrase(str(value or "").replace("_", "-"))
    if not cleaned:
        return ""
    lower = cleaned.lower()
    aliases = {
        "hopper-v3": "Hopper-v3",
        "hopper": "Hopper",
        "walker2d-v3": "Walker2d-v3",
        "walker2d": "Walker2d",
        "reacher-v2": "Reacher-v2",
        "reacher": "Reacher",
        "halfcheetah-v3": "HalfCheetah-v3",
        "halfcheetah": "HalfCheetah",
        "mountaincarcontinuous-v0": "MountainCarContinuous-v0",
        "mountaincarcontinuous": "MountainCarContinuous",
        "selfish mining": "selfish mining",
        "cage": "CAGE Challenge 2",
        "cage challenge": "CAGE Challenge 2",
        "cage challenge 2": "CAGE Challenge 2",
        "cyborg": "CybORG",
        "autonomous driving": "autonomous driving",
        "metadrive": "MetaDrive",
        "malware mutation": "Malware Mutation",
        "malconv": "MalConv",
    }
    return aliases.get(lower, cleaned)


def _unit_text_blob(unit: Any) -> str:
    return " ".join(
        _dedupe_nonempty(
            [
                str(getattr(unit, "statement", "") or ""),
                *_unit_positive_obligations(unit, limit=8),
                *[str(item) for item in list(getattr(unit, "verification_targets", []) or [])],
            ]
        )
    )


def _unit_focused_text_blob(unit: Any) -> str:
    return " ".join(
        _dedupe_nonempty(
            [
                str(getattr(unit, "statement", "") or ""),
                *_unit_positive_obligations(unit, limit=8),
                *[str(item) for item in list(getattr(unit, "runtime_interfaces", []) or [])],
                *[str(item) for item in list(getattr(unit, "expected_artifacts", []) or [])],
            ]
        )
    )


def _unit_inventory_text_blob(unit: Any) -> str:
    return "\n".join(
        _dedupe_nonempty(
            [
                str(getattr(unit, "statement", "") or ""),
                *_unit_positive_obligations(unit, limit=8),
                *[str(item) for item in list(getattr(unit, "code_obligations", []) or [])],
                *[str(item) for item in list(getattr(unit, "implementation_notes", []) or [])],
                *[str(item) for item in list(getattr(unit, "expected_artifacts", []) or [])],
            ]
        )
    )


def _inventory_context_tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{1,}", str(text or ""))
    }


def _candidate_sentences(text: str, hints: set[str]) -> list[str]:
    sentences = [
        item.strip()
        for item in re.split(r"(?<=[.;!?])\s+|\n+", str(text or ""))
        if item.strip()
    ]
    selected: list[str] = []
    for sentence in sentences:
        tokens = _inventory_context_tokens(sentence)
        if tokens.intersection(hints):
            selected.append(sentence)
    return selected


def _clean_inventory_phrase(value: str) -> str:
    words = [item.strip(" ,.;:()[]{}") for item in str(value or "").split() if item.strip(" ,.;:()[]{}")]
    while words and words[0].lower() in _INVENTORY_PHRASE_STOPWORDS:
        words.pop(0)
    while words and words[-1].lower() in _INVENTORY_PHRASE_STOPWORDS:
        words.pop()
    return " ".join(words).strip(" ,.;:")


def _is_generic_inventory_phrase(value: str) -> bool:
    cleaned = _clean_inventory_phrase(value)
    if not cleaned:
        return True
    if any(char in cleaned for char in "\\{}^"):
        return True
    if cleaned in _GENERIC_INVENTORY_NAMES:
        return True
    lowered = cleaned.lower()
    parts = [part for part in re.split(r"[\s/_-]+", lowered) if part]
    if not parts:
        return True
    if parts[0] in _GENERIC_PHRASE_HEADS:
        return True
    if len(parts) > 1 and all(len(part) == 1 for part in parts):
        return True
    if len(parts) == 1 and cleaned.islower() and not any(char.isdigit() for char in cleaned):
        return True
    if any(part in _INVENTORY_PHRASE_STOPWORDS for part in parts[1:-1]):
        return True
    if all(part in _INVENTORY_PHRASE_STOPWORDS for part in parts):
        return True
    if re.fullmatch(r"[ivx]+(?:\s*-\s*[ivx]+)?|\d+[a-z]?", lowered):
        return True
    return False


def _looks_like_specific_variant_name(value: str) -> bool:
    cleaned = _clean_inventory_phrase(value)
    if not cleaned or _is_generic_inventory_phrase(cleaned):
        return False
    parts = [part for part in re.split(r"[\s/_-]+", cleaned) if part]
    if not parts:
        return False
    if len(parts) == 1:
        return (
            parts[0].isupper()
            or bool(re.search(r"[a-z][A-Z]", cleaned))
            or cleaned in {"Random", "Ours"}
        )
    return (
        any(part.isupper() for part in parts)
        or any(re.search(r"[a-z][A-Z]", part) for part in parts)
        or "-" in cleaned
        or any(char.isdigit() for char in cleaned)
        or parts[-1].lower() in {"learning", "actorcritic", "finetuning", "fine-tuning"}
    )


def _extract_named_experiment_inventory(text: str) -> list[str]:
    matches = [match.group(0).strip() for match in _EXPERIMENT_NAME_RE.finditer(str(text or ""))]
    return _bounded_inventory_items(matches, limit=12)


def _extract_table_figure_inventory(text: str) -> list[str]:
    matches = [match.group(0).strip() for match in _TABLE_FIGURE_RE.finditer(str(text or ""))]
    return _bounded_inventory_items(matches, limit=10)


_EXPLICIT_INVENTORY_KEYS = {
    "obligation_matrix",
    "experiment_inventory",
    "environment_inventory",
    "dataset_inventory",
    "method_inventory",
    "policy_inventory",
    "model_inventory",
    "baseline_inventory",
    "metric_inventory",
    "measurement_inventory",
    "parameter_inventory",
    "result_trend_inventory",
    "artifact_inventory",
    "result_artifact_inventory",
    "implementation_surface_inventory",
}


def _extract_explicit_inventory_lines(text: str) -> dict[str, list[str]]:
    inventories: dict[str, list[str]] = {}
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip(" -\t")
        if not line:
            continue
        match = re.match(r"(?P<key>[a-z_]{3,64})\s*:\s*(?P<values>.+)$", line, flags=re.I)
        if not match:
            continue
        key = match.group("key").strip().lower()
        if key not in _EXPLICIT_INVENTORY_KEYS:
            continue
        raw_values = match.group("values").strip()
        if key == "obligation_matrix":
            values = [item.strip() for item in re.split(r"\s*;\s*", raw_values) if item.strip()]
        else:
            values = [
                item.strip()
                for item in re.split(r"\s*;\s*|\s*\|\s*", raw_values)
                if item.strip()
            ]
        if values:
            inventories.setdefault(key, [])
            inventories[key].extend(values)
    return {key: _bounded_inventory_items(values, limit=24) for key, values in inventories.items() if values}


def _extract_named_variant_inventory(text: str) -> list[str]:
    tokens = _inventory_context_tokens(text)
    if not tokens.intersection(_BASELINE_HINTS):
        return []
    environment_items = _extract_environment_inventory(text)
    environment_terms = {
        term
        for item in environment_items
        for term in [item, *re.split(r"[\s/]+", item)]
        if term
    }
    candidates: list[str] = []
    method_sentences = _candidate_sentences(text, _BASELINE_HINTS.union({"explanation", "explanations", "refining", "refinement", "algorithm", "algorithms"}))
    scoped_text = "\n".join(method_sentences) or str(text or "")
    if any(re.search(r"\bour (?:proposed )?method\b", sentence, re.I) for sentence in method_sentences):
        candidates.append("Ours")
    candidates.extend(match.group(1).strip() for match in _QUOTED_VARIANT_RE.finditer(scoped_text))
    for parenthetical in _PAREN_LIST_RE.findall(scoped_text):
        if "et al" in parenthetical.lower() or "," not in parenthetical:
            continue
        for item in re.split(r",|\band\b", parenthetical):
            cleaned_item = _clean_inventory_phrase(item)
            if cleaned_item.lower() in {"our method", "our proposed method", "ours method"}:
                candidates.append("Ours")
            elif cleaned_item:
                candidates.append(cleaned_item)
    candidates.extend(match.group(0).strip() for match in _ACRONYM_NAME_RE.finditer(scoped_text))
    candidates.extend(
        match.group(0).strip()
        for match in _TITLE_PHRASE_RE.finditer(scoped_text)
        if _looks_like_specific_variant_name(match.group(0).strip())
    )
    candidates.extend(
        match.group(0).strip()
        for match in _CAMEL_OR_VERSIONED_NAME_RE.finditer(scoped_text)
        if _looks_like_specific_variant_name(match.group(0).strip())
    )
    return _bounded_inventory_items(
        [
            _clean_inventory_phrase(item)
            for item in candidates
            if not _is_generic_inventory_phrase(item)
            and _looks_like_specific_variant_name(_clean_inventory_phrase(item))
            and _clean_inventory_phrase(item) not in environment_terms
            and "-v" not in _clean_inventory_phrase(item).lower()
            and not _clean_inventory_phrase(item).lower().startswith(("section", "appendix", "table", "figure"))
        ],
        limit=18,
    )


def _extract_measurement_inventory(text: str) -> list[str]:
    tokens = _inventory_context_tokens(text)
    if not tokens.intersection(_MEASUREMENT_HINTS):
        return []
    phrases: list[str] = []
    lowered = str(text or "").lower()
    phrase_map = {
        "cumulative reward": "cumulative reward",
        "episode reward": "episode reward",
        "reward curve": "reward curve",
        "learning curve": "learning curve",
        "fidelity": "fidelity score",
        "fidelity score": "fidelity score",
        "success rate": "success rate",
        "accuracy": "accuracy",
        "macro f1": "macro F1",
        "f1": "F1",
        "auc": "AUC",
        "final reward": "final reward",
        "reward change": "reward change",
        "reward improvement": "reward improvement",
        "evasion probability": "evasion probability",
        "runtime": "runtime",
        "training time": "training time",
        "sample efficiency": "sample efficiency",
        "return": "return",
    }
    for needle, label in phrase_map.items():
        if needle in lowered:
            phrases.append(label)
    for table_ref in _extract_table_figure_inventory(text):
        phrases.append(f"{table_ref} reproduction artifact")
    return _bounded_inventory_items(phrases, limit=14)


def _extract_environment_inventory(text: str) -> list[str]:
    tokens = _inventory_context_tokens(text)
    versioned_phrases = [match.group(0).strip() for match in _VERSIONED_ENV_PHRASE_RE.finditer(str(text or ""))]
    common_env_phrases = [match.group(0).strip() for match in _COMMON_ENV_PHRASE_RE.finditer(str(text or ""))]
    if not tokens.intersection(_ENVIRONMENT_HINTS) and not versioned_phrases and not common_env_phrases:
        return []
    candidates = list(versioned_phrases) + list(common_env_phrases)
    env_sentences = _candidate_sentences(text, _ENVIRONMENT_HINTS)
    for sentence in env_sentences:
        candidates.extend(match.group(1).strip() for match in _TASK_NOUN_PHRASE_RE.finditer(sentence))
        for parenthetical in _PAREN_LIST_RE.findall(sentence):
            if "et al" in parenthetical.lower() or "," not in parenthetical:
                continue
            for item in re.split(r",|\band\b", parenthetical):
                cleaned_item = re.sub(r"\s+of\s+the\s+.+$", "", item.strip(), flags=re.I)
                if cleaned_item:
                    candidates.append(cleaned_item)
        candidates.extend(match.group(0).strip() for match in _TITLE_PHRASE_RE.finditer(sentence))
        candidates.extend(
            match.group(0).strip()
            for match in _CAMEL_OR_VERSIONED_NAME_RE.finditer(sentence)
            if "-v" in match.group(0).lower()
        )
    return _bounded_inventory_items(
        [
            _normalize_environment_item(item)
            for item in candidates
            if _normalize_environment_item(item)
            and not _normalize_environment_item(item).lower().startswith(("section", "appendix", "table", "figure"))
            and (
                "-v" in _normalize_environment_item(item).lower()
                or _COMMON_ENV_PHRASE_RE.fullmatch(_normalize_environment_item(item))
                or not _is_generic_inventory_phrase(_normalize_environment_item(item))
            )
        ],
        limit=18,
    )


def _derive_unit_inventory(owned_units: list[Any]) -> dict[str, list[str]]:
    inventories: dict[str, list[str]] = {}
    for unit in owned_units:
        text = _unit_text_blob(unit)
        focused_text = _unit_focused_text_blob(unit)
        for key, values in _extract_explicit_inventory_lines(_unit_inventory_text_blob(unit)).items():
            inventories.setdefault(key, [])
            inventories[key].extend(values)
        experiments = _extract_named_experiment_inventory(text)
        environments = _extract_environment_inventory(text)
        baselines = _extract_named_variant_inventory(text)
        measurements = _extract_measurement_inventory(text)
        result_refs = _extract_table_figure_inventory(text)
        if experiments:
            inventories.setdefault("experiment_inventory", [])
            inventories["experiment_inventory"].extend(experiments)
        if environments:
            inventories.setdefault("environment_inventory", [])
            inventories["environment_inventory"].extend(environments)
        if baselines:
            inventories.setdefault("baseline_inventory", [])
            inventories["baseline_inventory"].extend(baselines)
        if measurements:
            inventories.setdefault("measurement_inventory", [])
            inventories["measurement_inventory"].extend(measurements)
        if result_refs:
            inventories.setdefault("result_artifact_inventory", [])
            inventories["result_artifact_inventory"].extend(result_refs)
    return {key: _bounded_inventory_items(values) for key, values in inventories.items() if values}


def _augment_inventory_obligations(inventories: dict[str, list[str]]) -> list[str]:
    obligations: list[str] = []
    obligation_matrix = list(inventories.get("obligation_matrix", []) or [])
    experiments = list(inventories.get("experiment_inventory", []) or [])
    environments = list(inventories.get("environment_inventory", []) or [])
    baselines = (
        list(inventories.get("baseline_inventory", []) or [])
        + list(inventories.get("method_inventory", []) or [])
        + list(inventories.get("policy_inventory", []) or [])
        + list(inventories.get("model_inventory", []) or [])
    )
    measurements = list(inventories.get("measurement_inventory", []) or [])
    parameters = list(inventories.get("parameter_inventory", []) or [])
    trends = list(inventories.get("result_trend_inventory", []) or [])
    artifacts = list(inventories.get("artifact_inventory", []) or []) + list(
        inventories.get("result_artifact_inventory", []) or []
    )
    if obligation_matrix:
        obligations.append(
            "Preserve the paper-derived evidence obligation matrix in code/config/artifacts: "
            + "; ".join(obligation_matrix[:8])
        )
    if experiments:
        obligations.append(
            "Expose named experiment protocols in code/config rather than collapsing them into a generic runner: "
            + "; ".join(experiments[:8])
        )
    if environments:
        obligations.append(
            "Preserve explicit environment/task coverage and initialization surfaces: "
            + "; ".join(environments[:10])
        )
    if baselines:
        obligations.append(
            "Preserve explicit baseline or method-variant selection surfaces: "
            + "; ".join(baselines[:10])
        )
    if measurements:
        obligations.append(
            "Implement measurement collection and result aggregation for: "
            + "; ".join(measurements[:10])
        )
    if parameters:
        obligations.append(
            "Expose required parameter sweeps through bounded config/registry entries: "
            + "; ".join(parameters[:10])
        )
    if trends:
        obligations.append(
            "Preserve required result-trend assertions in reporting/review artifacts: "
            + "; ".join(trends[:10])
        )
    if artifacts:
        obligations.append(
            "Write or declare concrete reproduction artifacts for result verification: "
            + "; ".join(artifacts[:10])
        )
    return _dedupe_nonempty(obligations)


def _merge_inventory_maps(*maps: dict[str, list[str]]) -> dict[str, list[str]]:
    merged: dict[str, list[str]] = {}
    for inventory_map in maps:
        for key, values in dict(inventory_map or {}).items():
            normalized_key = str(key or "").strip()
            if normalized_key not in _IMPLEMENTATION_INVENTORY_KEYS:
                continue
            merged.setdefault(normalized_key, [])
            merged[normalized_key].extend(_normalize_contract_list(values))
    return {key: _bounded_inventory_items(values) for key, values in merged.items() if values}


def _slugify_contract_id(value: str) -> str:
    characters: list[str] = []
    previous_separator = False
    for char in str(value or "").lower():
        if char.isascii() and (char.isalnum() or char == "_"):
            characters.append(char)
            previous_separator = False
            continue
        if previous_separator:
            continue
        characters.append("_")
        previous_separator = True
    return "".join(characters).strip("_") or "item"


def _looks_like_encoded_artifact_name(path: str) -> bool:
    normalized = _normalize_repo_path(path)
    name = normalized.rsplit("/", 1)[-1].lower()
    if not name:
        return False
    parts = [part for part in name.replace("-", "_").split("_") if part]
    if len(parts) < 2:
        return name in {"results", "outputs", "metrics", "predictions", "figures", "plots", "checkpoints"}
    if parts[0] in {"result", "results", "output", "outputs", "prediction", "predictions", "figure", "figures", "plot", "plots", "metric", "metrics", "checkpoint", "checkpoints"}:
        return True
    if parts[-1] in _ARTIFACT_ENCODED_SUFFIXES and any(
        part in {"result", "results", "output", "outputs", "prediction", "predictions", "figure", "figures", "plot", "plots", "metric", "metrics", "table", "tables", "summary", "checkpoint", "checkpoints"}
        for part in parts[:-1]
    ):
        return True
    return False


def _looks_like_artifact_path(path: str) -> bool:
    normalized = _normalize_repo_path(path)
    if not normalized:
        return False
    lowered = normalized.lower()
    if any(token in lowered for token in ("(", ")", "->", "=", ",")):
        return False
    if any(lowered == prefix.rstrip("/") or lowered.startswith(prefix) for prefix in _ARTIFACT_ROOT_PREFIXES):
        return True
    if _looks_like_encoded_artifact_name(normalized):
        return True
    if any(char in lowered for char in "*[]{}") and "/" in lowered:
        return True
    name = lowered.rsplit("/", 1)[-1]
    if name.endswith(_ARTIFACT_FILE_SUFFIXES):
        return True
    if name.endswith(".json") and any(
        token in name
        for token in (
            "ablation",
            "artifact",
            "curve",
            "curves",
            "evidence",
            "experiment",
            "figure",
            "metric",
            "metrics",
            "registry",
            "report",
            "result",
            "results",
            "summary",
            "table",
            "trace",
        )
    ):
        return True
    if name.endswith(".json"):
        stem = name.rsplit(".", 1)[0]
        if stem in {"metrics", "results", "summary", "trend_summary", "experiment_plan", "ablation_matrix"}:
            return True
        if any(token in stem for token in ("prediction", "checkpoint")):
            return True
    if name in {"metrics", "results", "summary", "figures", "plots", "checkpoints", "predictions"}:
        return True
    if _looks_like_implementation_path(normalized):
        return False
    return bool(
        "/" in normalized
        and any(token in normalized.lower() for token in ("result", "artifact", "metric", "report", "prediction", "output"))
    )


def _looks_like_contract_output_path(path: str) -> bool:
    normalized = _normalize_repo_path(path)
    if not normalized:
        return False
    lowered = normalized.lower()
    if any(token in lowered for token in ("(", ")", "->", "=", ",")):
        return False
    return (
        _looks_like_artifact_path(normalized)
        or lowered.startswith(_ARTIFACT_ROOT_PREFIXES)
        or _looks_like_encoded_artifact_name(normalized)
        or lowered.endswith(_ARTIFACT_FILE_SUFFIXES)
        or any(token in lowered for token in ("result/", "results_", "prediction", "output/"))
        or (
            lowered.endswith(".json")
            and any(
                token in lowered.rsplit("/", 1)[-1]
                for token in (
                    "ablation",
                    "artifact",
                    "curve",
                    "curves",
                    "evidence",
                    "experiment",
                    "figure",
                    "metric",
                    "metrics",
                    "registry",
                    "report",
                    "result",
                    "results",
                    "summary",
                    "table",
                    "trace",
                )
            )
        )
    )


def _extract_repo_paths(values: list[str]) -> list[str]:
    return [
        normalized
        for item in values
        for normalized in [_normalize_repo_path(item)]
        if _looks_like_implementation_path(normalized) and not _looks_like_contract_output_path(normalized)
    ]


_EMBEDDED_REPO_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_./-])"
    r"(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+"
    r"\.(?:py|md|txt|toml|ya?ml|json|ini|cfg|sh)"
    r"(?![A-Za-z0-9_./-])",
    re.I,
)


def _embedded_implementation_paths_from_text(value: str) -> list[str]:
    """Extract repo paths mentioned inside contract prose."""
    paths: list[str] = []
    for match in _EMBEDDED_REPO_PATH_RE.finditer(str(value or "")):
        candidate = _normalize_repo_path(match.group(0).strip("`'\"()[]{}.,;:"))
        if not candidate:
            continue
        hinted = _source_path_for_surface_hint(candidate) or candidate
        if _is_valid_source_repo_path(hinted):
            paths.append(hinted)
    return _canonicalize_semantic_repo_paths(paths)


def _is_entrypoint_like_path(path: str) -> bool:
    lowered = _normalize_repo_path(path).lower()
    if not lowered:
        return False
    basename = lowered.rsplit("/", 1)[-1]
    if basename in {
        "main.py",
        "cli.py",
        "entrypoint.py",
        "run.py",
        "runner.py",
        "train.py",
        "eval.py",
        "evaluate.py",
    }:
        return True
    if lowered.startswith(("scripts/", "bin/")) and basename.endswith(".py"):
        stem = basename[:-3]
        return stem in {"run", "runner", "train", "eval", "evaluate"} or stem.startswith(
            ("run_", "train_", "eval_", "evaluate_")
        )
    return False


def _surface_path_from_text(value: str, *, package_id: str = "") -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    normalized = _normalize_repo_path(text)
    if (
        _looks_like_repo_relative_path(normalized)
        and _looks_like_implementation_path(normalized)
        and not _looks_like_contract_output_path(normalized)
    ):
        return normalized
    hinted = _source_path_for_surface_hint(text)
    if hinted:
        return hinted
    terms = _tokenize_text(text, package_id)
    for term in sorted(terms):
        if term in _IMPLEMENTATION_SURFACE_PATHS:
            return _semantic_fallback_source_path(text, package_id=package_id)
    lowered_text = text.lower()
    if terms.intersection({"entrypoint", "entry", "cli", "run", "command"}) or re.search(
        r"\bmain(?:\.py|\s+entry|\s+script|\s+cli)\b",
        lowered_text,
    ):
        return "main.py"
    if terms.intersection({"config", "configuration", "hyperparameter", "parameter", "settings"}):
        return _semantic_fallback_source_path(text, package_id=package_id)
    if terms.intersection({"environment", "environments", "env", "simulator", "simulation", "benchmark"}):
        return _semantic_fallback_source_path(text, package_id=package_id)
    if terms.intersection({"dataset", "data", "loader", "preprocess", "preprocessing", "sampling"}):
        return _semantic_fallback_source_path(text, package_id=package_id)
    if terms.intersection({"evaluation", "evaluate", "metric", "metrics", "score", "report", "results"}):
        return _semantic_fallback_source_path(text, package_id=package_id)
    if terms.intersection({"policy", "agent", "network", "model", "encoder", "decoder"}):
        return _semantic_fallback_source_path(text, package_id=package_id)
    if terms.intersection({"train", "training", "trainer", "finetune", "fine", "optimization", "optimize"}):
        return _semantic_fallback_source_path(text, package_id=package_id)
    if terms.intersection({"method", "algorithm", "baseline", "ablation", "explanation", "mask"}) or _has_refinement_semantic_signal(text, package_id):
        return _semantic_fallback_source_path(text, package_id=package_id)
    if terms.intersection({"artifact", "writer", "serialize", "checkpoint", "logging"}):
        return _semantic_fallback_source_path(text, package_id=package_id)
    return ""


def _project_work_package_semantic_surfaces(work_package: dict[str, Any]) -> list[str]:
    package_id = str(work_package.get("work_package_id", "") or "").strip()
    inventories = dict(work_package.get("inventories", {}) or {})
    semantic_items: list[str] = []
    semantic_items.extend(str(item) for item in list(work_package.get("interface_contract", []) or []))
    for key, values in inventories.items():
        normalized_key = str(key or "").strip().lower()
        if "artifact" in normalized_key or "result" in normalized_key or "measurement" in normalized_key:
            continue
        semantic_items.extend(str(item) for item in list(values or []))
    projected = [
        path
        for item in semantic_items
        for path in [_surface_path_from_text(item, package_id=package_id)]
        if path and _looks_like_repo_relative_path(path) and not _looks_like_contract_output_path(path)
    ]
    return _canonicalize_semantic_repo_paths(projected)


def _merge_work_package_inventories(work_package: dict[str, Any]) -> dict[str, list[str]]:
    inventories = _normalize_inventory_map(work_package.get("inventories"))
    for produced in _normalize_contract_list(work_package.get("produces", [])):
        normalized = _normalize_repo_path(produced)
        if not normalized:
            continue
        inventory_name = (
            "artifact_inventory"
            if _looks_like_contract_output_path(normalized)
            else "source_files" if _looks_like_implementation_path(normalized)
            else "artifact_inventory"
        )
        inventories.setdefault(inventory_name, [])
        if normalized not in inventories[inventory_name]:
            inventories[inventory_name].append(normalized)
    return inventories


def _append_inventory_item(
    inventories: dict[str, list[str]],
    inventory_owners: dict[str, dict[str, list[str]]],
    inventory_name: str,
    item: str,
    owner_work_package: str,
) -> None:
    normalized_name = str(inventory_name or "").strip()
    normalized_item = str(item or "").strip()
    normalized_owner = str(owner_work_package or "").strip()
    if not normalized_name or not normalized_item or not normalized_owner:
        return
    inventories.setdefault(normalized_name, [])
    if normalized_item not in inventories[normalized_name]:
        inventories[normalized_name].append(normalized_item)
    inventory_owners.setdefault(normalized_name, {})
    inventory_owners[normalized_name].setdefault(normalized_owner, [])
    if normalized_item not in inventory_owners[normalized_name][normalized_owner]:
        inventory_owners[normalized_name][normalized_owner].append(normalized_item)


def _collect_inventory_contract(
    work_packages: list[dict[str, Any]],
) -> tuple[dict[str, list[str]], dict[str, dict[str, list[str]]]]:
    inventories: dict[str, list[str]] = {}
    inventory_owners: dict[str, dict[str, list[str]]] = {}
    for work_package in work_packages:
        work_package_id = str(work_package.get("work_package_id", "") or "").strip()
        if not work_package_id:
            continue
        for inventory_name, items in _merge_work_package_inventories(work_package).items():
            for item in items:
                _append_inventory_item(inventories, inventory_owners, inventory_name, item, work_package_id)
    return inventories, inventory_owners


def _default_requirement_ids(input_payload: dict[str, Any]) -> list[str]:
    requirements = list(dict(input_payload.get("boundary_requirements", {}) or {}).get("boundary_requirements", []) or [])
    requirement_ids = _dedupe_nonempty(
        [str(item.get("requirement_id", "") or "") for item in requirements if isinstance(item, dict)]
    )
    return requirement_ids if len(requirement_ids) == 1 else []


def _normalize_work_package_contracts(
    input_payload: dict[str, Any],
    draft: GlobalContractOutput,
) -> list[dict[str, Any]]:
    work_packages = list(dict(input_payload.get("work_package_planning", {}) or {}).get("work_packages", []) or [])
    evidence_map = {
        str(item.get("work_package_id", "") or ""): item
        for item in list(input_payload.get("evidence_bundles", []) or [])
        if isinstance(item, dict)
    }
    draft_map = {
        item.work_package_id: item
        for item in draft.work_package_contracts
    }
    default_requirement_ids = _default_requirement_ids(input_payload)
    normalized_contracts: list[dict[str, Any]] = []
    for work_package in work_packages:
        if not isinstance(work_package, dict):
            continue
        work_package_id = str(work_package.get("work_package_id", "") or "").strip()
        if not work_package_id:
            continue
        draft_contract = draft_map.get(work_package_id)
        evidence_bundle = evidence_map.get(work_package_id, {})
        work_package_text_scope = " ".join(
            _dedupe_nonempty(
                [
                    work_package_id,
                    str(work_package.get("goal", "") or ""),
                    str(work_package.get("hypothesis", "") or ""),
                    str(work_package.get("decision_value", "") or ""),
                    *[str(item) for item in list(work_package.get("tags", []) or [])],
                    *[str(item) for item in list(work_package.get("interface_contract", []) or [])],
                    *[str(item) for item in list(work_package.get("method_obligations", []) or [])],
                    *[
                        str(item)
                        for values in dict(work_package.get("inventories", {}) or {}).values()
                        for item in list(values or [])
                    ],
                ]
            )
        )
        formula_obligations = _formula_contract_anchor_obligations(
            input_payload,
            text_scope=work_package_text_scope,
            limit=8,
        )
        scope_boundary = (
            dict(work_package.get("scope_boundary", {}) or {})
            if isinstance(work_package.get("scope_boundary", {}), dict)
            else {}
        )
        if formula_obligations:
            scope_boundary["preserve"] = _dedupe_nonempty(
                [str(item) for item in list(scope_boundary.get("preserve", []) or [])]
                + formula_obligations[:4]
            )
            scope_boundary["implementation_focus"] = _dedupe_nonempty(
                [str(item) for item in list(scope_boundary.get("implementation_focus", []) or [])]
                + ["formula/algorithm anchors must be represented in executable code/config and reached by the canonical route"]
            )
        normalized_contracts.append(
            {
                "work_package_id": work_package_id,
                "goal": str(work_package.get("goal", "") or "").strip() or (draft_contract.goal if draft_contract else ""),
                "depends_on": _normalize_contract_list(work_package.get("depends_on", [])),
                "requirement_ids": (
                    _normalize_contract_list(draft_contract.requirement_ids)
                    if draft_contract is not None and draft_contract.requirement_ids
                    else list(default_requirement_ids)
                ),
                "reference_ids": _normalize_contract_list(
                    work_package.get("reference_ids", [])
                    or (list(draft_contract.reference_ids) if draft_contract is not None else [])
                ),
                "interface_contract": _dedupe_nonempty(
                    _normalize_contract_list(work_package.get("interface_contract", []))
                ),
                "method_obligations": _dedupe_nonempty(
                    _normalize_contract_list(work_package.get("method_obligations", []))
                    + formula_obligations
                ),
                "produces": _dedupe_nonempty(
                    [
                        _normalize_repo_path(path)
                        for path in _normalize_contract_list(work_package.get("produces", []))
                    ]
                    + (
                        [_normalize_repo_path(path) for path in draft_contract.produces]
                        if draft_contract is not None
                        else []
                    )
                ),
                "inventories": _merge_work_package_inventories(work_package),
                "scope_boundary": scope_boundary,
                "grounding_status": (
                    str(evidence_bundle.get("grounding_status", "") or "").strip()
                    or (draft_contract.grounding_status if draft_contract is not None else "")
                ),
                "evidence_summary": _dedupe_nonempty(
                    _normalize_contract_list(evidence_bundle.get("context_summary", []))
                    + (list(draft_contract.evidence_summary) if draft_contract is not None else [])
                )[:6],
            }
        )
    return normalized_contracts


def _owner_work_packages_for_inventory_item(
    inventory_owners: dict[str, dict[str, list[str]]],
    inventory_name: str,
    item: str,
) -> list[str]:
    normalized_item = str(item or "").strip()
    owners: list[str] = []
    for work_package_id, values in dict(inventory_owners.get(inventory_name, {}) or {}).items():
        if normalized_item in values:
            owners.append(work_package_id)
    return owners


def _known_work_package_ids_from_contracts(work_package_contracts: list[dict[str, Any]]) -> set[str]:
    return {
        str(item.get("work_package_id", "") or "").strip()
        for item in work_package_contracts
        if str(item.get("work_package_id", "") or "").strip()
    }


def _normalized_target_artifact_paths(paths: list[str]) -> list[str]:
    return _dedupe_nonempty(
        [
            normalized
            for path in list(paths or [])
            for normalized in [_normalize_repo_path(path)]
            if _looks_like_repo_relative_path(normalized) and _looks_like_contract_output_path(normalized)
        ]
    )


def _normalized_target_owner_work_packages(
    owner_work_packages: list[str],
    *,
    work_package_contracts: list[dict[str, Any]],
    inventory_owners: dict[str, dict[str, list[str]]],
    artifact_paths: list[str],
) -> list[str]:
    known_work_package_ids = _known_work_package_ids_from_contracts(work_package_contracts)
    normalized_owners = _dedupe_nonempty(
        [
            str(item or "").strip()
            for item in list(owner_work_packages or [])
            if str(item or "").strip() in known_work_package_ids
        ]
    )
    if normalized_owners:
        return normalized_owners

    inferred_owners: list[str] = []
    for artifact_path in artifact_paths:
        for inventory_name in ("artifact_inventory", "measurement_inventory", "benchmark_artifacts"):
            inferred_owners.extend(_owner_work_packages_for_inventory_item(inventory_owners, inventory_name, artifact_path))

    if inferred_owners:
        return _dedupe_nonempty(
            [owner for owner in inferred_owners if owner in known_work_package_ids]
        )

    for work_package in work_package_contracts:
        work_package_id = str(work_package.get("work_package_id", "") or "").strip()
        if not work_package_id:
            continue
        produces = _normalized_target_artifact_paths(_normalize_contract_list(work_package.get("produces", [])))
        inventories = {
            normalized
            for values in dict(work_package.get("inventories", {}) or {}).values()
            for normalized in _normalized_target_artifact_paths(values if isinstance(values, list) else [values])
        }
        if set(artifact_paths).intersection(produces) or set(artifact_paths).intersection(inventories):
            normalized_owners.append(work_package_id)

    return _dedupe_nonempty(normalized_owners)


def _result_target_kind(inventory_name: str, item: str) -> str:
    lowered_inventory = str(inventory_name or "").lower()
    if "baseline" in lowered_inventory:
        return "baseline"
    if "ablation" in lowered_inventory:
        return "ablation"
    if "variant" in lowered_inventory:
        return "variant"
    if "measurement" in lowered_inventory:
        return "measurement"
    if "prerequisite" in lowered_inventory:
        return "prerequisite"
    if _looks_like_artifact_path(item):
        return "artifact"
    return "experiment"


def _benchmark_expectations(input_payload: dict[str, Any], draft: GlobalContractOutput) -> dict[str, Any]:
    experiment_design = dict(input_payload.get("experiment_design", {}) or {})
    benchmark_expectations = dict(draft.benchmark_expectations)
    for key in ("benchmark_name", "case_id", "task_id"):
        value = str(experiment_design.get(key, "") or "").strip()
        if value:
            benchmark_expectations[key] = value
    expected_artifacts = _dedupe_nonempty(
        _normalize_contract_list(benchmark_expectations.get("expected_artifacts", []))
        + _normalize_contract_list(experiment_design.get("expected_artifacts", []))
        + [
            _normalize_repo_path(path)
            for path in _normalize_contract_list(dict(input_payload.get("normalized_input", {}) or {}).get("expected_outputs", []))
            if _looks_like_contract_output_path(_normalize_repo_path(path))
        ]
    )
    if expected_artifacts:
        benchmark_expectations["expected_artifacts"] = expected_artifacts
    notes = _dedupe_nonempty(
        _normalize_contract_list(benchmark_expectations.get("notes", []))
        + _normalize_contract_list(experiment_design.get("notes", []))
    )
    if notes:
        benchmark_expectations["notes"] = notes
    return benchmark_expectations


def _draft_result_target_map(draft: GlobalContractOutput) -> dict[str, GlobalContractResultTarget]:
    return {item.target_id: item for item in draft.result_targets if item.target_id}


def _synthesize_result_targets(
    input_payload: dict[str, Any],
    draft: GlobalContractOutput,
    work_package_contracts: list[dict[str, Any]],
    inventories: dict[str, list[str]],
    inventory_owners: dict[str, dict[str, list[str]]],
) -> list[dict[str, Any]]:
    draft_by_id = _draft_result_target_map(draft)
    work_package_inventory_map = {
        str(item.get("work_package_id", "") or ""): dict(item.get("inventories", {}) or {})
        for item in work_package_contracts
    }
    owner_surface_map: dict[str, list[str]] = {}
    for work_package in work_package_contracts:
        work_package_id = str(work_package.get("work_package_id", "") or "").strip()
        if not work_package_id:
            continue
        owner_surface_map[work_package_id] = _extract_repo_paths(
            _normalize_contract_list(work_package.get("produces", []))
            + [
                candidate
                for values in dict(work_package_inventory_map.get(work_package_id, {}) or {}).values()
                for candidate in values
            ]
        )

    entry_surfaces = _dedupe_nonempty(
        [
            path
            for work_package in work_package_contracts
            for path in _extract_repo_paths(work_package.get("produces", []))
            if _is_entrypoint_like_path(path)
        ]
    )

    result_targets: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_artifacts: set[str] = set()

    for draft_target in draft.result_targets:
        target_id = str(draft_target.target_id or "").strip()
        if not target_id:
            continue
        normalized_paths = _normalized_target_artifact_paths(list(draft_target.artifact_paths or []))
        normalized_owner_work_packages = _normalized_target_owner_work_packages(
            list(draft_target.owner_work_packages or []),
            work_package_contracts=work_package_contracts,
            inventory_owners=inventory_owners,
            artifact_paths=normalized_paths,
        )
        result_targets.append(
            {
                "target_id": target_id,
                "kind": str(draft_target.kind or "").strip() or "artifact",
                "name": str(draft_target.name or "").strip() or target_id,
                "owner_work_packages": normalized_owner_work_packages,
                "required_inputs": _normalize_contract_list(draft_target.required_inputs),
                "artifact_paths": normalized_paths,
                "coverage_notes": _normalize_contract_list(draft_target.coverage_notes),
            }
        )
        seen_ids.add(target_id)
        seen_artifacts.update(normalized_paths)

    candidate_items: list[tuple[str, str]] = []
    for inventory_name, items in inventories.items():
        for item in items:
            if inventory_name == "source_files" and item not in seen_artifacts:
                continue
            candidate_items.append((inventory_name, item))
    for inventory_name, item in candidate_items:
        normalized_item = _normalize_repo_path(item)
        if not normalized_item:
            continue
        kind = "benchmark_artifact" if inventory_name == "benchmark_artifacts" else _result_target_kind(inventory_name, normalized_item)
        target_id = f"{kind}:{_slugify_contract_id(normalized_item)}"
        if target_id in seen_ids:
            draft_target = draft_by_id.get(target_id)
            if draft_target is not None and normalized_item not in seen_artifacts:
                seen_artifacts.add(normalized_item)
            continue
        owner_work_packages = (
            _owner_work_packages_for_inventory_item(inventory_owners, inventory_name, normalized_item)
            if inventory_name != "benchmark_artifacts"
            else []
        )
        if inventory_name == "benchmark_artifacts" and not owner_work_packages:
            owner_work_packages = _dedupe_nonempty(
                [
                    work_package_id
                    for work_package_id, surfaces in owner_surface_map.items()
                    if any(_looks_like_artifact_path(path) for path in surfaces)
                ]
            )
        coverage_notes = [f"Derived from {inventory_name}."]
        if kind == "benchmark_artifact":
            coverage_notes.append("Expected benchmark-facing artifact must remain reproducible in the canonical repo.")
        if draft_by_id.get(target_id) is not None:
            coverage_notes = _dedupe_nonempty(
                coverage_notes + list(draft_by_id[target_id].coverage_notes)
            )
        result_targets.append(
            {
                "target_id": target_id,
                "kind": kind,
                "name": normalized_item,
                "owner_work_packages": owner_work_packages,
                "required_inputs": list(entry_surfaces) if kind in {"artifact", "measurement", "benchmark_artifact"} else [],
                "artifact_paths": [normalized_item] if _looks_like_artifact_path(normalized_item) else [],
                "coverage_notes": coverage_notes,
            }
        )
        seen_ids.add(target_id)
        if _looks_like_artifact_path(normalized_item):
            seen_artifacts.add(normalized_item)

    return result_targets


def _contract_notes(
    input_payload: dict[str, Any],
    draft: GlobalContractOutput,
    work_package_contracts: list[dict[str, Any]],
    result_targets: list[dict[str, Any]],
) -> list[str]:
    del draft
    evidence_bundles = list(input_payload.get("evidence_bundles", []) or [])
    ungrounded = [
        str(item.get("work_package_id", "") or "")
        for item in evidence_bundles
        if isinstance(item, dict) and str(item.get("grounding_status", "") or "").strip().lower() == "ungrounded"
    ]
    notes = _dedupe_nonempty(
        [
            f"Contract covers {len(work_package_contracts)} work packages.",
            f"Contract exposes {len(result_targets)} result targets.",
            "reproagent delivers one canonical validated repository from the projected work-package implementation contracts.",
        ]
    )
    if ungrounded:
        notes.append("Ungrounded work packages still require closed contracts: " + ", ".join(ungrounded[:6]))
    return notes


def _validation_gates(
    input_payload: dict[str, Any],
    draft: GlobalContractOutput,
    result_targets: list[dict[str, Any]],
) -> list[str]:
    gates = list(draft.validation_gates)
    if not gates:
        gates.extend(
            [
                "canonical_entry_surface_present",
                "repo_plan_contract_closed",
                "docker_validation_ready",
            ]
        )
    if result_targets and "artifact_contract_declared" not in gates:
        gates.append("artifact_contract_declared")
    if list(input_payload.get("evidence_bundles", []) or []) and "work_package_trace_closed" not in gates:
        gates.append("work_package_trace_closed")
    return _dedupe_nonempty(gates)


def _synthesize_global_contract_output(
    input_payload: dict[str, Any],
    draft: GlobalContractOutput,
) -> GlobalContractOutput:
    work_package_contracts = _normalize_work_package_contracts(input_payload, draft)
    work_packages = [
        {
            "work_package_id": item["work_package_id"],
            "produces": item["produces"],
            "inventories": item["inventories"],
        }
        for item in work_package_contracts
    ]
    inventories, inventory_owners = _collect_inventory_contract(work_packages)
    result_targets = _synthesize_result_targets(
        input_payload,
        draft,
        work_package_contracts,
        inventories,
        inventory_owners,
    )
    benchmark_expectations = _benchmark_expectations(input_payload, draft)
    return GlobalContractOutput.model_validate(
        {
            "contract_version": str(draft.contract_version or "1.0").strip() or "1.0",
            "canonical_stage_sequence": (
                _normalize_contract_list(draft.canonical_stage_sequence)
                or ["scope", "contract", "design", "generate"]
            ),
            "work_package_contracts": work_package_contracts,
            "inventories": inventories,
            "inventory_owners": inventory_owners,
            "result_targets": result_targets,
            "benchmark_expectations": benchmark_expectations,
            "validation_gates": _validation_gates(input_payload, draft, result_targets),
            "contract_notes": _contract_notes(input_payload, draft, work_package_contracts, result_targets),
        }
    )


def _architecture_has_path(architecture: ArchitectureOutput, path: str) -> bool:
    normalized = _normalize_repo_path(path)
    if not normalized:
        return False
    return normalized in {
        _normalize_repo_path(candidate)
        for candidate in architecture.target_file_tree
    }


def _contract_text_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for raw in str(text or "").lower().replace("/", " ").replace("-", " ").split():
        cleaned = "".join(char for char in raw if char.isalnum() or char == "_").strip("_")
        if len(cleaned) < 4 or cleaned in _CONTRACT_TOKEN_STOPWORDS:
            continue
        tokens.add(cleaned)
    return tokens


def _architecture_text_corpus(architecture: ArchitectureOutput) -> tuple[str, set[str]]:
    text_parts = [str(architecture.rationale or "").lower()]
    tokens: set[str] = set()
    for blueprint in architecture.file_blueprints:
        text_parts.append(str(blueprint.path or "").lower())
        text_parts.append(str(blueprint.purpose or "").lower())
        tokens.update(_contract_text_tokens(blueprint.path))
        tokens.update(_contract_text_tokens(blueprint.purpose))
    return " ".join(text_parts), tokens


def _method_spine_item_materialized(
    architecture: ArchitectureOutput,
    item: str,
    contract_targets: dict[str, Any],
) -> bool:
    normalized_item = str(item or "").strip()
    lowered_item = normalized_item.lower()
    if not normalized_item:
        return True
    repo_paths = {_normalize_repo_path(path) for path in architecture.target_file_tree}
    if _looks_like_implementation_path(normalized_item):
        return _normalize_repo_path(normalized_item) in repo_paths
    required_entry_files = {
        _normalize_repo_path(path)
        for path in list(contract_targets.get("required_entry_files", []) or [])
    }
    required_result_artifacts = _normalize_contract_list(contract_targets.get("required_result_artifacts", []))
    config_paths = [path for path in repo_paths if _infer_file_kind(path) == "config"]
    corpus, corpus_tokens = _architecture_text_corpus(architecture)
    if any(token in lowered_item for token in _METHOD_SPINE_HINTS["entrypoint"]):
        return bool(required_entry_files.intersection(repo_paths)) or any(_is_entrypoint_like_path(path) for path in repo_paths)
    if any(token in lowered_item for token in _METHOD_SPINE_HINTS["artifact"]):
        return bool(required_result_artifacts) and (
            bool(required_entry_files.intersection(repo_paths))
            or any(path.endswith(".py") for path in repo_paths)
        )
    if any(token in lowered_item for token in _METHOD_SPINE_HINTS["config"]):
        return bool(config_paths)
    if any(token in lowered_item for token in _METHOD_SPINE_HINTS["data"]):
        return any(any(token in path.lower() for token in ("data", "dataset", "loader", "preprocess")) for path in repo_paths)
    if any(token in lowered_item for token in _METHOD_SPINE_HINTS["evaluation"]):
        return any(any(token in path.lower() for token in ("eval", "metric", "benchmark", "score")) for path in repo_paths)
    if lowered_item in corpus:
        return True
    return bool(_contract_text_tokens(normalized_item).intersection(corpus_tokens))


def _path_pattern_matches_file_tree(path: str, file_tree: list[str]) -> bool:
    normalized = _normalize_repo_path(path)
    if not normalized:
        return False
    stem = normalized.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    parent = normalized.rsplit("/", 1)[0] if "/" in normalized else ""
    for candidate in file_tree:
        current = _normalize_repo_path(candidate)
        if current == normalized or current.endswith("/" + normalized) or current.endswith(normalized.rsplit("/", 1)[-1]):
            return True
        if stem and current.rsplit("/", 1)[-1].startswith(stem):
            return True
        if parent and current.startswith(parent + "/"):
            return True
    return False


def _architecture_prune_paths_without_semantic_signal(
    architecture: ArchitectureOutput,
    *,
    contract_targets: dict[str, Any],
    input_payload: dict[str, Any] | None = None,
) -> ArchitectureOutput:
    keep_domain_llm_tree = _architecture_keep_llm_file_tree(architecture)
    domain_root = _architecture_domain_package_root(architecture) if keep_domain_llm_tree else ""
    semantic_items: list[str] = []
    for key in (
        "task_model_method_spine",
        "task_model_interface_closure",
        "task_model_runnable_flow",
        "required_interface_surfaces",
    ):
        semantic_items.extend(str(item) for item in list(contract_targets.get(key, []) or []))
    if input_payload:
        for work_package in list(input_payload.get("work_package_planning", {}).get("work_packages", []) or []):
            if not isinstance(work_package, dict):
                continue
            semantic_items.extend(str(item) for item in list(work_package.get("tags", []) or []))
            semantic_items.extend(str(item) for item in list(work_package.get("interface_contract", []) or []))
            semantic_items.extend(str(item) for item in list(work_package.get("implementation_surfaces", []) or []))
            semantic_items.extend(str(item) for item in list(work_package.get("method_obligations", []) or []))
            semantic_items.extend(str(work_package.get(field, "") or "") for field in ("goal", "hypothesis", "decision_value"))
        for node in list(input_payload.get("pipeline_plan", {}).get("plan_nodes", []) or []):
            if not isinstance(node, dict):
                continue
            semantic_items.extend(str(node.get(field, "") or "") for field in ("title", "action", "insight", "hypothesis"))
    has_refinement_signal = _has_refinement_semantic_signal(" ".join(semantic_items))
    pruned_paths: set[str] = set()
    fixed_scaffold_roles = _fixed_architecture_scaffold_roles()
    fixed_scaffold_paths = set(fixed_scaffold_roles)
    candidate_paths = _dedupe_nonempty(
        [_normalize_repo_path(path) for path in list(architecture.target_file_tree or [])]
        + [_normalize_repo_path(item.path) for item in list(architecture.file_blueprints or [])]
    )
    non_generic_source_paths = [
        path
        for path in candidate_paths
        if path and path not in fixed_scaffold_paths and _architecture_path_roles(path)
    ]
    task_specific_roles: set[str] = set()
    for path in non_generic_source_paths:
        task_specific_roles.update(_architecture_path_roles(path))
    if not has_refinement_signal:
        for path in candidate_paths:
            if "refinement" in _architecture_path_roles(path):
                if domain_root and path.startswith(domain_root + "/"):
                    continue
                pruned_paths.add(path)
    if len(non_generic_source_paths) >= 6:
        for path, roles in fixed_scaffold_roles.items():
            if path in pruned_paths:
                continue
            if roles and roles.issubset(task_specific_roles):
                pruned_paths.add(path)
    if len(non_generic_source_paths) >= 6:
        for item in list(architecture.file_blueprints or []):
            path = _normalize_repo_path(item.path)
            if domain_root and path.startswith(domain_root + "/"):
                continue
            purpose = str(item.purpose or "").strip().lower()
            if (
                path in fixed_scaffold_paths
                and purpose == f"implement {path.lower()}."
                and not list(item.related_node_ids or [])
            ):
                pruned_paths.add(path)
    task_specific_package_paths = [
        path
        for path in candidate_paths
        if path.endswith(".py")
        and "/" in path
        and not path.startswith(("tests/", "scripts/", "configs/"))
        and not path.endswith("__init__.py")
        and (not path.startswith("src/") or path.count("/") >= 2)
    ]
    task_specific_package_roots: dict[str, int] = {}
    for path in task_specific_package_paths:
        parts = path.split("/")
        root = "/".join(parts[:2]) if parts[0] == "src" and len(parts) >= 3 else parts[0]
        task_specific_package_roots[root] = task_specific_package_roots.get(root, 0) + 1
    task_specific_package_roles: set[str] = set()
    if any(count >= 3 for count in task_specific_package_roots.values()):
        for path in task_specific_package_paths:
            task_specific_package_roles.update(_architecture_path_roles(path))
        for item in list(architecture.file_blueprints or []):
            path = _normalize_repo_path(item.path)
            if not path.startswith("src/") or not path.endswith(".py") or path.endswith("__init__.py"):
                continue
            if list(item.based_on_references or []):
                continue
            purpose = str(item.purpose or "").strip().lower()
            if purpose != f"implement {path.lower()}.":
                continue
            pruned_paths.add(path)
    if not pruned_paths:
        return architecture
    file_tree = [path for path in list(architecture.target_file_tree or []) if _normalize_repo_path(path) not in pruned_paths]
    blueprint_paths = {_normalize_repo_path(path) for path in file_tree}
    kept_blueprints = [
        item
        for item in list(architecture.file_blueprints or [])
        if _normalize_repo_path(item.path) in blueprint_paths
    ]
    kept_by_path = {_normalize_repo_path(item.path): item for item in kept_blueprints}
    for item in list(architecture.file_blueprints or []):
        pruned_path = _normalize_repo_path(item.path)
        if pruned_path not in pruned_paths or not list(item.related_node_ids or []):
            continue
        pruned_roles = _architecture_path_roles(pruned_path)
        target_path = ""
        for candidate in kept_blueprints:
            candidate_path = _normalize_repo_path(candidate.path)
            if pruned_roles and pruned_roles.intersection(_architecture_path_roles(candidate_path)):
                target_path = candidate_path
                break
        if not target_path and kept_blueprints:
            target_path = _normalize_repo_path(kept_blueprints[0].path)
        if not target_path or target_path not in kept_by_path:
            continue
        target = kept_by_path[target_path]
        kept_by_path[target_path] = target.model_copy(
            update={
                "related_node_ids": _dedupe_nonempty(
                    list(target.related_node_ids or []) + list(item.related_node_ids or [])
                ),
                "based_on_references": _dedupe_nonempty(
                    list(target.based_on_references or []) + list(item.based_on_references or [])
                ),
            }
        )
    return architecture.model_copy(
        update={
            "target_file_tree": file_tree,
            "file_blueprints": [kept_by_path[path] for path in file_tree if path in kept_by_path],
            "dependency_graph": [
                edge
                for edge in list(architecture.dependency_graph or [])
                if _normalize_repo_path(edge.source_path) in blueprint_paths
                and _normalize_repo_path(edge.target_path) in blueprint_paths
            ],
            "stable_interfaces": [
                path for path in list(architecture.stable_interfaces or []) if _normalize_repo_path(path) in blueprint_paths
            ],
            "execution_entrypoints": [
                path for path in list(architecture.execution_entrypoints or []) if _normalize_repo_path(path) in blueprint_paths
            ],
            "config_surfaces": [
                path for path in list(architecture.config_surfaces or []) if _normalize_repo_path(path) in blueprint_paths
            ],
            "package_layout": {
                work_package_id: [
                    path for path in list(paths or []) if _normalize_repo_path(path) in blueprint_paths
                ]
                for work_package_id, paths in dict(architecture.package_layout or {}).items()
            },
        }
    )


def _fixed_architecture_scaffold_roles() -> dict[str, set[str]]:
    return {}


def _architecture_path_roles(path: str) -> set[str]:
    normalized = _normalize_repo_path(path).lower()
    tokens = set(
        _tokenize_text(
            normalized.replace("/", " ").replace("_", " ").replace("-", " ").replace(".", " ")
        )
    )
    roles: set[str] = set()
    if "config" in tokens or normalized.startswith("configs/"):
        roles.add("config")
    if tokens.intersection({"environment", "environments", "env", "problem", "problems", "target", "targets"}):
        roles.add("environment")
    if tokens.intersection({"data", "dataset", "datasets", "loader", "sampling", "sampler", "samples"}):
        roles.add("data")
    if tokens.intersection({"experiment", "experiments", "runner", "run", "matrix", "protocol"}):
        roles.add("experiment")
    if tokens.intersection({"evaluation", "evaluate", "metrics", "metric", "score", "scores", "landscape"}):
        roles.add("evaluation")
    if tokens.intersection({"train", "training", "trainer", "optimizer", "optimizers", "optimization", "fit", "runner"}):
        roles.add("training")
    if tokens.intersection({"baseline", "baselines", "comparison"}):
        roles.add("baseline")
    if tokens.intersection({"method", "methods", "algorithm", "algorithms", "loss", "losses"}):
        roles.add("method")
    if tokens.intersection({"model", "models", "network", "networks", "latent", "decoder", "encoder"}):
        roles.add("model")
    if tokens.intersection({"agent", "agents", "policy", "policies"}):
        roles.add("agent")
    if tokens.intersection({"refine", "refinement", "adaptation", "adapter"}):
        roles.add("refinement")
    if tokens.intersection({"artifact", "artifacts", "report", "reports", "reporting", "manifest", "writer", "tables", "table"}):
        roles.add("artifact")
    if tokens.intersection({"plot", "plots", "plotting", "figure", "figures", "visualization", "visualize", "reporting"}):
        roles.add("plotting")
    return roles


def _architecture_work_package_roles(text: str) -> set[str]:
    lowered = str(text or "").lower()
    tokens = set(
        _tokenize_text(
            lowered.replace("/", " ").replace("_", " ").replace("-", " ").replace(".", " ")
        )
    )
    roles = _architecture_path_roles(text)
    if tokens.intersection({"method", "methods", "algorithm", "algorithms", "objective", "loss", "attack", "selector", "mask", "smm", "backbone", "policy"}):
        roles.add("method")
    if tokens.intersection({"model", "models", "network", "networks", "encoder", "decoder"}):
        roles.add("model")
    if tokens.intersection({"agent", "agents", "policy", "policies"}):
        roles.add("agent")
    if tokens.intersection({"dataset", "datasets", "data", "loader", "preprocess", "preprocessing", "sampling", "imagenet"}):
        roles.add("data")
    if tokens.intersection({"environment", "environments", "env", "simulator", "task", "benchmark"}):
        roles.add("environment")
    if tokens.intersection({"train", "training", "trainer", "optimizer", "optimizers", "finetune", "pretrain"}):
        roles.add("training")
    if tokens.intersection({"baseline", "baselines", "ablation", "variant", "variants", "comparison", "compare"}):
        roles.add("baseline")
    if tokens.intersection({"evaluate", "evaluation", "metric", "metrics", "score", "scores", "fidelity", "aggregate", "aggregator"}):
        roles.add("evaluation")
    if tokens.intersection({"experiment", "experiments", "protocol", "matrix", "study"}):
        roles.add("experiment")
    if tokens.intersection({"artifact", "artifacts", "result", "results", "report", "reporting", "plot", "figure", "figures", "writer", "summary", "manifest"}):
        roles.add("artifact")
    if tokens.intersection({"config", "configuration", "hyperparameter", "parameter", "sweep", "seed"}):
        roles.add("config")
    if tokens.intersection({"test", "tests", "validation", "assert", "smoke"}):
        roles.add("test")
    return roles


def _architecture_path_role_covered(file_tree: set[str], required_path: str) -> bool:
    required_roles = _architecture_path_roles(required_path)
    if not required_roles:
        return False
    for candidate in file_tree:
        if candidate == required_path:
            return True
        candidate_roles = _architecture_path_roles(candidate)
        if required_roles.intersection(candidate_roles):
            return True
    return False


def _architecture_keep_llm_file_tree(architecture: ArchitectureOutput) -> bool:
    """Return whether the draft already carries a domain-specific package layout.

    Deterministic architecture repair should close missing entry/config/artifact
    paths, but it must not replace a concrete LLM-authored package such as
    `src/<paper_package>/lora.py`/`pruning.py`/`train.py` with generic scaffold
    owners.  This detector is intentionally structural rather than paper-name
    specific.
    """
    paths = _dedupe_nonempty(
        [_normalize_repo_path(path) for path in list(architecture.target_file_tree or [])]
        + [_normalize_repo_path(item.path) for item in list(architecture.file_blueprints or [])]
    )
    package_counts: dict[str, int] = {}
    domain_role_paths = 0
    for path in paths:
        if not path.endswith(".py") or path.startswith("tests/") or path.endswith("__init__.py"):
            continue
        parts = path.split("/")
        if parts[0] == "src" and len(parts) >= 3:
            package_root = "/".join(parts[:2])
        elif len(parts) >= 2:
            package_root = parts[0]
        else:
            continue
        package_counts[package_root] = package_counts.get(package_root, 0) + 1
        roles = _architecture_path_roles(path)
        if roles.intersection({"method", "model", "training", "evaluation", "data", "baseline", "artifact", "config"}):
            domain_role_paths += 1
    return any(count >= 5 for count in package_counts.values()) and domain_role_paths >= 4


def _architecture_domain_package_root(architecture: ArchitectureOutput) -> str:
    paths = _dedupe_nonempty(
        [_normalize_repo_path(path) for path in list(architecture.target_file_tree or [])]
        + [_normalize_repo_path(item.path) for item in list(architecture.file_blueprints or [])]
    )
    package_root = _detect_src_package_root(paths)
    if package_root:
        return package_root
    counts: dict[str, int] = {}
    for path in paths:
        if not path.endswith(".py") or path.startswith("tests/") or path.endswith("__init__.py"):
            continue
        parts = path.split("/")
        if len(parts) >= 3 and parts[0] == "src" and parts[1] not in _NON_PACKAGE_SRC_DIRS:
            root = "/".join(parts[:2])
        elif len(parts) >= 2 and parts[0] not in _NON_PACKAGE_SRC_DIRS.union({"configs", "scripts"}):
            root = parts[0]
        else:
            continue
        counts[root] = counts.get(root, 0) + 1
    if not counts:
        return ""
    return sorted(counts, key=lambda root: (-counts[root], root))[0]


def _architecture_existing_domain_paths(architecture: ArchitectureOutput) -> list[str]:
    root = _architecture_domain_package_root(architecture)
    if not root:
        return []
    return _dedupe_nonempty(
        [
            path
            for path in (
                [_normalize_repo_path(item.path) for item in list(architecture.file_blueprints or [])]
                + [_normalize_repo_path(path) for path in list(architecture.target_file_tree or [])]
            )
            if path.startswith(root + "/")
            and path.endswith(".py")
            and not path.endswith("__init__.py")
        ]
    )


def _architecture_path_semantic_score(source_path: str, candidate_path: str, source_text: str = "") -> int:
    source_roles = _architecture_path_roles(source_path)
    candidate_roles = _architecture_path_roles(candidate_path)
    score = len(source_roles.intersection(candidate_roles)) * 8
    source_tokens = _architecture_owner_tokens_for_path(source_path, source_text)
    candidate_tokens = _architecture_owner_tokens_for_path(candidate_path, "")
    score += len(source_tokens.intersection(candidate_tokens)) * 3
    source_stem = _normalize_repo_path(source_path).rsplit("/", 1)[-1].rsplit(".", 1)[0]
    candidate_stem = _normalize_repo_path(candidate_path).rsplit("/", 1)[-1].rsplit(".", 1)[0]
    if source_stem and candidate_stem and (source_stem == candidate_stem or source_stem in candidate_stem or candidate_stem in source_stem):
        score += 10
    return score


def _architecture_map_generic_paths_to_llm_tree(
    architecture: ArchitectureOutput,
    paths: list[str],
    *,
    text_by_path: dict[str, str] | None = None,
) -> list[str]:
    """Map deterministic scaffold paths onto an existing paper-specific package tree."""
    if not _architecture_keep_llm_file_tree(architecture):
        return _dedupe_nonempty([_normalize_repo_path(path) for path in paths if _normalize_repo_path(path)])
    domain_paths = _architecture_existing_domain_paths(architecture)
    if not domain_paths:
        return _dedupe_nonempty([_normalize_repo_path(path) for path in paths if _normalize_repo_path(path)])
    mapped: list[str] = []
    for raw_path in paths:
        path = _normalize_repo_path(raw_path)
        if not path:
            continue
        if (
            path.startswith(("src/methods/", "src/method/", "src/models/", "src/model/", "src/data/", "src/datasets/", "src/experiments/", "src/reporting/"))
            or (path.startswith("src/") and len(path.split("/")) == 2 and path.endswith(".py"))
        ):
            source_text = str(dict(text_by_path or {}).get(path, "") or "")
            scored = sorted(
                (
                    (_architecture_path_semantic_score(path, candidate, source_text), candidate)
                    for candidate in domain_paths
                ),
                key=lambda item: (-item[0], item[1]),
            )
            if scored and scored[0][0] > 0:
                mapped.append(scored[0][1])
                continue
        mapped.append(path)
    return _dedupe_nonempty(mapped)


def _architecture_remap_contract_targets_for_llm_tree(
    architecture: ArchitectureOutput,
    contract_targets: dict[str, Any],
) -> dict[str, Any]:
    if not _architecture_keep_llm_file_tree(architecture):
        return contract_targets
    path_text: dict[str, str] = {}
    for work_package_id, paths in dict(contract_targets.get("required_work_package_owner_paths", {}) or {}).items():
        joined = str(work_package_id or "")
        for path in list(paths or []):
            normalized = _normalize_repo_path(path)
            if normalized:
                path_text.setdefault(normalized, joined)
    remapped_owner_paths = {
        str(work_package_id): _architecture_map_generic_paths_to_llm_tree(
            architecture,
            [_normalize_repo_path(path) for path in list(paths or []) if _normalize_repo_path(path)],
            text_by_path=path_text,
        )
        for work_package_id, paths in dict(contract_targets.get("required_work_package_owner_paths", {}) or {}).items()
        if str(work_package_id or "").strip()
    }
    remapped_required = _architecture_map_generic_paths_to_llm_tree(
        architecture,
        [_normalize_repo_path(path) for path in list(contract_targets.get("required_generated_files", []) or []) if _normalize_repo_path(path)],
        text_by_path=path_text,
    )
    return {
        **contract_targets,
        "required_generated_files": remapped_required,
        "recommended_generated_files": _architecture_map_generic_paths_to_llm_tree(
            architecture,
            [_normalize_repo_path(path) for path in list(contract_targets.get("recommended_generated_files", []) or []) if _normalize_repo_path(path)],
            text_by_path=path_text,
        ),
        "required_work_package_owner_paths": remapped_owner_paths,
    }


def _file_path_accepts_formula_obligations(path: str) -> bool:
    roles = _architecture_path_roles(path)
    if roles.intersection({"method", "model", "training", "evaluation", "experiment", "baseline", "refinement", "config", "test"}):
        return True
    lowered = _normalize_repo_path(path).lower()
    return any(
        token in lowered
        for token in (
            "adapter",
            "prun",
            "tuning",
            "distill",
            "train",
            "metric",
            "eval",
            "config",
        )
    )


def _formula_symbols_as_code_names(symbols: list[str]) -> list[str]:
    names: list[str] = []
    for symbol in symbols:
        lowered = str(symbol or "").strip().lower()
        lowered = re.sub(r"[^a-z0-9_]+", "_", lowered)
        lowered = re.sub(r"_+", "_", lowered).strip("_")
        if lowered and not lowered[0].isdigit():
            names.append(lowered)
    return _dedupe_nonempty(names)


def _augment_package_file_plans_with_formula_contract(
    state: PaperBenchReproState,
    file_planning: PackageFilePlanningOutput,
    input_payload: dict[str, Any],
) -> PackageFilePlanningOutput:
    payload = _input_payload_with_formula_contract(input_payload, state)
    if not _formula_algorithm_contract_from_input(payload):
        return file_planning
    patched_plans: list[dict[str, Any]] = []
    saw_formula_owner = False
    for item in list(file_planning.file_plans or []):
        plan = item.model_dump(mode="json")
        target_file = _normalize_repo_path(str(plan.get("target_file", "") or ""))
        scope_parts = [
            target_file,
            str(plan.get("work_package_id", "") or ""),
            str(plan.get("purpose", "") or ""),
            str(plan.get("generation_prompt", "") or ""),
            *[str(value) for key in ("method_obligations", "implementation_surfaces", "interface_contract", "review_points", "defines_symbols", "calls_symbols") for value in list(plan.get(key, []) or [])],
        ]
        obligations = _formula_contract_anchor_obligations(
            payload,
            text_scope=" ".join(_dedupe_nonempty(scope_parts)),
            limit=6,
        )
        if obligations and _file_path_accepts_formula_obligations(target_file):
            saw_formula_owner = True
            plan["method_obligations"] = _dedupe_nonempty(
                [str(value) for value in list(plan.get("method_obligations", []) or [])]
                + obligations
            )[:48]
            plan["review_points"] = _dedupe_nonempty(
                [str(value) for value in list(plan.get("review_points", []) or [])]
                + ["Verify paper formula/algorithm anchors are implemented as executable code/config and reached by the canonical route."]
                + obligations[:3]
            )[:48]
            plan["defines_symbols"] = _dedupe_nonempty(
                [str(value) for value in list(plan.get("defines_symbols", []) or [])]
                + _formula_symbols_as_code_names(
                    _formula_anchor_symbols_for_scope(payload, text_scope=" ".join(scope_parts), limit=12)
                )
            )[:40]
            scope_boundary = dict(plan.get("scope_boundary", {}) or {})
            scope_boundary["preserve"] = _dedupe_nonempty(
                [str(value) for value in list(scope_boundary.get("preserve", []) or [])]
                + obligations[:4]
            )[:32]
            scope_boundary["implementation_focus"] = _dedupe_nonempty(
                [str(value) for value in list(scope_boundary.get("implementation_focus", []) or [])]
                + ["executable formula/algorithm route"]
            )[:24]
            plan["scope_boundary"] = scope_boundary
            plan["generation_prompt"] = _append_text_with_obligations(
                str(plan.get("generation_prompt", "") or ""),
                obligations[:4],
                limit=2000,
            )
        patched_plans.append(plan)

    if not saw_formula_owner and patched_plans:
        ranked = sorted(
            (
                (
                    0 if _file_path_accepts_formula_obligations(str(plan.get("target_file", "") or "")) else 1,
                    str(plan.get("target_file", "") or ""),
                    index,
                )
                for index, plan in enumerate(patched_plans)
            ),
            key=lambda item: (item[0], item[1]),
        )
        index = ranked[0][2]
        plan = dict(patched_plans[index])
        obligations = _formula_contract_anchor_obligations(payload, limit=8)
        plan["method_obligations"] = _dedupe_nonempty(
            [str(value) for value in list(plan.get("method_obligations", []) or [])]
            + obligations
        )[:48]
        plan["review_points"] = _dedupe_nonempty(
            [str(value) for value in list(plan.get("review_points", []) or [])]
            + ["Verify paper formula/algorithm anchors are implemented as executable code/config and reached by the canonical route."]
        )[:48]
        plan["generation_prompt"] = _append_text_with_obligations(
            str(plan.get("generation_prompt", "") or ""),
            obligations[:5],
            limit=2000,
        )
        patched_plans[index] = plan

    return PackageFilePlanningOutput.model_validate(
        {
            "file_plans": patched_plans,
            "planning_notes": _dedupe_nonempty(
                list(file_planning.planning_notes)
                + ["Formula/algorithm anchors were projected into executable file-plan obligations."]
            ),
            "unresolved_review_failures": list(file_planning.unresolved_review_failures),
        }
    )


def _architecture_contract_target_errors(
    architecture: ArchitectureOutput,
    contract_targets: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    file_tree = [_normalize_repo_path(path) for path in architecture.target_file_tree]

    package_layout = dict(architecture.package_layout or {})
    for work_package_id, required_paths in dict(contract_targets.get("required_work_package_owner_paths", {}) or {}).items():
        normalized_work_package_id = str(work_package_id or "").strip()
        if not normalized_work_package_id:
            continue
        expected_paths = [
            _normalize_repo_path(path)
            for path in list(required_paths or [])
            if _normalize_repo_path(path)
        ]
        if not expected_paths:
            continue
        owned_paths = {
            _normalize_repo_path(path)
            for path in list(package_layout.get(normalized_work_package_id, []) or [])
            if _normalize_repo_path(path)
        }
        if not owned_paths.intersection(set(expected_paths).intersection(file_tree)):
            errors.append(
                f"work package `{normalized_work_package_id}` has no compact owner file from required paths: "
                + ", ".join(expected_paths[:6])
            )
    for work_package_id, paths in package_layout.items():
        invalid_paths = [
            _normalize_repo_path(path)
            for path in list(paths or [])
            if _normalize_repo_path(path) and _normalize_repo_path(path) not in file_tree
        ]
        if invalid_paths:
            errors.append(
                f"work package `{work_package_id}` references files outside target_file_tree: "
                + ", ".join(invalid_paths[:8])
            )
    return errors


def _architecture_ref_model_errors(
    architecture: ArchitectureOutput,
    ref_repo_model: dict[str, Any] | None,
) -> list[str]:
    ref_repo_model = dict(ref_repo_model or {})
    errors: list[str] = []
    file_tree = [_normalize_repo_path(path) for path in architecture.target_file_tree]
    if ref_repo_model.get("entrypoint_candidates") and not any(_is_entrypoint_like_path(path) for path in file_tree):
        errors.append("reference repositories expose entry surfaces but architecture has no clear entrypoint-like file")
    if ref_repo_model.get("config_candidates") and not any(_infer_file_kind(path) == "config" for path in file_tree):
        errors.append("reference repositories expose config surfaces but architecture has no config-like file")
    if ref_repo_model.get("surveyed_repos") or ref_repo_model.get("ref_candidate_paths"):
        generic_scaffold_paths: list[str] = []
        for item in architecture.file_blueprints:
            path = _normalize_repo_path(item.path)
            if not path or not path.endswith(".py") or path.endswith("__init__.py"):
                continue
            purpose = str(item.purpose or "").strip().lower()
            if purpose == f"implement {path.lower()}." and not list(item.based_on_references or []):
                generic_scaffold_paths.append(path)
        generic_scaffold_paths = _dedupe_nonempty(generic_scaffold_paths)
        if generic_scaffold_paths:
            errors.append(
                "architecture generic scaffold blueprint lacks reference grounding: "
                + ", ".join(generic_scaffold_paths[:12])
            )
    return errors


def _write_architecture_debug_artifact(
    output_dir: Any,
    filename: str,
    payload: Any,
    *,
    json_default: Callable[[Any], Any],
) -> None:
    debug_dir = output_dir / "nodes" / "plan" / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    path = debug_dir / filename
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )
    register_existing_file(
        path,
        run_dir=output_dir,
        logical_name=filename.rsplit(".", 1)[0],
        kind="output",
        stage="plan",
        node="plan",
        authority="debug_snapshot",
        retention="debug",
    )


def _build_architecture_task_model_input(input_payload: dict[str, Any]) -> dict[str, Any]:
    work_packages = list(dict(input_payload.get("work_package_planning", {}) or {}).get("work_packages", []) or [])
    evidence_bundles = list(input_payload.get("evidence_bundles", []) or [])
    return {
        "target": input_payload.get("target", ""),
        "normalized_input": dict(input_payload.get("normalized_input", {}) or {}),
        "units": list(input_payload.get("units", []) or []),
        "experiment_design": dict(input_payload.get("experiment_design", {}) or {}),
        "dataset_preparation": dict(input_payload.get("dataset_preparation", {}) or {}),
        "topic_profile": dict(input_payload.get("topic_profile", {}) or {}),
        "global_contract": dict(input_payload.get("global_contract", {}) or {}),
        "pipeline_plan": dict(input_payload.get("pipeline_plan", {}) or {}),
        "work_packages": [
            {
                "work_package_id": str(item.get("work_package_id", "") or ""),
                "goal": str(item.get("goal", "") or ""),
                "owned_unit_ids": list(item.get("owned_unit_ids", []) or []),
                "depends_on": list(item.get("depends_on", []) or []),
                "interface_contract": list(item.get("interface_contract", []) or []),
                "inventories": dict(item.get("inventories", {}) or {}),
                "method_obligations": list(item.get("method_obligations", []) or []),
            }
            for item in work_packages
            if isinstance(item, dict)
        ],
        "evidence_bundles": [
            {
                "work_package_id": str(item.get("work_package_id", "") or ""),
                "focus": str(item.get("focus", "") or ""),
                "grounding_status": str(item.get("grounding_status", "") or ""),
                "context_summary": list(item.get("context_summary", []) or []),
                "evidence_links": [
                    {
                        "unit_id": str(link.get("unit_id", "") or ""),
                        "ref_id": str(link.get("ref_id", "") or ""),
                        "file_path": str(link.get("file_path", "") or ""),
                        "why_relevant": str(link.get("why_relevant", "") or ""),
                        "matched_keywords": list(link.get("matched_keywords", []) or []),
                    }
                    for link in list(item.get("evidence_links", []) or [])
                    if isinstance(link, dict)
                ],
            }
            for item in evidence_bundles
            if isinstance(item, dict)
        ],
        "prepared_reference_repositories": list(input_payload.get("prepared_reference_repositories", []) or []),
    }


def _synthesize_architecture_task_model(
    input_payload: dict[str, Any],
    ref_repo_model: dict[str, Any],
) -> ArchitectureTaskModelOutput:
    """Build a local task model so architecture planning needs only one LLM design call."""
    work_packages = [
        item
        for item in list(dict(input_payload.get("work_package_planning", {}) or {}).get("work_packages", []) or [])
        if isinstance(item, dict)
    ]
    pipeline_nodes = [
        item
        for item in list(dict(input_payload.get("pipeline_plan", {}) or {}).get("plan_nodes", []) or [])
        if isinstance(item, dict)
    ]
    global_contract = dict(input_payload.get("global_contract", {}) or {})
    result_targets = [
        item
        for item in list(global_contract.get("result_targets", []) or [])
        if isinstance(item, dict)
    ]
    evidence_bundles = [
        item
        for item in list(input_payload.get("evidence_bundles", []) or [])
        if isinstance(item, dict)
    ]
    evidence_by_package = {
        str(item.get("work_package_id", "") or "").strip(): item
        for item in evidence_bundles
        if str(item.get("work_package_id", "") or "").strip()
    }

    entry_candidates = _dedupe_nonempty(
        [
            str(item.get("name", "") or "")
            for item in pipeline_nodes
            if str(item.get("level", "") or "").strip().lower() == "experiment"
        ]
        + [str(item) for item in list(ref_repo_model.get("entrypoint_candidates", []) or [])]
    )
    execution_entry = entry_candidates[0] if entry_candidates else "main.py"

    stage_sequence = _dedupe_nonempty(
        [str(item) for item in list(global_contract.get("canonical_stage_sequence", []) or [])]
        + [
            str(item.get("name", "") or "")
            for item in pipeline_nodes
            if str(item.get("level", "") or "").strip().lower() in {"experiment", "module"}
        ]
    )
    if not stage_sequence:
        stage_sequence = ["configure", "run core method", "write artifacts"]

    method_spine = _dedupe_nonempty(
        [
            str(obligation)
            for work_package in work_packages
            for obligation in list(work_package.get("method_obligations", []) or [])
        ]
        + [
            str(item)
            for item in list(dict(input_payload.get("topic_profile", {}) or {}).get("prompt_guidance", []) or [])
        ]
    )[:16]

    package_responsibilities: list[dict[str, Any]] = []
    evidence_to_module_mapping: list[dict[str, Any]] = []
    for work_package in work_packages:
        work_package_id = str(work_package.get("work_package_id", "") or "").strip()
        if not work_package_id:
            continue
        inventories = dict(work_package.get("inventories", {}) or {})
        implementation_surfaces = _dedupe_nonempty(
            [str(item) for item in list(inventories.get("implementation_surface_inventory", []) or [])]
            + [str(item) for item in list(work_package.get("produces", []) or [])]
            + [str(item) for item in list(work_package.get("interface_contract", []) or [])]
        )
        interface_surfaces = implementation_surfaces
        package_responsibilities.append(
            {
                "work_package_id": work_package_id,
                "responsibilities": _dedupe_nonempty(
                    [str(work_package.get("goal", "") or "")]
                    + [str(item) for item in list(work_package.get("evidence_needs", []) or [])]
                )[:8],
                "method_obligations": _dedupe_nonempty(
                    [str(item) for item in list(work_package.get("method_obligations", []) or [])]
                )[:10],
                "interface_surfaces": interface_surfaces[:8],
                "owned_unit_ids": _dedupe_nonempty(
                    [str(item) for item in list(work_package.get("owned_unit_ids", []) or [])]
                ),
            }
        )

        evidence = evidence_by_package.get(work_package_id, {})
        influenced_paths = _dedupe_nonempty(
            [
                _source_path_for_surface_hint(str(link.get("file_path", "") or ""))
                or _normalize_repo_path(str(link.get("file_path", "") or ""))
                for link in list(evidence.get("evidence_links", []) or [])
                if isinstance(link, dict)
            ]
        )
        influenced_paths = _sanitize_architecture_source_paths(influenced_paths)
        supporting_references = _dedupe_nonempty(
            [
                str(ref_id)
                for ref_id in list(work_package.get("reference_ids", []) or [])
            ]
            + [
                str(link.get("ref_id", "") or "")
                for link in list(evidence.get("evidence_links", []) or [])
                if isinstance(link, dict)
            ]
        )
        if influenced_paths or supporting_references:
            evidence_to_module_mapping.append(
                {
                    "work_package_id": work_package_id,
                    "influenced_paths": influenced_paths[:8],
                    "supporting_references": supporting_references[:8],
                    "notes": _dedupe_nonempty([str(item) for item in list(evidence.get("context_summary", []) or [])])[:4],
                }
            )

    interface_closure = _dedupe_nonempty(
        [
            str(item)
            for work_package in work_packages
            for item in list(work_package.get("interface_contract", []) or [])
        ]
        + [
            path
            for responsibility in package_responsibilities
            for path in list(responsibility.get("interface_surfaces", []) or [])
        ]
    )[:20]
    artifact_paths = _dedupe_nonempty(
        [
            _normalize_repo_path(str(path))
            for target in result_targets
            for path in list(target.get("artifact_paths", []) or [])
            if _normalize_repo_path(str(path))
        ]
    )
    reproducibility_readiness = _dedupe_nonempty(
        ["entrypoint", "dependency manifest"]
        + (["config surface"] if list(ref_repo_model.get("config_candidates", []) or []) else [])
        + [f"write artifact {path}" for path in artifact_paths[:8]]
    )

    return ArchitectureTaskModelOutput.model_validate(
        {
            "execution_entry": execution_entry,
            "runnable_flow": stage_sequence[:16],
            "method_spine": method_spine,
            "package_responsibilities": package_responsibilities,
            "interface_closure": interface_closure,
            "evidence_to_module_mapping": evidence_to_module_mapping,
            "reproducibility_readiness": reproducibility_readiness,
        }
    )


def _build_ref_repo_model(input_payload: dict[str, Any]) -> dict[str, Any]:
    reference_selection = dict(input_payload.get("reference_selection", {}) or {})
    actionable_ref_ids = {
        str(item.get("ref_id", "") or "").strip()
        for item in list(reference_selection.get("actionable_references", []) or [])
        if isinstance(item, dict) and str(item.get("ref_id", "") or "").strip()
    }
    repositories = [
        item
        for item in list(input_payload.get("prepared_reference_repositories", []) or [])
        if isinstance(item, dict) and str(item.get("ref_id", "") or "").strip() in actionable_ref_ids
    ]
    evidence_bundles = list(input_payload.get("evidence_bundles", []) or [])

    likely_reusable_files: list[str] = []
    protocol_clues: list[str] = []
    entrypoint_candidates: list[str] = []
    config_candidates: list[str] = []
    repo_structure_patterns: list[list[str]] = []
    reference_path_owners: dict[str, list[str]] = {}

    def _record_reference_path(path: str, ref_id: str) -> None:
        normalized = _normalize_repo_path(path)
        if not normalized or not ref_id:
            return
        reference_path_owners.setdefault(normalized, [])
        reference_path_owners[normalized] = _dedupe_nonempty(reference_path_owners[normalized] + [ref_id])

    for item in repositories:
        if not isinstance(item, dict):
            continue
        ref_id = str(item.get("ref_id", "") or "").strip()
        repo_likely_reusable_files = [str(value) for value in item.get("likely_reusable_files", []) or []]
        likely_reusable_files.extend(repo_likely_reusable_files)
        protocol_clues.extend(str(value) for value in item.get("protocol_clues", []) or [])
        top_level_files = [str(value) for value in item.get("top_level_files", []) or []]
        top_python_files = [str(value) for value in item.get("top_python_files", []) or []]
        repo_structure_patterns.append(top_level_files[:12])
        for path in [*top_python_files, *repo_likely_reusable_files]:
            _record_reference_path(path, ref_id)
            lowered = str(path).lower()
            if any(token in lowered for token in ("main", "run", "train", "eval", "cli")):
                entrypoint_candidates.append(str(path))
        for path in [*top_level_files, *top_python_files]:
            _record_reference_path(path, ref_id)
            lowered = str(path).lower()
            if any(token in lowered for token in ("config", "configs", ".yaml", ".yml", ".json", ".toml")):
                config_candidates.append(str(path))

    evidence_file_patterns: list[str] = []
    for bundle in evidence_bundles:
        if not isinstance(bundle, dict):
            continue
        for link in list(bundle.get("evidence_links", []) or []):
            if not isinstance(link, dict):
                continue
            ref_id = str(link.get("ref_id", "") or bundle.get("ref_id", "") or "").strip()
            if ref_id not in actionable_ref_ids:
                continue
            file_path = _normalize_repo_path(str(link.get("file_path", "") or ""))
            if _looks_like_implementation_path(file_path):
                evidence_file_patterns.append(file_path)
                _record_reference_path(file_path, ref_id)

    ref_candidate_paths = _dedupe_nonempty(
        [
            *_sanitize_architecture_source_paths(entrypoint_candidates),
            *_sanitize_architecture_source_paths(config_candidates),
            *_sanitize_architecture_source_paths(evidence_file_patterns),
            *_sanitize_architecture_source_paths(likely_reusable_files),
        ]
    )[:40]

    return {
        "surveyed_repos": _dedupe_nonempty([str(item.get("ref_id", "") or "") for item in repositories if isinstance(item, dict)]),
        "likely_reusable_files": _dedupe_nonempty(likely_reusable_files)[:20],
        "protocol_clues": _dedupe_nonempty(protocol_clues)[:20],
        "entrypoint_candidates": _dedupe_nonempty(entrypoint_candidates)[:12],
        "config_candidates": _dedupe_nonempty(config_candidates)[:12],
        "evidence_file_patterns": _dedupe_nonempty(evidence_file_patterns)[:20],
        "ref_candidate_paths": ref_candidate_paths,
        "reference_path_owners": reference_path_owners,
        "repo_structure_patterns": [item for item in repo_structure_patterns if item][:8],
    }


def _contract_inventory_values(work_packages: list[dict[str, Any]], global_contract: dict[str, Any]) -> dict[str, list[str]]:
    inventories: dict[str, list[str]] = {}
    for inventory_map in [
        dict(global_contract.get("inventories", {}) or {}),
        *[dict(item.get("inventories", {}) or {}) for item in work_packages if isinstance(item, dict)],
    ]:
        for key, values in inventory_map.items():
            normalized_key = str(key or "").strip()
            if not normalized_key:
                continue
            inventories.setdefault(normalized_key, [])
            inventories[normalized_key].extend(_normalize_contract_list(values if isinstance(values, list) else [values]))
    return {key: _dedupe_nonempty(values) for key, values in inventories.items() if values}


_ARCHITECTURE_OWNER_GENERIC_TOKENS = {
    "wp",
    "paper",
    "repo",
    "repository",
    "reproduction",
    "surface",
    "contract",
    "protocol",
    "implementation",
    "impl",
    "package",
    "module",
}


def _architecture_ordered_slug_tokens(*values: str) -> list[str]:
    tokens: list[str] = []
    seen: set[str] = set()
    for value in values:
        for token in re.findall(r"[A-Za-z][A-Za-z0-9]*", str(value or "").lower()):
            if token in seen or token.isdigit():
                continue
            seen.add(token)
            tokens.append(token)
    return tokens


def _architecture_owner_slug(work_package_id: str, text: str = "") -> str:
    """Build one stable owner slug per work package, not one file per claim phrase."""
    package_slug = _slugify_contract_id(str(work_package_id or "").strip())
    if package_slug:
        package_slug = re.sub(r"^(wp|paper)_+", "", package_slug).strip("_")
    package_tokens = [
        token
        for token in _architecture_ordered_slug_tokens(package_slug)
        if token not in _ARCHITECTURE_OWNER_GENERIC_TOKENS and len(token) > 1 and not token.isdigit()
    ]
    if package_tokens:
        return _slugify_contract_id(" ".join(package_tokens[:3])) or "work_package"
    text_tokens = [
        token
        for token in _architecture_ordered_slug_tokens(text)
        if token not in _ARCHITECTURE_OWNER_GENERIC_TOKENS
        and token not in _GENERIC_SURFACE_TERMS
        and len(token) > 1
        and not token.isdigit()
    ]
    return _slugify_contract_id(" ".join(text_tokens[:3]) or package_slug or "work_package") or "work_package"


def _architecture_work_package_text(work_package: dict[str, Any], responsibility: Any | None = None) -> str:
    parts: list[str] = [
        str(work_package.get("work_package_id", "") or ""),
        str(work_package.get("goal", "") or ""),
        str(work_package.get("hypothesis", "") or ""),
        str(work_package.get("decision_value", "") or ""),
        *[str(item) for item in list(work_package.get("tags", []) or [])],
        *[str(item) for item in list(work_package.get("produces", []) or [])],
        *[str(item) for item in list(work_package.get("interface_contract", []) or [])],
        *[str(item) for item in list(work_package.get("method_obligations", []) or [])],
    ]
    inventories = dict(work_package.get("inventories", {}) or {})
    for values in inventories.values():
        parts.extend(str(item) for item in list(values or []))
    if responsibility is not None:
        parts.extend(str(item) for item in list(getattr(responsibility, "responsibilities", []) or []))
        parts.extend(str(item) for item in list(getattr(responsibility, "method_obligations", []) or []))
        parts.extend(str(item) for item in list(getattr(responsibility, "interface_surfaces", []) or []))
    return " ".join(part for part in parts if part)


def _architecture_work_package_role_text(work_package: dict[str, Any]) -> str:
    """Use ownership-level fields for file role projection; inventories are evidence, not owners."""
    primary_parts = [
        str(work_package.get("work_package_id", "") or ""),
        str(work_package.get("goal", "") or ""),
        *[str(item) for item in list(work_package.get("produces", []) or [])],
    ]
    primary_text = " ".join(part for part in primary_parts if part)
    primary_roles = _architecture_work_package_roles(primary_text)
    tag_parts = [str(item) for item in list(work_package.get("tags", []) or [])]
    surface_inventory = list(dict(work_package.get("inventories", {}) or {}).get("implementation_surface_inventory", []) or [])
    if len(primary_roles) <= 1:
        primary_parts.extend(tag_parts)
        primary_parts.extend(str(item) for item in surface_inventory[:6])
    return " ".join(part for part in primary_parts if part)


def _architecture_owner_paths_for_work_package(
    work_package: dict[str, Any],
    responsibility: Any | None,
) -> list[str]:
    """Choose compact, task-specific owner files for one package."""
    work_package_id = str(work_package.get("work_package_id", "") or "").strip()
    text = _architecture_work_package_text(work_package, responsibility)
    role_text = _architecture_work_package_role_text(work_package)
    lowered = text.lower()
    slug = _architecture_owner_slug(work_package_id, text)
    explicit_paths = _dedupe_nonempty(
        [
            path
            for field in ("produces", "interface_contract", "method_obligations")
            for path in _extract_repo_paths([str(item) for item in list(work_package.get(field, []) or [])])
            if _is_valid_source_repo_path(path)
        ]
    )[:3]
    roles = _architecture_work_package_roles(role_text)
    role_lowered = role_text.lower()
    primary_paths: list[str] = list(explicit_paths)
    support_paths: list[str] = []
    has_core_role = bool(
        roles.intersection(
            {
                "method",
                "model",
                "refinement",
                "agent",
                "data",
                "environment",
                "training",
                "experiment",
                "baseline",
                "evaluation",
                "artifact",
                "plotting",
            }
        )
    )

    if roles.intersection({"method", "model", "refinement", "agent"}) or any(
        token in role_lowered for token in ("mask", "smm", "algorithm", "core method", "policy")
    ):
        primary_paths.append(f"src/methods/{slug}.py")
    if roles.intersection({"data", "environment"}) or any(
        token in role_lowered for token in ("dataset", "preprocess", "loader", "imagenet")
    ):
        primary_paths.append(f"src/data/{slug}.py")
    if roles.intersection({"training", "experiment", "baseline"}) or any(
        token in role_lowered for token in ("ablation", "comparison", "table ", "experiment", "train", "evaluation")
    ):
        primary_paths.append(f"src/experiments/{slug}.py")
    if roles.intersection({"evaluation", "artifact", "plotting"}) or any(
        token in lowered
        for token in ("metric", "metrics", "report", "reporting", "validation", "audit", "artifact", "figure", "table")
    ):
        primary_paths.append(f"src/reporting/{slug}.py")
    if roles.intersection({"config"}) or any(token in role_lowered for token in ("config", "configuration", "hyperparameter", "sweep")):
        support_paths.append(f"configs/{slug}.yaml")
    if any(token in role_lowered for token in ("entrypoint", "entry point", "cli", "command line", "run script", "runner")):
        support_paths.append("main.py")
    if any(token in role_lowered for token in ("unit test", "unit tests", "smoke test", "pytest", "contract test")):
        support_paths.append(f"tests/test_{slug}.py")
    if not any(path.endswith(".py") and not path.startswith("tests/") for path in primary_paths):
        if not has_core_role and support_paths:
            primary_paths.append(support_paths[0])
        else:
            primary_paths.append(f"src/{slug}.py")
    return _dedupe_nonempty(_canonicalize_semantic_repo_paths(primary_paths + support_paths))[:3]


def _responsibility_by_work_package_id(task_model: ArchitectureTaskModelOutput) -> dict[str, Any]:
    return {
        str(item.work_package_id or "").strip(): item
        for item in list(task_model.package_responsibilities or [])
        if str(item.work_package_id or "").strip()
    }


def _required_work_package_owner_paths(
    work_packages: list[dict[str, Any]],
    task_model: ArchitectureTaskModelOutput,
) -> dict[str, list[str]]:
    responsibility_by_id = _responsibility_by_work_package_id(task_model)
    owner_paths: dict[str, list[str]] = {}
    for work_package in work_packages:
        work_package_id = str(work_package.get("work_package_id", "") or "").strip()
        if not work_package_id:
            continue
        paths = _architecture_owner_paths_for_work_package(
            work_package,
            responsibility_by_id.get(work_package_id),
        )
        if paths:
            owner_paths[work_package_id] = paths
    return owner_paths


def _recommended_generated_files_from_contract(
    work_packages: list[dict[str, Any]],
    global_contract: dict[str, Any],
    task_model: ArchitectureTaskModelOutput,
) -> list[str]:
    """Project paper-derived inventories into compact architecture owner files."""
    inventories = _contract_inventory_values(work_packages, global_contract)
    priority_files: list[str] = []
    files: list[str] = []
    semantic_items: list[str] = []
    for item in [
        str(getattr(task_model, "execution_entry", "") or ""),
        *[str(value) for value in list(getattr(task_model, "runnable_flow", []) or [])],
        *[str(value) for value in list(getattr(task_model, "interface_closure", []) or [])],
        *[str(value) for value in list(getattr(task_model, "reproducibility_readiness", []) or [])],
    ]:
        embedded_paths = _embedded_implementation_paths_from_text(item)
        files.extend(embedded_paths)
        priority_files.extend(path for path in embedded_paths if _is_entrypoint_like_path(path))
    for work_package in work_packages:
        if not isinstance(work_package, dict):
            continue
        semantic_items.extend(str(item) for item in list(work_package.get("tags", []) or []))
        semantic_items.extend(str(item) for item in list(work_package.get("interface_contract", []) or []))
        semantic_items.extend(str(item) for item in list(work_package.get("method_obligations", []) or []))
        semantic_items.extend(str(item) for item in list(work_package.get("produces", []) or []))
    owner_paths_by_package = _required_work_package_owner_paths(work_packages, task_model)
    for paths in owner_paths_by_package.values():
        priority_files.extend(paths)
    for values in inventories.values():
        semantic_items.extend(str(item) for item in values)
    for contract in list(global_contract.get("work_package_contracts", []) or []):
        if not isinstance(contract, dict):
            continue
        semantic_items.extend(str(item) for item in list(contract.get("interface_contract", []) or []))
        semantic_items.extend(str(item) for item in list(contract.get("method_obligations", []) or []))
        semantic_items.extend(str(item) for item in list(contract.get("produces", []) or []))
        for values in dict(contract.get("inventories", {}) or {}).values():
            semantic_items.extend(str(item) for item in list(values or []))
    for target in list(global_contract.get("result_targets", []) or []):
        if not isinstance(target, dict):
            continue
        semantic_items.extend(
            str(item)
            for item in [
                target.get("target_id", ""),
                target.get("kind", ""),
                target.get("name", ""),
            ]
        )
        semantic_items.extend(str(item) for item in list(target.get("required_inputs", []) or []))
        semantic_items.extend(str(item) for item in list(target.get("artifact_paths", []) or []))
        semantic_items.extend(str(item) for item in list(target.get("coverage_notes", []) or []))
    for responsibility in list(task_model.package_responsibilities or []):
        semantic_items.extend(str(item) for item in list(responsibility.interface_surfaces or []))
        semantic_items.extend(str(item) for item in list(responsibility.method_obligations or []))

    for item in semantic_items:
        embedded_paths = _embedded_implementation_paths_from_text(item)
        files.extend(embedded_paths)
        priority_files.extend(path for path in embedded_paths if _is_entrypoint_like_path(path))

    semantic_text = " ".join(semantic_items).lower()
    if not any(_is_entrypoint_like_path(path) for path in priority_files + files):
        priority_files.append("main.py")
    if any(token in semantic_text for token in ("readme", "documentation", "usage", "instruction")):
        priority_files.append("README.md")
    if any(token in semantic_text for token in ("dependency", "dependencies", "requirements", "install", "package")):
        priority_files.append("requirements.txt")
    if inventories or semantic_items:
        priority_files.append("configs/default.yaml")

    prioritized = _canonicalize_semantic_repo_paths(priority_files)
    remainder = [
        path
        for path in _canonicalize_semantic_repo_paths(files)
        if path not in set(prioritized)
    ]
    return _dedupe_nonempty(prioritized + remainder)[:18]


def _derive_architecture_contract_targets(
    input_payload: dict[str, Any],
    task_model: ArchitectureTaskModelOutput,
) -> dict[str, Any]:
    work_packages = list(dict(input_payload.get("work_package_planning", {}) or {}).get("work_packages", []) or [])
    pipeline_nodes = list(dict(input_payload.get("pipeline_plan", {}) or {}).get("plan_nodes", []) or [])
    global_contract = dict(input_payload.get("global_contract", {}) or {})
    reference_selection = dict(input_payload.get("reference_selection", {}) or {})
    actionable_references = [
        item
        for item in list(reference_selection.get("actionable_references", []) or [])
        if isinstance(item, dict)
    ]
    preferred_reference_ids = _dedupe_nonempty(
        [str(item.get("ref_id", "") or "") for item in actionable_references]
    )
    preferred_reference_id_set = set(preferred_reference_ids)
    reference_keyword_map: dict[str, list[str]] = {}
    for reference in actionable_references:
        ref_id = str(reference.get("ref_id", "") or "").strip()
        if not ref_id:
            continue
        reference_keyword_map[ref_id] = sorted(
            _tokenize_text(
                str(reference.get("title", "") or ""),
                str(reference.get("repository_url", "") or ""),
                str(reference.get("readme_summary", "") or ""),
                str(reference.get("file_tree", "") or ""),
                *[str(item) for item in list(reference.get("reusable_modules", []) or [])],
                *[str(item) for item in list(reference.get("likely_reusable_files", []) or [])],
                *[str(item) for item in list(reference.get("top_python_files", []) or [])],
                *[str(item) for item in list(reference.get("protocol_clues", []) or [])],
            )
        )[:80]

    required_generated_files: list[str] = _recommended_generated_files_from_contract(
        work_packages,
        global_contract,
        task_model,
    )
    architecture_context = dict(input_payload.get("architecture", {}) or {})
    architecture_paths = _dedupe_nonempty(
        [_normalize_repo_path(path) for path in list(architecture_context.get("target_file_tree", []) or []) if _normalize_repo_path(path)]
        + [
            _normalize_repo_path(str(item.get("path", "") or ""))
            for item in list(architecture_context.get("file_blueprints", []) or [])
            if isinstance(item, dict) and _normalize_repo_path(str(item.get("path", "") or ""))
        ]
    )
    if architecture_paths:
        try:
            context_architecture = ArchitectureOutput.model_validate(
                {
                    "target_stack": list(architecture_context.get("target_stack", []) or []),
                    "target_file_tree": architecture_paths,
                    "file_blueprints": [
                        item
                        for item in list(architecture_context.get("file_blueprints", []) or [])
                        if isinstance(item, dict)
                    ],
                    "dependency_graph": list(architecture_context.get("dependency_graph", []) or []),
                    "stable_interfaces": list(architecture_context.get("stable_interfaces", []) or []),
                    "execution_entrypoints": list(architecture_context.get("execution_entrypoints", []) or []),
                    "config_surfaces": list(architecture_context.get("config_surfaces", []) or []),
                    "package_layout": dict(architecture_context.get("package_layout", {}) or {}),
                    "dependency_rules": list(architecture_context.get("dependency_rules", []) or []),
                    "protocol_stages": list(architecture_context.get("protocol_stages", []) or []),
                    "result_targets": list(architecture_context.get("result_targets", []) or []),
                    "architecture_reference_ids": list(architecture_context.get("architecture_reference_ids", []) or []),
                    "unresolved_review_failures": list(architecture_context.get("unresolved_review_failures", []) or []),
                    "rationale": str(architecture_context.get("rationale", "") or ""),
                }
            )
            remapped = _architecture_remap_contract_targets_for_llm_tree(
                context_architecture,
                {
                    "required_generated_files": required_generated_files,
                    "recommended_generated_files": required_generated_files,
                    "required_work_package_owner_paths": _required_work_package_owner_paths(work_packages, task_model),
                },
            )
            required_generated_files = list(remapped.get("required_generated_files", []) or required_generated_files)
        except Exception:
            pass

    required_entry_files = [
        path
        for path in required_generated_files
        if _is_entrypoint_like_path(path)
    ]

    result_artifact_paths: list[str] = []
    artifact_owner_surfaces: dict[str, list[str]] = {}
    for target in list(global_contract.get("result_targets", []) or []):
        if not isinstance(target, dict):
            continue
        normalized_paths = [_normalize_repo_path(str(path)) for path in target.get("artifact_paths", []) or []]
        result_artifact_paths.extend(path for path in normalized_paths if path)
        owner_surfaces: list[str] = []
        for path in normalized_paths:
            if path:
                artifact_owner_surfaces[path] = owner_surfaces

    required_interface_surfaces: list[str] = []
    package_surface_expectations: dict[str, list[str]] = {}

    evidence_bundle_map = {
        str(item.get("work_package_id", "") or "").strip(): dict(item)
        for item in list(input_payload.get("evidence_bundles", []) or [])
        if isinstance(item, dict)
    }
    package_reference_map: dict[str, list[str]] = {}
    critical_grounding_failures: list[str] = []
    critical_grounding_work_packages: list[str] = []
    for work_package in work_packages:
        if not isinstance(work_package, dict):
            continue
        work_package_id = str(work_package.get("work_package_id", "") or "").strip()
        if not work_package_id:
            continue
        evidence_bundle = evidence_bundle_map.get(work_package_id, {})
        repo_reference_ids = _dedupe_nonempty(
            [
                str(ref_id)
                for ref_id in list(work_package.get("reference_ids", []) or [])
                if str(ref_id) in preferred_reference_id_set
            ]
            + [
                str(link.get("ref_id", "") or "")
                for link in list(evidence_bundle.get("evidence_links", []) or [])
                if isinstance(link, dict) and str(link.get("ref_id", "") or "") in preferred_reference_id_set
            ]
        )
        if repo_reference_ids:
            package_reference_map[work_package_id] = repo_reference_ids
        tags = {
            str(item or "").strip().lower()
            for item in list(work_package.get("tags", []) or [])
            if str(item or "").strip()
        }
        goal = str(work_package.get("goal", "") or "").lower()
        has_reference_scope = bool(repo_reference_ids)
        is_critical = bool(tags.intersection(_CRITICAL_GROUNDING_TAGS)) or any(
            token in goal
            for token in ("method", "dataset", "evaluation", "artifact", "entry", "protocol")
        )
        if not (has_reference_scope and is_critical):
            continue
        grounding_status = str(evidence_bundle.get("grounding_status", "") or "").strip().lower()
        if grounding_status in {"grounded", "self_contained"}:
            continue
        critical_grounding_work_packages.append(work_package_id)
        critical_grounding_failures.append(
            f"critical package `{work_package_id}` needs grounded evidence before architecture planning"
        )

    return {
        "required_work_package_ids": _dedupe_nonempty(
            [str(item.get("work_package_id", "") or "") for item in work_packages if isinstance(item, dict)]
        ),
        "required_plan_node_ids": _dedupe_nonempty(
            [str(item.get("node_id", "") or "") for item in pipeline_nodes if isinstance(item, dict)]
        ),
        "required_generated_files": _dedupe_nonempty(required_generated_files),
        "recommended_generated_files": _dedupe_nonempty(required_generated_files),
        "required_work_package_owner_paths": _required_work_package_owner_paths(work_packages, task_model),
        "required_entry_files": _dedupe_nonempty(required_entry_files),
        "required_result_artifacts": _dedupe_nonempty(result_artifact_paths),
        "required_interface_surfaces": required_interface_surfaces,
        "package_surface_expectations": package_surface_expectations,
        "artifact_owner_surfaces": artifact_owner_surfaces,
        "preferred_reference_ids": preferred_reference_ids,
        "package_reference_map": package_reference_map,
        "reference_keyword_map": reference_keyword_map,
        "task_model_runnable_flow": list(task_model.runnable_flow),
        "task_model_method_spine": list(task_model.method_spine),
        "task_model_interface_closure": list(task_model.interface_closure),
        "task_model_reproducibility_readiness": list(task_model.reproducibility_readiness),
        "critical_grounding_failures": _dedupe_nonempty(critical_grounding_failures),
        "critical_grounding_work_packages": _dedupe_nonempty(critical_grounding_work_packages),
    }


def _normalize_architecture_output(architecture: ArchitectureOutput) -> ArchitectureOutput:
    raw_file_tree = _sanitize_architecture_source_paths(
        [_normalize_repo_path(path) for path in architecture.target_file_tree]
        + [_normalize_repo_path(item.path) for item in architecture.file_blueprints]
    )
    if not raw_file_tree:
        raw_file_tree = ["main.py"]
    package_root = _detect_src_package_root(raw_file_tree)
    package_alias_map = _package_module_alias_map(raw_file_tree, package_root)
    raw_file_tree = _canonicalize_semantic_repo_paths(
        raw_file_tree,
        package_root=package_root,
        package_alias_map=package_alias_map,
    )

    blueprint_by_path: dict[str, dict[str, Any]] = {}
    for item in architecture.file_blueprints:
        path = _align_repo_path_to_package_layout(
            item.path,
            package_root=package_root,
            package_alias_map=package_alias_map,
        )
        hinted_path = _source_path_for_surface_hint(path)
        path = _canonical_repo_path(
            hinted_path or path,
            package_root=package_root,
            package_alias_map=package_alias_map,
        )
        if not _is_valid_source_repo_path(path) or path in blueprint_by_path:
            continue
        blueprint_by_path[path] = {
            "path": path,
            "purpose": str(item.purpose or "").strip() or _infer_file_purpose(path),
            "kind": _normalize_architecture_blueprint_kind(item.kind, path),
            "related_node_ids": _dedupe_nonempty(list(item.related_node_ids)),
            "based_on_references": _dedupe_nonempty(list(item.based_on_references)),
            "implementation_strategy": item.implementation_strategy,
        }

    semantic_path_map = {
        _normalize_repo_path(original_path): _canonical_repo_path(
            original_path,
            package_root=package_root,
            package_alias_map=package_alias_map,
        )
        for original_path in raw_file_tree
        if _normalize_repo_path(original_path)
    }
    for original_path, blueprint in list(blueprint_by_path.items()):
        canonical_path = semantic_path_map.get(original_path, original_path)
        if canonical_path == original_path:
            continue
        existing = blueprint_by_path.get(canonical_path, {})
        blueprint_by_path[canonical_path] = {
            **blueprint,
            **existing,
            "path": canonical_path,
            "purpose": str(existing.get("purpose") or blueprint.get("purpose") or _infer_file_purpose(canonical_path)),
            "related_node_ids": _dedupe_nonempty(
                list(existing.get("related_node_ids", []) or []) + list(blueprint.get("related_node_ids", []) or [])
            ),
            "based_on_references": _dedupe_nonempty(
                list(existing.get("based_on_references", []) or []) + list(blueprint.get("based_on_references", []) or [])
            ),
        }
        if original_path != canonical_path:
            blueprint_by_path.pop(original_path, None)

    file_tree = _canonicalize_semantic_repo_paths(
        raw_file_tree,
        package_root=package_root,
        package_alias_map=package_alias_map,
    )
    for path in file_tree:
        blueprint_by_path.setdefault(
            path,
            {
                "path": path,
                "purpose": _infer_file_purpose(path),
                "kind": _infer_file_kind(path),
                "related_node_ids": [],
                "based_on_references": [],
                "implementation_strategy": "new",
            },
        )

    dependency_seen: set[tuple[str, str, str]] = set()
    dependency_graph: list[dict[str, Any]] = []
    file_tree_set = set(file_tree)
    for edge in architecture.dependency_graph:
        source_path = _canonical_repo_path(
            edge.source_path,
            package_root=package_root,
            package_alias_map=package_alias_map,
        )
        target_path = _canonical_repo_path(
            edge.target_path,
            package_root=package_root,
            package_alias_map=package_alias_map,
        )
        dependency_type = str(edge.dependency_type or "imports").strip() or "imports"
        if (
            not source_path
            or not target_path
            or source_path == target_path
            or source_path not in file_tree_set
            or target_path not in file_tree_set
        ):
            continue
        key = (source_path, target_path, dependency_type)
        if key in dependency_seen:
            continue
        dependency_seen.add(key)
        dependency_graph.append(
            {
                "source_path": source_path,
                "target_path": target_path,
                "dependency_type": dependency_type,
            }
        )

    normalized_stable_interfaces = _dedupe_nonempty(
        [
            path
            for path in _sanitize_architecture_source_paths(
                [_normalize_repo_path(path) for path in list(architecture.stable_interfaces or [])]
            )
            for path in [
                _canonical_repo_path(
                    path,
                    package_root=package_root,
                    package_alias_map=package_alias_map,
                )
            ]
            if path in file_tree_set
        ]
    )
    normalized_execution_entrypoints = _dedupe_nonempty(
        [
            path
            for path in _sanitize_architecture_source_paths(
                [_normalize_repo_path(path) for path in list(architecture.execution_entrypoints or [])]
            )
            for path in [
                _canonical_repo_path(
                    path,
                    package_root=package_root,
                    package_alias_map=package_alias_map,
                )
            ]
            if path in file_tree_set
        ]
    )
    normalized_config_surfaces = _dedupe_nonempty(
        [
            path
            for path in _sanitize_architecture_source_paths(
                [_normalize_repo_path(path) for path in list(architecture.config_surfaces or [])]
            )
            for path in [
                _canonical_repo_path(
                    path,
                    package_root=package_root,
                    package_alias_map=package_alias_map,
                )
            ]
            if path in file_tree_set
        ]
    )
    normalized_package_layout = {
        str(work_package_id): [
            path
            for path in _sanitize_architecture_source_paths([_normalize_repo_path(path) for path in list(paths or [])])
            for path in [
                _canonical_repo_path(
                    path,
                    package_root=package_root,
                    package_alias_map=package_alias_map,
                )
            ]
            if path in file_tree_set
        ]
        for work_package_id, paths in dict(architecture.package_layout or {}).items()
        if str(work_package_id or "").strip()
    }
    return ArchitectureOutput.model_validate(
        {
            "target_stack": _dedupe_nonempty([str(item) for item in architecture.target_stack]),
            "target_file_tree": file_tree,
            "file_blueprints": [blueprint_by_path[path] for path in file_tree],
            "dependency_graph": dependency_graph,
            "stable_interfaces": normalized_stable_interfaces,
            "execution_entrypoints": normalized_execution_entrypoints,
            "config_surfaces": normalized_config_surfaces,
            "package_layout": normalized_package_layout,
            "dependency_rules": _positive_architecture_dependency_rules(
                dependency_graph,
                stable_interfaces=normalized_stable_interfaces,
                execution_entrypoints=normalized_execution_entrypoints,
                config_surfaces=normalized_config_surfaces,
                package_layout=normalized_package_layout,
            ),
            "protocol_stages": _dedupe_nonempty([str(item) for item in list(architecture.protocol_stages or [])]),
            "result_targets": _dedupe_nonempty(
                [_normalize_repo_path(path) for path in list(architecture.result_targets or [])]
            ),
            "architecture_reference_ids": _dedupe_nonempty([str(item) for item in architecture.architecture_reference_ids]),
            "rationale": str(architecture.rationale or "").strip(),
        }
    )


def _filter_architecture_to_known_work_packages(
    architecture: ArchitectureOutput,
    known_work_package_ids: set[str],
) -> ArchitectureOutput:
    if not known_work_package_ids:
        return architecture
    return architecture.model_copy(
        update={
            "package_layout": {
                work_package_id: paths
                for work_package_id, paths in dict(architecture.package_layout or {}).items()
                if str(work_package_id or "").strip() in known_work_package_ids
            },
            "unresolved_review_failures": list(architecture.unresolved_review_failures or []),
        }
    )


def _close_architecture_package_layout(
    architecture: ArchitectureOutput,
    *,
    contract_targets: dict[str, Any],
    task_model: ArchitectureTaskModelOutput,
) -> ArchitectureOutput:
    file_tree = {_normalize_repo_path(path) for path in list(architecture.target_file_tree or [])}
    package_layout = {
        str(work_package_id): [
            _normalize_repo_path(path)
            for path in list(paths or [])
            if _normalize_repo_path(path) in file_tree
        ]
        for work_package_id, paths in dict(architecture.package_layout or {}).items()
        if str(work_package_id or "").strip()
    }
    for work_package_id, required_paths in dict(contract_targets.get("required_work_package_owner_paths", {}) or {}).items():
        normalized_work_package_id = str(work_package_id or "").strip()
        if not normalized_work_package_id:
            continue
        owned = list(package_layout.get(normalized_work_package_id, []) or [])
        for path in list(required_paths or []):
            normalized = _normalize_repo_path(path)
            if normalized in file_tree:
                owned.append(normalized)
        package_layout[normalized_work_package_id] = _dedupe_nonempty(owned)

    assigned_paths = {
        path
        for paths in package_layout.values()
        for path in list(paths or [])
        if path in file_tree
    }
    blueprint_by_path = {
        _normalize_repo_path(item.path): item.model_dump(mode="json")
        for item in list(architecture.file_blueprints or [])
        if _normalize_repo_path(item.path)
    }
    fallback_owner = next(iter(package_layout), "")
    for path in list(architecture.target_file_tree or []):
        normalized = _normalize_repo_path(path)
        if not normalized or normalized in assigned_paths:
            continue
        owner = _infer_architecture_owner_for_path(
            normalized,
            blueprint_by_path.get(normalized, {}),
            task_model,
            fallback_owner=fallback_owner,
        )
        if not owner:
            continue
        package_layout[owner] = _dedupe_nonempty(list(package_layout.get(owner, []) or []) + [normalized])
        assigned_paths.add(normalized)
    return architecture.model_copy(
        update={
            "package_layout": {
                work_package_id: paths
                for work_package_id, paths in package_layout.items()
                if paths
            }
        }
    )


def _coerce_architecture_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Best-effort payload coercion before ArchitectureOutput schema validation."""
    normalized = dict(payload or {})
    file_blueprints: list[dict[str, Any]] = []
    for item in list(normalized.get("file_blueprints", []) or []):
        if not isinstance(item, dict):
            continue
        path = _normalize_repo_path(str(item.get("path", "") or ""))
        coerced = dict(item)
        if path:
            coerced["path"] = path
        coerced["kind"] = _normalize_architecture_blueprint_kind(item.get("kind"), path)
        coerced["implementation_strategy"] = _normalize_architecture_implementation_strategy(
            item.get("implementation_strategy")
        )
        file_blueprints.append(coerced)
    if file_blueprints:
        normalized["file_blueprints"] = file_blueprints
    normalized["target_file_tree"] = [
        _normalize_repo_path(str(path))
        for path in list(normalized.get("target_file_tree", []) or [])
        if _normalize_repo_path(str(path))
    ]
    return normalized


def _architecture_deviation_report(
    architecture: ArchitectureOutput,
    contract_targets: dict[str, Any],
    ref_repo_model: dict[str, Any] | None = None,
) -> dict[str, Any]:
    valid_node_ids = set(str(item) for item in contract_targets.get("required_plan_node_ids", []) or [])
    valid_reference_ids = set(str(item) for item in contract_targets.get("preferred_reference_ids", []) or [])

    failures: list[str] = []
    warnings: list[str] = []

    file_tree = list(architecture.target_file_tree)
    blueprint_by_path = {item.path: item for item in architecture.file_blueprints}

    if not file_tree:
        failures.append("target_file_tree is empty")
    if not any(path.endswith(".py") for path in file_tree):
        failures.append("architecture must declare at least one Python source file")

    missing_blueprints = [path for path in file_tree if path not in blueprint_by_path]
    if missing_blueprints:
        failures.append("missing file_blueprints for: " + ", ".join(missing_blueprints[:8]))

    extra_blueprints = [path for path in blueprint_by_path if path not in file_tree]
    if extra_blueprints:
        failures.append("file_blueprints contain undeclared paths: " + ", ".join(extra_blueprints[:8]))

    invalid_related_node_refs: list[str] = []
    for item in architecture.file_blueprints:
        invalid = [node_id for node_id in item.related_node_ids if node_id not in valid_node_ids]
        if invalid:
            invalid_related_node_refs.append(f"{item.path}: {', '.join(invalid[:6])}")
    if invalid_related_node_refs:
        failures.append("invalid related_node_ids: " + "; ".join(invalid_related_node_refs[:6]))

    invalid_reference_refs: list[str] = []
    if valid_reference_ids:
        for item in architecture.file_blueprints:
            invalid = [ref_id for ref_id in item.based_on_references if ref_id not in valid_reference_ids]
            if invalid:
                invalid_reference_refs.append(f"{item.path}: {', '.join(invalid[:6])}")
        invalid_arch_refs = [ref_id for ref_id in architecture.architecture_reference_ids if ref_id not in valid_reference_ids]
        if invalid_arch_refs:
            invalid_reference_refs.append("architecture_reference_ids: " + ", ".join(invalid_arch_refs[:6]))
    if invalid_reference_refs:
        failures.append("unknown reference ids: " + "; ".join(invalid_reference_refs[:6]))

    invalid_edges = []
    for edge in architecture.dependency_graph:
        if edge.source_path not in file_tree or edge.target_path not in file_tree:
            invalid_edges.append(f"{edge.source_path}->{edge.target_path}")
    if invalid_edges:
        failures.append("dependency_graph references undeclared files: " + ", ".join(invalid_edges[:8]))

    required_entry_files = set(str(item) for item in contract_targets.get("required_entry_files", []) or [])
    if required_entry_files and not required_entry_files.intersection(file_tree) and not any(
        _architecture_path_role_covered(file_tree, path) for path in required_entry_files
    ):
        failures.append(
            "architecture missing canonical entry surface; expected one of: "
            + ", ".join(sorted(required_entry_files)[:8])
        )

    uncovered_nodes = []
    covered_nodes = {node_id for item in architecture.file_blueprints for node_id in item.related_node_ids}
    for node_id in valid_node_ids:
        if node_id not in covered_nodes:
            uncovered_nodes.append(node_id)
    if uncovered_nodes:
        failures.append("pipeline plan nodes are not covered by file_blueprints: " + ", ".join(uncovered_nodes[:8]))

    required_generated_files = set(str(item) for item in contract_targets.get("required_generated_files", []) or [])
    critical_files = {path for path in required_generated_files if path in required_entry_files}
    missing_critical_files = [
        path
        for path in critical_files
        if path not in file_tree and not _architecture_path_role_covered(file_tree, path)
    ]
    if missing_critical_files:
        failures.append("architecture omitted critical generated files: " + ", ".join(missing_critical_files[:8]))

    if not architecture.rationale.strip():
        warnings.append("architecture rationale is empty")

    failures.extend(_normalize_contract_list(contract_targets.get("critical_grounding_failures", [])))
    failures.extend(_architecture_contract_target_errors(architecture, contract_targets))
    failures.extend(_architecture_ref_model_errors(architecture, ref_repo_model))

    categorized_failures: dict[str, list[str]] = {
        "task_drift": [],
        "ref_drift": [],
        "closure_drift": [],
        "evidence_gap": [],
    }

    for failure in _dedupe_nonempty(failures):
        lowered = failure.lower()
        if any(token in lowered for token in ("critical package", "grounded evidence", "trace")):
            categorized_failures["evidence_gap"].append(failure)
        elif any(
            token in lowered
            for token in (
                "reference repositories expose",
                "grounded reference-repo",
                "unknown reference ids",
                "reference grounding",
                "generic scaffold blueprint",
            )
        ):
            categorized_failures["ref_drift"].append(failure)
        elif any(
            token in lowered
            for token in (
                "canonical entry surface",
                "interface closure surfaces",
                "artifact target",
                "dependency_graph",
                "missing file_blueprints",
                "undeclared files",
                "critical generated files",
                "target_file_tree",
            )
        ):
            categorized_failures["closure_drift"].append(failure)
        else:
            categorized_failures["task_drift"].append(failure)

    categorized_failures = {
        category: _dedupe_nonempty(items)
        for category, items in categorized_failures.items()
        if items
    }
    repair_actions: list[dict[str, Any]] = []
    if categorized_failures.get("evidence_gap"):
        repair_actions.append(
            {
                "action": "refresh_trace_failed_evidence",
                "category": "evidence_gap",
                "reason": "critical work packages are missing grounded evidence",
            }
        )
    if categorized_failures.get("task_drift"):
        repair_actions.append(
            {
                "action": "resynthesize_from_task_model",
                "category": "task_drift",
                "reason": "architecture is not materializing required task obligations",
            }
        )
    if categorized_failures.get("ref_drift"):
        repair_actions.append(
            {
                "action": "resynthesize_from_ref_repo_model",
                "category": "ref_drift",
                "reason": "architecture drifted from reference-repo structure and evidence patterns",
            }
        )
    if categorized_failures.get("closure_drift"):
        repair_actions.append(
            {
                "action": "patch_architecture_field",
                "category": "closure_drift",
                "reason": "architecture contract closure is incomplete and needs deterministic field repair",
            }
        )

    return {
        "status": "passed" if not failures else "failed",
        "failures": _dedupe_nonempty(failures),
        "warnings": warnings,
        "failure_categories": sorted(categorized_failures),
        "failure_groups": categorized_failures,
        "repair_actions": repair_actions,
    }


def _architecture_blueprint_payload_for_path(
    path: str,
    blueprint_candidates: dict[str, dict[str, Any]],
    *,
    valid_node_ids: set[str],
    preferred_reference_ids: set[str],
) -> dict[str, Any]:
    """Project one blueprint onto a required path while stripping invalid refs."""
    normalized_path = _normalize_repo_path(path)
    candidate = dict(blueprint_candidates.get(normalized_path, {}) or {})
    if not candidate:
        basename = normalized_path.rsplit("/", 1)[-1]
        for candidate_path, payload in blueprint_candidates.items():
            if candidate_path.rsplit("/", 1)[-1] == basename:
                candidate = dict(payload or {})
                break
    related_node_ids = [
        node_id
        for node_id in _dedupe_nonempty([str(item) for item in list(candidate.get("related_node_ids", []) or [])])
        if node_id in valid_node_ids
    ]
    based_on_references = [
        ref_id
        for ref_id in _dedupe_nonempty([str(item) for item in list(candidate.get("based_on_references", []) or [])])
        if preferred_reference_ids and ref_id in preferred_reference_ids
    ]
    return {
        "path": normalized_path,
        "purpose": str(candidate.get("purpose", "") or "").strip() or _infer_file_purpose(normalized_path),
        "kind": _normalize_architecture_blueprint_kind(candidate.get("kind"), normalized_path),
        "related_node_ids": related_node_ids,
        "based_on_references": based_on_references,
        "implementation_strategy": _normalize_architecture_implementation_strategy(
            candidate.get("implementation_strategy")
        ),
    }


def _match_plan_nodes_for_path(path: str, plan_nodes: list[Any]) -> list[str]:
    """Heuristically bind uncovered plan nodes onto a likely owning file."""
    normalized_path = _normalize_repo_path(path)
    basename = normalized_path.rsplit("/", 1)[-1].rsplit(".", 1)[0].lower()
    matched: list[str] = []
    for node in list(plan_nodes or []):
        node_id = str(getattr(node, "node_id", "") or "").strip()
        if not node_id:
            continue
        node_terms = " ".join(
            [
                str(getattr(node, "name", "") or ""),
                str(getattr(node, "description", "") or ""),
                str(getattr(node, "reusable_module", "") or ""),
                str(getattr(node, "ref_id", "") or ""),
            ]
        ).lower()
        if basename and basename in node_terms:
            matched.append(node_id)
            continue
        if normalized_path.endswith("main.py") and str(getattr(node, "level", "") or "").strip().lower() == "experiment":
            matched.append(node_id)
    return _dedupe_nonempty(matched)


def _plan_node_owner_tokens(node: Any) -> set[str]:
    return _tokenize_text(
        str(getattr(node, "node_id", "") or ""),
        str(getattr(node, "name", "") or ""),
        str(getattr(node, "description", "") or ""),
        str(getattr(node, "reusable_module", "") or ""),
        str(getattr(node, "ref_id", "") or ""),
        str(getattr(node, "insight", "") or ""),
        str(getattr(node, "hypothesis", "") or ""),
        str(getattr(node, "decision_value", "") or ""),
        str(getattr(node, "stop_rule_or_pruning_rationale", "") or ""),
    )


def _plan_node_path_score(path: str, blueprint: dict[str, Any], node: Any, *, entry_path: str = "") -> int:
    normalized_path = _normalize_repo_path(path)
    lowered_path = normalized_path.lower()
    node_tokens = _plan_node_owner_tokens(node)
    path_tokens = _architecture_owner_tokens_for_path(normalized_path, str(blueprint.get("purpose", "") or ""))
    score = len(path_tokens.intersection(node_tokens)) * 4
    basename = lowered_path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    if basename and basename in node_tokens:
        score += 8
    if lowered_path == entry_path:
        level = str(getattr(node, "level", "") or "").strip().lower()
        if level == "experiment" or node_tokens.intersection({"entry", "entrypoint", "cli", "main", "run", "command"}):
            score += 4
        else:
            score -= 6
    if lowered_path.endswith(("requirements.txt", "pyproject.toml", "setup.py", "setup.cfg")):
        if node_tokens.intersection({"dependency", "dependencies", "package", "install", "requirement"}):
            score += 8
        else:
            score -= 8
    if lowered_path.endswith(("readme.md", "readme.rst")):
        if node_tokens.intersection({"readme", "document", "usage", "instruction"}):
            score += 8
        else:
            score -= 6
    if lowered_path.startswith("tests/"):
        if node_tokens.intersection({"test", "tests", "validation", "contract", "smoke"}):
            score += 8
        else:
            score -= 3
    if "config" in lowered_path or lowered_path.endswith((".yaml", ".yml", ".toml", ".ini", ".cfg")):
        if node_tokens.intersection({"config", "configuration", "parameter", "hyperparameter", "sweep", "seed"}):
            score += 8
    return score


def _assign_uncovered_plan_nodes_to_architecture_paths(
    uncovered_node_ids: list[str],
    normalized_blueprints: list[dict[str, Any]],
    plan_nodes: list[Any],
    *,
    entry_path: str,
) -> dict[str, list[str]]:
    """Distribute uncovered plan nodes by semantic file role instead of overloading the entrypoint."""
    node_by_id = {
        str(getattr(node, "node_id", "") or "").strip(): node
        for node in list(plan_nodes or [])
        if str(getattr(node, "node_id", "") or "").strip()
    }
    paths = [
        _normalize_repo_path(str(item.get("path", "") or ""))
        for item in list(normalized_blueprints or [])
        if _normalize_repo_path(str(item.get("path", "") or ""))
    ]
    non_support_paths = [
        path
        for path in paths
        if path != entry_path
        and not path.startswith("tests/")
        and not path.endswith(("readme.md", "requirements.txt", "pyproject.toml"))
    ]
    assignments: dict[str, list[str]] = {path: [] for path in paths}
    blueprint_by_path = {
        _normalize_repo_path(str(item.get("path", "") or "")): item
        for item in list(normalized_blueprints or [])
        if _normalize_repo_path(str(item.get("path", "") or ""))
    }
    for node_id in _dedupe_nonempty(uncovered_node_ids):
        node = node_by_id.get(node_id)
        if node is None:
            continue
        scored = sorted(
            (
                (
                    _plan_node_path_score(
                        path,
                        blueprint_by_path.get(path, {}),
                        node,
                        entry_path=entry_path,
                    ),
                    path,
                )
                for path in paths
            ),
            key=lambda item: (-item[0], item[1]),
        )
        best_score, best_path = scored[0] if scored else (0, "")
        if best_score <= 0:
            node_text = " ".join(_dedupe_nonempty(list(_plan_node_owner_tokens(node))))
            hinted_path = _source_path_for_surface_hint(node_text)
            if hinted_path in paths:
                best_path = hinted_path
            elif str(getattr(node, "level", "") or "").strip().lower() == "experiment" and entry_path:
                best_path = entry_path
            elif non_support_paths:
                best_path = non_support_paths[0]
            elif paths:
                best_path = paths[0]
        if best_path:
            assignments.setdefault(best_path, []).append(node_id)
    return {path: _dedupe_nonempty(nodes) for path, nodes in assignments.items() if nodes}


def _architecture_owner_tokens_for_path(path: str, purpose: str) -> set[str]:
    normalized = _normalize_repo_path(path).lower()
    tokens = _tokenize_text(normalized, purpose)
    hint_groups: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
        (("main.py", "run", "cli"), ("entrypoint", "entry", "experiment", "protocol")),
        (("config", "yaml", "yml", "toml"), ("config", "configuration", "hyperparameter")),
        (("environment", "environments", "env"), ("environment", "env", "simulator")),
        (("agent", "agents", "policy", "policies"), ("agent", "policy", "model")),
        (("trajectory", "trajectories", "rollout"), ("trajectory", "collector", "episode")),
        (("explainer", "explainers", "explanation", "mask"), ("explainer", "explanation", "mask")),
        (("refinement", "refine"), ("refinement", "refiner", "adaptation")),
        (("baseline", "baselines", "ablation"), ("baseline", "ablation")),
        (("evaluation", "eval", "metric"), ("evaluation", "metric", "score")),
        (("experiment", "experiments", "protocol"), ("experiment", "protocol", "matrix")),
        (("report", "reporting", "artifact"), ("report", "reporting", "artifact", "result")),
        (("test", "tests", "contract"), ("test", "validation", "smoke")),
        (("data", "dataset", "loader"), ("data", "dataset", "preprocess")),
        (("training", "train", "trainer"), ("training", "train", "trainer")),
        (("method", "methods", "algorithm"), ("method", "algorithm")),
    )
    for path_hints, owner_hints in hint_groups:
        if any(hint in normalized for hint in path_hints):
            tokens.update(owner_hints)
    return tokens


def _architecture_owner_score_for_responsibility(
    path: str,
    blueprint: dict[str, Any],
    responsibility: Any,
) -> int:
    file_tokens = _architecture_owner_tokens_for_path(path, str(blueprint.get("purpose", "") or ""))
    owner_tokens = _tokenize_text(
        str(getattr(responsibility, "work_package_id", "") or ""),
        *list(getattr(responsibility, "responsibilities", []) or []),
        *list(getattr(responsibility, "method_obligations", []) or []),
        *list(getattr(responsibility, "interface_surfaces", []) or []),
        *list(getattr(responsibility, "owned_unit_ids", []) or []),
    )
    return len(file_tokens.intersection(owner_tokens))


def _infer_architecture_owner_for_path(
    path: str,
    blueprint: dict[str, Any],
    task_model: ArchitectureTaskModelOutput,
    *,
    fallback_owner: str = "",
) -> str:
    scored: list[tuple[int, str]] = []
    for responsibility in list(task_model.package_responsibilities or []):
        work_package_id = str(responsibility.work_package_id or "").strip()
        if not work_package_id:
            continue
        score = _architecture_owner_score_for_responsibility(path, blueprint, responsibility)
        if score > 0:
            scored.append((score, work_package_id))
    if scored:
        best_score = max(score for score, _work_package_id in scored)
        best_ids = [work_package_id for score, work_package_id in scored if score == best_score]
        for responsibility in list(task_model.package_responsibilities or []):
            work_package_id = str(responsibility.work_package_id or "").strip()
            if work_package_id in best_ids:
                return work_package_id
    normalized = _normalize_repo_path(path)
    if fallback_owner and ("/" not in normalized or normalized.startswith("scripts/") or normalized.endswith("__init__.py")):
        return fallback_owner
    if list(task_model.package_responsibilities or []):
        return str(task_model.package_responsibilities[0].work_package_id or "").strip()
    return fallback_owner


def _blueprint_should_inherit_package_references(path: str, purpose: str, owner: str) -> bool:
    """Decide whether a file should carry owner-level reference provenance."""
    lowered = " ".join([str(path or ""), str(purpose or ""), str(owner or "")]).lower()
    reference_scoped_tokens = (
        "data",
        "dataset",
        "benchmark",
        "loader",
        "preprocess",
        "environment",
        "config",
        "protocol",
        "evaluation",
        "metric",
        "baseline",
        "artifact",
        "table",
        "figure",
        "prompt",
        "cot",
        "toxicity",
        "gsm8k",
        "strategyqa",
        "truthfulqa",
        "scienceqa",
        "toxigen",
    )
    return any(token in lowered for token in reference_scoped_tokens)


def _infer_architecture_references_for_blueprint(
    *,
    path: str,
    blueprint: dict[str, Any],
    owner: str,
    contract_targets: dict[str, Any],
) -> list[str]:
    preferred_reference_ids = {
        str(item or "").strip()
        for item in list(contract_targets.get("preferred_reference_ids", []) or [])
        if str(item or "").strip()
    }
    if not preferred_reference_ids:
        return []
    existing = [
        str(ref_id or "").strip()
        for ref_id in list(blueprint.get("based_on_references", []) or [])
        if str(ref_id or "").strip() in preferred_reference_ids
    ]
    package_reference_map = {
        str(work_package_id): [
            str(ref_id)
            for ref_id in list(ref_ids or [])
            if str(ref_id) in preferred_reference_ids
        ]
        for work_package_id, ref_ids in dict(contract_targets.get("package_reference_map", {}) or {}).items()
    }
    purpose = str(blueprint.get("purpose", "") or "")
    inferred = list(existing)
    if _blueprint_should_inherit_package_references(path, purpose, owner):
        inferred.extend(package_reference_map.get(str(owner or "").strip(), []))

    blueprint_tokens = _tokenize_text(path, purpose, owner)
    for ref_id, keywords in dict(contract_targets.get("reference_keyword_map", {}) or {}).items():
        normalized_ref_id = str(ref_id or "").strip()
        if normalized_ref_id not in preferred_reference_ids:
            continue
        keyword_tokens = _tokenize_text(*[str(item) for item in list(keywords or [])])
        if blueprint_tokens.intersection(keyword_tokens):
            inferred.append(normalized_ref_id)
    return _dedupe_nonempty(inferred)


def _move_path_between_architecture_packages(
    package_layout: dict[str, list[str]],
    *,
    path: str,
    target_work_package_id: str,
) -> None:
    normalized_path = _normalize_repo_path(path)
    target = str(target_work_package_id or "").strip()
    if not normalized_path or not target:
        return
    for work_package_id, paths in list(package_layout.items()):
        if work_package_id == target:
            continue
        package_layout[work_package_id] = [
            item for item in list(paths or []) if _normalize_repo_path(item) != normalized_path
        ]
    package_layout[target] = _dedupe_nonempty(list(package_layout.get(target, []) or []) + [normalized_path])


def _repair_architecture_deterministically(
    architecture: ArchitectureOutput,
    *,
    contract_targets: dict[str, Any],
    task_model: ArchitectureTaskModelOutput,
    plan_nodes: list[Any],
    task_view_architecture: ArchitectureOutput | None = None,
    ref_view_architecture: ArchitectureOutput | None = None,
) -> ArchitectureOutput:
    """Apply local guardrail repairs for repeatable closure/task drift without aborting the run."""
    contract_targets = _architecture_remap_contract_targets_for_llm_tree(architecture, contract_targets)
    candidate_architectures = [
        item
        for item in [architecture, task_view_architecture, ref_view_architecture]
        if item is not None
    ]
    blueprint_candidates: dict[str, dict[str, Any]] = {}
    for candidate in candidate_architectures:
        for blueprint in list(candidate.file_blueprints or []):
            candidate_path = _normalize_repo_path(blueprint.path)
            if not candidate_path or candidate_path in blueprint_candidates:
                continue
            blueprint_candidates[candidate_path] = blueprint.model_dump(mode="json")

    valid_node_ids = {
        str(item or "").strip()
        for item in list(contract_targets.get("required_plan_node_ids", []) or [])
        if str(item or "").strip()
    }
    preferred_reference_ids = {
        str(item or "").strip()
        for item in list(contract_targets.get("preferred_reference_ids", []) or [])
        if str(item or "").strip()
    }
    required_paths = _dedupe_nonempty(
        [_normalize_repo_path(path) for path in list(architecture.target_file_tree or []) if _normalize_repo_path(path)]
        + [
            _normalize_repo_path(path)
            for path in list(contract_targets.get("required_generated_files", []) or [])
            if _normalize_repo_path(path)
        ]
    )
    owner_paths_by_package = {
        str(work_package_id): [
            _normalize_repo_path(path)
            for path in list(paths or [])
            if _normalize_repo_path(path)
        ]
        for work_package_id, paths in dict(contract_targets.get("required_work_package_owner_paths", {}) or {}).items()
        if str(work_package_id or "").strip()
    }
    if not _architecture_keep_llm_file_tree(architecture):
        required_paths = _dedupe_nonempty(
            required_paths
            + [
                path
                for paths in owner_paths_by_package.values()
                for path in paths
            ]
        )
    if not required_paths:
        required_paths = [
            _normalize_repo_path(path)
            for path in list(contract_targets.get("required_entry_files", []) or [])
            if _normalize_repo_path(path)
        ] or ["main.py"]

    normalized_blueprints = [
        _architecture_blueprint_payload_for_path(
            path,
            blueprint_candidates,
            valid_node_ids=valid_node_ids,
            preferred_reference_ids=preferred_reference_ids,
        )
        for path in required_paths
    ]
    covered_node_ids = {
        node_id
        for item in normalized_blueprints
        for node_id in list(item.get("related_node_ids", []) or [])
        if node_id in valid_node_ids
    }
    uncovered_node_ids = [
        node_id
        for node_id in list(valid_node_ids)
        if node_id not in covered_node_ids
    ]
    entry_candidates = [
        _normalize_repo_path(path)
        for path in list(contract_targets.get("required_entry_files", []) or [])
        if _normalize_repo_path(path)
    ]
    entry_path = next((path for path in entry_candidates if path in required_paths), "")
    if not entry_path:
        entry_path = next(
            (
                path
                for path in required_paths
                if _is_entrypoint_like_path(path)
            ),
            "",
        )
    if not entry_path:
        entry_path = entry_candidates[0] if entry_candidates else ""
        if entry_path not in required_paths:
            required_paths = _dedupe_nonempty([entry_path] + required_paths)
            normalized_blueprints = [
                _architecture_blueprint_payload_for_path(
                    path,
                    blueprint_candidates,
                    valid_node_ids=valid_node_ids,
                    preferred_reference_ids=preferred_reference_ids,
                )
                for path in required_paths
            ]

    uncovered_assignments = _assign_uncovered_plan_nodes_to_architecture_paths(
        uncovered_node_ids,
        normalized_blueprints,
        plan_nodes,
        entry_path=entry_path,
    )
    for blueprint in normalized_blueprints:
        path = _normalize_repo_path(str(blueprint.get("path", "") or ""))
        heuristic_nodes = _match_plan_nodes_for_path(path, plan_nodes)
        blueprint["related_node_ids"] = _dedupe_nonempty(
            list(blueprint.get("related_node_ids", []) or [])
            + heuristic_nodes
            + list(uncovered_assignments.get(path, []) or [])
        )

    rationale_lines = [str(architecture.rationale or "").strip()]
    for item in list(task_model.method_spine or []):
        rendered = str(item or "").strip()
        if rendered and rendered not in rationale_lines:
            rationale_lines.append(rendered)

    package_layout = {
        str(work_package_id): _dedupe_nonempty(
            [
                _normalize_repo_path(path)
                for path in list(paths or [])
                if _normalize_repo_path(path) in required_paths
            ]
        )
        for work_package_id, paths in dict(architecture.package_layout or {}).items()
        if str(work_package_id or "").strip()
    }
    root_support_files = [
        path
        for path in required_paths
        if path == entry_path or path.startswith("scripts/")
    ]
    entry_owner = next(
        (
            str(work_package_id)
            for work_package_id, paths in package_layout.items()
            if entry_path in list(paths or [])
        ),
        "",
    )
    if not entry_owner:
        entry_owner = next(
            (
                str(item or "").strip()
                for item in list(contract_targets.get("required_work_package_ids", []) or [])
                if str(item or "").strip()
            ),
            "",
        )
    if entry_owner and root_support_files:
        package_layout[entry_owner] = _dedupe_nonempty(
            list(package_layout.get(entry_owner, []) or []) + root_support_files
        )
    if not _architecture_keep_llm_file_tree(architecture):
        for work_package_id, paths in owner_paths_by_package.items():
            package_layout[work_package_id] = _dedupe_nonempty(
                list(package_layout.get(work_package_id, []) or []) + [
                    path for path in paths if path in required_paths
                ]
            )
    assigned_paths = {
        _normalize_repo_path(path)
        for paths in package_layout.values()
        for path in list(paths or [])
        if _normalize_repo_path(path)
    }
    blueprint_by_path = {
        str(item.get("path", "") or ""): item
        for item in normalized_blueprints
        if str(item.get("path", "") or "")
    }
    for path in required_paths:
        if path in assigned_paths:
            continue
        owner = _infer_architecture_owner_for_path(
            path,
            blueprint_by_path.get(path, {}),
            task_model,
            fallback_owner=entry_owner,
        )
        if not owner:
            continue
        package_layout[owner] = _dedupe_nonempty(list(package_layout.get(owner, []) or []) + [path])
        assigned_paths.add(path)

    owner_by_path: dict[str, str] = {}
    for work_package_id, paths in dict(package_layout or {}).items():
        for path in list(paths or []):
            normalized_path = _normalize_repo_path(path)
            if normalized_path and normalized_path not in owner_by_path:
                owner_by_path[normalized_path] = str(work_package_id or "").strip()
    for blueprint in normalized_blueprints:
        path = _normalize_repo_path(str(blueprint.get("path", "") or ""))
        if not path:
            continue
        owner = owner_by_path.get(path) or _infer_architecture_owner_for_path(
            path,
            blueprint,
            task_model,
            fallback_owner=entry_owner,
        )
        inferred_refs = _infer_architecture_references_for_blueprint(
            path=path,
            blueprint=blueprint,
            owner=owner,
            contract_targets=contract_targets,
        )
        if inferred_refs:
            blueprint["based_on_references"] = inferred_refs
            if blueprint.get("implementation_strategy") == "new":
                blueprint["implementation_strategy"] = "adapted"

    repaired = ArchitectureOutput.model_validate(
        {
            "target_stack": _dedupe_nonempty([str(item) for item in list(architecture.target_stack or [])]),
            "target_file_tree": [item["path"] for item in normalized_blueprints],
            "file_blueprints": normalized_blueprints,
            "dependency_graph": [
                {
                    "source_path": _normalize_repo_path(edge.source_path),
                    "target_path": _normalize_repo_path(edge.target_path),
                    "dependency_type": str(edge.dependency_type or "imports").strip() or "imports",
                }
                for edge in list(architecture.dependency_graph or [])
                if _normalize_repo_path(edge.source_path) and _normalize_repo_path(edge.target_path)
            ],
            "stable_interfaces": _dedupe_nonempty(
                [_normalize_repo_path(path) for path in list(architecture.stable_interfaces or []) if _normalize_repo_path(path)]
                + [entry_path]
            ),
            "execution_entrypoints": _dedupe_nonempty(
                [_normalize_repo_path(path) for path in list(architecture.execution_entrypoints or []) if _normalize_repo_path(path)]
                + entry_candidates
                + [entry_path]
            ),
            "config_surfaces": _dedupe_nonempty(
                [_normalize_repo_path(path) for path in list(architecture.config_surfaces or []) if _normalize_repo_path(path)]
            ),
            "package_layout": package_layout,
            "dependency_rules": _dedupe_nonempty([str(item) for item in list(architecture.dependency_rules or [])]),
            "protocol_stages": _dedupe_nonempty([str(item) for item in list(architecture.protocol_stages or [])]),
            "result_targets": _dedupe_nonempty(
                [_normalize_repo_path(path) for path in list(architecture.result_targets or []) if _normalize_repo_path(path)]
                + [_normalize_repo_path(path) for path in list(contract_targets.get("required_result_artifacts", []) or []) if _normalize_repo_path(path)]
            ),
            "architecture_reference_ids": _dedupe_nonempty(
                [
                    str(item)
                    for item in list(architecture.architecture_reference_ids or [])
                    if not preferred_reference_ids or str(item) in preferred_reference_ids
                ]
                + list(preferred_reference_ids)
            ),
            "unresolved_review_failures": list(architecture.unresolved_review_failures or []),
            "rationale": "\n".join(item for item in rationale_lines if item),
        }
    )
    return _normalize_architecture_output(repaired)


def _artifact_owner_priority(file_path: str) -> int:
    normalized = _normalize_repo_path(file_path).lower()
    if "artifact" in normalized:
        return 0
    if "report" in normalized:
        return 1
    if "plot" in normalized or "figure" in normalized:
        return 2
    if "evaluation" in normalized:
        return 3
    if "experiment" in normalized:
        return 4
    if normalized.endswith(("main.py", "cli.py")):
        return 20
    if "retrieval" in normalized or "pipeline" in normalized:
        return 21
    if "config" in normalized or "models.py" in normalized:
        return 22
    return 10


def _repair_package_file_artifact_coverage(
    file_planning: PackageFilePlanningOutput,
    *,
    global_contract: GlobalContractOutput | None,
) -> PackageFilePlanningOutput:
    """Deterministically assign artifact ownership when review keeps reporting empty writes_artifacts."""
    if global_contract is None or not file_planning.file_plans:
        return file_planning

    artifact_paths = [
        _normalize_repo_path(path)
        for target in list(global_contract.result_targets or [])
        for path in list(target.artifact_paths or [])
        if _normalize_repo_path(path)
    ]
    artifact_paths = _dedupe_nonempty(artifact_paths)
    if not artifact_paths:
        return file_planning

    file_plans = list(file_planning.file_plans or [])
    plan_by_path = {str(item.target_file or "").strip(): item for item in file_plans if str(item.target_file or "").strip()}

    owner_by_artifact: dict[str, str] = {}
    for artifact_path in artifact_paths:
        lowered = artifact_path.lower()
        artifact_name = lowered.rsplit("/", 1)[-1]
        owner_candidates: list[str] = []
        if any(token in artifact_name for token in ("figure", "fig", "plot", "curve", "curves")):
            owner_candidates = [
                path for path in plan_by_path
                if any(token in path.lower() for token in ("plot", "figure", "report", "artifact"))
            ]
        elif "metrics" in lowered or "report" in lowered:
            owner_candidates = [
                path for path in plan_by_path
                if any(token in path.lower() for token in ("artifact", "report"))
            ]
        elif "prediction" in lowered:
            owner_candidates = [
                path for path in plan_by_path
                if any(token in path.lower() for token in ("artifact", "report", "evaluation", "retrieval"))
            ]
        elif lowered.endswith(".jsonl"):
            owner_candidates = [
                path for path in plan_by_path
                if any(token in path.lower() for token in ("artifact", "report", "evaluation"))
            ]
        else:
            owner_candidates = [
                path for path in plan_by_path
                if any(token in path.lower() for token in ("artifact", "report", "evaluation"))
            ]
        if any("artifact" in path.lower() for path in plan_by_path):
            owner_candidates = [path for path in owner_candidates if "artifact" in path.lower()] or owner_candidates
        if not owner_candidates:
            owner_candidates = [
                path for path in plan_by_path
                if not path.lower().endswith(("main.py", "cli.py", "requirements.txt", "pyproject.toml"))
                and not path.lower().startswith("tests/")
            ] or list(plan_by_path)
        if not owner_candidates:
            continue
        owner_by_artifact[artifact_path] = sorted(owner_candidates, key=_artifact_owner_priority)[0]

    repaired_file_plans: list[RepoFilePlan] = []
    for item in file_plans:
        target_file = str(item.target_file or "").strip()
        assigned_writes = [
            artifact_path
            for artifact_path, owner_path in owner_by_artifact.items()
            if owner_path == target_file
        ]
        assigned_reads = [
            artifact_path
            for artifact_path, owner_path in owner_by_artifact.items()
            if owner_path != target_file and any(token in target_file.lower() for token in ("cli.py", "main.py", "pipeline", "evaluation"))
        ]
        repaired_file_plans.append(
            item.model_copy(
                update={
                    "writes_artifacts": _dedupe_nonempty(list(item.writes_artifacts or []) + assigned_writes),
                    "reads_artifacts": _dedupe_nonempty(list(item.reads_artifacts or []) + assigned_reads),
                }
            )
        )

    notes = list(file_planning.planning_notes or [])
    if owner_by_artifact:
        notes.append("Deterministically assigned artifact ownership to file plans after repeated artifact-coverage review failures.")
    return file_planning.model_copy(
        update={
            "file_plans": repaired_file_plans,
            "planning_notes": _dedupe_nonempty(notes),
        }
    )


def _normalize_topic_profile_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload or {})
    hints = normalized.get("coverage_hints")
    if isinstance(hints, dict):
        normalized_hints: dict[str, list[str]] = {}
        for key, value in hints.items():
            if isinstance(value, str):
                normalized_hints[str(key)] = [value] if value.strip() else []
            elif isinstance(value, (list, tuple, set)):
                normalized_hints[str(key)] = [str(item) for item in value if str(item).strip()]
            elif value is None:
                normalized_hints[str(key)] = []
            else:
                normalized_hints[str(key)] = [str(value)]
        normalized["coverage_hints"] = normalized_hints
    return normalized


def topic_profile_impl(
    state: PaperBenchReproState,
    *,
    build_topic_profile_context: Callable[[PaperBenchReproState, Any], dict[str, Any]],
    limit_json_for_prompt: Callable[[Any], str],
    invoke_json_stage: Callable[[str, str, str, str], dict[str, Any]],
    get_output_dir: Callable[[PaperBenchReproState], Any],
    write_stage_output: Callable[[PaperBenchReproState, str, object], None],
    run_or_resume_stage: Callable[..., Any],
    save_tracking_artifacts: Callable[[PaperBenchReproState], None],
) -> PaperBenchReproState:
    logger.info("topic - Synthesizing topic profile...")
    input_payload = build_topic_profile_context(state, state.boundary_requirements)

    def _compute() -> TopicProfileOutput:
        system, user = build_topic_profile_prompt(
            limit_json_for_prompt(input_payload),
            language=state.input.language,
        )
        payload = invoke_json_stage("topic_profile", "topic_profile", system, user)
        return TopicProfileOutput.model_validate(_normalize_topic_profile_payload(payload))

    def _load() -> TopicProfileOutput:
        output_dir = get_output_dir(state)
        return TopicProfileOutput.model_validate(
            _read_plan_json_artifact(output_dir, CANONICAL_ARTIFACTS["topic_profile"])
        )

    def _write(result: TopicProfileOutput) -> None:
        write_stage_output(state, CANONICAL_ARTIFACTS["topic_profile"], result)

    state.topic_profile = run_or_resume_stage(
        state,
        "topic_profile_synthesis",
        input_payload,
        _compute,
        _load,
        _write,
    )
    save_tracking_artifacts(state)
    return state


_WORK_PACKAGE_UNIT_ANCHORS: dict[str, str] = {
    "paper_addendum_constraints": "setup",
    "paper_contract_environment_protocol": "environment",
    "paper_contract_dataset_metric_protocol": "evaluation",
    "paper_contract_experiment_artifact_protocol": "artifact",
    "paper_contract_method_baseline_protocol": "method",
    "paper_contract_sweep_hyperparameter_protocol": "configuration",
    "paper_evidence_matrix": "experiment",
    "paper_dataset_inventory": "data",
    "paper_evaluation_protocol": "evaluation",
    "paper_method_core": "method",
    "paper_named_experiment_protocols": "experiment",
    "paper_task_environment_setup": "setup",
    "paper_training_or_optimization_loop": "training",
}

_WORK_PACKAGE_ROLE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "setup": (
        "addendum",
        "setup",
        "install",
        "dependency",
        "readme",
        "documentation",
        "config",
        "environment",
        "runnable",
        "repository",
    ),
    "environment": ("environment", "task", "simulator", "benchmark", "protocol"),
    "data": ("data", "dataset", "benchmark", "loader", "preprocess", "sampling"),
    "method": ("method", "algorithm", "model", "baseline", "variant", "objective", "loss", "policy"),
    "training": ("train", "training", "pretrain", "fine", "finetune", "optimizer", "checkpoint", "loop"),
    "evaluation": ("evaluate", "evaluation", "metric", "score", "measure", "validation", "compare"),
    "configuration": ("config", "configuration", "parameter", "hyperparameter", "sweep", "seed"),
    "artifact": (
        "artifact",
        "artifacts",
        "result",
        "results",
        "table",
        "tables",
        "figure",
        "figures",
        "plot",
        "plots",
        "report",
        "reports",
        "output",
        "outputs",
        "writer",
    ),
    "experiment": ("experiment", "protocol", "matrix", "ablation", "robustness", "comparison"),
}

_WORK_PACKAGE_EXISTING_OWNER_HINTS: dict[str, tuple[str, ...]] = {
    "setup": ("project_skeleton", "setup", "configuration", "config"),
    "configuration": ("project_skeleton", "setup", "configuration", "config"),
    "artifact": ("main_comparison", "experiment", "evaluation", "comparison", "artifact"),
    "experiment": ("main_comparison", "experiment", "evaluation", "comparison"),
    "evaluation": ("main_comparison", "experiment", "evaluation", "comparison"),
}

_WORK_PACKAGE_ID_STOPWORDS = {
    "paper",
    "contract",
    "protocol",
    "surface",
    "unit",
    "task",
    "implementation",
    "obligation",
    "inventory",
    "evidence",
    "matrix",
    "module",
    "modules",
}


def _unit_text_blob(unit: Any) -> str:
    parts: list[str] = []
    for field_name in (
        "unit_id",
        "type",
        "statement",
        "hypothesis",
        "decision_value",
        "stop_rule_or_pruning_rationale",
    ):
        value = str(getattr(unit, field_name, "") or "").strip()
        if value:
            parts.append(value)
    for field_name in (
        "paper_evidence",
        "implementation_surfaces",
        "code_obligations",
        "runtime_interfaces",
        "expected_artifacts",
        "suggested_module_kinds",
        "implementation_notes",
    ):
        for item in list(getattr(unit, field_name, []) or []):
            value = str(item or "").strip()
            if value:
                parts.append(value)
    return " ".join(parts).lower()


def _work_package_role_for_unit(unit: Any) -> str:
    unit_id = str(getattr(unit, "unit_id", "") or "").strip()
    anchored_role = _WORK_PACKAGE_UNIT_ANCHORS.get(unit_id)
    if anchored_role:
        return anchored_role
    blob = _unit_text_blob(unit)
    scored: list[tuple[int, str]] = []
    for role, keywords in _WORK_PACKAGE_ROLE_KEYWORDS.items():
        score = sum(1 for token in keywords if token in blob)
        if score:
            scored.append((score, role))
    if scored:
        return sorted(scored, key=lambda item: (-item[0], item[1]))[0][1]
    return "implementation"


def _unit_work_package_slug(unit: Any, *, role: str) -> str:
    candidates: list[str] = []
    for field_name in ("statement", "hypothesis", "decision_value"):
        value = str(getattr(unit, field_name, "") or "").strip()
        if value:
            candidates.append(value)
    for field_name in ("expected_artifacts", "implementation_surfaces", "suggested_module_kinds", "code_obligations"):
        candidates.extend(str(item) for item in list(getattr(unit, field_name, []) or []) if str(item).strip())
    candidates.append(str(getattr(unit, "unit_id", "") or ""))
    tokens = [
        token
        for token in _tokenize_text(*candidates)
        if len(token) > 2
        and token not in _WORK_PACKAGE_ID_STOPWORDS
        and token not in _CONTRACT_TOKEN_STOPWORDS
        and token not in set(_WORK_PACKAGE_ROLE_KEYWORDS.get(role, ()))
    ]
    slug = _slugify_contract_id(" ".join(tokens[:5]) or str(getattr(unit, "unit_id", "") or role))
    if slug in {"item", role}:
        slug = _slugify_contract_id(str(getattr(unit, "unit_id", "") or role))
    return f"{role}_{slug}"[:64].strip("_") or role


def _dynamic_work_package_id_for_unit(unit: Any) -> str:
    role = _work_package_role_for_unit(unit)
    if role in {"setup", "configuration"}:
        return "project_skeleton"
    if role in {"artifact", "experiment", "evaluation"}:
        return "main_comparison"
    return f"wp_{_unit_work_package_slug(unit, role=role)}"


def _existing_package_score_for_unit(
    package_id: str,
    unit: Any,
    *,
    role: str,
    package_payload: dict[str, Any] | None = None,
) -> int:
    package_text = package_id
    if package_payload:
        package_text += " " + json.dumps(package_payload, ensure_ascii=False, default=str)
    ignored_tokens = set(_CONTRACT_TOKEN_STOPWORDS).union(_WORK_PACKAGE_ID_STOPWORDS)
    package_tokens = {token for token in _tokenize_text(package_text) if token not in ignored_tokens}
    if not package_tokens:
        return 0
    unit_tokens = {
        token
        for token in _tokenize_text(_unit_text_blob(unit), role, *_WORK_PACKAGE_ROLE_KEYWORDS.get(role, ()))
        if token not in ignored_tokens
    }
    score = len(package_tokens.intersection(unit_tokens))
    if role in package_tokens:
        score += 6
    for owner_hint in _WORK_PACKAGE_EXISTING_OWNER_HINTS.get(role, ()):
        if owner_hint in package_id:
            score += 12
        else:
            score += 4 * len(package_tokens.intersection(_tokenize_text(owner_hint)))
    role_keywords = set(_WORK_PACKAGE_ROLE_KEYWORDS.get(role, ()))
    role_signal = role in package_tokens or bool(package_tokens.intersection(role_keywords))
    if not role_signal and _WORK_PACKAGE_EXISTING_OWNER_HINTS.get(role):
        role_signal = any(owner_hint in package_id for owner_hint in _WORK_PACKAGE_EXISTING_OWNER_HINTS[role])
    if not role_signal:
        return 0
    score += len(package_tokens.intersection(role_keywords)) * 3
    return score


def _unit_work_package_target(unit: Any) -> str:
    return _dynamic_work_package_id_for_unit(unit)


def _synthesize_work_package_from_units(work_package_id: str, owned_units: list[Any]) -> dict[str, Any]:
    first_unit = owned_units[0]
    unit_ids = _dedupe_nonempty([str(getattr(unit, "unit_id", "") or "") for unit in owned_units])
    statements = _dedupe_nonempty([str(getattr(unit, "statement", "") or "") for unit in owned_units])
    surfaces = _dedupe_nonempty(
        [
            str(surface)
            for unit in owned_units
            for surface in list(getattr(unit, "implementation_surfaces", []) or [])
        ]
    )
    runtime_interfaces = _dedupe_nonempty(
        [
            str(interface)
            for unit in owned_units
            for interface in list(getattr(unit, "runtime_interfaces", []) or [])
        ]
    )
    artifacts = _dedupe_nonempty(
        [
            str(artifact)
            for unit in owned_units
            for artifact in list(getattr(unit, "expected_artifacts", []) or [])
        ]
    )
    code_obligations = _dedupe_nonempty(
        [
            str(obligation)
            for unit in owned_units
            for obligation in _unit_positive_obligations(unit, limit=8)
        ]
    )
    role = _work_package_role_for_unit(first_unit)
    if role == "artifact":
        goal = "Own appendix Table/Figure artifact obligations and the associated evidence matrix."
    elif role in {"setup", "configuration"}:
        goal = "Own configuration, setup, and addendum-derived reproducibility coverage."
    elif role in {"experiment", "evaluation"}:
        goal = "Own the paper-derived experiment/evaluation contract and keep required comparisons explicit."
    elif role in {"method", "training"}:
        goal = "Own the paper-derived method/training obligations and expose the active implementation route."
    elif role in {"data", "environment"}:
        goal = "Own paper-derived dataset, environment, and task coverage for the reproduction route."
    else:
        goal = statements[0] if statements else f"Cover {work_package_id}"
    unit_summary = ", ".join(unit_ids[:4]) if unit_ids else work_package_id
    hypothesis = str(getattr(first_unit, "hypothesis", "") or "").strip() or f"Owning {unit_summary} keeps the paper contract explicit."
    decision_value = str(getattr(first_unit, "decision_value", "") or "").strip() or f"Decides coverage for {unit_summary}."
    stop_rule = f"Implementation scope: cover {unit_summary} through the paper-stated contract surfaces."
    inventories = _merge_inventory_maps(
        _implementation_inventory_map(_derive_unit_inventory(owned_units)),
        {
            key: values
            for key, values in {
                "implementation_surface_inventory": surfaces,
                "artifact_inventory": artifacts,
            }.items()
            if values
        },
    )
    if artifacts:
        inventories["artifact_inventory"] = _dedupe_nonempty(
            list(inventories.get("artifact_inventory", [])) + artifacts
        )
    scope_boundary = _scope_boundary_from_units(
        owned_units,
        existing={},
        fallback_focus=[goal, *surfaces[:6], *runtime_interfaces[:6], *artifacts[:6]],
    )
    return {
        "work_package_id": work_package_id,
        "goal": goal,
        "hypothesis": hypothesis,
        "decision_value": decision_value,
        "stop_rule_or_pruning_rationale": stop_rule,
        "owned_unit_ids": unit_ids,
        "tags": _dedupe_nonempty([str(getattr(first_unit, "type", "") or "")] + surfaces),
        "reference_ids": [],
        "depends_on": [],
        "produces": [],
        "interface_contract": _dedupe_nonempty(runtime_interfaces),
        "evidence_needs": _dedupe_nonempty(
            [f"cover unit {unit_id}" for unit_id in unit_ids[:8]]
            + [statement for statement in statements[:4]]
        ),
        "inventories": inventories,
        "scope_boundary": scope_boundary,
        "method_obligations": _dedupe_nonempty(
            code_obligations + _augment_inventory_obligations(inventories)
        ),
    }


def _scope_boundary_from_units(
    owned_units: list[Any],
    *,
    existing: dict[str, Any] | None = None,
    fallback_focus: list[str] | None = None,
) -> dict[str, list[str]]:
    """Build a positive implementation-scope contract from Prepare units."""
    existing = dict(existing or {})
    preserve: list[str] = []
    focus: list[str] = []
    for unit in owned_units:
        unit_id = str(getattr(unit, "unit_id", "") or "").strip()
        statement = str(getattr(unit, "statement", "") or "").strip()
        obligations = _dedupe_nonempty(
            _unit_positive_obligations(unit, limit=8)
            + [str(item) for item in list(getattr(unit, "runtime_interfaces", []) or [])]
            + [str(item) for item in list(getattr(unit, "expected_artifacts", []) or [])]
        )
        pieces = _dedupe_nonempty(obligations[:6] or ([statement] if statement else []))
        if pieces:
            prefix = f"{unit_id}: " if unit_id else ""
            preserve.append(prefix + "; ".join(pieces[:5]))
        focus.extend(str(item) for item in list(getattr(unit, "implementation_surfaces", []) or []))
        focus.extend(str(item) for item in list(getattr(unit, "runtime_interfaces", []) or []))
        focus.extend(str(item) for item in list(getattr(unit, "expected_artifacts", []) or []))

    for item in list(existing.get("preserve", []) or []):
        if str(item or "").strip():
            preserve.append(str(item).strip())
    for item in list(existing.get("implementation_focus", []) or []):
        if str(item or "").strip():
            focus.append(str(item).strip())
    focus.extend(str(item) for item in list(fallback_focus or []) if str(item).strip())

    preserve = _dedupe_nonempty(preserve)
    focus = _dedupe_nonempty(focus)
    if not preserve:
        preserve = [
            "Implement all owned paper/addendum method, protocol, metric, artifact, and bounded execution obligations as code."
        ]
    if not focus:
        focus = [
            "Active code routes, artifact writers, selectors, metric functions, and bounded execution defaults."
        ]
    return {
        "preserve": preserve[:24],
        "implementation_focus": focus[:24],
    }


def _repair_unowned_work_packages(
    result: WorkPackagePlanningOutput,
    *,
    units_by_id: dict[str, Any],
) -> WorkPackagePlanningOutput:
    unit_ids = sorted(units_by_id)
    if not unit_ids or not result.work_packages:
        return result

    package_items = [item.model_dump(mode="json") for item in result.work_packages]
    package_units: dict[str, list[str]] = {}
    covered_unit_ids: set[str] = set()
    for item in package_items:
        package_id = str(item.get("work_package_id", "") or "").strip()
        if not package_id:
            continue
        owned_unit_ids = _dedupe_nonempty(
            [str(unit_id) for unit_id in list(item.get("owned_unit_ids", []) or []) if str(unit_id).strip() in units_by_id]
        )
        package_units[package_id] = list(owned_unit_ids)
        covered_unit_ids.update(owned_unit_ids)

    uncovered_unit_ids = sorted(set(unit_ids) - covered_unit_ids)
    if not uncovered_unit_ids:
        return result

    for unit_id in uncovered_unit_ids:
        target_package_id = _unit_work_package_target(units_by_id[unit_id])
        package_units.setdefault(target_package_id, []).append(unit_id)

    package_lookup = {str(item.get("work_package_id", "") or "").strip(): item for item in package_items}
    final_packages: list[dict[str, Any]] = []
    for package_id, item in package_lookup.items():
        owned_unit_ids = _dedupe_nonempty(package_units.get(package_id, []))
        owned_units = [units_by_id[unit_id] for unit_id in owned_unit_ids if unit_id in units_by_id]
        if owned_units:
            rebuilt = dict(item)
            rebuilt.update(_synthesize_work_package_from_units(package_id, owned_units))
            rebuilt["owned_unit_ids"] = owned_unit_ids
            final_packages.append(rebuilt)
        else:
            final_packages.append(item)

    for package_id, owned_unit_ids in sorted(package_units.items()):
        if package_id in package_lookup:
            continue
        owned_unit_ids = _dedupe_nonempty(owned_unit_ids)
        owned_units = [units_by_id[unit_id] for unit_id in owned_unit_ids if unit_id in units_by_id]
        if not owned_units:
            continue
        final_packages.append(_synthesize_work_package_from_units(package_id, owned_units))

    covered_unit_ids = {
        unit_id
        for item in final_packages
        for unit_id in list(item.get("owned_unit_ids", []) or [])
        if unit_id in units_by_id
    }
    uncovered_unit_ids = sorted(set(unit_ids) - covered_unit_ids)
    planning_notes = _dedupe_nonempty(
        list(result.planning_notes)
        + (
            [f"Recovered unowned work-package units: {', '.join(uncovered_unit_ids[:8])}"]
            if uncovered_unit_ids
            else []
        )
    )
    normalized = WorkPackagePlanningOutput.model_validate(
        {
            "work_packages": final_packages,
            "coverage_summary": {
                "total_units": len(unit_ids),
                "covered_units": len(covered_unit_ids),
                "uncovered_unit_ids": uncovered_unit_ids,
            },
            "planning_notes": planning_notes,
        }
    )
    return normalized


def work_package_planning_impl(
    state: PaperBenchReproState,
    *,
    build_work_package_planning_context: Callable[[PaperBenchReproState], dict[str, Any]],
    build_work_package_local_context: Callable[[PaperBenchReproState, str], dict[str, Any]],
    limit_json_for_prompt: Callable[[Any], str],
    invoke_json_stage: Callable[[str, str, str, str], dict[str, Any]],
    get_output_dir: Callable[[PaperBenchReproState], Any],
    json_default: Callable[[Any], Any],
    write_stage_output: Callable[[PaperBenchReproState, str, object], None],
    run_or_resume_stage: Callable[..., Any],
    save_tracking_artifacts: Callable[[PaperBenchReproState], None],
) -> PaperBenchReproState:
    logger.info("work_packages - Planning work packages...")
    input_payload = build_work_package_planning_context(state)
    stage_debug: dict[str, Any] = {
        "attempts": 0,
        "fallback_used": False,
        "review_status": "accepted",
        "validation_errors": [],
    }

    def _normalize_work_package_output(result: WorkPackagePlanningOutput) -> WorkPackagePlanningOutput:
        units_by_id = {
            str(item.unit_id or "").strip(): item
            for item in (state.unit_extraction.units if state.unit_extraction else [])
            if str(item.unit_id or "").strip()
        }
        unit_ids = set(units_by_id)
        normalized_packages: list[dict[str, Any]] = []
        covered_unit_ids: set[str] = set()
        for item in result.work_packages:
            owned_unit_ids = [
                unit_id for unit_id in _dedupe_nonempty([str(unit_id) for unit_id in item.owned_unit_ids])
                if unit_id in unit_ids
            ]
            owned_units = [units_by_id[unit_id] for unit_id in owned_unit_ids if unit_id in units_by_id]
            unit_surfaces = _dedupe_nonempty(
                [
                    str(surface)
                    for unit in owned_units
                    for surface in list(getattr(unit, "implementation_surfaces", []) or [])
                ]
            )
            unit_obligations = _dedupe_nonempty(
                [
                    str(obligation)
                    for unit in owned_units
                    for obligation in _unit_positive_obligations(unit, limit=8)
                ]
            )
            unit_interfaces = _dedupe_nonempty(
                [
                    str(interface)
                    for unit in owned_units
                    for interface in list(getattr(unit, "runtime_interfaces", []) or [])
                ]
            )
            unit_artifacts = _dedupe_nonempty(
                [
                    str(artifact)
                    for unit in owned_units
                    for artifact in list(getattr(unit, "expected_artifacts", []) or [])
                ]
            )
            covered_unit_ids.update(owned_unit_ids)
            unit_inventories = _implementation_inventory_map(_derive_unit_inventory(owned_units))
            inventories = _merge_inventory_maps(_normalize_inventory_map(item.inventories), unit_inventories)
            if unit_surfaces:
                inventories["implementation_surface_inventory"] = _dedupe_nonempty(
                    list(inventories.get("implementation_surface_inventory", [])) + unit_surfaces
                )
            if unit_artifacts:
                inventories["artifact_inventory"] = _dedupe_nonempty(
                    list(inventories.get("artifact_inventory", [])) + unit_artifacts
                )
            normalized_packages.append(
                {
                    "work_package_id": str(item.work_package_id or "").strip(),
                    "goal": str(item.goal or "").strip(),
                    "hypothesis": str(
                        item.hypothesis
                        or (getattr(owned_units[0], "hypothesis", "") if owned_units else "")
                        or ""
                    ).strip(),
                    "decision_value": str(
                        item.decision_value
                        or (getattr(owned_units[0], "decision_value", "") if owned_units else "")
                        or ""
                    ).strip(),
                    "stop_rule_or_pruning_rationale": "Implementation scope: preserve owned-unit routes, artifacts, and bounded execution defaults.",
                    "owned_unit_ids": owned_unit_ids,
                    "tags": _dedupe_nonempty([str(tag) for tag in item.tags] + unit_surfaces),
                    "reference_ids": _dedupe_nonempty([str(ref_id) for ref_id in item.reference_ids]),
                    "depends_on": _dedupe_nonempty([str(dep) for dep in item.depends_on]),
                    "produces": _dedupe_nonempty([_normalize_repo_path(path) for path in item.produces]),
                    "interface_contract": _dedupe_nonempty([str(contract) for contract in item.interface_contract] + unit_interfaces),
                    "evidence_needs": _dedupe_nonempty([str(need) for need in item.evidence_needs]),
                    "inventories": inventories,
                    "scope_boundary": _scope_boundary_from_units(
                        owned_units,
                        existing=dict(getattr(item, "scope_boundary", {}) or {}),
                        fallback_focus=[
                            str(item.goal or "").strip(),
                            *unit_surfaces[:6],
                            *unit_interfaces[:6],
                            *unit_artifacts[:6],
                        ],
                    ),
                    "method_obligations": _dedupe_nonempty(
                        [str(obligation) for obligation in item.method_obligations]
                        + unit_obligations
                        + _augment_inventory_obligations(inventories)
                    ),
                }
            )
        uncovered_unit_ids = sorted(unit_ids - covered_unit_ids)
        normalized = WorkPackagePlanningOutput.model_validate(
            {
                "work_packages": normalized_packages,
                "coverage_summary": {
                    "total_units": len(unit_ids),
                    "covered_units": len(covered_unit_ids),
                    "uncovered_unit_ids": uncovered_unit_ids,
                },
                "planning_notes": _dedupe_nonempty(
                    list(result.planning_notes)
                    + (
                        [f"Uncovered units after normalization: {', '.join(uncovered_unit_ids[:8])}"]
                        if uncovered_unit_ids
                        else []
                    )
                ),
            }
        )
        return _repair_unowned_work_packages(normalized, units_by_id=units_by_id)

    def _work_package_planning_issues(result: WorkPackagePlanningOutput) -> list[str]:
        issues: list[str] = []
        work_package_ids = [str(item.work_package_id or "").strip() for item in result.work_packages]
        duplicate_ids = sorted({item for item in work_package_ids if item and work_package_ids.count(item) > 1})
        if duplicate_ids:
            issues.append("duplicate work_package_id values: " + ", ".join(duplicate_ids[:8]))
        if not result.work_packages:
            issues.append("work_package_planning returned no work packages")
        uncovered = list(result.coverage_summary.uncovered_unit_ids)
        if uncovered:
            issues.append("uncovered active units remain: " + ", ".join(uncovered[:8]))
        quality_gate = work_package_quality_report(
            units=list(state.unit_extraction.units if state.unit_extraction else []),
            work_packages=list(result.work_packages or []),
        )
        stage_debug["quality_gate"] = quality_gate
        for group in list(quality_gate.get("missing_groups", []) or []):
            issues.append(f"quality gate missing implementation group in work packages: {group}")
        for unit_id in list(quality_gate.get("unowned_units", []) or []):
            issues.append(f"quality gate found unowned unit: {unit_id}")
        for group, values in dict(quality_gate.get("evidence_contract_gaps", {}) or {}).items():
            issues.append(
                "quality gate missing paper-derived evidence in work packages: "
                + str(group)
                + "="
                + ",".join(str(value) for value in list(values or [])[:8])
            )
        for group, values in dict(quality_gate.get("implementation_obligation_gaps", {}) or {}).items():
            issues.append(
                "quality gate missing executable implementation obligation in work packages: "
                + str(group)
                + "="
                + ",".join(str(value) for value in list(values or [])[:8])
            )
        issues.extend(
            "quality gate " + issue
            for issue in claim_inventory_quality_issues(
                dict(quality_gate.get("claim_inventory_coverage", {}) or {}),
                stage_label="work packages",
            )
        )
        return issues

    review_budget = _stage_review_repair_budget(state)

    def _compute() -> WorkPackagePlanningOutput:
        if bool(getattr(get_workflow_config(), "deterministic_work_package_planning", True)):
            stage_debug["review_status"] = "deterministic_unit_plan"
            fallback = _fallback_work_package_output(state, stage_debug)
            issues = _work_package_planning_issues(fallback)
            if issues:
                stage_debug["validation_errors"] = _dedupe_nonempty(
                    list(stage_debug.get("validation_errors", []) or []) + issues
                )
                stage_debug["review_status"] = "deterministic_unit_plan_degraded_continue"
                _record_degraded_planning_issue(
                    state,
                    stage="work_package_planning",
                    code="work_package_quality_degraded_continue",
                    message="deterministic work package plan has quality issues; continuing with best available package ownership for repair",
                    reasons=issues,
                )
            return fallback
        max_attempts = max(1, 1 + review_budget)
        context_json = limit_json_for_prompt(input_payload)
        system, user = build_work_package_planning_prompt(
            context_json,
            language=state.input.language,
        )
        last_result: WorkPackagePlanningOutput | None = None
        for attempt_index in range(1, max_attempts + 1):
            stage_debug["attempts"] = attempt_index
            if attempt_index == 1 or last_result is None:
                try:
                    payload = invoke_json_stage("work_package_planning", "work_package_planning", system, user)
                except (TimeoutError, ValueError, ValidationError) as exc:
                    stage_debug["validation_errors"].append(str(exc))
                    stage_debug["review_status"] = "structured_output_fallback"
                    _record_degraded_planning_issue(
                        state,
                        stage="work_package_planning",
                        code="work_package_structured_output_degraded_continue",
                        message="work-package structured output failed; continuing with unit-derived package ownership",
                        reasons=[str(exc)],
                    )
                    return _fallback_work_package_output(state, stage_debug)
            else:
                repair_system, repair_user = build_work_package_planning_repair_prompt(
                    context_json=context_json,
                    previous_output_json=json.dumps(last_result.model_dump(mode="json"), ensure_ascii=False, indent=2),
                    validation_errors=list(stage_debug.get("validation_errors", []) or []),
                    language=state.input.language,
                )
                try:
                    payload = invoke_json_stage(
                        "work_package_planning_repair",
                        "work_package_planning",
                        repair_system,
                        repair_user,
                    )
                except (TimeoutError, ValueError, ValidationError) as exc:
                    stage_debug["validation_errors"].append(str(exc))
                    stage_debug["review_status"] = "structured_output_fallback"
                    _record_degraded_planning_issue(
                        state,
                        stage="work_package_planning",
                        code="work_package_structured_output_degraded_continue",
                        message="work-package repair structured output failed; continuing with unit-derived package ownership",
                        reasons=[str(exc)],
                    )
                    return _fallback_work_package_output(state, stage_debug)
            normalized = _normalize_work_package_output(WorkPackagePlanningOutput.model_validate(payload))
            if _should_fan_out_stage(
                state,
                stage_name="work_package_planning",
                work_package_count=len(list(normalized.work_packages)),
                prompt_payload=input_payload,
                reference_count=len(list(input_payload.get("prepared_reference_repositories", []) or [])),
                retry_count=max(0, attempt_index - 1),
            ):
                refined_outputs: list[WorkPackagePlanningOutput] = []
                for work_package in normalized.work_packages:
                    work_package_id = str(work_package.work_package_id or "").strip()
                    if not work_package_id:
                        continue
                    local_context = build_work_package_local_context(
                        state,
                        work_package_id,
                        base_context=input_payload,
                        current_work_package=work_package.model_dump(mode="json"),
                    )
                    local_system, local_user = build_work_package_planning_prompt(
                        limit_json_for_prompt(local_context),
                        language=state.input.language,
                    )
                    local_payload = invoke_json_stage(
                        f"work_package_planning_{work_package_id}",
                        "work_package_planning",
                        local_system,
                        local_user,
                    )
                    local_output = _normalize_work_package_output(WorkPackagePlanningOutput.model_validate(local_payload))
                    refined_outputs.append(
                        WorkPackagePlanningOutput.model_validate(
                            {
                                "work_packages": [
                                    item.model_dump(mode="json")
                                    for item in local_output.work_packages
                                    if str(item.work_package_id or "").strip() == work_package_id
                                ] or [work_package.model_dump(mode="json")],
                                "coverage_summary": local_output.coverage_summary.model_dump(mode="json"),
                                "planning_notes": list(local_output.planning_notes) + [f"fanout refine: {work_package_id}"],
                            }
                        )
                    )
                if refined_outputs:
                    normalized = _merge_work_package_results(normalized, refined_outputs)
                    normalized = _normalize_work_package_output(normalized)
            issues = _work_package_planning_issues(normalized)
            if not issues:
                return normalized
            stage_debug["validation_errors"].extend(issues)
            last_result = normalized
        stage_debug["review_status"] = "fallback"
        if last_result:
            last_issues = _work_package_planning_issues(last_result)
            if not last_issues:
                return last_result
            stage_debug["validation_errors"] = _dedupe_nonempty(
                list(stage_debug.get("validation_errors", []) or []) + last_issues
            )
        fallback = _fallback_work_package_output(state, stage_debug)
        fallback_issues = _work_package_planning_issues(fallback)
        if fallback_issues:
            stage_debug["validation_errors"] = _dedupe_nonempty(
                list(stage_debug.get("validation_errors", []) or []) + fallback_issues
            )
            _record_degraded_planning_issue(
                state,
                stage="work_package_planning",
                code="work_package_fallback_quality_degraded_continue",
                message="work package fallback still has quality issues after repair budget; continuing with best available package ownership",
                reasons=fallback_issues,
            )
        return fallback

    def _load() -> WorkPackagePlanningOutput:
        output_dir = get_output_dir(state)
        loaded = _normalize_work_package_output(
            WorkPackagePlanningOutput.model_validate(
                _read_plan_json_artifact(output_dir, "work_packages.json")
            )
        )
        issues = _work_package_planning_issues(loaded)
        if issues:
            _record_degraded_planning_issue(
                state,
                stage="work_package_planning",
                code="existing_work_package_quality_degraded_continue",
                message="existing work package artifact has quality issues; reusing it for best-effort continuation",
                reasons=issues,
            )
        return loaded

    def _write(result: WorkPackagePlanningOutput) -> None:
        result = _normalize_work_package_output(result)
        final_quality_gate = work_package_quality_report(
            units=list(state.unit_extraction.units if state.unit_extraction else []),
            work_packages=list(result.work_packages or []),
        )
        stage_debug["quality_gate"] = final_quality_gate
        final_issues = _work_package_planning_issues(result)
        if final_issues:
            stage_debug["validation_errors"] = _dedupe_nonempty(
                list(stage_debug.get("validation_errors", []) or []) + final_issues
            )
            _record_degraded_planning_issue(
                state,
                stage="work_package_planning",
                code="work_package_final_quality_degraded_continue",
                message="work package final quality gate has unresolved issues; continuing with best available package ownership",
                reasons=final_issues,
            )
        else:
            stage_debug["validation_errors"] = []
        write_stage_output(state, "work_packages.json", result)
        workflow_runtime.write_review_artifact(
            state,
            "work_package_planning",
            {
                "stage_name": "work_package_planning",
                "budget": review_budget,
                "attempts": int(stage_debug.get("attempts", 1) or 1),
                "fallback_used": bool(stage_debug.get("fallback_used", False)),
                "review_status": str(stage_debug.get("review_status", "accepted") or "accepted"),
                "validation_errors": list(stage_debug.get("validation_errors", []) or []),
                "quality_gate": final_quality_gate,
                "notes": list(result.planning_notes),
            },
            get_output_dir=get_output_dir,
            json_default=json_default,
        )

    state.work_package_planning = run_or_resume_stage(
        state,
        "work_package_planning",
        input_payload,
        _compute,
        _load,
        _write,
    )
    state.work_package_planning, autobind_notes = _autobind_work_package_references(state, state.work_package_planning)
    if autobind_notes:
        write_stage_output(state, "work_packages.json", state.work_package_planning)
    save_tracking_artifacts(state)
    return state

def _fallback_work_package_output(
    state: PaperBenchReproState,
    stage_debug: dict[str, Any],
) -> WorkPackagePlanningOutput:
        stage_debug["fallback_used"] = True
        units = [item for item in (state.unit_extraction.units if state.unit_extraction else []) if str(item.unit_id or "").strip()]
        fallback_work_packages = [
            {
                "work_package_id": f"wp_{index:03d}",
                "goal": _unit_scope_label(unit) or f"Implement unit {unit.unit_id}",
                "hypothesis": str(getattr(unit, "hypothesis", "") or "").strip(),
                "decision_value": str(getattr(unit, "decision_value", "") or "").strip(),
                "stop_rule_or_pruning_rationale": "Implementation scope: preserve the unit-owned routes, artifacts, and bounded execution defaults.",
                "owned_unit_ids": [unit.unit_id],
                "tags": _dedupe_nonempty([str(unit.type or "").strip()] + [str(item) for item in list(getattr(unit, "implementation_surfaces", []) or [])]),
                "reference_ids": [],
                "depends_on": [],
                "produces": [],
                "interface_contract": _dedupe_nonempty([str(item) for item in list(getattr(unit, "runtime_interfaces", []) or [])]),
                "evidence_needs": [f"cover unit {unit.unit_id}"],
                "inventories": _merge_inventory_maps(
                    _implementation_inventory_map(_derive_unit_inventory([unit])),
                    {
                        key: values
                        for key, values in {
                            "implementation_surface_inventory": _dedupe_nonempty(
                                [str(item) for item in list(getattr(unit, "implementation_surfaces", []) or [])]
                            ),
                            "artifact_inventory": _dedupe_nonempty(
                                [str(item) for item in list(getattr(unit, "expected_artifacts", []) or [])]
                            ),
                        }.items()
                        if values
                    },
                ),
                "scope_boundary": _scope_boundary_from_units(
                    [unit],
                    existing={},
                    fallback_focus=[
                        _unit_scope_label(unit),
                        *[str(item) for item in list(getattr(unit, "implementation_surfaces", []) or [])],
                        *[str(item) for item in list(getattr(unit, "runtime_interfaces", []) or [])],
                        *[str(item) for item in list(getattr(unit, "expected_artifacts", []) or [])],
                    ],
                ),
                "method_obligations": _dedupe_nonempty(
                    _unit_positive_obligations(unit, limit=8)
                    + _augment_inventory_obligations(_merge_inventory_maps(_implementation_inventory_map(_derive_unit_inventory([unit]))))
                ),
            }
            for index, unit in enumerate(units, start=1)
        ]
        return WorkPackagePlanningOutput.model_validate(
            {
                "work_packages": fallback_work_packages,
                "coverage_summary": {
                    "total_units": len(units),
                    "covered_units": len(units),
                    "uncovered_unit_ids": [],
                },
                "planning_notes": ["Fallback work-package plan synthesized to restore full unit coverage."],
        }
    )


def _preferred_owner_work_package_id_for_unit(
    unit: Any,
    existing_work_package_ids: set[str],
    package_payloads: dict[str, dict[str, Any]] | None = None,
) -> str:
    role = _work_package_role_for_unit(unit)
    scored = sorted(
        [
            (
                _existing_package_score_for_unit(
                    package_id,
                    unit,
                    role=role,
                    package_payload=dict(package_payloads or {}).get(package_id),
                ),
                package_id,
            )
            for package_id in existing_work_package_ids
            if str(package_id or "").strip()
        ],
        key=lambda item: (-item[0], item[1]),
    )
    if scored and scored[0][0] > 0:
        return scored[0][1]
    return _dynamic_work_package_id_for_unit(unit)


def _materialize_work_package_item(
    item: Any,
    owned_units: list[Any],
    *,
    units_by_id: dict[str, Any],
) -> dict[str, Any]:
    owned_unit_ids = _dedupe_nonempty(
        [str(unit_id) for unit_id in list(getattr(item, "owned_unit_ids", []) or [])]
        + [str(getattr(unit, "unit_id", "") or "").strip() for unit in owned_units]
    )
    owned_units = [units_by_id[unit_id] for unit_id in owned_unit_ids if unit_id in units_by_id]
    unit_surfaces = _dedupe_nonempty(
        [
            str(surface)
            for unit in owned_units
            for surface in list(getattr(unit, "implementation_surfaces", []) or [])
        ]
    )
    unit_obligations = _dedupe_nonempty(
        [
            str(obligation)
            for unit in owned_units
            for obligation in _unit_positive_obligations(unit, limit=8)
        ]
    )
    unit_interfaces = _dedupe_nonempty(
        [
            str(interface)
            for unit in owned_units
            for interface in list(getattr(unit, "runtime_interfaces", []) or [])
        ]
    )
    unit_artifacts = _dedupe_nonempty(
        [
            str(artifact)
            for unit in owned_units
            for artifact in list(getattr(unit, "expected_artifacts", []) or [])
        ]
    )
    unit_inventories = _implementation_inventory_map(_derive_unit_inventory(owned_units))
    inventories = _merge_inventory_maps(_normalize_inventory_map(_item_payload(item).get("inventories")), unit_inventories)
    if unit_surfaces:
        inventories["implementation_surface_inventory"] = _dedupe_nonempty(
            list(inventories.get("implementation_surface_inventory", [])) + unit_surfaces
        )
    if unit_artifacts:
        inventories["artifact_inventory"] = _dedupe_nonempty(
            list(inventories.get("artifact_inventory", [])) + unit_artifacts
        )
    owned_labels = _dedupe_nonempty(
        [
            _unit_scope_label(unit) or str(getattr(unit, "unit_id", "") or "").strip()
            for unit in owned_units
        ]
    )
    artifact_labels = _dedupe_nonempty(
        [
            str(artifact).strip()
            for unit in owned_units
            for artifact in list(getattr(unit, "expected_artifacts", []) or [])
            if str(artifact).strip()
        ]
    )
    goal = str(_item_payload(item).get("goal", "") or "").strip()
    if not goal:
        focus = ", ".join(artifact_labels[:4] or owned_labels[:4] or [str(_item_payload(item).get("work_package_id", "") or "work package")])
        goal = f"Cover {focus}"
    hypothesis = str(_item_payload(item).get("hypothesis", "") or "").strip()
    if not hypothesis:
        hypothesis = (
            str(next((getattr(unit, "hypothesis", "") for unit in owned_units if str(getattr(unit, "hypothesis", "") or "").strip()), "") or "").strip()
            or f"These units preserve the paper/addendum obligations for {goal.lower()}."
        )
    decision_value = str(_item_payload(item).get("decision_value", "") or "").strip()
    if not decision_value:
        decision_focus = ", ".join(artifact_labels[:4] or owned_labels[:4] or [str(_item_payload(item).get("work_package_id", "") or "work package")])
        decision_value = f"This package determines coverage for {decision_focus}."
    stop_rule = "Implementation scope: own the paper/addendum-stated table, figure, dataset, and contract surfaces, with repeated variants represented in config."
    return {
        "work_package_id": str(_item_payload(item).get("work_package_id", "") or "").strip(),
        "goal": goal,
        "hypothesis": hypothesis,
        "decision_value": decision_value,
        "stop_rule_or_pruning_rationale": stop_rule,
        "owned_unit_ids": owned_unit_ids,
        "tags": _dedupe_nonempty([str(tag) for tag in list(_item_payload(item).get("tags", []) or [])] + unit_surfaces),
        "reference_ids": _dedupe_nonempty([str(ref_id) for ref_id in list(_item_payload(item).get("reference_ids", []) or [])]),
        "depends_on": _dedupe_nonempty([str(dep) for dep in list(_item_payload(item).get("depends_on", []) or [])]),
        "produces": _dedupe_nonempty([_normalize_repo_path(path) for path in list(_item_payload(item).get("produces", []) or [])]),
        "interface_contract": _dedupe_nonempty(
            [str(contract) for contract in list(_item_payload(item).get("interface_contract", []) or [])] + unit_interfaces
        ),
        "evidence_needs": _dedupe_nonempty([str(need) for need in list(_item_payload(item).get("evidence_needs", []) or [])]),
        "inventories": inventories,
        "scope_boundary": _scope_boundary_from_units(
            owned_units,
            existing=dict(_item_payload(item).get("scope_boundary", {}) or {}),
            fallback_focus=[goal, *unit_surfaces[:6], *unit_interfaces[:6], *unit_artifacts[:6]],
        ),
        "method_obligations": _dedupe_nonempty(
            [str(obligation) for obligation in list(_item_payload(item).get("method_obligations", []) or [])]
            + unit_obligations
            + _augment_inventory_obligations(inventories)
        ),
    }


def _item_payload(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        return dict(item)
    if hasattr(item, "model_dump"):
        payload = item.model_dump(mode="json")
        if isinstance(payload, dict):
            return payload
    return {
        "work_package_id": str(getattr(item, "work_package_id", "") or "").strip(),
        "goal": str(getattr(item, "goal", "") or "").strip(),
        "hypothesis": str(getattr(item, "hypothesis", "") or "").strip(),
        "decision_value": str(getattr(item, "decision_value", "") or "").strip(),
        "stop_rule_or_pruning_rationale": "Implementation scope: preserve owned routes, artifacts, and bounded execution defaults.",
        "owned_unit_ids": list(getattr(item, "owned_unit_ids", []) or []),
        "tags": list(getattr(item, "tags", []) or []),
        "reference_ids": list(getattr(item, "reference_ids", []) or []),
        "depends_on": list(getattr(item, "depends_on", []) or []),
        "produces": list(getattr(item, "produces", []) or []),
        "interface_contract": list(getattr(item, "interface_contract", []) or []),
        "evidence_needs": list(getattr(item, "evidence_needs", []) or []),
        "inventories": dict(getattr(item, "inventories", {}) or {}),
        "scope_boundary": dict(getattr(item, "scope_boundary", {}) or {}),
        "method_obligations": list(getattr(item, "method_obligations", []) or []),
    }


def _repair_unowned_work_packages(
    result: WorkPackagePlanningOutput,
    *,
    units_by_id: dict[str, Any],
) -> WorkPackagePlanningOutput:
    active_units_by_id = {
        unit_id: item
        for unit_id, item in units_by_id.items()
        if str(getattr(item, "status", "active") or "active").strip().lower() == "active"
    }
    if not active_units_by_id:
        return result

    package_by_id: dict[str, dict[str, Any]] = {}
    package_order: list[str] = []
    for item in result.work_packages:
        payload = _item_payload(item)
        package_id = str(payload.get("work_package_id", "") or "").strip()
        if not package_id:
            continue
        package_order.append(package_id)
        package_by_id[package_id] = dict(payload)
    original_package_by_id = {package_id: dict(payload) for package_id, payload in package_by_id.items()}

    owned_unit_ids = {
        str(unit_id).strip()
        for package in package_by_id.values()
        for unit_id in list(package.get("owned_unit_ids", []) or [])
        if str(unit_id).strip()
    }
    active_unit_ids = set(active_units_by_id)
    uncovered_unit_ids = sorted(active_unit_ids - owned_unit_ids)
    if not uncovered_unit_ids:
        return result

    routed_notes: list[str] = []
    existing_package_ids = set(package_by_id)
    for unit_id in uncovered_unit_ids:
        unit = units_by_id[unit_id]
        owner_id = _preferred_owner_work_package_id_for_unit(
            unit,
            existing_package_ids,
            package_payloads=original_package_by_id,
        )
        if owner_id not in package_by_id:
            package_by_id[owner_id] = {
                "work_package_id": owner_id,
                "goal": "",
                "hypothesis": "",
                "decision_value": "",
                "stop_rule_or_pruning_rationale": "",
                "owned_unit_ids": [],
                "tags": [],
                "reference_ids": [],
                "depends_on": [],
                "produces": [],
                "interface_contract": [],
                "evidence_needs": [],
                "inventories": {},
                "method_obligations": [],
            }
            package_order.append(owner_id)
        package = package_by_id[owner_id]
        package["owned_unit_ids"] = _dedupe_nonempty(list(package.get("owned_unit_ids", []) or []) + [unit_id])
        package_by_id[owner_id] = _materialize_work_package_item(package, [units_by_id[item_id] for item_id in package["owned_unit_ids"] if item_id in units_by_id], units_by_id=units_by_id)
        routed_notes.append(f"routed orphan unit `{unit_id}` to work package `{owner_id}`")

    repaired_packages = [package_by_id[package_id] for package_id in package_order if package_id in package_by_id]
    repaired = WorkPackagePlanningOutput.model_validate(
        {
            "work_packages": repaired_packages,
            "coverage_summary": {
                "total_units": len(active_unit_ids),
                "covered_units": len(active_unit_ids),
                "uncovered_unit_ids": [],
            },
            "planning_notes": _dedupe_nonempty(
                list(result.planning_notes) + routed_notes + ["Deterministically routed uncovered active units into the most relevant work packages."]
            ),
        }
    )
    return repaired


def evidence_grounding_impl(
    state: PaperBenchReproState,
    *,
    build_evidence_bundles: Callable[[PaperBenchReproState, WorkPackagePlanningOutput], tuple[list[Any], list[Any]]],
    get_output_dir: Callable[[PaperBenchReproState], Any],
    write_stage_output: Callable[[PaperBenchReproState, str, object], None],
    run_or_resume_stage: Callable[..., Any],
    save_tracking_artifacts: Callable[[PaperBenchReproState], None],
) -> PaperBenchReproState:
    logger.info("evidence - Grounding work packages to prepared references...")
    if state.work_package_planning is None:
        raise ValueError("evidence grounding requires work package planning")
    input_payload = {
        "work_package_planning": state.work_package_planning.model_dump(mode="json"),
        "reference_repo_surveys": [item.model_dump(mode="json") for item in state.reference_repo_surveys],
    }

    def _compute() -> tuple[list[Any], list[Any]]:
        bundles, graph = build_evidence_bundles(state, state.work_package_planning)
        return bundles, graph

    def _load() -> tuple[list[Any], list[Any]]:
        output_dir = get_output_dir(state)
        return (
            _read_plan_json_artifact(output_dir, "evidence_bundles.json"),
            _read_plan_json_artifact(output_dir, "evidence_graph.json"),
        )

    def _write(result: tuple[list[Any], list[Any]]) -> None:
        bundles, graph = result
        write_stage_output(state, "evidence_bundles.json", bundles)
        write_stage_output(state, "evidence_graph.json", graph)
        diagnostics = state.temp_data.get("evidence_diagnostics")
        if diagnostics:
            write_stage_output(state, "evidence_diagnostics.json", diagnostics)

    evidence_bundles, evidence_graph = run_or_resume_stage(
        state,
        "package_evidence_grounding",
        input_payload,
        _compute,
        _load,
        _write,
    )
    state.evidence_bundles = [
        item if hasattr(item, "model_dump") else EvidenceBundleOutput.model_validate(item)
        for item in evidence_bundles
    ]
    state.evidence_graph = [
        item if hasattr(item, "model_dump") else EvidenceLinkOutput.model_validate(item)
        for item in evidence_graph
    ]
    save_tracking_artifacts(state)
    return state


def scope_alignment_impl(
    state: PaperBenchReproState,
    *,
    build_boundary_requirements_context: Callable[[PaperBenchReproState], dict[str, Any]],
    limit_json_for_prompt: Callable[[Any], str],
    invoke_json_stage: Callable[[str, str, str, str], dict[str, Any]],
    get_output_dir: Callable[[PaperBenchReproState], Any] | None = None,
    write_stage_output: Callable[[PaperBenchReproState, str, object], None],
) -> PaperBenchReproState:
    logger.info("scope - Building boundary requirements...")
    forked_boundary_payload = _load_forked_json_artifact(
        state,
        get_output_dir=get_output_dir,
        relative_path=CANONICAL_ARTIFACTS["boundary_requirements"],
    )
    if forked_boundary_payload is not None:
        forked_payload = dict(forked_boundary_payload)
        patched_requirements = []
        for item in list(forked_payload.get("boundary_requirements", []) or []):
            if not isinstance(item, dict):
                continue
            payload = dict(item)
            existing_source_ids = _dedupe_nonempty(
                [str(value) for value in list(payload.get("source_unit_ids", []) or [])]
            )
            payload["source_unit_ids"] = _dedupe_nonempty(
                [*existing_source_ids, *_boundary_requirement_unit_links(state, payload)]
            )
            patched_requirements.append(payload)
        forked_payload["boundary_requirements"] = patched_requirements
        if not patched_requirements:
            forked_payload = _synthesize_boundary_requirements_from_units(state)
        state.boundary_requirements = BoundaryRequirementsOutput.model_validate(forked_payload)
        write_stage_output(state, CANONICAL_ARTIFACTS["boundary_requirements"], state.boundary_requirements)
        return state
    system, user = build_boundary_requirements_prompt(
        limit_json_for_prompt(build_boundary_requirements_context(state)),
        language=state.input.language,
    )
    boundary_payload = invoke_json_stage("boundary_requirements", "boundary_requirements", system, user)
    patched_boundary_payload = dict(boundary_payload)
    patched_requirements = []
    for item in list(patched_boundary_payload.get("boundary_requirements", []) or []):
        if not isinstance(item, dict):
            continue
        payload = dict(item)
        existing_source_ids = _dedupe_nonempty(
            [str(value) for value in list(payload.get("source_unit_ids", []) or [])]
        )
        payload["source_unit_ids"] = _dedupe_nonempty(
            [*existing_source_ids, *_boundary_requirement_unit_links(state, payload)]
        )
        patched_requirements.append(payload)
    patched_boundary_payload["boundary_requirements"] = patched_requirements
    if not patched_requirements:
        patched_boundary_payload = _synthesize_boundary_requirements_from_units(state)
    state.boundary_requirements = BoundaryRequirementsOutput.model_validate(patched_boundary_payload)
    write_stage_output(state, CANONICAL_ARTIFACTS["boundary_requirements"], state.boundary_requirements)
    return state


def contract_planning_impl(
    state: PaperBenchReproState,
    *,
    build_reference_selection_context: Callable[[PaperBenchReproState, Any], dict[str, Any]],
    build_pipeline_plan_context: Callable[[PaperBenchReproState, Any, Any], dict[str, Any]],
    limit_json_for_prompt: Callable[[Any], str],
    invoke_json_stage: Callable[[str, str, str, str], dict[str, Any]],
    get_output_dir: Callable[[PaperBenchReproState], Any] | None = None,
    write_stage_output: Callable[[PaperBenchReproState, str, object], None],
    run_or_resume_stage: Callable[..., Any] | None = None,
    save_tracking_artifacts: Callable[[PaperBenchReproState], None] | None = None,
) -> PaperBenchReproState:
    logger.info("contract - Building reference selection and pipeline plan...")
    if get_output_dir is None:
        raise ValueError("contract planning requires get_output_dir")

    def _load_reference_selection() -> ReferenceSelectionOutput:
        return _merge_reference_selection_with_surveys(
            state,
            ReferenceSelectionOutput.model_validate(
                _normalize_reference_selection_payload(
                    _read_plan_json_artifact(get_output_dir(state), CANONICAL_ARTIFACTS["reference_selection"])
                )
            ),
        )

    def _write_reference_selection(result: ReferenceSelectionOutput) -> None:
        write_stage_output(state, CANONICAL_ARTIFACTS["reference_selection"], result)

    forked_reference_payload = _load_forked_json_artifact(
        state,
        get_output_dir=get_output_dir,
        relative_path=CANONICAL_ARTIFACTS["reference_selection"],
    )
    resume_reference_payload = (
        None
        if forked_reference_payload is not None
        else _load_in_place_resume_json_artifact(
            state,
            get_output_dir=get_output_dir,
            relative_path=CANONICAL_ARTIFACTS["reference_selection"],
        )
    )
    if forked_reference_payload is not None:
        state.reference_selection = _merge_reference_selection_with_surveys(
            state,
            ReferenceSelectionOutput.model_validate(_normalize_reference_selection_payload(forked_reference_payload)),
        )
        _write_reference_selection(state.reference_selection)
    elif resume_reference_payload is not None:
        state.reference_selection = _merge_reference_selection_with_surveys(
            state,
            ReferenceSelectionOutput.model_validate(_normalize_reference_selection_payload(resume_reference_payload)),
        )
        _write_reference_selection(state.reference_selection)
    elif _should_force_empty_reference_selection(state):
        state.reference_selection = _merge_reference_selection_with_surveys(
            state,
            ReferenceSelectionOutput.model_validate(
                {
                    "actionable_references": [],
                    "reference_relations": [],
                }
            ),
        )
        _write_reference_selection(state.reference_selection)
    else:
        reference_input_payload = build_reference_selection_context(state, state.boundary_requirements)

        def _compute_reference_selection() -> ReferenceSelectionOutput:
            system, user = build_reference_selection_prompt(
                limit_json_for_prompt(reference_input_payload),
                language=state.input.language,
            )
            reference_payload = invoke_json_stage("select_references", "select_references", system, user)
            return _merge_reference_selection_with_surveys(
                state,
                ReferenceSelectionOutput.model_validate(_normalize_reference_selection_payload(reference_payload)),
            )

        if run_or_resume_stage is not None:
            state.reference_selection = run_or_resume_stage(
                state,
                "reference_selection",
                reference_input_payload,
                _compute_reference_selection,
                _load_reference_selection,
                _write_reference_selection,
            )
        else:
            state.reference_selection = _compute_reference_selection()
            _write_reference_selection(state.reference_selection)

    forked_pipeline_payload = _load_forked_json_artifact(
        state,
        get_output_dir=get_output_dir,
        relative_path=CANONICAL_ARTIFACTS["pipeline_plan"],
    )
    resume_pipeline_payload = (
        None
        if forked_pipeline_payload is not None
        else _load_in_place_resume_json_artifact(
            state,
            get_output_dir=get_output_dir,
            relative_path=CANONICAL_ARTIFACTS["pipeline_plan"],
        )
    )

    def _load_pipeline_plan() -> PipelinePlanOutput:
        return _sanitize_pipeline_plan_with_surveys(
            state,
            PipelinePlanOutput.model_validate(
                _read_plan_json_artifact(get_output_dir(state), CANONICAL_ARTIFACTS["pipeline_plan"])
            ),
        )

    def _write_pipeline_plan(result: PipelinePlanOutput) -> None:
        sanitized = _sanitize_pipeline_plan_with_surveys(state, result)
        state.pipeline_plan = sanitized
        write_stage_output(state, CANONICAL_ARTIFACTS["pipeline_plan"], sanitized)

    if forked_pipeline_payload is not None:
        state.pipeline_plan = _sanitize_pipeline_plan_with_surveys(
            state,
            PipelinePlanOutput.model_validate(forked_pipeline_payload),
        )
        _write_pipeline_plan(state.pipeline_plan)
        if save_tracking_artifacts is not None:
            save_tracking_artifacts(state)
        return state
    if resume_pipeline_payload is not None:
        state.pipeline_plan = _sanitize_pipeline_plan_with_surveys(
            state,
            PipelinePlanOutput.model_validate(resume_pipeline_payload),
        )
        _write_pipeline_plan(state.pipeline_plan)
        if save_tracking_artifacts is not None:
            save_tracking_artifacts(state)
        return state

    pipeline_input_payload = build_pipeline_plan_context(
        state,
        state.boundary_requirements,
        state.reference_selection,
    )
    pipeline_debug: dict[str, Any] = {
        "fallback_used": False,
        "fallback_reason": "",
    }

    def _compute_pipeline_plan() -> PipelinePlanOutput:
        if bool(getattr(get_workflow_config(), "deterministic_pipeline_plan", False)):
            pipeline_debug["fallback_used"] = True
            pipeline_debug["fallback_reason"] = "deterministic_pipeline_plan_enabled"
            return _sanitize_pipeline_plan_with_surveys(
                state,
                _synthesize_pipeline_plan_output(
                    pipeline_input_payload,
                    reason=str(pipeline_debug["fallback_reason"]),
                ),
            )
        system, user = build_pipeline_plan_prompt(
            limit_json_for_prompt(pipeline_input_payload),
            language=state.input.language,
        )
        try:
            pipeline_plan_payload = invoke_json_stage("pipeline_plan", "pipeline_plan", system, user)
        except Exception as exc:
            pipeline_debug["fallback_used"] = True
            pipeline_debug["fallback_reason"] = f"{type(exc).__name__}: {exc}"
            return _sanitize_pipeline_plan_with_surveys(
                state,
                _synthesize_pipeline_plan_output(
                    pipeline_input_payload,
                    reason=str(pipeline_debug["fallback_reason"]),
                ),
            )
        pipeline_plan_payload = _normalize_pipeline_plan_payload(pipeline_plan_payload)
        return _sanitize_pipeline_plan_with_surveys(
            state,
            PipelinePlanOutput.model_validate(pipeline_plan_payload),
        )

    if run_or_resume_stage is not None:
        state.pipeline_plan = run_or_resume_stage(
            state,
            "pipeline_plan",
            pipeline_input_payload,
            _compute_pipeline_plan,
            _load_pipeline_plan,
            _write_pipeline_plan,
        )
    else:
        state.pipeline_plan = _compute_pipeline_plan()
        _write_pipeline_plan(state.pipeline_plan)
    if pipeline_debug.get("fallback_used"):
        workflow_runtime.write_review_artifact(
            state,
            "pipeline_plan",
            {
                "stage_name": "pipeline_plan",
                "review_status": "deterministic_fallback_continue",
                "fallback_used": True,
                "fallback_reason": str(pipeline_debug.get("fallback_reason", "") or ""),
                "coverage_summary": state.pipeline_plan.coverage_summary.model_dump(mode="json") if state.pipeline_plan else {},
                "notes": [
                    "Pipeline plan was synthesized from verified work packages and unit contracts.",
                    "This preserves hypothesis, decision-value, positive implementation scope, artifact, and reference anchors for downstream planning.",
                ],
            },
            get_output_dir=get_output_dir,
            json_default=lambda obj: str(obj),
        )
    pipeline_quality_failures = _pipeline_plan_quality_failures(
        state,
        fallback_used=bool(pipeline_debug.get("fallback_used")),
    )
    if pipeline_quality_failures:
        _mark_pipeline_plan_blocked(state, pipeline_quality_failures)
        workflow_runtime.write_review_artifact(
            state,
            "pipeline_plan",
            {
                "stage_name": "pipeline_plan",
                "review_status": "degraded_best_effort",
                "fallback_used": bool(pipeline_debug.get("fallback_used")),
                "fallback_reason": str(pipeline_debug.get("fallback_reason", "") or ""),
                "validation_errors": list(pipeline_quality_failures),
                "coverage_summary": state.pipeline_plan.coverage_summary.model_dump(mode="json") if state.pipeline_plan else {},
                "next_action": "continue_to_architecture_and_repair",
            },
            get_output_dir=get_output_dir,
            json_default=lambda obj: str(obj),
        )
    if save_tracking_artifacts is not None:
        save_tracking_artifacts(state)
    return state


def global_contract_impl(
    state: PaperBenchReproState,
    *,
    build_global_contract_context: Callable[[PaperBenchReproState, Any, Any, Any], dict[str, Any]],
    limit_json_for_prompt: Callable[[Any], str],
    invoke_json_stage: Callable[[str, str, str, str], dict[str, Any]],
    get_output_dir: Callable[[PaperBenchReproState], Any],
    write_stage_output: Callable[[PaperBenchReproState, str, object], None],
    run_or_resume_stage: Callable[..., Any],
    save_tracking_artifacts: Callable[[PaperBenchReproState], None],
) -> PaperBenchReproState:
    logger.info("global_contract - Freezing global contract...")
    input_payload = build_global_contract_context(
        state,
        state.boundary_requirements,
        state.reference_selection,
        state.pipeline_plan,
    )

    def _compute() -> GlobalContractOutput:
        if bool(getattr(get_workflow_config(), "deterministic_global_contract", False)):
            return _synthesize_global_contract_output(input_payload, GlobalContractOutput())
        system, user = build_global_contract_prompt(
            limit_json_for_prompt(input_payload),
            language=state.input.language,
        )
        try:
            payload = invoke_json_stage("global_contract", "global_contract", system, user)
            draft = GlobalContractOutput.model_validate(payload)
        except Exception:
            draft = GlobalContractOutput()
        return _synthesize_global_contract_output(input_payload, draft)

    def _load() -> GlobalContractOutput:
        output_dir = get_output_dir(state)
        return GlobalContractOutput.model_validate(
            _read_plan_json_artifact(output_dir, CANONICAL_ARTIFACTS["global_contract"])
        )

    def _write(result: GlobalContractOutput) -> None:
        write_stage_output(state, CANONICAL_ARTIFACTS["global_contract"], result)

    state.global_contract = run_or_resume_stage(
        state,
        "global_contract_synthesis",
        input_payload,
        _compute,
        _load,
        _write,
    )
    save_tracking_artifacts(state)
    return state


def architecture_planning_impl(
    state: PaperBenchReproState,
    *,
    build_architecture_context: Callable[[PaperBenchReproState, Any, Any, Any], dict[str, Any]],
    build_architecture_package_context: Callable[[PaperBenchReproState, str], dict[str, Any]],
    limit_json_for_prompt: Callable[[Any], str],
    invoke_json_stage: Callable[[str, str, str, str], dict[str, Any]],
    get_output_dir: Callable[[PaperBenchReproState], Any],
    json_default: Callable[[Any], Any],
    write_stage_output: Callable[[PaperBenchReproState, str, object], None],
    run_or_resume_stage: Callable[..., Any],
    save_tracking_artifacts: Callable[[PaperBenchReproState], None],
) -> PaperBenchReproState:
    logger.info("architecture - Building repository architecture...")
    input_payload = build_architecture_context(
        state,
        state.boundary_requirements,
        state.reference_selection,
        state.pipeline_plan,
    )
    stage_debug: dict[str, Any] = {
        "task_model": {},
        "ref_repo_model": {},
        "contract_targets": {},
        "task_view_draft": {},
        "ref_view_draft": {},
        "deterministic_synthesis_used": False,
        "attempts_used": 0,
        "used_review_repair": False,
        "review_status": "pending",
        "deviation_failures": [],
        "deviation_categories": [],
        "repair_actions": [],
        "schema_fix_rounds": 0,
        "validation_errors": [],
    }
    review_budget = 0

    def _compute() -> ArchitectureOutput:
        output_dir = get_output_dir(state)

        def _fallback_from_contract(reason: str) -> ArchitectureOutput:
            """Build a deterministic architecture when provider calls time out."""
            contract_paths = _dedupe_nonempty(
                [
                    path
                    for path in list(contract_targets.get("recommended_generated_files", []) or [])
                    if _normalize_repo_path(path) and not _looks_like_contract_output_path(path)
                ]
                + [
                    path
                    for paths in dict(contract_targets.get("required_work_package_owner_paths", {}) or {}).values()
                    for path in list(paths or [])
                    if _normalize_repo_path(path) and not _looks_like_contract_output_path(path)
                ]
                + [
                    path
                    for path in list(contract_targets.get("required_entry_files", []) or [])
                    if _normalize_repo_path(path) and not _looks_like_contract_output_path(path)
                ]
            )
            reference_hint_paths = _dedupe_nonempty(
                [
                    path
                    for path in _sanitize_architecture_source_paths(
                        [str(item) for item in list(ref_repo_model.get("ref_candidate_paths", []) or [])]
                    )
                    if path and not _looks_like_contract_output_path(path)
                ]
            )
            entry_files = _dedupe_nonempty(
                [path for path in list(contract_targets.get("required_entry_files", []) or []) if _normalize_repo_path(path)]
                + [path for path in contract_paths if _is_entrypoint_like_path(path)]
            )
            config_files = _dedupe_nonempty([path for path in contract_paths if _infer_file_kind(path) == "config"])
            if not config_files:
                config_files = ["configs/default.yaml"]
            candidate_paths = _dedupe_nonempty([*entry_files, *config_files, *contract_paths]) or ["main.py"]
            reference_path_owners = dict(ref_repo_model.get("reference_path_owners", {}) or {})
            fallback_blueprints = [
                {
                    "path": path,
                    "purpose": _infer_file_purpose(path),
                    "kind": _normalize_architecture_blueprint_kind(None, path),
                    "related_node_ids": [],
                    "based_on_references": _dedupe_nonempty(
                        [str(item) for item in list(reference_path_owners.get(path, []) or [])]
                    ),
                    "implementation_strategy": "adapted" if reference_path_owners.get(path) else "new",
                }
                for path in candidate_paths
            ]
            fallback_seed = ArchitectureOutput.model_validate(
                {
                    "target_stack": ["python"],
                    "target_file_tree": candidate_paths,
                    "file_blueprints": fallback_blueprints,
                    "dependency_graph": [],
                    "stable_interfaces": [],
                    "execution_entrypoints": entry_files,
                    "config_surfaces": config_files,
                    "package_layout": {},
                    "dependency_rules": ["Generated files should depend on package-local stable interfaces."],
                    "protocol_stages": list(getattr(task_model, "runnable_flow", []) or []),
                    "result_targets": list(contract_targets.get("required_result_artifacts", []) or []),
                    "architecture_reference_ids": _dedupe_nonempty(
                        list(contract_targets.get("preferred_reference_ids", []) or [])
                        + list(ref_repo_model.get("surveyed_repos", []) or [])
                    ),
                    "unresolved_review_failures": [
                        f"architecture provider fallback used after {reason}; review generated file boundaries carefully"
                    ],
                    "rationale": f"Deterministic architecture fallback from global contract, work-package owner paths, and task model after {reason}. Reference hints available: {len(reference_hint_paths)}.",
                }
            )
            return _repair_architecture_deterministically(
                fallback_seed,
                contract_targets=contract_targets,
                task_model=task_model,
                plan_nodes=list(state.pipeline_plan.plan_nodes or []),
            )

        def _fallback_architecture(
            raw_payload: dict[str, Any] | None,
            *,
            reason: str,
        ) -> ArchitectureOutput:
            raw_payload = dict(raw_payload or {})
            candidate_paths = _dedupe_nonempty(
                [
                    _normalize_repo_path(str(path))
                    for path in list(raw_payload.get("target_file_tree", []) or [])
                    if _normalize_repo_path(str(path))
                ]
                + [
                    _normalize_repo_path(str(item.get("path", "") or ""))
                    for item in list(raw_payload.get("file_blueprints", []) or [])
                    if isinstance(item, dict) and _normalize_repo_path(str(item.get("path", "") or ""))
                ]
                + [path for path in list(contract_targets.get("required_entry_files", []) or []) if _normalize_repo_path(path)]
            )
            if not candidate_paths:
                candidate_paths = ["main.py"]

            raw_blueprints = {
                _normalize_repo_path(str(item.get("path", "") or "")): dict(item)
                for item in list(raw_payload.get("file_blueprints", []) or [])
                if isinstance(item, dict) and _normalize_repo_path(str(item.get("path", "") or ""))
            }

            fallback_payload = {
                "target_stack": _dedupe_nonempty([str(item) for item in list(raw_payload.get("target_stack", []) or [])] or ["python"]),
                "target_file_tree": candidate_paths,
                "file_blueprints": [
                    {
                        "path": path,
                        "purpose": str(raw_blueprints.get(path, {}).get("purpose", "") or "").strip() or _infer_file_purpose(path),
                        "kind": _normalize_architecture_blueprint_kind(raw_blueprints.get(path, {}).get("kind"), path),
                        "related_node_ids": _dedupe_nonempty(
                            [str(item) for item in list(raw_blueprints.get(path, {}).get("related_node_ids", []) or [])]
                        ),
                        "based_on_references": _dedupe_nonempty(
                            [str(item) for item in list(raw_blueprints.get(path, {}).get("based_on_references", []) or [])]
                        ),
                        "implementation_strategy": _normalize_architecture_implementation_strategy(
                            raw_blueprints.get(path, {}).get("implementation_strategy")
                        ),
                    }
                    for path in candidate_paths
                ],
                "dependency_graph": [
                    {
                        "source_path": _normalize_repo_path(str(edge.get("source_path", "") or "")),
                        "target_path": _normalize_repo_path(str(edge.get("target_path", "") or "")),
                        "dependency_type": str(edge.get("dependency_type", "imports") or "imports").strip() or "imports",
                    }
                    for edge in list(raw_payload.get("dependency_graph", []) or [])
                    if isinstance(edge, dict)
                    and _normalize_repo_path(str(edge.get("source_path", "") or ""))
                    and _normalize_repo_path(str(edge.get("target_path", "") or ""))
                ],
                "stable_interfaces": _dedupe_nonempty(
                    [_normalize_repo_path(str(path)) for path in list(raw_payload.get("stable_interfaces", []) or []) if _normalize_repo_path(str(path))]
                ),
                "execution_entrypoints": _dedupe_nonempty(
                    [_normalize_repo_path(str(path)) for path in list(raw_payload.get("execution_entrypoints", []) or []) if _normalize_repo_path(str(path))]
                    + [path for path in list(contract_targets.get("required_entry_files", []) or []) if _normalize_repo_path(path)]
                ),
                "config_surfaces": _dedupe_nonempty(
                    [_normalize_repo_path(str(path)) for path in list(raw_payload.get("config_surfaces", []) or []) if _normalize_repo_path(str(path))]
                ),
                "package_layout": {
                    str(work_package_id): _dedupe_nonempty(
                        [_normalize_repo_path(str(path)) for path in list(paths or []) if _normalize_repo_path(str(path))]
                    )
                    for work_package_id, paths in dict(raw_payload.get("package_layout", {}) or {}).items()
                    if str(work_package_id or "").strip()
                },
                "dependency_rules": _dedupe_nonempty([str(item) for item in list(raw_payload.get("dependency_rules", []) or [])]),
                "protocol_stages": _dedupe_nonempty([str(item) for item in list(raw_payload.get("protocol_stages", []) or [])]),
                "result_targets": _dedupe_nonempty(
                    [_normalize_repo_path(str(path)) for path in list(raw_payload.get("result_targets", []) or []) if _normalize_repo_path(str(path))]
                    + [path for path in list(contract_targets.get("required_result_artifacts", []) or []) if _normalize_repo_path(path)]
                ),
                "architecture_reference_ids": _dedupe_nonempty(
                    [str(item) for item in list(raw_payload.get("architecture_reference_ids", []) or [])]
                    + [str(item) for item in list(contract_targets.get("preferred_reference_ids", []) or [])]
                ),
                "rationale": (
                    str(raw_payload.get("rationale", "") or "").strip()
                    or f"Fallback architecture synthesized after schema validation failure: {reason}"
                ),
            }
            return _normalize_architecture_output(ArchitectureOutput.model_validate(fallback_payload))

        def _deterministic_architecture_synthesis(
            *,
            reason: str,
            task_view_architecture: ArchitectureOutput,
            ref_view_architecture: ArchitectureOutput,
        ) -> ArchitectureOutput:
            merged = _merge_architecture_outputs(task_view_architecture, [ref_view_architecture])
            merged = _repair_architecture_deterministically(
                merged,
                contract_targets=contract_targets,
                task_model=task_model,
                plan_nodes=list(state.pipeline_plan.plan_nodes or []),
                task_view_architecture=task_view_architecture,
                ref_view_architecture=ref_view_architecture,
            )
            stage_debug["deterministic_synthesis_used"] = True
            stage_debug["used_review_repair"] = True
            stage_debug["review_status"] = "deterministic_synthesis_continue"
            return merged.model_copy(
                update={
                    "unresolved_review_failures": _dedupe_nonempty(
                        list(merged.unresolved_review_failures)
                        + [f"deterministic architecture synthesis used after {reason}"]
                    ),
                    "rationale": "\n".join(
                        item
                        for item in [
                            str(merged.rationale or "").strip(),
                            f"Deterministically merged task-view and reference-view architecture after {reason}.",
                        ]
                        if item
                    ),
                }
            )

        def _invoke_architecture_payload(
            *,
            stage_name: str,
            system: str,
            user: str,
            schema_fix_context: dict[str, Any] | None = None,
        ) -> ArchitectureOutput:
            del schema_fix_context
            try:
                payload = invoke_json_stage(stage_name, "architecture", system, user)
            except Exception as exc:
                stage_debug["validation_errors"] = _dedupe_nonempty(
                    list(stage_debug.get("validation_errors", []) or [])
                    + [f"{stage_name}: {type(exc).__name__}: {exc}"]
                )
                stage_debug["used_review_repair"] = True
                stage_debug["review_status"] = "provider_fallback_continue"
                return _fallback_from_contract(f"{stage_name} provider error: {type(exc).__name__}")
            try:
                return _normalize_architecture_output(
                    ArchitectureOutput.model_validate(_coerce_architecture_payload(payload))
                )
            except ValidationError as exc:
                rendered_error = str(exc)
                stage_debug["validation_errors"] = _dedupe_nonempty(
                    list(stage_debug.get("validation_errors", []) or []) + [f"{stage_name}: {rendered_error}"]
                )
                _write_architecture_debug_artifact(
                    output_dir,
                    f"architecture.{stage_name}.schema_error.json",
                    {
                        "stage_name": stage_name,
                        "validation_error": rendered_error,
                        "payload": payload,
                    },
                    json_default=json_default,
                )

            fallback = _fallback_architecture(
                payload,
                reason=str(stage_debug.get("validation_errors", [])[-1])
                if stage_debug.get("validation_errors")
                else f"{stage_name} returned no valid architecture payload",
            )
            fallback = fallback.model_copy(
                update={
                    "unresolved_review_failures": _dedupe_nonempty(
                        list(fallback.unresolved_review_failures)
                        + [f"{stage_name} schema validation exhausted; continued with fallback architecture"]
                    )
                }
            )
            stage_debug["review_status"] = "schema_fix_exhausted_continue"
            stage_debug["used_review_repair"] = True
            return fallback

        ref_repo_model = _build_ref_repo_model(input_payload)
        task_model = _synthesize_architecture_task_model(input_payload, ref_repo_model)
        contract_targets = _derive_architecture_contract_targets(input_payload, task_model)
        known_work_package_ids = set(contract_targets.get("required_work_package_ids", []) or [])
        stage_debug["task_model"] = task_model.model_dump(mode="json")
        stage_debug["ref_repo_model"] = dict(ref_repo_model)
        stage_debug["contract_targets"] = dict(contract_targets)
        architecture_context = {
            **input_payload,
            "task_model": stage_debug["task_model"],
            "ref_repo_model": stage_debug["ref_repo_model"],
            "contract_targets": stage_debug["contract_targets"],
        }
        architecture_system, architecture_user = build_architecture_prompt(
            limit_json_for_prompt(architecture_context),
            language=state.input.language,
        )
        architecture = _invoke_architecture_payload(
            stage_name="architecture",
            system=architecture_system,
            user=architecture_user,
        )
        architecture = _filter_architecture_to_known_work_packages(
            architecture,
            known_work_package_ids,
        )
        stage_debug["task_view_draft"] = architecture.model_dump(mode="json")
        stage_debug["ref_view_draft"] = {}
        _write_architecture_debug_artifact(
            output_dir,
            "architecture.draft.llm.json",
            architecture.model_dump(mode="json"),
            json_default=json_default,
        )
        contract_targets = _architecture_remap_contract_targets_for_llm_tree(architecture, contract_targets)
        stage_debug["contract_targets"] = dict(contract_targets)
        _write_architecture_debug_artifact(
            output_dir,
            "architecture.task_model.json",
            {
                "task_model": stage_debug["task_model"],
                "ref_repo_model": stage_debug["ref_repo_model"],
                "contract_targets": stage_debug["contract_targets"],
            },
            json_default=json_default,
        )
        if bool(getattr(get_workflow_config(), "deterministic_architecture_synthesis", True)):
            stage_debug["deterministic_synthesis_used"] = True
            stage_debug["used_review_repair"] = True
            stage_debug["review_status"] = "llm_architecture_with_deterministic_closure"
            deterministic_seed = _fallback_from_contract("deterministic_architecture_synthesis_enabled")
            deterministic_seed = _filter_architecture_to_known_work_packages(
                deterministic_seed,
                known_work_package_ids,
            )
            _write_architecture_debug_artifact(
                output_dir,
                "architecture.draft.deterministic.json",
                deterministic_seed.model_dump(mode="json"),
                json_default=json_default,
            )
            architecture = _merge_architecture_outputs(architecture, [deterministic_seed])
            architecture = _normalize_architecture_output(architecture)
        architecture = _repair_architecture_deterministically(
            architecture,
            contract_targets=contract_targets,
            task_model=task_model,
            plan_nodes=list(state.pipeline_plan.plan_nodes or []),
        )
        architecture = _close_architecture_package_layout(
            architecture,
            contract_targets=contract_targets,
            task_model=task_model,
        )
        architecture = _architecture_prune_paths_without_semantic_signal(
            architecture,
            contract_targets=contract_targets,
            input_payload=input_payload,
        )
        architecture = _close_architecture_package_layout(
            architecture,
            contract_targets=contract_targets,
            task_model=task_model,
        )
        architecture = _filter_architecture_to_known_work_packages(
            architecture,
            known_work_package_ids,
        )
        _write_architecture_debug_artifact(
            output_dir,
            "architecture.draft.initial.json",
            architecture.model_dump(mode="json"),
            json_default=json_default,
        )

        stage_debug["attempts_used"] = 1
        deviation_report = _architecture_deviation_report(architecture, contract_targets, ref_repo_model)
        last_failures = list(deviation_report.get("failures", []) or [])
        stage_debug["deviation_categories"] = list(deviation_report.get("failure_categories", []) or [])
        stage_debug["repair_actions"] = list(deviation_report.get("repair_actions", []) or [])
        stage_debug["deviation_failures"] = list(last_failures)
        _write_architecture_debug_artifact(
            output_dir,
            "architecture.deviation.round1.json",
            deviation_report,
            json_default=json_default,
        )
        _write_architecture_debug_artifact(
            output_dir,
            "architecture.fix_plan.round1.json",
            {
                "mode": "single_pass_deterministic_closure",
                "failure_categories": list(deviation_report.get("failure_categories", []) or []),
                "failure_groups": dict(deviation_report.get("failure_groups", {}) or {}),
                "repair_actions": list(deviation_report.get("repair_actions", []) or []),
            },
            json_default=json_default,
        )
        if not last_failures:
            stage_debug["review_status"] = "passed"
            return architecture.model_copy(update={"unresolved_review_failures": []})
        stage_debug["review_status"] = "deterministic_closure_continue"
        return architecture.model_copy(
            update={
                "unresolved_review_failures": _dedupe_nonempty(
                    list(last_failures)
                    + ["architecture used single-pass planning; unresolved deviations are recorded without LLM repair"]
                )
            }
        )

    def _load() -> ArchitectureOutput:
        output_dir = get_output_dir(state)
        return ArchitectureOutput.model_validate(
            _read_plan_json_artifact(output_dir, CANONICAL_ARTIFACTS["architecture"])
        )

    def _write(result: ArchitectureOutput) -> None:
        write_stage_output(state, CANONICAL_ARTIFACTS["architecture"], result)
        state.temp_data["architecture_task_model"] = dict(stage_debug.get("task_model", {}) or {})
        state.temp_data["architecture_ref_repo_model"] = dict(stage_debug.get("ref_repo_model", {}) or {})
        state.temp_data["architecture_contract_targets"] = dict(stage_debug.get("contract_targets", {}) or {})
        workflow_runtime.write_review_artifact(
            state,
            "architecture_planning",
            {
                "stage_name": "architecture_planning",
                "budget": review_budget,
                "attempts": int(stage_debug.get("attempts_used", 1) or 1),
                "schema_fix_rounds": int(stage_debug.get("schema_fix_rounds", 0) or 0),
                "review_status": str(stage_debug.get("review_status", "accepted") or "accepted"),
                "validation_errors": list(stage_debug.get("validation_errors", []) or []),
                "failure_categories": list(stage_debug.get("deviation_categories", []) or []),
                "repair_actions": list(stage_debug.get("repair_actions", []) or []),
                "notes": [
                    *([result.rationale] if result.rationale else []),
                    *[
                        f"review_failure: {item}"
                        for item in list(stage_debug.get("deviation_failures", []) or [])[:6]
                    ],
                ],
            },
            get_output_dir=get_output_dir,
            json_default=json_default,
        )

    state.architecture = run_or_resume_stage(
        state,
        "architecture_planning",
        input_payload,
        _compute,
        _load,
        _write,
    )
    unresolved_architecture_failures = [
        str(item).strip()
        for item in list(getattr(state.architecture, "unresolved_review_failures", []) or [])
        if str(item).strip()
    ]
    if unresolved_architecture_failures:
        _record_degraded_planning_issue(
            state,
            stage="architecture_planning",
            code="architecture_review_degraded_continue",
            message="architecture planning has unresolved review failures; continuing so file planning/generate/repair can handle them",
            reasons=unresolved_architecture_failures,
        )
    save_tracking_artifacts(state)
    return state


def package_file_planning_impl(
    state: PaperBenchReproState,
    *,
    project_file_plans_from_architecture: Callable[[Any, Any], Any],
    close_package_file_plans: Callable[[Any, Any, Any], Any],
    validate_file_plans: Callable[[Any, Any], None],
    order_file_plans_for_execution_closure: Callable[[Any, Any], Any],
    build_package_file_planning_context: Callable[[PaperBenchReproState, Any, Any, Any], dict[str, Any]],
    build_package_file_planning_local_context: Callable[[PaperBenchReproState, Any, Any, Any, str], dict[str, Any]],
    limit_json_for_prompt: Callable[[Any], str],
    invoke_json_stage: Callable[[str, str, str, str], dict[str, Any]],
    get_output_dir: Callable[[PaperBenchReproState], Any],
    json_default: Callable[[Any], Any],
    write_stage_output: Callable[[PaperBenchReproState, str, object], None],
    run_or_resume_stage: Callable[..., Any],
    save_tracking_artifacts: Callable[[PaperBenchReproState], None],
    derive_steps_from_file_plans: Callable[[PackageFilePlanningOutput], list[Any]],
    render_pipeline_plan_markdown: Callable[[PaperBenchReproState], str],
) -> PaperBenchReproState:
    logger.info("file_planning - Freezing file-level task plan...")
    unresolved_architecture_failures = [
        str(item).strip()
        for item in list(getattr(state.architecture, "unresolved_review_failures", []) or [])
        if str(item).strip()
    ]
    if unresolved_architecture_failures:
        _record_degraded_planning_issue(
            state,
            stage="package_file_planning",
            code="unresolved_architecture_degraded_continue",
            message="package file planning is continuing from architecture review issues so generate/repair can address them",
            reasons=unresolved_architecture_failures,
        )
    projected_file_plans = project_file_plans_from_architecture(state.architecture, state.pipeline_plan)
    input_payload = build_package_file_planning_context(
        state,
        state.architecture,
        state.pipeline_plan,
        projected_file_plans,
    )
    stage_debug: dict[str, Any] = {
        "attempts": 0,
        "schema_fix_rounds": 0,
        "review_status": "accepted",
        "validation_errors": [],
    }
    review_budget = _stage_review_repair_budget(state)

    def _file_plan_contract_issues(result: PackageFilePlanningOutput) -> list[str]:
        issues: list[str] = []
        if not result.file_plans:
            issues.append("package_file_planning returned no file plans")
            return issues
        if state.global_contract is not None:
            required_artifacts = {
                str(path or "").strip()
                for target in state.global_contract.result_targets
                for path in target.artifact_paths
                if str(path or "").strip()
            }
            planned_artifacts = {
                str(path or "").strip()
                for item in result.file_plans
                for path in item.writes_artifacts
                if str(path or "").strip()
            }
            missing_artifacts = sorted(required_artifacts - planned_artifacts)
            if missing_artifacts:
                issues.append("declared artifact targets are not covered by file plans: " + ", ".join(missing_artifacts[:8]))
        entrypoints = {str(path or "").strip() for path in list(state.architecture.execution_entrypoints or []) if str(path or "").strip()}
        if entrypoints:
            planned_paths = {item.target_file for item in result.file_plans}
            missing_entrypoints = sorted(entrypoints - planned_paths)
            if missing_entrypoints:
                issues.append("execution entrypoints are missing from file plans: " + ", ".join(missing_entrypoints[:8]))
        quality_gate = file_plan_quality_report(
            units=list(state.unit_extraction.units if state.unit_extraction else []),
            work_packages=list(state.work_package_planning.work_packages if state.work_package_planning else []),
            file_plans=list(result.file_plans or []),
        )
        stage_debug["quality_gate"] = quality_gate
        issues.extend(file_plan_quality_issues(quality_gate))
        return issues

    def _compute() -> PackageFilePlanningOutput:
        if bool(getattr(get_workflow_config(), "deterministic_package_file_planning", True)):
            stage_debug["review_status"] = "deterministic_projection"
            projected_closed = close_package_file_plans(state.architecture, state.pipeline_plan, projected_file_plans)
            projected_closed = _repair_package_file_artifact_coverage(
                projected_closed,
                global_contract=state.global_contract,
            )
            projected_closed = _augment_package_file_plans_with_formula_contract(
                state,
                projected_closed,
                input_payload,
            )
            validate_file_plans(state.architecture, projected_closed)
            ordered = order_file_plans_for_execution_closure(state.architecture, projected_closed)
            ordered = _augment_package_file_plans_with_formula_contract(state, ordered, input_payload)
            issues = _file_plan_contract_issues(ordered)
            if not issues:
                return ordered.model_copy(update={"unresolved_review_failures": []})
            stage_debug["validation_errors"] = _dedupe_nonempty(
                list(stage_debug.get("validation_errors", []) or []) + issues
            )
            stage_debug["review_status"] = "deterministic_projection_degraded_continue"
            _record_degraded_planning_issue(
                state,
                stage="package_file_planning",
                code="deterministic_file_plan_quality_degraded_continue",
                message="deterministic package file plan has unresolved contract issues; continuing for generate/repair",
                reasons=issues,
            )
            return ordered.model_copy(update={"unresolved_review_failures": issues})
        max_attempts = max(1, 1 + review_budget)
        context_json = limit_json_for_prompt(input_payload)
        system, user = build_package_file_planning_prompt(
            context_json,
            language=state.input.language,
        )
        payload: dict[str, Any] | None = None
        last_error: Exception | None = None
        last_planned: PackageFilePlanningOutput | None = None
        last_closed: PackageFilePlanningOutput | None = None
        previous_issue_signature = ""
        no_progress_rounds = 0
        for attempt_index in range(1, max_attempts + 1):
            stage_debug["attempts"] = attempt_index
            if attempt_index == 1:
                try:
                    payload = invoke_json_stage("package_file_planning", "package_file_planning", system, user)
                except Exception as exc:
                    last_error = exc
                    stage_debug.setdefault("validation_errors", []).append(f"{type(exc).__name__}: {exc}")
                    stage_debug["review_status"] = "llm_error_continue"
                    break
            else:
                if last_planned is not None:
                    repair_system, repair_user = build_package_file_planning_repair_prompt(
                        context_json=context_json,
                        previous_output_json=json.dumps(last_planned.model_dump(mode="json"), ensure_ascii=False, indent=2),
                        validation_errors=list(stage_debug.get("validation_errors", []) or []),
                        language=state.input.language,
                    )
                    try:
                        payload = invoke_json_stage(
                            "package_file_planning_repair",
                            "package_file_planning",
                            repair_system,
                            repair_user,
                        )
                    except Exception as exc:
                        last_error = exc
                        stage_debug.setdefault("validation_errors", []).append(f"{type(exc).__name__}: {exc}")
                        stage_debug["review_status"] = "llm_error_continue"
                        break
                else:
                    stage_debug["schema_fix_rounds"] = attempt_index - 1
                    fix_system, fix_user = build_package_file_planning_schema_fix_prompt(
                        context_json=context_json,
                        invalid_payload_json=json.dumps(payload or {}, ensure_ascii=False, indent=2, default=json_default),
                        validation_error=str(last_error or ""),
                        language=state.input.language,
                    )
                    try:
                        payload = invoke_json_stage(
                            "package_file_planning_schema_fix",
                            "package_file_planning",
                            fix_system,
                            fix_user,
                        )
                    except Exception as exc:
                        last_error = exc
                        stage_debug.setdefault("validation_errors", []).append(f"{type(exc).__name__}: {exc}")
                        stage_debug["review_status"] = "llm_error_continue"
                        break
            try:
                planned = PackageFilePlanningOutput.model_validate(payload)
                last_planned = planned
                closed = close_package_file_plans(state.architecture, state.pipeline_plan, planned)
                closed = _repair_package_file_artifact_coverage(
                    closed,
                    global_contract=state.global_contract,
                )
                closed = _augment_package_file_plans_with_formula_contract(state, closed, input_payload)
                if bool(getattr(get_workflow_config(), "package_file_planning_fanout_enabled", False)) and _should_fan_out_stage(
                    state,
                    stage_name="package_file_planning",
                    work_package_count=len(_fanout_work_package_ids(state)),
                    prompt_payload=input_payload,
                    reference_count=len(
                        {
                            ref_id
                            for item in _work_package_items(state)
                            for ref_id in list(dict(item).get("reference_ids", []) or [])
                            if str(ref_id).strip()
                        }
                    ),
                    retry_count=max(0, attempt_index - 1),
                ):
                    refined_outputs: list[PackageFilePlanningOutput] = []
                    for work_package_id in _fanout_work_package_ids(state):
                        local_context = build_package_file_planning_local_context(
                            state,
                            state.architecture,
                            state.pipeline_plan,
                            projected_file_plans,
                            work_package_id,
                            base_context=input_payload,
                        )
                        local_system, local_user = build_package_file_planning_prompt(
                            limit_json_for_prompt(local_context),
                            language=state.input.language,
                        )
                        local_payload = invoke_json_stage(
                            f"package_file_planning_{work_package_id}",
                            "package_file_planning",
                            local_system,
                            local_user,
                        )
                        local_planned = PackageFilePlanningOutput.model_validate(local_payload)
                        local_closed = close_package_file_plans(state.architecture, state.pipeline_plan, local_planned)
                        local_closed = _repair_package_file_artifact_coverage(
                            local_closed,
                            global_contract=state.global_contract,
                        )
                        local_closed = _augment_package_file_plans_with_formula_contract(
                            state,
                            local_closed,
                            local_context,
                        )
                        refined_outputs.append(
                            local_closed.model_copy(
                                update={
                                    "file_plans": [
                                        item
                                        for item in local_closed.file_plans
                                        if str(item.work_package_id or "").strip() == work_package_id
                                    ],
                                    "planning_notes": list(local_closed.planning_notes) + [f"fanout refine: {work_package_id}"],
                                }
                            )
                        )
                    if refined_outputs:
                        closed = _merge_package_file_plan_outputs(closed, refined_outputs)
                last_closed = closed
                validate_file_plans(state.architecture, closed)
                ordered = order_file_plans_for_execution_closure(state.architecture, closed)
                ordered = _augment_package_file_plans_with_formula_contract(state, ordered, input_payload)
                issues = _file_plan_contract_issues(ordered)
                if not issues:
                    return ordered.model_copy(update={"unresolved_review_failures": []})
                issue_signature = " | ".join(sorted(issues))
                if issue_signature == previous_issue_signature:
                    no_progress_rounds += 1
                else:
                    no_progress_rounds = 0
                previous_issue_signature = issue_signature
                last_error = RuntimeError("; ".join(issues))
                stage_debug.setdefault("validation_errors", []).extend(issues)
                if no_progress_rounds >= 1:
                    stage_debug["review_status"] = "no_progress_continue"
                    break
                if attempt_index >= max_attempts:
                    break
            except Exception as exc:
                last_error = exc
                stage_debug.setdefault("validation_errors", []).append(str(exc))
                if attempt_index >= max_attempts:
                    break
        unresolved_failures = _dedupe_nonempty(list(stage_debug.get("validation_errors", []) or []))
        if stage_debug.get("review_status") != "llm_error_continue":
            stage_debug["review_status"] = "budget_exhausted_continue"
        _record_degraded_planning_issue(
            state,
            stage="package_file_planning",
            code="package_file_planning_budget_exhausted_continue",
            message="package/file planning repair budget exhausted; continuing with the best available closed file plan for generate/repair",
            reasons=unresolved_failures,
        )
        continuation_notes = [
            "Package/file planning review budget exhausted; continuing with the last closed file plan and delegating unresolved plan issues to repair."
        ]
        if stage_debug.get("review_status") == "llm_error_continue":
            continuation_notes = [
                "Package/file planning LLM call failed; continuing with deterministic architecture projection."
            ]
        if last_closed is not None:
            ordered_last = order_file_plans_for_execution_closure(state.architecture, last_closed)
            ordered_last = _augment_package_file_plans_with_formula_contract(state, ordered_last, input_payload)
            if not ordered_last.file_plans and projected_file_plans.file_plans:
                projected_closed = close_package_file_plans(state.architecture, state.pipeline_plan, projected_file_plans)
                ordered_last = order_file_plans_for_execution_closure(state.architecture, projected_closed)
                ordered_last = _augment_package_file_plans_with_formula_contract(state, ordered_last, input_payload)
            final_issues = _file_plan_contract_issues(ordered_last)
            if final_issues:
                stage_debug["validation_errors"] = _dedupe_nonempty(unresolved_failures + final_issues)
                _record_degraded_planning_issue(
                    state,
                    stage="package_file_planning",
                    code="package_file_plan_after_budget_degraded_continue",
                    message="last closed package file plan still has quality issues; continuing with it for generate/repair",
                    reasons=final_issues,
                )
            return ordered_last.model_copy(
                update={
                    "planning_notes": _dedupe_nonempty(list(ordered_last.planning_notes) + continuation_notes),
                    "unresolved_review_failures": unresolved_failures,
                }
            )
        if last_planned is not None:
            fallback_closed = close_package_file_plans(state.architecture, state.pipeline_plan, last_planned)
            ordered_fallback = order_file_plans_for_execution_closure(state.architecture, fallback_closed)
            ordered_fallback = _augment_package_file_plans_with_formula_contract(state, ordered_fallback, input_payload)
            if not ordered_fallback.file_plans and projected_file_plans.file_plans:
                projected_closed = close_package_file_plans(state.architecture, state.pipeline_plan, projected_file_plans)
                ordered_fallback = order_file_plans_for_execution_closure(state.architecture, projected_closed)
                ordered_fallback = _augment_package_file_plans_with_formula_contract(state, ordered_fallback, input_payload)
            final_issues = _file_plan_contract_issues(ordered_fallback)
            if final_issues:
                stage_debug["validation_errors"] = _dedupe_nonempty(unresolved_failures + final_issues)
                _record_degraded_planning_issue(
                    state,
                    stage="package_file_planning",
                    code="package_file_plan_fallback_degraded_continue",
                    message="fallback package file plan still has quality issues; continuing with it for generate/repair",
                    reasons=final_issues,
                )
            return ordered_fallback.model_copy(
                update={
                    "planning_notes": _dedupe_nonempty(list(ordered_fallback.planning_notes) + continuation_notes),
                    "unresolved_review_failures": unresolved_failures,
                }
            )
        projected_closed = close_package_file_plans(state.architecture, state.pipeline_plan, projected_file_plans)
        ordered_projected = order_file_plans_for_execution_closure(state.architecture, projected_closed)
        ordered_projected = _augment_package_file_plans_with_formula_contract(state, ordered_projected, input_payload)
        final_issues = _file_plan_contract_issues(ordered_projected)
        if final_issues:
            stage_debug["validation_errors"] = _dedupe_nonempty(unresolved_failures + final_issues)
            _record_degraded_planning_issue(
                state,
                stage="package_file_planning",
                code="package_file_plan_projected_degraded_continue",
                message="projected package file plan still has quality issues; continuing with it for generate/repair",
                reasons=final_issues,
            )
        return ordered_projected.model_copy(
            update={
                "planning_notes": _dedupe_nonempty(list(ordered_projected.planning_notes) + continuation_notes),
                "unresolved_review_failures": unresolved_failures,
            }
        )

    def _load() -> PackageFilePlanningOutput:
        output_dir = get_output_dir(state)
        loaded = PackageFilePlanningOutput.model_validate(
            _read_plan_json_artifact(output_dir, CANONICAL_ARTIFACTS["package_file_planning"])
        )
        loaded = _augment_package_file_plans_with_formula_contract(state, loaded, input_payload)
        issues = _file_plan_contract_issues(loaded)
        unresolved = list(getattr(loaded, "unresolved_review_failures", []) or [])
        if unresolved:
            issues.extend("existing file plan has unresolved review failure: " + str(item) for item in unresolved[:12])
        if issues:
            _record_degraded_planning_issue(
                state,
                stage="package_file_planning",
                code="existing_package_file_plan_degraded_continue",
                message="existing package file planning artifact has unresolved quality issues; reusing it for best-effort continuation",
                reasons=issues,
            )
        return loaded

    def _write(result: PackageFilePlanningOutput) -> None:
        result = _augment_package_file_plans_with_formula_contract(state, result, input_payload)
        final_issues = _file_plan_contract_issues(result)
        unresolved = list(getattr(result, "unresolved_review_failures", []) or [])
        if unresolved:
            final_issues.extend("file plan has unresolved review failure: " + str(item) for item in unresolved[:12])
        if final_issues:
            stage_debug["validation_errors"] = _dedupe_nonempty(
                list(stage_debug.get("validation_errors", []) or []) + final_issues
            )
            _record_degraded_planning_issue(
                state,
                stage="package_file_planning",
                code="package_file_plan_final_degraded_continue",
                message="package file planning final quality gate has unresolved issues; continuing with best available file plan",
                reasons=final_issues,
            )
            result = result.model_copy(
                update={
                    "unresolved_review_failures": _dedupe_nonempty(
                        list(getattr(result, "unresolved_review_failures", []) or []) + final_issues
                    )
                }
            )
        else:
            stage_debug["validation_errors"] = []
        write_stage_output(state, CANONICAL_ARTIFACTS["package_file_planning"], result)
        workflow_runtime.write_review_artifact(
            state,
            "package_file_planning",
            {
                "stage_name": "package_file_planning",
                "budget": review_budget,
                "attempts": int(stage_debug.get("attempts", 1) or 1),
                "schema_fix_rounds": int(stage_debug.get("schema_fix_rounds", 0) or 0),
                "review_status": str(stage_debug.get("review_status", "accepted") or "accepted"),
                "validation_errors": list(stage_debug.get("validation_errors", []) or []),
                "quality_gate": dict(stage_debug.get("quality_gate", {}) or {}),
                "notes": list(result.planning_notes),
            },
            get_output_dir=get_output_dir,
            json_default=json_default,
        )

    file_planning_output = run_or_resume_stage(
        state,
        "package_file_planning",
        input_payload,
        _compute,
        _load,
        _write,
    )
    state.package_file_planning_output = file_planning_output
    state.temp_data["steps"] = derive_steps_from_file_plans(file_planning_output)
    state.plan = render_pipeline_plan_markdown(state)
    save_tracking_artifacts(state)
    return state
