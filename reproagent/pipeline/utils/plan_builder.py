"""Plan normalization, projection, and rendering helpers for reproagent."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from reproagent.pipeline.schemas import (
    ArchitectureOutput,
    PaperBenchReproState,
    PackageFilePlanningOutput,
    PipelinePlanOutput,
    RepoFilePlan,
)

from .dataset_manager import _get_dataset_preparation, _get_resource_manifest
from .evidence_contracts import flatten_evidence_contract, infer_evidence_contract, object_values
from .contract_sanitizer import sanitize_contract_list, sanitize_contract_text, sanitize_scope_boundary

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


def _positive_addendum_line(line: str) -> str:
    """Project addendum lines into positive implementation duties."""
    text = " ".join(str(line or "").split())
    if not text:
        return ""
    lowered = text.lower()
    pieces: list[str] = []
    if "inference throughput" in lowered or "sampled processed per second" in lowered or "samples processed per second" in lowered:
        pieces.append("Implement inference throughput as samples processed per second.")
    if "train time" in lowered or "time-to-accuracy" in lowered or "97%" in lowered:
        pieces.append("Implement train-time/TTA as time to reach 97% of finetuning-baseline dev/test performance.")
    if "relative accuracy" in lowered or "sst2" in lowered or "mnli" in lowered:
        pieces.append("Implement relative accuracy as the SST2/MNLI average relative to the finetuned-baseline average.")
    if "mask tuning" in lowered or "retraining-free-pruning" in lowered:
        pieces.append("Implement the mask_tuning baseline with LoRA-tuned model adaptation and config-visible defaults.")
    if "cofi" in lowered or "l0" in lowered:
        pieces.append("Implement the CoFi baseline with LoRA and L0 module controls plus default hyperparameters.")
    if "default hyperparameters" in lowered:
        pieces.append("Expose referenced baseline default hyperparameters through configuration.")
    if "outlier-aware salience" in lowered or "0.85" in lowered or "0.15" in lowered:
        pieces.append("Implement outlier-aware block-salience EMA with 0.85 previous and 0.15 current weighting.")
    if "pruning_start_step" in lowered or "pruning_end_step" in lowered or "global_step" in lowered or "\\mu" in lowered or "$\\mu$" in lowered:
        pieces.append("Implement the mu pruning schedule from pruning_start_step to pruning_end_step with bounded config defaults.")
    if "teacher-student" in lowered or "layer-mapping" in lowered:
        pieces.append("Recompute teacher-student layer mapping every training step.")
    if "classification" in lowered or "glue" in lowered or "distillation loss" in lowered:
        pieces.append("Implement GLUE classification distillation loss with named prediction and layer-loss terms.")
    if "squad" in lowered or "cnn/dm" in lowered or "cnn" in lowered and "dm" in lowered:
        pieces.append("Expose SQuAD and CNN/DM distillation-loss interfaces named by the paper/addendum.")
    if "gpu memory" in lowered or "max gpu memory" in lowered or "max_memory_allocated" in lowered:
        pieces.append("Measure max GPU memory through torch.cuda.max_memory_allocated() or an equivalent backend hook.")
    if "tau" in lowered or "\\tau" in lowered or "$\\tau$" in lowered:
        pieces.append("Set tau to 4 in the equation-7/CoFi-related calculation path.")
    if pieces:
        return " ".join(_dedupe(pieces))
    if any(token in lowered for token in ("should be implemented", "is measured as", "computed as", "calculated as", "use `", "use https://")):
        return text
    return ""


def _read_optional_text(path: str) -> str:
    normalized = str(path or "").strip()
    if not normalized:
        return ""
    try:
        candidate = Path(normalized)
        if candidate.exists() and candidate.is_file():
            return candidate.read_text(encoding="utf-8")
    except Exception:
        return ""
    return ""


def _source_text_for_state(state: PaperBenchReproState | None) -> str:
    """Return PaperBench source text only: paper.md, addendum.md, and prepared chunks."""
    if state is None:
        return ""
    parts: list[str] = []
    paper_text = str(getattr(state.input, "paper_text", "") or "").strip()
    if paper_text:
        parts.append(paper_text)
    loaded_paper = _read_optional_text(str(getattr(state.input, "paper_path", "") or ""))
    if loaded_paper:
        parts.append(loaded_paper)
    experiment_design = dict(getattr(state.input, "experiment_design", {}) or {})
    paperbench_payload = dict(experiment_design.get("paperbench", {}) or {})
    addendum_text = str(paperbench_payload.get("addendum_text", "") or "").strip()
    if addendum_text:
        parts.append(addendum_text)
    loaded_addendum = _read_optional_text(str(paperbench_payload.get("addendum_path", "") or ""))
    if loaded_addendum:
        parts.append(loaded_addendum)
    if not parts:
        for chunk in list(getattr(state, "paper_chunks", []) or []):
            text = str(getattr(chunk, "text", "") or "").strip()
            if text:
                parts.append(text)
    return "\n\n".join(_dedupe(parts))


def _addendum_text_for_state(state: PaperBenchReproState | None) -> str:
    if state is None:
        return ""
    experiment_design = dict(getattr(state.input, "experiment_design", {}) or {})
    paperbench_payload = dict(experiment_design.get("paperbench", {}) or {})
    addendum_text = str(paperbench_payload.get("addendum_text", "") or "").strip()
    if addendum_text:
        return addendum_text
    return _read_optional_text(str(paperbench_payload.get("addendum_path", "") or ""))


def _source_mentions_term(term: str, source_text: str) -> bool:
    """Conservative source-presence check used to filter LLM-added inventory noise."""
    value = str(term or "").strip()
    if not value:
        return False
    lowered = str(source_text or "").lower()
    if not lowered:
        return True
    variants = _dedupe([
        value,
        value.replace("_", " "),
        value.replace("_", "-"),
        value.replace("_", ""),
        value.replace("-", " "),
        value.replace("-", "_"),
    ])
    for variant in variants:
        needle = str(variant or "").strip().lower()
        if not needle:
            continue
        if needle in lowered:
            return True
        pattern = r"(?<![a-z0-9])" + re.escape(needle).replace(r"\ ", r"[\s_-]+") + r"(?![a-z0-9])"
        if re.search(pattern, lowered):
            return True
    return False


def _paper_artifact_path(name: str) -> str:
    """Map paper-visible table/figure labels to stable repo artifact paths."""
    label = " ".join(str(name or "").strip().split())
    match = re.search(r"\btable\s+(\d+[A-Za-z]?)\b", label, flags=re.IGNORECASE)
    if match:
        return f"results/tables/table_{match.group(1).lower()}.csv"
    match = re.search(r"\b(?:figure|fig\.?)\s+(\d+[A-Za-z]?)\b", label, flags=re.IGNORECASE)
    if match:
        return f"results/figures/figure_{match.group(1).lower()}.png"
    normalized = label.strip().lower()
    if normalized in {"metrics_json", "metrics.json"}:
        return "results/metrics.json"
    if normalized == "result_table":
        return "results/tables/experiment_results.csv"
    if normalized == "result_figure":
        return "results/figures/experiment_results.png"
    if normalized == "config":
        return "results/config_resolved.json"
    if normalized in {"log", "logs", "training_log", "training log"}:
        return "results/training_log.json"
    if normalized == "predictions":
        return "results/predictions.jsonl"
    return ""


def _paper_artifact_paths(names: list[str]) -> list[str]:
    return _dedupe([path for path in (_paper_artifact_path(name) for name in list(names or [])) if path])

def _normalize_repo_path(path: str) -> str:
    normalized = str(path or "").strip().replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    while "//" in normalized:
        normalized = normalized.replace("//", "/")
    return normalized.strip("/")


def _looks_like_repo_relative_file_path(path: str) -> bool:
    normalized = _normalize_repo_path(path)
    if not normalized or normalized.endswith("/"):
        return False
    if normalized.startswith(("/", "-", "$")) or "://" in normalized:
        return False
    if any(char.isspace() for char in normalized):
        return False
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._/-")
    if any(char not in allowed for char in normalized):
        return False
    basename = normalized.rsplit("/", 1)[-1]
    return "." in basename


def _work_package_by_id(state: PaperBenchReproState | None) -> dict[str, object]:
    if state is None or state.work_package_planning is None:
        return {}
    return {
        str(item.work_package_id or "").strip(): item
        for item in list(state.work_package_planning.work_packages or [])
        if str(item.work_package_id or "").strip()
    }


_WORK_PACKAGE_SLUG_GENERIC_TOKENS = {
    "wp",
    "paper",
    "repo",
    "repository",
    "reproduction",
    "implementation",
    "impl",
    "surface",
    "contract",
    "protocol",
    "package",
    "module",
}

_WORK_PACKAGE_PROJECTION_GENERIC_TOKENS = _WORK_PACKAGE_SLUG_GENERIC_TOKENS.union(
    {
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
        "dataset",
        "datasets",
        "data",
        "loader",
        "preprocess",
        "model",
        "models",
        "method",
        "methods",
        "algorithm",
        "algorithms",
        "baseline",
        "baselines",
        "ablation",
        "train",
        "training",
        "evaluation",
        "evaluate",
        "metric",
        "metrics",
        "artifact",
        "artifacts",
        "report",
        "reporting",
        "plot",
        "figure",
        "protocol",
        "matrix",
        "validation",
    }
)

_GENERIC_SURFACE_TERMS = {
    "config",
    "configs",
    "configuration",
    "entry",
    "entrypoint",
    "main",
    "cli",
    "command",
    "package",
    "module",
    "protocol",
    "registry",
    "schema",
    "schemas",
    "spec",
    "specs",
    "surface",
    "surface contract",
}

_WORK_PACKAGE_PROJECTION_SYMBOL_SUPPORT_TOKENS = {
    "config",
    "configs",
    "configuration",
    "schema",
    "schemas",
    "spec",
    "specs",
    "result",
    "results",
    "layout",
    "settings",
    "types",
}


def _ordered_slug_tokens(*values: str) -> list[str]:
    tokens: list[str] = []
    seen: set[str] = set()
    for value in values:
        for token in re.findall(r"[A-Za-z][A-Za-z0-9]*", str(value or "").lower()):
            if token in seen or token.isdigit():
                continue
            seen.add(token)
            tokens.append(token)
    return tokens


def _work_package_slug(work_package_id: str, work_package: object) -> str:
    package_slug = _symbol_slug(str(work_package_id or "")) or "work_package"
    package_slug = re.sub(r"^(wp|paper)_+", "", package_slug).strip("_") or package_slug
    package_tokens = [
        token
        for token in _ordered_slug_tokens(package_slug)
        if token not in _WORK_PACKAGE_SLUG_GENERIC_TOKENS
    ]
    if package_tokens:
        return "_".join(package_tokens[:3])
    goal_tokens = [
        token
        for token in _ordered_slug_tokens(
            str(getattr(work_package, "goal", "") or ""),
            " ".join([str(item) for item in list(getattr(work_package, "tags", []) or [])[:4]]),
        )
        if token not in _WORK_PACKAGE_SLUG_GENERIC_TOKENS
        and token not in _WORK_PACKAGE_PROJECTION_GENERIC_TOKENS
    ]
    if goal_tokens:
        return "_".join(goal_tokens[:3])
    return package_slug or "work_package"


def _shared_role_paths_for_work_package(
    work_package_id: str,
    work_package: object,
    *,
    existing_paths: list[str] | None = None,
) -> list[str]:
    text = _evidence_model_text(work_package)
    roles = _text_roles(text)
    slug = _work_package_slug(work_package_id, work_package)
    existing = [_normalize_repo_path(path) for path in list(existing_paths or []) if _normalize_repo_path(path)]
    existing_by_role = [
        path
        for path in existing
        if _file_roles(path).intersection(roles)
        and "doc" not in _file_roles(path)
        and "packaging" not in _file_roles(path)
    ]
    if existing_by_role:
        return existing_by_role[:3]
    paths: list[str] = []
    if roles.intersection({"artifact", "reporting", "evaluation"}) and not roles.intersection({"method", "training"}):
        paths.append("src/reporting/reports.py")
    if roles.intersection({"training", "experiment", "baseline"}):
        paths.append("src/experiments/runner.py")
    if roles.intersection({"method", "model", "agent", "explainer", "refinement"}):
        paths.append("src/methods/core.py")
    if "data" in roles and not roles.intersection({"method", "training", "evaluation"}):
        paths.append("src/data/datasets.py")
    if "environment" in roles and not roles.intersection({"method", "training"}):
        paths.append("src/envs/tasks.py")
    if "config" in roles and not roles.intersection({"method", "training", "evaluation", "artifact", "experiment"}):
        paths.append("configs/default.yaml")
    if "entrypoint" in roles:
        paths.append("main.py")
    if "packaging" in roles and not roles.intersection({"method", "training", "evaluation", "artifact", "experiment"}):
        paths.append("pyproject.toml")
    if "doc" in roles and not roles.intersection({"method", "training", "evaluation", "artifact", "experiment"}):
        paths.append("README.md")
    if "test" in roles and not roles.intersection({"method", "training", "evaluation", "artifact"}):
        paths.append(f"tests/test_{slug}.py")
    if not paths:
        paths.append("src/experiments/runner.py")
    return _dedupe(paths)


def _semantic_work_package_source_path(work_package_id: str, work_package: object) -> str:
    return _shared_role_paths_for_work_package(work_package_id, work_package)[0]


def _work_package_projection_paths_and_owners(
    state: PaperBenchReproState | None,
) -> tuple[list[str], dict[str, str]]:
    """Recover concrete producer files directly from work-package contracts."""
    if state is None:
        return [], {}
    paths: list[str] = []
    owners: dict[str, str] = {}
    existing_paths = []
    if state.architecture is not None:
        existing_paths.extend(_normalize_repo_path(path) for path in list(state.architecture.target_file_tree or []))
    if state.canonical_ir is not None:
        existing_paths.extend(_canonical_registered_paths(state))
    for work_package_id, work_package in _work_package_by_id(state).items():
        text = _evidence_model_text(work_package)
        roles = _text_roles(text)
        package_paths = [
            _normalize_repo_path(path)
            for path in list(getattr(work_package, "produces", []) or [])
            if _looks_like_repo_relative_file_path(_normalize_repo_path(path))
            and not _normalize_repo_path(path).lower().startswith(("results/", "outputs/", "artifacts/"))
        ]
        if not package_paths:
            package_paths = _shared_role_paths_for_work_package(
                work_package_id,
                work_package,
                existing_paths=existing_paths,
            )
        if roles.intersection({"entrypoint", "experiment"}):
            package_paths.append("main.py")
        if "config" in roles:
            package_paths.append("configs/default.yaml")
        if "doc" in roles or "readme" in text.lower() or "documentation" in text.lower():
            package_paths.append("README.md")
        if "packaging" in roles or "requirements.txt" in text.lower() or "pyproject.toml" in text.lower():
            package_paths.append("pyproject.toml")
        if "test" in roles or "smoke test" in text.lower() or "contract test" in text.lower():
            package_paths.append(f"tests/test_{_work_package_slug(work_package_id, work_package) or 'contracts'}.py")
        for path in _dedupe(package_paths):
            paths.append(path)
            owners.setdefault(path, work_package_id)
    return _dedupe(paths), owners


def _canonical_registered_paths(state: PaperBenchReproState) -> list[str]:
    if state.canonical_ir is None:
        raise ValueError("canonical IR is required for canonical-only file plan projection")
    return [
        _normalize_repo_path(path)
        for path in list(state.canonical_ir.validation_index.get("registered_paths", []) or [])
        if _normalize_repo_path(path)
    ]


def _projection_paths(
    architecture: ArchitectureOutput,
    state: PaperBenchReproState | None,
) -> list[str]:
    architecture_paths = [
        _normalize_repo_path(path)
        for path in list(architecture.target_file_tree or [])
        if _normalize_repo_path(path)
    ]
    package_paths, _owners = _work_package_projection_paths_and_owners(state)
    shard_paths = _contract_shard_paths(state)
    if state is not None and state.canonical_ir is not None:
        canonical_paths = _canonical_registered_paths(state)
        if canonical_paths:
            return _dedupe(list(architecture_paths) + list(canonical_paths) + package_paths + shard_paths)
    return _dedupe(architecture_paths + package_paths + shard_paths)


def _projection_owner_map(
    architecture: ArchitectureOutput,
    state: PaperBenchReproState | None,
) -> dict[str, str]:
    shard_owner_map = _contract_shard_owner_map(state)
    _package_paths, package_owner_map = _work_package_projection_paths_and_owners(state)
    if state is not None and state.canonical_ir is not None and _canonical_registered_paths(state):
        return {
            **package_owner_map,
            **_canonical_owner_map(state),
            **_architecture_package_map(architecture),
            **shard_owner_map,
        }
    return {**package_owner_map, **_architecture_package_map(architecture), **shard_owner_map}


def _canonical_owner_map(state: PaperBenchReproState) -> dict[str, str]:
    if state.canonical_ir is None:
        raise ValueError("canonical IR is required for canonical-only owner projection")
    owner_map: dict[str, str] = {}
    for item in state.canonical_ir.file_nodes:
        path = _normalize_repo_path(item.canonical_path)
        owner = str(item.owner_work_package_id or "").strip()
        if path and owner:
            owner_map[path] = owner
    return owner_map


def _dependency_map(architecture: ArchitectureOutput) -> dict[str, list[str]]:
    dependency_map: dict[str, list[str]] = {}
    for edge in architecture.dependency_graph:
        dependency_map.setdefault(edge.source_path, [])
        if edge.target_path not in dependency_map[edge.source_path]:
            dependency_map[edge.source_path].append(edge.target_path)
    return dependency_map


def _architecture_package_map(architecture: ArchitectureOutput) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for work_package_id, paths in dict(architecture.package_layout or {}).items():
        for path in list(paths or []):
            normalized = str(path or "").strip()
            if normalized and normalized not in mapping:
                mapping[normalized] = str(work_package_id or "").strip()
    return mapping


_CONTRACT_SHARD_OWNER_ROLES: dict[str, set[str]] = {
    "src/experiment_registry.py": {"experiment", "evaluation", "artifact", "config", "test"},
    "src/environment_registry.py": {"environment", "data", "config", "test"},
    "src/dataset_registry.py": {"data", "environment", "evaluation", "config", "test"},
    "src/method_registry.py": {
        "method",
        "model",
        "agent",
        "training",
        "baseline",
        "explainer",
        "refinement",
        "config",
        "test",
    },
    "src/sweep_registry.py": {"config", "experiment", "training", "method", "baseline", "refinement", "test"},
    "src/artifact_contract.py": {"artifact", "evaluation", "experiment", "test"},
    "src/trend_assertions.py": {"evaluation", "artifact", "experiment", "test"},
}

_CONTRACT_SHARD_ROLE_HINTS: dict[str, tuple[str, ...]] = {
    "src/experiment_registry.py": ("experiment", "protocol", "matrix", "evidence obligation"),
    "src/environment_registry.py": ("environment", "task", "simulator", "benchmark"),
    "src/dataset_registry.py": ("dataset", "benchmark", "loader"),
    "src/method_registry.py": ("method", "baseline", "variant", "policy", "model", "adapter"),
    "src/sweep_registry.py": ("parameter", "sweep", "hyperparameter"),
    "src/artifact_contract.py": ("artifact", "result", "table", "figure", "writer", "checkpoint"),
    "src/trend_assertions.py": ("trend", "endpoint", "outperformance", "insensitive", "assertion"),
}


def _contract_shard_paths(state: PaperBenchReproState | None) -> list[str]:
    """Add small support modules when a paper has broad paper-derived evidence contracts."""
    if os.getenv("PAPERBENCH_REPRO_ENABLE_LEGACY_CONTRACT_SHARDS", "").strip().lower() not in {"1", "true", "yes"}:
        return []
    if state is None or state.unit_extraction is None:
        return []
    contract = _state_evidence_contract(state)
    paths: list[str] = []
    if _contract_terms(contract, "named_experiments"):
        paths.append("src/experiment_registry.py")
    if _contract_terms(contract, "environments"):
        paths.append("src/environment_registry.py")
    if _contract_terms(contract, "datasets"):
        paths.append("src/dataset_registry.py")
    if _contract_terms(contract, "methods"):
        paths.append("src/method_registry.py")
    if list(contract.get("parameter_sweeps", []) or []):
        paths.append("src/sweep_registry.py")
    if _contract_terms(contract, "artifacts") or _contract_terms(contract, "metrics"):
        paths.append("src/artifact_contract.py")
    if _contract_terms(contract, "trend_obligations"):
        paths.append("src/trend_assertions.py")
    return _dedupe(paths)


def _work_package_score_for_roles(state: PaperBenchReproState | None, roles: set[str]) -> dict[str, int]:
    if state is None or state.work_package_planning is None:
        return {}
    scores: dict[str, int] = {}
    for package in list(state.work_package_planning.work_packages or []):
        work_package_id = str(package.work_package_id or "").strip()
        if not work_package_id:
            continue
        text = _evidence_model_text(package)
        package_roles = _text_roles(text)
        score = len(package_roles.intersection(roles))
        for surface in list(getattr(package, "implementation_surfaces", []) or []):
            if _text_roles(str(surface)).intersection(roles):
                score += 1
        if score:
            scores[work_package_id] = score
    return scores


def _contract_shard_owner_map(state: PaperBenchReproState | None) -> dict[str, str]:
    if state is None or state.work_package_planning is None:
        return {}
    first_work_package = next(
        (
            str(item.work_package_id or "").strip()
            for item in list(state.work_package_planning.work_packages or [])
            if str(item.work_package_id or "").strip()
        ),
        "",
    )
    owner_map: dict[str, str] = {}
    for path in _contract_shard_paths(state):
        scores = _work_package_score_for_roles(state, _CONTRACT_SHARD_OWNER_ROLES.get(path, set()))
        if scores:
            owner_map[path] = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[0][0]
        elif first_work_package:
            owner_map[path] = first_work_package
    return owner_map


def _bounded(items: list[str], limit: int = 18) -> list[str]:
    return _dedupe(items)[:limit]


def _inventory_obligations(inventories: dict[str, list[str]]) -> list[str]:
    """Turn preserved inventories into generation-facing implementation obligations."""
    obligation_matrix = _bounded(list(inventories.get("obligation_matrix", []) or []), 10)
    experiments = _bounded(list(inventories.get("experiment_inventory", []) or []), 10)
    environments = _bounded(list(inventories.get("environment_inventory", []) or []), 10)
    datasets = _bounded(list(inventories.get("dataset_inventory", []) or []), 10)
    baselines = _bounded(
        list(inventories.get("baseline_inventory", []) or [])
        + list(inventories.get("method_inventory", []) or [])
        + list(inventories.get("policy_inventory", []) or [])
        + list(inventories.get("model_inventory", []) or []),
        10,
    )
    measurements = _bounded(
        list(inventories.get("measurement_inventory", []) or [])
        + list(inventories.get("metric_inventory", []) or [])
        + list(inventories.get("required_measurement_inventory", []) or []),
        10,
    )
    parameters = _bounded(list(inventories.get("parameter_inventory", []) or []), 10)
    trends = _bounded(list(inventories.get("result_trend_inventory", []) or []), 10)
    result_artifacts = _bounded(
        list(inventories.get("result_artifact_inventory", []) or [])
        + list(inventories.get("artifact_inventory", []) or []),
        10,
    )
    obligations: list[str] = []
    if obligation_matrix:
        obligations.append("Preserve paper-derived evidence obligation matrix rows: " + " | ".join(obligation_matrix))
    if experiments:
        obligations.append("Materialize named experiment protocols: " + " | ".join(experiments))
    if environments:
        obligations.append("Expose explicit environment/task coverage: " + " | ".join(environments))
    if datasets:
        obligations.append("Expose explicit dataset/benchmark coverage: " + " | ".join(datasets))
    if baselines:
        obligations.append("Expose explicit method, baseline, or variant selectors: " + " | ".join(baselines))
    if measurements:
        obligations.append("Collect and aggregate required measurements: " + " | ".join(measurements))
    if parameters:
        obligations.append("Expose bounded parameter sweep configs/registries: " + " | ".join(parameters))
    if trends:
        obligations.append("Preserve required result-trend assertions: " + " | ".join(trends))
    if result_artifacts:
        obligations.append("Write or declare verifiable result artifacts: " + " | ".join(result_artifacts))
    return _dedupe(obligations)


def _file_roles(file_path: str) -> set[str]:
    normalized = _normalize_repo_path(file_path).lower()
    basename = normalized.rsplit("/", 1)[-1]
    stem = basename.rsplit(".", 1)[0]
    roles: set[str] = set()
    if normalized == "main.py" or basename in {"cli.py", "run.py"} or normalized.startswith("scripts/"):
        roles.add("entrypoint")
    if basename in {"readme.md", "readme.rst"} or normalized.startswith("docs/"):
        roles.add("doc")
    if basename in {"requirements.txt", "pyproject.toml", "setup.py", "setup.cfg", "environment.yml"}:
        roles.add("packaging")
    if (
        normalized.startswith(("configs/", "config/"))
        or basename in {"config.py", "configs.py", "constants.py", "settings.py"}
        or basename.endswith((".yaml", ".yml", ".toml", ".ini", ".cfg"))
    ):
        roles.add("config")
    if normalized.startswith("tests/") or basename.startswith("test_") or basename.endswith("_test.py"):
        roles.add("test")
    if any(token in normalized for token in ("environment", "environments", "/env", "envs", "simulator", "task")):
        roles.add("environment")
    if any(token in normalized for token in ("data", "dataset", "datasets", "loader", "preprocess")):
        roles.add("data")
    if any(token in normalized for token in ("agent", "agents", "policy", "policies")):
        roles.add("agent")
    if any(token in normalized for token in ("model", "models", "network", "encoder", "decoder")):
        roles.add("model")
    if any(token in normalized for token in ("method", "methods", "algorithm", "attack", "loss")):
        roles.add("method")
    if any(token in normalized for token in ("train", "training", "trainer", "finetune", "optim")):
        roles.add("training")
    if any(token in normalized for token in ("explain", "explainer", "explanation", "mask", "saliency")):
        roles.add("explainer")
    if any(token in normalized for token in ("refine", "refinement", "adaptation", "adapter")):
        roles.add("refinement")
    if any(token in normalized for token in ("baseline", "baselines", "ablation", "variant")):
        roles.add("baseline")
    if any(token in normalized for token in ("evaluation", "evaluate", "metric", "metrics", "score", "fidelity", "benchmark")):
        roles.add("evaluation")
    if any(token in normalized for token in ("experiment", "experiments", "protocol", "runner", "matrix")):
        roles.add("experiment")
    if any(token in normalized for token in ("artifact", "artifacts", "report", "reporting", "plot", "figure", "result", "output")):
        roles.add("artifact")
    if any(token in normalized for token in ("plot", "plotting", "figure", "figures")):
        roles.add("plotting")
    if any(token in normalized for token in ("report", "reporting")):
        roles.add("reporting")
    if normalized == "src/experiment_registry.py":
        roles.update({"experiment", "registry_shard"})
    if normalized == "src/environment_registry.py":
        roles.update({"environment", "registry_shard"})
    if normalized == "src/dataset_registry.py":
        roles.update({"data", "registry_shard"})
    if normalized == "src/method_registry.py":
        roles.update({"method", "baseline", "registry_shard"})
    if normalized == "src/sweep_registry.py":
        roles.update({"config", "experiment", "registry_shard"})
    if normalized == "src/artifact_contract.py":
        roles.update({"artifact", "registry_shard"})
    if normalized == "src/trend_assertions.py":
        roles.update({"evaluation", "artifact", "registry_shard"})
    if not roles and normalized.endswith(".py") and normalized.startswith("src/"):
        roles.add("method")
    if not roles and stem in {"metrics", "results", "summary"}:
        roles.add("artifact")
    return roles


def _has_refinement_semantic_signal(text: str) -> bool:
    lowered = str(text or "").lower().replace("_", " ").replace("-", " ")
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
    return bool(re.search(r"\b(adapt|adaptive|adaptation|adapter|refiner|refining)\b", lowered))


def _text_roles(text: str) -> set[str]:
    lowered = str(text or "").lower().replace("_", " ").replace("-", " ")
    tokens = set(re.findall(r"[a-z0-9]+", lowered))

    def has_any(*values: str) -> bool:
        for value in values:
            normalized = str(value or "").lower()
            if " " in normalized:
                if normalized in lowered:
                    return True
            elif normalized in tokens:
                return True
        return False

    roles: set[str] = set()
    if (
        has_any("entrypoint", "entry point", "cli", "command line", "run script", "dispatch", "dry run", "manifest", "smoke mode")
        or re.search(r"\bmain(?:\.py|\s+entry|\s+script|\s+cli)?\b", lowered)
    ):
        roles.add("entrypoint")
    if has_any("config", "configuration", "hyperparameter", "parameter", "sweep", "seed", "registry"):
        roles.add("config")
    if has_any("environment", "task", "simulator", "benchmark"):
        roles.add("environment")
    if has_any("dataset", "data loader", "dataloader", "preprocess", "sampling"):
        roles.add("data")
    if has_any("policy", "agent", "checkpoint"):
        roles.add("agent")
    if has_any("model", "network", "encoder", "decoder", "backbone"):
        roles.add("model")
    if has_any("method", "algorithm", "objective", "loss", "attack", "adapter", "selector"):
        roles.add("method")
    if has_any("train", "training", "pretrain", "fine tuning", "finetune", "optimizer", "epoch"):
        roles.add("training")
    if has_any("explain", "explanation", "mask", "saliency", "importance"):
        roles.add("explainer")
    if _has_refinement_semantic_signal(text):
        roles.add("refinement")
    if has_any("baseline", "ablation", "variant", "comparison", "compare"):
        roles.add("baseline")
    if has_any("evaluate", "evaluation", "metric", "score", "accuracy", "reward", "return", "fidelity", "aggregate"):
        roles.add("evaluation")
    if has_any("experiment", "protocol", "table", "figure", "study", "matrix"):
        roles.add("experiment")
    if has_any("artifact", "result", "output", "report", "plot", "figure", "writer", "summary"):
        roles.add("artifact")
    if has_any("report", "reporting", "aggregate", "aggregation", "summary"):
        roles.add("reporting")
    if has_any("test", "contract", "validation", "assert"):
        roles.add("test")
    if has_any("dependency", "dependencies", "package", "install", "requirement"):
        roles.add("packaging")
    if has_any("readme", "document", "usage", "instruction"):
        roles.add("doc")
    return roles


def _inventory_owner_roles(inventory_name: str) -> set[str]:
    key = str(inventory_name or "").strip().lower()
    roles: set[str] = set()
    if "obligation" in key:
        roles.update({"experiment", "evaluation", "artifact", "config", "test"})
    if "experiment" in key or "protocol" in key:
        roles.update({"experiment", "evaluation", "config", "test"})
    if "environment" in key or key in {"env_inventory"}:
        roles.update({"environment", "config", "data", "test"})
    if "dataset" in key or "benchmark" in key:
        roles.update({"data", "environment", "evaluation", "config", "test"})
    if any(token in key for token in ("baseline", "method", "policy", "model", "variant")):
        roles.update({"method", "model", "agent", "training", "baseline", "explainer", "config", "test"})
    if "measurement" in key or "metric" in key:
        roles.update({"evaluation", "artifact", "experiment", "test"})
    if "parameter" in key or "sweep" in key:
        roles.update({"config", "experiment", "training", "method", "baseline", "test"})
    if "trend" in key:
        roles.update({"evaluation", "artifact", "experiment", "test"})
    if "scope" in key or "constraint" in key:
        roles.update({"config", "experiment", "doc", "test"})
    if "artifact" in key or "result" in key:
        roles.update({"artifact", "evaluation", "experiment", "test"})
    if "surface" in key or "interface" in key:
        roles.update(
            {
                "entrypoint",
                "config",
                "environment",
                "data",
                "agent",
                "model",
                "method",
                "training",
                "explainer",
                "baseline",
                "evaluation",
                "experiment",
                "artifact",
                "test",
            }
        )
    return roles


def _filter_inventories_for_file(file_path: str, inventories: dict[str, list[str]]) -> dict[str, list[str]]:
    roles = _file_roles(file_path)
    if not roles:
        return inventories
    filtered: dict[str, list[str]] = {}
    for key, values in dict(inventories or {}).items():
        owner_roles = _inventory_owner_roles(key)
        if owner_roles and not roles.intersection(owner_roles):
            continue
        scoped_values = [
            str(item)
            for item in list(values or [])
            if str(item).strip()
            and (
                not _text_roles(str(item))
                or roles.intersection(_text_roles(str(item)))
                or roles.intersection(owner_roles)
            )
        ]
        if scoped_values:
            filtered[str(key)] = _bounded(scoped_values, 24)
    return filtered


def _file_scoped_surfaces(file_path: str, surfaces: list[str]) -> list[str]:
    roles = _file_roles(file_path)
    if not roles:
        return _dedupe(surfaces)
    scoped = [
        str(surface)
        for surface in list(surfaces or [])
        if str(surface).strip()
        and (
            not _text_roles(str(surface))
            or roles.intersection(_text_roles(str(surface)))
            or ("entrypoint" in roles and str(surface).strip().lower() in {"entry", "entrypoint", "cli", "main"})
        )
    ]
    if not scoped and roles.intersection({"doc", "packaging", "entrypoint"}):
        return []
    return _dedupe(scoped)


def _obligation_relevant_to_file(file_path: str, obligation: str) -> bool:
    roles = _file_roles(file_path)
    if not roles:
        return True
    rendered = str(obligation or "").strip()
    if not rendered:
        return False
    text_roles = _text_roles(rendered)
    lowered = rendered.lower()
    if "entrypoint" in roles:
        return bool(
            text_roles.intersection({"entrypoint"})
            or any(token in lowered for token in ("dry-run", "dry run", "manifest", "parse", "dispatch", "canonical run"))
        )
    if "packaging" in roles:
        return bool(text_roles.intersection({"packaging"}))
    if "doc" in roles:
        return bool(text_roles.intersection({"doc", "entrypoint", "config", "experiment", "artifact"}))
    if "test" in roles:
        return bool(text_roles.intersection({"test", "config", "experiment", "evaluation", "artifact", "method", "baseline", "environment", "data"}))
    if not text_roles:
        return not roles.intersection({"doc", "packaging"})
    if roles.intersection(text_roles):
        return True
    if "experiment" in roles and text_roles.intersection({"config", "evaluation", "artifact", "baseline", "environment", "data"}):
        return True
    if "evaluation" in roles and text_roles.intersection({"baseline", "artifact", "experiment", "data", "environment"}):
        return True
    if "artifact" in roles and text_roles.intersection({"evaluation", "experiment"}):
        return True
    if "config" in roles and text_roles.intersection({"environment", "data", "method", "baseline", "experiment", "training"}):
        return True
    if roles.intersection({"method", "training", "model", "agent", "baseline", "explainer", "refinement"}) and text_roles.intersection(
        {"method", "training", "model", "agent", "baseline", "explainer", "refinement", "config"}
    ):
        return True
    return False


def _obligation_limit_for_file(file_path: str) -> int:
    roles = _file_roles(file_path)
    if "registry_shard" in roles:
        return 10
    if "entrypoint" in roles:
        return 12
    if roles.intersection({"doc", "packaging"}):
        return 6
    if "test" in roles:
        return 10
    if roles.intersection({"experiment", "evaluation"}):
        return 10
    if roles.intersection({"artifact", "config"}):
        return 10
    return 8


def _file_scoped_obligations(file_path: str, obligations: list[str]) -> list[str]:
    scoped = [
        str(obligation)
        for obligation in list(obligations or [])
        if _obligation_relevant_to_file(file_path, str(obligation))
    ]
    return _bounded(scoped, _obligation_limit_for_file(file_path))


def _contract_terms(contract: dict[str, object], key: str) -> list[str]:
    return _dedupe([str(item) for item in list(contract.get(key, []) or []) if str(item).strip()])


def _render_sweep_contract(item: dict[str, object]) -> str:
    """Render a sweep with exact values so static gates can verify coverage."""
    name = str(item.get("name", "") or "").strip()
    values = [
        str(value).strip()
        for value in list(item.get("values", []) or [])
        if str(value).strip()
    ]
    if values:
        return f"{name} values " + ", ".join(values)
    return name


def _trend_contract_phrase(name: str) -> str:
    normalized = str(name or "").strip()
    if normalized == "endpoint_low":
        return "endpoint_low with p=0 and p=1 endpoint/boundary cases expected to be lowest/minimum/worst"
    if normalized == "sweep_insensitive":
        return "sweep_insensitive with stable/insensitive/robust parameter-sweep behavior"
    if normalized == "baseline_outperformance":
        return "baseline_outperformance with explicit comparison showing improvement over baselines"
    return normalized


def _executable_contract_obligations(file_path: str, obligations: list[str]) -> list[str]:
    """Add generation-facing constraints that force paper anchors into code paths."""
    roles = _file_roles(file_path)
    if not roles.intersection(
        {
            "entrypoint",
            "config",
            "method",
            "training",
            "evaluation",
            "experiment",
            "artifact",
            "model",
            "agent",
            "baseline",
            "refinement",
            "test",
        }
    ):
        return []
    text = "\n".join(str(item or "") for item in list(obligations or []))
    lowered = text.lower()
    anchors: list[str] = []
    has_numeric_anchor = bool(
        re.search(r"(?<![a-z0-9])(?:\d+\.\d+|\d+)(?:e[-+]?\d+)?(?![a-z0-9])", lowered)
        or any(token in lowered for token in ("=", "values ", "sweep", "hyperparameter", "batch size", "learning rate"))
    )
    has_formula_anchor = bool(
        any(token in lowered for token in ("formula", "objective", "loss", "metric", "aggregation", "gradient", "posterior", "likelihood", "ratio"))
    )
    has_algorithm_anchor = bool(
        any(token in lowered for token in ("algorithm", "training loop", "evaluation loop", "attack", "adaptation", "optimizer", "simulator", "sampler", "factory"))
    )
    has_artifact_anchor = bool(any(token in lowered for token in ("table", "figure", "artifact", "report", "metrics.json", "prediction")))
    if has_numeric_anchor or has_formula_anchor:
        anchors.append(
            "Executable anchor contract: exact numeric constants, defaults, sweep values, formulas, objectives, and metric aggregations named here must be implemented as constants/dataclasses/functions used by runtime code, not only README text, schemas, or registry rows."
        )
    if has_algorithm_anchor:
        anchors.append(
            "Executable algorithm contract: paper-visible algorithms must expose concrete functions/classes and be called by train/evaluate/sample/compare routes."
        )
    if has_artifact_anchor:
        anchors.append(
            "Executable artifact contract: table/figure/metric/prediction writers must call concrete method, simulator, dataset, and metric functions on bounded inputs; schema-only or dry-run-only result files are not valid paper-visible outputs."
        )
    if roles.intersection({"entrypoint", "experiment", "training", "evaluation"}):
        anchors.append(
            "Canonical route contract: smoke mode must execute the same implementation path with bounded data/config, while full mode can scale it; do not create a separate fake path for validation."
        )
    return _dedupe(anchors)


def _priority_evidence_contract_obligations(state: PaperBenchReproState | None, file_path: str) -> list[str]:
    """Keep complete paper-derived enumerations ahead of lossy per-file summaries."""
    if state is None or state.unit_extraction is None:
        return []
    roles = _file_roles(file_path)
    normalized_path = _normalize_repo_path(file_path).lower()
    if "registry_shard" in roles:
        if normalized_path != "src/method_registry.py":
            return []
    owns_method = bool(
        roles.intersection({"method", "training", "model", "agent", "baseline", "explainer", "refinement", "test"})
    )
    owns_metric = bool(roles.intersection({"experiment", "evaluation", "artifact", "test"}))
    if not (owns_method or owns_metric):
        return []
    contract = _state_evidence_contract(state)
    obligations: list[str] = []
    if owns_method:
        methods = _contract_terms(contract, "methods")
        sweeps = [
            _render_sweep_contract(item)
            for item in list(contract.get("parameter_sweeps", []) or [])
            if isinstance(item, dict) and str(item.get("name", "") or "").strip()
        ]
        fixed_hyperparameters = _contract_terms(contract, "fixed_hyperparameters")
        if methods:
            obligations.append(
                "Paper evidence contract priority methods: complete method/baseline selector set must include "
                + ", ".join(methods)
                + "."
            )
        if sweeps:
            obligations.append(
                "Paper evidence contract priority sweeps: complete bounded parameter sweeps must include "
                + "; ".join(sweeps)
                + "."
            )
        if fixed_hyperparameters:
            obligations.append(
                "Paper evidence contract priority fixed hyperparameters: preserve exact anchors "
                + ", ".join(fixed_hyperparameters)
                + "."
            )
    if owns_metric:
        trends = [_trend_contract_phrase(item) for item in _contract_terms(contract, "trend_obligations")]
        if trends:
            obligations.append(
                "Paper evidence contract priority trends: preserve complete trend_obligations including "
                + "; ".join(trends)
                + "."
            )
    return _file_scoped_obligations(file_path, obligations)


def _formula_algorithm_contract_from_state(state: PaperBenchReproState | None) -> dict[str, object]:
    if state is None:
        return {}
    cache_key = "plan_builder_formula_algorithm_contract_v1"
    cached = state.temp_data.get(cache_key)
    if isinstance(cached, dict):
        return cached
    try:
        from reproagent.pipeline.utils.prompt_context_builder import _paper_evidence_contract_payload

        evidence_contract = _paper_evidence_contract_payload(state)
        contract = dict(evidence_contract.get("formula_algorithm_contract", {}) or {})
    except Exception:
        contract = {}
    state.temp_data[cache_key] = contract
    return contract


def _formula_algorithm_obligations_for_file(state: PaperBenchReproState | None, file_path: str) -> list[str]:
    """Project paper-derived formula/algorithm anchors into executable owner files."""
    roles = _file_roles(file_path)
    if not roles.intersection({"method", "model", "training", "evaluation", "experiment", "config", "baseline", "refinement", "test"}):
        return []
    contract = _formula_algorithm_contract_from_state(state)
    if not contract:
        return []
    anchors = [item for item in list(contract.get("anchors", []) or []) if isinstance(item, dict)]
    symbol_inventory = [
        str(item).strip()
        for item in list(contract.get("required_symbol_inventory", []) or [])
        if str(item).strip()
    ]
    numeric_inventory = [
        str(item).strip()
        for item in list(contract.get("required_numeric_inventory", []) or [])
        if str(item).strip()
    ]
    implementation_obligations = [
        str(item).strip()
        for item in list(contract.get("implementation_obligations", []) or [])
        if str(item).strip()
    ]
    if not any([anchors, symbol_inventory, numeric_inventory, implementation_obligations]):
        return []
    obligations: list[str] = []
    if roles.intersection({"method", "model", "training", "evaluation", "experiment", "baseline", "refinement"}):
        anchor_phrases: list[str] = []
        for anchor in anchors[:10]:
            section = str(anchor.get("section_title", "") or anchor.get("source_id", "") or "").strip()
            excerpts = [
                str(item).strip()
                for item in list(anchor.get("formula_or_algorithm_excerpts", []) or [])[:2]
                if str(item).strip()
            ]
            steps = [
                str(item).strip()
                for item in list(anchor.get("algorithm_steps", []) or [])[:2]
                if str(item).strip()
            ]
            symbols = [
                str(item).strip()
                for item in list(anchor.get("required_symbols", []) or [])[:8]
                if str(item).strip()
            ]
            numeric_values = [
                str(item).strip()
                for item in list(anchor.get("required_numeric_values", []) or [])[:6]
                if str(item).strip()
            ]
            parts = []
            if section:
                parts.append(section)
            if symbols:
                parts.append("symbols=" + ", ".join(symbols))
            if numeric_values:
                parts.append("values=" + ", ".join(numeric_values))
            if steps:
                parts.append("steps=" + " ; ".join(steps))
            elif excerpts:
                parts.append("formula=" + " ; ".join(excerpts))
            if parts:
                anchor_phrases.append(" | ".join(parts))
        if anchor_phrases:
            obligations.append(
                "Paper formula/algorithm contract: implement these anchors as executable code/config and canonical-route calls, not README or schema text: "
                + " || ".join(anchor_phrases)
            )
    if roles.intersection({"config", "method", "training", "evaluation", "test"}) and (symbol_inventory or numeric_inventory):
        obligations.append(
            "Paper formula/algorithm symbol inventory must be code-visible: "
            + ", ".join(_bounded(symbol_inventory, 24))
            + ("; numeric/default anchors: " + ", ".join(_bounded(numeric_inventory, 16)) if numeric_inventory else "")
        )
    if implementation_obligations and roles.intersection({"method", "training", "evaluation", "experiment", "test"}):
        obligations.append(
            "Paper formula/algorithm implementation obligations: "
            + " | ".join(_bounded(implementation_obligations, 20))
        )
    return _file_scoped_obligations(file_path, sanitize_contract_list(obligations, field="method_obligations"))


def _global_contract_inventories(state: PaperBenchReproState | None) -> dict[str, list[str]]:
    if state is None or state.global_contract is None:
        return {}
    inventories: dict[str, list[str]] = {}
    for key, values in dict(state.global_contract.inventories or {}).items():
        normalized_key = str(key or "").strip()
        if not normalized_key:
            continue
        cleaned_values: list[str] = []
        for item in list(values or []):
            value = str(item or "").strip()
            if not value:
                continue
            cleaned_values.append(value)
        if cleaned_values:
            inventories.setdefault(normalized_key, [])
            inventories[normalized_key].extend(cleaned_values)
    if state.unit_extraction is not None:
        contract = _state_evidence_contract(state)
        artifact_paths = _paper_artifact_paths(_contract_terms(contract, "artifacts"))
        if artifact_paths:
            inventories.setdefault("artifact_inventory", [])
            inventories["artifact_inventory"].extend(artifact_paths)
            inventories.setdefault("result_artifact_inventory", [])
            inventories["result_artifact_inventory"].extend(artifact_paths)
    return {key: _bounded(values, 32) for key, values in inventories.items() if values}

def _work_package_inventories(state: PaperBenchReproState | None, work_package_id: str) -> dict[str, list[str]]:
    if state is None or state.work_package_planning is None:
        return {}
    work_package = next(
        (
            item
            for item in list(state.work_package_planning.work_packages or [])
            if str(item.work_package_id or "").strip() == str(work_package_id or "").strip()
        ),
        None,
    )
    if work_package is None:
        return {}
    return {
        str(key): _bounded([str(item) for item in list(values or []) if str(item).strip()], 24)
        for key, values in dict(work_package.inventories or {}).items()
        if str(key).strip()
    }


def _merge_inventories(*inventory_maps: dict[str, list[str]]) -> dict[str, list[str]]:
    merged: dict[str, list[str]] = {}
    for inventory_map in inventory_maps:
        for key, values in dict(inventory_map or {}).items():
            normalized_key = str(key or "").strip()
            if not normalized_key:
                continue
            merged.setdefault(normalized_key, [])
            merged[normalized_key].extend(str(item) for item in list(values or []) if str(item).strip())
    return {key: _bounded(values, 24) for key, values in merged.items() if values}


def _generation_inventories_for_file(
    state: PaperBenchReproState | None,
    work_package_id: str,
    file_path: str,
) -> dict[str, list[str]]:
    package_inventories = _filter_inventories_for_file(
        file_path,
        _work_package_inventories(state, work_package_id),
    )
    global_inventories = _filter_inventories_for_file(file_path, _global_contract_inventories(state))
    return _merge_inventories(global_inventories, package_inventories)


def _file_inventory_obligations(file_path: str, inventories: dict[str, list[str]]) -> list[str]:
    roles = _file_roles(file_path)
    normalized_path = _normalize_repo_path(file_path).lower()
    obligation_matrix = _bounded(list(inventories.get("obligation_matrix", []) or []), 12)
    experiments = _bounded(list(inventories.get("experiment_inventory", []) or []), 10)
    environments = _bounded(list(inventories.get("environment_inventory", []) or []), 12)
    datasets = _bounded(list(inventories.get("dataset_inventory", []) or []), 12)
    baselines = _bounded(
        list(inventories.get("baseline_inventory", []) or [])
        + list(inventories.get("method_inventory", []) or [])
        + list(inventories.get("policy_inventory", []) or [])
        + list(inventories.get("model_inventory", []) or []),
        12,
    )
    variants = _bounded(list(inventories.get("variant_inventory", []) or []), 10)
    measurements = _bounded(
        list(inventories.get("measurement_inventory", []) or [])
        + list(inventories.get("metric_inventory", []) or [])
        + list(inventories.get("required_measurement_inventory", []) or []),
        10,
    )
    parameters = _bounded(list(inventories.get("parameter_inventory", []) or []), 12)
    trends = _bounded(list(inventories.get("result_trend_inventory", []) or []), 12)
    artifacts = _bounded(
        list(inventories.get("result_artifact_inventory", []) or [])
        + list(inventories.get("artifact_inventory", []) or []),
        12,
    )
    obligations: list[str] = []
    canonical_measurements = _dedupe(
        [
            slug
            for measurement in measurements
            for slug in (
                _symbol_slug(measurement),
                _symbol_slug(measurement, prefix="metric"),
            )
            if slug
        ]
    )
    canonical_artifacts = _dedupe(
        [
            slug
            for artifact in artifacts
            for slug in (
                _symbol_slug(artifact),
                _symbol_slug(artifact, prefix="artifact"),
            )
            if slug
        ]
    )
    is_shard = "registry_shard" in roles
    owns_environment = (
        normalized_path in {"src/environment_registry.py", "src/dataset_registry.py"}
        or (not is_shard and bool(roles.intersection({"environment", "data", "config", "test"})))
    )
    owns_experiment = (
        normalized_path == "src/experiment_registry.py"
        or (not is_shard and bool(roles.intersection({"experiment", "evaluation", "config", "artifact", "test", "doc"})))
    )
    owns_methods = (
        normalized_path == "src/method_registry.py"
        or (not is_shard and bool(roles.intersection({"method", "training", "agent", "model", "baseline", "explainer", "refinement", "test"})))
    )
    owns_measurements = (
        normalized_path == "src/artifact_contract.py"
        or (not is_shard and bool(roles.intersection({"evaluation", "artifact", "experiment", "test"})))
    )
    owns_artifacts = (
        normalized_path == "src/artifact_contract.py"
        or (not is_shard and bool(roles.intersection({"artifact", "evaluation", "experiment", "test", "doc"})))
    )
    owns_parameters = (
        normalized_path == "src/sweep_registry.py"
        or (not is_shard and bool(roles.intersection({"config", "experiment", "training", "method", "baseline", "refinement", "test"})))
    )
    owns_trends = (
        normalized_path == "src/trend_assertions.py"
        or (not is_shard and bool(roles.intersection({"evaluation", "artifact", "experiment", "test"})))
    )
    if obligation_matrix and owns_experiment:
        obligations.append(
            "In this file, implement paper-derived evidence obligation matrix rows as callable experiment specs that bind environments, methods, parameter defaults, metric functions, and artifact writer call sites: "
            + " | ".join(obligation_matrix)
        )
    if environments and owns_environment:
        obligations.append(
            "In this file, expose paper-derived environment/task factories with ids, aliases, setup metadata, availability checks, and runnable config hooks for: "
            + " | ".join(environments)
        )
    if datasets and owns_environment:
        obligations.append(
            "In this file, expose paper-derived dataset/benchmark loaders with ids, setup metadata, validation checks, and runnable config hooks for: "
            + " | ".join(datasets)
        )
    if experiments and owns_experiment:
        obligations.append(
            "In this file, materialize a callable protocol matrix linking named experiments to environments/tasks, method selectors, metric functions, and artifact writer functions: "
            + " | ".join(experiments)
        )
    method_items = _bounded(baselines + variants, 14)
    if method_items and owns_methods:
        obligations.append(
            "In this file, expose selectable method/baseline/variant factories or adapters backed by concrete implementation functions/classes for: "
            + " | ".join(method_items)
        )
    if measurements and owns_measurements:
        obligations.append(
            "In this file, implement metric formulas, aggregation functions, and result field writers for: "
            + " | ".join(measurements)
        )
        if canonical_measurements:
            obligations.append(
                "In this file, preserve canonical metric identifiers for static review: "
                + " | ".join(canonical_measurements)
            )
    if parameters and owns_parameters:
        obligations.append(
            "In this file, expose required parameter sweeps as executable constants/default accessors used by train/evaluate/report routes, not only registry values: "
            + " | ".join(parameters)
        )
    if trends and owns_trends:
        obligations.append(
            "In this file, preserve required result-trend assertions for semantic review: "
            + " | ".join(trends)
        )
    if artifacts and owns_artifacts:
        obligations.append(
            "In this file, make result artifact paths statically discoverable and implement writer functions that call evaluation/metric code for: "
            + " | ".join(artifacts)
        )
        if canonical_artifacts:
            obligations.append(
                "In this file, preserve canonical artifact identifiers for static review: "
                + " | ".join(canonical_artifacts)
            )
    if not is_shard and roles.intersection({"entrypoint", "experiment", "training", "evaluation", "method", "model", "baseline", "refinement"}):
        matrix_parts = []
        if experiments:
            matrix_parts.append("experiments=" + " | ".join(experiments[:8]))
        if environments or datasets:
            matrix_parts.append("datasets_or_tasks=" + " | ".join(_bounded(environments + datasets, 10)))
        if method_items:
            matrix_parts.append("methods_or_models=" + " | ".join(method_items[:10]))
        if parameters:
            matrix_parts.append("parameters=" + " | ".join(parameters[:8]))
        if measurements:
            matrix_parts.append("metrics=" + " | ".join(measurements[:8]))
        if matrix_parts:
            obligations.append(
                "Full experiment-matrix route contract: implement executable orchestration over the declared "
                "paper-derived dimensions, not only a registry or prose summary: "
                + " ; ".join(matrix_parts)
            )
    if not is_shard and roles.intersection({"training", "refinement", "baseline", "experiment", "method"}):
        obligations.append(
            "Implement the full data/model/training/evaluation route implied by the paper-derived method inventory. "
            "A dry-run or smoke mode may use tiny fixtures while calling the same real loader, model factory, optimizer/refinement, pairwise evaluation, metric, checkpoint/artifact, or comparison code."
        )
        obligations.append(
            "Wire paper-derived objective, reward, metric, sweep, and baseline obligations into callable primary functions/classes reached by train/evaluate/compare paths."
        )
    return _file_scoped_obligations(file_path, sanitize_contract_list(obligations, field="method_obligations"))




def _addendum_obligations_for_file(state: PaperBenchReproState | None, file_path: str) -> list[str]:
    """Project binding addendum clarifications into concrete file-level responsibilities."""
    addendum = _addendum_text_for_state(state)
    if not addendum.strip():
        return []
    roles = _file_roles(file_path)
    normalized_path = _normalize_repo_path(file_path).lower()
    candidate_lines: list[str] = []
    for raw_line in addendum.splitlines():
        line = _positive_addendum_line(" ".join(str(raw_line or "").strip("-* ").split()))
        if not line:
            continue
        lowered = line.lower()
        has_binding_signal = bool(
            re.search(r"\d", line)
            or any(token in lowered for token in (
                "use ", "used ", "taken from", "implementation", "repository", "trust_remote_code",
                "attack", "epsilon", "parameter", "iteration", "target", "table", "figure", "version",
                "momentum", "uniform", "non-normalized", "half precision", "single precision",
            ))
        )
        if not has_binding_signal:
            continue
        line_roles = _text_roles(line)
        if any(token in lowered for token in ("dataset", "data", "huggingface", "trust_remote_code")):
            line_roles.update({"data", "environment", "config"})
        if any(token in lowered for token in ("vision encoder", "encoder", "backbone", "model", "network")):
            line_roles.update({"model", "method", "config"})
        if any(token in lowered for token in ("attack", "target", "epsilon", "parameter", "iterations", "momentum")):
            line_roles.update({"evaluation", "method", "experiment", "config"})
        if any(token in lowered for token in ("table", "figure", "artifact", "benchmark")):
            line_roles.update({"artifact", "evaluation", "experiment"})
        if any(token in lowered for token in ("activation", "version", "optimizer")):
            line_roles.update({"method", "training", "config", "experiment"})
        if "doc" in roles or "test" in roles or roles.intersection(line_roles):
            candidate_lines.append(line)
    if not candidate_lines:
        return []
    prefix = "Binding addendum clarification for this file: "
    if normalized_path in {"configs/default.yaml", "src/artifact_contract.py"} or "doc" in roles:
        limit = 12
    elif roles.intersection({"evaluation", "experiment", "method", "model", "data", "config", "test"}):
        limit = 8
    else:
        limit = 4
    return _file_scoped_obligations(file_path, [prefix + item for item in _dedupe(candidate_lines)[:limit]])


def _source_artifact_contexts(state: PaperBenchReproState | None) -> list[str]:
    """Extract paper-visible table/figure caption and nearby comparison context."""
    source = _source_text_for_state(state)
    if not source.strip():
        return []
    caption_contexts: list[str] = []
    mention_contexts: list[str] = []
    artifact_label_re = re.compile(r"\b(?:Table|Figure|Fig\.?)\s+\d+[A-Za-z]?\b", flags=re.IGNORECASE)
    caption_re = re.compile(r"^\s*(?:Table|Figure|Fig\.?)\s+\d+[A-Za-z]?\s*[:.]", flags=re.IGNORECASE)
    for raw_line in source.splitlines():
        line = " ".join(str(raw_line or "").strip().split())
        if not line or not artifact_label_re.search(line):
            continue
        cleaned = line.replace("Fig.", "Figure")
        if len(cleaned) > 320:
            cleaned = cleaned[:317].rstrip() + "..."
        if caption_re.search(line):
            caption_contexts.append(cleaned)
        elif any(token in cleaned.lower() for token in ("outperform", "compare", "comparison", "sensitivity", "ablation", "baseline", "memo", "noadapt")):
            mention_contexts.append(cleaned)
    return _dedupe(caption_contexts + mention_contexts)


def _source_artifact_context_obligations_for_file(state: PaperBenchReproState | None, file_path: str) -> list[str]:
    roles = _file_roles(file_path)
    if not roles.intersection({"artifact", "plotting", "reporting", "evaluation", "experiment", "test", "doc"}):
        return []
    contexts = _source_artifact_contexts(state)
    if not contexts:
        return []
    limit = 36 if roles.intersection({"artifact", "evaluation", "experiment", "plotting"}) else 18
    obligation = (
        "Paper artifact context: preserve table/figure captions, named baselines, comparison semantics, and output mapping for "
        + " | ".join(contexts[:limit])
    )
    return _file_scoped_obligations(file_path, [obligation])

def _state_evidence_contract_obligations(state: PaperBenchReproState | None, file_path: str) -> list[str]:
    """Project inferred paper-derived terms into files that own registries or evaluation surfaces."""
    if state is None or state.unit_extraction is None:
        return []
    roles = _file_roles(file_path)
    normalized_path = _normalize_repo_path(file_path).lower()
    is_shard = "registry_shard" in roles
    owns_registry = (
        normalized_path in {"src/environment_registry.py", "src/dataset_registry.py", "src/experiment_registry.py"}
        or (not is_shard and bool(roles.intersection({"config", "experiment", "data", "environment", "test"})))
    )
    owns_method = (
        normalized_path in {"src/method_registry.py", "src/sweep_registry.py"}
        or (not is_shard and bool(roles.intersection({"config", "method", "training", "model", "agent", "baseline", "explainer", "refinement", "test"})))
    )
    owns_metric = (
        normalized_path in {"src/artifact_contract.py", "src/trend_assertions.py"}
        or (not is_shard and bool(roles.intersection({"experiment", "evaluation", "artifact", "test"})))
    )
    if not (owns_registry or owns_method or owns_metric):
        return []
    contract = _state_evidence_contract(state)
    obligations: list[str] = []
    if owns_registry:
        environments = _contract_terms(contract, "environments")
        datasets = _contract_terms(contract, "datasets")
        if environments and normalized_path != "src/dataset_registry.py":
            obligations.append(
                "Paper evidence contract: explicitly register environment/task aliases for "
                + ", ".join(environments)
                + "."
            )
        if datasets and normalized_path != "src/environment_registry.py":
            obligations.append(
                "Paper evidence contract: explicitly register dataset/benchmark aliases for "
                + ", ".join(datasets)
                + "."
            )
    if owns_method:
        methods = _contract_terms(contract, "methods")
        sweeps = [
            _render_sweep_contract(item)
            for item in list(contract.get("parameter_sweeps", []) or [])
            if isinstance(item, dict) and str(item.get("name", "") or "").strip()
        ]
        fixed_hyperparameters = _contract_terms(contract, "fixed_hyperparameters")
        if methods and normalized_path != "src/sweep_registry.py":
            obligations.append(
                "Paper evidence contract: expose method/baseline/attack selectors for "
                + ", ".join(methods)
                + "."
            )
        if sweeps and normalized_path != "src/method_registry.py":
            obligations.append(
                "Paper evidence contract: expose bounded sweep/config entries for "
                + "; ".join(sweeps)
                + "."
            )
        if fixed_hyperparameters and normalized_path not in {"src/method_registry.py", "src/sweep_registry.py"}:
            obligations.append(
                "Paper evidence contract: expose fixed hyperparameter anchors for "
                + ", ".join(fixed_hyperparameters)
                + "."
            )
    if owns_metric:
        metrics = _contract_terms(contract, "metrics")
        artifacts = _contract_terms(contract, "artifacts")
        artifact_paths = _paper_artifact_paths(artifacts)
        trends = [_trend_contract_phrase(item) for item in _contract_terms(contract, "trend_obligations")]
        if metrics and normalized_path != "src/trend_assertions.py":
            obligations.append(
                "Paper evidence contract: declare metric schemas/aggregations for "
                + ", ".join(metrics)
                + "."
            )
        if artifacts and normalized_path != "src/trend_assertions.py":
            artifact_clause = ", ".join(artifacts)
            if artifact_paths:
                artifact_clause += " with stable output paths " + ", ".join(artifact_paths)
            obligations.append(
                "Paper evidence contract: declare result artifact writers for "
                + artifact_clause
                + "."
            )
        if trends and normalized_path != "src/artifact_contract.py":
            obligations.append(
                "Paper evidence contract: preserve expected result-trend assertions for "
                + ", ".join(trends)
                + "."
            )
    return _file_scoped_obligations(file_path, obligations)


def _merge_evidence_contracts(*contracts: dict[str, object]) -> dict[str, object]:
    merged: dict[str, object] = {
        "requires_evidence_matrix": False,
        "named_experiments": [],
        "environments": [],
        "datasets": [],
        "methods": [],
        "metrics": [],
        "artifacts": [],
        "parameter_sweeps": [],
        "trend_obligations": [],
        "protocol_obligations": [],
        "fixed_hyperparameters": [],
        "implementation_obligations": [],
    }
    for contract in contracts:
        if not isinstance(contract, dict):
            continue
        merged["requires_evidence_matrix"] = bool(merged["requires_evidence_matrix"] or contract.get("requires_evidence_matrix"))
        for key in (
            "named_experiments",
            "environments",
            "datasets",
            "methods",
            "metrics",
            "artifacts",
            "trend_obligations",
            "protocol_obligations",
            "fixed_hyperparameters",
            "implementation_obligations",
        ):
            merged[key] = _dedupe(list(merged.get(key, []) or []) + [str(item) for item in list(contract.get(key, []) or []) if str(item).strip()])
        sweeps_by_name = {
            str(item.get("name", "") or "").strip(): dict(item)
            for item in list(merged.get("parameter_sweeps", []) or [])
            if isinstance(item, dict) and str(item.get("name", "") or "").strip()
        }
        for item in list(contract.get("parameter_sweeps", []) or []):
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "") or "").strip()
            if not name:
                continue
            existing = sweeps_by_name.setdefault(name, {"name": name, "values": []})
            existing["values"] = _dedupe(list(existing.get("values", []) or []) + [str(value) for value in list(item.get("values", []) or []) if str(value).strip()])
        merged["parameter_sweeps"] = list(sweeps_by_name.values())
    return merged


def _filter_contract_against_source(contract: dict[str, object], source_text: str) -> dict[str, object]:
    """Remove planning-only contract terms that are not grounded in paper/addendum text."""
    if not str(source_text or "").strip():
        return contract
    filtered = dict(contract)
    for key in (
        "named_experiments",
        "environments",
        "datasets",
        "methods",
        "metrics",
        "artifacts",
        "trend_obligations",
        "protocol_obligations",
        "fixed_hyperparameters",
        "implementation_obligations",
    ):
        filtered[key] = [
            item
            for item in list(contract.get(key, []) or [])
            if _source_mentions_term(str(item), source_text)
        ]
    filtered_sweeps: list[dict[str, object]] = []
    for item in list(contract.get("parameter_sweeps", []) or []):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "") or "").strip()
        values = [str(value) for value in list(item.get("values", []) or []) if str(value).strip()]
        if name and (
            _source_mentions_term(name, source_text)
            or any(_source_mentions_term(value, source_text) for value in values)
        ):
            filtered_sweeps.append(dict(item))
    filtered["parameter_sweeps"] = filtered_sweeps
    filtered["requires_evidence_matrix"] = bool(
        filtered_sweeps
        or any(list(filtered.get(key, []) or []) for key in (
            "named_experiments",
            "environments",
            "datasets",
            "methods",
            "metrics",
            "artifacts",
            "trend_obligations",
            "protocol_obligations",
            "fixed_hyperparameters",
            "implementation_obligations",
        ))
    )
    return filtered


def _state_evidence_contract(state: PaperBenchReproState) -> dict[str, object]:
    cache_key = "plan_builder_evidence_contract_v6"
    cached = state.temp_data.get(cache_key)
    if isinstance(cached, dict):
        return cached
    planning_text_parts: list[str] = []
    for unit in list(state.unit_extraction.units if state.unit_extraction else []):
        planning_text_parts.append(_evidence_model_text(unit))
    if state.work_package_planning is not None:
        for package in list(state.work_package_planning.work_packages or []):
            planning_text_parts.append(_evidence_model_text(package))
    source_text = _source_text_for_state(state)
    planning_contract = flatten_evidence_contract(infer_evidence_contract("\n".join(planning_text_parts)))
    planning_contract = _filter_contract_against_source(planning_contract, source_text)
    source_contract = flatten_evidence_contract(infer_evidence_contract(source_text)) if source_text else {}
    contract = _merge_evidence_contracts(source_contract, planning_contract)
    state.temp_data[cache_key] = contract
    return contract

def _evidence_model_text(item: object) -> str:
    names = (
        "statement",
        "goal",
        "purpose",
        "description",
        "generation_prompt",
        "paper_evidence",
        "implementation_surfaces",
        "code_obligations",
        "method_obligations",
        "runtime_interfaces",
        "interface_contract",
        "review_points",
        "defines_symbols",
        "tags",
        "hypothesis",
        "decision_value",
        "stop_rule_or_pruning_rationale",
    )
    values = object_values(item, names)
    if values:
        return " ".join(values)
    return _compact_model_text(item)


def _compact_model_text(item: object) -> str:
    if hasattr(item, "model_dump"):
        try:
            payload = item.model_dump(mode="json")
        except Exception:
            payload = str(item)
    else:
        payload = item
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)


def _file_owns_artifact_outputs(file_path: str) -> bool:
    return bool(_file_roles(file_path).intersection({"artifact", "plotting", "reporting"}))


def _artifact_is_plot_like(path: str) -> bool:
    lowered = _normalize_repo_path(path).lower()
    name = lowered.rsplit("/", 1)[-1]
    return (
        any(token in name for token in ("figure", "fig", "plot", "curve", "curves", "chart"))
        or lowered.startswith(("figures/", "plots/"))
        or name.endswith((".png", ".jpg", ".jpeg", ".pdf", ".svg"))
    )


def _file_scoped_artifact_outputs(file_path: str, artifact_paths: list[str]) -> list[str]:
    roles = _file_roles(file_path)
    normalized_file = _normalize_repo_path(file_path).lower()
    artifacts = _dedupe([path for path in list(artifact_paths or []) if _artifact_like(path)])
    if not artifacts or not _file_owns_artifact_outputs(file_path):
        return []
    is_dedicated_artifact_writer = any(token in normalized_file for token in ("artifact", "artifacts", "report", "reporting"))
    if is_dedicated_artifact_writer:
        return artifacts
    if "plotting" in roles:
        return [path for path in artifacts if _artifact_is_plot_like(path)]
    return []


def _artifact_inventory_paths(inventories: dict[str, list[str]]) -> list[str]:
    candidates = (
        list(inventories.get("artifact_inventory", []) or [])
        + list(inventories.get("result_artifact_inventory", []) or [])
    )
    return _dedupe(
        [
            _normalize_repo_path(path)
            for path in candidates
            if _artifact_like(_normalize_repo_path(path))
        ]
    )


def _artifact_like(path: str) -> bool:
    normalized = _normalize_repo_path(path)
    if not normalized:
        return False
    lowered = normalized.lower()
    return (
        lowered.startswith(("results/", "outputs/", "artifacts/", "reports/", "figures/", "plots/", "metrics/"))
        or lowered.endswith((".json", ".csv", ".tsv", ".txt", ".md", ".png", ".pdf", ".npy", ".npz", ".pt", ".pth", ".ckpt"))
        and any(token in lowered for token in ("result", "metric", "score", "table", "figure", "plot", "curve", "summary", "checkpoint"))
    )


def _global_result_artifacts_for_owner(state: PaperBenchReproState | None, work_package_id: str) -> list[str]:
    if state is None or state.global_contract is None:
        return []
    owner = str(work_package_id or "").strip()
    artifacts: list[str] = []
    for target in list(state.global_contract.result_targets or []):
        owners = {str(item or "").strip() for item in list(target.owner_work_packages or [])}
        if owner and owners and owner not in owners:
            continue
        artifacts.extend(_normalize_repo_path(path) for path in list(target.artifact_paths or []))
    return _dedupe([path for path in artifacts if _artifact_like(path)])


def _global_result_target_obligations_for_file(
    state: PaperBenchReproState | None,
    work_package_id: str,
    file_path: str,
) -> list[str]:
    """Project result-target names into executable metric/artifact owners."""
    if state is None or state.global_contract is None:
        return []
    roles = _file_roles(file_path)
    normalized_path = _normalize_repo_path(file_path).lower()
    owns_result_route = bool(
        roles.intersection({"entrypoint", "experiment", "evaluation", "artifact", "plotting", "reporting", "test"})
        or normalized_path in {"main.py", "src/artifact_contract.py", "src/trend_assertions.py"}
    )
    if not owns_result_route:
        return []
    owner = str(work_package_id or "").strip()
    obligations: list[str] = []
    for target in list(state.global_contract.result_targets or []):
        owners = {str(item or "").strip() for item in list(target.owner_work_packages or [])}
        if owner and owners and owner not in owners:
            continue
        name = str(target.name or target.target_id or "").strip()
        canonical_name = _symbol_slug(name, prefix="metric") or _symbol_slug(name, prefix="artifact")
        kind = str(target.kind or "result").strip()
        artifacts = [
            _normalize_repo_path(path)
            for path in list(target.artifact_paths or [])
            if _artifact_like(_normalize_repo_path(path))
        ]
        coverage = [str(item).strip() for item in list(target.coverage_notes or []) if str(item).strip()]
        if not name and not artifacts:
            continue
        artifact_clause = f" and write {', '.join(artifacts[:6])}" if artifacts else ""
        coverage_clause = f" Coverage notes: {' | '.join(coverage[:4])}." if coverage else ""
        canonical_clause = f" Canonical identifier: `{canonical_name}`." if canonical_name else ""
        obligations.append(
            f"Global result target: implement executable {kind} metric/result `{name}`{artifact_clause}; "
            "route it through experiment/evaluation/artifact code instead of leaving it as a registry or report label."
            + canonical_clause
            + coverage_clause
        )
    return _file_scoped_obligations(file_path, obligations)


def _global_measurement_obligations_for_file(
    state: PaperBenchReproState | None,
    file_path: str,
) -> list[str]:
    if state is None or state.global_contract is None:
        return []
    roles = _file_roles(file_path)
    normalized_path = _normalize_repo_path(file_path).lower()
    owns_measurement_route = bool(
        roles.intersection({"entrypoint", "experiment", "evaluation", "artifact", "plotting", "reporting", "test"})
        or normalized_path in {"main.py", "src/artifact_contract.py", "src/trend_assertions.py"}
    )
    if not owns_measurement_route:
        return []
    inventories = dict(state.global_contract.inventories or {})
    measurement_terms = _dedupe(
        [
            str(item or "").strip()
            for key in ("measurement_inventory", "metric_inventory", "required_measurement_inventory")
            for item in list(inventories.get(key, []) or [])
            if str(item or "").strip()
        ]
    )
    if not measurement_terms:
        return []
    rendered_terms: list[str] = []
    for term in measurement_terms[:16]:
        canonical = _symbol_slug(term) or _symbol_slug(term, prefix="metric")
        if canonical and canonical not in {term.lower(), term}:
            rendered_terms.append(f"{term} ({canonical})")
        else:
            rendered_terms.append(term)
    return _file_scoped_obligations(
        file_path,
        [
            "Global measurement inventory for canonical run entrypoint/evaluation route: "
            + ", ".join(rendered_terms)
            + ". These measurement names must appear in executable evaluation or artifact code.",
        ],
    )


def _work_package_contract_for_file(state: PaperBenchReproState | None, work_package_id: str) -> dict[str, list[str]]:
    """Return unit-derived implementation hints for one work package."""
    if state is None or state.work_package_planning is None:
        return {"interfaces": [], "surfaces": [], "obligations": [], "artifacts": []}
    work_package = next(
        (
            item
            for item in list(state.work_package_planning.work_packages or [])
            if str(item.work_package_id or "").strip() == str(work_package_id or "").strip()
        ),
        None,
    )
    if work_package is None:
        return {"interfaces": [], "surfaces": [], "obligations": [], "artifacts": []}
    inventories = dict(work_package.inventories or {})
    inventory_artifacts = [
        _normalize_repo_path(path)
        for path in list(inventories.get("artifact_inventory", []) or [])
        if _artifact_like(_normalize_repo_path(path))
    ]
    return {
        "interfaces": _dedupe(list(work_package.interface_contract or [])),
        "surfaces": _dedupe(list(inventories.get("implementation_surface_inventory", []) or [])),
        "obligations": _dedupe(list(work_package.method_obligations or []) + _inventory_obligations(inventories)),
        "artifacts": _dedupe(inventory_artifacts + _global_result_artifacts_for_owner(state, work_package_id)),
    }


def _scoped_work_package_contract_for_file(
    state: PaperBenchReproState | None,
    work_package_id: str,
    file_path: str,
) -> dict[str, list[str]]:
    contract = _work_package_contract_for_file(state, work_package_id)
    obligations = _file_scoped_obligations(file_path, list(contract["obligations"]))
    if "registry_shard" in _file_roles(file_path):
        obligations = _compact_registry_shard_obligations(file_path, obligations)
    return {
        "interfaces": _file_scoped_obligations(file_path, list(contract["interfaces"])),
        "surfaces": _file_scoped_surfaces(file_path, list(contract["surfaces"])),
        "obligations": obligations,
        "artifacts": list(contract["artifacts"]) if _file_owns_artifact_outputs(file_path) else [],
    }


def _compact_registry_shard_obligations(file_path: str, obligations: list[str]) -> list[str]:
    """Keep shard plans complete but avoid repeating huge cross-product matrix prose."""
    normalized = _normalize_repo_path(file_path).lower()
    compacted: list[str] = []
    replacement_added = False
    for obligation in list(obligations or []):
        text = str(obligation or "").strip()
        lowered = text.lower()
        is_cross_product_matrix = (
            len(text) > 900
            and (
                "obligation_matrix:" in lowered
                or (
                    " -> environments=" in lowered
                    and " -> methods=" in lowered
                    and " -> metrics=" in lowered
                )
            )
        )
        if is_cross_product_matrix:
            if not replacement_added and normalized == "src/experiment_registry.py":
                compacted.append(
                    "Build a code-visible evidence obligation matrix by binding experiment registry rows to environment, dataset, method, sweep, trend, metric, and artifact registries."
                )
                replacement_added = True
            continue
        if len(text) > 900 and "registry_shard" in _file_roles(file_path):
            continue
        compacted.append(text)
    return _dedupe(compacted)


def _plan_node_roles(node: object) -> set[str]:
    return _text_roles(
        " ".join(
            [
                str(getattr(node, "name", "") or ""),
                str(getattr(node, "description", "") or ""),
                str(getattr(node, "reusable_module", "") or ""),
                str(getattr(node, "insight", "") or ""),
                str(getattr(node, "hypothesis", "") or ""),
                str(getattr(node, "decision_value", "") or ""),
                str(getattr(node, "stop_rule_or_pruning_rationale", "") or ""),
            ]
        )
    )


def _node_relevant_to_file(file_path: str, node: object) -> bool:
    roles = _file_roles(file_path)
    if not roles:
        return True
    node_roles = _plan_node_roles(node)
    if "entrypoint" in roles:
        return bool(node_roles.intersection({"entrypoint"})) or str(getattr(node, "level", "") or "").strip().lower() == "experiment"
    if roles.intersection({"doc", "packaging"}):
        return False
    if not node_roles:
        return True
    if roles.intersection(node_roles):
        return True
    if "experiment" in roles and node_roles.intersection({"evaluation", "artifact", "baseline", "config", "environment", "data"}):
        return True
    if "evaluation" in roles and node_roles.intersection({"experiment", "artifact", "baseline", "data", "environment"}):
        return True
    if "artifact" in roles and node_roles.intersection({"experiment", "evaluation"}):
        return True
    if "config" in roles and node_roles.intersection({"experiment", "environment", "data", "method", "training", "baseline", "refinement"}):
        return True
    if "test" in roles:
        return True
    return False


def _related_node_ids_for_file(
    file_path: str,
    related_node_ids: list[str],
    plan_node_by_id: dict[str, object],
) -> list[str]:
    return _dedupe(
        [
            node_id
            for node_id in list(related_node_ids or [])
            if node_id in plan_node_by_id and _node_relevant_to_file(file_path, plan_node_by_id[node_id])
        ]
    )


def _work_package_decision_contract(state: PaperBenchReproState | None, work_package_id: str) -> dict[str, str]:
    """Return decision-value metadata for one work package."""
    empty = {"hypothesis": "", "decision_value": "", "stop_rule_or_pruning_rationale": ""}
    if state is None or state.work_package_planning is None:
        return empty
    work_package = next(
        (
            item
            for item in list(state.work_package_planning.work_packages or [])
            if str(item.work_package_id or "").strip() == str(work_package_id or "").strip()
        ),
        None,
    )
    if work_package is None:
        return empty
    goal = str(getattr(work_package, "goal", "") or "").strip()
    obligations = _dedupe([str(item) for item in list(getattr(work_package, "method_obligations", []) or [])])[:3]
    owned_units = _dedupe([str(item) for item in list(getattr(work_package, "owned_unit_ids", []) or [])])[:4]
    produced = _dedupe([str(item) for item in list(getattr(work_package, "produces", []) or [])])[:4]
    obligation_text = "; ".join(obligations)
    return {
        "hypothesis": str(getattr(work_package, "hypothesis", "") or "").strip()
        or (
            f"Implementing `{work_package.work_package_id}` should satisfy the paper obligation"
            + (f" `{goal}`" if goal else "")
            + (f" through {obligation_text}." if obligation_text else ".")
        ),
        "decision_value": str(getattr(work_package, "decision_value", "") or "").strip()
        or (
            f"This package determines coverage for {', '.join(owned_units) if owned_units else work_package.work_package_id}"
            + (f" and materializes {', '.join(produced)}." if produced else ".")
        ),
        "stop_rule_or_pruning_rationale": "Implementation scope: paper-specified protocol, registry, artifact, smoke-test surfaces, and bounded default execution.",
    }


def _work_package_reference_ids(state: PaperBenchReproState | None, work_package_id: str) -> list[str]:
    if state is None or state.work_package_planning is None:
        return []
    owner = str(work_package_id or "").strip()
    if not owner:
        return []
    valid_refs = {
        str(getattr(survey, "ref_id", "") or "").strip()
        for survey in list(getattr(state, "reference_repo_surveys", []) or [])
        if str(getattr(survey, "ref_id", "") or "").strip()
    }
    if not valid_refs:
        return []
    for work_package in list(state.work_package_planning.work_packages or []):
        if str(work_package.work_package_id or "").strip() == owner:
            refs = _dedupe([str(ref_id) for ref_id in list(work_package.reference_ids or [])])
            return [ref_id for ref_id in refs if ref_id in valid_refs]
    return []


def _work_package_owned_unit_ids(state: PaperBenchReproState | None, work_package_id: str) -> list[str]:
    if state is None or state.work_package_planning is None:
        return []
    owner = str(work_package_id or "").strip()
    if not owner:
        return []
    for work_package in list(state.work_package_planning.work_packages or []):
        if str(work_package.work_package_id or "").strip() == owner:
            return _dedupe([str(unit_id) for unit_id in list(work_package.owned_unit_ids or [])])
    return []


def _units_by_id(state: PaperBenchReproState | None) -> dict[str, object]:
    if state is None or state.unit_extraction is None:
        return {}
    return {
        str(getattr(unit, "unit_id", "") or "").strip(): unit
        for unit in list(state.unit_extraction.units or [])
        if str(getattr(unit, "unit_id", "") or "").strip()
    }


def _scope_boundary_for_file(
    state: PaperBenchReproState | None,
    work_package_id: str,
    owned_unit_ids: list[str],
    *,
    file_path: str,
    focus: list[str],
) -> dict[str, list[str]]:
    package_scope: dict[str, list[str]] = {}
    if state is not None and state.work_package_planning is not None:
        for work_package in list(state.work_package_planning.work_packages or []):
            if str(getattr(work_package, "work_package_id", "") or "").strip() == str(work_package_id or "").strip():
                package_scope = dict(getattr(work_package, "scope_boundary", {}) or {})
                break
    unit_map = _units_by_id(state)
    preserve = list(package_scope.get("preserve", []) or [])
    implementation_focus = list(package_scope.get("implementation_focus", []) or [])
    for unit_id in owned_unit_ids:
        unit = unit_map.get(str(unit_id or "").strip())
        if unit is None:
            continue
        obligations = _dedupe(
            _implementation_obligation_projection(
                list(getattr(unit, "code_obligations", []) or [])
                + list(getattr(unit, "implementation_notes", []) or [])
            )
            + [str(item) for item in list(getattr(unit, "runtime_interfaces", []) or [])]
            + [str(item) for item in list(getattr(unit, "expected_artifacts", []) or [])]
        )
        if obligations:
            preserve.append(f"{unit_id}: " + "; ".join(obligations[:5]))
        implementation_focus.extend(str(item) for item in list(getattr(unit, "implementation_surfaces", []) or []))
        implementation_focus.extend(str(item) for item in list(getattr(unit, "runtime_interfaces", []) or []))
        implementation_focus.extend(str(item) for item in list(getattr(unit, "expected_artifacts", []) or []))
    implementation_focus.extend(focus)
    return sanitize_scope_boundary(
        {
            "preserve": _dedupe(preserve)
            or [f"{file_path}: implement the owned paper/addendum obligations as active code."],
            "implementation_focus": _dedupe(implementation_focus)
            or [f"{file_path}: active implementation route and artifact writer."],
        }
    )


def _implementation_obligation_projection(items: list[object], *, limit: int = 6) -> list[str]:
    """Return unit obligations that describe implementation work instead of planning policy."""
    projected: list[str] = []
    direct_action_terms = (
        "implement",
        "write",
        "compute",
        "measure",
        "route",
        "artifact",
        "metric",
        "checkpoint",
        "config",
        "interface",
        "writer",
        "loader",
        "factory",
        "results/",
        "checkpoints/",
        "实现",
        "写出",
        "计算",
        "度量",
        "指标",
        "配置",
        "接口",
        "路径",
        "产物",
    )
    bookkeeping_anchor_terms = (
        "method",
        "configuration",
        "checkpoint",
        "metric",
        "code path",
        "dry-run",
        "方法",
        "配置",
        "指标",
        "路径",
        "代码",
    )
    for raw in list(items or []):
        text = " ".join(str(raw or "").strip().split())
        if not text:
            continue
        lowered = text.lower()
        has_direct_action = any(token in lowered for token in direct_action_terms)
        has_bookkeeping_route = (
            any(token in lowered for token in ("record", "记录"))
            and any(token in lowered for token in bookkeeping_anchor_terms)
        )
        has_callable_symbol = bool(re.search(r"\b[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*\b", text))
        if has_direct_action or has_bookkeeping_route or has_callable_symbol:
            projected.append(text)
    return _dedupe(projected)[:limit]


def _known_unit_ids(state: PaperBenchReproState | None) -> set[str]:
    if state is None or state.unit_extraction is None:
        return set()
    return {
        str(getattr(unit, "unit_id", "") or "").strip()
        for unit in list(state.unit_extraction.units or [])
        if str(getattr(unit, "unit_id", "") or "").strip()
    }


def _filter_known_unit_ids(state: PaperBenchReproState | None, unit_ids: list[str]) -> list[str]:
    values = _dedupe([str(unit_id) for unit_id in list(unit_ids or [])])
    known = _known_unit_ids(state)
    if not known:
        return values
    return [unit_id for unit_id in values if unit_id in known]


def _known_work_package_ids(state: PaperBenchReproState | None) -> set[str]:
    if state is None or state.work_package_planning is None:
        return set()
    return {
        str(item.work_package_id or "").strip()
        for item in list(state.work_package_planning.work_packages or [])
        if str(item.work_package_id or "").strip()
    }


def _filter_reference_ids_for_state(state: PaperBenchReproState | None, reference_ids: list[str]) -> list[str]:
    refs = _dedupe([str(ref_id) for ref_id in list(reference_ids or [])])
    if state is None:
        return refs
    valid_refs = {
        str(getattr(survey, "ref_id", "") or "").strip()
        for survey in list(getattr(state, "reference_repo_surveys", []) or [])
        if str(getattr(survey, "ref_id", "") or "").strip()
    }
    if not valid_refs:
        return []
    return [ref_id for ref_id in refs if ref_id in valid_refs]


def _canonical_work_package_id_for_path(
    file_path: str,
    *,
    package_by_path: dict[str, str],
    proposed_work_package_id: str = "",
    state: PaperBenchReproState | None = None,
) -> str:
    known_ids = _known_work_package_ids(state)
    architecture_owner = str(package_by_path.get(file_path, "") or "").strip()
    proposed = str(proposed_work_package_id or "").strip()
    if not known_ids:
        return proposed or architecture_owner
    if architecture_owner in known_ids:
        return architecture_owner
    if proposed in known_ids:
        return proposed
    return ""


def _append_code_contract_prompt(
    base_prompt: str,
    *,
    surfaces: list[str],
    obligations: list[str],
    defines: list[str] | None = None,
    calls: list[str] | None = None,
) -> str:
    parts = [sanitize_contract_text(str(base_prompt or "").strip())]
    if surfaces:
        parts.append("Implementation surfaces: " + " | ".join(surfaces[:8]))
    if obligations:
        parts.append("Unit-derived method obligations: " + " | ".join(sanitize_contract_list(obligations, field="method_obligations")[:8]))
        parts.append(
            "Generation acceptance contract: every paper-visible formula, numeric anchor, hyperparameter default, algorithm step, dataset/model route, metric aggregation, and table/figure writer named above must be represented in executable/importable code or config and reached by the canonical smoke/full route."
        )
    if defines:
        parts.append(
            "Active route contract - define these public symbols/classes/functions in this file: "
            + " | ".join(_dedupe([str(item) for item in defines])[:12])
        )
    if calls:
        parts.append(
            "Active route contract - import/call/wire these symbols from executable routes, not just comments or registry text: "
            + " | ".join(_dedupe([str(item) for item in calls])[:12])
        )
    return sanitize_contract_text("\n".join(part for part in parts if part))


def _identifier_stem(file_path: str) -> str:
    stem = _normalize_repo_path(file_path).rsplit("/", 1)[-1].rsplit(".", 1)[0]
    cleaned = "".join(char if char.isalnum() else "_" for char in stem).strip("_").lower()
    return cleaned or "module"


def _class_name_from_stem(stem: str) -> str:
    parts = [part for part in stem.split("_") if part]
    return "".join(part[:1].upper() + part[1:] for part in parts) or "Module"


def _symbol_slug(text: str, *, prefix: str = "") -> str:
    tokens = re.findall(r"[A-Za-z0-9]+", str(text or "").lower())
    if not tokens:
        return ""
    slug = "_".join(tokens[:8])
    if slug and slug[0].isdigit():
        slug = f"{prefix or 'route'}_{slug}"
    if prefix and not slug.startswith(prefix + "_"):
        slug = f"{prefix}_{slug}"
    return slug


def _artifact_writer_symbol(path_or_label: str) -> str:
    normalized = _normalize_repo_path(path_or_label)
    basename = normalized.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    if not basename:
        basename = str(path_or_label or "")
    slug = _symbol_slug(basename)
    return f"write_{slug}_artifact" if slug else ""


def _figure_table_route_symbols(text: str) -> list[str]:
    symbols: list[str] = []
    lowered = str(text or "").lower()
    for kind, number in re.findall(r"\b(figure|fig|table)\.?\s*([0-9]+[a-z]?)\b", lowered):
        normalized_kind = "figure" if kind in {"fig", "figure"} else "table"
        symbols.append(f"write_{normalized_kind}_{number}_artifact")
        symbols.append(f"run_{normalized_kind}_{number}_route")
    for token in re.findall(r"\b(?:figure|fig|table)_[0-9]+[a-z]?\b", lowered):
        normalized = token.replace("fig_", "figure_")
        symbols.append(f"write_{normalized}_artifact")
        symbols.append(f"run_{normalized}_route")
    return _dedupe(symbols)


def _high_signal_symbol_terms(text: str) -> bool:
    lowered = str(text or "").lower().replace("_", " ").replace("-", " ")
    return any(
        token in lowered
        for token in (
            "metric",
            "score",
            "fidelity",
            "accuracy",
            "rouge",
            "loss",
            "objective",
            "gradient",
            "sampler",
            "diffusion",
            "mask",
            "embedding",
            "adapter",
            "baseline",
            "factory",
            "dataset",
            "environment",
            "simulator",
            "training",
            "evaluation",
            "artifact",
            "figure",
            "table",
        )
    )




def _has_symbol_signal(lowered: str, required: tuple[str, ...]) -> bool:
    return all(token in lowered for token in required)


_SYMBOL_CANDIDATE_STOPWORDS = {
    "and",
    "for",
    "from",
    "with",
    "without",
    "into",
    "paper",
    "method",
    "model",
    "dataset",
    "table",
    "figure",
    "appendix",
    "section",
    "results",
    "experiment",
    "experiments",
    "implementation",
    "interface",
    "interfaces",
    "protocol",
    "baseline",
    "baselines",
    "backed",
    "complete",
    "concrete",
    "executable",
    "factories",
    "factory",
    "here",
    "include",
    "implemented",
    "include",
    "must",
    "selector",
    "selectors",
    "set",
    "should",
    "visible",
}

_SYMBOL_META_TOKENS = {
    "backed",
    "complete",
    "concrete",
    "executable",
    "factories",
    "factory",
    "here",
    "include",
    "implemented",
    "must",
    "selector",
    "selectors",
}


def _is_valid_public_symbol_hint(symbol: str) -> bool:
    rendered = str(symbol or "").strip()
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", rendered):
        return False
    lowered = rendered.lower()
    if "heremustbeimplementedas" in lowered:
        return False
    parts = [part for part in re.split(r"_+", lowered) if part]
    if any(part in _SYMBOL_META_TOKENS for part in parts):
        return False
    if len(parts) > 6 and not rendered.startswith(("DEFAULT_", "compute_", "aggregate_", "resolve_", "write_", "run_")):
        return False
    return True


def _symbol_identifier_from_label(label: str, *, style: str = "snake", suffix: str = "") -> str:
    tokens = [
        token
        for token in re.findall(r"[A-Za-z][A-Za-z0-9]*", str(label or ""))
        if token.lower() not in _SYMBOL_CANDIDATE_STOPWORDS
    ]
    if not tokens:
        return ""
    if style == "class":
        rendered = "".join(token[:1].upper() + token[1:] for token in tokens[:5])
    else:
        rendered = "_".join(token.lower() for token in tokens[:6])
    if suffix and rendered and not rendered.lower().endswith(suffix.lower()):
        rendered = f"{rendered}{suffix}"
    return rendered


def _evidence_symbol_candidates(text: str) -> list[str]:
    """Derive public symbol names from paper/unit evidence without benchmark-specific tokens."""
    raw = _strip_meta_contract_lines(str(text or ""))
    candidates: list[str] = []
    for token in re.findall(r"`([A-Za-z_][A-Za-z0-9_]{2,})`", raw):
        candidates.append(token)
    for token in re.findall(r"\b[A-Z][A-Za-z0-9]+(?:[A-Z][A-Za-z0-9]+)+\b", raw):
        candidates.append(token)
    for token in re.findall(r"\b[A-Z][A-Z0-9]{1,}(?:[-_/][A-Z0-9]{1,})*\b", raw):
        normalized = token.replace("-", "_").replace("/", "_")
        if normalized.lower() not in _SYMBOL_CANDIDATE_STOPWORDS:
            candidates.append(normalized)
    for phrase in re.findall(
        r"\b(?:called|named|method|algorithm|baseline|variant|model|task|environment|dataset)\s+([A-Za-z][A-Za-z0-9 _/-]{2,40})",
        raw,
        flags=re.IGNORECASE,
    ):
        identifier = _symbol_identifier_from_label(phrase, style="class")
        if identifier:
            candidates.append(identifier)
    return _dedupe([item for item in candidates if _is_valid_public_symbol_hint(item)])[:16]


def _strip_meta_contract_lines(text: str) -> str:
    meta_prefixes = (
        "executable anchor contract:",
        "executable algorithm contract:",
        "executable artifact contract:",
        "canonical route contract:",
        "generation acceptance contract:",
        "executable-route acceptance:",
        "full experiment-matrix route contract:",
    )
    return "\n".join(
        line
        for line in str(text or "").splitlines()
        if not any(prefix in line.lower() for prefix in meta_prefixes)
    )


def _symbol_source_obligations(obligations: list[str]) -> list[str]:
    return [
        stripped
        for item in list(obligations or [])
        for stripped in [" ".join(str(item or "").strip().split())]
        if stripped and _strip_meta_contract_lines(stripped).strip()
    ]


def _executable_anchor_symbol_hints(text: str) -> list[str]:
    """Synthesize default/formula/metric symbols from paper-derived obligation text."""
    raw = _strip_meta_contract_lines(str(text or ""))
    lowered = raw.lower().replace("-", " ")
    symbols: list[str] = []
    parameter_aliases = {
        "learning_rate": ("learning rate", "learning_rate", "lr"),
        "weight_decay": ("weight decay", "weight_decay"),
        "batch_size": ("batch size", "batch_size"),
        "epochs": ("epoch", "epochs"),
        "seed": ("seed", "seeds"),
        "temperature": ("temperature",),
        "alpha": ("alpha",),
        "beta": ("beta",),
        "gamma": ("gamma",),
        "epsilon": ("epsilon",),
        "lambda": ("lambda",),
        "sigma_min": ("sigma_min", "sigma min"),
        "sigma_max": ("sigma_max", "sigma max"),
        "hidden_dim": ("hidden dim", "hidden_dim", "hidden size"),
        "token_dim": ("token dim", "token_dim", "token dimension"),
        "num_layers": ("num layers", "number of layers", "layers"),
        "num_steps": ("num steps", "steps", "iterations"),
    }
    for slug, aliases in parameter_aliases.items():
        if any(alias in lowered for alias in aliases):
            symbols.extend([f"DEFAULT_{slug.upper()}", f"resolve_{slug}_defaults"])
            if any(token in lowered for token in ("sweep", "values", "grid", "vary", "varied", "varies")):
                symbols.append(f"{slug}_values")
    for name in re.findall(r"\b([A-Za-z][A-Za-z0-9_]{1,40})\s*(?:=|:)\s*[-+]?\d", raw):
        slug = _symbol_slug(name)
        if slug and slug not in _SYMBOL_CANDIDATE_STOPWORDS:
            symbols.append(f"DEFAULT_{slug.upper()}")
    metric_aliases = {
        "accuracy": ("accuracy", "top-1", "top1"),
        "loss": ("loss", "objective"),
        "reward": ("reward", "return"),
        "f1": ("f1", "f1 score"),
        "auc": ("auc", "auroc"),
        "c2st": ("c2st", "two-sample", "two sample"),
        "nll": ("nll", "negative log"),
        "mse": ("mse", "mean squared"),
        "mae": ("mae", "mean absolute"),
        "correlation": ("spearman", "pearson", "kendall"),
    }
    for slug, aliases in metric_aliases.items():
        if any(alias in lowered for alias in aliases):
            symbols.extend([f"compute_{slug}", f"aggregate_{slug}"])
    if any(token in lowered for token in ("formula", "objective", "loss", "gradient", "likelihood", "posterior", "score")):
        evidence_slug = _symbol_slug(" ".join(_evidence_symbol_candidates(raw)[:3]))
        if evidence_slug:
            symbols.extend([f"compute_{evidence_slug}_objective", f"compute_{evidence_slug}_score"])
    return _dedupe([symbol for symbol in symbols if _is_valid_public_symbol_hint(symbol)])[:16]


def _file_call_hints(
    file_path: str,
    *,
    surfaces: list[str],
    obligations: list[str],
    writes_artifacts: list[str],
    defines: list[str],
) -> list[str]:
    """Infer active route calls so plan does not leave helpers disconnected."""
    normalized = _normalize_repo_path(file_path).lower()
    if not normalized.endswith(".py") or normalized.endswith("__init__.py"):
        return []
    text = " ".join([normalized, *surfaces, *obligations, *writes_artifacts, *defines]).lower()
    stem = _identifier_stem(normalized)
    calls: list[str] = []
    is_entry = normalized == "main.py" or normalized.endswith(("/main.py", "/cli.py", "run_experiments.py"))
    is_experiment = any(token in normalized for token in ("experiment", "run_"))
    is_training = any(token in normalized for token in ("train", "training"))
    is_evaluation = any(token in normalized for token in ("eval", "evaluation", "metric"))
    is_artifact = any(token in normalized for token in ("artifact", "report", "plot", "figure"))
    anchor_symbols = _executable_anchor_symbol_hints(text)
    metric_slug_candidates = []
    if "fidelity" in text:
        metric_slug_candidates.extend(["compute_fidelity_score", "aggregate_fidelity_score", "write_fidelity_score_artifact"])
    if "accuracy" in text:
        metric_slug_candidates.extend(["compute_accuracy", "aggregate_accuracy"])
    if metric_slug_candidates and (is_entry or is_experiment or is_evaluation or is_artifact):
        calls.extend(metric_slug_candidates)
    calls.extend(
        symbol
        for symbol in anchor_symbols
        if symbol.startswith(("resolve_", "compute_", "aggregate_"))
    )

    if is_entry:
        calls.extend(["run_from_config", "run_experiment"])
    if is_experiment or is_entry:
        calls.extend(["load_inputs", "run_evaluation", "write_named_result_artifacts"])
        calls.append(f"run_{stem}")
    if is_training:
        calls.extend(["run_training_loop", "compute_training_objective"])
        calls.append(f"train_{stem}")
    if is_evaluation:
        calls.extend(["compute_metrics", "aggregate_metrics", "write_named_result_artifacts"])
        calls.append(f"evaluate_{stem}")
    if is_artifact:
        calls.extend(["write_json_artifact", "write_artifact_manifest", "write_summary_report"])
        calls.extend(_artifact_writer_symbol(path) for path in writes_artifacts[:16])

    evidence_symbols = _evidence_symbol_candidates(text)
    evidence_slug = _symbol_slug(" ".join(evidence_symbols[:3]))
    if evidence_slug and (is_entry or is_experiment):
        calls.extend([f"run_{evidence_slug}_experiment", f"evaluate_{evidence_slug}"])
    if evidence_slug and is_training:
        calls.append(f"train_{evidence_slug}")
    if evidence_slug and is_evaluation:
        calls.append(f"compute_{evidence_slug}_metrics")
    if is_entry or is_experiment or is_training or is_evaluation:
        calls.extend(symbol for symbol in evidence_symbols[:8] if symbol not in set(defines))
    if writes_artifacts and (is_entry or is_experiment or is_evaluation or is_artifact):
        calls.extend(_artifact_writer_symbol(path) for path in writes_artifacts[:16])
    calls.extend(_figure_table_route_symbols(text)[:16])
    limit = 48 if is_entry else 32
    return _dedupe([item for item in calls if item])[:limit]


def _generic_role_symbol_hints(file_path: str, roles: set[str], evidence_symbols: list[str]) -> list[str]:
    normalized = _normalize_repo_path(file_path).lower()
    stem = _identifier_stem(normalized)
    class_name = _class_name_from_stem(stem)
    symbols: list[str] = []

    registry_symbols: dict[str, list[str]] = {
        "src/experiment_registry.py": ["ExperimentSpec", "EXPERIMENT_REGISTRY", "build_evidence_obligation_matrix"],
        "src/environment_registry.py": ["EnvironmentSpec", "ENVIRONMENT_REGISTRY", "get_environment_spec"],
        "src/dataset_registry.py": ["DatasetSpec", "DATASET_REGISTRY", "get_dataset_spec"],
        "src/method_registry.py": ["MethodSpec", "METHOD_REGISTRY", "get_method_spec"],
        "src/sweep_registry.py": ["SweepSpec", "PARAMETER_SWEEPS", "get_sweep_values"],
        "src/artifact_contract.py": ["ArtifactSpec", "ARTIFACT_CONTRACT", "write_artifact_manifest"],
        "src/trend_assertions.py": ["TrendAssertion", "TREND_ASSERTIONS", "validate_expected_trends"],
    }
    symbols.extend(registry_symbols.get(normalized, []))

    if roles.intersection({"method", "model", "agent", "baseline", "refinement", "explainer"}):
        symbols.extend(evidence_symbols[:8])
        symbols.extend([f"{class_name}Config", f"build_{stem}"])
    if "environment" in roles:
        symbols.extend([f"{class_name}Spec", f"make_{stem}", f"check_{stem}_available"])
    if "data" in roles:
        symbols.extend([f"{class_name}Spec", f"load_{stem}", f"prepare_{stem}"])
    if "training" in roles:
        symbols.extend([f"{class_name}Config", f"train_{stem}", "run_training_loop"])
    if "evaluation" in roles:
        symbols.extend([f"{class_name}Result", f"evaluate_{stem}", f"compute_{stem}_metrics", "aggregate_metrics"])
    if "experiment" in roles:
        symbols.extend([f"{class_name}Spec", f"run_{stem}", "run_experiment"])
    if "artifact" in roles or "plotting" in roles or "reporting" in roles:
        symbols.extend([f"{class_name}Layout", f"write_{stem}_artifact", "write_artifact_manifest"])
    if "config" in roles:
        symbols.extend([f"{class_name}Config", f"load_{stem}_config"])
    if "test" in roles:
        symbols.extend([f"test_{stem}_contract"])
    return _dedupe(symbols)


def _file_symbol_hints(file_path: str, *, surfaces: list[str], obligations: list[str]) -> list[str]:
    """Infer concrete public symbols from generic implementation surfaces."""
    normalized = _normalize_repo_path(file_path).lower()
    if not normalized.endswith(".py"):
        return []
    if normalized.endswith("__init__.py"):
        return ["__all__"]

    surface_text = " ".join([normalized, *surfaces]).lower()
    obligation_text = " ".join(obligations).lower()
    text = " ".join([surface_text, obligation_text])
    symbols: list[str] = []
    symbols.extend(_executable_anchor_symbol_hints(text))
    evidence_symbols = _evidence_symbol_candidates(text)
    roles = _file_roles(normalized)
    if normalized == "main.py" or normalized.endswith("/cli.py"):
        symbols.extend(["main", "parse_args", "run_from_config"])

    symbols.extend(_generic_role_symbol_hints(normalized, roles, evidence_symbols))
    evidence_slug = _symbol_slug(" ".join(evidence_symbols[:3]))
    if evidence_slug:
        if any(token in normalized for token in ("method", "model", "agent", "baseline", "refinement")):
            symbols.extend(evidence_symbols[:8])
            symbols.append(_symbol_identifier_from_label(evidence_slug, style="class", suffix="Config"))
        if any(token in normalized for token in ("train", "training")):
            symbols.append(f"train_{evidence_slug}")
        if any(token in normalized for token in ("eval", "evaluation", "metric")):
            symbols.append(f"evaluate_{evidence_slug}")
            symbols.append(f"compute_{evidence_slug}_metrics")
        if any(token in normalized for token in ("experiment", "run")):
            symbols.append(f"run_{evidence_slug}_experiment")

    if any(token in surface_text for token in ("environment", "simulator", "task", "env_factory", "dataset")) or "env" in normalized:
        symbols.extend(_generic_role_symbol_hints(normalized, {"environment"}, evidence_symbols))
    if any(token in surface_text for token in ("policy", "agent", "model", "network", "checkpoint")):
        symbols.extend(_generic_role_symbol_hints(normalized, {"model"}, evidence_symbols))
    if any(token in surface_text for token in ("train", "training", "pretrain", "optimizer")):
        symbols.extend(_generic_role_symbol_hints(normalized, {"training"}, evidence_symbols))
    if any(token in surface_text for token in ("evaluate", "evaluation", "metric", "reward", "table", "fidelity")):
        symbols.extend(_generic_role_symbol_hints(normalized, {"evaluation"}, evidence_symbols))
    if any(token in surface_text for token in ("artifact", "report", "result", "figure", "summary")):
        symbols.extend(_generic_role_symbol_hints(normalized, {"artifact"}, evidence_symbols))
    if any(token in surface_text for token in ("baseline", "ablation", "compare", "comparison", "variant")):
        symbols.extend(_generic_role_symbol_hints(normalized, {"baseline"}, evidence_symbols))
    if _has_refinement_semantic_signal(surface_text):
        symbols.extend(_generic_role_symbol_hints(normalized, {"refinement"}, evidence_symbols))
    if any(token in surface_text for token in ("experiment", "protocol", "matrix")):
        symbols.extend(_generic_role_symbol_hints(normalized, {"experiment"}, evidence_symbols))
    if any(token in surface_text for token in ("explain", "explanation", "mask")):
        symbols.extend(_generic_role_symbol_hints(normalized, {"explainer"}, evidence_symbols))
    if any(token in surface_text for token in ("data", "dataset", "loader")):
        symbols.extend(_generic_role_symbol_hints(normalized, {"data"}, evidence_symbols))
    if "fidelity" in text:
        symbols.extend(["compute_fidelity_score", "aggregate_fidelity_score", "write_fidelity_score_artifact"])
    if any(token in text for token in ("fig. 4", "figure 4", "figure_4", "table 4")):
        symbols.extend(["write_figure_4_artifact", "run_figure_4_route", "write_table_4_artifact", "run_table_4_route"])
    if not symbols and any(token in obligation_text for token in ("evaluate", "evaluation", "metric", "reward", "fidelity")):
        symbols.extend(_generic_role_symbol_hints(normalized, {"evaluation"}, evidence_symbols))
    if not symbols and any(token in obligation_text for token in ("train", "training", "pretrain")):
        symbols.extend(_generic_role_symbol_hints(normalized, {"training"}, evidence_symbols))
    if not symbols and any(token in obligation_text for token in ("environment", "simulator", "dataset")):
        symbols.extend(_generic_role_symbol_hints(normalized, {"environment"}, evidence_symbols))
    if not symbols and normalized.startswith("src/"):
        stem = _identifier_stem(normalized)
        symbols.extend([f"{_class_name_from_stem(stem)}Config", f"build_{stem}"])
    return _dedupe(symbols)[:16]


def _bench_visible_contract_obligations(file_path: str) -> list[str]:
    """Add generic PaperBench-facing surfaces without encoding a paper/case template."""
    roles = _file_roles(file_path)
    obligations: list[str] = []
    if "doc" in roles:
        obligations.append(
            "Document the canonical run path, environment/readiness expectations, configuration surfaces, named protocol coverage, and declared output artifacts."
        )
    if "packaging" in roles:
        obligations.append(
            "Declare lightweight core dependencies separately from optional heavy simulator/training dependencies when possible."
        )
    if roles.intersection({"config", "experiment"}):
        obligations.append(
            "Expose a paper-derived configuration contract with task/environment factories, callable method/baseline selectors, seeds, hyperparameter defaults/constants, output paths, and active reproduction scope notes."
        )
    if roles.intersection({"entrypoint", "experiment", "training", "evaluation", "artifact"}):
        obligations.append(
            "Provide a dry-run or runtime-smoke mode for cost control as a bounded execution of the same concrete implementation route, with paper-visible results backed by measured code paths."
        )
    if roles.intersection({"experiment", "evaluation", "artifact", "test"}):
        obligations.append(
            "Make named experiment protocols statically discoverable as callable matrices connecting tasks, methods, metric functions, and artifact writer paths."
        )
    if roles.intersection({"artifact", "evaluation", "test"}):
        obligations.append(
            "Expose artifact layout helpers or constants for metrics, tables, figures, config snapshots, run manifests, and reports so static review can find output contracts."
        )
    if roles.intersection({"environment", "data"}):
        obligations.append(
            "Represent external environments or datasets through import-light descriptors/factories with clear availability checks and faithful fallback errors."
        )
    return obligations


def _project_file_plans_from_architecture(
    architecture: ArchitectureOutput,
    pipeline_plan: PipelinePlanOutput,
    state: PaperBenchReproState | None = None,
) -> PackageFilePlanningOutput:
    """Project architecture into deterministic package-scoped file plans."""
    blueprint_by_path = {item.path: item for item in architecture.file_blueprints}
    plan_node_by_id = {item.node_id: item for item in pipeline_plan.plan_nodes}
    dependency_map = _dependency_map(architecture)
    package_by_path = _projection_owner_map(architecture, state)
    target_paths = _projection_paths(architecture, state)
    file_plans: list[RepoFilePlan] = []

    for index, file_path in enumerate(target_paths, start=1):
        blueprint = blueprint_by_path.get(file_path)
        related_node_ids = _related_node_ids_for_file(
            file_path,
            _dedupe(list(blueprint.related_node_ids if blueprint else [])),
            plan_node_by_id,
        )
        related_nodes = [plan_node_by_id[node_id] for node_id in related_node_ids if node_id in plan_node_by_id]
        work_package_id = package_by_path.get(file_path, "")
        reference_ids = _dedupe(
            list(blueprint.based_on_references if blueprint else [])
            + [node.ref_id for node in related_nodes if node.ref_id]
            + _work_package_reference_ids(state, work_package_id)
        )
        reference_ids = _filter_reference_ids_for_state(state, reference_ids)
        owned_unit_ids = _filter_known_unit_ids(state, _work_package_owned_unit_ids(state, work_package_id))
        code_contract = _scoped_work_package_contract_for_file(state, work_package_id, file_path)
        decision_contract = _work_package_decision_contract(state, work_package_id)
        file_inventories = _generation_inventories_for_file(state, work_package_id, file_path)
        file_obligations = _file_inventory_obligations(file_path, file_inventories)
        contract_artifact_paths = _paper_artifact_paths(_contract_terms(_state_evidence_contract(state), "artifacts")) if state is not None and _file_owns_artifact_outputs(file_path) else []
        artifact_outputs = _dedupe(
            list(code_contract["artifacts"])
            + contract_artifact_paths
            + (_artifact_inventory_paths(file_inventories) if _file_owns_artifact_outputs(file_path) else [])
        )
        artifact_outputs = _file_scoped_artifact_outputs(file_path, artifact_outputs)
        method_obligations = _dedupe(
            _priority_evidence_contract_obligations(state, file_path)
            + _source_artifact_context_obligations_for_file(state, file_path)
            + file_obligations
            + _global_measurement_obligations_for_file(state, file_path)
            + _global_result_target_obligations_for_file(state, work_package_id, file_path)
            + _state_evidence_contract_obligations(state, file_path)
            + _formula_algorithm_obligations_for_file(state, file_path)
            + _addendum_obligations_for_file(state, file_path)
            + list(code_contract["obligations"])
            + _bench_visible_contract_obligations(file_path)
        )
        base_method_obligations = sanitize_contract_list(
            _file_scoped_obligations(file_path, method_obligations),
            field="method_obligations",
        )
        symbol_obligations = _symbol_source_obligations(base_method_obligations)
        executable_obligations = _file_scoped_obligations(
            file_path,
            _executable_contract_obligations(file_path, symbol_obligations),
        )
        method_obligations = sanitize_contract_list(
            _dedupe(base_method_obligations + executable_obligations),
            field="method_obligations",
        )
        symbol_names = _dedupe(
            [node.name for node in related_nodes if getattr(node, "name", "").strip()]
            + _file_symbol_hints(
                file_path,
                surfaces=list(code_contract["surfaces"]),
                obligations=symbol_obligations,
            )
        )
        call_symbols = _file_call_hints(
            file_path,
            surfaces=list(code_contract["surfaces"]),
            obligations=symbol_obligations,
            writes_artifacts=artifact_outputs,
            defines=symbol_names,
        )
        scope_boundary = _scope_boundary_for_file(
            state,
            work_package_id,
            owned_unit_ids,
            file_path=file_path,
            focus=[
                *list(code_contract["surfaces"]),
                *artifact_outputs[:8],
                *symbol_names[:8],
                *call_symbols[:8],
            ],
        )
        review_points = [
            f"Implement the contract-owned responsibility for {file_path}.",
            "Keep dependency wiring and package interfaces stable.",
            "Do not violate declared artifact or execution closure.",
            *[
                f"Active route contract: define `{symbol}` in {file_path}."
                for symbol in symbol_names[:8]
            ],
            *[
                f"Active route contract: wire/call `{symbol}` from {file_path} or a downstream executable route."
                for symbol in call_symbols[:8]
            ],
            *[f"Satisfy file-scoped method obligation: {item}" for item in method_obligations[:6]],
        ]
        file_plans.append(
            RepoFilePlan(
                target_file=file_path,
                task_id=f"task_{index:03d}",
                work_package_id=work_package_id,
                purpose=(blueprint.purpose if blueprint and blueprint.purpose else f"Implement {file_path}"),
                hypothesis=decision_contract["hypothesis"],
                decision_value=decision_contract["decision_value"],
                stop_rule_or_pruning_rationale=decision_contract["stop_rule_or_pruning_rationale"],
                scope_boundary=scope_boundary,
                related_node_ids=related_node_ids,
                owned_units=owned_unit_ids,
                reference_ids=reference_ids,
                depends_on=_dedupe(list(dependency_map.get(file_path, []))),
                blocking_dependencies=_dedupe(list(dependency_map.get(file_path, []))),
                requires_stable_dependencies=True,
                interface_contract=sanitize_contract_list(code_contract["interfaces"], field="interface_contract"),
                implementation_surfaces=code_contract["surfaces"],
                method_obligations=method_obligations,
                context_sources=_dedupe(
                    [f"unit:{unit_id}" for unit_id in owned_unit_ids]
                    + [f"node:{node_id}" for node_id in related_node_ids]
                    + [f"ref:{ref_id}" for ref_id in reference_ids]
                ),
                consumes=_dedupe(list(dependency_map.get(file_path, []))),
                produces=[file_path],
                defines_symbols=symbol_names,
                calls_symbols=call_symbols,
                writes_artifacts=artifact_outputs,
                reads_artifacts=[],
                allowed_scope={"write": [file_path], "read": _dedupe(list(dependency_map.get(file_path, [])))},
                generation_prompt=_append_code_contract_prompt(
                    f"Implement `{file_path}` according to its package contract and repository execution closure.",
                    surfaces=code_contract["surfaces"],
                    obligations=method_obligations,
                    defines=symbol_names,
                    calls=call_symbols,
                ),
                validation_hooks=["python_syntax" if file_path.endswith(".py") else "file_exists"],
                review_points=sanitize_contract_list(
                    _dedupe(review_points + [f"Preserve artifact output: {artifact}" for artifact in artifact_outputs[:8]]),
                    field="review_points",
                ),
            )
        )

    return PackageFilePlanningOutput(
        file_plans=file_plans,
        planning_notes=["Projected deterministic file plans from architecture before package-level closure."],
    )


def _close_package_file_plans(
    architecture: ArchitectureOutput,
    pipeline_plan: PipelinePlanOutput,
    file_planning: PackageFilePlanningOutput,
    state: PaperBenchReproState | None = None,
) -> PackageFilePlanningOutput:
    """Deterministically close missing fields against architecture and plan bindings."""
    blueprint_by_path = {item.path: item for item in architecture.file_blueprints}
    plan_node_by_id = {item.node_id: item for item in pipeline_plan.plan_nodes}
    dependency_map = _dependency_map(architecture)
    package_by_path = _projection_owner_map(architecture, state)
    expected_paths = _projection_paths(architecture, state)
    closed: list[RepoFilePlan] = []
    seen_paths: set[str] = set()

    for index, item in enumerate(file_planning.file_plans, start=1):
        file_path = _normalize_repo_path(item.target_file)
        if not file_path or file_path in seen_paths:
            continue
        if expected_paths and file_path not in set(expected_paths):
            continue
        seen_paths.add(file_path)
        blueprint = blueprint_by_path.get(file_path)
        related_node_ids = _related_node_ids_for_file(
            file_path,
            _dedupe(list(item.related_node_ids) + list(blueprint.related_node_ids if blueprint else [])),
            plan_node_by_id,
        )
        related_nodes = [plan_node_by_id[node_id] for node_id in related_node_ids if node_id in plan_node_by_id]
        reference_ids = _dedupe(
            list(item.reference_ids)
            + list(blueprint.based_on_references if blueprint else [])
            + [node.ref_id for node in related_nodes if node.ref_id]
        )
        depends_on = _dedupe(list(item.depends_on) + list(dependency_map.get(file_path, [])))
        work_package_id = _canonical_work_package_id_for_path(
            file_path,
            package_by_path=package_by_path,
            proposed_work_package_id=item.work_package_id,
            state=state,
        )
        reference_ids = _dedupe(reference_ids + _work_package_reference_ids(state, work_package_id))
        reference_ids = _filter_reference_ids_for_state(state, reference_ids)
        owned_unit_ids = _filter_known_unit_ids(state, _work_package_owned_unit_ids(state, work_package_id))
        code_contract = _scoped_work_package_contract_for_file(state, work_package_id, file_path)
        decision_contract = _work_package_decision_contract(state, work_package_id)
        file_inventories = _generation_inventories_for_file(state, work_package_id, file_path)
        file_obligations = _file_inventory_obligations(file_path, file_inventories)
        method_obligations = _dedupe(
            _priority_evidence_contract_obligations(state, file_path)
            + _source_artifact_context_obligations_for_file(state, file_path)
            + file_obligations
            + _global_measurement_obligations_for_file(state, file_path)
            + _global_result_target_obligations_for_file(state, work_package_id, file_path)
            + _state_evidence_contract_obligations(state, file_path)
            + _formula_algorithm_obligations_for_file(state, file_path)
            + _addendum_obligations_for_file(state, file_path)
            + list(item.method_obligations)
            + code_contract["obligations"]
            + _bench_visible_contract_obligations(file_path)
        )
        base_method_obligations = sanitize_contract_list(
            _file_scoped_obligations(file_path, method_obligations),
            field="method_obligations",
        )
        symbol_obligations = _symbol_source_obligations(base_method_obligations)
        executable_obligations = _file_scoped_obligations(
            file_path,
            _executable_contract_obligations(file_path, symbol_obligations),
        )
        method_obligations = sanitize_contract_list(
            _dedupe(base_method_obligations + executable_obligations),
            field="method_obligations",
        )
        contract_artifact_paths = _paper_artifact_paths(_contract_terms(_state_evidence_contract(state), "artifacts")) if state is not None and _file_owns_artifact_outputs(file_path) else []
        artifact_outputs = _dedupe(
            list(item.writes_artifacts)
            + code_contract["artifacts"]
            + contract_artifact_paths
            + (_artifact_inventory_paths(file_inventories) if _file_owns_artifact_outputs(file_path) else [])
        )
        artifact_outputs = _file_scoped_artifact_outputs(file_path, artifact_outputs)
        base_review_points = list(item.review_points) or [
            f"Implement the contract-owned responsibility for {file_path}.",
            "Keep dependency wiring and package interfaces stable.",
            "Do not violate declared artifact or execution closure.",
        ]
        review_points = _dedupe(
            base_review_points
            + [f"Satisfy file-scoped method obligation: {obligation}" for obligation in base_method_obligations[:6]]
        )
        defines_symbols = _dedupe(
            list(item.defines_symbols)
            + [node.name for node in related_nodes if node.name]
            + _file_symbol_hints(
                file_path,
                surfaces=list(item.implementation_surfaces) + list(code_contract["surfaces"]),
                obligations=symbol_obligations,
            )
        )
        calls_symbols = _dedupe(
            list(item.calls_symbols)
            + _file_call_hints(
                file_path,
                surfaces=list(item.implementation_surfaces) + list(code_contract["surfaces"]),
                obligations=symbol_obligations,
                writes_artifacts=artifact_outputs,
                defines=defines_symbols,
            )
        )
        scope_boundary = sanitize_scope_boundary(
            dict(getattr(item, "scope_boundary", {}) or {})
            or _scope_boundary_for_file(
                state,
                work_package_id,
                _filter_known_unit_ids(state, list(item.owned_units) + owned_unit_ids),
                file_path=file_path,
                focus=[
                    *list(item.implementation_surfaces),
                    *list(code_contract["surfaces"]),
                    *artifact_outputs[:8],
                    *defines_symbols[:8],
                    *calls_symbols[:8],
                ],
            )
        )
        closed.append(
            item.model_copy(
                update={
                    "task_id": item.task_id or f"task_{index:03d}",
                    "work_package_id": work_package_id,
                    "purpose": item.purpose or (blueprint.purpose if blueprint else f"Implement {file_path}"),
                    "hypothesis": item.hypothesis or decision_contract["hypothesis"],
                    "decision_value": item.decision_value or decision_contract["decision_value"],
                    "stop_rule_or_pruning_rationale": decision_contract["stop_rule_or_pruning_rationale"],
                    "scope_boundary": scope_boundary,
                    "related_node_ids": related_node_ids,
                    "owned_units": _filter_known_unit_ids(state, list(item.owned_units) + owned_unit_ids),
                    "reference_ids": reference_ids,
                    "depends_on": depends_on,
                    "blocking_dependencies": _dedupe(list(item.blocking_dependencies) + depends_on),
                    "interface_contract": sanitize_contract_list(
                        _dedupe(list(item.interface_contract) + code_contract["interfaces"]),
                        field="interface_contract",
                    ),
                    "implementation_surfaces": _dedupe(list(item.implementation_surfaces) + code_contract["surfaces"]),
                    "method_obligations": method_obligations,
                    "context_sources": _dedupe(
                        list(item.context_sources)
                        + [f"unit:{unit_id}" for unit_id in owned_unit_ids]
                        + [f"node:{node_id}" for node_id in related_node_ids]
                        + [f"ref:{ref_id}" for ref_id in reference_ids]
                    ),
                    "consumes": _dedupe(list(item.consumes) + depends_on),
                    "produces": _dedupe(list(item.produces) or [file_path]),
                    "writes_artifacts": artifact_outputs,
                    "allowed_scope": item.allowed_scope or {"write": [file_path], "read": depends_on},
                    "generation_prompt": _append_code_contract_prompt(
                        item.generation_prompt or f"Implement `{file_path}` according to its package contract and repository execution closure.",
                        surfaces=code_contract["surfaces"],
                        obligations=method_obligations,
                        defines=defines_symbols,
                        calls=calls_symbols,
                    ),
                    "validation_hooks": _dedupe(
                        list(item.validation_hooks)
                        or ["python_syntax" if file_path.endswith(".py") else "file_exists"]
                    ),
                    "defines_symbols": defines_symbols,
                    "calls_symbols": calls_symbols,
                    "review_points": sanitize_contract_list(
                        _dedupe(
                            review_points
                            + [f"Preserve artifact output: {artifact}" for artifact in artifact_outputs[:6]]
                            + [f"Active route contract: define `{symbol}`." for symbol in defines_symbols[:8]]
                            + [f"Active route contract: wire/call `{symbol}`." for symbol in calls_symbols[:8]]
                        ),
                        field="review_points",
                    ),
                }
            )
        )

    missing_paths = [path for path in expected_paths if path not in seen_paths]
    if missing_paths:
        fallback = _project_file_plans_from_architecture(architecture, pipeline_plan, state=state)
        fallback_by_path = {item.target_file: item for item in fallback.file_plans}
        for path in missing_paths:
            if path in fallback_by_path:
                closed.append(fallback_by_path[path])

    closed = _wire_work_package_active_routes(closed)

    return PackageFilePlanningOutput(
        file_plans=closed,
        planning_notes=_dedupe(
            list(file_planning.planning_notes)
            + ["Closed package file plans against architecture file coverage and dependency closure."]
        ),
    )


def _wire_work_package_active_routes(file_plans: list[RepoFilePlan]) -> list[RepoFilePlan]:
    """Ensure route files call high-signal helpers from the same work package."""
    if not file_plans:
        return []
    by_package: dict[str, list[RepoFilePlan]] = {}
    for plan in file_plans:
        by_package.setdefault(str(plan.work_package_id or "").strip(), []).append(plan)

    def _is_support_symbol(symbol: str) -> bool:
        lowered = str(symbol or "").strip().lower()
        if not lowered:
            return True
        if lowered in {"main", "__all__"}:
            return True
        if lowered.startswith("test_"):
            return True
        if lowered.endswith(("config", "spec", "schema", "result", "layout", "settings", "types", "configs")):
            return True
        return False

    def _high_signal_plan_symbol(plan: RepoFilePlan, *, path: str) -> bool:
        text = " ".join(
            [
                path,
                *list(plan.implementation_surfaces or []),
                *list(plan.method_obligations or []),
                *list(plan.review_points or []),
                *list(plan.defines_symbols or []),
                *list(plan.calls_symbols or []),
            ]
        )
        return bool(
            plan.writes_artifacts
            or _high_signal_symbol_terms(text)
            or any(
                token in text.lower()
                for token in (
                    "metric",
                    "formula",
                    "factory",
                    "train",
                    "evaluate",
                    "sampler",
                    "adapter",
                    "baseline",
                    "dataset",
                    "environment",
                    "artifact",
                    "report",
                    "result",
                )
            )
        )

    def _helper_symbol_priority(symbol: str) -> tuple[int, str]:
        rendered = str(symbol or "").strip()
        name = rendered.rsplit("::", 1)[-1]
        lowered = name.lower()
        if lowered.startswith(("build_", "load_", "prepare_", "evaluate_", "compute_", "aggregate_", "write_", "run_", "train_")):
            return (0, rendered)
        if "fidelity" in lowered or "metric" in lowered or "artifact" in lowered:
            return (1, rendered)
        if lowered.startswith(("default_", "resolve_")) or lowered.endswith("_values"):
            return (3, rendered)
        return (2, rendered)

    all_high_signal_helpers: list[tuple[str, RepoFilePlan]] = []
    for plan in file_plans:
        path = _normalize_repo_path(plan.target_file)
        if path and _high_signal_plan_symbol(plan, path=path):
            all_high_signal_helpers.append((path, plan))

    rewired: list[RepoFilePlan] = []
    for plan in file_plans:
        file_path = _normalize_repo_path(plan.target_file)
        package_plans = by_package.get(str(plan.work_package_id or "").strip(), [])
        roles = _file_roles(file_path)
        is_route = bool(
            roles.intersection({"entrypoint", "experiment", "training", "evaluation", "artifact", "plotting", "reporting"})
            or file_path.lower().rsplit("/", 1)[-1] in {"main.py", "cli.py", "run.py", "run_experiments.py"}
        )
        if not is_route:
            rewired.append(plan)
            continue
        route_calls = list(plan.calls_symbols)
        route_dependencies = list(plan.depends_on)
        route_review_points = list(plan.review_points)
        route_prompt = str(plan.generation_prompt or "")
        helper_symbols: list[str] = []
        for helper_path, helper in all_high_signal_helpers:
            if helper_path == file_path:
                continue
            if helper_path != file_path and helper_path not in route_dependencies:
                route_dependencies.append(helper_path)
            if helper_path == file_path or not helper_path.endswith(".py"):
                continue
            if not _high_signal_plan_symbol(helper, path=helper_path):
                continue
            if helper_path not in route_dependencies:
                route_dependencies.append(helper_path)
            helper_defined_support = False
            for symbol in list(helper.defines_symbols or []):
                rendered = str(symbol or "").strip()
                if _is_support_symbol(rendered):
                    helper_defined_support = True
                    continue
                helper_symbols.append(rendered)
            if helper_defined_support:
                for symbol in list(helper.defines_symbols or []):
                    rendered = str(symbol or "").strip()
                    if _is_support_symbol(rendered):
                        continue
                    helper_symbols.append(rendered)
        for helper in package_plans:
            helper_path = _normalize_repo_path(helper.target_file)
            if helper_path == file_path or not helper_path.endswith(".py"):
                continue
            if not _high_signal_plan_symbol(helper, path=helper_path):
                continue
            helper_defined_support = False
            for symbol in list(helper.defines_symbols or []):
                rendered = str(symbol or "").strip()
                if _is_support_symbol(rendered):
                    helper_defined_support = True
                    continue
                helper_symbols.append(rendered)
            if helper_defined_support:
                for symbol in list(helper.defines_symbols or []):
                    rendered = str(symbol or "").strip()
                    if _is_support_symbol(rendered):
                        continue
                    helper_symbols.append(rendered)
        helper_symbols = _dedupe([item for _, item in sorted((_helper_symbol_priority(symbol) for symbol in helper_symbols), key=lambda item: item[0])])[:48]
        if helper_symbols:
            route_calls = _dedupe(route_calls + helper_symbols)
            route_review_points = _dedupe(
                route_review_points
                + [
                    "Active route closure: import or call same-package helper symbol "
                    f"`{symbol}` from the canonical route."
                    for symbol in helper_symbols[:12]
                ]
            )
            route_prompt = _append_code_contract_prompt(
                route_prompt or f"Implement `{file_path}` as the active route for its work package.",
                surfaces=[],
                obligations=[
                    "Active route closure: call same-package helper implementations rather than re-declaring registry rows or creating a toy route."
                ],
                defines=[],
                calls=helper_symbols,
            )
        rewired.append(
            plan.model_copy(
                update={
                    "depends_on": _dedupe(route_dependencies),
                    "blocking_dependencies": _dedupe(list(plan.blocking_dependencies) + route_dependencies),
                    "consumes": _dedupe(list(plan.consumes) + route_dependencies),
                    "calls_symbols": route_calls,
                    "generation_prompt": route_prompt,
                    "review_points": sanitize_contract_list(route_review_points, field="review_points"),
                }
            )
        )
    return rewired


def _validate_file_plans(
    architecture: ArchitectureOutput,
    file_planning: PackageFilePlanningOutput,
    state: PaperBenchReproState | None = None,
) -> None:
    """Validate package file plans against architecture closure."""
    expected_paths = set(_projection_paths(architecture, state))
    actual_paths = [_normalize_repo_path(item.target_file) for item in file_planning.file_plans]
    duplicate_paths = sorted({path for path in actual_paths if actual_paths.count(path) > 1})
    if duplicate_paths:
        raise ValueError(f"file_plans must contain unique target_file values: {duplicate_paths}")
    actual_path_set = set(actual_paths)
    missing_paths = sorted(expected_paths - actual_path_set)
    extra_paths = sorted(actual_path_set - expected_paths)
    if missing_paths or extra_paths:
        raise ValueError(
            "file_plans do not match architecture file coverage: "
            f"missing={missing_paths}, extra={extra_paths}"
        )
    if state is not None and state.canonical_ir is not None:
        allowed_paths = set(expected_paths)
        disallowed_paths = sorted(path for path in actual_path_set if path not in allowed_paths)
        if disallowed_paths:
            raise ValueError(
                "file_plans contain paths outside canonical registered file graph: "
                f"{disallowed_paths}"
            )


def _order_file_plans_for_execution_closure(
    architecture: ArchitectureOutput,
    file_planning: PackageFilePlanningOutput,
) -> PackageFilePlanningOutput:
    """Order file plans so dependency closure is respected before generation."""
    dependency_map = _dependency_map(architecture)
    indegree = {item.target_file: 0 for item in file_planning.file_plans}
    adjacency: dict[str, list[str]] = {item.target_file: [] for item in file_planning.file_plans}
    by_path = {item.target_file: item for item in file_planning.file_plans}

    for source, targets in dependency_map.items():
        if source not in indegree:
            continue
        for target in targets:
            if target not in indegree:
                continue
            adjacency.setdefault(target, []).append(source)
            indegree[source] += 1

    planning_order = _dedupe(list(architecture.target_file_tree) + [item.target_file for item in file_planning.file_plans])
    queue = [path for path in planning_order if path in indegree and indegree[path] == 0]
    ordered_paths: list[str] = []
    seen: set[str] = set()
    while queue:
        current = queue.pop(0)
        if current in seen:
            continue
        seen.add(current)
        ordered_paths.append(current)
        for dependent in adjacency.get(current, []):
            indegree[dependent] -= 1
            if indegree[dependent] <= 0:
                queue.append(dependent)

    for path in planning_order:
        if path in by_path and path not in seen:
            ordered_paths.append(path)

    return PackageFilePlanningOutput(
        file_plans=[by_path[path] for path in ordered_paths if path in by_path],
        planning_notes=_dedupe(
            list(file_planning.planning_notes)
            + ["Ordered file plans by deterministic execution closure."]
        ),
    )


def _derive_steps_from_file_plans(file_planning: PackageFilePlanningOutput) -> list[str]:
    """Render file plans into stable high-level execution steps."""
    return [f"{item.task_id or item.target_file}: generate {item.target_file}" for item in file_planning.file_plans]


def _render_pipeline_plan_markdown(state: PaperBenchReproState) -> str:
    """Render the planning pipeline into a compact markdown plan."""
    dataset_preparation = _get_dataset_preparation(state)
    resource_manifest = _get_resource_manifest(state)
    lines = ["# Experiment Implementation Plan", "", "## Target", state.input.target, "", "## Prepared Datasets"]
    if dataset_preparation.get("downloaded_datasets"):
        for item in dataset_preparation["downloaded_datasets"]:
            lines.append(f"- {item.get('name', '')}: {item.get('local_path', '')}")
    elif dataset_preparation.get("requested_datasets"):
        lines.append("- Dataset requests exist but no prepared dataset artifact was recorded.")
    else:
        lines.append("- None")

    lines.extend(["", "## Prepared Resources"])
    if resource_manifest:
        for key in ("benchmarks", "baselines", "ref_repos"):
            items = list(resource_manifest.get(key, []) or [])
            lines.append(f"- {key}: {len(items)}")
    else:
        lines.append("- None")

    lines.extend(["", "## Work Packages"])
    if state.work_package_planning:
        for item in state.work_package_planning.work_packages:
            lines.append(f"- [{item.work_package_id}] {item.goal}")
    else:
        lines.append("- None")

    lines.extend(["", "## Architecture"])
    if state.architecture:
        for path in state.architecture.target_file_tree:
            lines.append(f"- {path}")
    else:
        lines.append("- None")

    lines.extend(["", "## File Plans"])
    if state.repo_plan is not None and state.repo_plan.files:
        for item in state.repo_plan.files:
            lines.append(f"- [{item.task_id or item.target_file}] {item.target_file}: {item.purpose}")
    elif state.package_file_planning_output:
        for item in state.package_file_planning_output.file_plans:
            lines.append(f"- [{item.task_id or item.target_file}] {item.target_file}: {item.purpose}")
    else:
        lines.append("- None")

    return "\n".join(lines) + "\n"
