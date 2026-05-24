"""Early semantic quality checks before repository generation."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, replace
from typing import Any

from reproagent.pipeline.utils.evidence_contracts import (
    evidence_contract_gaps,
    flatten_evidence_contract,
    infer_evidence_contract,
    implementation_obligation_gaps,
)


_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+-]*")
_QUALITY_POLICY_ENV_JSON = "PAPERBENCH_REPRO_QUALITY_POLICY_JSON"
_QUALITY_POLICY_ENV_PATH = "PAPERBENCH_REPRO_QUALITY_POLICY_PATH"


@dataclass(frozen=True)
class QualityGatePolicy:
    """Configurable structural quality policy for prepare/plan gates.

    Defaults are intentionally domain-generic for reproduction repositories. They
    can be overridden with JSON from PAPERBENCH_REPRO_QUALITY_POLICY_JSON or
    PAPERBENCH_REPRO_QUALITY_POLICY_PATH without changing this module.
    """

    surface_rules: dict[str, dict[str, tuple[str, ...]]] = field(
        default_factory=lambda: {
            "environment": {
                "keywords": (
                    "environment",
                    "environments",
                    "benchmark",
                    "simulator",
                    "task",
                ),
                "surfaces": (
                    "environment",
                    "environment_adapter",
                    "environment_factory",
                    "data_pipeline",
                    "config",
                ),
            },
            "policy_model": {
                "keywords": (
                    "agent",
                    "policy",
                    "policies",
                    "model",
                    "neural network",
                    "policy network",
                    "checkpoint",
                    "pretrained",
                    "pretraining",
                ),
                "surfaces": (
                    "policy",
                    "policy_adapter",
                    "policy_factory",
                    "model",
                    "model_or_method",
                    "pretraining",
                    "training_loop",
                ),
            },
            "training": {
                "keywords": (
                    "train",
                    "training",
                    "pretrain",
                    "pretraining",
                    "fine-tuning",
                    "finetuning",
                    "optimizer",
                    "epoch",
                ),
                "surfaces": (
                    "training",
                    "training_loop",
                    "pretraining",
                    "model_or_method",
                    "policy_adapter",
                ),
            },
            "evaluation_metric": {
                "keywords": (
                    "evaluate",
                    "evaluation",
                    "metric",
                    "metrics",
                    "score",
                    "fidelity",
                    "reward",
                    "return",
                    "table",
                ),
                "surfaces": (
                    "evaluation",
                    "metric",
                    "metric_formula",
                    "baseline_or_ablation",
                    "artifact_writer",
                ),
            },
            "baseline": {
                "keywords": (
                    "baseline",
                    "baselines",
                    "ablation",
                    "compare",
                    "comparison",
                    "variant",
                    "variants",
                ),
                "surfaces": (
                    "baseline",
                    "baseline_or_ablation",
                    "model_or_method",
                    "evaluation",
                ),
            },
            "refinement": {
                "keywords": (
                    "refine",
                    "refinement",
                    "fine-tune",
                    "fine-tuning",
                ),
                "surfaces": (
                    "refinement",
                    "refinement_algorithm",
                    "training_loop",
                    "evaluation",
                    "model_or_method",
                ),
            },
            "experiment_protocol": {
                "keywords": ("experiment", "experiments", "figure", "table", "result", "results"),
                "surfaces": ("entrypoint", "evaluation", "config", "artifact_writer"),
            },
        }
    )
    support_only_surfaces: frozenset[str] = frozenset({"artifact_writer", "config", "tests", "entrypoint"})
    repository_skeleton_rules: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            "entrypoint": ("main.py", "cli.py", "scripts/", "bin/"),
            "documentation": ("readme.md", "docs/"),
            "packaging": ("pyproject.toml", "setup.py", "setup.cfg"),
            "config": ("configs/", "config/", ".yaml", ".yml", ".toml", ".json"),
            "tests": ("tests/", "test_"),
        }
    )
    repository_skeleton_surfaces: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            "entrypoint": ("entrypoint", "entry", "cli", "main"),
            "documentation": ("documentation", "doc", "docs", "readme"),
            "packaging": ("packaging", "package", "install", "dependency", "dependencies"),
            "config": ("config", "configuration", "settings"),
            "tests": ("tests", "test"),
        }
    )
    repository_skeleton_explicit_terms: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            "entrypoint": (
                "entrypoint",
                "entry point",
                "command line",
                "cli",
                "run script",
                "runner",
                "main.py",
            ),
            "documentation": ("documentation", "readme", "docs/", "docs "),
            "packaging": (
                "packaging",
                "package metadata",
                "installable",
                "installation",
                "dependency manifest",
                "requirements.txt",
                "pyproject.toml",
                "setup.py",
            ),
            "config": (
                "config",
                "configuration",
                "hyperparameter file",
                "sweep file",
                "yaml",
                "toml",
                "json config",
            ),
            "tests": (
                "unit test",
                "unit tests",
                "test suite",
                "pytest",
                "contract test",
                "smoke test",
                "tests/",
            ),
        }
    )
    decision_value_triggers: tuple[str, ...] = (
        "ablation",
        "ablations",
        "baseline",
        "baselines",
        "benchmark",
        "compare",
        "comparison",
        "decision",
        "evaluate",
        "evaluation",
        "experiment",
        "experiments",
        "fidelity",
        "hypothesis",
        "metric",
        "metrics",
        "refine",
        "refinement",
        "result",
        "results",
        "score",
        "sweep",
        "variant",
        "variants",
    )
    unit_route_support_only_surfaces: frozenset[str] = frozenset(
        {"config", "tests", "documentation", "packaging", "doc"}
    )
    prepare_forbidden_unit_tokens: tuple[str, ...] = (
        "rubric.json",
        "judge metadata",
        "scoring metadata",
    )
    prepare_min_active_units: int = 10
    prepare_min_claim_coverage: float = 0.75
    prepare_min_artifact_coverage: float = 0.85
    weak_reference_path_tokens: tuple[str, ...] = (
        ".github/",
        "code_of_conduct",
        "contributing",
        "issue_template",
        "pull_request_template",
        "security.md",
        "license",
        "readme",
    )
    actionable_reference_extensions: tuple[str, ...] = (
        ".py",
        ".ipynb",
        ".yaml",
        ".yml",
        ".json",
        ".toml",
        ".cfg",
        ".ini",
        ".sh",
        ".bash",
        ".md",
    )
    non_actionable_symbol_kinds: frozenset[str] = frozenset(
        {"doc_protocol", "documentation", "readme", "license", "community_protocol"}
    )
    generic_paper_unit_ids: frozenset[str] = frozenset(
        {
            "paper_evidence_matrix",
            "paper_named_experiment_protocols",
            "paper_addendum_constraints",
            "paper_environment_inventory",
            "paper_dataset_inventory",
            "paper_task_environment_setup",
            "paper_method_core",
            "paper_training_or_optimization_loop",
            "paper_evaluation_protocol",
            "paper_contract_dataset_metric_protocol",
            "paper_contract_method_baseline_protocol",
            "paper_contract_experiment_artifact_protocol",
            "paper_contract_sweep_hyperparameter_protocol",
            "paper_contract_environment_protocol",
        }
    )
    active_route_filenames: frozenset[str] = frozenset({"main.py", "cli.py", "run.py", "run_experiments.py"})
    active_route_path_terms: tuple[str, ...] = (
        "experiment",
        "evaluation",
        "evaluate",
        "training",
        "train",
        "runner",
        "report",
        "artifact",
        "plot",
        "figure",
    )
    route_support_filenames: frozenset[str] = frozenset(
        {
            "__init__.py",
            "config.py",
            "configs.py",
            "constants.py",
            "registry.py",
            "schema.py",
            "schemas.py",
            "settings.py",
            "types.py",
        }
    )
    route_support_path_prefixes: tuple[str, ...] = ("configs/", "config/", "tests/")
    route_support_suffixes: tuple[str, ...] = (
        ".yaml",
        ".yml",
        ".toml",
        ".json",
        ".md",
        ".txt",
    )
    support_plan_suffixes: tuple[str, ...] = (
        "readme.md",
        "requirements.txt",
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
    )
    active_route_required_surfaces: frozenset[str] = frozenset(
        {
            "training",
            "training_loop",
            "pretraining",
            "evaluation",
            "metric",
            "metric_formula",
            "baseline",
            "baseline_or_ablation",
            "environment",
            "environment_adapter",
            "data_pipeline",
            "model",
            "model_or_method",
            "policy",
            "policy_adapter",
            "refinement",
            "refinement_algorithm",
            "artifact_writer",
        }
    )
    active_route_required_terms: tuple[str, ...] = (
        "ablation",
        "artifact",
        "baseline",
        "benchmark",
        "dataset",
        "evaluate",
        "evaluation",
        "experiment",
        "figure",
        "metric",
        "plot",
        "report",
        "result",
        "score",
        "table",
        "train",
        "training",
    )
    callable_symbol_prefixes: tuple[str, ...] = (
        "build_",
        "prepare_",
        "load_",
        "run_",
        "evaluate_",
        "compute_",
        "write_",
        "train_",
        "fit_",
        "make_",
    )
    non_callable_exact_symbols: frozenset[str] = frozenset({"main", "__all__"})
    non_callable_prefixes: tuple[str, ...] = ("test_", "default_")
    non_callable_suffixes: tuple[str, ...] = (
        "_values",
        "config",
        "spec",
        "schema",
        "result",
        "layout",
        "settings",
        "types",
    )
    method_plan_signal_terms: tuple[str, ...] = (
        "train",
        "evaluate",
        "metric",
        "policy",
        "environment",
        "refine",
        "baseline",
    )
    method_plan_symbol_terms: tuple[str, ...] = ("class", "function", "factory", "adapter", "loop", "compute")

    @property
    def implementation_surfaces(self) -> frozenset[str]:
        return frozenset(
            surface
            for rule in self.surface_rules.values()
            for surface in rule["surfaces"]
            if surface not in self.support_only_surfaces
        )


_QUALITY_POLICY_CACHE: QualityGatePolicy | None = None


def _as_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(str(item) for item in value)
    return tuple()


def _as_frozenset(value: Any) -> frozenset[str]:
    return frozenset(item for item in _as_tuple(value) if item)


def _normalize_surface_rules(value: Any, default: dict[str, dict[str, tuple[str, ...]]]) -> dict[str, dict[str, tuple[str, ...]]]:
    if not isinstance(value, dict):
        return default
    normalized: dict[str, dict[str, tuple[str, ...]]] = {}
    for group, payload in value.items():
        if not isinstance(payload, dict):
            continue
        normalized[str(group)] = {
            "keywords": _as_tuple(payload.get("keywords")),
            "surfaces": _as_tuple(payload.get("surfaces")),
        }
    return normalized or default


def _normalize_tuple_dict(value: Any, default: dict[str, tuple[str, ...]]) -> dict[str, tuple[str, ...]]:
    if not isinstance(value, dict):
        return default
    normalized = {str(key): _as_tuple(items) for key, items in value.items()}
    return normalized or default


def _load_policy_override_payload() -> dict[str, Any]:
    raw_json = os.getenv(_QUALITY_POLICY_ENV_JSON, "").strip()
    raw_path = os.getenv(_QUALITY_POLICY_ENV_PATH, "").strip()
    if raw_json:
        try:
            payload = json.loads(raw_json)
            return payload if isinstance(payload, dict) else {}
        except json.JSONDecodeError:
            return {}
    if raw_path:
        try:
            with open(raw_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            return payload if isinstance(payload, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}
    return {}


def _apply_policy_overrides(policy: QualityGatePolicy, overrides: dict[str, Any]) -> QualityGatePolicy:
    if not overrides:
        return policy
    fields = set(QualityGatePolicy.__dataclass_fields__)
    replacements: dict[str, Any] = {}
    for key, value in overrides.items():
        if key not in fields:
            continue
        current = getattr(policy, key)
        if key == "surface_rules":
            replacements[key] = _normalize_surface_rules(value, current)
        elif isinstance(current, frozenset):
            replacements[key] = _as_frozenset(value)
        elif isinstance(current, tuple):
            replacements[key] = _as_tuple(value)
        elif isinstance(current, dict):
            replacements[key] = _normalize_tuple_dict(value, current)
        elif isinstance(current, int):
            try:
                replacements[key] = int(value)
            except (TypeError, ValueError):
                continue
        elif isinstance(current, float):
            try:
                replacements[key] = float(value)
            except (TypeError, ValueError):
                continue
    return replace(policy, **replacements) if replacements else policy


def _quality_policy() -> QualityGatePolicy:
    global _QUALITY_POLICY_CACHE
    if _QUALITY_POLICY_CACHE is None:
        _QUALITY_POLICY_CACHE = _apply_policy_overrides(
            QualityGatePolicy(),
            _load_policy_override_payload(),
        )
    return _QUALITY_POLICY_CACHE


def _dedupe(items: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for item in items:
        value = str(item or "").strip()
        if not value or value in seen:
            continue
        ordered.append(value)
        seen.add(value)
    return ordered


def _get_value(item: Any, name: str) -> Any:
    return item.get(name) if isinstance(item, dict) else getattr(item, name, None)


def _object_values(item: Any, names: tuple[str, ...]) -> list[str]:
    values: list[str] = []
    for name in names:
        value = _get_value(item, name)
        if isinstance(value, str):
            values.append(value)
        elif isinstance(value, (list, tuple, set)):
            values.extend(str(part) for part in value if str(part).strip())
        elif isinstance(value, dict):
            for nested in value.values():
                if isinstance(nested, (list, tuple, set)):
                    values.extend(str(part) for part in nested if str(part).strip())
                elif str(nested).strip():
                    values.append(str(nested))
    return values


def _object_text(item: Any) -> str:
    return " ".join(
        _object_values(
            item,
            (
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
                "calls_symbols",
                "writes_artifacts",
                "tags",
                "hypothesis",
                "decision_value",
            ),
        )
    ).lower()


def _object_text_raw(item: Any) -> str:
    return " ".join(
        _object_values(
            item,
            (
                "unit_id",
                "type",
                "statement",
                "hypothesis",
                "decision_value",
                "paper_evidence",
                "source_paragraph_ids",
                "citation_refs",
                "implementation_surfaces",
                "code_obligations",
                "runtime_interfaces",
                "expected_artifacts",
                "suggested_module_kinds",
                "implementation_notes",
            ),
        )
    )


def _surface_values(items: list[Any]) -> list[str]:
    surfaces: list[str] = []
    for item in items:
        surfaces.extend(_object_values(item, ("implementation_surfaces", "tags", "defines_symbols")))
        inventories = _get_value(item, "inventories")
        if isinstance(inventories, dict):
            surfaces.extend(str(part) for part in inventories.get("implementation_surface_inventory", []) or [])
    return _dedupe([surface.lower() for surface in surfaces])


def _target_path(item: Any) -> str:
    return str(_get_value(item, "target_file") or _get_value(item, "path") or "").strip().replace("\\", "/").lower()


def _normalize_plan_path(path: str) -> str:
    normalized = str(path or "").strip().replace("\\", "/").lower()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized.strip("/")


def _is_source_plan(item: Any) -> bool:
    path = _target_path(item)
    return path.endswith(".py") and not path.endswith("__init__.py")


def _is_active_route_file(path: str) -> bool:
    policy = _quality_policy()
    normalized = str(path or "").lower()
    basename = normalized.rsplit("/", 1)[-1]
    if basename in policy.active_route_filenames:
        return True
    return any(token in normalized for token in policy.active_route_path_terms)


def _is_route_support_file(path: str) -> bool:
    policy = _quality_policy()
    normalized = str(path or "").lower()
    basename = normalized.rsplit("/", 1)[-1]
    return (
        not normalized
        or basename in policy.route_support_filenames
        or normalized.startswith(policy.route_support_path_prefixes)
        or normalized.endswith(policy.route_support_suffixes)
    )


def _is_support_plan(item: Any) -> bool:
    policy = _quality_policy()
    path = _target_path(item)
    basename = path.rsplit("/", 1)[-1]
    return (
        not path
        or basename in policy.route_support_filenames
        or path.endswith(policy.support_plan_suffixes)
        or path.startswith(policy.route_support_path_prefixes)
        or path.endswith(policy.route_support_suffixes)
    )


def _has_active_route_signal(text: str) -> bool:
    lowered = str(text or "").lower()
    return any(token in lowered for token in _quality_policy().active_route_required_terms)


def _plan_has_active_route_signal(plan: Any) -> bool:
    policy = _quality_policy()
    text = _object_text(plan)
    surfaces = set(_surface_values([plan]))
    return (
        bool(surfaces.intersection(policy.active_route_required_surfaces))
        or bool(_object_values(plan, ("writes_artifacts",)))
        or _has_active_route_signal(text)
    )


def _active_route_contract_required(file_plans: list[Any]) -> bool:
    source_plans = [
        plan
        for plan in file_plans
        if _is_source_plan(plan) and not _is_support_plan(plan)
    ]
    if any(_plan_has_active_route_signal(plan) for plan in source_plans):
        return True
    return any(
        _is_active_route_file(_target_path(plan))
        and bool(_object_values(plan, ("calls_symbols", "writes_artifacts")))
        for plan in file_plans
        if _is_source_plan(plan)
    )


def _active_route_contract_report(file_plans: list[Any]) -> dict[str, Any]:
    """Check that plan-level symbols are wired before generation starts.

    The gate is triggered by structural plan signals: implementation surfaces,
    artifact-writing, callable symbols, or active train/eval/report routes. This
    keeps the check paper-agnostic while still preventing orphaned method files.
    """
    policy = _quality_policy()
    if not _active_route_contract_required(file_plans):
        return {
            "required": False,
            "status": "skipped",
            "route_files": [],
            "implementation_files": [],
            "unwired_symbols": [],
            "route_files_without_calls": [],
            "missing_route_files": [],
            "notes": ["Active-route contract skipped because no implementation route signal was detected."],
        }

    route_plans = [
        plan
        for plan in file_plans
        if _is_source_plan(plan) and _is_active_route_file(_target_path(plan))
    ]
    implementation_plans = [
        plan
        for plan in file_plans
        if _is_source_plan(plan) and not _is_support_plan(plan)
    ]
    route_text = "\n".join(
        " ".join(
            _object_values(
                plan,
                (
                    "target_file",
                    "purpose",
                    "generation_prompt",
                    "implementation_surfaces",
                    "method_obligations",
                    "review_points",
                    "calls_symbols",
                    "writes_artifacts",
                ),
            )
        ).lower()
        for plan in route_plans
    )
    route_calls = set()
    for plan in route_plans:
        route_calls.update(value.lower() for value in _object_values(plan, ("calls_symbols",)) if value.strip())

    def is_callable_symbol(value: str) -> bool:
        rendered = str(value or "").strip()
        if not rendered:
            return False
        name = rendered.rsplit("::", 1)[-1]
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
            return False
        if name[:1].isupper():
            return False
        lowered_name = name.lower()
        if lowered_name in policy.non_callable_exact_symbols:
            return False
        if name.isupper() and "_" in name:
            return False
        if lowered_name.startswith(policy.non_callable_prefixes):
            return False
        if lowered_name.startswith("resolve_") and lowered_name.endswith("_defaults"):
            return False
        if lowered_name.endswith(policy.non_callable_suffixes):
            return False
        return True

    def requires_direct_route_reference(value: str) -> bool:
        rendered = str(value or "").strip()
        lowered = rendered.lower()
        if not is_callable_symbol(rendered):
            return False
        return (
            lowered in route_calls
            or lowered.startswith(policy.callable_symbol_prefixes)
            or _has_active_route_signal(rendered)
        )

    def has_callable_definitions(plan: Any) -> bool:
        return any(is_callable_symbol(symbol) for symbol in _object_values(plan, ("defines_symbols",)))

    candidate_symbols: list[dict[str, str]] = []
    for plan in implementation_plans:
        path = _target_path(plan)
        if _is_route_support_file(path):
            continue
        if _is_active_route_file(path) and route_calls:
            continue
        plan_text = _object_text(plan)
        surfaces = set(_surface_values([plan]))
        if surfaces and surfaces.issubset(policy.support_only_surfaces):
            continue
        plan_candidates = [
            str(symbol or "").strip()
            for symbol in _object_values(plan, ("defines_symbols",))
            if requires_direct_route_reference(str(symbol or "").strip())
        ]
        if any(symbol.lower() in route_text for symbol in plan_candidates):
            continue
        candidate_symbols.extend({"path": path, "symbol": symbol} for symbol in plan_candidates)

    unwired_symbols = [
        item
        for item in candidate_symbols
        if item["symbol"].lower() not in route_text
    ]
    route_files_without_calls = [
        _target_path(plan)
        for plan in route_plans
        if _plan_has_active_route_signal(plan)
        and not has_callable_definitions(plan)
        and not _object_values(plan, ("calls_symbols",))
        and not any("active route contract" in value.lower() for value in _object_values(plan, ("review_points", "generation_prompt")))
    ]
    missing_route_files: list[str] = []
    if not route_plans:
        missing_route_files.append("no active entry/experiment/evaluation/reporting route file is planned")

    status = "passed" if not unwired_symbols and not route_files_without_calls and not missing_route_files else "needs_attention"
    notes = []
    if unwired_symbols:
        notes.append(
            "Implementation symbols not wired into active route plans: "
            + "; ".join(f"{item['path']}::{item['symbol']}" for item in unwired_symbols[:12])
        )
    else:
        notes.append("Implementation symbols are referenced by active route plans.")
    if route_files_without_calls:
        notes.append("Active route files missing calls_symbols: " + ", ".join(route_files_without_calls[:12]))
    if missing_route_files:
        notes.extend(missing_route_files)
    return {
        "required": True,
        "status": status,
        "route_files": [_target_path(plan) for plan in route_plans],
        "implementation_files": [_target_path(plan) for plan in implementation_plans],
        "unwired_symbols": unwired_symbols,
        "route_files_without_calls": route_files_without_calls,
        "missing_route_files": missing_route_files,
        "notes": notes,
    }


def _requires_decision_value(item: Any) -> bool:
    policy = _quality_policy()
    text = _object_text(item)
    if any(token in text for token in policy.decision_value_triggers):
        return True
    surfaces = set(_surface_values([item]))
    return bool(surfaces.intersection({"evaluation", "metric", "metric_formula", "baseline", "baseline_or_ablation"}))


def _decision_value_gaps(items: list[Any], *, id_field: str) -> list[dict[str, Any]]:
    gaps: list[dict[str, Any]] = []
    for index, item in enumerate(items, start=1):
        if not _requires_decision_value(item):
            continue
        missing = [
            field
            for field in ("hypothesis", "decision_value")
            if not str(_get_value(item, field) or "").strip()
        ]
        if not missing:
            continue
        identifier = str(_get_value(item, id_field) or _get_value(item, "target_file") or f"item_{index:03d}").strip()
        gaps.append({"id": identifier, "missing_fields": missing})
    return gaps


def _required_repository_skeleton_groups(units: list[Any], work_packages: list[Any]) -> list[str]:
    policy = _quality_policy()
    required: list[str] = []
    source_items = [*units, *work_packages]
    source_text = "\n".join(_object_text(item) for item in source_items)
    source_surfaces = set(_surface_values(source_items))
    for group, surface_hints in policy.repository_skeleton_surfaces.items():
        explicit_surface = bool(source_surfaces.intersection(set(surface_hints)))
        explicit_text = any(term in source_text for term in policy.repository_skeleton_explicit_terms.get(group, ()))
        if explicit_surface or explicit_text:
            required.append(group)
    return sorted(set(required))


def _repository_skeleton_report(file_plans: list[Any], *, required_groups: list[str]) -> dict[str, Any]:
    policy = _quality_policy()
    required_set = set(required_groups)
    if not file_plans and not required_set:
        return {
            "required": False,
            "required_groups": [],
            "present_groups": [],
            "missing_groups": [],
            "optional_missing_groups": [],
            "notes": ["Repository support-surface check skipped because no explicit support surfaces were required."],
        }
    present: list[str] = []
    paths = [_target_path(plan) for plan in file_plans]
    surfaces_by_group = {
        group: set(values)
        for group, values in policy.repository_skeleton_surfaces.items()
    }
    for group, path_hints in policy.repository_skeleton_rules.items():
        surface_hints = surfaces_by_group.get(group, set())
        has_path = any(
            path
            and (
                any(path == hint for hint in path_hints if not hint.endswith("/") and not hint.startswith("."))
                or any(path.startswith(hint) for hint in path_hints if hint.endswith("/"))
                or any(path.endswith(hint) for hint in path_hints if hint.startswith("."))
                or (group == "tests" and "/test_" in path)
            )
            for path in paths
        )
        has_surface = any(
            set(_surface_values([plan])).intersection(surface_hints)
            for plan in file_plans
        )
        if has_path or has_surface:
            present.append(group)
    present_set = set(present)
    missing = sorted(required_set - present_set)
    optional_missing = sorted(set(policy.repository_skeleton_rules) - present_set - required_set)
    return {
        "required": bool(required_set),
        "required_groups": sorted(required_set),
        "present_groups": sorted(present),
        "missing_groups": missing,
        "optional_missing_groups": optional_missing,
        "notes": [
            "Repository support groups present: " + ", ".join(sorted(present)) if present else "No repository support groups are present.",
            "Missing required repository support groups: " + ", ".join(missing)
            if missing
            else "All explicitly required repository support groups are represented.",
            "Optional repository support groups not planned: " + ", ".join(optional_missing)
            if optional_missing
            else "All optional repository support groups are already present.",
        ],
    }


def _covers_group(group: str, available_surfaces: set[str], text: str, accepted_surfaces: set[str]) -> bool:
    implementation_surfaces = accepted_surfaces - _quality_policy().support_only_surfaces
    if implementation_surfaces:
        return bool(available_surfaces.intersection(implementation_surfaces)) or any(
            surface in text for surface in implementation_surfaces
        )
    return bool(available_surfaces.intersection(accepted_surfaces)) or any(surface in text for surface in accepted_surfaces)


def _missing_work_package_file_plans(work_packages: list[Any], file_plans: list[Any]) -> list[str]:
    required_ids = {
        str(_get_value(work_package, "work_package_id") or "").strip()
        for work_package in work_packages
        if str(_get_value(work_package, "work_package_id") or "").strip()
    }
    if not required_ids:
        return []
    covered_ids = {
        str(_get_value(plan, "work_package_id") or "").strip()
        for plan in file_plans
        if str(_get_value(plan, "work_package_id") or "").strip()
        and str(_get_value(plan, "target_file") or "").strip()
    }

    def looks_like_repo_relative_file_path(path: str) -> bool:
        normalized = _normalize_plan_path(path)
        if not normalized or normalized.endswith("/"):
            return False
        if normalized.startswith(("/", "-", "$")) or "://" in normalized:
            return False
        if any(char.isspace() for char in normalized):
            return False
        if normalized.startswith(("results/", "outputs/", "artifacts/")):
            return False
        basename = normalized.rsplit("/", 1)[-1]
        return "." in basename

    produces_by_work_package = {
        str(_get_value(work_package, "work_package_id") or "").strip(): {
            _normalize_plan_path(str(path or ""))
            for path in list(_get_value(work_package, "produces") or [])
            if looks_like_repo_relative_file_path(str(path or ""))
        }
        for work_package in work_packages
        if str(_get_value(work_package, "work_package_id") or "").strip()
    }
    planned_paths = {
        _normalize_plan_path(str(_get_value(plan, "target_file") or ""))
        for plan in file_plans
        if str(_get_value(plan, "target_file") or "").strip()
    }
    for work_package_id, produced_paths in produces_by_work_package.items():
        normalized_produced_paths = {_normalize_plan_path(path) for path in produced_paths if _normalize_plan_path(path)}
        if planned_paths.intersection(normalized_produced_paths):
            covered_ids.add(work_package_id)
    package_units = {
        str(_get_value(work_package, "work_package_id") or "").strip(): {
            str(unit_id).strip()
            for unit_id in list(_get_value(work_package, "owned_unit_ids") or [])
            if str(unit_id).strip()
        }
        for work_package in work_packages
        if str(_get_value(work_package, "work_package_id") or "").strip()
    }
    for plan in file_plans:
        if not str(_get_value(plan, "target_file") or "").strip():
            continue
        plan_units = {
            str(unit_id).strip()
            for unit_id in list(
                _get_value(plan, "owned_unit_ids")
                or _get_value(plan, "owned_units")
                or _get_value(plan, "source_unit_ids")
                or []
            )
            if str(unit_id).strip()
        }
        if not plan_units:
            continue
        for work_package_id, owned_units in package_units.items():
            if plan_units.intersection(owned_units):
                covered_ids.add(work_package_id)
    return sorted(required_ids - covered_ids)


def _plan_owned_unit_ids(plan: Any) -> set[str]:
    return {
        str(unit_id).strip()
        for unit_id in list(
            _get_value(plan, "owned_unit_ids")
            or _get_value(plan, "owned_units")
            or _get_value(plan, "source_unit_ids")
            or []
        )
        if str(unit_id).strip()
    }


def _work_package_owned_unit_map(work_packages: list[Any]) -> dict[str, set[str]]:
    return {
        str(_get_value(work_package, "work_package_id") or "").strip(): {
            str(unit_id).strip()
            for unit_id in list(_get_value(work_package, "owned_unit_ids") or [])
            if str(unit_id).strip()
        }
        for work_package in work_packages
        if str(_get_value(work_package, "work_package_id") or "").strip()
    }


def _unit_requires_active_route(unit: Any) -> bool:
    policy = _quality_policy()
    surfaces = {str(surface).strip().lower() for surface in _surface_values([unit]) if str(surface).strip()}
    if surfaces.intersection(policy.implementation_surfaces):
        return True
    if surfaces and surfaces.issubset(policy.support_only_surfaces | policy.unit_route_support_only_surfaces):
        return False
    return bool(_nonempty_list_value(unit, "code_obligations") or _nonempty_list_value(unit, "expected_artifacts"))


def _unit_active_route_report(*, units: list[Any], work_packages: list[Any], file_plans: list[Any]) -> dict[str, Any]:
    """Check that each paper-derived implementation unit survives into an active file route."""
    active_units = _active_prepare_units(units)
    work_package_units = _work_package_owned_unit_map(work_packages)
    owner_packages_by_unit: dict[str, list[str]] = {}
    for work_package_id, unit_ids in work_package_units.items():
        for unit_id in unit_ids:
            owner_packages_by_unit.setdefault(unit_id, []).append(work_package_id)

    source_plans = [plan for plan in file_plans if _is_source_plan(plan)]
    route_plans = [plan for plan in source_plans if _is_active_route_file(_target_path(plan))]
    rows: list[dict[str, Any]] = []
    missing_owner_units: list[str] = []
    missing_file_plan_units: list[str] = []
    support_only_file_units: list[str] = []
    missing_active_owner_units: list[str] = []
    missing_route_units: list[str] = []

    for index, unit in enumerate(active_units, start=1):
        unit_id = _unit_identifier(unit, index)
        required = _unit_requires_active_route(unit)
        owner_work_packages = sorted(owner_packages_by_unit.get(unit_id, []))
        owner_file_plans = [
            plan
            for plan in file_plans
            if (
                unit_id in _plan_owned_unit_ids(plan)
                or str(_get_value(plan, "work_package_id") or "").strip() in owner_work_packages
            )
        ]
        owner_paths = _dedupe([_target_path(plan) for plan in owner_file_plans if _target_path(plan)])
        active_owner_paths = _dedupe(
            [
                _target_path(plan)
                for plan in owner_file_plans
                if _is_source_plan(plan) and not _is_support_plan(plan)
            ]
        )
        route_paths = _dedupe(
            [
                _target_path(plan)
                for plan in route_plans
                if (
                    _target_path(plan) in set(active_owner_paths)
                    or unit_id in _plan_owned_unit_ids(plan)
                    or any(
                        str(symbol or "").strip().lower() in _object_text(plan)
                        for owner_plan in owner_file_plans
                        for symbol in _object_values(owner_plan, ("defines_symbols",))
                    )
                    or any(
                        path and path in _object_text(plan)
                        for path in active_owner_paths
                    )
                )
            ]
        )
        if required:
            if not owner_work_packages:
                missing_owner_units.append(unit_id)
            if not owner_file_plans:
                missing_file_plan_units.append(unit_id)
            elif owner_paths and not active_owner_paths:
                support_only_file_units.append(unit_id)
            if owner_file_plans and not active_owner_paths:
                missing_active_owner_units.append(unit_id)
            if active_owner_paths and not route_paths:
                missing_route_units.append(unit_id)
        rows.append(
            {
                "unit_id": unit_id,
                "requires_active_route": required,
                "owner_work_package_ids": owner_work_packages,
                "owner_file_paths": owner_paths,
                "active_owner_paths": active_owner_paths,
                "called_by_or_route_paths": route_paths,
                "support_only_file_paths": [
                    path for path in owner_paths if path not in set(active_owner_paths)
                ],
                "closure_status": (
                    "closed"
                    if (not required or (owner_work_packages and active_owner_paths and route_paths))
                    else "needs_attention"
                ),
            }
        )

    blocking_units = _dedupe(
        missing_owner_units
        + missing_file_plan_units
        + support_only_file_units
        + missing_active_owner_units
        + missing_route_units
    )
    status = "passed" if not blocking_units else "needs_attention"
    return {
        "status": status,
        "unit_count": len(active_units),
        "required_unit_count": sum(1 for row in rows if row["requires_active_route"]),
        "closed_required_unit_count": sum(
            1
            for row in rows
            if row["requires_active_route"] and row["closure_status"] == "closed"
        ),
        "unit_routes": rows,
        "missing_owner_units": _dedupe(missing_owner_units),
        "missing_file_plan_units": _dedupe(missing_file_plan_units),
        "support_only_file_units": _dedupe(support_only_file_units),
        "missing_active_owner_units": _dedupe(missing_active_owner_units),
        "missing_route_units": _dedupe(missing_route_units),
        "blocking_units": blocking_units,
        "notes": [
            "Every implementation-bearing prepare unit has active owner and route file coverage."
            if status == "passed"
            else "Implementation-bearing prepare units missing active owner/route coverage: "
            + ", ".join(blocking_units[:12]),
            "README/config/registry/test-only ownership is advisory and does not close method, dataset, model, baseline, metric, training, or evaluation units.",
        ],
    }


def infer_required_surface_groups(text: str) -> dict[str, dict[str, Any]]:
    policy = _quality_policy()
    lowered = str(text or "").lower()
    tokens = set(_WORD_RE.findall(lowered))
    required: dict[str, dict[str, Any]] = {}
    for group, rule in policy.surface_rules.items():
        hits: list[str] = []
        for keyword in rule["keywords"]:
            if keyword in tokens or keyword in lowered:
                hits.append(keyword)
        if hits:
            required[group] = {
                "matched_keywords": _dedupe(hits),
                "accepted_surfaces": list(rule["surfaces"]),
            }
    return required


def expected_groups_from_units(units: list[Any]) -> dict[str, dict[str, Any]]:
    return infer_required_surface_groups("\n".join(_object_text(unit) for unit in units))


def _contract_category_names(contract: dict[str, Any], category: str) -> list[str]:
    names: list[str] = []
    for item in list(contract.get(category, []) or []):
        if isinstance(item, dict):
            if category == "parameter_sweeps":
                name = str(item.get("name", "") or "").strip()
                values = [str(value) for value in list(item.get("values", []) or []) if str(value).strip()]
                names.append(name + (f"[{','.join(values[:8])}]" if values else ""))
            else:
                names.append(str(item.get("name", "") or "").strip())
        else:
            names.append(str(item or "").strip())
    return _dedupe([item for item in names if item])


def _claim_inventory_coverage_report(required_contract: dict[str, Any], candidate_text: str) -> dict[str, Any]:
    """Measure whether a stage preserves the paper-derived claim inventory.

    This is intentionally stricter than group coverage. A unit set that merely
    says "implement baselines" should not pass if the paper names Azure-SFT,
    LoRA, CoT, Table 3, and Figure 3 explicitly.
    """

    categories = (
        "named_experiments",
        "environments",
        "datasets",
        "methods",
        "metrics",
        "artifacts",
        "parameter_sweeps",
        "trend_obligations",
        "protocol_obligations",
        "fixed_hyperparameters",
    )
    required_by_category = {
        category: _contract_category_names(required_contract, category)
        for category in categories
    }
    evidence_gaps = evidence_contract_gaps(required_contract, candidate_text)
    implementation_gaps = implementation_obligation_gaps(required_contract, candidate_text)
    missing_by_category: dict[str, list[str]] = {
        key: [str(item) for item in list(values or []) if str(item).strip()]
        for key, values in {**evidence_gaps, **implementation_gaps}.items()
        if values
    }
    total = sum(len(items) for items in required_by_category.values()) + len(
        _contract_category_names(required_contract, "implementation_obligations")
    )
    missing_total = sum(len(items) for items in missing_by_category.values())
    covered = max(0, total - missing_total)
    coverage_ratio = (covered / total) if total else 1.0
    artifact_total = len(required_by_category.get("artifacts", []))
    artifact_missing = len(missing_by_category.get("artifacts", []))
    artifact_coverage_ratio = ((artifact_total - artifact_missing) / artifact_total) if artifact_total else 1.0
    return {
        "total_claim_items": total,
        "covered_claim_items": covered,
        "missing_claim_items": missing_total,
        "coverage_ratio": coverage_ratio,
        "artifact_total": artifact_total,
        "artifact_missing": artifact_missing,
        "artifact_coverage_ratio": artifact_coverage_ratio,
        "required_by_category": required_by_category,
        "missing_by_category": missing_by_category,
    }


def claim_inventory_quality_issues(report: dict[str, Any], *, stage_label: str) -> list[str]:
    policy = _quality_policy()
    issues: list[str] = []
    total = int(report.get("total_claim_items", 0) or 0)
    coverage_ratio = float(report.get("coverage_ratio", 1.0) or 0.0)
    artifact_total = int(report.get("artifact_total", 0) or 0)
    artifact_ratio = float(report.get("artifact_coverage_ratio", 1.0) or 0.0)
    if total >= 12 and coverage_ratio < policy.prepare_min_claim_coverage:
        issues.append(
            f"{stage_label} preserves only {coverage_ratio:.0%} of paper-derived claim inventory "
            f"({int(report.get('covered_claim_items', 0) or 0)}/{total})"
        )
    if artifact_total >= 3 and artifact_ratio < policy.prepare_min_artifact_coverage:
        issues.append(
            f"{stage_label} preserves only {artifact_ratio:.0%} of named table/figure/artifact inventory "
            f"({artifact_total - int(report.get('artifact_missing', 0) or 0)}/{artifact_total})"
        )
    for category, values in dict(report.get("missing_by_category", {}) or {}).items():
        missing = [str(item) for item in list(values or []) if str(item).strip()]
        if not missing:
            continue
        issues.append(
            f"{stage_label} missing paper-derived {category}: "
            + ",".join(missing[:10])
        )
    return issues


def unit_extraction_quality_report(*, paper_text: str, units: list[Any]) -> dict[str, Any]:
    policy = _quality_policy()
    required = infer_required_surface_groups(paper_text)
    evidence_contract = infer_evidence_contract(paper_text)
    unit_text = "\n".join(_object_text(unit) for unit in units)
    evidence_gaps = evidence_contract_gaps(evidence_contract, unit_text)
    implementation_gaps = implementation_obligation_gaps(evidence_contract, unit_text)
    claim_coverage = _claim_inventory_coverage_report(evidence_contract, unit_text)
    surfaces = set(_surface_values(units))
    missing: list[str] = []
    covered: list[str] = []
    for group, payload in required.items():
        accepted = {str(surface).lower() for surface in payload.get("accepted_surfaces", [])}
        if _covers_group(group, surfaces, "", accepted):
            covered.append(group)
        else:
            missing.append(group)
    implementation_unit_count = sum(
        1 for unit in units if set(_surface_values([unit])).intersection(policy.implementation_surfaces)
    )
    support_only_count = sum(
        1
        for unit in units
        if set(_surface_values([unit])) and set(_surface_values([unit])).issubset(policy.support_only_surfaces)
    )
    protocol_bias = support_only_count > implementation_unit_count and bool(required)
    decision_value_gaps = _decision_value_gaps(units, id_field="unit_id")
    status = (
        "passed"
        if (
            not missing
            and not protocol_bias
            and not decision_value_gaps
            and not evidence_gaps
            and not implementation_gaps
            and not claim_inventory_quality_issues(claim_coverage, stage_label="unit extraction")
        )
        else "needs_attention"
    )
    return {
        "status": status,
        "required_groups": required,
        "covered_groups": covered,
        "missing_groups": missing,
        "evidence_contract": flatten_evidence_contract(evidence_contract),
        "evidence_contract_gaps": evidence_gaps,
        "implementation_obligation_gaps": implementation_gaps,
        "claim_inventory_coverage": claim_coverage,
        "implementation_unit_count": implementation_unit_count,
        "support_only_unit_count": support_only_count,
        "protocol_bias": protocol_bias,
        "decision_value_gaps": decision_value_gaps,
        "unit_count": len(units),
        "notes": [
            "Missing implementation groups before planning: " + ", ".join(missing)
            if missing
            else "All inferred implementation groups are represented in extracted units.",
            "Support/protocol-only units outnumber implementation units."
            if protocol_bias
            else "Implementation units are not dominated by support-only protocol units.",
            "Units missing hypothesis/decision fields: "
            + ", ".join(
                f"{item['id']}({','.join(item['missing_fields'])})"
                for item in decision_value_gaps[:12]
            )
            if decision_value_gaps
            else "Decision-bearing units include hypothesis and decision value.",
            "Missing paper-derived evidence obligations in extracted units: "
            + "; ".join(f"{key}={','.join(values[:8])}" for key, values in evidence_gaps.items())
            if evidence_gaps
            else "Extracted units preserve inferred paper-derived experiment/method/parameter evidence obligations.",
            "Missing executable implementation obligations in extracted units: "
            + "; ".join(f"{key}={','.join(values[:8])}" for key, values in implementation_gaps.items())
            if implementation_gaps
            else "Extracted units preserve executable implementation obligations.",
            "Claim inventory coverage issues: "
            + "; ".join(claim_inventory_quality_issues(claim_coverage, stage_label="unit extraction")[:8])
            if claim_inventory_quality_issues(claim_coverage, stage_label="unit extraction")
            else "Extracted units preserve the paper-derived claim inventory at sufficient density.",
        ],
    }


def _unit_identifier(unit: Any, index: int) -> str:
    return str(_get_value(unit, "unit_id") or f"unit_{index:03d}").strip()


def _nonempty_list_value(item: Any, name: str) -> list[Any]:
    value = _get_value(item, name)
    if isinstance(value, (list, tuple, set)):
        return [part for part in value if str(part or "").strip()]
    if str(value or "").strip():
        return [value]
    return []


def _active_prepare_units(units: list[Any]) -> list[Any]:
    return [
        unit
        for unit in list(units or [])
        if str(_get_value(unit, "status") or "active").strip().lower() not in {"inactive", "dropped", "out_of_scope"}
    ]


def _unit_field_gaps(units: list[Any]) -> list[dict[str, Any]]:
    gaps: list[dict[str, Any]] = []
    required_scalar_fields = ("statement",)
    required_list_fields = (
        "paper_evidence",
        "source_paragraph_ids",
        "verification_targets",
        "implementation_surfaces",
        "code_obligations",
    )
    for index, unit in enumerate(units, start=1):
        missing: list[str] = []
        for field in required_scalar_fields:
            if not str(_get_value(unit, field) or "").strip():
                missing.append(field)
        for field in required_list_fields:
            if not _nonempty_list_value(unit, field):
                missing.append(field)
        if missing:
            gaps.append({"unit_id": _unit_identifier(unit, index), "missing_fields": missing})
    return gaps


def _generic_fallback_unit_ids(units: list[Any]) -> list[str]:
    policy = _quality_policy()
    return [
        _unit_identifier(unit, index)
        for index, unit in enumerate(units, start=1)
        if _unit_identifier(unit, index) in policy.generic_paper_unit_ids
    ]


def _specialized_paper_unit_ids(units: list[Any]) -> list[str]:
    policy = _quality_policy()
    return [
        _unit_identifier(unit, index)
        for index, unit in enumerate(units, start=1)
        if _unit_identifier(unit, index).startswith("paper_")
        and _unit_identifier(unit, index) not in policy.generic_paper_unit_ids
    ]


def _forbidden_unit_hits(units: list[Any]) -> list[dict[str, Any]]:
    policy = _quality_policy()
    hits: list[dict[str, Any]] = []
    for index, unit in enumerate(units, start=1):
        text = _object_text_raw(unit).lower()
        matched = [token for token in policy.prepare_forbidden_unit_tokens if token in text]
        if matched:
            hits.append({"unit_id": _unit_identifier(unit, index), "tokens": matched})
    return hits


def _prepared_reference_repositories(reference_repo_preparation: dict[str, Any]) -> list[dict[str, Any]]:
    prepared = reference_repo_preparation.get("prepared_repositories", [])
    if not isinstance(prepared, list):
        return []
    return [dict(item) for item in prepared if isinstance(item, dict)]


def _requested_reference_repositories(reference_repo_preparation: dict[str, Any]) -> list[dict[str, Any]]:
    requested = reference_repo_preparation.get("requested_repositories", [])
    if not isinstance(requested, list):
        return []
    return [dict(item) for item in requested if isinstance(item, dict)]


def _failed_reference_repositories(reference_repo_preparation: dict[str, Any]) -> list[dict[str, Any]]:
    failed = reference_repo_preparation.get("failed_repositories", [])
    if not isinstance(failed, list):
        return []
    return [dict(item) for item in failed if isinstance(item, dict)]


def _resolved_unprepared_reference_repositories(reference_repo_preparation: dict[str, Any]) -> list[dict[str, Any]]:
    """Return resolved reference candidates that cannot be silently downgraded."""
    requested = _requested_reference_repositories(reference_repo_preparation)
    prepared_ids = {
        str(item.get("ref_id") or item.get("repository_url") or "").strip()
        for item in _prepared_reference_repositories(reference_repo_preparation)
    }
    failed_by_ref_id = {
        str(item.get("ref_id") or item.get("repository_url") or "").strip(): item
        for item in _failed_reference_repositories(reference_repo_preparation)
    }
    unresolved: list[dict[str, Any]] = []
    for item in requested:
        ref_key = str(item.get("ref_id") or item.get("repository_url") or "").strip()
        if ref_key in prepared_ids:
            continue
        repository_url = str(item.get("repository_url") or "").strip()
        resolve_status = str(item.get("resolve_status") or item.get("status") or "").strip().lower()
        repository_type = str(item.get("repository_type") or "").strip().lower()
        repository_origin = str(item.get("repository_origin") or "").strip().lower()
        is_resolved_candidate = bool(repository_url) or resolve_status == "found" or repository_type in {
            "official",
            "reference",
            "reproduction",
        } or repository_origin == "official"
        if not is_resolved_candidate:
            continue
        failed = failed_by_ref_id.get(ref_key, {})
        unresolved.append(
            {
                "ref_id": ref_key,
                "title": str(item.get("title") or "").strip(),
                "repository_url": repository_url,
                "repository_origin": repository_origin,
                "repository_type": repository_type,
                "status": str(failed.get("status") or resolve_status or "unprepared").strip(),
                "error_message": str(failed.get("error_message") or item.get("resolve_reason") or "").strip(),
            }
        )
    return unresolved


def _survey_payloads(reference_repo_surveys: list[Any]) -> list[Any]:
    return [item for item in list(reference_repo_surveys or []) if item is not None]


def _survey_ref_id(survey: Any) -> str:
    return str(_get_value(survey, "ref_id") or "").strip()


def _survey_symbol_evidence(survey: Any) -> list[Any]:
    value = _get_value(survey, "symbol_evidence")
    return list(value or []) if isinstance(value, (list, tuple)) else []


def _survey_actionability_label(survey: Any, prepared_item: Any | None = None) -> str:
    """Classify whether a prepared repo can ground units at symbol/path level."""
    symbol_evidence = _survey_symbol_evidence(survey)
    protocol_clues = _nonempty_list_value(survey, "protocol_clues")
    likely_reusable_files = _nonempty_list_value(survey, "likely_reusable_files")
    source_file_count = int(_get_value(survey, "source_file_count") or 0)
    repository_origin = str(_get_value(survey, "repository_origin") or "").strip().lower()
    repository_type = str(_get_value(survey, "repository_type") or "").strip().lower()
    search_only = bool(_get_value(prepared_item, "search_only")) if prepared_item is not None else False

    has_actionable_evidence = bool(symbol_evidence) and bool(source_file_count)
    has_protocol_shape = bool(protocol_clues or likely_reusable_files) and bool(source_file_count)
    if has_actionable_evidence:
        return "actionable_symbol_grounding"
    if source_file_count <= 0 and not symbol_evidence:
        if search_only:
            return "search_only"
        if repository_origin == "community" or repository_type in {"community", "explicit", "reproduction"}:
            return "community_weak"
        return "non_actionable"
    if has_protocol_shape:
        return "protocol_only"
    return "weak_or_empty"


def _is_actionable_reference_evidence(evidence: dict[str, Any]) -> bool:
    """Return whether a ref evidence row can ground implementation work."""
    policy = _quality_policy()
    path = str(evidence.get("file_path") or "").strip().lower()
    kind = str(evidence.get("symbol_kind") or "").strip().lower()
    symbol_name = str(evidence.get("symbol_name") or "").strip().lower()
    if not path and not symbol_name:
        return False
    if any(token in path for token in policy.weak_reference_path_tokens):
        return False
    if kind in policy.non_actionable_symbol_kinds:
        return False
    if path and not path.endswith(policy.actionable_reference_extensions):
        return False
    return True


def _has_self_contained_paper_evidence(unit: Any) -> bool:
    """Return whether a unit can stand on paper/addendum evidence alone."""
    paper_evidence = _nonempty_list_value(unit, "paper_evidence")
    source_paragraph_ids = _nonempty_list_value(unit, "source_paragraph_ids")
    statement = str(_get_value(unit, "statement") or "").strip()
    return bool(statement and paper_evidence and source_paragraph_ids)


def _survey_quality_report(reference_repo_preparation: dict[str, Any], reference_repo_surveys: list[Any]) -> dict[str, Any]:
    prepared = _prepared_reference_repositories(reference_repo_preparation)
    surveys = _survey_payloads(reference_repo_surveys)
    prepared_by_ref_id = {
        str(item.get("ref_id") or "").strip(): item
        for item in prepared
        if isinstance(item, dict) and str(item.get("ref_id") or "").strip()
    }
    survey_by_ref_id = {_survey_ref_id(survey): survey for survey in surveys if _survey_ref_id(survey)}
    missing_surveys = [
        str(item.get("ref_id") or item.get("repository_url") or item.get("title") or "").strip()
        for item in prepared
        if str(item.get("ref_id") or "").strip() not in survey_by_ref_id
    ]
    survey_rows: list[dict[str, Any]] = []
    actionable_count = 0
    for survey in surveys:
        symbol_evidence = _survey_symbol_evidence(survey)
        protocol_clues = _nonempty_list_value(survey, "protocol_clues")
        likely_reusable_files = _nonempty_list_value(survey, "likely_reusable_files")
        source_file_count = int(_get_value(survey, "source_file_count") or 0)
        actionability = _survey_actionability_label(
            survey,
            prepared_by_ref_id.get(_survey_ref_id(survey)),
        )
        has_actionable_evidence = actionability == "actionable_symbol_grounding"
        has_protocol_shape = actionability == "protocol_only"
        if has_actionable_evidence:
            actionable_count += 1
        survey_rows.append(
            {
                "ref_id": _survey_ref_id(survey),
                "title": str(_get_value(survey, "title") or "").strip(),
                "repository_url": str(_get_value(survey, "repository_url") or "").strip(),
                "repository_origin": str(_get_value(survey, "repository_origin") or "").strip(),
                "status": str(_get_value(survey, "status") or "").strip(),
                "source_file_count": source_file_count,
                "symbol_evidence_count": len(symbol_evidence),
                "protocol_clue_count": len(protocol_clues),
                "likely_reusable_file_count": len(likely_reusable_files),
                "quality": actionability,
            }
        )
    return {
        "prepared_reference_count": len(prepared),
        "survey_count": len(surveys),
        "actionable_survey_count": actionable_count,
        "missing_survey_ref_ids": [item for item in missing_surveys if item],
        "surveys": survey_rows,
    }


def _unit_reference_grounding(units: list[Any], reference_repo_surveys: list[Any], *, prepared_reference_count: int) -> dict[str, Any]:
    evidence_by_unit: dict[str, list[dict[str, Any]]] = {}
    weak_evidence_by_unit: dict[str, int] = {}
    for survey in _survey_payloads(reference_repo_surveys):
        ref_id = _survey_ref_id(survey)
        for evidence in _survey_symbol_evidence(survey):
            matched_unit_ids = _nonempty_list_value(evidence, "matched_unit_ids")
            if not matched_unit_ids:
                continue
            summary = {
                "ref_id": ref_id or str(_get_value(evidence, "ref_id") or "").strip(),
                "evidence_id": str(_get_value(evidence, "evidence_id") or "").strip(),
                "file_path": str(_get_value(evidence, "file_path") or "").strip(),
                "symbol_name": str(_get_value(evidence, "symbol_name") or "").strip(),
                "symbol_kind": str(_get_value(evidence, "symbol_kind") or "").strip(),
                "score": float(_get_value(evidence, "score") or 0.0),
            }
            if not summary["file_path"] and not summary["symbol_name"]:
                continue
            for unit_id in matched_unit_ids:
                key = str(unit_id or "").strip()
                if key:
                    if _is_actionable_reference_evidence(summary):
                        evidence_by_unit.setdefault(key, []).append(summary)
                    else:
                        weak_evidence_by_unit[key] = weak_evidence_by_unit.get(key, 0) + 1

    unit_rows: list[dict[str, Any]] = []
    missing_ref_grounding_units: list[str] = []
    for index, unit in enumerate(units, start=1):
        unit_id = _unit_identifier(unit, index)
        evidence = evidence_by_unit.get(unit_id, [])
        if prepared_reference_count <= 0:
            unit_rows.append(
                {
                    "unit_id": unit_id,
                    "grounding_status": "paper_only_no_reference_evidence",
                    "paper_only_reason": "No prepared reference repositories were available; carry this unit forward from paper/addendum evidence only.",
                    "reference_evidence": [],
                    "weak_reference_evidence_count": weak_evidence_by_unit.get(unit_id, 0),
                }
            )
            continue
        if evidence:
            unit_rows.append(
                {
                    "unit_id": unit_id,
                    "grounding_status": "grounded_to_reference_symbol",
                    "reference_evidence": sorted(evidence, key=lambda item: (-item["score"], item["ref_id"], item["file_path"]))[:6],
                    "weak_reference_evidence_count": weak_evidence_by_unit.get(unit_id, 0),
                }
            )
        elif _has_self_contained_paper_evidence(unit):
            unit_rows.append(
                {
                    "unit_id": unit_id,
                    "grounding_status": "self_contained_paper_grounded",
                    "paper_only_reason": (
                        "Prepared references did not expose a matching symbol, but the unit has "
                        "paper evidence and source paragraph ids for self-contained implementation."
                    ),
                    "reference_evidence": [],
                    "weak_reference_evidence_count": weak_evidence_by_unit.get(unit_id, 0),
                }
            )
        else:
            missing_ref_grounding_units.append(unit_id)
            unit_rows.append(
                {
                    "unit_id": unit_id,
                    "grounding_status": "missing_reference_grounding",
                    "reference_evidence": [],
                    "weak_reference_evidence_count": weak_evidence_by_unit.get(unit_id, 0),
                }
            )
    grounded_count = sum(1 for row in unit_rows if row.get("grounding_status") == "grounded_to_reference_symbol")
    return {
        "unit_grounding": unit_rows,
        "grounded_unit_count": grounded_count,
        "missing_ref_grounding_units": missing_ref_grounding_units,
        "paper_only_unit_count": sum(1 for row in unit_rows if row.get("grounding_status") == "paper_only_no_reference_evidence"),
        "self_contained_paper_unit_count": sum(1 for row in unit_rows if row.get("grounding_status") == "self_contained_paper_grounded"),
        "weak_reference_evidence_count": sum(weak_evidence_by_unit.values()),
    }


def prepare_quality_gate_report(
    *,
    paper_text: str,
    units: list[Any],
    reference_repo_preparation: dict[str, Any],
    reference_repo_surveys: list[Any],
) -> dict[str, Any]:
    """Build a deterministic prepare-phase gate before any planning/generation spend."""
    policy = _quality_policy()
    active_units = _active_prepare_units(units)
    unit_quality = unit_extraction_quality_report(paper_text=paper_text, units=active_units)
    requested_repositories = _requested_reference_repositories(reference_repo_preparation)
    prepared_repositories = _prepared_reference_repositories(reference_repo_preparation)
    resolved_unprepared_repositories = _resolved_unprepared_reference_repositories(reference_repo_preparation)
    survey_quality = _survey_quality_report(reference_repo_preparation, reference_repo_surveys)
    grounding = _unit_reference_grounding(
        active_units,
        reference_repo_surveys,
        prepared_reference_count=len(prepared_repositories),
    )
    field_gaps = _unit_field_gaps(active_units)
    fallback_unit_ids = _generic_fallback_unit_ids(active_units)
    specialized_paper_unit_ids = _specialized_paper_unit_ids(active_units)
    fallback_dominated = (
        bool(active_units)
        and len(fallback_unit_ids) > max(5, len(active_units) / 2)
        and len(specialized_paper_unit_ids) < 3
    )
    forbidden_hits = _forbidden_unit_hits(active_units)
    blocking_reasons: list[str] = []
    warnings: list[str] = []

    if len(active_units) < policy.prepare_min_active_units:
        blocking_reasons.append(
            f"low active unit count: {len(active_units)} < {policy.prepare_min_active_units}"
        )
    claim_coverage = dict(unit_quality.get("claim_inventory_coverage", {}) or {})
    total_claim_items = int(claim_coverage.get("total_claim_items", 0) or 0)
    minimum_units_for_claims = max(
        policy.prepare_min_active_units,
        min(40, max(0, total_claim_items // 3)),
    )
    if total_claim_items >= 30 and len(active_units) < minimum_units_for_claims:
        blocking_reasons.append(
            "unit extraction is too coarse for paper claim inventory: "
            + f"{len(active_units)} active units for {total_claim_items} inferred claim items; "
            + f"require at least {minimum_units_for_claims} atomic units"
        )
    if fallback_dominated:
        blocking_reasons.append(
            "fallback paper-derived units dominate extraction: "
            + f"{len(fallback_unit_ids)}/{len(active_units)} units are generic paper_* anchors "
            + f"and only {len(specialized_paper_unit_ids)} specialized paper-derived units are present"
        )
    elif active_units and len(fallback_unit_ids) > max(5, len(active_units) / 2):
        warnings.append(
            "Generic paper-derived anchors are numerous, but specialized paper-derived units are present: "
            + f"generic={len(fallback_unit_ids)}/{len(active_units)}, specialized={len(specialized_paper_unit_ids)}"
        )
    if unit_quality.get("status") != "passed":
        for group in list(unit_quality.get("missing_groups", []) or []):
            blocking_reasons.append(f"unit extraction missing implementation group: {group}")
        for group, values in dict(unit_quality.get("evidence_contract_gaps", {}) or {}).items():
            blocking_reasons.append(
                "unit extraction missing paper-derived evidence: "
                + str(group)
                + "="
                + ",".join(str(value) for value in list(values or [])[:8])
            )
        for group, values in dict(unit_quality.get("implementation_obligation_gaps", {}) or {}).items():
            blocking_reasons.append(
                "unit extraction missing executable implementation obligation: "
                + str(group)
                + "="
                + ",".join(str(value) for value in list(values or [])[:8])
            )
        for issue in claim_inventory_quality_issues(
            claim_coverage,
            stage_label="unit extraction",
        )[:12]:
            blocking_reasons.append(issue)
        if unit_quality.get("protocol_bias"):
            blocking_reasons.append("unit extraction is dominated by support/protocol-only units")
        for item in list(unit_quality.get("decision_value_gaps", []) or [])[:12]:
            blocking_reasons.append(
                "decision-bearing unit lacks hypothesis/decision fields: "
                + str(item.get("id", "unknown"))
                + " missing="
                + ",".join(str(field) for field in list(item.get("missing_fields", []) or []))
            )
    for gap in field_gaps[:16]:
        blocking_reasons.append(
            "unit lacks required prepare fields: "
            + str(gap.get("unit_id", "unknown"))
            + " missing="
            + ",".join(str(field) for field in list(gap.get("missing_fields", []) or []))
        )
    for hit in forbidden_hits:
        blocking_reasons.append(
            "unit contains forbidden evaluator/rubric token: "
            + str(hit.get("unit_id", "unknown"))
            + " tokens="
            + ",".join(str(token) for token in list(hit.get("tokens", []) or []))
        )

    if requested_repositories and not prepared_repositories:
        if resolved_unprepared_repositories:
            blocking_reasons.append(
                "resolved reference repositories were not prepared; repair reference acquisition before plan: "
                + "; ".join(
                    (
                        str(item.get("ref_id") or "unknown")
                        + "="
                        + str(item.get("repository_url") or item.get("title") or "unknown")
                        + " status="
                        + str(item.get("status") or "unprepared")
                    )
                    for item in resolved_unprepared_repositories[:6]
                )
            )
        else:
            warnings.append(
                "reference repository search produced no resolved trusted repository; carrying paper-derived units forward without reference grounding"
            )
    if prepared_repositories:
        if survey_quality["missing_survey_ref_ids"]:
            blocking_reasons.append(
                "prepared references missing surveys: "
                + ", ".join(survey_quality["missing_survey_ref_ids"][:8])
            )
        actionable_survey_count = int(survey_quality.get("actionable_survey_count", 0) or 0)
        weak_survey_count = sum(
            1
            for row in list(survey_quality.get("surveys", []) or [])
            if str(row.get("quality", "")).strip() in {"community_weak", "search_only", "non_actionable", "weak_or_empty"}
        )
        grounded_unit_count = int(grounding.get("grounded_unit_count", 0) or 0)
        required_grounded_units = min(len(active_units), max(3, int(len(active_units) * 0.35))) if active_units else 0
        if actionable_survey_count <= 0:
            warnings.append(
                "prepared references are non-actionable or weak; continuing with self-contained paper-grounded units "
                + f"without symbol grounding requirements ({weak_survey_count}/{int(survey_quality.get('survey_count', 0) or 0)} weak surveys)"
            )
        else:
            if grounded_unit_count <= 0:
                blocking_reasons.append("prepared references did not ground any unit to symbol/path evidence")
            elif grounded_unit_count < required_grounded_units:
                blocking_reasons.append(
                    "insufficient unit-to-reference grounding coverage: "
                    + f"{grounded_unit_count}/{len(active_units)} grounded, require at least {required_grounded_units}"
                )
        missing_grounding_units = list(grounding.get("missing_ref_grounding_units", []) or [])
        if missing_grounding_units:
            allowed_missing_units = max(4, int(len(active_units) * 0.25)) if active_units else 0
            message = (
                "Some units remain paper-only despite prepared refs: "
                + ", ".join(str(item) for item in missing_grounding_units[:12])
            )
            if len(missing_grounding_units) > allowed_missing_units:
                blocking_reasons.append(
                    "too many active units lack actionable reference grounding despite prepared refs: "
                    + f"{len(missing_grounding_units)}/{len(active_units)} missing, allow at most {allowed_missing_units}"
                )
                warnings.append(message)
            else:
                warnings.append(message)
    else:
        warnings.append("No prepared reference repositories; all units are marked paper_only_no_reference_evidence.")

    passed = not blocking_reasons
    status = "passed" if passed else "degraded_best_effort"
    return {
        "schema_version": "1.1",
        "status": status,
        "passed": passed,
        "degraded": not passed,
        "continue_with_best_effort": True,
        "next_action": "enter_plan" if passed else "enter_plan_degraded_best_effort",
        "repair_policy": (
            "repair_prepare_before_plan_when_budget_allows; "
            "continue_with_current_units_and_record_degraded_reasons_when_budget_is_exhausted"
        ),
        "blocking_reasons": blocking_reasons,
        "warnings": warnings,
        "unit_quality": unit_quality,
        "unit_field_gaps": field_gaps,
        "fallback_unit_ids": fallback_unit_ids,
        "generic_fallback_unit_ids": fallback_unit_ids,
        "specialized_paper_unit_ids": specialized_paper_unit_ids,
        "fallback_dominated": fallback_dominated,
        "forbidden_unit_hits": forbidden_hits,
        "reference_survey_quality": survey_quality,
        "resolved_unprepared_reference_repositories": resolved_unprepared_repositories,
        "reference_grounding": grounding,
        "active_unit_count": len(active_units),
        "minimum_active_units": policy.prepare_min_active_units,
        "requested_reference_count": len(requested_repositories),
        "prepared_reference_count": len(prepared_repositories),
        "notes": [
            "Prepare quality issues are diagnostic and repair-prioritized; they do not hard-stop planning.",
            "Reference acquisition is not grounding; units need symbol/path evidence when prepared refs exist.",
            "When no refs exist, units are explicitly marked paper_only_no_reference_evidence and must remain paper/addendum grounded downstream.",
        ],
    }


def work_package_quality_report(*, units: list[Any], work_packages: list[Any]) -> dict[str, Any]:
    required = expected_groups_from_units(units)
    evidence_contract = infer_evidence_contract("\n".join(_object_text(unit) for unit in units))
    package_surfaces = set(_surface_values(work_packages))
    package_text = "\n".join(_object_text(package) for package in work_packages)
    evidence_gaps = evidence_contract_gaps(evidence_contract, package_text)
    implementation_gaps = implementation_obligation_gaps(evidence_contract, package_text)
    claim_coverage = _claim_inventory_coverage_report(evidence_contract, package_text)
    missing: list[str] = []
    covered: list[str] = []
    for group, payload in required.items():
        accepted = {str(surface).lower() for surface in payload.get("accepted_surfaces", [])}
        if _covers_group(group, package_surfaces, package_text, accepted):
            covered.append(group)
        else:
            missing.append(group)
    owned_unit_ids = {
        str(unit_id).strip()
        for package in work_packages
        for unit_id in (_get_value(package, "owned_unit_ids") or [])
        if str(unit_id).strip()
    }
    active_unit_ids = {
        str(_get_value(unit, "unit_id") or "").strip()
        for unit in units
        if str(_get_value(unit, "unit_id") or "").strip()
    }
    unowned_units = sorted(active_unit_ids - owned_unit_ids)
    decision_value_gaps = _decision_value_gaps(work_packages, id_field="work_package_id")
    status = (
        "passed"
        if (
            not missing
            and not unowned_units
            and not decision_value_gaps
            and not evidence_gaps
            and not implementation_gaps
            and not claim_inventory_quality_issues(claim_coverage, stage_label="work packages")
        )
        else "needs_attention"
    )
    return {
        "status": status,
        "required_groups": required,
        "covered_groups": covered,
        "missing_groups": missing,
        "evidence_contract": flatten_evidence_contract(evidence_contract),
        "evidence_contract_gaps": evidence_gaps,
        "implementation_obligation_gaps": implementation_gaps,
        "claim_inventory_coverage": claim_coverage,
        "unowned_units": unowned_units,
        "decision_value_gaps": decision_value_gaps,
        "work_package_count": len(work_packages),
        "notes": [
            "Missing implementation groups in work packages: " + ", ".join(missing)
            if missing
            else "Work packages carry all inferred implementation groups.",
            "Unowned units: " + ", ".join(unowned_units[:12]) if unowned_units else "All active units are owned.",
            "Work packages missing hypothesis/decision fields: "
            + ", ".join(
                f"{item['id']}({','.join(item['missing_fields'])})"
                for item in decision_value_gaps[:12]
            )
            if decision_value_gaps
            else "Decision-bearing work packages include hypothesis and decision value.",
            "Missing paper-derived evidence obligations in work packages: "
            + "; ".join(f"{key}={','.join(values[:8])}" for key, values in evidence_gaps.items())
            if evidence_gaps
            else "Work packages preserve inferred paper-derived experiment/method/parameter evidence obligations.",
            "Missing executable implementation obligations in work packages: "
            + "; ".join(f"{key}={','.join(values[:8])}" for key, values in implementation_gaps.items())
            if implementation_gaps
            else "Work packages preserve executable implementation obligations.",
            "Claim inventory coverage issues: "
            + "; ".join(claim_inventory_quality_issues(claim_coverage, stage_label="work packages")[:8])
            if claim_inventory_quality_issues(claim_coverage, stage_label="work packages")
            else "Work packages preserve the paper-derived claim inventory at sufficient density.",
        ],
    }


def file_plan_quality_report(*, units: list[Any], work_packages: list[Any], file_plans: list[Any]) -> dict[str, Any]:
    policy = _quality_policy()
    required = expected_groups_from_units(units)
    evidence_contract = infer_evidence_contract(
        "\n".join([*[_object_text(unit) for unit in units], *[_object_text(package) for package in work_packages]])
    )
    file_surfaces = set(_surface_values(file_plans))
    file_text = "\n".join(_object_text(plan) for plan in file_plans)
    evidence_gaps = evidence_contract_gaps(evidence_contract, file_text)
    implementation_gaps = implementation_obligation_gaps(evidence_contract, file_text)
    claim_coverage = _claim_inventory_coverage_report(evidence_contract, file_text)
    missing: list[str] = []
    covered: list[str] = []
    for group, payload in required.items():
        accepted = {str(surface).lower() for surface in payload.get("accepted_surfaces", [])}
        if _covers_group(group, file_surfaces, file_text, accepted):
            covered.append(group)
        else:
            missing.append(group)
    weak_method_plans = [
        str(_get_value(plan, "target_file") or "").strip()
        for plan in file_plans
        if _target_path(plan).endswith(".py")
        and not _target_path(plan).endswith("__init__.py")
        if (
            set(_surface_values([plan])).intersection(policy.implementation_surfaces)
            or any(token in _object_text(plan) for token in policy.method_plan_signal_terms)
        )
        and not _object_values(plan, ("defines_symbols",))
        and not any(token in _object_text(plan) for token in policy.method_plan_symbol_terms)
    ]
    weak_method_plans = [item for item in weak_method_plans if item]
    generic_scaffold_plans = []
    for plan in file_plans:
        target = str(_get_value(plan, "target_file") or "").strip()
        if not target or not target.endswith(".py") or target.endswith("__init__.py"):
            continue
        purpose = str(_get_value(plan, "purpose") or "").strip().lower()
        prompt = str(_get_value(plan, "generation_prompt") or "").strip().lower()
        references = _object_values(plan, ("reference_ids", "context_sources"))
        obligations = _object_values(plan, ("method_obligations", "interface_contract", "review_points", "defines_symbols"))
        generic_purpose = purpose == f"implement {target.lower()}."
        generic_prompt = prompt.startswith(f"implement `{target.lower()}` according to its package contract")
        if (generic_purpose or generic_prompt) and not references and len(" ".join(obligations)) < 240:
            generic_scaffold_plans.append(target)
    generic_scaffold_plans = _dedupe(generic_scaffold_plans)
    skeleton_required_groups = _required_repository_skeleton_groups(units, work_packages)
    repository_skeleton = _repository_skeleton_report(file_plans, required_groups=skeleton_required_groups)
    skeleton_missing = list(repository_skeleton.get("missing_groups", []) or [])
    missing_work_package_file_plans = _missing_work_package_file_plans(work_packages, file_plans)
    decision_value_gaps = _decision_value_gaps(file_plans, id_field="target_file")
    active_route_contract = _active_route_contract_report(file_plans)
    unit_active_route_contract = _unit_active_route_report(
        units=units,
        work_packages=work_packages,
        file_plans=file_plans,
    )
    active_route_issues = (
        list(active_route_contract.get("unwired_symbols", []) or [])
        + list(active_route_contract.get("route_files_without_calls", []) or [])
        + list(active_route_contract.get("missing_route_files", []) or [])
    )
    unit_active_route_issues = list(unit_active_route_contract.get("blocking_units", []) or [])
    status = (
        "passed"
        if (
            not missing
            and not weak_method_plans
            and not skeleton_missing
            and not missing_work_package_file_plans
            and not decision_value_gaps
            and not active_route_issues
            and not unit_active_route_issues
            and not evidence_gaps
            and not implementation_gaps
            and not claim_inventory_quality_issues(claim_coverage, stage_label="file plans")
        )
        else "needs_attention"
    )
    return {
        "status": status,
        "required_groups": required,
        "covered_groups": covered,
        "missing_groups": missing,
        "evidence_contract": flatten_evidence_contract(evidence_contract),
        "evidence_contract_gaps": evidence_gaps,
        "implementation_obligation_gaps": implementation_gaps,
        "claim_inventory_coverage": claim_coverage,
        "weak_method_file_plans": weak_method_plans,
        "generic_scaffold_file_plans": generic_scaffold_plans,
        "repository_skeleton": repository_skeleton,
        "missing_work_package_file_plans": missing_work_package_file_plans,
        "decision_value_gaps": decision_value_gaps,
        "active_route_contract": active_route_contract,
        "unit_active_route_contract": unit_active_route_contract,
        "file_plan_count": len(file_plans),
        "work_package_count": len(work_packages),
        "notes": [
            "Missing implementation groups in file plans: " + ", ".join(missing)
            if missing
            else "File plans carry all inferred implementation groups.",
            "Method-level file plans without concrete symbols/checks: " + ", ".join(weak_method_plans[:12])
            if weak_method_plans
            else "Method-level file plans expose concrete symbol/check hints.",
            "Generic scaffold file plans without reference/task grounding: " + ", ".join(generic_scaffold_plans[:12])
            if generic_scaffold_plans
            else "File plans are grounded beyond generic scaffold prompts.",
            "Missing repository skeleton groups: " + ", ".join(skeleton_missing)
            if skeleton_missing
            else "Repository skeleton is represented in file plans.",
            "Work packages with no owned file plans: " + ", ".join(missing_work_package_file_plans[:12])
            if missing_work_package_file_plans
            else "Every work package owns at least one file plan.",
            "File plans missing hypothesis/decision fields: "
            + ", ".join(
                f"{item['id']}({','.join(item['missing_fields'])})"
                for item in decision_value_gaps[:12]
            )
            if decision_value_gaps
            else "Decision-bearing file plans include hypothesis and decision value.",
            "Active-route contract issues: " + "; ".join(list(active_route_contract.get("notes", []) or [])[:4])
            if active_route_issues
            else "Active-route contract is closed or not required.",
            "Unit active-route contract issues: "
            + "; ".join(list(unit_active_route_contract.get("notes", []) or [])[:4])
            if unit_active_route_issues
            else "Implementation-bearing prepare units have active file-route coverage.",
            "Missing paper-derived evidence obligations in file plans: "
            + "; ".join(f"{key}={','.join(values[:8])}" for key, values in evidence_gaps.items())
            if evidence_gaps
            else "File plans preserve inferred paper-derived experiment/method/parameter evidence obligations.",
            "Missing executable implementation obligations in file plans: "
            + "; ".join(f"{key}={','.join(values[:8])}" for key, values in implementation_gaps.items())
            if implementation_gaps
            else "File plans preserve executable implementation obligations.",
            "Claim inventory coverage issues: "
            + "; ".join(claim_inventory_quality_issues(claim_coverage, stage_label="file plans")[:8])
            if claim_inventory_quality_issues(claim_coverage, stage_label="file plans")
            else "File plans preserve the paper-derived claim inventory at sufficient density.",
        ],
    }


def file_plan_quality_issues(report: dict[str, Any]) -> list[str]:
    """Convert a file-plan quality report into deterministic repair issues."""
    issues: list[str] = []
    for group in list(report.get("missing_groups", []) or []):
        issues.append(f"quality gate missing implementation group in file plans: {group}")
    for path in list(report.get("weak_method_file_plans", []) or []):
        issues.append(f"quality gate weak method-level file plan lacks concrete symbols/checks: {path}")
    for path in list(report.get("generic_scaffold_file_plans", []) or []):
        issues.append(f"quality gate generic scaffold file plan lacks reference/task grounding: {path}")
    for item in list(report.get("decision_value_gaps", []) or []):
        issues.append(
            "quality gate decision-bearing file plan lacks hypothesis/decision fields: "
            + str(item.get("id", "unknown"))
            + " missing="
            + ",".join(str(field) for field in list(item.get("missing_fields", []) or []))
        )
    for group, values in dict(report.get("evidence_contract_gaps", {}) or {}).items():
        issues.append(
            "quality gate missing paper-derived evidence in file plans: "
            + str(group)
            + "="
            + ",".join(str(value) for value in list(values or [])[:8])
        )
    for group, values in dict(report.get("implementation_obligation_gaps", {}) or {}).items():
        issues.append(
            "quality gate missing executable implementation obligation in file plans: "
            + str(group)
            + "="
            + ",".join(str(value) for value in list(values or [])[:8])
        )
    issues.extend(
        "quality gate " + issue
        for issue in claim_inventory_quality_issues(
            dict(report.get("claim_inventory_coverage", {}) or {}),
            stage_label="file plans",
        )
    )
    skeleton = dict(report.get("repository_skeleton", {}) or {})
    for group in list(skeleton.get("missing_groups", []) or []):
        issues.append(f"quality gate missing repository skeleton group in file plans: {group}")
    for work_package_id in list(report.get("missing_work_package_file_plans", []) or []):
        issues.append(f"quality gate work package has no owned file plans: {work_package_id}")
    route_contract = dict(report.get("active_route_contract", {}) or {})
    for item in list(route_contract.get("unwired_symbols", []) or []):
        if isinstance(item, dict):
            issues.append(
                "quality gate implementation plan symbol is not wired into an active route: "
                + str(item.get("path", ""))
                + "::"
                + str(item.get("symbol", ""))
            )
    for path in list(route_contract.get("route_files_without_calls", []) or []):
        issues.append(f"quality gate active route file lacks calls_symbols: {path}")
    for item in list(route_contract.get("missing_route_files", []) or []):
        issues.append("quality gate missing active route file for implementation plan symbols: " + str(item))
    unit_route_contract = dict(report.get("unit_active_route_contract", {}) or {})
    for unit_id in list(unit_route_contract.get("missing_owner_units", []) or []):
        issues.append(f"quality gate implementation unit has no owning work package: {unit_id}")
    for unit_id in list(unit_route_contract.get("missing_file_plan_units", []) or []):
        issues.append(f"quality gate implementation unit has no owned file plan: {unit_id}")
    for unit_id in list(unit_route_contract.get("support_only_file_units", []) or []):
        issues.append(f"quality gate implementation unit is only assigned to support files: {unit_id}")
    for unit_id in list(unit_route_contract.get("missing_active_owner_units", []) or []):
        issues.append(f"quality gate implementation unit has no active Python owner file: {unit_id}")
    for unit_id in list(unit_route_contract.get("missing_route_units", []) or []):
        issues.append(f"quality gate implementation unit has no entry/train/eval/report route: {unit_id}")
    return issues
