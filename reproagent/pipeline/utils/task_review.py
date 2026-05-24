"""Task-level review helpers for reproagent generate and repair flows."""

import ast
import importlib.util
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from reproagent.pipeline.schemas import EvaluationDecision
from reproagent.pipeline.utils.contract_sanitizer import sanitize_contract_text, sanitize_task_contract

_IMPLEMENTATION_SURFACE_TOKENS = {
    "algorithm",
    "baseline",
    "baseline_or_ablation",
    "data_pipeline",
    "environment",
    "environment_adapter",
    "environment_factory",
    "evaluation",
    "metric",
    "metric_formula",
    "method",
    "model",
    "model_or_method",
    "policy",
    "policy_adapter",
    "policy_factory",
    "pretraining",
    "refinement",
    "refinement_algorithm",
    "training",
    "training_loop",
}

_IMPLEMENTATION_OBLIGATION_TOKENS = {
    "adapter",
    "agent",
    "algorithm",
    "baseline",
    "checkpoint",
    "class",
    "compute",
    "environment",
    "evaluate",
    "evaluation",
    "factory",
    "fidelity",
    "fine-tuning",
    "finetuning",
    "formula",
    "implement",
    "loop",
    "metric",
    "model",
    "policy",
    "pretrain",
    "refine",
    "refinement",
    "train",
    "training",
}

_PROTOCOL_ONLY_TOKENS = {
    "artifact",
    "contract",
    "dry-run",
    "dry_run",
    "manifest",
    "placeholder",
    "protocol",
    "readiness",
    "schema",
    "stub",
    "synthetic",
    "template",
}

_PLACEHOLDER_DOMINANCE_TOKENS = {
    "contract_placeholder_not_experimental_result",
    "does not train",
    "does not launch",
    "does not start",
    "dry-run",
    "dry_run",
    "executed\": false",
    "executed': false",
    "fallback",
    "not experimental result",
    "placeholder",
    "runtime_smoke",
    "smoke",
    "stub",
    "synthetic",
}


def _dedupe_nonempty_text(items: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for item in items:
        normalized = str(item or "").strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(normalized)
    return ordered


_CANONICAL_METHOD_PATHS = {
    "main.py",
    "run.py",
    "src/training.py",
    "src/evaluation.py",
    "src/refinement.py",
    "src/baselines.py",
    "src/experiments.py",
}

_DATASET_ALIASES = {
    "fashion_mnist": (
        "fashion-mnist",
        "fashion_mnist",
        "fashion mnist",
        "f-mnist",
        "f_mnist",
        "fmnist",
    ),
    "mnist": ("mnist",),
    "cifar10": ("cifar-10", "cifar_10", "cifar 10", "cifar10"),
    "cifar100": ("cifar-100", "cifar_100", "cifar 100", "cifar100"),
    "svhn": ("svhn",),
    "imagenet": ("imagenet", "image-net", "image net"),
    "imagenet_r": ("imagenet-r", "imagenet_r", "imagenet r"),
    "imagenet_a": ("imagenet-a", "imagenet_a", "imagenet a"),
    "imagenet_v2": ("imagenet-v2", "imagenet_v2", "imagenet v2"),
    "imagenet_sketch": ("imagenet-sketch", "imagenet_sketch", "imagenet sketch"),
    "objectnet": ("objectnet", "object-net", "object net"),
}
_IMPERFECT_SUPERVISION_TOKENS = (
    "class-imbalanced",
    "class imbalanced",
    "class_imbalanced",
    "imbalance",
    "imbalanced",
    "label noise",
    "label-noise",
    "noisy label",
    "noisy-label",
    "noisy",
    "imperfect supervision",
    "imperfect-supervision",
    "imperfect_supervision",
    "weak supervision",
    "weakly supervised",
)
_RUNTIME_FALLBACK_TOKENS = (
    "runtime_smoke",
    "synthetic",
    "fallback",
    "dry_run",
    "dry-run",
    "dry run",
    "stub",
    "placeholder",
)
_ROUTE_ACTION_PREFIXES = (
    "run",
    "plot",
    "write",
    "generate",
    "make",
    "build",
    "evaluate",
    "eval",
)


def _work_package_is_closed(project_files: dict[str, str], task_input: dict) -> tuple[bool, list[str]]:
    required_files = [
        str(path).strip()
        for path in list(task_input.get("work_package_required_files", []) or [])
        if str(path).strip()
    ]
    missing = [path for path in required_files if path not in project_files]
    return (not missing, missing)


def _smoke_module_name(surface_file: str) -> str:
    normalized = str(surface_file or "").strip().replace("\\", "/").rstrip("/")
    if normalized.endswith(".py"):
        normalized = normalized[:-3]
    parts = [item for item in normalized.split("/") if item and item != "__init__"]
    return ".".join(parts)


def _format_failed_check_suggestion(check: dict[str, Any]) -> str:
    name = str(check.get("name", "") or "").strip()
    details = str(check.get("details", "") or "").strip()
    error = str(check.get("error", "") or "").strip()
    pieces = [item for item in [name, details, error] if item]
    if not pieces:
        return "task review failed"
    return ": ".join(pieces)[:1400]


def _phrase_present(text: str, phrase: str) -> bool:
    value = str(phrase or "").strip().lower()
    if not value:
        return False
    variants = list(dict.fromkeys([
        value,
        value.replace("_", "-"),
        value.replace("_", " "),
        value.replace("-", "_"),
        value.replace("-", " "),
    ]))
    lowered = str(text or "").lower()
    for variant in variants:
        if not variant:
            continue
        pattern = r"(?<![a-z0-9])" + re.escape(variant).replace(r"\ ", r"[\s_-]+") + r"(?![a-z0-9])"
        if re.search(pattern, lowered):
            return True
    return False


def _dataset_mentions(text: str) -> set[str]:
    lowered = str(text or "").lower()
    mentions: set[str] = set()
    mnist_search = re.sub(
        r"fashion[\s_-]*mnist|f[\s_-]?mnist|fmnist",
        " ",
        lowered,
    )
    for canonical, aliases in _DATASET_ALIASES.items():
        search_text = mnist_search if canonical == "mnist" else lowered
        if any(_phrase_present(search_text, alias) for alias in aliases):
            mentions.add(canonical)
    return mentions


def _normal_numeric_token(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        numeric = float(raw)
    except ValueError:
        return raw
    if numeric.is_integer():
        return str(int(numeric))
    return ("%f" % numeric).rstrip("0").rstrip(".")


def _numbers_in_text(text: str) -> list[str]:
    values = [_normal_numeric_token(item) for item in re.findall(r"\d+(?:\.\d+)?", str(text or ""))]
    return [item for item in values if item]


def _declared_search_time_values(text: str) -> list[str]:
    lowered = str(text or "").lower()
    if not any(_phrase_present(lowered, token) for token in ("search_time", "search time", "search-time", "search_times", "search times")):
        return []
    values: list[str] = []
    pattern = re.compile(
        r"search[\s_-]*times?\s*(?:values?|grid|sweep|=|:|of|over|in|are|is|to)?\s*"
        r"(?:\[([^\]]+)\]|\(([^\)]+)\)|\{([^\}]+)\}|((?:\d+(?:\.\d+)?\s*(?:,|and|or|\s)+){1,}\d+(?:\.\d+)?))",
        flags=re.I,
    )
    for match in pattern.finditer(lowered):
        segment = next((group for group in match.groups() if group), "")
        values.extend(_numbers_in_text(segment))
    return list(dict.fromkeys(values))


def _declared_route_tokens(text: str) -> list[str]:
    lowered = str(text or "").lower()
    routes: list[str] = []
    routes.extend(f"figure_{number}" for number in re.findall(r"\bfig(?:ure)?\.?\s*(\d+)\b", lowered))
    routes.extend(f"table_{number}" for number in re.findall(r"\btable\.?\s*(\d+)\b", lowered))
    routes.extend(re.findall(r"\b(?:figure|fig|table)_\d+\b", lowered))
    routes.extend(
        token
        for token in re.findall(r"\b(?:run|plot|write|generate|evaluate|eval)_[a-z0-9_]*\b", lowered)
        if len(token) > 4
    )
    normalized: list[str] = []
    for route in routes:
        token = route.replace("fig_", "figure_").strip("_")
        if token and token not in normalized:
            normalized.append(token)
    return normalized


def _route_action_present(text: str, route_token: str) -> bool:
    lowered = str(text or "").lower()
    token = str(route_token or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not token:
        return False
    if token.startswith(tuple(f"{prefix}_" for prefix in _ROUTE_ACTION_PREFIXES)):
        route_stem = token
        direct_patterns = [
            rf"\bdef\s+{re.escape(route_stem)}\b",
            rf"\b{re.escape(route_stem)}\s*\(",
        ]
        if any(re.search(pattern, lowered) for pattern in direct_patterns):
            return True
    patterns = []
    for prefix in _ROUTE_ACTION_PREFIXES:
        patterns.extend([
            rf"\bdef\s+{prefix}_{re.escape(token)}\b",
            rf"\b{prefix}_{re.escape(token)}\s*\(",
            rf"\b{prefix}_{re.escape(token)}_",
        ])
    patterns.extend([
        rf"\b{re.escape(token)}_(?:writer|artifact|route|runner|experiment|evaluation)\b",
        rf"\b(?:writer|artifact|route|runner|experiment|evaluation)_{re.escape(token)}\b",
    ])
    return any(re.search(pattern, lowered) for pattern in patterns)


def _has_search_time_sweep_marker(text: str) -> bool:
    lowered = str(text or "").lower()
    if any(_phrase_present(lowered, token) for token in ("search_times", "search time grid", "search-time grid")):
        return True
    return bool(
        re.search(r"\bfor\s+[a-z_]*search[a-z_]*\s+in\b", lowered)
        or re.search(r"\bfor\s+\w+\s+in\s+[a-z_]*search[_a-z]*times\b", lowered)
    )


def declared_experiment_contract_gaps(
    contract_text: str,
    route_text: str,
    *,
    require_runtime_route: bool = True,
) -> list[dict[str, Any]]:
    """Return generic gaps between explicit experiment obligations and active code."""
    contract = sanitize_contract_text(contract_text).lower()
    route = sanitize_contract_text(route_text).lower()
    gaps: list[dict[str, Any]] = []

    expected_datasets = _dataset_mentions(contract)
    if expected_datasets and (
        any(_phrase_present(contract, token) for token in _IMPERFECT_SUPERVISION_TOKENS)
        or _phrase_present(contract, "dataset")
        or any(_phrase_present(contract, token) for token in ("figure", "table", "evaluation"))
    ):
        route_datasets = _dataset_mentions(route)
        unexpected = sorted(route_datasets - expected_datasets)
        expected_present = sorted(route_datasets & expected_datasets)
        if unexpected and not expected_present:
            gaps.append(
                {
                    "kind": "dataset_mismatch",
                    "expected": sorted(expected_datasets),
                    "found": unexpected,
                    "guidance": "Explicit dataset obligations must be reachable in code; a different hardcoded dataset is not enough.",
                }
            )

    declared_search_times = _declared_search_time_values(contract)
    if len(declared_search_times) >= 2:
        route_values = set(_numbers_in_text(route))
        missing_values = [value for value in declared_search_times if value not in route_values]
        if missing_values or not _has_search_time_sweep_marker(route):
            gaps.append(
                {
                    "kind": "search_time_sweep",
                    "expected": declared_search_times,
                    "missing": missing_values,
                    "guidance": "Search-time ablations must preserve the declared search_times sweep, not collapse to a fixed-k or fixed-epoch path.",
                }
            )

    if require_runtime_route:
        declared_routes = _declared_route_tokens(contract)
        if declared_routes:
            missing_active_routes = [
                route_token
                for route_token in declared_routes
                if not _route_action_present(route, route_token)
            ]
            fallback_markers = [
                token
                for token in _RUNTIME_FALLBACK_TOKENS
                if _phrase_present(route, token)
            ]
            if missing_active_routes and (
                fallback_markers
                or any(_phrase_present(contract, token) for token in ("figure", "table", "artifact", "route", "entrypoint", "evaluation"))
            ):
                gaps.append(
                    {
                        "kind": "runtime_route",
                        "expected": declared_routes,
                        "missing": missing_active_routes,
                        "fallback_markers": fallback_markers,
                        "guidance": "Declared figure/table or experiment routes must be wired into active runtime/reporting functions, not only registry or smoke payloads.",
                    }
                )
    return gaps


def format_declared_experiment_contract_gaps(gaps: list[dict[str, Any]]) -> str:
    pieces: list[str] = []
    for gap in gaps:
        kind = str(gap.get("kind", "") or "contract_gap")
        expected = ",".join(str(item) for item in list(gap.get("expected", []) or [])[:8])
        found = ",".join(str(item) for item in list(gap.get("found", []) or [])[:8])
        missing = ",".join(str(item) for item in list(gap.get("missing", []) or [])[:8])
        guidance = str(gap.get("guidance", "") or "")
        detail = kind
        if expected:
            detail += f" expected={expected}"
        if found:
            detail += f" found={found}"
        if missing:
            detail += f" missing={missing}"
        if guidance:
            detail += f": {guidance}"
        pieces.append(detail)
    return "; ".join(pieces)


def _local_import_roots(project_files: dict[str, str]) -> set[str]:
    roots: set[str] = set()
    for relative_path in project_files:
        normalized = str(relative_path or "").strip().replace("\\", "/").strip("/")
        if not normalized or not normalized.endswith(".py"):
            continue
        first_part = normalized.split("/", 1)[0]
        if first_part.endswith(".py"):
            first_part = first_part[:-3]
        if first_part:
            roots.add(first_part)
    return roots


def _is_stdlib_import(root_name: str) -> bool:
    root = str(root_name or "").strip()
    if not root:
        return True
    if root in sys.builtin_module_names:
        return True
    stdlib_names = getattr(sys, "stdlib_module_names", set())
    return root in stdlib_names


def _top_level_import_roots(tree: ast.Module) -> list[str]:
    roots: list[str] = []
    for statement in tree.body:
        if isinstance(statement, ast.Import):
            roots.extend(str(alias.name or "").split(".", 1)[0] for alias in statement.names)
        elif isinstance(statement, ast.ImportFrom):
            if int(statement.level or 0) > 0:
                continue
            roots.append(str(statement.module or "").split(".", 1)[0])
    return [item for item in roots if item and item != "__future__"]


def _missing_top_level_imports(project_files: dict[str, str], tree: ast.Module) -> list[str]:
    local_roots = _local_import_roots(project_files)
    missing: list[str] = []
    for root in _top_level_import_roots(tree):
        if root in local_roots or _is_stdlib_import(root):
            continue
        try:
            available = importlib.util.find_spec(root) is not None
        except Exception:
            available = False
        if not available and root not in missing:
            missing.append(root)
    return missing


def _task_requires_implementation_depth(task_input: dict, task_view: dict) -> bool:
    surfaces = " ".join(
        str(item).strip().lower()
        for source in (task_input, task_view)
        for item in list(source.get("implementation_surfaces", []) or [])
        if str(item).strip()
    )
    obligations = " ".join(
        str(item).strip().lower()
        for source in (task_input, task_view)
        for item in list(source.get("method_obligations", []) or [])
        if str(item).strip()
    )
    if any(token in surfaces for token in _IMPLEMENTATION_SURFACE_TOKENS):
        return True
    return any(token in obligations for token in _IMPLEMENTATION_OBLIGATION_TOKENS)


def _task_contract_text(task_input: dict, task_view: dict) -> str:
    parts: list[str] = []
    for source in (task_input, task_view):
        for key in (
            "method_obligations",
            "review_points",
            "implementation_surfaces",
            "writes_artifacts",
            "required_interfaces",
            "task_notes",
        ):
            value = source.get(key, [])
            if isinstance(value, dict):
                parts.extend(str(item or "") for item in value.values())
            elif isinstance(value, (list, tuple, set)):
                parts.extend(str(item or "") for item in value)
            else:
                parts.append(str(value or ""))
        parts.append(str(source.get("file_path", "") or ""))
        generation_context = source.get("generation_context", {})
        if isinstance(generation_context, dict):
            parts.append(str(generation_context))
    return sanitize_contract_text("\n".join(part for part in parts if str(part or "").strip())).lower()


def _implementation_depth_check(
    *,
    tree: ast.Module,
    content: str,
    task_input: dict,
    task_view: dict,
) -> dict[str, Any]:
    """Reject protocol/manifest-only files assigned real method obligations."""
    file_path = str(task_view.get("file_path", "") or task_input.get("file_path", "") or "")
    if not _task_requires_implementation_depth(task_input, task_view):
        return {
            "name": "implementation_depth",
            "task_id": task_view.get("task_id", ""),
            "file_path": file_path,
            "passed": True,
            "skipped": True,
            "error": "",
            "details": "Skipped because this file has no method-level implementation obligation.",
        }

    class_count = sum(isinstance(node, ast.ClassDef) for node in ast.walk(tree))
    function_count = sum(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in ast.walk(tree))
    statement_count = sum(isinstance(node, ast.stmt) for node in ast.walk(tree))
    lowered = str(content or "").lower()
    protocol_hits = sum(1 for token in _PROTOCOL_ONLY_TOKENS if token in lowered)
    executable_surface_count = class_count + function_count

    placeholder_markers = [
        marker
        for marker in ("notimplementederror", "todo", "pass  #", "pass\n", "placeholder", "stub")
        if marker in lowered
    ]
    shallow_protocol_file = (
        executable_surface_count == 0
        and protocol_hits >= 2
        and statement_count <= 12
    )
    insufficient_symbols = executable_surface_count == 0 and statement_count <= 6
    has_placeholder_surface = bool(placeholder_markers) and executable_surface_count <= 2

    passed = not (shallow_protocol_file or insufficient_symbols or has_placeholder_surface)
    details = (
        "Implementation-depth check passed: file exposes concrete classes/functions for its method obligations."
        if passed
        else (
            "File is assigned method-level obligations but looks like protocol/manifest/dry-run scaffolding "
            f"rather than implementation code: classes={class_count}, functions={function_count}, "
            f"statements={statement_count}, protocol_markers={protocol_hits}."
        )
    )
    return {
        "name": "implementation_depth",
        "task_id": task_view.get("task_id", ""),
        "file_path": file_path,
        "passed": passed,
        "skipped": False,
        "error": "" if passed else details,
        "details": details,
    }


def _placeholder_dominance_check(*, content: str, task_input: dict, task_view: dict) -> dict[str, Any]:
    """Reject canonical method files where placeholder/smoke paths dominate real obligations."""
    file_path = str(task_view.get("file_path", "") or task_input.get("file_path", "") or "").strip().replace("\\", "/")
    normalized_path = file_path.lower().strip("/")
    if not _task_requires_implementation_depth(task_input, task_view):
        return {
            "name": "placeholder_dominance",
            "task_id": task_view.get("task_id", ""),
            "file_path": file_path,
            "passed": True,
            "skipped": True,
            "error": "",
            "details": "Skipped because this file has no method-level implementation obligation.",
        }

    lowered = str(content or "").lower()
    marker_hits = sorted(token for token in _PLACEHOLDER_DOMINANCE_TOKENS if token in lowered)
    real_path_markers = sum(
        1
        for token in (
            "def train",
            "def evaluate",
            "def run_training",
            "def evaluate_model",
            "from src.",
            "get_dataset",
            "get_model",
            "metrics",
            "accuracy",
            "loss_curve",
            "selection_time",
        )
        if token in lowered
    )
    executable_surface_count = len(re.findall(r"^\s*(?:def|class)\s+", str(content or ""), flags=re.M))
    algorithm_markers = sum(
        1
        for token in (
            "loss.backward",
            ".backward(",
            "torch.",
            "gradient",
            "optimizer",
            "for _ in range",
            "for step in range",
            ".clamp(",
            "kl_div",
            "mse_loss",
            "cross_entropy",
            "train_step",
            "evaluate(",
            "json.dump",
            "savefig",
        )
        if token in lowered
    )
    bounded_fallback_context = ("except importerror" in lowered or "runtime_smoke" in lowered) and real_path_markers >= 4
    concrete_algorithm_context = (
        executable_surface_count >= 4
        and (real_path_markers >= 2 or algorithm_markers >= 3)
    )
    is_canonical_path = (
        normalized_path in _CANONICAL_METHOD_PATHS
        or normalized_path.startswith("scripts/")
        or normalized_path.endswith("/main.py")
        or any(token in normalized_path for token in ("train", "eval", "refine", "experiment", "baseline"))
    )
    obligation_text = " ".join(
        str(item or "").lower()
        for source in (task_input, task_view)
        for item in list(source.get("method_obligations", []) or [])
    )
    surfaces = {
        str(item or "").strip().lower()
        for source in (task_input, task_view)
        for item in list(source.get("implementation_surfaces", []) or [])
        if str(item or "").strip()
    }
    requires_canonical_method = bool(
        surfaces.intersection({"entrypoint", "training_loop", "training", "evaluation", "refinement", "refinement_algorithm", "baseline_or_ablation"})
        or any(token in obligation_text for token in ("canonical", "train", "evaluate", "refine", "baseline", "experiment"))
    )
    passed = not (
        is_canonical_path
        and requires_canonical_method
        and len(marker_hits) >= 3
        and not bounded_fallback_context
        and not concrete_algorithm_context
    )
    details = (
        "Placeholder-dominance check passed: placeholder/smoke markers do not dominate a canonical method file."
        if passed
        else (
            "Canonical method file is assigned training/evaluation/refinement obligations but is dominated by "
            "placeholder, dry-run, synthetic, smoke, or fallback markers: " + ", ".join(marker_hits[:10])
        )
    )
    return {
        "name": "placeholder_dominance",
        "task_id": task_view.get("task_id", ""),
        "file_path": file_path,
        "passed": passed,
        "skipped": False,
        "error": "" if passed else details,
        "details": details,
    }


def _full_implementation_route_check(*, content: str, task_input: dict, task_view: dict) -> dict[str, Any]:
    """Reject files where a paper-required full route is replaced by smoke/mock-only code."""
    file_path = str(task_view.get("file_path", "") or task_input.get("file_path", "") or "").strip().replace("\\", "/")
    contract_text = _task_contract_text(task_input, task_view).lower()
    if "full experiment-matrix route contract" not in contract_text and not any(
        token in contract_text
        for token in (
            "full data/model/training/evaluation route",
            "real loader",
            "model factory",
            "optimizer/refinement",
            "pairwise evaluation",
            "checkpoint/artifact",
        )
    ):
        return {
            "name": "full_implementation_route",
            "task_id": task_view.get("task_id", ""),
            "file_path": file_path,
            "passed": True,
            "skipped": True,
            "error": "",
            "details": "Skipped because this file has no full-route implementation contract.",
        }

    lowered = str(content or "").lower()
    placeholder_hits = [
        token
        for token in (
            "dry-run",
            "dry_run",
            "runtime_smoke",
            "smoke",
            "synthetic",
            "mock",
            "dummy",
            "placeholder",
            "fake",
            "deterministic placeholder",
            "contract artifact",
        )
        if token in lowered
    ]
    full_route_markers = [
        token
        for token in (
            "from_pretrained",
            "automodel",
            "autotokenizer",
            "load_dataset",
            "datasets.load",
            "torch.optim",
            "optimizer",
            ".backward(",
            "loss.backward",
            "save_pretrained",
            "save_checkpoint",
            "checkpoint",
            "train_test_split",
            "random.shuffle",
            "itertools.product",
            "for i,",
            "for j,",
            "cartesian",
            "pairwise",
            "exact_match",
            "file not found",
            "optional dependency",
            "raise importerror",
            "raise filenotfounderror",
        )
        if token in lowered
    ]
    executable_surface_count = len(re.findall(r"^\s*(?:def|class)\s+", str(content or ""), flags=re.M))
    has_real_route = len(full_route_markers) >= 4 and executable_surface_count >= 4
    mock_dominated = len(placeholder_hits) >= 3 and not has_real_route
    passed = not mock_dominated
    details = (
        "Full implementation route check passed: smoke/mock support does not dominate the required full route."
        if passed
        else (
            "Full-route contract appears to be satisfied mainly by smoke/mock/synthetic code. "
            "Implement lazy real loaders, train/evaluate/refine loops, pairwise evaluation, metrics, and checkpoint/artifact outputs. "
            f"placeholder_markers={placeholder_hits[:8]}, full_route_markers={full_route_markers[:8]}"
        )
    )
    return {
        "name": "full_implementation_route",
        "task_id": task_view.get("task_id", ""),
        "file_path": file_path,
        "passed": passed,
        "skipped": False,
        "error": "" if passed else details,
        "details": details,
    }


def _runtime_path_bypass_check(
    *,
    content: str,
    task_input: dict,
    task_view: dict,
    project_files: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Reject entrypoints that pass smoke by bypassing the generated implementation."""
    file_path = str(task_view.get("file_path", "") or task_input.get("file_path", "") or "").strip().replace("\\", "/")
    normalized_path = file_path.lower().strip("/")
    obligation_text = " ".join(
        str(item or "").lower()
        for source in (task_input, task_view)
        for key in ("method_obligations", "review_points")
        for item in list(source.get(key, []) or [])
    )
    surfaces = {
        str(item or "").strip().lower()
        for source in (task_input, task_view)
        for item in list(source.get("implementation_surfaces", []) or [])
        if str(item or "").strip()
    }
    is_entrypoint_or_runtime = (
        normalized_path.endswith("main.py")
        or normalized_path.startswith("scripts/")
        or "entrypoint" in surfaces
        or any(token in obligation_text for token in ("entrypoint", "orchestrate", "run experiment", "artifact generation"))
    )
    if not is_entrypoint_or_runtime:
        return {
            "name": "runtime_path_bypass",
            "task_id": task_view.get("task_id", ""),
            "file_path": file_path,
            "passed": True,
            "skipped": True,
            "error": "",
            "details": "Skipped because this file is not an entrypoint or runtime orchestration surface.",
        }

    lowered = str(content or "").lower()
    bypass_markers = [
        "generate_dry_run_artifacts",
        "dry_run_contract",
        "reproagent_validation_materializer",
        "dry-run schema",
        "not real experiment",
        "not a claimed experimental result",
        "using minimal",
        "using simulated",
        "synthetic result",
        "fake result",
        "module not available",
        "method not available",
        "data module not available",
        "model module not available",
    ]
    activation_markers = [
        "run_all_experiments",
        "run_benchmark",
        "run_training",
        "run_evaluation",
        "algorithm",
        "train_",
        "evaluate_",
        "baseline",
        "dataset",
    ]
    bypass_hits = [marker for marker in bypass_markers if marker in lowered]
    activation_count = sum(1 for marker in activation_markers if marker in lowered)
    route_gaps = [
        gap
        for gap in declared_experiment_contract_gaps(obligation_text, lowered, require_runtime_route=True)
        if gap.get("kind") == "runtime_route"
    ]
    if route_gaps and project_files is not None:
        closed, missing_files = _work_package_is_closed(project_files, task_input)
        if not closed:
            return {
                "name": "runtime_path_bypass",
                "task_id": task_view.get("task_id", ""),
                "file_path": file_path,
                "passed": True,
                "skipped": True,
                "error": "",
                "details": (
                    "Skipped runtime route closure because this work package is not closed yet: "
                    + ", ".join(missing_files[:8])
                ),
            }
    fallback_dominated_route = bool(route_gaps) and any(
        _phrase_present(lowered, marker)
        for marker in _RUNTIME_FALLBACK_TOKENS
    )
    passed = not (
        ("generate_dry_run_artifacts" in lowered and "run_all_experiments" not in lowered)
        or (len(bypass_hits) >= 2 and activation_count < 3)
        or fallback_dominated_route
    )
    failure_details = bypass_hits[:8]
    if route_gaps:
        failure_details.append(format_declared_experiment_contract_gaps(route_gaps))
    details = (
        "Runtime path check passed: entrypoint invokes generated implementation paths."
        if passed
        else (
            "Entrypoint appears to satisfy validation by dry-run/fallback artifacts instead of "
            "running generated dataset, method, evaluation, and artifact paths: "
            + ", ".join(item for item in failure_details if item)
        )
    )
    return {
        "name": "runtime_path_bypass",
        "task_id": task_view.get("task_id", ""),
        "file_path": file_path,
        "passed": passed,
        "skipped": False,
        "error": "" if passed else details,
        "details": details,
    }


def _declared_experiment_contract_check(
    *,
    content: str,
    task_input: dict,
    task_view: dict,
    project_files: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Reject explicit dataset/search-time/route contracts that are only smoke or registry-level."""
    file_path = str(task_view.get("file_path", "") or task_input.get("file_path", "") or "").strip().replace("\\", "/")
    contract_text = _task_contract_text(task_input, task_view)
    has_explicit_contract = bool(
        _dataset_mentions(contract_text)
        or _declared_search_time_values(contract_text)
        or _declared_route_tokens(contract_text)
    )
    if not has_explicit_contract:
        return {
            "name": "declared_experiment_contract",
            "task_id": task_view.get("task_id", ""),
            "file_path": file_path,
            "passed": True,
            "skipped": True,
            "error": "",
            "details": "Skipped because this file has no explicit dataset/search-time/figure/table route contract.",
        }

    gaps = declared_experiment_contract_gaps(contract_text, content, require_runtime_route=True)
    if gaps and _can_defer_result_route_contract(file_path):
        gaps = [gap for gap in gaps if gap.get("kind") != "runtime_route"]
    if gaps and project_files is not None and all(gap.get("kind") == "runtime_route" for gap in gaps):
        closed, missing_files = _work_package_is_closed(project_files, task_input)
        if not closed:
            return {
                "name": "declared_experiment_contract",
                "task_id": task_view.get("task_id", ""),
                "file_path": file_path,
                "passed": True,
                "skipped": True,
                "error": "",
                "details": (
                    "Skipped declared route closure because this work package is not closed yet: "
                    + ", ".join(missing_files[:8])
                ),
            }
    passed = not gaps
    details = (
        "Declared experiment contract check passed: dataset, search-time, and route obligations are visible in active code."
        if passed
        else "Declared experiment contract gaps: " + format_declared_experiment_contract_gaps(gaps)
    )
    return {
        "name": "declared_experiment_contract",
        "task_id": task_view.get("task_id", ""),
        "file_path": file_path,
        "passed": passed,
        "skipped": False,
        "error": "" if passed else details,
        "details": details,
    }


def _can_defer_result_route_contract(file_path: str) -> bool:
    """Return True for method/training/data files that should not own report routes."""
    normalized = str(file_path or "").strip().lower().replace("\\", "/")
    if not normalized:
        return False
    route_owner_tokens = (
        "eval",
        "evaluate",
        "report",
        "plot",
        "figure",
        "table",
        "artifact",
        "result",
        "main.py",
        "experiment",
        "registry",
    )
    if any(token in normalized for token in route_owner_tokens):
        return False
    basename = normalized.rsplit("/", 1)[-1]
    if basename in {"train.py", "training.py", "trainer.py"}:
        return True
    deferred_tokens = (
        "/trainer/",
        "/training/",
        "/models/",
        "/model/",
        "/methods/",
        "/method/",
        "/data/",
        "/dataset/",
        "/classifier/",
        "/adaptor/",
        "/adapter/",
        "/policy/",
        "/policies/",
    )
    return any(token in normalized for token in deferred_tokens)


def _toy_or_stub_environment_check(*, tree: ast.Module, content: str, task_input: dict, task_view: dict) -> dict[str, Any]:
    file_path = str(task_view.get("file_path", "") or task_input.get("file_path", "") or "").strip().replace("\\", "/")
    text = " ".join(
        str(item or "").lower()
        for source in (task_input, task_view)
        for key in ("implementation_surfaces", "method_obligations", "review_points")
        for item in list(source.get(key, []) or [])
    )
    requires_env_or_data = any(token in text for token in ("environment", "dataset", "data_pipeline", "rollout", "simulation"))
    if not requires_env_or_data:
        return {
            "name": "toy_or_stub_environment",
            "task_id": task_view.get("task_id", ""),
            "file_path": file_path,
            "passed": True,
            "skipped": True,
            "error": "",
            "details": "Skipped because this file has no environment/data obligation.",
        }

    class_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    function_names = [node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    lowered = str(content or "").lower()
    suspicious_symbols = [
        name
        for name in class_names + function_names
        if any(token in name.lower() for token in ("toy", "dummy", "stub", "synthetic", "fake"))
    ]
    suspicious_text = [token for token in ("toyenv", "dummy", "stub", "syntheticdataset", "fakeenv") if token in lowered]
    has_real_backend = any(
        token in lowered
        for token in (
            "environment_registry",
            "dataset_registry",
            "task_specs",
            "adapter",
            "optional_external_backend",
            "gym",
            "gymnasium",
            "mujoco",
            "dataset",
            "env_id",
            "benchmark",
            "loader",
        )
    )
    has_only_toy_surface = bool(suspicious_symbols or suspicious_text) and not has_real_backend
    passed = not has_only_toy_surface
    details = (
        "Environment/data check passed: no toy/stub implementation dominates the obligation."
        if passed
        else "Environment/data obligation appears satisfied mainly by toy/stub/synthetic symbols: "
        + ", ".join((suspicious_symbols + suspicious_text)[:8])
    )
    return {
        "name": "toy_or_stub_environment",
        "task_id": task_view.get("task_id", ""),
        "file_path": file_path,
        "passed": passed,
        "skipped": False,
        "error": "" if passed else details,
        "details": details,
    }


def _external_backend_route_check(*, content: str, task_input: dict, task_view: dict) -> dict[str, Any]:
    file_path = str(task_view.get("file_path", "") or task_input.get("file_path", "") or "").strip().replace("\\", "/")
    contract_text = _task_contract_text(task_input, task_view)
    lowered = str(content or "").lower()
    backend_aliases: dict[str, tuple[str, ...]] = {
        "nle": ("nle", "net_hack", "nethack"),
        "transformers": ("transformers", "from_pretrained", "automodel", "autotokenizer", "gpt2", "roberta", "t5"),
        "datasets": ("datasets", "load_dataset", "cnn_dailymail", "jigsaw", "realtoxicityprompts", "squad", "svhn"),
        "sbi": ("sbi", "npe", "nre", "neural posterior estimation", "neural ratio estimation"),
        "torch": ("torch", "nn.module", "optimizer", "cuda", "backward"),
        "gym": ("gym", "gymnasium"),
    }

    def has_explicit_alias(text: str, alias: str) -> bool:
        rendered = str(alias or "").strip().lower()
        if not rendered:
            return False
        if "." in rendered or "_" in rendered or " " in rendered:
            return rendered in text
        return re.search(rf"(?<![a-z0-9_]){re.escape(rendered)}(?![a-z0-9_])", text) is not None

    required = [
        name
        for name, aliases in backend_aliases.items()
        if any(has_explicit_alias(contract_text, alias) for alias in aliases)
    ]
    if not required:
        return {
            "name": "external_backend_route",
            "task_id": task_view.get("task_id", ""),
            "file_path": file_path,
            "passed": True,
            "skipped": True,
            "error": "",
            "details": "Skipped because no external backend/library route is named in the task contract.",
        }
    has_lazy_import = any(token in lowered for token in ("importlib", "__import__(", "find_spec", "import_optional", "optional dependency", "raise importerror"))
    has_loader_or_factory = any(
        token in lowered
        for token in (
            "factory",
            "loader",
            "load_",
            "make_",
            "build_",
            "from_pretrained",
            "load_dataset",
            "find_spec",
            "importlib",
        )
    )
    missing = []
    for backend in required:
        aliases = backend_aliases[backend]
        if not any(has_explicit_alias(lowered, alias) for alias in aliases):
            missing.append(backend)
    toy_dominated = any(token in lowered for token in ("toy", "dummy", "fake", "synthetic")) and not any(
        token in lowered for token in ("full", "from_pretrained", "load_dataset", "nle", "sbi", "gymnasium", "importlib", "find_spec")
    )
    passed = not missing and has_loader_or_factory and (has_lazy_import or any(token in lowered for token in ("from_pretrained", "load_dataset", "sbi.", "nle."))) and not toy_dominated
    details = (
        "External backend route check passed: named backend has a lazy loader/factory/full-mode route."
        if passed
        else (
            "External backend/library named in prepare/plan is not represented by a real lazy import/load factory route. "
            f"required={required}, missing={missing}, has_loader_or_factory={has_loader_or_factory}, has_lazy_import={has_lazy_import}, toy_dominated={toy_dominated}"
        )
    )
    return {
        "name": "external_backend_route",
        "task_id": task_view.get("task_id", ""),
        "file_path": file_path,
        "passed": passed,
        "skipped": False,
        "error": "" if passed else details,
        "details": details,
    }


def _metric_semantics_check(*, tree: ast.Module, content: str, task_input: dict, task_view: dict) -> dict[str, Any]:
    file_path = str(task_view.get("file_path", "") or task_input.get("file_path", "") or "").strip().replace("\\", "/")
    text = " ".join(
        str(item or "").lower()
        for source in (task_input, task_view)
        for key in ("implementation_surfaces", "method_obligations", "writes_artifacts", "review_points")
        for item in list(source.get(key, []) or [])
    )
    requires_metric = any(token in text for token in ("metric", "evaluation", "artifact", "report", "result", "fidelity", "reward"))
    if not requires_metric:
        return {
            "name": "metric_semantics",
            "task_id": task_view.get("task_id", ""),
            "file_path": file_path,
            "passed": True,
            "skipped": True,
            "error": "",
            "details": "Skipped because this file has no metric/artifact obligation.",
        }

    suspicious_returns: list[str] = []

    def _payload_has_placeholder_semantics(node: ast.AST) -> bool:
        if isinstance(node, ast.Constant):
            if node.value is None:
                return True
            if isinstance(node.value, str):
                lowered = node.value.lower()
                return any(
                    token in lowered
                    for token in (
                        "contract_placeholder_not_experimental_result",
                        "dry_run_contract",
                        "reproagent_validation_materializer",
                        "placeholder payload",
                        "synthetic result",
                        "simulated result",
                        "fake result",
                        "not a claimed experimental result",
                    )
                )
            return False
        if isinstance(node, ast.Dict):
            if not node.values:
                return True
            return any(_payload_has_placeholder_semantics(value) for value in node.values)
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            if not node.elts:
                return True
            return any(_payload_has_placeholder_semantics(value) for value in node.elts)
        return False

    metric_function_tokens = ("metric", "eval", "artifact", "report", "result", "score", "accuracy", "fidelity")
    for function in [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        function_name = str(function.name or "").lower()
        should_check_function = any(token in function_name for token in metric_function_tokens)
        for node in ast.walk(function):
            if not isinstance(node, ast.Return):
                continue
            value = node.value
            if not should_check_function:
                continue
            if isinstance(value, ast.Dict):
                for dict_value in value.values:
                    if isinstance(dict_value, ast.Constant) and dict_value.value is None:
                        suspicious_returns.append("return metric payload with None")
                        break
                continue
            if value is None or isinstance(value, ast.Constant) and value.value is None:
                suspicious_returns.append("return None")
            elif _payload_has_placeholder_semantics(value):
                suspicious_returns.append("return placeholder payload")
    passed = not suspicious_returns
    details = (
        "Metric semantics check passed: metric/artifact code exposes non-empty measurable outputs."
        if passed
        else "Metric/artifact obligation has empty, None, or placeholder result semantics: "
        + ", ".join(suspicious_returns[:8] or ["placeholder payload"])
    )
    return {
        "name": "metric_semantics",
        "task_id": task_view.get("task_id", ""),
        "file_path": file_path,
        "passed": passed,
        "skipped": False,
        "error": "" if passed else details,
        "details": details,
    }


def _run_closed_workpackage_smoke(project_files: dict[str, str], task_input: dict) -> dict[str, Any]:
    smoke_config = dict(task_input.get("work_package_smoke", {}) or {})
    surface_file = str(smoke_config.get("surface_file", "") or "").strip()
    smoke_mode = str(smoke_config.get("mode", "") or "").strip()
    smoke_command = str(smoke_config.get("command", "") or "").strip()
    timeout_seconds = int(smoke_config.get("timeout_seconds", 10) or 10)
    work_package_id = str(task_input.get("work_package_id", "") or "").strip()

    declared_artifacts = _dedupe_nonempty_text(
        [str(item or "").strip() for item in list(task_input.get("writes_artifacts", []) or [])]
        + [
            str(item or "").strip()
            for item in list(dict(task_input.get("canonical_route", {}) or {}).get("expected_outputs", []) or [])
        ]
    )

    closed, missing_files = _work_package_is_closed(project_files, task_input)
    if not work_package_id:
        return {
            "name": "workpackage_dynamic_smoke",
            "task_id": task_input.get("task_id", ""),
            "file_path": task_input.get("file_path", ""),
            "work_package_id": work_package_id,
            "passed": True,
            "skipped": True,
            "error": "",
            "details": "Skipped workpackage smoke because the task has no work package owner.",
        }
    if not closed:
        return {
            "name": "workpackage_dynamic_smoke",
            "task_id": task_input.get("task_id", ""),
            "file_path": task_input.get("file_path", ""),
            "work_package_id": work_package_id,
            "passed": True,
            "skipped": True,
            "error": "",
            "details": "Skipped workpackage smoke because the work package is not closed yet: " + ", ".join(missing_files[:8]),
        }
    if not surface_file or not smoke_mode:
        return {
            "name": "workpackage_dynamic_smoke",
            "task_id": task_input.get("task_id", ""),
            "file_path": task_input.get("file_path", ""),
            "work_package_id": work_package_id,
            "passed": True,
            "skipped": True,
            "error": "",
            "details": "Skipped workpackage smoke because no runnable smoke surface is declared for the closed work package.",
        }

    with tempfile.TemporaryDirectory(prefix="reproagent_wp_smoke_") as tmp_dir:
        root = Path(tmp_dir)
        for relative_path, content in project_files.items():
            if not str(relative_path).strip():
                continue
            path = root / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

        if smoke_mode == "command" and smoke_command:
            try:
                smoke_args = shlex.split(smoke_command)
                if not smoke_args:
                    raise ValueError("Smoke command is empty")
                completed = subprocess.run(
                    smoke_args,
                    cwd=root,
                    shell=False,
                    text=True,
                    capture_output=True,
                    timeout=timeout_seconds,
                )
                passed = completed.returncode == 0
                error = completed.stderr.strip() or completed.stdout.strip()
                missing_artifacts: list[str] = []
                if passed and declared_artifacts:
                    for artifact_path in declared_artifacts:
                        candidate = root / artifact_path
                        if not candidate.exists():
                            missing_artifacts.append(artifact_path)
                    if missing_artifacts:
                        passed = False
                        error = (
                            "smoke command returned 0 but did not materialize declared artifacts: "
                            + ", ".join(missing_artifacts[:12])
                        )
                return {
                    "name": "workpackage_dynamic_smoke",
                    "task_id": task_input.get("task_id", ""),
                    "file_path": task_input.get("file_path", ""),
                    "work_package_id": work_package_id,
                    "passed": passed,
                    "skipped": False,
                    "error": "" if passed else error[:1200],
                    "details": (
                        f"Closed workpackage smoke via command: {smoke_command}; "
                        f"checked {len(declared_artifacts)} declared artifact paths"
                    ),
                }
            except Exception as exc:
                return {
                    "name": "workpackage_dynamic_smoke",
                    "task_id": task_input.get("task_id", ""),
                    "file_path": task_input.get("file_path", ""),
                    "work_package_id": work_package_id,
                    "passed": False,
                    "skipped": False,
                    "error": str(exc),
                    "details": f"Closed workpackage smoke command raised: {smoke_command}",
                }

        if smoke_mode == "import" and surface_file.endswith(".py"):
            module_name = _smoke_module_name(surface_file)
            module_path = root / surface_file
            try:
                sys.path.insert(0, str(root))
                submodule_search_locations = None
                if module_path.name == "__init__.py":
                    submodule_search_locations = [str(module_path.parent)]
                spec = importlib.util.spec_from_file_location(
                    module_name or "_reproagent_workpackage_smoke",
                    module_path,
                    submodule_search_locations=submodule_search_locations,
                )
                if spec is None:
                    raise ImportError(f"Cannot resolve module spec for {module_name or module_path.name}")
                module = importlib.util.module_from_spec(spec)
                assert spec.loader is not None
                sys.modules[spec.name] = module
                spec.loader.exec_module(module)
                return {
                    "name": "workpackage_dynamic_smoke",
                    "task_id": task_input.get("task_id", ""),
                    "file_path": task_input.get("file_path", ""),
                    "work_package_id": work_package_id,
                    "passed": True,
                    "skipped": False,
                    "error": "",
                    "details": f"Closed workpackage import smoke passed for {module_name}",
                }
            except Exception as exc:
                return {
                    "name": "workpackage_dynamic_smoke",
                    "task_id": task_input.get("task_id", ""),
                    "file_path": task_input.get("file_path", ""),
                    "work_package_id": work_package_id,
                    "passed": False,
                    "skipped": False,
                    "error": str(exc),
                    "details": f"Closed workpackage import smoke failed for {module_name}",
                }
            finally:
                if "spec" in locals() and spec is not None:
                    sys.modules.pop(spec.name, None)
                if sys.path and sys.path[0] == str(root):
                    sys.path.pop(0)

    return {
        "name": "workpackage_dynamic_smoke",
        "task_id": task_input.get("task_id", ""),
        "file_path": task_input.get("file_path", ""),
        "work_package_id": work_package_id,
        "passed": True,
        "skipped": True,
        "error": "",
        "details": "Skipped workpackage smoke because no supported smoke mode was available.",
    }


def _work_package_code_content(project_files: dict[str, str], task_input: dict) -> str:
    """Concatenate generated package code for work-package-level semantic checks."""
    package_paths = _dedupe_nonempty_text(
        [str(item or "").strip() for item in list(task_input.get("work_package_required_files", []) or [])]
        or [str(item or "").strip() for item in list(task_input.get("work_package_files", []) or [])]
    )
    code_paths = [path for path in package_paths if path.endswith(".py") and path in project_files]
    if not code_paths:
        code_paths = [path for path in package_paths if path in project_files]
    parts: list[str] = []
    for path in code_paths:
        content = str(project_files.get(path, "") or "")
        if not content.strip():
            continue
        parts.append(f"\n# reproagent work-package file: {path}\n{content}")
    return "\n".join(parts)


def _run_task_review(project_files: dict[str, str], task_input: dict, task_view: dict) -> dict:
    """Run review for one generated task or one task inside a closed work package."""
    task_input = sanitize_task_contract(dict(task_input or {}) if isinstance(task_input, dict) else {})
    task_view = sanitize_task_contract(dict(task_view or {}) if isinstance(task_view, dict) else {})
    checks: list[dict] = []
    file_path = task_view.get("file_path", "")
    file_present = file_path in project_files
    content = project_files.get(file_path, "")
    route = str(task_view.get("route", "") or "").strip()
    work_package_review = route == "work_package_review" or bool(task_view.get("work_package_review", False))
    semantic_content = _work_package_code_content(project_files, task_input) if work_package_review else ""
    if not semantic_content:
        semantic_content = content
    file_checks = [
        {
            "name": "file_exists",
            "task_id": task_view.get("task_id", ""),
            "file_path": file_path,
            "passed": file_present,
            "error": "" if file_present else f"Missing generated file: {file_path}",
        }
    ]
    if file_present and file_path.endswith(".py"):
        try:
            tree = ast.parse(content)
            file_checks.append({
                "name": "python_syntax",
                "task_id": task_view.get("task_id", ""),
                "file_path": file_path,
                "passed": True,
                "error": "",
            })
            missing_imports = _missing_top_level_imports(project_files, tree)
            if missing_imports:
                file_checks.append({
                    "name": "python_top_level_imports",
                    "task_id": task_view.get("task_id", ""),
                    "file_path": file_path,
                    "passed": False,
                    "error": (
                        "Top-level imports require external packages that are unavailable "
                        f"in the code-only smoke environment: {', '.join(missing_imports)}. "
                        "Move optional dependencies into the methods/functions that use them, "
                        "provide lightweight fallbacks for import-time type references, and keep "
                        "module import side effects dependency-light."
                    ),
                })
            file_checks.append(
                _implementation_depth_check(
                    tree=tree,
                    content=content,
                    task_input=task_input,
                    task_view=task_view,
                )
            )
            file_checks.append(
                _placeholder_dominance_check(
                    content=content,
                    task_input=task_input,
                    task_view=task_view,
                )
            )
            file_checks.append(
                _full_implementation_route_check(
                    content=semantic_content,
                    task_input=task_input,
                    task_view=task_view,
                )
            )
            file_checks.append(
                _runtime_path_bypass_check(
                    content=semantic_content,
                    task_input=task_input,
                    task_view=task_view,
                    project_files=project_files,
                )
            )
            file_checks.append(
                _declared_experiment_contract_check(
                    content=semantic_content,
                    task_input=task_input,
                    task_view=task_view,
                    project_files=project_files,
                )
            )
            file_checks.append(
                _toy_or_stub_environment_check(
                    tree=tree,
                    content=content,
                    task_input=task_input,
                    task_view=task_view,
                )
            )
            file_checks.append(
                _external_backend_route_check(
                    content=semantic_content,
                    task_input=task_input,
                    task_view=task_view,
                )
            )
            file_checks.append(
                _metric_semantics_check(
                    tree=tree,
                    content=content,
                    task_input=task_input,
                    task_view=task_view,
                )
            )
        except SyntaxError as exc:
            file_checks.append({
                "name": "python_syntax",
                "task_id": task_view.get("task_id", ""),
                "file_path": file_path,
                "passed": False,
                "error": str(exc),
            })
    checks.extend(file_checks)
    run_workpackage_smoke = bool(task_view.get("run_workpackage_smoke", True))
    if all(item.get("passed", False) for item in file_checks) and run_workpackage_smoke:
        checks.append(_run_closed_workpackage_smoke(project_files, task_input))
    elif all(item.get("passed", False) for item in file_checks) and work_package_review:
        checks.append(
            {
                "name": "workpackage_dynamic_smoke",
                "task_id": task_view.get("task_id", ""),
                "file_path": file_path,
                "work_package_id": str(task_input.get("work_package_id", "") or ""),
                "passed": True,
                "skipped": True,
                "error": "",
                "details": "Skipped duplicate work-package smoke; package smoke is run once per work-package review.",
            }
        )
    passed = all(item.get("passed", False) for item in checks) if checks else True
    failed_checks = [item for item in checks if not item.get("passed", False)]
    suggestions = [_format_failed_check_suggestion(item) for item in failed_checks][:8]
    return {
        "task_id": task_input.get("task_id", ""),
        "review_stage": "task_review",
        "success": passed,
        "checks": checks,
        "review_points": task_input.get("review_points", []),
        "failure_summary": suggestions[:6],
        "suggestions": suggestions,
    }

def _task_touches_runtime_surface(file_path: str) -> bool:
    """Return True when a reviewed task changes files that can invalidate prior runtime validation."""
    runtime_sensitive_names = {"requirements.txt", "pyproject.toml", "setup.py"}
    if file_path in runtime_sensitive_names:
        return True
    return file_path.endswith((".py", ".sh"))

def _build_terminal_evaluation(
    *,
    engine,
    latest_execution_result: dict[str, Any],
    latest_preflight: dict[str, Any],
    latest_runtime_smoke: dict[str, Any],
    latest_docker_validate: dict[str, Any],
    latest_experiment_results: dict[str, Any],
    latest_suggestions: list[str],
) -> EvaluationDecision:
    """Build a terminal evaluation for the generate flow."""
    checks = latest_execution_result.get("checks", [])
    failed_checks = [item for item in checks if not item.get("passed", False)]

    if latest_preflight.get("status") == "failed":
        return EvaluationDecision(
            action="COMPLETE",
            reason="All generate tasks completed, but final preflight still failed.",
            suggestions=list(latest_suggestions),
        )
    if latest_runtime_smoke.get("status") == "failed":
        return EvaluationDecision(
            action="COMPLETE",
            reason="All generate tasks completed, but final runtime smoke still failed.",
            suggestions=list(latest_suggestions),
        )
    if latest_docker_validate.get("status") == "failed":
        return EvaluationDecision(
            action="COMPLETE",
            reason="All generate tasks completed, but final docker validation failed.",
            suggestions=list(latest_suggestions),
        )
    if latest_docker_validate.get("status") == "partial":
        return EvaluationDecision(
            action="COMPLETE",
            reason="All generate tasks completed, but final docker validation is incomplete.",
            suggestions=list(latest_suggestions),
        )
    if latest_docker_validate.get("status") == "success":
        if latest_experiment_results and engine._targets_met(latest_experiment_results, latest_execution_result.get("raw_metrics")):
            return EvaluationDecision(action="COMPLETE", reason="Experiment targets met.", suggestions=[])
        if latest_experiment_results:
            return EvaluationDecision(
                action="COMPLETE",
                reason="All generate tasks completed with successful runtime validation.",
                suggestions=[],
            )
        return EvaluationDecision(
            action="COMPLETE",
            reason="Runtime validation passed but structured experiment results are missing.",
            suggestions=list(latest_suggestions),
        )
    if latest_execution_result.get("success", False) and not failed_checks:
        return EvaluationDecision(
            action="COMPLETE",
            reason="All generate review tasks passed.",
            suggestions=[],
        )
    return EvaluationDecision(
        action="COMPLETE",
        reason="All generate tasks completed with unresolved review or execution issues.",
        suggestions=list(latest_suggestions),
    )
