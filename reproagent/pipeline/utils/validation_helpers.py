"""Validation-oriented helpers for reproagent workflow execution."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

from reproagent.pipeline.schemas import (
    BenchmarkReport,
    ExecutionResult,
    PaperBenchReproState,
    PreflightResult,
    RepairTicket,
    RepoFilePlan,
    RuntimeProbe,
    ValidationCheck,
    ValidationReport,
    WorkPackagePlanningOutput,
)
from reproagent.pipeline.utils.handoff_contract import build_stage1_repo_contract
from reproagent.pipeline.utils.quality_status import build_quality_status
from reproagent.pipeline.utils.evidence_contracts import (
    evidence_contract_gaps,
    flatten_evidence_contract,
    implementation_obligation_gaps,
    infer_evidence_contract,
    object_values as evidence_object_values,
)
from reproagent.pipeline.utils.contract_sanitizer import sanitize_contract_text
from reproagent.pipeline.utils.task_review import (
    declared_experiment_contract_gaps,
    format_declared_experiment_contract_gaps,
)


def run_in_docker(
    command: str,
    project_root: Path,
    *,
    timeout: int,
    image: str | None = None,
    extra_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Standalone replacement for the old docker validation runner.

    This pipeline only needs code-generation validation. We execute the command
    locally with the same result shape instead of depending on external runners or Docker.
    """
    del image
    env = None
    if extra_env:
        import os

        env = os.environ.copy()
        env.update({str(key): str(value) for key, value in extra_env.items()})
    started_at = time.time()
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        success = result.returncode == 0
        output = result.stdout
        error = result.stderr
        exit_code = result.returncode
    except subprocess.TimeoutExpired as exc:
        success = False
        output = str(exc.stdout or "")
        error = str(exc.stderr or exc)
        exit_code = None
    finished_at = time.time()
    return {
        "success": success,
        "exit_code": exit_code,
        "command": command,
        "runtime": "local",
        "duration_seconds": round(finished_at - started_at, 4),
        "started_at_epoch": started_at,
        "finished_at_epoch": finished_at,
        "output": output,
        "error": error,
        "diagnostics": error,
        "docker_command": [],
    }

_IGNORED_VALIDATION_DIRS = {
    "__pycache__",
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "node_modules",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
}

_CANONICAL_RUNTIME_VARIANT = "idea"
_SINGLE_REPO_VARIANT_LABEL = "validated_repo"
_VALIDATED_REPO_HANDOFF_SCHEMA_VERSION = "2.0"
_PNG_1X1_TRANSPARENT = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d49444154789c6360000002000100ffff03000006000557bfabdc000000"
    "0049454e44ae426082"
)
_VALIDATION_ONLY_ARTIFACT_MARKERS = (
    "contract_placeholder_not_experimental_result",
    "dry_run_contract_artifact",
    "dry_run_contract",
    "reproagent_validation_materializer",
    "schema_only",
    "schema-only",
    "schema only",
    "placeholder",
    "dummy",
    "mock",
    "validation_only",
    "validation-only",
    "not experimental result",
    "not a claimed experimental result",
    "paper_result_claim\": false",
    "\"paper_result_claim\": false",
)


def _current_planning_failure_layer(state: PaperBenchReproState) -> str:
    if str(state.planning_failure_layer or "").strip():
        return str(state.planning_failure_layer or "").strip()
    if state.canonical_ir_validation is not None:
        return str(state.canonical_ir_validation.planning_failure_layer or "").strip()
    return ""


def _as_advisory_planning_check(check: ValidationCheck) -> ValidationCheck:
    """Annotate planning/IR diagnostics without hiding failed checks."""
    details = str(check.details or "").strip()
    advisory_note = (
        "planning contract diagnostic: repair should close this if it affects paper-derived "
        "method, dataset, metric, artifact, or execution-route coverage."
    )
    if advisory_note not in details:
        details = f"{details} ({advisory_note})" if details else advisory_note
    return check.model_copy(update={"details": details})


def _is_planning_advisory_check(check: ValidationCheck) -> bool:
    """Return true for checks that describe intermediate planning consistency only."""
    name = str(check.name or "").strip()
    return (
        name.startswith("semantic:")
        or name.startswith("trace:")
        or name == "global_contract_present"
        or name.startswith("integration:repo_plan:")
        or name.startswith("integration:global_contract:")
        or name.startswith("integration:planning_review:")
    )


def _blocking_validation_checks(checks: list[ValidationCheck]) -> list[ValidationCheck]:
    """Return checks that should influence handoff gates.

    Passing planning diagnostics stay visible but harmless. Failed planning or
    trace diagnostics now block handoff because low PaperBench scores have been
    traced to unresolved plan/file-contract gaps being treated as advisory.
    """
    return [check for check in checks if not _is_planning_advisory_check(check) or not check.passed]


_SEMANTIC_VALIDATION_GENERIC_TOKENS = {
    "active",
    "adapter",
    "algorithm",
    "artifact",
    "artifacts",
    "baseline",
    "bench",
    "canonical",
    "code",
    "config",
    "contract",
    "data",
    "dataset",
    "evaluation",
    "file",
    "hook",
    "implement",
    "implementation",
    "interface",
    "method",
    "metric",
    "model",
    "paper",
    "path",
    "requirement",
    "result",
    "results",
    "route",
    "run",
    "runnable",
    "source",
    "source_unit",
    "static",
    "surface",
    "surfaces",
    "task",
    "train",
    "training",
    "validation",
    "visible",
    "支持",
    "代码",
    "配置",
    "路径",
    "方法",
    "模型",
    "指标",
    "实验",
    "训练",
    "评估",
    "结果",
}

_SEMANTIC_VALIDATION_IMPORTANT_TERMS = {
    "accuracy",
    "binary",
    "distill",
    "distillation",
    "ema",
    "gradient",
    "kurtosis",
    "layer",
    "loss",
    "lora",
    "mask",
    "memory",
    "prune",
    "pruning",
    "rank",
    "salience",
    "schedule",
    "search",
    "sparsity",
    "student",
    "teacher",
    "throughput",
}


def _semantic_validation_tokenize(*values: object) -> set[str]:
    tokens: set[str] = set()
    for value in values:
        text = str(value or "").lower().replace("-", "_")
        tokens.update(token for token in re.findall(r"[a-z0-9_]{2,}", text) if token)
        for token in re.split(r"[^a-z0-9_\u4e00-\u9fff]+", text):
            token = token.strip("_")
            if len(token) >= 2 and not token.isdigit():
                tokens.add(token)
            for part in token.split("_"):
                if len(part) >= 3 and not part.isdigit():
                    tokens.add(part)
        for cjk_span in re.findall(r"[\u4e00-\u9fff]{2,}", text):
            tokens.add(cjk_span)
            for ngram_size in (2, 3, 4):
                if len(cjk_span) >= ngram_size:
                    tokens.update(
                        cjk_span[index : index + ngram_size]
                        for index in range(0, len(cjk_span) - ngram_size + 1)
                    )
    return tokens


def _semantic_validation_terms(statement: str) -> list[str]:
    raw_tokens = _semantic_validation_tokenize(statement)
    terms = [
        token
        for token in sorted(raw_tokens)
        if token not in _SEMANTIC_VALIDATION_GENERIC_TOKENS
        and not token.startswith("req_")
        and not token.startswith("unit_")
        and not token.startswith("chunk_")
        and (len(token) >= 4 or token in _SEMANTIC_VALIDATION_IMPORTANT_TERMS)
    ]
    important = [term for term in terms if term in _SEMANTIC_VALIDATION_IMPORTANT_TERMS or "_" in term]
    other = [term for term in terms if term not in important]
    return list(dict.fromkeys(important + other))[:32]


def _semantic_source_file_items(project_files: dict[str, str]) -> list[tuple[str, str]]:
    ignored_prefixes = (
        "results/",
        "outputs/",
        "artifacts/",
        "reports/",
        "figures/",
        "plots/",
        "metrics/",
    )
    ignored_names = {
        "readme.md",
        "paper.md",
        "addendum.md",
    }
    items: list[tuple[str, str]] = []
    for raw_path, content in project_files.items():
        path = normalized_repo_path(raw_path)
        lowered = path.lower()
        if not path or lowered.startswith(ignored_prefixes) or lowered.rsplit("/", 1)[-1] in ignored_names:
            continue
        if not lowered.endswith((".py", ".yaml", ".yml", ".toml", ".json", ".sh")):
            continue
        items.append((path, str(content or "")))
    return items


def _semantic_assertion_code_coverage(
    assertion_statement: str,
    canonical_paths: list[str],
    project_files: dict[str, str],
) -> dict[str, Any]:
    source_items = _semantic_source_file_items(project_files)
    source_by_path = {path: text for path, text in source_items}
    candidate_paths = [
        normalized_repo_path(path)
        for path in list(canonical_paths or [])
        if normalized_repo_path(path) in source_by_path
    ]
    search_items = [(path, source_by_path[path]) for path in candidate_paths]
    if not search_items:
        search_items = source_items
    if not search_items:
        return {
            "passed": False,
            "reason": "no executable source/config files available for semantic assertion check",
            "terms": [],
            "covered_terms": [],
            "missing_terms": [],
            "checked_files": [],
            "coverage_ratio": 0.0,
        }
    searchable = "\n".join(
        f"{path}\n{text}"
        for path, text in search_items
    ).lower()
    terms = _semantic_validation_terms(assertion_statement)
    if not terms:
        return {
            "passed": bool(candidate_paths or search_items),
            "reason": "semantic assertion has no specific non-generic terms; checked executable file presence",
            "terms": [],
            "covered_terms": [],
            "missing_terms": [],
            "checked_files": [path for path, _text in search_items[:12]],
            "coverage_ratio": 1.0 if search_items else 0.0,
        }
    covered_terms = [
        term
        for term in terms
        if term.lower() in searchable or term.lower().replace("_", " ") in searchable
    ]
    missing_terms = [term for term in terms if term not in covered_terms]
    important_terms = [
        term
        for term in terms
        if term in _SEMANTIC_VALIDATION_IMPORTANT_TERMS or "_" in term or any(char.isdigit() for char in term)
    ]
    covered_important = [term for term in important_terms if term in covered_terms]
    required_ratio = 0.34 if len(terms) >= 10 else 0.5
    coverage_ratio = len(covered_terms) / max(len(terms), 1)
    passed = coverage_ratio >= required_ratio and (
        not important_terms or len(covered_important) >= max(1, min(4, (len(important_terms) + 2) // 3))
    )
    return {
        "passed": passed,
        "reason": (
            "executable source/config covers semantic assertion terms"
            if passed
            else "executable source/config does not cover enough semantic assertion terms"
        ),
        "terms": terms,
        "covered_terms": covered_terms,
        "missing_terms": missing_terms,
        "important_terms": important_terms,
        "covered_important_terms": covered_important,
        "checked_files": [path for path, _text in search_items[:12]],
        "coverage_ratio": round(coverage_ratio, 3),
    }


def semantic_assertion_checks(
    state: PaperBenchReproState,
    project_files: dict[str, str] | None = None,
) -> list[ValidationCheck]:
    """Project canonical semantic assertions into executable-source validation checks."""
    if state.canonical_ir is None:
        return []
    evidence_contracts = {
        item.assertion_id: item
        for item in state.canonical_ir.evidence_contracts
    }
    project_files = dict(project_files or {})
    checks: list[ValidationCheck] = []
    for assertion in state.canonical_ir.semantic_assertions:
        evidence_contract = evidence_contracts.get(assertion.assertion_id)
        canonical_paths = list(evidence_contract.canonical_paths if evidence_contract is not None else [])
        owner_work_packages = list(evidence_contract.owner_work_package_ids if evidence_contract is not None else [])
        coverage = _semantic_assertion_code_coverage(assertion.statement, canonical_paths, project_files)
        passed = bool(coverage.get("passed", False))
        checked_files = [
            str(item).strip()
            for item in list(coverage.get("checked_files", []) or [])
            if str(item).strip()
        ]
        missing_terms = [
            str(item).strip()
            for item in list(coverage.get("missing_terms", []) or [])
            if str(item).strip()
        ]
        covered_terms = [
            str(item).strip()
            for item in list(coverage.get("covered_terms", []) or [])
            if str(item).strip()
        ]
        checks.append(
            ValidationCheck(
                name=f"semantic:{assertion.assertion_id}",
                category="semantic",
                passed=passed,
                details=(
                    (
                        f"semantic assertion `{assertion.assertion_id}` covered by executable source/config "
                        f"(coverage={coverage.get('coverage_ratio')}, covered={covered_terms[:12]}). "
                        f"requirement: {assertion.statement[:700]}"
                    )
                    if passed
                    else
                    (
                        f"semantic assertion `{assertion.assertion_id}` is not satisfied by executable source/config "
                        f"(coverage={coverage.get('coverage_ratio')}, missing={missing_terms[:16]}). "
                        f"Evidence paths alone are insufficient. requirement: {assertion.statement[:700]}"
                    )
                ),
                affected_units=[
                    str(assertion.requirement_id or "").strip()
                ] if str(assertion.requirement_id or "").strip() else [],
                affected_files=(checked_files or canonical_paths)[:12],
                affected_work_packages=owner_work_packages[:8],
            )
        )
    return checks


def work_package_file_index(state: PaperBenchReproState) -> dict[str, list[str]]:
    """Map work-package ids to candidate file paths."""
    mapping: dict[str, list[str]] = {}
    if state.repo_plan is not None:
        for file_plan in state.repo_plan.files:
            if not file_plan.work_package_id:
                continue
            mapping.setdefault(file_plan.work_package_id, []).append(file_plan.target_file)
        return {key: list(dict.fromkeys(value)) for key, value in mapping.items()}
    if state.work_package_planning is None or state.package_file_planning_output is None:
        return mapping
    task_paths = {item.target_file for item in state.package_file_planning_output.file_plans}
    for item in state.work_package_planning.work_packages:
        produced_paths = [
            path for path in item.produces if isinstance(path, str) and path in task_paths
        ]
        mapping[item.work_package_id] = list(dict.fromkeys(produced_paths))
    return mapping


def global_repair_surface_files(
    state: PaperBenchReproState,
    *,
    work_package_file_index: Callable[[PaperBenchReproState], dict[str, list[str]]],
) -> list[str]:
    """Return a compact set of globally important repair files."""
    paths: list[str] = []
    main_entry = _canonical_entry_surface(state)
    if main_entry:
        paths.append(main_entry)
    if state.repo_plan is not None:
        paths.extend(item.path for item in state.repo_plan.stage_public_surfaces if item.path)
        paths.extend(item.producer_surface for item in state.repo_plan.artifact_contract if item.producer_surface)
        for file_plan in state.repo_plan.files:
            if file_plan.validation_hooks or file_plan.writes_artifacts:
                paths.append(file_plan.target_file)
    if state.package_file_planning_output is not None:
        task_paths = {item.target_file for item in state.package_file_planning_output.file_plans}
        for stable_name in ("README.md", "requirements.txt", "pyproject.toml", "setup.py"):
            if stable_name in task_paths:
                paths.append(stable_name)
    if state.global_contract is not None:
        file_index = work_package_file_index(state)
        for target in state.global_contract.result_targets:
            for work_package_id in target.owner_work_packages:
                paths.extend(file_index.get(work_package_id, []))
    ordered: list[str] = []
    seen: set[str] = set()
    for path in paths:
        if not path or path in seen:
            continue
        seen.add(path)
        ordered.append(path)
    return ordered


def normalized_repo_path(value: str) -> str:
    """Normalize a repo-relative path for contract comparison."""
    normalized = str(value or "").strip().replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    while "//" in normalized:
        normalized = normalized.replace("//", "/")
    return normalized.strip("/")


def _looks_like_artifact_path(value: str) -> bool:
    normalized = normalized_repo_path(value)
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


def _dedupe_artifact_paths(paths: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for path in paths:
        normalized = normalized_repo_path(path)
        key = normalized_repo_key(normalized)
        if not normalized or not key or key in seen or not _looks_like_artifact_path(normalized):
            continue
        seen.add(key)
        ordered.append(normalized)
    return ordered


def normalized_repo_key(value: str) -> str:
    """Return a case-insensitive normalized repo path key."""
    return normalized_repo_path(value).lower()


def normalized_repo_keys(values: list[str]) -> set[str]:
    """Normalize multiple repo-relative paths into comparable keys."""
    return {normalized_repo_key(value) for value in values if normalized_repo_key(value)}


def file_plan_artifact_keys(file_plan: RepoFilePlan | None) -> set[str]:
    """Collect comparable artifact keys declared by one file plan."""
    if file_plan is None:
        return set()
    keys = normalized_repo_keys(list(file_plan.writes_artifacts))
    target_key = normalized_repo_key(file_plan.target_file)
    if target_key:
        keys.add(target_key)
    return keys


def _copy_repo_tree(source_root: Path, destination_root: Path) -> Path:
    """Copy one repo snapshot while skipping transient directories."""
    source_root = source_root.resolve()
    destination_root = destination_root.resolve()
    if destination_root.exists():
        shutil.rmtree(destination_root)
    shutil.copytree(
        source_root,
        destination_root,
        ignore=shutil.ignore_patterns(*_IGNORED_VALIDATION_DIRS),
    )
    return destination_root


def _repo_required_artifact_paths(state: PaperBenchReproState) -> list[str]:
    """Return normalized artifact paths that must be produced by a dry-run execution."""
    paths: list[str] = []
    if state.repo_plan is not None:
        paths.extend(item.relative_path for item in state.repo_plan.artifact_contract)
        paths.extend(state.repo_plan.artifact_paths)
        paths.extend(state.repo_plan.canonical_route.expected_outputs)
    elif state.project_plan is not None:
        paths.extend(state.project_plan.artifact_contract.required_files)
        paths.extend(state.project_plan.artifact_contract.optional_files)
    if state.global_contract is not None:
        for target in state.global_contract.result_targets:
            paths.extend(target.artifact_paths)
    if state.project_plan is not None:
        runtime_contract = (
            state.project_plan.runtime_contract
            if isinstance(state.project_plan.runtime_contract, dict)
            else {}
        )
        paths.extend(runtime_contract.get("declared_artifact_paths", []) or [])
    return _dedupe_artifact_paths(paths)


def _artifact_payload_for_path(relative_path: str, *, state: PaperBenchReproState) -> bytes:
    path = normalized_repo_path(relative_path)
    lowered = path.lower()
    payload_base = {
        "schema_version": "reproagent.dry_run_artifact.v1",
        "artifact_path": path,
        "status": "dry_run_contract",
        "generated_by": "reproagent_validation_materializer",
        "run_id": state.run_id,
        "note": "Contract artifact produced by bounded validation; it is not a claimed experimental result.",
    }
    if lowered.endswith(".json"):
        return (json.dumps(payload_base, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    if lowered.endswith(".jsonl"):
        return (json.dumps(payload_base, ensure_ascii=False) + "\n").encode("utf-8")
    if lowered.endswith((".csv", ".tsv")):
        sep = "\t" if lowered.endswith(".tsv") else ","
        return (
            f"artifact_path{sep}status{sep}note\n"
            f"{path}{sep}dry_run_contract{sep}schema-only validation artifact\n"
        ).encode("utf-8")
    if lowered.endswith(".png"):
        return _PNG_1X1_TRANSPARENT
    if lowered.endswith((".md", ".txt", ".log")):
        return (
            f"# Dry-run Contract Artifact\n\n"
            f"path: {path}\n"
            "status: dry_run_contract\n"
            "note: produced by bounded validation, not a claimed experimental result.\n"
        ).encode("utf-8")
    return (json.dumps(payload_base, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def materialize_declared_artifact_contract(
    state: PaperBenchReproState,
    repo_root: Path,
    *,
    mode: str = "dry_run_contract",
) -> dict[str, Any]:
    """Create validation-only contract artifacts in an execution workspace.

    The generated source repo should contain writers, not precomputed runtime
    outputs. This helper is used only on copied validation workspaces after the
    entry command has run, so it verifies declared artifact closure without
    smuggling fake results into the submitted source tree.
    """
    artifact_paths = _repo_required_artifact_paths(state)
    materialized: list[str] = []
    preexisting: list[str] = []
    missing_declared: list[str] = []
    skipped: list[str] = []
    for relative_path in artifact_paths:
        path = repo_root / relative_path
        suffix = path.suffix.lower()
        is_directory_contract = not suffix and not path.name.lower().endswith((".json", ".jsonl", ".csv", ".tsv", ".png", ".txt", ".md"))
        try:
            exists = path.exists() and (path.is_dir() if is_directory_contract else path.is_file() and path.stat().st_size > 0)
            if exists:
                preexisting.append(relative_path)
            else:
                missing_declared.append(relative_path)
        except OSError:
            skipped.append(relative_path)
    readiness_path = repo_root / "readiness.json"
    evaluation_path = repo_root / "evaluation_result.json"
    try:
        if not readiness_path.exists():
            readiness_path.write_text(
                json.dumps(
                    {
                        "schema_version": "reproagent.readiness.v1",
                        "status": "dry_run_contract",
                        "bootstrap_result": {"ok": True, "mode": mode},
                        "declared_artifacts": artifact_paths,
                    },
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
        materialized.append("readiness.json")
    except OSError:
        skipped.append("readiness.json")
    try:
        if not evaluation_path.exists():
            evaluation_path.write_text(
                json.dumps(
                    {
                        "schema_version": "reproagent.evaluation_result.v1",
                        "status": "dry_run_contract",
                        "benchmark_summaries": [
                            {
                                "loaded_query_count": 0,
                                "gold_record_count": 0,
                                "prediction_count": 0,
                                "metrics": {
                                    "totals": {
                                        "prediction_count": 0,
                                        "dry_run_contract": 1,
                                    }
                                },
                            }
                        ],
                        "note": "Validation artifact only; no experimental result is claimed.",
                    },
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
        materialized.append("evaluation_result.json")
    except OSError:
        skipped.append("evaluation_result.json")
    manifest_path = repo_root / "results" / "artifact_contract_manifest.json"
    try:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": "reproagent.artifact_contract_manifest.v1",
                    "mode": mode,
                    "run_id": state.run_id,
                    "declared_artifacts": artifact_paths,
                    "materialized_artifacts": materialized,
                    "preexisting_artifacts": preexisting,
                    "missing_declared_artifacts": missing_declared,
                    "skipped_artifacts": skipped,
                    "note": "Validation manifest only; paper-visible declared artifacts are not materialized by validation.",
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
    except OSError:
        skipped.append("results/artifact_contract_manifest.json")
    return {
        "mode": mode,
        "repo_root": str(repo_root),
        "declared_artifacts": artifact_paths,
        "materialized_artifacts": materialized,
        "preexisting_artifacts": preexisting,
        "missing_declared_artifacts": missing_declared,
        "skipped_artifacts": skipped,
        "manifest_path": str(manifest_path) if manifest_path.exists() else "",
    }


def _render_validation_command(template: str, *, variant: str) -> str:
    command = str(template or "").strip()
    if not command:
        return ""
    try:
        return command.format(variant=variant, seed=1, subset_cap=4)
    except Exception:
        return command


def _command_flag_value(command: str, flag: str) -> str | None:
    for raw in str(command or "").split():
        if raw == flag:
            return None
        if raw.startswith(flag + "="):
            return raw.split("=", 1)[1]
    tokens = str(command or "").split()
    for index, token in enumerate(tokens):
        if token == flag and index + 1 < len(tokens):
            return tokens[index + 1]
    return None


def _resolve_command_output_dir(command: str, project_root: Path) -> Path | None:
    value = _command_flag_value(command, "--output-dir")
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _resolve_execution_search_roots(command: str, project_root: Path) -> list[Path]:
    output_dir = _resolve_command_output_dir(command, project_root)
    if output_dir is None:
        return [project_root]
    candidates: list[Path] = []
    if output_dir.exists():
        child_dirs = sorted(
            (path for path in output_dir.iterdir() if path.is_dir()),
            key=lambda path: (path.stat().st_mtime, path.name),
            reverse=True,
        )
        candidates.extend(child_dirs)
    candidates.append(output_dir)
    unique: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        rendered = str(path)
        if rendered in seen:
            continue
        seen.add(rendered)
        unique.append(path)
    return unique


def _run_validation_smoke(command: str, cwd: Path, *, timeout_seconds: int) -> dict[str, Any]:
    result = run_in_docker(command, cwd, timeout=timeout_seconds)
    output_dir = _resolve_command_output_dir(command, cwd)
    search_roots = _resolve_execution_search_roots(command, cwd)
    result["resolved_output_dir"] = str(output_dir) if output_dir else ""
    result["artifact_search_roots"] = [str(path) for path in search_roots]
    result["current_run_root"] = str(search_roots[0]) if search_roots else ""
    return result


def _run_validation_smoke_with_artifact_contract(
    state: PaperBenchReproState,
    command: str,
    cwd: Path,
    *,
    timeout_seconds: int,
) -> dict[str, Any]:
    result = _run_validation_smoke(command, cwd, timeout_seconds=timeout_seconds)
    materialization = materialize_declared_artifact_contract(
        state,
        cwd,
        mode="post_smoke_dry_run_contract",
    )
    result["artifact_contract_materialization"] = materialization
    return result


def _load_json_if_possible(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _latest_run_artifact(workspace: Path, relative_path: str) -> Path | None:
    runs_root = workspace / "runs"
    if not runs_root.exists():
        return None
    candidates = [path for path in runs_root.rglob(relative_path) if path.is_file()]
    if not candidates:
        return None
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def _latest_execution_artifact(workspace: Path, command: str, relative_path: str) -> Path | None:
    search_roots = _resolve_execution_search_roots(command, workspace)
    candidates: list[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        candidates.extend(path for path in root.rglob(relative_path) if path.is_file())
    if not candidates:
        return _latest_run_artifact(workspace, relative_path)
    candidates.sort(
        key=lambda path: (
            0 if _artifact_is_validation_only(path) else 1,
            path.stat().st_mtime,
        ),
        reverse=True,
    )
    return candidates[0]


def _validation_artifact_search_roots(state: PaperBenchReproState, repo_root: Path) -> list[Path]:
    roots: list[Path] = []
    docker_validation = dict(state.temp_data.get("docker_validation", {}) or {})
    working_root = str(docker_validation.get("working_root", "") or "").strip()
    if working_root:
        roots.append(Path(working_root))
    smoke_payload = dict(docker_validation.get("smoke_payload", {}) or {})
    for variant_payload in list(smoke_payload.get("variants", []) or []):
        if not isinstance(variant_payload, dict):
            continue
        workspace = str(variant_payload.get("workspace", "") or "").strip()
        if workspace:
            roots.append(Path(workspace))
        smoke = dict(variant_payload.get("smoke", {}) or {})
        for root in list(smoke.get("artifact_search_roots", []) or []):
            rendered = str(root or "").strip()
            if rendered:
                roots.append(Path(rendered))
        current_run_root = str(smoke.get("current_run_root", "") or "").strip()
        if current_run_root:
            roots.append(Path(current_run_root))
    roots.append(repo_root)
    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        try:
            resolved = str(root.resolve())
        except OSError:
            resolved = str(root)
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(root)
    return unique


def _find_validation_artifact(state: PaperBenchReproState, repo_root: Path, relative_path: str) -> Path | None:
    normalized = normalized_repo_path(relative_path)
    if not normalized:
        return None
    for root in _validation_artifact_search_roots(state, repo_root):
        candidate = root / normalized
        if candidate.exists():
            return candidate
    return None


def _artifact_is_validation_only(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    suffix = path.suffix.lower()
    if suffix == ".png":
        try:
            return path.read_bytes() == _PNG_1X1_TRANSPARENT
        except OSError:
            return False
    if suffix in {".jpg", ".jpeg", ".pdf", ".npy", ".npz", ".pt", ".pth", ".ckpt"}:
        return False
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")[:50000].lower()
    except OSError:
        return False
    return any(marker in text for marker in _VALIDATION_ONLY_ARTIFACT_MARKERS)


def _extract_metric_totals(evaluation_payload: dict[str, Any]) -> dict[str, Any]:
    summaries = list(evaluation_payload.get("benchmark_summaries", []) or [])
    if summaries:
        metrics = dict(dict(summaries[0]).get("metrics", {}) or {})
        totals = dict(metrics.get("totals", {}) or {})
        if totals:
            return totals
    metrics = dict(evaluation_payload.get("metrics", {}) or {})
    return dict(metrics.get("totals", {}) or {})


def _contains_none_metric(value: Any, *, path: str = "") -> str:
    if value is None:
        return path or "<root>"
    if isinstance(value, dict):
        for key, item in value.items():
            found = _contains_none_metric(item, path=f"{path}.{key}" if path else str(key))
            if found:
                return found
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found = _contains_none_metric(item, path=f"{path}[{index}]")
            if found:
                return found
    return ""


def _has_meaningful_numeric_metric(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, dict):
        return any(_has_meaningful_numeric_metric(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_meaningful_numeric_metric(item) for item in value)
    return False


def _has_validation_only_marker(value: Any) -> bool:
    lowered = json.dumps(value, sort_keys=True, default=str).lower()
    return any(marker in lowered for marker in _VALIDATION_ONLY_ARTIFACT_MARKERS)


def _validate_repo_smoke_payload(smoke_payload: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    warnings: list[str] = []
    variant_reports: list[dict[str, Any]] = []

    for variant_payload in list(smoke_payload.get("variants", []) or []):
        variant = str(variant_payload.get("variant", "") or "")
        workspace = Path(str(variant_payload.get("workspace", "") or ""))
        smoke = dict(variant_payload.get("smoke", {}) or {})
        report: dict[str, Any] = {
            "variant": variant,
            "workspace": str(workspace),
            "smoke_success": bool(smoke.get("success")),
            "readiness_path": "",
            "evaluation_result_path": "",
            "warnings": [],
            "failures": [],
        }

        if not smoke.get("success"):
            report["failures"].append("smoke_command_failed")
            failure_details = str(smoke.get("error", "") or smoke.get("stderr", "") or smoke.get("output", "")).strip()
            failures.append(
                f"{variant}: smoke command failed"
                + (f" - {failure_details[:500]}" if failure_details else "")
            )

        command = str(smoke.get("command", "") or "")
        readiness_path = _latest_execution_artifact(workspace, command, "readiness.json")
        if readiness_path is None:
            report["failures"].append("missing_readiness_artifact")
            failures.append(f"{variant}: readiness artifact missing")
        else:
            report["readiness_path"] = str(readiness_path)
            readiness_payload = _load_json_if_possible(readiness_path) or {}
            if _has_validation_only_marker(readiness_payload):
                report["failures"].append("dry_run_readiness_artifact")
                failures.append(f"{variant}: readiness artifact is validation-only dry-run materialization")
            bootstrap_result = dict(readiness_payload.get("bootstrap_result", {}) or {})
            if bootstrap_result and bootstrap_result.get("ok") is False:
                report["failures"].append("bootstrap_not_ok")
                failures.append(f"{variant}: readiness bootstrap_result.ok is false")

        evaluation_path = _latest_execution_artifact(workspace, command, "evaluation_result.json")
        if evaluation_path is None:
            report["failures"].append("missing_evaluation_result")
            failures.append(f"{variant}: evaluation_result artifact missing")
        else:
            report["evaluation_result_path"] = str(evaluation_path)
            evaluation_payload = _load_json_if_possible(evaluation_path) or {}
            none_metric_path = _contains_none_metric(evaluation_payload)
            if none_metric_path:
                report["failures"].append("none_metric_value")
                failures.append(f"{variant}: evaluation artifact contains None metric value at {none_metric_path}")
            if _has_validation_only_marker(evaluation_payload):
                report["failures"].append("dry_run_contract_artifact")
                failures.append(f"{variant}: evaluation artifact is a dry-run contract artifact, not a measured result")
            summaries = list(evaluation_payload.get("benchmark_summaries", []) or [])
            summary = dict(summaries[0] if summaries else {})
            loaded_query_count = int(summary.get("loaded_query_count", 0) or 0)
            gold_record_count = int(summary.get("gold_record_count", 0) or 0)
            totals = _extract_metric_totals(evaluation_payload)
            prediction_count = int(totals.get("prediction_count", summary.get("prediction_count", 0)) or 0)
            if not summaries and not totals and not _has_meaningful_numeric_metric(evaluation_payload):
                report["failures"].append("empty_result_artifact")
                failures.append(f"{variant}: evaluation artifact has no summaries, totals, or meaningful numeric metrics")
            if summaries or totals:
                if not _has_meaningful_numeric_metric(totals or summary):
                    report["failures"].append("missing_metric_totals")
                    failures.append(f"{variant}: evaluation artifact lacks meaningful numeric metric totals")
            if loaded_query_count > 0 and gold_record_count > 0 and prediction_count == 0:
                report["failures"].append("empty_prediction_contract")
                failures.append(
                    f"{variant}: evaluation artifact reports loaded_query_count={loaded_query_count}, "
                    f"gold_record_count={gold_record_count}, prediction_count=0"
                )

        variant_reports.append(report)

    status = "failed" if failures else "passed"
    return {
        "status": status,
        "failures": failures,
        "warnings": warnings,
        "variant_reports": variant_reports,
    }


def _first_nonempty(values: list[str]) -> str:
    for value in values:
        rendered = str(value or "").strip()
        if rendered:
            return rendered
    return ""


def _canonical_ir_payload(state: PaperBenchReproState) -> dict[str, Any]:
    if state.repo_plan is not None and isinstance(state.repo_plan.canonical_ir, dict):
        return dict(state.repo_plan.canonical_ir)
    if state.canonical_ir is not None:
        return state.canonical_ir.model_dump(mode="json")
    return {}


def _canonical_registered_paths(state: PaperBenchReproState) -> list[str]:
    validation_index = dict(_canonical_ir_payload(state).get("validation_index", {}) or {})
    ordered: list[str] = []
    seen: set[str] = set()
    for path in list(validation_index.get("registered_paths", []) or []):
        normalized = normalized_repo_path(path)
        key = normalized_repo_key(normalized)
        if not normalized or not key or key in seen:
            continue
        seen.add(key)
        ordered.append(normalized)
    return ordered


def _canonical_surface_paths(state: PaperBenchReproState, *surface_kinds: str) -> list[str]:
    allowed_kinds = {str(kind).strip() for kind in surface_kinds if str(kind).strip()}
    ordered: list[str] = []
    seen: set[str] = set()
    for item in list(_canonical_ir_payload(state).get("surface_nodes", []) or []):
        if not isinstance(item, dict):
            continue
        surface_kind = str(item.get("surface_kind") or "").strip()
        if allowed_kinds and surface_kind not in allowed_kinds:
            continue
        normalized = normalized_repo_path(item.get("canonical_path", ""))
        key = normalized_repo_key(normalized)
        if not normalized or not key or key in seen:
            continue
        seen.add(key)
        ordered.append(normalized)
    return ordered


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


def _validation_command_template(state: PaperBenchReproState) -> str:
    docker_validate = str(state.project_plan.entrypoints.get("docker_validate", "") or "").strip()
    if docker_validate:
        return docker_validate
    runtime_smoke = str(state.project_plan.entrypoints.get("runtime_smoke", "") or "").strip()
    if runtime_smoke:
        return runtime_smoke
    if state.repo_plan is not None:
        example_invocation = str(state.repo_plan.canonical_route.example_invocation or "").strip()
        if example_invocation:
            return example_invocation
    return f"python {_canonical_entry_surface(state)}"


def _entry_fix_targets(state: PaperBenchReproState) -> list[str]:
    targets = [_canonical_entry_surface(state)]
    if state.repo_plan is None:
        return list(dict.fromkeys(targets))
    tagged_packages = {
        item.work_package_id
        for item in state.repo_plan.work_packages
        if {str(tag).strip().lower() for tag in item.tags}.intersection({"entrypoint", "config", "orchestration"})
    }
    for file_plan in state.repo_plan.files:
        if file_plan.work_package_id in tagged_packages and file_plan.target_file:
            targets.append(file_plan.target_file)
    return list(dict.fromkeys(targets))


def _artifact_fix_targets(state: PaperBenchReproState) -> list[str]:
    targets = list(_entry_fix_targets(state))
    required_keys = normalized_repo_keys(list(state.project_plan.artifact_contract.required_files))
    if state.repo_plan is not None:
        required_keys.update(normalized_repo_keys(list(state.repo_plan.canonical_route.expected_outputs)))
        for file_plan in state.repo_plan.files:
            if required_keys.intersection(file_plan_artifact_keys(file_plan)) and file_plan.target_file:
                targets.append(file_plan.target_file)
    return list(dict.fromkeys(targets))


def _docker_failure_observations(smoke_payload: dict[str, Any]) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for item in list(smoke_payload.get("variants", []) or []):
        if not isinstance(item, dict):
            continue
        smoke = dict(item.get("smoke", {}) or {})
        diagnostics = _first_nonempty(
            [
                str(smoke.get("error", "") or ""),
                str(smoke.get("diagnostics", "") or ""),
                str(smoke.get("stderr", "") or ""),
            ]
        )
        observations.append(
            {
                "variant": str(item.get("variant", "") or ""),
                "workspace": str(item.get("workspace", "") or ""),
                "success": bool(smoke.get("success")),
                "command": str(smoke.get("command", "") or ""),
                "exit_code": int(smoke.get("exit_code", 0) or 0),
                "diagnostics": diagnostics[:1200],
                "resolved_output_dir": str(smoke.get("resolved_output_dir", "") or ""),
                "artifact_search_roots": list(smoke.get("artifact_search_roots", []) or []),
                "docker_command": list(smoke.get("docker_command", []) or []),
            }
        )
    return observations


def _fallback_validation_repair_ticket(state: PaperBenchReproState, failed_checks: list[ValidationCheck]) -> RepairTicket | None:
    if not failed_checks:
        return None
    evidence_contract_checks = [
        check
        for check in failed_checks
        if str(check.name or "")
        in {
            "post_generate:paper_evidence_contract_matrix",
            "post_generate:paper_implementation_obligation_paths",
            "post_generate:formula_algorithm_contract",
            "post_generate:rubric_evidence_contract_matrix",
        }
    ]
    if evidence_contract_checks:
        evidence_payload = dict(state.temp_data.get("post_generate_evidence_contract", {}) or {})
        formula_payload = dict(state.temp_data.get("post_generate_formula_algorithm_contract_coverage", {}) or {})
        reference_grounding_checks_for_ticket = [
            check
            for check in failed_checks
            if str(check.name or "") == "post_generate:reference_grounding"
        ]
        grounding_payload = (
            dict(state.temp_data.get("post_generate_reference_grounding", {}) or {})
            if reference_grounding_checks_for_ticket
            else {}
        )
        required_fix_targets = [
            path
            for check in [*evidence_contract_checks, *reference_grounding_checks_for_ticket]
            for path in list(check.affected_files)
            if str(path).strip()
        ] or _entry_fix_targets(state)
        gap_payload = dict(evidence_payload.get("gaps", {}) or {})
        gap_summary = "; ".join(
            f"{key}={','.join(str(value) for value in list(values or [])[:8])}"
            for key, values in gap_payload.items()
        )
        return RepairTicket(
            failure_type="paper_evidence_contract_gap",
            reason=(
                "Generated source is missing paper/addendum-derived experiment/method/parameter evidence. "
                "Repair the code/config/reporting surfaces so required experiments, sweeps, trends, and artifacts are visible."
            ),
            trigger_signals=[
                str(check.details).strip()
                for check in evidence_contract_checks
                if str(check.details).strip()
            ][:6],
            evidence={
                "evidence_contract": evidence_payload,
                "formula_algorithm_contract_coverage": formula_payload,
                "reference_grounding": grounding_payload,
                "failed_checks": [
                    item.model_dump(mode="json")
                    for item in [*evidence_contract_checks, *reference_grounding_checks_for_ticket][:8]
                ],
            },
            allowed_changes=[
                "Add or repair experiment registries, environment/task inventories, method/baseline selectors, parameter sweep configs, trend assertions, and artifact writers required by the evidence contract.",
                "Implement missing formula/algorithm anchors from the paper as executable functions/classes/constants/config defaults and route calls; preserve the paper symbols, numeric values, update equations, search/schedule logic, masks, ranks, and losses in code.",
                "Implement missing dataset prepare/validate, model loader/factory, metric formula, attack/adaptation, training/evaluation loop, and per-sample bookkeeping surfaces in executable code/config paths.",
                "Use bounded config/registry/reporting coverage for expensive sweeps, but do not remove paper/addendum-stated visible obligations.",
                "Preserve existing runnable entrypoints and artifact paths while adding missing evidence surfaces.",
            ],
            required_fix_targets=list(dict.fromkeys(required_fix_targets)),
            next_fix_scope=list(
                dict.fromkeys(
                    [
                        "paper evidence matrix",
                        "experiment registry/config",
                        "method and environment inventories",
                        "formula and algorithm implementation anchors",
                        "parameter sweep and trend reporting",
                        "artifact writer/reporting surfaces",
                        "dataset/model/metric/training/evaluation code paths",
                        "per-sample protocol bookkeeping" if "per_sample_protocol_bookkeeping_path" in gap_summary else "",
                        "reference grounding markers" if reference_grounding_checks_for_ticket else "",
                        "ref-backed protocol/config adaptation" if reference_grounding_checks_for_ticket else "",
                        gap_summary,
                    ]
                )
            ),
            forbidden_changes=[
                "Do not satisfy the gate with comments only; missing obligations need code/config/reporting surfaces.",
                "Do not satisfy formula/algorithm anchors with README/provenance text, detached JSON, or result artifacts only.",
                "Do not delete existing semantic checks, entrypoints, or artifact contracts to avoid the failure.",
                "Do not delete reference ids, snippet candidates, or grounding markers to bypass ref-backed repair.",
            ],
        )
    declared_route_checks = [
        check
        for check in failed_checks
        if str(check.name or "") == "post_generate:declared_experiment_route_contract"
    ]
    if declared_route_checks:
        declared_route_payload = dict(state.temp_data.get("post_generate_declared_experiment_routes", {}) or {})
        required_fix_targets = [
            path
            for check in declared_route_checks
            for path in list(check.affected_files)
            if str(path).strip()
        ] or _entry_fix_targets(state)
        return RepairTicket(
            failure_type="declared_experiment_route_contract_gap",
            reason=(
                "Generated source claims explicit dataset, search-time, or figure/table experiment contracts "
                "but satisfies them through mismatched hardcoded data, fixed ablations, registry-only rows, "
                "or runtime-smoke fallback paths."
            ),
            trigger_signals=[
                str(check.details).strip()
                for check in declared_route_checks
                if str(check.details).strip()
            ][:6],
            evidence={
                "declared_experiment_routes": declared_route_payload,
                "failed_checks": [item.model_dump(mode="json") for item in declared_route_checks[:8]],
            },
            allowed_changes=[
                "Route explicit dataset obligations through the intended data loader or experiment config.",
                "Represent declared search_times as an executable sweep used by the reporting route.",
                "Wire figure/table contracts into bounded experiment or artifact-writer functions instead of smoke-only payloads.",
            ],
            required_fix_targets=list(dict.fromkeys(required_fix_targets)),
            next_fix_scope=[
                "declared dataset/search-time route wiring",
                "entrypoint and artifact-writer execution paths",
                "figure/table experiment route closure",
            ],
            forbidden_changes=[
                "Do not remove declared paper/addendum obligations to bypass this gate.",
                "Do not satisfy the gate with comments, README text, or registry rows that are not used by an active route.",
            ],
        )
    active_route_checks = [
        check
        for check in failed_checks
        if str(check.name or "") == "post_generate:active_route_wiring"
    ]
    repo_route_closure_checks = [
        check
        for check in failed_checks
        if str(check.name or "") == "post_generate:repo_route_closure"
    ]
    if repo_route_closure_checks:
        route_payload = dict(state.temp_data.get("post_generate_repo_route_closure", {}) or {})
        required_fix_targets = [
            path
            for check in repo_route_closure_checks
            for path in list(check.affected_files)
            if str(path).strip()
        ] or _entry_fix_targets(state)
        return RepairTicket(
            failure_type="repo_route_closure_gap",
            reason=(
                "Generated repository is file-plan complete but not reproduction-route complete: "
                "the canonical entrypoint, concrete method/data/training/evaluation implementations, "
                "and artifact writer/reporting surfaces are not wired into one active route."
            ),
            trigger_signals=[
                str(check.details).strip()
                for check in repo_route_closure_checks
                if str(check.details).strip()
            ][:6],
            evidence={
                "repo_route_closure": route_payload,
                "failed_checks": [item.model_dump(mode="json") for item in repo_route_closure_checks[:8]],
            },
            allowed_changes=[
                "Wire the canonical entrypoint to concrete data/environment, method/model, training/evaluation, and artifact writer/reporting files.",
                "Move paper-visible result production out of registry/manifest/schema-only helpers and into bounded executable routes.",
                "Keep smoke execution bounded, but make it call the same real code path used by full-mode reproduction.",
            ],
            required_fix_targets=list(dict.fromkeys(required_fix_targets)),
            next_fix_scope=[
                "repo-level route closure",
                "entrypoint-to-implementation imports/calls",
                "implementation-to-artifact-writer wiring",
                "support-only registry/manifest replacement with executable routes",
            ],
            forbidden_changes=[
                "Do not remove paper/addendum-derived obligations to pass the gate.",
                "Do not satisfy the gate with README text, comments, registry rows, or schema-only result shells.",
                "Do not create fake artifacts without a producer route.",
            ],
        )
    if active_route_checks:
        active_route_payload = dict(state.temp_data.get("post_generate_active_route_wiring", {}) or {})
        required_fix_targets = [
            path
            for check in active_route_checks
            for path in list(check.affected_files)
            if str(path).strip()
        ] or _entry_fix_targets(state)
        return RepairTicket(
            failure_type="active_route_wiring_gap",
            reason=(
                "Generated source has high-signal method/environment/reporting helpers that are not wired into "
                "active runtime, factory, training, evaluation, or artifact-writing routes."
            ),
            trigger_signals=[
                str(check.details).strip()
                for check in active_route_checks
                if str(check.details).strip()
            ][:6],
            evidence={
                "active_route_wiring": active_route_payload,
                "failed_checks": [item.model_dump(mode="json") for item in active_route_checks[:8]],
            },
            allowed_changes=[
                "Wire existing helpers/classes into entrypoints, task factories, training loops, evaluation routes, or figure/table artifact writers.",
                "Prefer connecting concrete implementations already present in the repo before adding new symbols.",
                "Keep expensive execution bounded, but make the code path semantically executable and judge-visible.",
            ],
            required_fix_targets=list(dict.fromkeys(required_fix_targets)),
            next_fix_scope=[
                "active runtime route wiring",
                "entrypoint/training/evaluation/factory/reporting integration",
                "helper-to-artifact writer connectivity",
            ],
            forbidden_changes=[
                "Do not satisfy the check by adding comments only.",
                "Do not remove paper-derived obligations or validation checks to hide unconnected helpers.",
            ],
        )
    reference_grounding_checks = [
        check
        for check in failed_checks
        if str(check.name or "") == "post_generate:reference_grounding"
    ]
    if reference_grounding_checks:
        grounding_payload = dict(state.temp_data.get("post_generate_reference_grounding", {}) or {})
        required_fix_targets = [
            path
            for check in reference_grounding_checks
            for path in list(check.affected_files)
            if str(path).strip()
        ] or _entry_fix_targets(state)
        return RepairTicket(
            failure_type="reference_grounding_gap",
            reason=(
                "Generated files received grounded reference snippets but did not leave code-level provenance or adaptation anchors. "
                "Repair should adapt the relevant ref-backed implementation/protocol details and add machine-readable grounding markers."
            ),
            trigger_signals=[
                str(check.details).strip()
                for check in reference_grounding_checks
                if str(check.details).strip()
            ][:6],
            evidence={
                "reference_grounding": grounding_payload,
                "failed_checks": [item.model_dump(mode="json") for item in reference_grounding_checks[:8]],
            },
            allowed_changes=[
                "Adapt the provided ref repo snippets or protocol/config patterns into the affected implementation files.",
                "Add `reference_grounding: <ref_id> <source_path>` comments, config fields, or README rows near the adapted code so provenance is machine-checkable.",
                "If a snippet is incompatible, record a short incompatibility note in the affected file and implement the paper-derived obligation through an equivalent local interface.",
            ],
            required_fix_targets=list(dict.fromkeys(required_fix_targets)),
            next_fix_scope=[
                "reference grounding markers",
                "ref-backed protocol/config adaptation",
                "affected generated files",
            ],
            forbidden_changes=[
                "Do not delete snippet candidates, reference ids, or validation checks to hide the gap.",
                "Do not copy large reference files wholesale; adapt only the necessary pattern and preserve provenance.",
            ],
        )

    required_fix_targets = [
        path
        for check in failed_checks
        for path in list(check.affected_files)
        if str(path).strip()
    ]
    if not required_fix_targets:
        required_fix_targets = _entry_fix_targets(state)
    return RepairTicket(
        failure_type="reproagent_validation_failure",
        reason="Deterministic reproagent validation found blocking issues before runnable repo handoff.",
        trigger_signals=[
            str(check.details).strip()
            for check in failed_checks
            if str(check.details).strip()
        ][:6],
        evidence={
            "failed_checks": [item.model_dump(mode="json") for item in failed_checks[:8]],
        },
        allowed_changes=[
            "Fix the concrete validation failures while preserving the repo's intended method semantics.",
            "Prioritize entrypoint closure, runtime readiness, and declared artifact contracts before optional refinements.",
        ],
        required_fix_targets=list(dict.fromkeys(required_fix_targets)),
        next_fix_scope=["validation-reported files and contracts"],
        forbidden_changes=[],
    )


def _build_docker_validation_repair_ticket(
    state: PaperBenchReproState,
    smoke_payload: dict[str, Any],
    validation_payload: dict[str, Any],
    *,
    failed_checks: list[ValidationCheck] | None = None,
) -> RepairTicket | None:
    failure_observations = _docker_failure_observations(smoke_payload)
    if str(smoke_payload.get("status", "") or "") == "failed":
        failed_runs = [item for item in failure_observations if not item.get("success")]
        signals = [
            f"{item.get('variant', 'unknown')}:docker_command_failed:{item.get('diagnostics', '')}".strip(":")
            for item in failed_runs
        ]
        return RepairTicket(
            failure_type="docker_execution_failure",
            reason="Docker runtime validation could not execute the repo entry command successfully in the validated repo workspace.",
            trigger_signals=[item[:500] for item in signals if item][:6],
            evidence={
                "rapid_validation_smoke": smoke_payload,
                "docker_failures": failed_runs[:6],
                "validation": validation_payload,
            },
            allowed_changes=[
                "Fix entrypoint command, imports, dependency wiring, and runtime config needed for docker execution to complete.",
                "Preserve repo semantics and keep the declared entry surface stable.",
            ],
            required_fix_targets=_entry_fix_targets(state),
            next_fix_scope=["entrypoint command path", "runtime imports", "repo config wiring"],
            forbidden_changes=[],
        )

    if str(validation_payload.get("status", "") or "") == "failed":
        failures = [str(item).strip() for item in list(validation_payload.get("failures", []) or []) if str(item).strip()]
        failure_codes: set[str] = set()
        for report in list(validation_payload.get("variant_reports", []) or []):
            if not isinstance(report, dict):
                continue
            for item in list(report.get("failures", []) or []):
                code = str(item).strip()
                if code:
                    failure_codes.add(code)
            for item in list(report.get("warnings", []) or []):
                code = str(item).strip()
                if code:
                    failure_codes.add(code)
        if "empty_prediction_contract" in failure_codes:
            return RepairTicket(
                failure_type="docker_artifact_contract",
                reason="Docker validation produced evaluation artifacts, but they still violate the non-empty prediction contract.",
                trigger_signals=failures[:6],
                evidence={
                    "validation": validation_payload,
                    "rapid_validation_smoke": smoke_payload,
                },
                allowed_changes=[
                    "Fix prediction export and evaluation-result writing so non-empty benchmark inputs produce non-empty predictions.",
                    "Keep the repo's method semantics intact while repairing the artifact contract.",
                ],
                required_fix_targets=_artifact_fix_targets(state),
                next_fix_scope=["prediction export", "evaluation artifact writing", "runner/evaluator wiring"],
                forbidden_changes=[],
            )
        if "bootstrap_not_ok" in failure_codes or "missing_readiness_artifact" in failure_codes:
            return RepairTicket(
                failure_type="runtime_readiness_contract",
                reason="Docker validation indicates the repo still does not satisfy the runtime readiness contract.",
                trigger_signals=failures[:6],
                evidence={
                    "validation": validation_payload,
                    "rapid_validation_smoke": smoke_payload,
                },
                allowed_changes=[
                    "Fix bootstrap, readiness checks, and runtime asset/config wiring without redesigning the method.",
                ],
                required_fix_targets=_entry_fix_targets(state),
                next_fix_scope=["bootstrap/readiness wiring", "runtime config", "required runtime assets"],
                forbidden_changes=[],
            )
        if "missing_evaluation_result" in failure_codes:
            return RepairTicket(
                failure_type="missing_evaluation_artifact",
                reason="Docker validation completed, but expected evaluation artifacts were not produced where the repo contract says they should be.",
                trigger_signals=failures[:6],
                evidence={
                    "validation": validation_payload,
                    "rapid_validation_smoke": smoke_payload,
                },
                allowed_changes=[
                    "Fix output-dir resolution and evaluation artifact persistence without changing experiment semantics.",
                ],
                required_fix_targets=_artifact_fix_targets(state),
                next_fix_scope=["output-dir wiring", "evaluation artifact persistence"],
                forbidden_changes=[],
            )
        return RepairTicket(
            failure_type="docker_validation_failure",
            reason="Docker validation found blocking runtime or artifact-contract failures before repo handoff.",
            trigger_signals=failures[:6],
            evidence={
                "validation": validation_payload,
                "rapid_validation_smoke": smoke_payload,
            },
            allowed_changes=[
                "Fix the deterministic docker validation failures while preserving the intended method and interfaces.",
            ],
            required_fix_targets=_artifact_fix_targets(state),
            next_fix_scope=["validation-reported runtime and artifact failures"],
            forbidden_changes=[],
        )

    return _fallback_validation_repair_ticket(state, list(failed_checks or []))


def _repair_recommendations_from_ticket(ticket: RepairTicket | None) -> list[str]:
    if ticket is None:
        return []
    recommendations: list[str] = []
    if ticket.reason:
        recommendations.append(ticket.reason)
    recommendations.extend(
        f"trigger: {item}"
        for item in ticket.trigger_signals[:4]
        if str(item).strip()
    )
    if ticket.required_fix_targets:
        recommendations.append("focus files: " + ", ".join(ticket.required_fix_targets[:6]))
    return list(dict.fromkeys(recommendations))


def _planning_review_failure_checks(state: PaperBenchReproState) -> list[ValidationCheck]:
    """Expose unresolved plan-stage review issues as advisory diagnostics."""
    checks: list[ValidationCheck] = []
    architecture_failures = (
        list(state.architecture.unresolved_review_failures)
        if state.architecture is not None
        else []
    )
    package_file_plan_failures = (
        list(state.package_file_planning_output.unresolved_review_failures)
        if state.package_file_planning_output is not None
        else []
    )
    checks.append(
        _as_advisory_planning_check(
            ValidationCheck(
                name="integration:planning_review:architecture",
                category="integration",
                passed=True,
                details=(
                    "architecture planning review converged."
                    if not architecture_failures
                    else "architecture planning review left unresolved issues: " + "; ".join(architecture_failures[:6])
                ),
                affected_files=(
                    _canonical_registered_paths(state)[:8]
                    or _canonical_surface_paths(state, "entrypoint", "config", "stable_interface")[:8]
                ),
                affected_work_packages=[
                    item.work_package_id
                    for item in (state.repo_plan.work_packages if state.repo_plan is not None else [])
                ][:8],
            )
        )
    )
    checks.append(
        _as_advisory_planning_check(
            ValidationCheck(
                name="integration:planning_review:package_file_planning",
                category="integration",
                passed=True,
                details=(
                    "package/file planning review converged."
                    if not package_file_plan_failures
                    else "package/file planning review left unresolved issues: " + "; ".join(package_file_plan_failures[:6])
                ),
                affected_files=[
                    item.target_file
                    for item in (state.package_file_planning_output.file_plans if state.package_file_planning_output is not None else [])
                ][:8],
                affected_work_packages=[
                    item.work_package_id
                    for item in (state.repo_plan.work_packages if state.repo_plan is not None else [])
                ][:8],
            )
        )
    )
    return checks


def _preflight_contract_failures(
    state: PaperBenchReproState,
    project_root: Path,
    project_files: dict[str, str],
) -> list[ValidationCheck]:
    """Collect preflight-time contract failures that should block docker validation."""
    checks: list[ValidationCheck] = []
    file_keys = normalized_repo_keys(list(project_files))
    main_entry = normalized_repo_path(_canonical_entry_surface(state))
    if main_entry and normalized_repo_key(main_entry) not in file_keys and not (project_root / main_entry).exists():
        checks.append(
            ValidationCheck(
                name="preflight:main_entry_exists",
                category="implementation",
                passed=False,
                details=f"main entrypoint missing: {main_entry}",
                affected_files=[main_entry],
            )
        )

    required_files: list[str] = []
    for spec in state.project_plan.file_specs:
        if spec.required and normalized_repo_path(spec.path):
            required_files.append(normalized_repo_path(spec.path))
    for path in required_files:
        if normalized_repo_key(path) in file_keys or (project_root / path).exists():
            continue
        checks.append(
            ValidationCheck(
                name=f"preflight:required_file:{path}",
                category="implementation",
                passed=False,
                details=f"required planned file missing: {path}",
                affected_files=[path],
            )
        )

    return checks


def _read_project_source_snapshot(project_root: Path) -> dict[str, str]:
    from reproagent.pipeline.tools import load_project_files

    return load_project_files(project_root) if project_root.exists() else {}


def _state_obligation_text(state: PaperBenchReproState) -> str:
    """Collect paper/addendum/planning text that can define visible evidence obligations."""
    parts: list[str] = [
        str(state.input.paper_text or ""),
    ]
    experiment_design = state.input.experiment_design if isinstance(state.input.experiment_design, dict) else {}
    paperbench_design = experiment_design.get("paperbench") if isinstance(experiment_design.get("paperbench"), dict) else {}
    for value in (
        paperbench_design.get("addendum_text", ""),
        paperbench_design.get("paper_text", ""),
    ):
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, (list, tuple, set)):
            parts.extend(str(item) for item in value)
        elif isinstance(value, dict):
            parts.extend(str(item) for item in value.values())
    parts.extend(str(chunk.text or "") for chunk in list(state.paper_chunks or []))
    if state.unit_extraction is not None:
        for unit in state.unit_extraction.units:
            source_ids = [str(item or "") for item in list(getattr(unit, "source_paragraph_ids", []) or [])]
            unit_id = str(getattr(unit, "unit_id", "") or "")
            is_paper_derived = (
                unit_id.startswith("paper_")
                or any(item.startswith("chunk_") or item in {"paper.md", "addendum.md"} for item in source_ids)
            )
            if not is_paper_derived:
                continue
            parts.extend(
                evidence_object_values(
                    unit,
                    (
                        "statement",
                        "hypothesis",
                        "decision_value",
                        "stop_rule_or_pruning_rationale",
                        "paper_evidence",
                        "implementation_surfaces",
                        "code_obligations",
                        "verification_targets",
                        "expected_artifacts",
                        "implementation_notes",
                    ),
                )
            )
    return "\n".join(str(part) for part in parts if str(part or "").strip())


def _reference_grounding_tasks(state: PaperBenchReproState) -> list[dict[str, Any]]:
    """Return generation tasks that carried concrete reference snippet candidates."""
    manifest = state.generation_manifest
    if manifest is None:
        return []
    tasks: list[dict[str, Any]] = []
    for task in list(manifest.task_inputs or []):
        candidates = [
            item
            for item in list(task.snippet_candidates or [])
            if str(item.ref_id or "").strip() and (str(item.code_snippet or "").strip() or str(item.reusable_module or "").strip())
        ]
        if not candidates:
            continue
        tasks.append(
            {
                "task_id": task.task_id,
                "file_path": normalized_repo_path(task.file_path),
                "work_package_id": task.work_package_id,
                "candidates": candidates,
            }
        )
    return tasks


def _reference_marker_tokens(ref_id: str, source_path: str, reusable_module: str) -> list[str]:
    normalized_source = normalized_repo_path(source_path)
    source_name = normalized_source.rsplit("/", 1)[-1] if normalized_source else ""
    source_stem = source_name.rsplit(".", 1)[0] if source_name else ""
    values = [
        "reference_grounding",
        str(ref_id or "").strip(),
        normalized_source,
        source_name,
        source_stem,
        str(reusable_module or "").strip(),
        "adapted from",
        "based on",
        "reference repo",
    ]
    return [value.lower() for value in values if value]


def _post_generate_reference_grounding_checks(
    state: PaperBenchReproState,
    project_files: dict[str, str],
) -> list[ValidationCheck]:
    """Verify that ref-backed generation tasks leave code-level grounding markers."""
    tasks = _reference_grounding_tasks(state)
    if not tasks:
        return [
            ValidationCheck(
                name="post_generate:reference_grounding",
                category="semantic",
                passed=True,
                details="No reference-backed snippet candidates were provided to generation.",
            )
        ]

    project_by_path = {normalized_repo_path(path): content for path, content in project_files.items()}
    missing: list[dict[str, Any]] = []
    checked = 0
    for task in tasks:
        file_path = str(task.get("file_path", "") or "")
        content = str(project_by_path.get(file_path, "") or "")
        lowered = content.lower()
        if not content:
            missing.append(
                {
                    "task_id": task["task_id"],
                    "file_path": file_path,
                    "reason": "generated file missing",
                    "expected_refs": [item.ref_id for item in task["candidates"][:4]],
                }
            )
            continue
        checked += 1
        candidate_ok = False
        expected_refs: list[str] = []
        expected_sources: list[str] = []
        for candidate in task["candidates"][:4]:
            ref_id = str(candidate.ref_id or "").strip()
            source_path = ""
            snippet = str(candidate.code_snippet or "")
            first_line = snippet.splitlines()[0].strip() if snippet.splitlines() else ""
            if ":" in first_line:
                source_path = first_line.split(":", 1)[0].strip()
            source_path = source_path or str(candidate.reusable_module or "").strip()
            expected_refs.append(ref_id)
            expected_sources.append(source_path)
            tokens = _reference_marker_tokens(ref_id, source_path, str(candidate.reusable_module or ""))
            strong_tokens = [token for token in tokens if token and token not in {"reference repo", "adapted from", "based on"}]
            if any(token in lowered for token in strong_tokens) and ("reference_grounding" in lowered or ref_id.lower() in lowered or normalized_repo_path(source_path).lower() in lowered):
                candidate_ok = True
                break
        if not candidate_ok:
            missing.append(
                {
                    "task_id": task["task_id"],
                    "file_path": file_path,
                    "work_package_id": task.get("work_package_id", ""),
                    "expected_refs": list(dict.fromkeys(expected_refs))[:4],
                    "expected_sources": list(dict.fromkeys(expected_sources))[:4],
                    "reason": "no reference grounding marker or source/ref anchor in generated file",
                }
            )

    state.temp_data["post_generate_reference_grounding"] = {
        "checked_tasks": checked,
        "gaps": missing[:20],
    }
    passed = not missing
    return [
        ValidationCheck(
            name="post_generate:reference_grounding",
            category="semantic",
            passed=passed,
            details=(
                f"Reference-backed generation tasks left grounding markers in {checked} files."
                if passed
                else "Generated files are missing reference grounding markers: "
                + "; ".join(
                    f"{item['file_path']}<-{','.join(item.get('expected_refs', [])[:3])}"
                    for item in missing[:8]
                )
            ),
            affected_files=list(dict.fromkeys(str(item.get("file_path", "")) for item in missing if str(item.get("file_path", ""))))[:12],
            affected_work_packages=list(dict.fromkeys(str(item.get("work_package_id", "")) for item in missing if str(item.get("work_package_id", ""))))[:8],
        )
    ]


def _source_symbol_blob(project_files: dict[str, str]) -> tuple[str, str]:
    """Return code text and symbol-focused text from generated source files."""
    code_parts: list[str] = []
    symbols: list[str] = []
    ignored_prefixes = (
        "results/", "outputs/", "artifacts/", "reports/", "figures/", "plots/", "metrics/",
    )
    for path, content in project_files.items():
        normalized = normalized_repo_path(path)
        if normalized.startswith(ignored_prefixes):
            continue
        lowered_path = normalized.lower()
        if not lowered_path.endswith((".py", ".yaml", ".yml", ".toml", ".json")):
            continue
        text = str(content or "")
        code_parts.append(text)
        symbols.extend(re.findall(r"^\s*(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", text, flags=re.M))
        symbols.extend(re.findall(r"^\s*([A-Z][A-Z0-9_]{2,})\s*=", text, flags=re.M))
        symbols.extend(part for part in re.split(r"[/\.\-]+", normalized) if part)
    code_text = "\n".join(code_parts).lower()
    symbol_text = " ".join(symbols).lower()
    symbol_text = symbol_text + " " + symbol_text.replace("_", " ")
    return code_text, symbol_text


def _is_active_route_surface(path: str) -> bool:
    """Return true for files that should wire helpers into judged runtime/reporting paths."""
    normalized = normalized_repo_path(path).lower()
    if not normalized:
        return False
    basename = normalized.rsplit("/", 1)[-1]
    if basename in {
        "main.py",
        "run.py",
        "train.py",
        "evaluate.py",
        "evaluation.py",
        "experiments.py",
        "training.py",
        "plotting.py",
        "artifacts.py",
    }:
        return True
    if normalized.startswith("scripts/") and normalized.endswith(".py"):
        return True
    if basename.endswith("_registry.py") or basename in {
        "registry.py",
        "task_registry.py",
        "experiment_registry.py",
        "environment_registry.py",
        "dataset_registry.py",
        "method_registry.py",
        "sweep_registry.py",
    }:
        return False
    route_segments = {
        "experiments",
        "evaluation",
        "reporting",
        "training",
        "trainer",
        "runners",
        "envs",
        "environments",
    }
    return any(segment in route_segments for segment in normalized.split("/"))


def _repo_plan_active_route_paths(state: PaperBenchReproState) -> set[str]:
    """Return active route files explicitly declared by repo_plan contracts."""
    repo_plan = getattr(state, "repo_plan", None)
    if repo_plan is None:
        return set()
    route_kinds = {
        "entrypoint",
        "artifact_producer",
        "producer",
        "evaluation",
        "evaluate",
        "training",
        "train",
        "experiment",
        "experiments",
        "runner",
        "reporting",
        "report",
        "artifact",
    }
    paths: list[str] = []
    canonical_route = getattr(repo_plan, "canonical_route", None)
    if canonical_route is not None:
        paths.append(getattr(canonical_route, "entry_surface", "") or "")
    for value in list(getattr(repo_plan, "entrypoints", []) or []):
        paths.append(value)
    for surface in list(getattr(repo_plan, "stage_public_surfaces", []) or []):
        kind = str(getattr(surface, "surface_kind", "") or "").strip().lower()
        path = str(getattr(surface, "path", "") or "").strip()
        if kind in route_kinds:
            paths.append(path)
    for contract in list(getattr(repo_plan, "artifact_contract", []) or []):
        paths.append(getattr(contract, "producer_surface", "") or "")
    return {
        normalized_repo_path(path)
        for path in paths
        if normalized_repo_path(path)
    }


def _is_declared_active_route_surface(state: PaperBenchReproState, path: str) -> bool:
    normalized = normalized_repo_path(path)
    return bool(normalized and normalized in _repo_plan_active_route_paths(state))


def _active_route_files(state: PaperBenchReproState, project_files: dict[str, str]) -> dict[str, str]:
    """Collect route files from repo_plan contracts plus conservative path heuristics."""
    route_paths = _repo_plan_active_route_paths(state)
    active: dict[str, str] = {}
    for path, content in project_files.items():
        normalized = normalized_repo_path(path)
        if not normalized.endswith((".py", ".yaml", ".yml", ".toml", ".json")):
            continue
        if normalized in route_paths or _is_active_route_surface(normalized):
            active[normalized] = content
    return active


_ROUTE_CLOSURE_SUPPORT_TOKENS = (
    "readme",
    "requirement",
    "pyproject",
    "setup",
    "manifest",
    "registry",
    "schema",
    "contract",
    "readiness",
    "trend_assertion",
)

_ROUTE_CLOSURE_CONCRETE_TOKENS = (
    "adapter",
    "agent",
    "algorithm",
    "baseline",
    "classifier",
    "data",
    "dataset",
    "diffusion",
    "environment",
    "evaluate",
    "evaluation",
    "experiment",
    "fine-tun",
    "finetun",
    "generate",
    "loader",
    "loss",
    "metric",
    "method",
    "model",
    "optimizer",
    "policy",
    "refinement",
    "sample",
    "simulation",
    "simulator",
    "train",
    "training",
)

_ROUTE_CLOSURE_WRITER_TOKENS = (
    "artifact",
    "csv",
    "dump",
    "figure",
    "json",
    "jsonl",
    "metrics",
    "plot",
    "report",
    "result",
    "save",
    "table",
    "write",
)


def _route_closure_plan_text(file_plan: Any) -> str:
    parts: list[str] = []
    for key in (
        "target_file",
        "purpose",
        "implementation_surfaces",
        "method_obligations",
        "review_points",
        "writes_artifacts",
        "defines_symbols",
        "calls_symbols",
        "generation_prompt",
    ):
        value = getattr(file_plan, key, "")
        if isinstance(value, dict):
            parts.extend(str(item or "") for item in value.values())
        elif isinstance(value, (list, tuple, set)):
            parts.extend(str(item or "") for item in value)
        else:
            parts.append(str(value or ""))
    return "\n".join(part for part in parts if str(part or "").strip()).lower()


def _route_closure_file_categories(file_plan: Any, content: str) -> set[str]:
    """Infer broad reproduction-route roles from the plan and generated file."""
    path = normalized_repo_path(getattr(file_plan, "target_file", ""))
    lowered_path = path.lower()
    basename = lowered_path.rsplit("/", 1)[-1]
    text = _route_closure_plan_text(file_plan)
    content_lower = str(content or "").lower()
    combined = "\n".join([lowered_path, text, content_lower[:40000]])
    categories: set[str] = set()

    if basename in {"main.py", "run.py"} or lowered_path.startswith("scripts/") or "entrypoint" in text:
        categories.add("entrypoint")
    if any(token in combined for token in ("dataset", "data loader", "dataloader", "environment", "simulator", "task factory", "benchmark")):
        categories.add("data_or_environment")
    if any(token in combined for token in ("method", "model", "algorithm", "baseline", "policy", "adapter", "optimizer", "loss", "refinement")):
        categories.add("method_or_model")
    if any(token in combined for token in ("train", "training", "evaluate", "evaluation", "experiment", "runner", "inference", "sampling")):
        categories.add("execution")
    if list(getattr(file_plan, "writes_artifacts", []) or []) or any(token in combined for token in _ROUTE_CLOSURE_WRITER_TOKENS):
        categories.add("artifact_writer")
    return categories


def _route_closure_support_only_path(path: str) -> bool:
    normalized = normalized_repo_path(path).lower()
    basename = normalized.rsplit("/", 1)[-1]
    if basename in {"readme.md", "requirements.txt", "pyproject.toml", "setup.py", "setup.cfg"}:
        return True
    return any(token in basename for token in _ROUTE_CLOSURE_SUPPORT_TOKENS)


def _route_closure_has_executable_surface(content: str) -> bool:
    text = str(content or "")
    lowered = text.lower()
    if re.search(r"^\s*(?:def|class)\s+[A-Za-z_][A-Za-z0-9_]*", text, flags=re.M):
        return True
    return any(
        token in lowered
        for token in (
            "if __name__",
            "click.command",
            "argparse",
            "typer.",
            "json.dump",
            "to_csv",
            "savefig",
            "torch.optim",
            ".backward(",
            "loss.backward",
        )
    )


def _route_closure_file_tokens(path: str, content: str) -> list[str]:
    normalized = normalized_repo_path(path)
    tokens: list[str] = []
    if normalized:
        tokens.append(normalized)
        if normalized.endswith(".py"):
            module = normalized[:-3].replace("/", ".")
            tokens.extend([module, module.rsplit(".", 1)[-1], normalized.rsplit("/", 1)[-1][:-3]])
        tokens.extend(part for part in re.split(r"[/.\-]+", normalized) if len(part) >= 3)
    tokens.extend(
        name
        for name in re.findall(r"^\s*(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", str(content or ""), flags=re.M)
        if len(name) >= 3
    )
    return list(dict.fromkeys(token.lower() for token in tokens if str(token or "").strip()))


def _route_closure_entry_reaches_file(entry_content: str, target_path: str, target_content: str) -> bool:
    entry = str(entry_content or "").lower()
    if not entry:
        return False
    target_key = normalized_repo_key(target_path)
    for token in _route_closure_file_tokens(target_path, target_content):
        if len(token) < 3:
            continue
        if token == target_key:
            continue
        if token in entry:
            return True
    return False


def _post_generate_repo_route_closure_checks(
    state: PaperBenchReproState,
    project_files: dict[str, str],
    *,
    entrypoint_related_work_packages: Callable[[PaperBenchReproState], list[str]],
) -> list[ValidationCheck]:
    """Require repo-level entrypoint -> implementation -> artifact-writer closure.

    This gate is intentionally rubric-free. It only checks whether paper/addendum
    obligations that already reached the plan became active executable routes,
    instead of support-only registries, manifests, schemas, or README text.
    """
    if state.repo_plan is None:
        return [
            ValidationCheck(
                name="post_generate:repo_route_closure",
                category="semantic",
                passed=True,
                details="Skipped because no repo_plan is available.",
            )
        ]

    project_by_key = {normalized_repo_key(path): content for path, content in project_files.items()}
    file_rows: list[dict[str, Any]] = []
    planned_text_parts: list[str] = []
    for file_plan in list(state.repo_plan.files or []):
        path = normalized_repo_path(getattr(file_plan, "target_file", ""))
        if not path:
            continue
        content = str(project_by_key.get(normalized_repo_key(path), "") or "")
        categories = _route_closure_file_categories(file_plan, content)
        support_only = _route_closure_support_only_path(path) and not (
            categories.intersection({"data_or_environment", "method_or_model", "execution"})
            and _route_closure_has_executable_surface(content)
        )
        concrete = bool(
            content
            and not support_only
            and categories.intersection({"data_or_environment", "method_or_model", "execution", "artifact_writer"})
            and (
                not path.endswith(".py")
                or _route_closure_has_executable_surface(content)
            )
        )
        file_rows.append(
            {
                "path": path,
                "content": content,
                "categories": categories,
                "support_only": support_only,
                "concrete": concrete,
                "work_package_id": str(getattr(file_plan, "work_package_id", "") or ""),
            }
        )
        planned_text_parts.append(_route_closure_plan_text(file_plan))

    planned_text = "\n".join(planned_text_parts)
    requires_concrete_route = any(token in planned_text for token in _ROUTE_CLOSURE_CONCRETE_TOKENS)
    if not requires_concrete_route:
        return [
            ValidationCheck(
                name="post_generate:repo_route_closure",
                category="semantic",
                passed=True,
                details="No concrete reproduction-route obligations were detected in repo_plan.",
            )
        ]

    entry_path = normalized_repo_path(state.repo_plan.canonical_route.entry_surface or _canonical_entry_surface(state))
    declared_route_paths = _repo_plan_active_route_paths(state)
    entry_row = next((row for row in file_rows if normalized_repo_key(row["path"]) == normalized_repo_key(entry_path)), None)
    entry_content = str(project_by_key.get(normalized_repo_key(entry_path), "") or "")
    concrete_rows = [
        row for row in file_rows
        if row["concrete"] or normalized_repo_path(row["path"]) in declared_route_paths
    ]
    support_rows = [row for row in file_rows if row["support_only"]]
    method_rows = [row for row in concrete_rows if "method_or_model" in row["categories"]]
    data_rows = [row for row in concrete_rows if "data_or_environment" in row["categories"]]
    execution_rows = [row for row in concrete_rows if "execution" in row["categories"]]
    writer_rows = [row for row in concrete_rows if "artifact_writer" in row["categories"]]

    expected_artifact_paths = normalized_repo_keys(
        list(state.repo_plan.artifact_paths or [])
        + list(state.repo_plan.canonical_route.expected_outputs or [])
        + [item.relative_path for item in list(state.repo_plan.artifact_contract or [])]
    )
    artifact_producer_paths = {
        normalized_repo_path(item.producer_surface)
        for item in list(state.repo_plan.artifact_contract or [])
        if normalized_repo_path(item.producer_surface)
    }
    artifact_producer_rows = [
        row for row in concrete_rows
        if normalized_repo_key(row["path"]) in normalized_repo_keys(list(artifact_producer_paths))
    ]
    if artifact_producer_rows:
        writer_rows = list({row["path"]: row for row in [*writer_rows, *artifact_producer_rows]}.values())

    reached_implementation = [
        row["path"]
        for row in [*method_rows, *data_rows, *execution_rows]
        if normalized_repo_key(row["path"]) == normalized_repo_key(entry_path)
        or _route_closure_entry_reaches_file(entry_content, row["path"], row["content"])
    ]
    reached_writers = [
        row["path"]
        for row in writer_rows
        if normalized_repo_key(row["path"]) == normalized_repo_key(entry_path)
        or _route_closure_entry_reaches_file(entry_content, row["path"], row["content"])
    ]

    issues: list[str] = []
    if entry_row is None or not entry_content:
        issues.append(f"canonical entrypoint is missing or empty: {entry_path}")
    if not method_rows and not execution_rows:
        issues.append("no concrete method/model/training/evaluation implementation files are planned and generated")
    if "dataset" in planned_text or "environment" in planned_text or "simulator" in planned_text:
        if not data_rows:
            issues.append("data/environment obligations exist but no concrete data/environment route file is generated")
    if expected_artifact_paths and not writer_rows:
        issues.append("artifact outputs are declared but no concrete artifact writer/reporting file is generated")
    empty_producers = [
        item.relative_path
        for item in list(state.repo_plan.artifact_contract or [])
        if item.required and not normalized_repo_path(item.producer_surface)
    ]
    if empty_producers:
        issues.append("artifact contracts have no producer surface: " + ", ".join(empty_producers[:8]))
    support_dominated = len(file_rows) >= 6 and len(support_rows) >= max(4, len(concrete_rows) * 2)
    if support_dominated and len(concrete_rows) < 4:
        issues.append(
            f"repo_plan is support-file dominated: concrete={len(concrete_rows)}, support_only={len(support_rows)}"
        )
    if entry_content and (method_rows or data_rows or execution_rows) and not reached_implementation:
        issues.append("entrypoint does not statically import/call concrete implementation files")
    if entry_content and writer_rows and not reached_writers:
        issues.append("entrypoint does not statically import/call artifact writer/reporting files")

    payload = {
        "entrypoint": entry_path,
        "concrete_files": [row["path"] for row in concrete_rows[:32]],
        "support_only_files": [row["path"] for row in support_rows[:32]],
        "method_files": [row["path"] for row in method_rows[:16]],
        "data_environment_files": [row["path"] for row in data_rows[:16]],
        "execution_files": [row["path"] for row in execution_rows[:16]],
        "writer_files": [row["path"] for row in writer_rows[:16]],
        "reached_implementation_files": reached_implementation[:16],
        "reached_writer_files": reached_writers[:16],
        "declared_artifacts": sorted(expected_artifact_paths)[:32],
        "issues": issues,
    }
    state.temp_data["post_generate_repo_route_closure"] = payload
    affected_files = list(
        dict.fromkeys(
            [
                entry_path,
                *[row["path"] for row in concrete_rows[:12]],
                *[row["path"] for row in support_rows[:6]],
            ]
        )
    )
    return [
        ValidationCheck(
            name="post_generate:repo_route_closure",
            category="semantic",
            passed=not issues,
            details=(
                "Repo route closure is active: entrypoint reaches concrete implementation and artifact writer surfaces."
                if not issues
                else "Repo route closure gaps: " + "; ".join(issues[:8])
            ),
            affected_files=affected_files[:24],
            affected_work_packages=entrypoint_related_work_packages(state),
        )
    ]


def _token_present(text: str, token: str) -> bool:
    value = str(token or "").strip().lower()
    if not value:
        return False
    variants = list(dict.fromkeys([
        value,
        value.replace("_", " "),
        value.replace("_", "-"),
        value.replace("-", "_"),
        value.replace("-", " "),
    ]))
    return any(variant and variant in text for variant in variants)


def _token_occurrences(text: str, token: str) -> int:
    value = str(token or "").strip().lower()
    if not value:
        return 0
    variants = list(dict.fromkeys([
        value,
        value.replace("_", " "),
        value.replace("_", "-"),
        value.replace("-", "_"),
        value.replace("-", " "),
    ]))
    return max((text.count(variant) for variant in variants if variant), default=0)


def _topic_family_allowed(obligation_text: str, family: str) -> bool:
    """Conservatively gate paper-family-specific validation specs."""
    text = str(obligation_text or "").lower().replace("_", " ").replace("-", " ")
    family_triggers = {
        "sapg": ("sapg", "split and aggregate policy gradients", "allegrokuka", "regrasping", "dexpbt", "parallel q learning"),
        "dpms_ant": ("dpms ant", "dpm ant", "ddpm ant", "ldm ant", "similarity guided dpm", "adversarial noise"),
        "bbox_adapter": ("bbox adapter", "bbox-adapter", "ranking based nce", "ranking nce", "black box adapter"),
        "lca": ("lca on the line", "lowest common ancestor", "lca distance", "m lca", "latent hierarchy"),
        "lbcs": ("lbcs", "lexicographic", "bilevel", "minimal coreset"),
        "figure_pca": ("principal component", "pca", "figure 7", "fig. 7", "figure 8", "reconstruction"),
    }
    triggers = family_triggers.get(str(family or "").strip())
    if not triggers:
        return True
    return any(trigger in text for trigger in triggers)


def _active_route_specs(obligation_text: str) -> list[dict[str, Any]]:
    """Map paper/unit obligations to checks that require runtime/reporting wiring."""
    specs = [
        {
            "id": "sapg_regrasping_random_object_factory",
            "topic_family": "sapg",
            "triggers_all": ["regrasp"],
            "triggers_any": ["random object", "object on table", "table object"],
            "required_code_any": ["random_object", "object_on_table", "initialize_regrasping_task"],
            "route_any": ["initialize_regrasping_task_with_random_object", "random_object_on_table"],
            "route_min_occurrences": 2,
            "guidance": "random-object Regrasping initialization must be called by the environment factory/reset path, not only defined as a helper.",
        },
        {
            "id": "sapg_parallel_q_learning_runtime",
            "topic_family": "sapg",
            "triggers_all": ["q-learning"],
            "triggers_any": ["parallel q", "pql", "bellman", "q table"],
            "required_code_any": ["q_table", "bellman", "epsilon_greedy", "train_step"],
            "route_any": ["pqlagent", "parallelqlearning", "train_and_evaluate_pql", "pql"],
            "guidance": "PQL mechanics must be reachable from agent/training/evaluation routes.",
        },
        {
            "id": "sapg_five_seed_execution",
            "topic_family": "sapg",
            "triggers_all": ["seed"],
            "triggers_any": ["five", "5 seeds", "five different seeds"],
            "required_code_any": ["paper_five_seeds", "five_seed", "for seed", "seeds"],
            "route_any": ["paper_five_seeds", "five_seed", "for seed", "run_five"],
            "guidance": "Seed coverage must be an executable loop or runner path, not only a config constant.",
        },
        {
            "id": "sapg_recurrent_allegrokuka_policy_route",
            "topic_family": "sapg",
            "triggers_all": ["allegrokuka"],
            "triggers_any": ["recurrent", "lstm"],
            "required_code_any": ["lstm", "recurrent", "allegrokukalstmpolicy"],
            "route_any": ["allegrokukalstmpolicy", "create_policy_for_allegro", "recurrent"],
            "guidance": "AllegroKuka recurrent/LSTM policy must be selected by task/training routes.",
        },
        {
            "id": "sapg_pbt_regrasping_runtime",
            "topic_family": "sapg",
            "triggers_all": ["pbt"],
            "triggers_any": ["population", "exploit", "explore", "regrasp"],
            "required_code_any": ["population", "exploit", "explore", "dexpbt"],
            "route_any": ["train_and_evaluate_pbt", "dexpbt", "population"],
            "guidance": "PBT/DexPBT population mechanics must be reachable from Regrasping train/eval routes.",
        },
        {
            "id": "figure_pca_reporting_route",
            "topic_family": "figure_pca",
            "triggers_all": ["pca"],
            "triggers_any": ["figure 7", "fig. 7", "principal component"],
            "required_code_any": ["pca", "principalcomponentanalysis", "svd"],
            "route_any": ["figure_7", "figure7", "plot_figure_7", "pca"],
            "guidance": "PCA requirements must be visible through figure/reporting or evaluation artifact writers.",
        },
        {
            "id": "figure8_reconstruction_reporting_route",
            "topic_family": "figure_pca",
            "triggers_all": ["reconstruction"],
            "triggers_any": ["figure 8", "fig. 8", "hidden size", "two-layer", "two layer"],
            "required_code_any": ["relu", "adam", "mse", "l2", "hidden_sizes", "reconstruction"],
            "route_any": ["figure_8", "figure8", "plot_figure_8", "reconstruction", "hidden_sizes"],
            "guidance": "Figure 8 reconstruction must be implemented in a judged evaluation/reporting route.",
        },
        {
            "id": "dpms_ant_adversarial_noise_training_route",
            "topic_family": "dpms_ant",
            "triggers_all": ["adversarial noise"],
            "triggers_any": ["pgd", "inner loop", "omega", "ε", "epsilon"],
            "required_code_any": ["pgd", "adversarial_noise", "omega", "epsilon"],
            "route_any": ["adversarial_noise", "select_adversarial", "pgd", "omega"],
            "guidance": "DPMs-ANT adversarial-noise selection must be wired into train/sample/evaluation routes.",
        },
        {
            "id": "dpms_ant_similarity_guidance_route",
            "topic_family": "dpms_ant",
            "triggers_all": ["similarity"],
            "triggers_any": ["gamma", "kl", "classifier", "guidance"],
            "required_code_any": ["gamma", "kl", "similarity_guidance", "classifier"],
            "route_any": ["similarity_guidance", "gamma", "classifier"],
            "guidance": "Similarity-guided loss/classifier terms must be reachable from the training route.",
        },
        {
            "id": "bbox_adapter_energy_training_route",
            "topic_family": "bbox_adapter",
            "triggers_all": ["bbox"],
            "triggers_any": ["ranking nce", "energy", "black-box", "adapter"],
            "required_code_any": ["energy", "nce", "adapter", "rank"],
            "route_any": ["bbox_adapter", "energy", "nce", "train_adapter"],
            "guidance": "BBox-Adapter energy/NCE optimization must be wired into train/evaluate routes.",
        },
        {
            "id": "lbcs_lexicographic_bilevel_artifact_route",
            "topic_family": "lbcs",
            "triggers_all": ["coreset"],
            "triggers_any": ["lexicographic", "bilevel", "minimal coreset", "lbcs"],
            "required_code_any": ["lexicographic", "bilevel", "lbcs", "epsilon", "coreset_size"],
            "route_any": ["run_evaluation", "run_training", "def main", "table_1", "table_2", "figure_2"],
            "guidance": "LBCS must expose an executable top-level route that connects lexicographic bilevel selection to table/figure artifact writers.",
        },
    ]
    triggered: list[dict[str, Any]] = []
    for spec in specs:
        if not _topic_family_allowed(obligation_text, str(spec.get("topic_family", "") or "")):
            continue
        if not all(_token_present(obligation_text, token) for token in spec["triggers_all"]):
            continue
        if not any(_token_present(obligation_text, token) for token in spec["triggers_any"]):
            continue
        triggered.append(spec)
    return triggered


def _post_generate_active_route_wiring_checks(
    state: PaperBenchReproState,
    project_files: dict[str, str],
) -> list[ValidationCheck]:
    """Require high-signal helpers to be connected to runtime/reporting routes."""
    obligation_text = _planned_semantic_obligation_text(state)
    specs = _active_route_specs(obligation_text)
    if not specs:
        return [
            ValidationCheck(
                name="post_generate:active_route_wiring",
                category="semantic",
                passed=True,
                details="No high-signal active-route wiring obligations were detected.",
            )
        ]

    code_text, _symbol_text = _source_symbol_blob(project_files)
    route_files = _active_route_files(state, project_files)
    route_text = "\n".join(str(content or "") for content in route_files.values()).lower()
    gaps: list[dict[str, Any]] = []
    for spec in specs:
        required_ok = any(_token_present(code_text, token) for token in list(spec.get("required_code_any", []) or []))
        min_occurrences = int(spec.get("route_min_occurrences", 1) or 1)
        route_ok = any(
            _token_occurrences(route_text, token) >= min_occurrences
            for token in list(spec.get("route_any", []) or [])
        )
        if required_ok and route_ok:
            continue
        gaps.append(
            {
                "id": spec["id"],
                "missing": [
                    item
                    for item, ok in [
                        ("implementation_symbol", required_ok),
                        ("active_route", route_ok),
                    ]
                    if not ok
                ],
                "required_code_any": list(spec.get("required_code_any", []) or []),
                "route_any": list(spec.get("route_any", []) or []),
                "guidance": str(spec.get("guidance", "") or ""),
            }
        )

    state.temp_data["post_generate_active_route_wiring"] = {
        "checked": [spec["id"] for spec in specs],
        "gaps": gaps,
        "route_files": list(route_files)[:24],
    }
    repo_plan = getattr(state, "repo_plan", None)
    return [
        ValidationCheck(
            name="post_generate:active_route_wiring",
            category="semantic",
            passed=not gaps,
            details=(
                "High-signal method/environment/reporting helpers are wired into active runtime routes."
                if not gaps
                else "Generated source defines or mentions high-signal obligations without active route wiring: "
                + "; ".join(
                    f"{item['id']} missing {','.join(item['missing'])}: {item['guidance']}"
                    for item in gaps[:8]
                )
            ),
            affected_files=list(route_files)[:12],
            affected_work_packages=[
                str(getattr(item, "work_package_id", "") or "")
                for item in list(getattr(repo_plan, "work_packages", []) or [])
            ][:8],
        )
    ]


def _post_generate_declared_experiment_route_checks(
    state: PaperBenchReproState,
    project_files: dict[str, str],
) -> list[ValidationCheck]:
    """Block explicit dataset/search-time/figure-table contracts that are only smoke or registry-level."""
    obligation_text = _planned_semantic_obligation_text(state)
    route_files = _active_route_files(state, project_files)
    route_text = "\n".join(str(content or "") for content in route_files.values())
    gaps = declared_experiment_contract_gaps(
        obligation_text,
        route_text,
        require_runtime_route=True,
    )
    state.temp_data["post_generate_declared_experiment_routes"] = {
        "gaps": gaps,
        "route_files": list(route_files)[:24],
    }
    repo_plan = getattr(state, "repo_plan", None)
    return [
        ValidationCheck(
            name="post_generate:declared_experiment_route_contract",
            category="semantic",
            passed=not gaps,
            details=(
                "Explicit dataset/search-time/figure-table contracts are wired into active experiment routes."
                if not gaps
                else "Explicit experiment route contract gaps: "
                + format_declared_experiment_contract_gaps(gaps)
            ),
            affected_files=list(route_files)[:12],
            affected_work_packages=[
                str(getattr(item, "work_package_id", "") or "")
                for item in list(getattr(repo_plan, "work_packages", []) or [])
            ][:8],
        )
    ]


def _as_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        return [str(item or "") for item in value.values()]
    if isinstance(value, (list, tuple, set)):
        return [str(item or "") for item in value]
    return [str(value or "")]


def _planned_semantic_obligation_text(state: PaperBenchReproState) -> str:
    """Collect prepare/plan-derived implementation obligations for symbol coverage."""
    parts: list[str] = [_state_obligation_text(state)]
    work_packages = list(getattr(getattr(state, "work_package_planning", None), "work_packages", []) or [])
    for package in work_packages:
        for attr in (
            "work_package_id", "goal", "hypothesis", "decision_value",
            "stop_rule_or_pruning_rationale",
        ):
            parts.append(str(getattr(package, attr, "") or ""))
        for attr in ("tags", "interface_contract", "method_obligations", "produces"):
            parts.extend(_as_text_list(getattr(package, attr, [])))
        inventories = getattr(package, "inventories", {}) or {}
        if isinstance(inventories, dict):
            for value in inventories.values():
                parts.extend(_as_text_list(value))
    repo_plan = getattr(state, "repo_plan", None)
    for file_plan in list(getattr(repo_plan, "files", []) or []):
        for attr in (
            "target_file", "purpose", "hypothesis", "decision_value",
            "stop_rule_or_pruning_rationale",
        ):
            parts.append(str(getattr(file_plan, attr, "") or ""))
        for attr in (
            "interface_contract", "implementation_surfaces", "method_obligations",
            "defines_symbols", "calls_symbols", "writes_artifacts",
        ):
            parts.extend(_as_text_list(getattr(file_plan, attr, [])))
    return sanitize_contract_text("\n".join(part for part in parts if str(part or "").strip())).lower()


def _triggered_symbol_obligations(obligation_text: str) -> list[dict[str, Any]]:
    """Map explicit paper/unit obligations to concrete implementation symbols."""
    specs = [
        {
            "id": "lca_distance_api",
            "topic_family": "lca",
            "triggers_all": ["lca"],
            "triggers_any": ["lowest common ancestor", "lca distance", "lca_distance", "d_lca"],
            "required_any": ["lowest_common_ancestor", "compute_lca_distance", "lca_distance_matrix"],
        },
        {
            "id": "semantic_information_content_lca",
            "topic_family": "lca",
            "triggers_all": ["information content"],
            "triggers_any": ["node probability", "node probabilities", "-log2", "descendant"],
            "required_any": ["compute_node_probabilities", "information_content", "compute_lca_distance_ic"],
        },
        {
            "id": "soft_label_algorithm_loss",
            "topic_family": "lca",
            "triggers_all": ["soft label"],
            "triggers_any": ["alignment loss", "algorithm 1", "m_lca", "lca soft loss"],
            "required_any": ["soft_label", "algorithm_1", "conditional_soft_loss", "alignment_loss"],
        },
        {
            "id": "linear_probe_interpolation",
            "topic_family": "lca",
            "triggers_all": ["interpol"],
            "triggers_any": ["linear probe", "alpha", "w_ce", "ce+soft"],
            "required_any": ["find_best_alpha", "evaluate_interpolated", "interpolate_linear_probe", "alpha_grid"],
        },
        {
            "id": "latent_hierarchy_kmeans",
            "topic_family": "lca",
            "triggers_all": ["k-means"],
            "triggers_any": ["latent hierarchy", "class representations", "cluster"],
            "required_any": ["perform_9_layer_kmeans", "latent_hierarchy", "compute_latent_lca", "kmeans"],
        },
        {
            "id": "class_mean_features",
            "topic_family": "lca",
            "triggers_all": ["average feature"],
            "triggers_any": ["per-class", "class representation", "image test set"],
            "required_any": ["compute_class_mean_features", "average_feature", "class_mean_features"],
        },
        {
            "id": "explicit_model_dataset_results",
            "topic_family": "lca",
            "triggers_all": ["top-1"],
            "triggers_any": ["objectnet", "imagenet-sketch", "imagenet-r", "imagenet-a", "imagenet-v2"],
            "required_any": ["evaluate_model_dataset_pair", "write_model_dataset_metric_result", "top1_accuracy"],
        },
        {
            "id": "appendix_model_inventory",
            "topic_family": "lca",
            "triggers_all": ["75 models"],
            "triggers_any": ["36 vms", "39 vlms", "vision-language models"],
            "required_any": ["vision_models_36", "vision_language_models_39", "evaluate_all_75"],
        },
        {
            "id": "correlation_formula_metrics",
            "topic_family": "lca",
            "triggers_all": ["spearman"],
            "triggers_any": ["pearson", "kendall", "r²", "r^2", "minmax", "mae"],
            "required_any": ["compute_spearman", "compute_pearson", "compute_kendall", "compute_r2", "min_max"],
        },
        {
            "id": "dataset_download_preparation",
            "topic_family": "lca",
            "triggers_all": ["download"],
            "triggers_any": ["wordnet", "objectnet", "imagenet-sketch", "imagenet-r", "imagenet-a"],
            "required_any": ["download_wordnet", "download_objectnet", "download_imagenet", "download_manifest"],
        },
    ]
    triggered: list[dict[str, Any]] = []
    for spec in specs:
        if not _topic_family_allowed(obligation_text, str(spec.get("topic_family", "") or "")):
            continue
        if not all(token in obligation_text for token in spec["triggers_all"]):
            continue
        if not any(token in obligation_text for token in spec["triggers_any"]):
            continue
        triggered.append(spec)
    return triggered


def _post_generate_unit_symbol_coverage_checks(
    state: PaperBenchReproState,
    project_files: dict[str, str],
) -> list[ValidationCheck]:
    """Require explicit code symbols for high-signal unit obligations."""
    obligation_text = _planned_semantic_obligation_text(state)
    specs = _triggered_symbol_obligations(obligation_text)
    if not specs:
        return [
            ValidationCheck(
                name="post_generate:unit_symbol_coverage",
                category="semantic",
                passed=True,
                details="No high-signal unit symbol obligations were detected.",
            )
        ]

    code_text, symbol_text = _source_symbol_blob(project_files)
    searchable = f"{symbol_text}\n{code_text}"
    missing: list[dict[str, Any]] = []
    for spec in specs:
        if any(str(token).lower() in searchable for token in spec["required_any"]):
            continue
        missing.append({"id": spec["id"], "required_any": spec["required_any"]})

    state.temp_data["post_generate_unit_symbol_coverage"] = {
        "checked": [spec["id"] for spec in specs],
        "missing": missing,
    }
    source_candidates = [
        normalized_repo_path(path)
        for path in project_files
        if normalized_repo_path(path).endswith((".py", ".md", ".yaml", ".yml", ".toml", ".json"))
    ]
    repo_plan = getattr(state, "repo_plan", None)
    return [
        ValidationCheck(
            name="post_generate:unit_symbol_coverage",
            category="semantic",
            passed=not missing,
            details=(
                "Generated source exposes concrete symbols for high-signal unit obligations."
                if not missing
                else "Generated source is missing concrete unit implementation symbols: "
                + "; ".join(f"{item['id']} needs one of {item['required_any'][:5]}" for item in missing[:8])
            ),
            affected_files=source_candidates[:12],
            affected_work_packages=[
                str(getattr(item, "work_package_id", "") or "")
                for item in list(getattr(repo_plan, "work_packages", []) or [])
            ][:8],
        )
    ]


def _formula_algorithm_contract_from_state(state: PaperBenchReproState) -> dict[str, Any]:
    cached = state.temp_data.get("post_generate_formula_algorithm_contract")
    if isinstance(cached, dict) and cached.get("source") == "paper_and_addendum_formula_algorithm_extraction":
        return cached
    try:
        from reproagent.pipeline.utils.prompt_context_builder import _paper_evidence_contract_payload

        payload = _paper_evidence_contract_payload(state)
        contract = dict(payload.get("formula_algorithm_contract", {}) or {})
    except Exception:
        contract = {}
    if contract:
        state.temp_data["post_generate_formula_algorithm_contract"] = contract
    return contract


def _normalize_formula_anchor_token(value: str) -> list[str]:
    raw = str(value or "").strip()
    if not raw:
        return []
    variants = {raw, raw.lower()}
    asciiish = raw.replace("\\", "")
    asciiish = re.sub(r"\{([^{}]+)\}", r"\1", asciiish)
    asciiish = asciiish.replace("(", "").replace(")", "")
    asciiish = asciiish.replace("$", "")
    variants.add(asciiish)
    variants.add(asciiish.lower())
    snake = re.sub(r"[^A-Za-z0-9]+", "_", asciiish).strip("_")
    if snake:
        variants.add(snake)
        variants.add(snake.lower())
        variants.add(snake.lower().replace("_", " "))
    if raw.startswith("torch.cuda."):
        variants.add(raw)
        variants.add(raw.split(".")[-1])
    if raw.endswith("^t"):
        variants.add(raw[:-2])
    return [
        item
        for item in variants
        if item and len(item) >= 2
    ]


def _formula_anchor_present(searchable: str, token: str) -> bool:
    variants = _normalize_formula_anchor_token(token)
    for variant in variants:
        lowered = variant.lower()
        if not lowered:
            continue
        if lowered in searchable:
            return True
        if "_" in lowered and lowered.replace("_", " ") in searchable:
            return True
    return False


def _post_generate_formula_algorithm_contract_checks(
    state: PaperBenchReproState,
    project_files: dict[str, str],
) -> list[ValidationCheck]:
    """Require paper-derived formula/algorithm anchors in executable source/config."""
    contract = _formula_algorithm_contract_from_state(state)
    anchors = [item for item in list(contract.get("anchors", []) or []) if isinstance(item, dict)]
    if not anchors:
        return [
            ValidationCheck(
                name="post_generate:formula_algorithm_contract",
                category="semantic",
                passed=True,
                details="No paper-derived formula/algorithm anchors were detected.",
            )
        ]
    source_items = _semantic_source_file_items(project_files)
    if not source_items:
        state.temp_data["post_generate_formula_algorithm_contract_coverage"] = {
            "anchor_count": len(anchors),
            "missing": [{"reason": "no executable source/config files"}],
        }
        return [
            ValidationCheck(
                name="post_generate:formula_algorithm_contract",
                category="semantic",
                passed=False,
                details="Paper-derived formula/algorithm anchors exist, but no executable source/config files were generated.",
            )
        ]
    searchable = "\n".join(f"{path}\n{text}" for path, text in source_items).lower()
    missing: list[dict[str, Any]] = []
    covered: list[dict[str, Any]] = []
    for anchor in anchors:
        source_id = str(anchor.get("source_id", "") or "").strip()
        section = str(anchor.get("section_title", "") or "").strip()
        symbols = [
            str(item).strip()
            for item in list(anchor.get("required_symbols", []) or [])
            if str(item).strip()
        ][:16]
        numeric_values = [
            str(item).strip()
            for item in list(anchor.get("required_numeric_values", []) or [])
            if str(item).strip()
        ][:10]
        algorithm_terms = [
            str(item).strip()
            for item in list(anchor.get("algorithm_terms", []) or [])
            if str(item).strip()
        ][:12]
        required_groups = 0
        passed_groups = 0
        missing_symbols = [symbol for symbol in symbols if not _formula_anchor_present(searchable, symbol)]
        if symbols:
            required_groups += 1
            symbol_ratio = (len(symbols) - len(missing_symbols)) / max(len(symbols), 1)
            if symbol_ratio >= 0.35:
                passed_groups += 1
        missing_values = [
            value
            for value in numeric_values
            if value.lower() not in searchable
        ]
        if numeric_values:
            required_groups += 1
            value_ratio = (len(numeric_values) - len(missing_values)) / max(len(numeric_values), 1)
            if value_ratio >= 0.5:
                passed_groups += 1
        missing_terms = [
            term
            for term in algorithm_terms
            if term.lower() not in searchable and term.lower().replace(" ", "_") not in searchable
        ]
        if algorithm_terms:
            required_groups += 1
            term_ratio = (len(algorithm_terms) - len(missing_terms)) / max(len(algorithm_terms), 1)
            if term_ratio >= 0.35:
                passed_groups += 1
        anchor_passed = required_groups > 0 and passed_groups == required_groups
        row = {
            "source_id": source_id,
            "section_title": section,
            "missing_symbols": missing_symbols[:12],
            "missing_numeric_values": missing_values[:8],
            "missing_algorithm_terms": missing_terms[:8],
            "required_groups": required_groups,
            "passed_groups": passed_groups,
        }
        if anchor_passed:
            covered.append(row)
        else:
            missing.append(row)
    state.temp_data["post_generate_formula_algorithm_contract_coverage"] = {
        "anchor_count": len(anchors),
        "covered_count": len(covered),
        "missing_count": len(missing),
        "covered": covered[:20],
        "missing": missing[:20],
        "source_files": [path for path, _text in source_items[:20]],
    }
    passed = not missing
    return [
        ValidationCheck(
            name="post_generate:formula_algorithm_contract",
            category="semantic",
            passed=passed,
            details=(
                f"Executable source/config covers all {len(anchors)} paper-derived formula/algorithm anchors."
                if passed
                else "Executable source/config is missing paper-derived formula/algorithm anchors: "
                + "; ".join(
                    f"{item.get('section_title') or item.get('source_id')}: "
                    f"symbols={item.get('missing_symbols', [])[:6]}, "
                    f"values={item.get('missing_numeric_values', [])[:4]}, "
                    f"terms={item.get('missing_algorithm_terms', [])[:4]}"
                    for item in missing[:8]
                )
            ),
            affected_files=[path for path, _text in source_items[:12]],
            affected_work_packages=[
                str(getattr(item, "work_package_id", "") or "")
                for item in list(getattr(getattr(state, "repo_plan", None), "work_packages", []) or [])
                if str(getattr(item, "work_package_id", "") or "")
            ][:8],
        )
    ]


def _post_generate_semantic_surface_checks(
    state: PaperBenchReproState,
    project_root: Path,
    project_files: dict[str, str],
) -> list[ValidationCheck]:
    """Check that generated source exposes a named protocol/environment/method/artifact matrix."""
    source_files = {
        path: content
        for path, content in project_files.items()
        for normalized_path in [normalized_repo_path(path)]
        if path.endswith((".py", ".yaml", ".yml", ".toml", ".json"))
        and not normalized_path.startswith(("results/", "results_", "results-", "outputs/", "artifacts/", "reports/", "figures/", "plots/", "metrics/"))
    }
    source_text = "\n".join(source_files.values()).lower()
    obligation_text = " ".join(
        str(item or "").lower()
        for file_plan in (state.repo_plan.files if state.repo_plan is not None else [])
        for item in [
            file_plan.target_file,
            file_plan.purpose,
            file_plan.decision_value,
            *file_plan.implementation_surfaces,
            *file_plan.method_obligations,
            *file_plan.writes_artifacts,
        ]
    )
    required_artifacts = _repo_required_artifact_paths(state) if state.repo_plan is not None else list(state.project_plan.artifact_contract.required_files)
    requires_matrix = any(token in obligation_text for token in ("experiment", "evaluation", "training", "environment", "dataset", "method", "artifact", "metric", "baseline", "refine"))
    if not requires_matrix and not required_artifacts:
        return [
            ValidationCheck(
                name="semantic:named_protocol_environment_method_artifact_matrix",
                category="semantic",
                passed=True,
                details="No protocol-style semantic matrix obligation was detected.",
            )
        ]

    categories = {
        "protocol_or_experiment": ("protocol", "experiment", "benchmark", "study", "case_study", "run_matrix"),
        "environment_or_dataset": ("environment", "env_id", "dataset", "task_registry", "data_loader", "corpus", "benchmark_task"),
        "method_or_model": ("method", "model", "baseline", "algorithm", "policy", "refiner", "trainer"),
        "artifact_or_metric": ("artifact", "metric", "results/", "outputs/", "write_json", "json.dump", "to_csv", "savefig"),
    }
    missing_categories = [
        category
        for category, tokens in categories.items()
        if not any(token in source_text for token in tokens)
    ]
    generic_only_markers = (
        "toyenv",
        "dummy",
        "placeholder",
        "mock",
        "schema_only",
        "schema-only",
        "dry_run_placeholder",
        "contract_placeholder_not_experimental_result",
        "paper_result_claim\": false",
    )
    generic_hits = [token for token in generic_only_markers if token in source_text]
    passed = not missing_categories and not generic_hits
    details = (
        "Generated source exposes protocol/environment/method/artifact surfaces for semantic review."
        if passed
        else "Generated source lacks semantic scoring surfaces: "
        + ", ".join(missing_categories + generic_hits)
    )
    checks = [
        ValidationCheck(
            name="semantic:named_protocol_environment_method_artifact_matrix",
            category="semantic",
            passed=passed,
            details=details,
            affected_files=list(source_files)[:12],
            affected_work_packages=[
                item.work_package_id
                for item in (state.repo_plan.work_packages if state.repo_plan is not None else [])
            ][:8],
        )
    ]
    evidence_contract = infer_evidence_contract(_state_obligation_text(state))
    evidence_gaps = evidence_contract_gaps(evidence_contract, source_text)
    implementation_gaps = implementation_obligation_gaps(evidence_contract, source_text)
    if evidence_contract.get("requires_evidence_matrix"):
        affected_files = list(source_files)[:12]
        evidence_passed = not evidence_gaps
        evidence_details = (
            "Generated source preserves paper/addendum-derived experiment/method/parameter evidence contract."
            if evidence_passed
            else "Generated source is missing paper/addendum-derived evidence contract terms: "
            + "; ".join(f"{key}={','.join(values[:12])}" for key, values in evidence_gaps.items())
        )
        checks.append(
            ValidationCheck(
                name="post_generate:paper_evidence_contract_matrix",
                category="semantic",
                passed=evidence_passed,
                details=evidence_details,
                affected_files=affected_files,
                affected_work_packages=[
                    item.work_package_id
                    for item in (state.repo_plan.work_packages if state.repo_plan is not None else [])
                ][:8],
            )
        )
        implementation_passed = not implementation_gaps
        implementation_details = (
            "Generated source exposes executable paper-derived implementation paths."
            if implementation_passed
            else "Generated source is missing executable implementation paths: "
            + "; ".join(f"{key}={','.join(values[:12])}" for key, values in implementation_gaps.items())
        )
        checks.append(
            ValidationCheck(
                name="post_generate:paper_implementation_obligation_paths",
                category="semantic",
                passed=implementation_passed,
                details=implementation_details,
                affected_files=affected_files,
                affected_work_packages=[
                    item.work_package_id
                    for item in (state.repo_plan.work_packages if state.repo_plan is not None else [])
                ][:8],
            )
        )
        state.temp_data["post_generate_evidence_contract"] = {
            "contract": flatten_evidence_contract(evidence_contract),
            "gaps": {**evidence_gaps, **implementation_gaps},
            "evidence_gaps": evidence_gaps,
            "implementation_gaps": implementation_gaps,
            "source_files": affected_files,
        }
    bbox_signal = _topic_family_allowed(obligation_text, "bbox_adapter") and any(
        token in obligation_text
        for token in ("bbox-adapter", "bbox_adapter", "bbox adapter", "ranking nce", "ranking-based nce", "black-box adapter")
    )
    if bbox_signal:
        active_source_files = {
            normalized_repo_path(path): content
            for path, content in source_files.items()
            if _is_active_route_surface(path)
        }
        active_source_text = "\n".join(active_source_files.values()).lower()
        bbox_requirements = {
            "appendix_h2_backbones": (
                "microsoft/deberta-v3-base",
                "microsoft/deberta-v3-large",
                "bert-base-cased",
            ),
            "appendix_h2_backbone_task_routes": (
                "strategyqa",
                "gsm8k",
                "scienceqa",
                "truthfulqa",
                "deberta-v3-base",
                "deberta-v3-large",
                "bert-base-cased",
            ),
            "eq3_energy_loss": (
                "positive_quadratic_regularization",
                "negative_quadratic_regularization",
                "5e-6",
            ),
            "eq3_active_loss_route": (
                "positive",
                "negative",
                "nce",
                "train_adapter",
            ),
            "spectral_normalization": ("spectral_norm", "spectral_normalization"),
            "algorithm1_state_updates": ("y_i+^(t)", "y_i-^(t)", "eq5", "eq6", "eq7"),
            "sentence_partial_beam": ("s_1", "p_llm", "partial_chain", "stop_signal", "candidate"),
            "paper_dataset_splits": ("7473", "1319", "2059", "229", "717", "100", "2000", "500"),
            "appendix_j_cot_prompt_routes": ("strategyqa", "gsm8k", "scienceqa", "cot", "prompt"),
            "mixtral_lora_table8": ("128", "384", "256", "768", "paged_adamw_32bit", "cosine"),
            "azure_sft_and_costs": ("fine_tuning.jobs.create", "api_usage", "cost_per_1000", "loss_curve"),
        }
        active_route_requirement_ids = {
            "appendix_h2_backbone_task_routes",
            "eq3_active_loss_route",
            "algorithm1_state_updates",
            "sentence_partial_beam",
            "paper_dataset_splits",
            "appendix_j_cot_prompt_routes",
            "mixtral_lora_table8",
            "azure_sft_and_costs",
        }
        bbox_missing: list[str] = []
        bbox_missing_tokens: dict[str, list[str]] = {}
        for key, tokens in bbox_requirements.items():
            haystack = active_source_text if key in active_route_requirement_ids else source_text
            missing_tokens = [token for token in tokens if token.lower() not in haystack]
            if missing_tokens:
                bbox_missing.append(key)
                bbox_missing_tokens[key] = missing_tokens
        checks.append(
            ValidationCheck(
                name="post_generate:bbox_paper_exact_protocol",
                category="semantic",
                passed=not bbox_missing,
                details=(
                    "BBox-Adapter paper-exact protocol surfaces are present in active source."
                    if not bbox_missing
                    else "BBox-Adapter source is missing paper-exact active protocol surfaces: "
                    + ", ".join(bbox_missing)
                ),
                affected_files=list(source_files)[:12],
                affected_work_packages=[
                    item.work_package_id
                    for item in (state.repo_plan.work_packages if state.repo_plan is not None else [])
                ][:8],
            )
        )
        state.temp_data["post_generate_bbox_paper_exact_protocol"] = {
            "missing": bbox_missing,
            "missing_tokens": bbox_missing_tokens,
            "requirements": {key: list(tokens) for key, tokens in bbox_requirements.items()},
            "source_files": list(source_files)[:12],
            "active_route_files": list(active_source_files)[:12],
        }
    return checks


def _merge_preflight_contract_failures(
    preflight_result: PreflightResult,
    contract_failures: list[ValidationCheck],
) -> PreflightResult:
    """Merge explicit contract failures into the checker-produced preflight result."""
    if not contract_failures:
        return preflight_result
    checks_payload = list(preflight_result.checks)
    blocking_failures = list(preflight_result.blocking_failures)
    suggested_fixes = list(preflight_result.suggested_fixes)
    seen_messages = set(blocking_failures)
    for check in contract_failures:
        payload = check.model_dump(mode="json")
        payload["severity"] = "blocking"
        payload["message"] = check.details
        checks_payload.append(payload)
        if check.details not in seen_messages:
            blocking_failures.append(check.details)
            suggested_fixes.append(check.details)
            seen_messages.add(check.details)
    return preflight_result.model_copy(
        update={
            "status": "failed",
            "checks": checks_payload,
            "blocking_failures": blocking_failures,
            "suggested_fixes": suggested_fixes,
        }
    )


def _build_preflight_repair_ticket(
    state: PaperBenchReproState,
    preflight_result: PreflightResult,
    contract_failures: list[ValidationCheck],
) -> RepairTicket | None:
    """Build a concrete repair ticket when generation is incomplete before docker validation."""
    if not contract_failures:
        return None
    main_failures = [item for item in contract_failures if item.name == "preflight:main_entry_exists"]
    artifact_owner_failures = [item for item in contract_failures if item.name.startswith("preflight:artifact_owner_surface:")]
    other_missing_files = [
        item for item in contract_failures if item.name.startswith("preflight:required_file:")
    ]

    if main_failures:
        main_entry = main_failures[0].affected_files[0] if main_failures[0].affected_files else _canonical_entry_surface(state)
        return RepairTicket(
            failure_type="missing_entrypoint_file",
            reason="Generation is incomplete because the declared main entrypoint is missing.",
            trigger_signals=[item.details for item in main_failures[:4]],
            evidence={
                "preflight_result": preflight_result.model_dump(mode="json"),
                "contract_failures": [item.model_dump(mode="json") for item in contract_failures[:8]],
            },
            allowed_changes=[
                "Implement the declared main entrypoint and its direct wiring without inventing placeholder code.",
            ],
            required_fix_targets=[main_entry] if main_entry else [],
            next_fix_scope=["main entrypoint implementation", "entry wiring"],
            forbidden_changes=["Do not create NotImplementedError or placeholder entry files."],
        )

    if artifact_owner_failures:
        affected_files = [
            path
            for item in artifact_owner_failures
            for path in item.affected_files
            if path
        ]
        return RepairTicket(
            failure_type="artifact_owner_surface_missing",
            reason="Generation is incomplete because an artifact contract points to a missing producer surface.",
            trigger_signals=[item.details for item in artifact_owner_failures[:4]],
            evidence={
                "preflight_result": preflight_result.model_dump(mode="json"),
                "contract_failures": [item.model_dump(mode="json") for item in artifact_owner_failures[:8]],
            },
            allowed_changes=[
                "Implement the missing artifact-owner surfaces and restore the declared artifact contract wiring.",
            ],
            required_fix_targets=list(dict.fromkeys(affected_files)),
            next_fix_scope=["artifact owner surfaces", "artifact contract wiring"],
            forbidden_changes=["Do not fake artifact files without the producing code path."],
        )

    if other_missing_files:
        affected_files = [
            path
            for item in other_missing_files
            for path in item.affected_files
            if path
        ]
        return RepairTicket(
            failure_type="generation_incomplete",
            reason="Generation is incomplete because required planned files are still missing.",
            trigger_signals=[item.details for item in other_missing_files[:6]],
            evidence={
                "preflight_result": preflight_result.model_dump(mode="json"),
                "contract_failures": [item.model_dump(mode="json") for item in other_missing_files[:8]],
            },
            allowed_changes=[
                "Implement the missing planned files and keep the repo contract consistent.",
            ],
            required_fix_targets=list(dict.fromkeys(affected_files)),
            next_fix_scope=["missing planned files"],
            forbidden_changes=["Do not add placeholder modules or stub files just to satisfy file presence checks."],
        )

    return RepairTicket(
        failure_type="preflight_validation_failed",
        reason="Preflight validation surfaced contract issues before docker validation.",
        trigger_signals=list(preflight_result.blocking_failures)[:6],
        evidence={"preflight_result": preflight_result.model_dump(mode="json")},
        allowed_changes=["Fix the preflight contract failures while preserving forward progress through docker validation."],
        required_fix_targets=[],
        next_fix_scope=["preflight failures"],
        forbidden_changes=[],
    )


def _record_degraded_validation_issue(
    state: PaperBenchReproState,
    *,
    failure_type: str,
    reason: str,
    signals: list[str] | None = None,
    affected_files: list[str] | None = None,
) -> None:
    payload = {
        "stage": "preflight",
        "failure_type": str(failure_type or "").strip(),
        "reason": str(reason or "").strip(),
        "signals": [item for item in list(signals or []) if str(item).strip()],
        "affected_files": [item for item in list(affected_files or []) if str(item).strip()],
    }
    backlog = state.temp_data.setdefault("degraded_backlog", [])
    if isinstance(backlog, list) and payload not in backlog:
        backlog.append(payload)


def _augment_repair_ticket_with_planning_issues(
    state: PaperBenchReproState,
    ticket: RepairTicket | None,
) -> RepairTicket | None:
    """Attach unresolved plan-stage review issues as advisory repair context."""
    architecture_failures = (
        list(state.architecture.unresolved_review_failures)
        if state.architecture is not None
        else []
    )
    package_file_plan_failures = (
        list(state.package_file_planning_output.unresolved_review_failures)
        if state.package_file_planning_output is not None
        else []
    )
    if not architecture_failures and not package_file_plan_failures:
        return ticket
    trigger_signals = list(ticket.trigger_signals) if ticket is not None else []
    trigger_signals.extend(
        [f"architecture_review_issue:{item}" for item in architecture_failures if str(item).strip()]
    )
    trigger_signals.extend(
        [f"package_file_planning_review_issue:{item}" for item in package_file_plan_failures if str(item).strip()]
    )
    evidence = dict(ticket.evidence) if ticket is not None else {}
    evidence["planning_review_issues"] = {
        "architecture": architecture_failures,
        "package_file_planning": package_file_plan_failures,
    }
    required_fix_targets = list(ticket.required_fix_targets) if ticket is not None else []
    required_fix_targets.extend(_canonical_surface_paths(state, "entrypoint")[:4])
    required_fix_targets.extend(_canonical_surface_paths(state, "config", "stable_interface")[:4])
    if state.repo_plan is not None and state.repo_plan.canonical_route.entry_surface:
        required_fix_targets.append(state.repo_plan.canonical_route.entry_surface)
    if state.package_file_planning_output is not None:
        required_fix_targets.extend(
            [
                item.target_file
                for item in state.package_file_planning_output.file_plans
                if str(item.target_file).strip()
            ][:8]
        )
    next_fix_scope = list(ticket.next_fix_scope) if ticket is not None else []
    next_fix_scope.append("advisory planning drift context")
    if ticket is None:
        return RepairTicket(
            failure_type="planning_review_advisory",
            reason=(
                "Plan-stage review left advisory issues. Do not treat them as a blocking contract; "
                "repair may rewrite repo surfaces, entrypoints, and artifact wiring to satisfy the target and runtime handoff."
            ),
            trigger_signals=list(dict.fromkeys(trigger_signals))[:12],
            evidence=evidence,
            allowed_changes=[
                "Use unresolved planning issues as hints, not as required closure gates.",
                "Rewrite repo structure, entrypoints, and artifact wiring when needed to satisfy the target and runtime handoff.",
            ],
            required_fix_targets=list(dict.fromkeys([item for item in required_fix_targets if str(item).strip()])),
            next_fix_scope=list(dict.fromkeys([item for item in next_fix_scope if str(item).strip()])),
            forbidden_changes=[],
        )
    return ticket.model_copy(
        update={
            "trigger_signals": list(dict.fromkeys(trigger_signals))[:12],
            "evidence": evidence,
            "required_fix_targets": list(dict.fromkeys([item for item in required_fix_targets if str(item).strip()])),
            "next_fix_scope": list(dict.fromkeys([item for item in next_fix_scope if str(item).strip()])),
        }
    )


def run_repo_execution_validation(
    state: PaperBenchReproState,
    *,
    get_workflow_config: Callable[[], Any],
    get_output_dir: Callable[[PaperBenchReproState], Path],
) -> tuple[PreflightResult, ExecutionResult]:
    """Run thin preflight plus docker-based repo validation."""
    from reproagent.pipeline.tools import load_project_files
    from reproagent.pipeline.utils.preflight_checker import PreflightChecker

    project_root = Path(state.project_root) if state.project_root else (get_output_dir(state) / "repo")
    project_files = load_project_files(project_root)
    if not project_files:
        failed_preflight = PreflightResult(
            status="failed",
            checks=[],
            blocking_failures=["generated repo snapshot is empty"],
            warning_messages=[],
            suggested_fixes=["Persist generated project files before validation."],
        )
        failed_execution = ExecutionResult(
            success=False,
            output="",
            error="generated repo snapshot is empty",
            exit_code=1,
            metrics={},
            checks=[],
            artifacts=[],
            artifact_summary={},
        )
        state.repair_ticket = RepairTicket(
            failure_type="repo_snapshot_missing",
            reason="Generated repo snapshot is empty, so reproagent cannot run docker validation or hand off a runnable repository.",
            trigger_signals=["generated repo snapshot is empty"],
            evidence={"project_root": str(project_root)},
            allowed_changes=["Persist the generated repo files before starting validation."],
            required_fix_targets=[],
            next_fix_scope=["repo snapshot persistence"],
            forbidden_changes=[],
        )
        state.temp_data["repair_ticket"] = state.repair_ticket.model_dump(mode="json")
        return failed_preflight, failed_execution

    project_plan_payload = state.project_plan.model_dump(mode="json")
    preflight_payload = PreflightChecker().run(project_root, project_plan_payload, project_files)
    preflight_result = PreflightResult.model_validate(preflight_payload)
    contract_failures = _preflight_contract_failures(state, project_root, project_files)
    preflight_result = _merge_preflight_contract_failures(preflight_result, contract_failures)
    if contract_failures:
        state.repair_ticket = _build_preflight_repair_ticket(state, preflight_result, contract_failures)
        for failure in contract_failures:
            _record_degraded_validation_issue(
                state,
                failure_type=(
                    "missing_entrypoint_file"
                    if failure.name == "preflight:main_entry_exists"
                    else "missing_required_file"
                    if failure.name.startswith("preflight:required_file:")
                    else "preflight_contract_issue"
                ),
                reason=failure.details,
                signals=[failure.name, failure.details],
                affected_files=list(failure.affected_files),
            )
        state.temp_data["repair_ticket"] = (
            state.repair_ticket.model_dump(mode="json")
            if state.repair_ticket is not None
            else {}
        )

    runtime_root = get_output_dir(state) / "repo_validation_runtime"
    original_root = _copy_repo_tree(project_root, runtime_root / "original")
    working_root = _copy_repo_tree(project_root, runtime_root / "working")

    workflow_config = get_workflow_config()
    timeout_seconds = int(getattr(workflow_config, "docker_validate_timeout", 600) or 600)
    command_template = _validation_command_template(state)
    smoke = _run_validation_smoke_with_artifact_contract(
        state,
        _render_validation_command(command_template, variant=_CANONICAL_RUNTIME_VARIANT),
        working_root,
        timeout_seconds=timeout_seconds,
    )
    variants = [
        {
            "variant": _SINGLE_REPO_VARIANT_LABEL,
            "workspace": str(working_root),
            "smoke": smoke,
        }
    ]
    smoke_payload = {
        "status": "passed" if bool(smoke.get("success")) else "failed",
        "command_template": command_template,
        "variants": variants,
    }
    validation_payload = _validate_repo_smoke_payload(smoke_payload)
    smoke_payload["validation"] = validation_payload
    state.temp_data["docker_validation"] = {
        "runtime_root": str(runtime_root),
        "original_root": str(original_root),
        "working_root": str(working_root),
        "command_template": command_template,
        "smoke_payload": smoke_payload,
        "validation_payload": validation_payload,
    }

    stdout = "\n\n".join(
        f"[{item.get('variant', 'unknown')}]\n{dict(item.get('smoke', {}) or {}).get('output', '')}".strip()
        for item in variants
        if str(dict(item.get("smoke", {}) or {}).get("output", "")).strip()
    ).strip()
    stderr_chunks: list[str] = []
    for item in variants:
        smoke = dict(item.get("smoke", {}) or {})
        error_text = _first_nonempty([
            str(smoke.get("error", "") or ""),
            str(smoke.get("diagnostics", "") or ""),
            str(smoke.get("stderr", "") or ""),
        ])
        if error_text:
            stderr_chunks.append(f"[{item.get('variant', 'unknown')}]\n{error_text}")
    validation_failures = "\n".join(str(item).strip() for item in validation_payload.get("failures", []) or [] if str(item).strip())
    if validation_failures:
        stderr_chunks.append(validation_failures)
    stderr = "\n\n".join(stderr_chunks).strip()
    metrics_path = _first_nonempty(
        [str(item.get("evaluation_result_path", "") or "") for item in validation_payload.get("variant_reports", []) or []]
    )
    artifact_summary = {
        str(item.get("variant", "") or "unknown"): {
            "workspace": str(item.get("workspace", "") or ""),
            "readiness_path": str(item.get("readiness_path", "") or ""),
            "evaluation_result_path": str(item.get("evaluation_result_path", "") or ""),
            "smoke_success": bool(item.get("smoke_success")),
            "failures": list(item.get("failures", []) or []),
        }
        for item in validation_payload.get("variant_reports", []) or []
        if isinstance(item, dict)
    }
    execution_result = ExecutionResult.model_validate(
        {
            "success": str(validation_payload.get("status", "") or "") == "passed",
            "output": stdout,
            "error": stderr,
            "exit_code": 0 if str(validation_payload.get("status", "") or "") == "passed" else 1,
            "metrics": {
                "validation_passed": 1.0 if str(validation_payload.get("status", "") or "") == "passed" else 0.0,
            },
            "checks": [
                {
                    "name": "docker_runtime_execution",
                    "passed": str(smoke_payload.get("status", "") or "") == "passed",
                    "details": "docker validation command status",
                },
                {
                    "name": "docker_runtime_artifact_contract",
                    "passed": str(validation_payload.get("status", "") or "") == "passed",
                    "details": "docker validation artifact-contract status",
                },
            ],
            "artifacts": [
                {"path": str(item.get("readiness_path", "") or ""), "kind": "readiness"}
                for item in validation_payload.get("variant_reports", []) or []
                if isinstance(item, dict) and str(item.get("readiness_path", "") or "")
            ]
            + [
                {"path": str(item.get("evaluation_result_path", "") or ""), "kind": "evaluation_result"}
                for item in validation_payload.get("variant_reports", []) or []
                if isinstance(item, dict) and str(item.get("evaluation_result_path", "") or "")
            ],
            "artifact_summary": artifact_summary,
        }
    )
    state.experiment_results = {}
    docker_repair_ticket = _build_docker_validation_repair_ticket(
        state,
        smoke_payload,
        validation_payload,
    )
    if docker_repair_ticket is not None:
        if state.repair_ticket is not None:
            merged_signals = list(dict.fromkeys([
                *list(state.repair_ticket.trigger_signals),
                *list(docker_repair_ticket.trigger_signals),
            ]))
            merged_targets = list(dict.fromkeys([
                *list(state.repair_ticket.required_fix_targets),
                *list(docker_repair_ticket.required_fix_targets),
            ]))
            merged_scope = list(dict.fromkeys([
                *list(state.repair_ticket.next_fix_scope),
                *list(docker_repair_ticket.next_fix_scope),
            ]))
            merged_evidence = dict(state.repair_ticket.evidence)
            merged_evidence["docker_validation"] = docker_repair_ticket.evidence
            state.repair_ticket = state.repair_ticket.model_copy(
                update={
                    "trigger_signals": merged_signals[:12],
                    "required_fix_targets": merged_targets[:12],
                    "next_fix_scope": merged_scope[:12],
                    "evidence": merged_evidence,
                }
            )
        else:
            state.repair_ticket = docker_repair_ticket
    state.temp_data["repair_ticket"] = (
        state.repair_ticket.model_dump(mode="json")
        if state.repair_ticket is not None
        else {}
    )
    return preflight_result, execution_result


def build_validated_repo_handoff(
    state: PaperBenchReproState,
    *,
    build_repo_handoff_payload: Callable[[PaperBenchReproState], dict[str, Any]],
    get_output_dir: Callable[[PaperBenchReproState], Path],
) -> dict[str, Any]:
    """Materialize a canonical rapid-validation repository handoff."""
    if not state.project_root or state.validation_report is None or not state.validation_report.passed:
        return {}
    validation_quality_level = str(state.validation_report.quality_level or "").strip().lower()
    if validation_quality_level in {"", "scaffold_only", "unverified"}:
        return {}

    source_root = Path(state.project_root)
    if not source_root.exists():
        return {}

    docker_validation = dict(state.temp_data.get("docker_validation", {}) or {})
    handoff_root = get_output_dir(state) / "validated_repo_init"
    original_source = Path(str(docker_validation.get("original_root", "") or "")) if docker_validation.get("original_root") else source_root
    working_source = Path(str(docker_validation.get("working_root", "") or "")) if docker_validation.get("working_root") else source_root
    original_root = _copy_repo_tree(original_source, handoff_root / "original")
    working_root = _copy_repo_tree(working_source, handoff_root / "working")

    smoke_payload = dict(docker_validation.get("smoke_payload", {}) or {})
    validation_payload = dict(docker_validation.get("validation_payload", {}) or smoke_payload.get("validation", {}) or {})
    if not smoke_payload:
        command_template = ""
        try:
            payload = build_repo_handoff_payload(state)
            workspace_config = dict(payload.get("workspace_config", {}) or {})
            init_repo = dict(payload.get("init_repo", {}) or {})
            command_template = str(
                workspace_config.get("smoke_command")
                or init_repo.get("entrypoint_hint")
                or ""
            ).strip()
        except Exception:
            command_template = ""
        if not command_template:
            command_template = _validation_command_template(state)
        timeout_seconds = 300
        smoke = _run_validation_smoke_with_artifact_contract(
            state,
            _render_validation_command(command_template, variant=_CANONICAL_RUNTIME_VARIANT),
            working_root,
            timeout_seconds=timeout_seconds,
        )
        variants = [
            {
                "variant": _SINGLE_REPO_VARIANT_LABEL,
                "workspace": str(working_root),
                "smoke": smoke,
            }
        ]
        smoke_payload = {
            "status": "passed" if bool(smoke.get("success")) else "failed",
            "command_template": command_template,
            "variants": variants,
        }
        validation_payload = _validate_repo_smoke_payload(smoke_payload)
    smoke_payload["validation"] = validation_payload

    repo_init_payload = {
        "status": "completed",
        "repo_scope": "single_canonical_repo",
        "canonical_validated_repo_root": str(source_root),
        "source_init_repo": str(source_root),
        "repo_init_root": str(handoff_root),
        "original_root": str(original_root),
        "working_root": str(working_root),
        "workspace_materialization_owner": "reproagent",
        "variant_materialization_mode": "single_repo_materialized",
        "repair_iterations": [],
        "review_history": [],
        "latest_review": {
            "verdict": "accept" if state.validation_report and state.validation_report.passed else "revise",
            "reason": (
                "reproagent validation produced a runnable repo handoff."
                if state.validation_report and state.validation_report.passed
                else "; ".join(state.validation_report.blocked_reasons) if state.validation_report else "validation failed"
            ),
            "must_fix": (
                list(state.repair_ticket.trigger_signals)
                if state.repair_ticket is not None
                else (list(state.validation_report.repair_recommendations) if state.validation_report else [])
            ),
        },
        "repair_ticket": state.repair_ticket.model_dump(mode="json") if state.repair_ticket is not None else {},
        "latest_smoke": smoke_payload,
        "latest_validation": validation_payload,
    }
    repo_contract = build_stage1_repo_contract(state, repo_path=working_root)
    repo_init_payload.update(
        {
            "repo_path": str(working_root),
            "repo_source": repo_contract["repo_source"],
            "entrypoint_hint": repo_contract["entrypoint_hint"],
            "install_command": repo_contract["install_command"],
            "variant_mode": repo_contract["variant_mode"],
            "baseline_command": repo_contract["baseline_command"],
            "idea_command": repo_contract["idea_command"],
            "variant_command": repo_contract["variant_command"],
            "smoke_command": repo_contract["smoke_command"],
            "command_contract": dict(repo_contract.get("command_contract", {}) or {}),
            "metric_paths": list(repo_contract["metric_paths"]),
            "editable_paths": list(repo_contract["editable_paths"]),
            "protected_paths": list(repo_contract["protected_paths"]),
            "repo_contract": repo_contract,
        }
    )
    handoff_ready = (
        bool(state.validation_report and state.validation_report.passed)
        and validation_quality_level not in {"", "scaffold_only", "unverified"}
        and bool(validation_payload)
        and str(validation_payload.get("status", "") or "").strip().lower() == "passed"
        and not list(validation_payload.get("failures", []) or [])
    )
    quality_status_payload = build_quality_status(state).model_dump(mode="json")
    quality_status_payload["handoff_ready"] = handoff_ready
    if str(quality_status_payload.get("quality_status", "") or "") in {"validated", "repaired"}:
        quality_status_payload["next_recommended_action"] = (
            "inspect_repo_handoff" if handoff_ready else "review_partial_artifacts_or_rerun"
        )
    return {
        "schema_version": _VALIDATED_REPO_HANDOFF_SCHEMA_VERSION,
        "producer": "reproagent",
        "run_id": state.run_id,
        "thread_id": state.input.thread_id or state.run_id,
        "target": state.input.target,
        "start_stage": "rapid_validation",
        "handoff_ready": handoff_ready,
        "quality_status": quality_status_payload,
        "repo_root": str(working_root),
        "project_root": str(source_root),
        "workspace_config": {
            "project_root": str(source_root),
            "target_repo_path": str(working_root),
            "baseline_command": repo_contract["baseline_command"],
            "idea_command": repo_contract["idea_command"],
            "variant_command": repo_contract["variant_command"],
            "smoke_command": repo_contract["smoke_command"],
            "metric_paths": list(repo_contract["metric_paths"]),
            "editable_paths": list(repo_contract["editable_paths"]),
            "protected_paths": list(repo_contract["protected_paths"]),
        },
        "init_repo": {
            "repo_path": str(working_root),
            "source": repo_contract["repo_source"],
            "variant_mode": repo_contract["variant_mode"],
            "entrypoint_hint": repo_contract["entrypoint_hint"],
            "install_command": repo_contract["install_command"],
            "editable_paths": list(repo_contract["editable_paths"]),
            "protected_paths": list(repo_contract["protected_paths"]),
        },
        "rapid_validation": {
            "repo_materialization": repo_init_payload,
            "smoke_validation": smoke_payload,
        }
    }


def _extract_file_from_entrypoint(command: str, file_keys: dict[str, str]) -> str:
    """Extract the backing file path from an execution entrypoint command string.

    Handles patterns like:
    - ``python -m src.spar.main`` -> ``src/spar/main.py``
    - ``python scripts/run_benchmark.py`` -> ``scripts/run_benchmark.py``
    - ``src/spar/main.py`` -> ``src/spar/main.py``
    """
    command = str(command or "").strip()
    if not command:
        return ""
    parts = command.split()
    # Direct file path
    if len(parts) == 1:
        key = normalized_repo_key(parts[0])
        if key in file_keys:
            return file_keys[key]
        # Try as module path
        module_path = parts[0].replace(".", "/") + ".py"
        key = normalized_repo_key(module_path)
        if key in file_keys:
            return file_keys[key]
        return ""
    # ``python -m module.path`` pattern
    if "-m" in parts:
        m_index = parts.index("-m")
        if m_index + 1 < len(parts):
            module = parts[m_index + 1]
            module_path = module.replace(".", "/") + ".py"
            key = normalized_repo_key(module_path)
            if key in file_keys:
                return file_keys[key]
            # Try __init__.py for package
            package_init = module.replace(".", "/") + "/__init__.py"
            key = normalized_repo_key(package_init)
            if key in file_keys:
                return file_keys[key]
        return ""
    # ``python script.py [args...]`` pattern
    for part in parts[1:]:
        if part.startswith("-"):
            continue
        key = normalized_repo_key(part)
        if key in file_keys:
            return file_keys[key]
    return ""


def _extract_file_from_interface(declaration: str, file_keys: dict[str, str]) -> str:
    """Extract the backing file path from a stable interface declaration.

    Handles patterns like:
    - ``src/spar/models.py: QueryPlan, CandidatePaper`` -> ``src/spar/models.py``
    - ``src/spar/pipeline.py: run_pipeline(...)`` -> ``src/spar/pipeline.py``
    - ``src/spar/models.py`` -> ``src/spar/models.py``
    """
    declaration = str(declaration or "").strip()
    if not declaration:
        return ""
    # Split on `:` to get file path prefix
    if ":" in declaration:
        file_part = declaration.split(":", 1)[0].strip()
    else:
        file_part = declaration
    key = normalized_repo_key(file_part)
    if key in file_keys:
        return file_keys[key]
    return ""


def _check_config_surface_coverage(
    config_names: list[str],
    state: PaperBenchReproState,
    file_keys: dict[str, str],
) -> list[str]:
    """Check config surface coverage by looking at config-like files in the repo plan.

    Config surfaces are typically env var names or config keys, not file paths.
    We check if config-like files exist that could declare them.
    """
    if not config_names:
        return []
    # Identify config-like files in the plan
    config_file_paths: list[str] = []
    for key, path in file_keys.items():
        lowered = key.lower()
        if any(token in lowered for token in (
            ".env", "config", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".json",
        )):
            config_file_paths.append(path)
    # If we have config files that could host these declarations, consider them covered
    if config_file_paths:
        return []
    # No config files at all -> these are genuinely unmapped
    return config_names


def repo_plan_static_contract_checks(
    state: PaperBenchReproState,
    *,
    entrypoint_related_work_packages: Callable[[PaperBenchReproState], list[str]],
) -> list[ValidationCheck]:
    """Validate repo-plan execution closure directly from formal repo-plan objects."""
    if state.repo_plan is None:
        return []

    file_map = {item.target_file: item for item in state.repo_plan.files}
    file_keys = {normalized_repo_key(path): path for path in file_map}
    surface_paths = [item.path for item in state.repo_plan.stage_public_surfaces if item.path]
    surface_keys = {normalized_repo_key(path): path for path in surface_paths}
    artifact_contract_keys = {
        normalized_repo_key(item.relative_path): item
        for item in state.repo_plan.artifact_contract
        if normalized_repo_key(item.relative_path)
    }

    checks: list[ValidationCheck] = []
    canonical_ir = dict(state.repo_plan.canonical_ir or {})
    canonical_surface_nodes = list(canonical_ir.get("surface_nodes", []) or [])
    canonical_entrypoints = [
        str(item.get("canonical_path") or "").strip()
        for item in canonical_surface_nodes
        if isinstance(item, dict) and str(item.get("surface_kind") or "").strip() == "entrypoint"
    ]
    canonical_config_surfaces = [
        str(item.get("canonical_path") or "").strip()
        for item in canonical_surface_nodes
        if isinstance(item, dict) and str(item.get("surface_kind") or "").strip() == "config"
    ]
    canonical_stable_interfaces = [
        str(item.get("canonical_path") or "").strip()
        for item in canonical_surface_nodes
        if isinstance(item, dict) and str(item.get("surface_kind") or "").strip() == "stable_interface"
    ]

    unmapped_entrypoints = [
        ep for ep in canonical_entrypoints
        if ep and normalized_repo_key(ep) not in file_keys
        and normalized_repo_key(ep) not in surface_keys
    ]
    checks.append(
        _as_advisory_planning_check(
            ValidationCheck(
                name="integration:repo_plan:execution_entrypoints",
                category="integration",
                passed=not unmapped_entrypoints,
                details=(
                    "all canonical entrypoint surfaces are backed by repo-plan files."
                    if not unmapped_entrypoints
                    else f"unmapped canonical entrypoint surfaces: {unmapped_entrypoints[:8]}"
                ),
                affected_files=list(unmapped_entrypoints),
                affected_work_packages=entrypoint_related_work_packages(state),
            )
        )
    )

    unmapped_configs = [
        path for path in canonical_config_surfaces
        if path and normalized_repo_key(path) not in file_keys
        and normalized_repo_key(path) not in surface_keys
    ]
    checks.append(
        _as_advisory_planning_check(
            ValidationCheck(
                name="integration:repo_plan:config_surfaces",
                category="integration",
                passed=not unmapped_configs,
                details=(
                    "all canonical config surfaces are backed by repo-plan files."
                    if not unmapped_configs
                    else f"unmapped canonical config surfaces: {unmapped_configs[:8]}"
                ),
                affected_files=list(unmapped_configs),
                affected_work_packages=entrypoint_related_work_packages(state),
            )
        )
    )

    unmapped_interfaces = [
        iface for iface in canonical_stable_interfaces
        if iface
        and normalized_repo_key(iface) not in file_keys
        and normalized_repo_key(iface) not in surface_keys
    ]
    checks.append(
        _as_advisory_planning_check(
            ValidationCheck(
                name="integration:repo_plan:stable_interfaces",
                category="integration",
                passed=not unmapped_interfaces,
                details=(
                    "all canonical stable interfaces are backed by repo-plan files."
                    if not unmapped_interfaces
                    else f"unmapped canonical stable interfaces: {unmapped_interfaces[:8]}"
                ),
                affected_files=list(unmapped_interfaces),
                affected_work_packages=entrypoint_related_work_packages(state),
            )
        )
    )

    canonical_route = state.repo_plan.canonical_route
    if canonical_route.entry_surface or canonical_route.expected_outputs:
        issues: list[str] = []
        entry_key = normalized_repo_key(canonical_route.entry_surface)
        if entry_key and entry_key not in file_keys and entry_key not in surface_keys:
            issues.append(f"canonical entry surface is not mapped: {canonical_route.entry_surface}")
        expected_output_keys = normalized_repo_keys(list(canonical_route.expected_outputs))
        planned_output_keys = set(artifact_contract_keys)
        for file_plan in state.repo_plan.files:
            planned_output_keys.update(file_plan_artifact_keys(file_plan))
        missing_outputs = sorted(expected_output_keys - planned_output_keys)
        if missing_outputs:
            issues.append(f"canonical expected outputs are not wired: {missing_outputs[:6]}")
        checks.append(
            _as_advisory_planning_check(
                ValidationCheck(
                    name="integration:repo_plan:canonical_route",
                    category="integration",
                    passed=not issues,
                    details=(
                        "canonical route is statically wired through repo-plan files and declared outputs."
                        if not issues
                        else "; ".join(issues)
                    ),
                    affected_files=[canonical_route.entry_surface] if canonical_route.entry_surface else [],
                    affected_work_packages=entrypoint_related_work_packages(state),
                )
            )
        )

    missing_stage_surfaces = [
        item.path
        for item in state.repo_plan.stage_public_surfaces
        if item.path and normalized_repo_key(item.path) not in file_keys
    ]
    checks.append(
        _as_advisory_planning_check(
            ValidationCheck(
                name="integration:repo_plan:stage_public_surfaces",
                category="integration",
                passed=not missing_stage_surfaces,
                details=(
                    "all stage public surfaces are mapped by repo-plan files."
                    if not missing_stage_surfaces
                    else f"unmapped stage public surfaces: {missing_stage_surfaces[:8]}"
                ),
                affected_files=missing_stage_surfaces,
                affected_work_packages=entrypoint_related_work_packages(state),
            )
        )
    )

    artifact_contract_issues: list[str] = []
    affected_work_packages: set[str] = set()
    for item in state.repo_plan.artifact_contract:
        producer_key = normalized_repo_key(item.producer_surface)
        relative_key = normalized_repo_key(item.relative_path)
        if item.owner_work_package:
            affected_work_packages.add(item.owner_work_package)
        if producer_key and producer_key not in file_keys and producer_key not in surface_keys:
            artifact_contract_issues.append(f"{item.artifact_key}:missing producer surface {item.producer_surface}")
            continue
        producer_file = file_map.get(item.producer_surface)
        planned_keys = file_plan_artifact_keys(producer_file)
        if relative_key and relative_key not in planned_keys and relative_key not in artifact_contract_keys:
            artifact_contract_issues.append(f"{item.artifact_key}:unwired artifact path {item.relative_path}")
    checks.append(
        _as_advisory_planning_check(
            ValidationCheck(
                name="integration:repo_plan:artifact_contract",
                category="integration",
                passed=not artifact_contract_issues,
                details=(
                    "artifact contract entries are mapped to producer surfaces and planned outputs."
                    if not artifact_contract_issues
                    else "; ".join(artifact_contract_issues[:8])
                ),
                affected_work_packages=sorted(affected_work_packages),
            )
        )
    )
    unresolved_failures = list(state.repo_plan.architecture.unresolved_review_failures or [])
    checks.append(
        _as_advisory_planning_check(
            ValidationCheck(
                name="integration:repo_plan:architecture_review_closure",
                category="integration",
                passed=not unresolved_failures,
                details=(
                    "architecture review closure is clean."
                    if not unresolved_failures
                    else "unresolved architecture review failures remain: " + "; ".join(unresolved_failures[:6])
                ),
                affected_work_packages=entrypoint_related_work_packages(state),
            )
        )
    )
    return checks


def repo_plan_artifact_owner_map(state: PaperBenchReproState) -> dict[str, list[str]]:
    """Build artifact-path to owner-work-package map directly from repo_plan."""
    if state.repo_plan is None:
        return {}
    owner_map: dict[str, list[str]] = {}
    for item in state.repo_plan.artifact_contract:
        artifact_key = normalized_repo_key(item.relative_path)
        if not artifact_key or not item.owner_work_package:
            continue
        owners = owner_map.setdefault(artifact_key, [])
        if item.owner_work_package not in owners:
            owners.append(item.owner_work_package)
    for file_plan in state.repo_plan.files:
        for path in file_plan.writes_artifacts:
            artifact_key = normalized_repo_key(path)
            if not artifact_key or not file_plan.work_package_id:
                continue
            owners = owner_map.setdefault(artifact_key, [])
            if file_plan.work_package_id not in owners:
                owners.append(file_plan.work_package_id)
    return owner_map


def _global_contract_artifact_satisfied_by_execution(
    state: PaperBenchReproState,
    artifact_path: str,
    repo_root: Path | None = None,
) -> bool:
    normalized = normalized_repo_path(artifact_path)
    if not normalized:
        return False
    search_roots: list[Path] = []
    if repo_root is not None:
        search_roots.extend(_validation_artifact_search_roots(state, repo_root))
    else:
        docker_validation = dict(state.temp_data.get("docker_validation", {}) or {})
        working_root = str(docker_validation.get("working_root", "") or "").strip()
        if working_root:
            search_roots.append(Path(working_root))
        smoke_payload = dict(docker_validation.get("smoke_payload", {}) or {})
        for variant_payload in list(smoke_payload.get("variants", []) or []):
            if not isinstance(variant_payload, dict):
                continue
            workspace = str(variant_payload.get("workspace", "") or "").strip()
            if workspace:
                search_roots.append(Path(workspace))
            smoke = dict(variant_payload.get("smoke", {}) or {})
            for root in list(smoke.get("artifact_search_roots", []) or []):
                rendered = str(root or "").strip()
                if rendered:
                    search_roots.append(Path(rendered))
    def _alias_keys(value: str) -> set[str]:
        lowered = normalized_repo_path(value).lower()
        keys = {lowered}
        basename = Path(lowered).name
        if basename:
            keys.add(basename)
            keys.add(basename.rsplit(".", 1)[0])
        keys.update(token for token in re.findall(r"[a-z0-9_]+", lowered) if len(token) > 2)
        return {key for key in keys if key}

    def _paths_from_manifest_value(value: Any) -> list[str]:
        paths: list[str] = []
        if isinstance(value, str):
            if _looks_like_artifact_path(value):
                paths.append(normalized_repo_path(value))
        elif isinstance(value, dict):
            for item in value.values():
                paths.extend(_paths_from_manifest_value(item))
        elif isinstance(value, list):
            for item in value:
                paths.extend(_paths_from_manifest_value(item))
        return paths

    def _manifest_alias_paths(root: Path, keys: set[str]) -> list[str]:
        manifests = [
            root / "results" / "artifact_manifest.json",
            root / "artifact_manifest.json",
            root / "results" / "evaluation_result.json",
            root / "evaluation_result.json",
        ]
        discovered: list[str] = []

        def visit(value: Any) -> None:
            if isinstance(value, dict):
                for key, item in value.items():
                    if str(key).strip().lower() in keys:
                        discovered.extend(_paths_from_manifest_value(item))
                    visit(item)
            elif isinstance(value, list):
                for item in value:
                    visit(item)

        for manifest in manifests:
            payload = _load_json_if_possible(manifest)
            if payload is None:
                continue
            visit(payload)
        return list(dict.fromkeys(discovered))

    aliases = _alias_keys(normalized)
    seen: set[str] = set()
    for root in search_roots:
        try:
            root_key = str(root.resolve())
        except OSError:
            root_key = str(root)
        if root_key in seen:
            continue
        seen.add(root_key)
        candidate = root / normalized
        try:
            if candidate.is_dir():
                return any(path.is_file() for path in candidate.rglob("*"))
            if candidate.is_file() and candidate.stat().st_size > 0:
                return not _artifact_is_validation_only(candidate)
        except OSError:
            continue
        for alias_path in _manifest_alias_paths(root, aliases):
            candidate = root / alias_path
            try:
                if candidate.is_dir() and any(path.is_file() for path in candidate.rglob("*")):
                    return True
                if candidate.is_file() and candidate.stat().st_size > 0:
                    return not _artifact_is_validation_only(candidate)
            except OSError:
                continue
    return False


def global_contract_wiring_checks(
    state: PaperBenchReproState,
    *,
    repo_root: Path | None = None,
) -> list[ValidationCheck]:
    """Validate owner/producer/artifact wiring between global contract and repo plan."""
    if state.global_contract is None or state.repo_plan is None:
        return []

    repo_work_package_ids = {item.work_package_id for item in state.repo_plan.work_packages}
    file_paths = {item.target_file for item in state.repo_plan.files}
    stage_surface_paths = {item.path for item in state.repo_plan.stage_public_surfaces if item.path}
    artifact_contract_by_key = {
        normalized_repo_key(item.relative_path): item
        for item in state.repo_plan.artifact_contract
        if normalized_repo_key(item.relative_path)
    }
    file_plans_by_work_package: dict[str, list[RepoFilePlan]] = {}
    for file_plan in state.repo_plan.files:
        if file_plan.work_package_id:
            file_plans_by_work_package.setdefault(file_plan.work_package_id, []).append(file_plan)

    missing_owner_packages: list[str] = []
    missing_producer_surfaces: list[str] = []
    missing_artifact_wiring: list[str] = []
    affected_work_packages: set[str] = set()

    for target in state.global_contract.result_targets:
        target_id = str(target.target_id).strip() or str(target.name).strip() or "unknown_target"
        owner_work_packages = [item for item in target.owner_work_packages if item]
        missing_owners = [item for item in owner_work_packages if item not in repo_work_package_ids]
        if missing_owners:
            missing_owner_packages.append(f"{target_id}:{','.join(missing_owners)}")

        owner_file_plans = [
            file_plan
            for work_package_id in owner_work_packages
            for file_plan in file_plans_by_work_package.get(work_package_id, [])
        ]
        if owner_work_packages and not owner_file_plans:
            missing_producer_surfaces.append(f"{target_id}:{','.join(owner_work_packages)}")
            affected_work_packages.update(owner_work_packages)
            continue

        artifact_keys = normalized_repo_keys(list(target.artifact_paths))
        for artifact_key in artifact_keys:
            contract_entry = artifact_contract_by_key.get(artifact_key)
            if contract_entry is None:
                continue
            producer_surface = normalized_repo_path(contract_entry.producer_surface)
            if producer_surface and producer_surface not in file_paths and producer_surface not in stage_surface_paths:
                missing_producer_surfaces.append(f"{target_id}:{producer_surface}")
                affected_work_packages.update(owner_work_packages)
        if artifact_keys:
            wired = any(
                artifact_keys.intersection(file_plan_artifact_keys(file_plan))
                for file_plan in owner_file_plans
            )
            if not wired:
                wired = any(
                    _global_contract_artifact_satisfied_by_execution(state, path, repo_root)
                    for path in target.artifact_paths
                )
            if not wired:
                missing_artifact_wiring.append(
                    f"{target_id}:{','.join(sorted(normalized_repo_path(path) for path in target.artifact_paths if path))}"
                )
                affected_work_packages.update(owner_work_packages)

    return [
        _as_advisory_planning_check(
            ValidationCheck(
                name="integration:global_contract:owner_packages",
                category="integration",
                passed=not missing_owner_packages,
                details=(
                    "All global-contract result targets reference known owner work packages."
                    if not missing_owner_packages
                    else f"Unknown owner work packages in global contract: {missing_owner_packages}"
                ),
                affected_work_packages=sorted(affected_work_packages),
            )
        ),
        _as_advisory_planning_check(
            ValidationCheck(
                name="integration:global_contract:producer_surfaces",
                category="integration",
                passed=not missing_producer_surfaces,
                details=(
                    "All global-contract result targets map to at least one repo-plan producer surface."
                    if not missing_producer_surfaces
                    else f"Missing repo-plan producer surfaces for global-contract targets: {missing_producer_surfaces}"
                ),
                affected_work_packages=sorted(affected_work_packages),
            )
        ),
        _as_advisory_planning_check(
            ValidationCheck(
                name="integration:global_contract:artifact_wiring",
                category="integration",
                passed=not missing_artifact_wiring,
                details=(
                    "All global-contract artifact paths are wired to repo-plan file outputs or verified execution artifacts."
                    if not missing_artifact_wiring
                    else f"Missing repo-plan artifact wiring for global-contract targets: {missing_artifact_wiring}"
                ),
                affected_work_packages=sorted(affected_work_packages),
            )
        ),
    ]


def repair_recommendations_from_checks(checks: list[ValidationCheck]) -> list[str]:
    """Derive repair recommendations from structured failed checks."""
    recommendations: list[str] = []
    for check in checks:
        if check.passed:
            continue
        if check.category == "artifact":
            recommendations.append("补齐缺失 artifact path，并保证 producer surface 与结果产物路径一致。")
            continue
        if check.category == "implementation":
            recommendations.append("收紧主入口和本地文件实现闭环，先保证关键执行面存在且可被局部验证。")
            continue
        if check.category == "trace":
            recommendations.append("补强 work-package 的 traceability 绑定，必要时刷新 evidence 和 contract 对应关系。")
            continue
        if check.name == "post_generate:paper_evidence_contract_matrix":
            recommendations.append("补齐 paper-derived evidence matrix：命名实验、环境/任务、方法/基线、参数 sweep、趋势断言和 artifact writer 必须在代码或配置中可见。")
            continue
        if check.name == "post_generate:paper_implementation_obligation_paths":
            recommendations.append("补齐 paper-derived implementation paths：dataset prepare/validate、model loader、metric formula、attack/adaptation、training/evaluation loop 和 per-sample bookkeeping 必须落到可执行代码或配置入口。")
            continue
        if check.name == "post_generate:formula_algorithm_contract":
            recommendations.append("补齐 paper-derived formula/algorithm anchors：论文公式符号、数值常量、mask/rank、loss、search/schedule 步骤必须落到可执行代码或配置，并被主训练/评估路径调用。")
            continue
        if check.name == "post_generate:active_route_wiring":
            recommendations.append("补齐 active-route wiring：已经实现的 helper/class 必须接到 entrypoint、factory/reset、training/evaluation loop 或 figure/table writer，不能只停留在孤立符号。")
            continue
        if check.name == "post_generate:declared_experiment_route_contract":
            recommendations.append("补齐 declared experiment route：显式 dataset/search_times/figure/table 合约必须接到真实执行或报告路径，不能用错数据集、固定 ablation 或 runtime_smoke/synthetic fallback 代替。")
            continue
        if check.category == "semantic":
            recommendations.append("修复语义证据缺口，优先补代码/配置/报告面的可见实验矩阵，而不是只补 README 描述。")
            continue
        if "canonical_route" in check.name:
            recommendations.append("优先修复 canonical route 的 entry surface、输入依赖和 expected outputs wiring。")
            continue
        if "artifact_contract" in check.name or "artifact_wiring" in check.name:
            recommendations.append("修复 artifact contract、producer surface 和 file plan outputs 之间的静态连线。")
            continue
        if "stage_public_surfaces" in check.name or "producer_surfaces" in check.name:
            recommendations.append("补齐 stage public surfaces 与 repo-plan files 的映射，再扩展可选模块。")
            continue
        if check.category == "integration":
            recommendations.append("先修复协议阶段、结果目标和公共 surface 的集成闭环，再做包级扩写。")
    ordered: list[str] = []
    seen: set[str] = set()
    for item in recommendations:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def traceability_checks(state: PaperBenchReproState) -> list[ValidationCheck]:
    """Validate package-scoped evidence grounding and trace coverage."""
    work_packages = (
        state.repo_plan.work_packages
        if state.repo_plan is not None
        else state.work_package_planning.work_packages if state.work_package_planning else []
    )
    bundle_by_package = {item.work_package_id: item for item in state.evidence_bundles}
    checks: list[ValidationCheck] = []
    for work_package in work_packages:
        bundle = bundle_by_package.get(work_package.work_package_id)
        needs_grounding = bool(work_package.reference_ids)
        grounded = bundle is not None and (
            bundle.grounding_status == "grounded"
            or (not needs_grounding and bundle.grounding_status in {"grounded", "ungrounded"})
        )
        checks.append(
            _as_advisory_planning_check(
                ValidationCheck(
                    name=f"trace:work_package:{work_package.work_package_id}",
                    category="trace",
                    passed=grounded,
                    details=(
                        f"evidence bundle available for {work_package.work_package_id}"
                        if grounded
                        else f"missing grounded evidence bundle for {work_package.work_package_id}"
                    ),
                    affected_work_packages=[work_package.work_package_id],
                )
            )
        )
    return checks


def trace_failed_work_package_ids(report: ValidationReport) -> list[str]:
    """Return work-package ids implicated by failed trace checks."""
    failed: set[str] = set()
    for check in report.trace_checks:
        if check.passed:
            continue
        failed.update(check.affected_work_packages)
        name = str(check.name or "").strip()
        if name.startswith("trace:work_package:"):
            work_package_id = name.split("trace:work_package:", 1)[1].strip()
            if work_package_id:
                failed.add(work_package_id)
    return sorted(failed)


def refresh_work_package_evidence(
    state: PaperBenchReproState,
    work_package_ids: list[str],
    *,
    build_evidence_bundles: Callable[[PaperBenchReproState, WorkPackagePlanningOutput], tuple[list[Any], list[Any]]],
    write_stage_output: Callable[[PaperBenchReproState, str, object], None],
) -> list[str]:
    """Refresh evidence bundles for selected work packages and merge them back into state."""
    if not work_package_ids or state.work_package_planning is None:
        return []
    target_ids = set(work_package_ids)
    target_packages = [
        item for item in state.work_package_planning.work_packages if item.work_package_id in target_ids
    ]
    if not target_packages:
        return []
    refreshed_bundles, _ = build_evidence_bundles(
        state,
        WorkPackagePlanningOutput(work_packages=target_packages),
    )
    refreshed_map = {item.work_package_id: item for item in refreshed_bundles}
    merged_bundles = [refreshed_map.get(item.work_package_id, item) for item in state.evidence_bundles]
    existing_ids = {item.work_package_id for item in merged_bundles}
    for item in refreshed_bundles:
        if item.work_package_id not in existing_ids:
            merged_bundles.append(item)
    state.evidence_bundles = merged_bundles
    state.evidence_graph = [link for bundle in merged_bundles for link in bundle.evidence_links]
    if state.repo_plan is not None:
        state.repo_plan = state.repo_plan.model_copy(update={"evidence_bundles": merged_bundles})
        write_stage_output(state, "repo_plan.json", state.repo_plan)
    write_stage_output(state, "evidence_bundles.json", state.evidence_bundles)
    write_stage_output(state, "evidence_graph.json", state.evidence_graph)
    return [item.work_package_id for item in refreshed_bundles]


def evaluate_validation_bundle(
    state: PaperBenchReproState,
    *,
    build_runtime_probe: Callable[[], RuntimeProbe],
    get_output_dir: Callable[[PaperBenchReproState], Path],
    entrypoint_related_work_packages: Callable[[PaperBenchReproState], list[str]],
) -> tuple[RuntimeProbe, ValidationReport, BenchmarkReport]:
    """Compute validation artifacts from the current generated project state."""
    runtime_probe = state.runtime_probe or build_runtime_probe()
    repo_root = Path(state.project_root) if state.project_root else (get_output_dir(state) / "repo")
    project_files = _read_project_source_snapshot(repo_root)
    main_entry = _canonical_entry_surface(state)
    main_path = repo_root / main_entry
    if state.repo_plan is not None:
        required_artifacts = _repo_required_artifact_paths(state)
    else:
        required_artifacts = _dedupe_artifact_paths(list(state.project_plan.artifact_contract.required_files))
    target_owner_map = repo_plan_artifact_owner_map(state)
    for target in (state.global_contract.result_targets if state.global_contract else []):
        for path in target.artifact_paths:
            owners = target_owner_map.setdefault(normalized_repo_key(path), [])
            for owner in target.owner_work_packages:
                if owner and owner not in owners:
                    owners.append(owner)
    file_owner_map: dict[str, list[str]] = {}
    if state.repo_plan is not None:
        for file_plan in state.repo_plan.files:
            if file_plan.work_package_id:
                owners = file_owner_map.setdefault(file_plan.target_file, [])
                if file_plan.work_package_id not in owners:
                    owners.append(file_plan.work_package_id)
            for path in file_plan.writes_artifacts:
                if not path:
                    continue
                owners = target_owner_map.setdefault(normalized_repo_key(path), [])
                if file_plan.work_package_id and file_plan.work_package_id not in owners:
                    owners.append(file_plan.work_package_id)

    artifact_checks = []
    for path in required_artifacts:
        artifact_path = _find_validation_artifact(state, repo_root, path)
        validation_only = bool(artifact_path is not None and _artifact_is_validation_only(artifact_path))
        artifact_checks.append(
            ValidationCheck(
                name=f"artifact:{path}",
                category="artifact",
                passed=artifact_path is not None and not validation_only,
                details=(
                    f"artifact exists: {artifact_path}"
                    if artifact_path is not None
                    and not validation_only
                    else (
                        f"artifact is validation-only/schema-only placeholder: {artifact_path}"
                        if artifact_path is not None
                        else "artifact missing"
                    )
                ),
                affected_files=[path],
                affected_work_packages=target_owner_map.get(normalized_repo_key(path), []),
            )
        )
    implementation_checks = [
        ValidationCheck(
            name="main_entry_exists",
            category="implementation",
            passed=main_path.exists(),
            details=f"entrypoint {'exists' if main_path.exists() else 'missing'}: {main_entry}",
            affected_files=[main_entry],
            affected_work_packages=file_owner_map.get(main_entry, []),
        )
    ]
    semantic_checks = semantic_assertion_checks(state, project_files)
    semantic_checks.extend(_post_generate_semantic_surface_checks(state, repo_root, project_files))
    semantic_checks.extend(_post_generate_unit_symbol_coverage_checks(state, project_files))
    semantic_checks.extend(_post_generate_formula_algorithm_contract_checks(state, project_files))
    semantic_checks.extend(
        _post_generate_repo_route_closure_checks(
            state,
            project_files,
            entrypoint_related_work_packages=entrypoint_related_work_packages,
        )
    )
    semantic_checks.extend(_post_generate_active_route_wiring_checks(state, project_files))
    semantic_checks.extend(_post_generate_declared_experiment_route_checks(state, project_files))
    semantic_checks.extend(_post_generate_reference_grounding_checks(state, project_files))
    trace_checks = [
        _as_advisory_planning_check(
            ValidationCheck(
                name="global_contract_present",
                category="trace",
                passed=state.global_contract is not None,
                details=(
                    "global contract available for downstream validation"
                    if state.global_contract
                    else "global contract missing"
                ),
                affected_work_packages=[
                    item.work_package_id
                    for item in (state.work_package_planning.work_packages if state.work_package_planning else [])
                ],
            )
        )
    ]
    trace_checks.extend(traceability_checks(state))
    docker_validation = dict(state.temp_data.get("docker_validation", {}) or {})
    smoke_payload = dict(docker_validation.get("smoke_payload", {}) or {})
    docker_validation_payload = dict(docker_validation.get("validation_payload", {}) or {})
    for report in list(docker_validation_payload.get("variant_reports", []) or []):
        if not isinstance(report, dict):
            continue
        variant = str(report.get("variant", "") or "unknown")
        readiness_path = str(report.get("readiness_path", "") or "")
        evaluation_path = str(report.get("evaluation_result_path", "") or "")
        artifact_checks.extend(
            [
                ValidationCheck(
                    name=f"artifact:docker:{variant}:readiness",
                    category="artifact",
                    passed=bool(readiness_path),
                    details="docker readiness artifact captured" if readiness_path else "docker readiness artifact missing",
                    affected_files=[readiness_path] if readiness_path else [],
                    affected_work_packages=entrypoint_related_work_packages(state),
                ),
                ValidationCheck(
                    name=f"artifact:docker:{variant}:evaluation_result",
                    category="artifact",
                    passed=bool(evaluation_path),
                    details=(
                        "docker evaluation result artifact captured"
                        if evaluation_path
                        else "docker evaluation result artifact missing"
                    ),
                    affected_files=[evaluation_path] if evaluation_path else [],
                    affected_work_packages=entrypoint_related_work_packages(state),
                ),
            ]
        )
    integration_checks = [
        ValidationCheck(
            name="generated_files_nonempty",
            category="integration",
            passed=bool(state.generated_files),
            details=f"generated_files={len(state.generated_files)}",
            affected_files=list(state.generated_files),
        )
    ]
    if state.repo_plan is not None:
        planned_file_keys = normalized_repo_keys(
            [
                item.target_file
                for item in list(state.repo_plan.files or [])
                if str(item.target_file or "").strip()
            ]
        )
        generated_file_keys = normalized_repo_keys(list(state.generated_files or []))
        missing_planned_files = sorted(planned_file_keys - generated_file_keys)
        integration_checks.append(
            ValidationCheck(
                name="integration:generated_file_plan_closure",
                category="integration",
                passed=not missing_planned_files,
                details=(
                    "generated_files closes over repo-plan target files."
                    if not missing_planned_files
                    else f"repo-plan files missing from generated_files: {missing_planned_files[:12]}"
                ),
                affected_files=list(missing_planned_files[:40]),
                affected_work_packages=entrypoint_related_work_packages(state),
            )
        )
    benchmark_visible_dry_run_artifacts: list[str] = []
    repo_root = Path(str(state.project_root or ""))
    if repo_root.exists():
        visible_name_tokens = {
            "metric",
            "metrics",
            "result",
            "results",
            "table",
            "figure",
            "trend",
            "cost",
            "prediction",
            "predictions",
            "report",
        }
        for path in sorted(repo_root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".json", ".jsonl", ".csv", ".tsv", ".md", ".txt", ".yaml", ".yml"}:
                continue
            rel_path = path.relative_to(repo_root).as_posix()
            if rel_path in {"results/artifact_contract_manifest.json"}:
                continue
            name_tokens = set(re.split(r"[^a-zA-Z0-9]+", path.stem.lower()))
            if not name_tokens.intersection(visible_name_tokens):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")[:20000].lower()
            except OSError:
                continue
            if any(marker in text for marker in _VALIDATION_ONLY_ARTIFACT_MARKERS):
                benchmark_visible_dry_run_artifacts.append(rel_path)
    integration_checks.append(
        ValidationCheck(
            name="integration:benchmark_visible_artifacts_not_dry_run_only",
            category="integration",
            passed=not benchmark_visible_dry_run_artifacts,
            details=(
                "benchmark-visible metrics/results/table/cost artifacts are not dry-run-only contract shells."
                if not benchmark_visible_dry_run_artifacts
                else "benchmark-visible artifacts are dry-run-only shells: "
                + ", ".join(benchmark_visible_dry_run_artifacts[:10])
            ),
            affected_files=list(benchmark_visible_dry_run_artifacts[:40]),
            affected_work_packages=entrypoint_related_work_packages(state),
        )
    )
    docker_failure_details = []
    for item in list(smoke_payload.get("variants", []) or []):
        if not isinstance(item, dict):
            continue
        smoke = dict(item.get("smoke", {}) or {})
        if smoke.get("success"):
            continue
        diagnostics = _first_nonempty([
            str(smoke.get("error", "") or ""),
            str(smoke.get("diagnostics", "") or ""),
            str(smoke.get("stderr", "") or ""),
        ])
        rendered = f"{item.get('variant', 'unknown')}: {diagnostics[:500]}" if diagnostics else f"{item.get('variant', 'unknown')}: docker command failed"
        docker_failure_details.append(rendered)
    validation_failure_details = [
        str(item).strip()
        for item in docker_validation_payload.get("failures", []) or []
        if str(item).strip()
    ]
    if smoke_payload or docker_validation_payload:
        integration_checks.extend(
            [
                ValidationCheck(
                    name="integration:docker_runtime_execution",
                    category="integration",
                    passed=str(smoke_payload.get("status", "") or "") == "passed",
                    details=(
                        "docker runtime validation command passed for the canonical validated-repo workspace."
                        if str(smoke_payload.get("status", "") or "") == "passed"
                        else "; ".join(docker_failure_details[:6]) or "docker runtime validation command failed"
                    ),
                    affected_files=[main_entry],
                    affected_work_packages=entrypoint_related_work_packages(state),
                ),
                ValidationCheck(
                    name="integration:docker_runtime_artifact_contract",
                    category="integration",
                    passed=str(docker_validation_payload.get("status", "") or "") == "passed",
                    details=(
                        "docker runtime artifacts satisfied readiness and evaluation-result contracts for the validated repo."
                        if str(docker_validation_payload.get("status", "") or "") == "passed"
                        else "; ".join(validation_failure_details[:6]) or "docker runtime artifact contract failed"
                    ),
                    affected_files=[main_entry],
                    affected_work_packages=entrypoint_related_work_packages(state),
                ),
            ]
        )
    integration_checks.extend(
        repo_plan_static_contract_checks(
            state,
            entrypoint_related_work_packages=entrypoint_related_work_packages,
        )
    )
    integration_checks.extend(global_contract_wiring_checks(state, repo_root=repo_root))
    integration_checks.extend(_planning_review_failure_checks(state))
    blocking_artifact_checks = _blocking_validation_checks(artifact_checks)
    blocking_implementation_checks = _blocking_validation_checks(implementation_checks)
    blocking_semantic_checks = _blocking_validation_checks(semantic_checks)
    blocking_trace_checks = _blocking_validation_checks(trace_checks)
    blocking_integration_checks = _blocking_validation_checks(integration_checks)

    static_status = state.preflight_result.status if state.preflight_result else "unknown"
    static_contract_status = "passed" if static_status == "passed" and main_path.exists() else static_status
    smoke_status = (
        "success"
        if str(smoke_payload.get("status", "") or "") == "passed"
        else "failed" if smoke_payload else "skipped"
    )
    dynamic_status = (
        "success"
        if str(docker_validation_payload.get("status", "") or "") == "passed"
        else "failed" if docker_validation_payload else "skipped"
    )
    quality_level = state.generate_stage_output.quality_level if state.generate_stage_output else "scaffold_only"
    failure_categories: list[str] = []
    blocked_reasons: list[str] = []
    if static_contract_status not in {"passed", "warnings"}:
        failure_categories.append("static_contract")
        blocked_reasons.append("static contract gate did not pass")
    if smoke_status not in {"success", "fixed"}:
        failure_categories.append("smoke")
        blocked_reasons.append("smoke gate not passed")
    if any(not item.passed for item in blocking_artifact_checks):
        failure_categories.append("artifact")
        blocked_reasons.append("required artifact contract not satisfied")
    if any(not item.passed for item in blocking_implementation_checks):
        failure_categories.append("implementation")
        blocked_reasons.append("implementation surface incomplete")
    if any(not item.passed for item in blocking_semantic_checks):
        failure_categories.append("semantic")
        blocked_reasons.append("semantic assertion contract not satisfied")
    if any(not item.passed for item in blocking_trace_checks):
        failure_categories.append("trace")
        blocked_reasons.append("traceability contract not satisfied")
    if any(not item.passed for item in blocking_integration_checks):
        failure_categories.append("integration")
        blocked_reasons.append("integration contract not satisfied")

    passed = not failure_categories
    overall_status = "passed" if passed else "partial" if static_contract_status in {"passed", "warnings"} else "failed"
    failed_checks = [
        item
        for item in [
            *blocking_artifact_checks,
            *blocking_implementation_checks,
            *blocking_trace_checks,
            *blocking_integration_checks,
        ]
        if not item.passed
    ]
    failed_checks.extend(item for item in blocking_semantic_checks if not item.passed)
    deterministic_repair_ticket = _fallback_validation_repair_ticket(state, failed_checks)
    repair_ticket = (
        deterministic_repair_ticket
        if deterministic_repair_ticket is not None and any(
            str(check.name or "")
            in {
                "post_generate:paper_evidence_contract_matrix",
                "post_generate:paper_implementation_obligation_paths",
                "post_generate:formula_algorithm_contract",
                "post_generate:repo_route_closure",
                "post_generate:active_route_wiring",
                "post_generate:declared_experiment_route_contract",
                "post_generate:reference_grounding",
            }
            for check in failed_checks
        )
        else (
            state.repair_ticket
            if state.repair_ticket is not None and not smoke_payload and not docker_validation_payload
            else _build_docker_validation_repair_ticket(
                state,
                smoke_payload,
                docker_validation_payload,
                failed_checks=failed_checks,
            )
        )
    )
    repair_ticket = _augment_repair_ticket_with_planning_issues(state, repair_ticket)
    state.repair_ticket = repair_ticket
    state.temp_data["repair_ticket"] = repair_ticket.model_dump(mode="json") if repair_ticket is not None else {}
    validation_report = ValidationReport(
        passed=passed,
        static_status=static_status,
        static_contract_status=static_contract_status,
        smoke_status=smoke_status,
        dynamic_status=dynamic_status,
        overall_status=overall_status,
        quality_level=quality_level,
        runtime_probe=runtime_probe,
        artifact_checks=artifact_checks,
        implementation_checks=implementation_checks,
        semantic_checks=semantic_checks,
        trace_checks=trace_checks,
        integration_checks=integration_checks,
        failure_categories=list(dict.fromkeys(failure_categories)),
        blocked_reasons=list(dict.fromkeys(blocked_reasons)),
        repair_recommendations=(
            _repair_recommendations_from_ticket(repair_ticket)
            or repair_recommendations_from_checks(failed_checks)
        ) if failure_categories else [],
        planning_failure_layer=_current_planning_failure_layer(state),
        semantic_validation_report=(
            dict(state.canonical_ir_validation.semantic_validation_report)
            if state.canonical_ir_validation is not None
            else {}
        ),
    )
    matched_artifacts = sum(1 for item in artifact_checks if item.passed)
    benchmark_report = BenchmarkReport(
        benchmark_name="reproagent_contract_validation",
        case_id=state.run_id,
        task_id=main_entry,
        expected_artifacts=required_artifacts,
        artifact_match_ratio=(matched_artifacts / len(required_artifacts)) if required_artifacts else 1.0,
        rubric_requirements=list(state.boundary_requirements.requirement_scope_items) if state.boundary_requirements else [],
        rubric_match_ratio=1.0 if passed else 0.0,
        notes=["derived from reproagent integration validation gates"],
    )
    return runtime_probe, validation_report, benchmark_report
