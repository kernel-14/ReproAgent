"""Open-source bundle export helpers for PaperBench reproduction runs."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


EXCLUDED_DIR_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "node",
    "score",
    "final_repo",
    "results",
    "simulated_data",
    "artifacts",
    "outputs",
}

EXCLUDED_FILE_SUFFIXES = {".pyc", ".pyo"}


@dataclass(frozen=True)
class BundleExportResult:
    """Paths and counts from one bundle export."""

    destination: Path
    source_run_dir: Path
    source_repo_dir: Path
    source_score_dir: Path | None
    node_dirs: tuple[str, ...]
    final_repo_files: int
    score_files: int


def _ignore_names(_src: str, names: list[str]) -> set[str]:
    ignored: set[str] = set()
    for name in names:
        if name in EXCLUDED_DIR_NAMES:
            ignored.add(name)
            continue
        if Path(name).suffix in EXCLUDED_FILE_SUFFIXES:
            ignored.add(name)
    return ignored


def _copytree_filtered(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"source path does not exist: {src}")
    shutil.copytree(src, dst, ignore=_ignore_names, symlinks=True)


def _count_files(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.rglob("*") if item.is_file())


def _write_no_artifacts_marker(path: Path, node_name: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    marker = {
        "node": node_name,
        "status": "not_run",
        "artifacts": [],
        "note": f"No {node_name} node artifacts were produced for this run.",
    }
    (path / "manifest.json").write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")


def _read_json_file(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _bundle_readiness_issues(run_dir: Path, source_repo_dir: Path) -> list[str]:
    issues: list[str] = []
    quality_path = run_dir / "quality_status.json"
    if not quality_path.exists():
        issues.append("quality_status.json is missing")
    else:
        try:
            quality = _read_json_file(quality_path)
        except Exception as exc:  # pragma: no cover - defensive file corruption path
            quality = {}
            issues.append(f"quality_status.json is unreadable: {exc}")
        if isinstance(quality, dict):
            if not bool(quality.get("handoff_ready", False)):
                issues.append("quality_status.handoff_ready is false")
            if not bool(quality.get("validation_passed", False)):
                issues.append("quality_status.validation_passed is false")
            quality_level = str(quality.get("validation_quality_level", "") or "").strip().lower()
            if quality_level in {"", "scaffold_only", "unverified"}:
                issues.append(f"validation quality level is not ready: {quality_level or 'missing'}")
            terminal = str(quality.get("terminal_outcome", "") or "").strip().lower()
            if terminal and terminal != "completed":
                issues.append(f"terminal outcome is not completed: {terminal}")
        elif quality_path.exists():
            issues.append("quality_status.json is not a JSON object")

    generate_summary_path = run_dir / "nodes" / "generate" / "summary.json"
    if generate_summary_path.exists():
        try:
            generate_summary = _read_json_file(generate_summary_path)
        except Exception as exc:  # pragma: no cover - defensive file corruption path
            generate_summary = {}
            issues.append(f"generate summary is unreadable: {exc}")
        if isinstance(generate_summary, dict):
            status = str(generate_summary.get("status", "") or "").strip().lower()
            if status in {"completed_unverified", "failed", "completed_with_runtime_risk", "completed_with_degraded_contract"}:
                issues.append(f"generate summary status is not ready: {status}")

    repo_files = [item for item in source_repo_dir.rglob("*") if item.is_file()] if source_repo_dir.exists() else []
    source_like_files = [
        item
        for item in repo_files
        if item.suffix.lower() in {".py", ".yaml", ".yml", ".toml", ".md", ".txt"}
        and "results" not in item.relative_to(source_repo_dir).parts
    ]
    if len(source_like_files) < 5:
        issues.append(f"final repo has too few source/config files: {len(source_like_files)}")
    return issues


def _score_readiness_issues(score_dir: Path | None) -> list[str]:
    """Return score-bundle issues that should block ready exports."""
    if score_dir is None:
        return ["score_dir is required for ready open-source export"]
    summary_path = score_dir / "summary.json"
    if not summary_path.exists():
        manifest_path = score_dir / "manifest.json"
        if manifest_path.exists():
            try:
                manifest = _read_json_file(manifest_path)
            except Exception as exc:  # pragma: no cover - defensive file corruption path
                return [f"score summary.json is missing and score manifest is unreadable: {exc}"]
            if isinstance(manifest, dict) and str(manifest.get("status", "")).strip().lower() == "not_run":
                return ["score summary.json is missing and score manifest says not_run"]
        return ["score summary.json is missing"]
    try:
        summary = _read_json_file(summary_path)
    except Exception as exc:  # pragma: no cover - defensive file corruption path
        return [f"score summary.json is unreadable: {exc}"]
    if not isinstance(summary, dict):
        return ["score summary.json is not a JSON object"]
    issues: list[str] = []
    score = summary.get("score")
    if not isinstance(score, int | float) or not 0 <= float(score) <= 1:
        issues.append("score summary has missing or invalid score")
    leaf_count = summary.get("num_leaf_nodes")
    invalid_count = summary.get("num_invalid_leaf_nodes", 0)
    if not isinstance(leaf_count, int) or leaf_count <= 0:
        issues.append("score summary has missing or invalid num_leaf_nodes")
    if not isinstance(invalid_count, int) or invalid_count < 0:
        issues.append("score summary has missing or invalid num_invalid_leaf_nodes")
    elif isinstance(leaf_count, int) and leaf_count > 0 and invalid_count / leaf_count > 0.10:
        issues.append(
            f"score summary has too many invalid leaves: {invalid_count}/{leaf_count}"
        )
    return issues


def export_open_source_bundle(
    *,
    run_dir: Path,
    source_repo_dir: Path,
    destination: Path,
    score_dir: Path | None = None,
    node_names: Iterable[str] = ("prepare", "plan", "generate", "repair"),
    replace: bool = False,
    allow_unverified: bool = False,
) -> BundleExportResult:
    """Export a run into the canonical open-source review bundle layout.

    The destination layout is intentionally narrow:
    - node/<stage>/ contains stage intermediate artifacts.
    - final_repo/ contains the final repository snapshot only.
    - score/ contains the final score output only, when provided.
    """

    run_dir = run_dir.resolve()
    source_repo_dir = source_repo_dir.resolve()
    destination = destination.resolve()
    score_dir = score_dir.resolve() if score_dir is not None else None

    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir does not exist: {run_dir}")
    if not source_repo_dir.exists():
        raise FileNotFoundError(f"source_repo_dir does not exist: {source_repo_dir}")
    if score_dir is not None and not score_dir.exists():
        raise FileNotFoundError(f"score_dir does not exist: {score_dir}")
    readiness_issues = _bundle_readiness_issues(run_dir, source_repo_dir)
    readiness_issues.extend(_score_readiness_issues(score_dir))
    if readiness_issues and not allow_unverified:
        raise RuntimeError(
            "refusing to export unready open-source bundle: "
            + "; ".join(readiness_issues[:12])
        )
    if destination.exists():
        if not replace:
            raise FileExistsError(f"destination already exists: {destination}")
        shutil.rmtree(destination)

    destination.mkdir(parents=True)
    nodes_root = run_dir / "nodes"
    exported_nodes: list[str] = []
    for node_name in node_names:
        source_node = nodes_root / node_name
        target_node = destination / "node" / node_name
        if source_node.exists():
            _copytree_filtered(source_node, target_node)
        else:
            _write_no_artifacts_marker(target_node, node_name)
        exported_nodes.append(node_name)

    _copytree_filtered(source_repo_dir, destination / "final_repo")

    score_file_count = 0
    if score_dir is not None:
        _copytree_filtered(score_dir, destination / "score")
        score_file_count = _count_files(destination / "score")
    else:
        _write_no_artifacts_marker(destination / "score", "score")

    return BundleExportResult(
        destination=destination,
        source_run_dir=run_dir,
        source_repo_dir=source_repo_dir,
        source_score_dir=score_dir,
        node_dirs=tuple(exported_nodes),
        final_repo_files=_count_files(destination / "final_repo"),
        score_files=score_file_count,
    )
