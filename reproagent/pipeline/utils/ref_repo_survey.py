"""Reference-repository survey helpers for reproagent planning stages."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

from reproagent.pipeline.schemas import (
    BoundaryRequirementsOutput,
    ExtractedUnit,
    PaperBenchReproState,
    PreparedReferenceRepositorySurvey,
    ReferenceRequirementCoverage,
    ReferenceSymbolEvidence,
)


_TEXT_EXTENSIONS = {
    ".py",
    ".md",
    ".rst",
    ".txt",
    ".yaml",
    ".yml",
    ".json",
    ".toml",
    ".ini",
    ".cfg",
    ".sh",
    ".bash",
    ".zsh",
    ".ipynb",
}
_SOURCE_EXTENSIONS = {
    ".py",
    ".ipynb",
    ".sh",
    ".bash",
    ".zsh",
    ".yaml",
    ".yml",
    ".json",
    ".toml",
}
_PROTOCOL_SNIPPET_EXTENSIONS = {
    ".md",
    ".rst",
    ".txt",
    ".yaml",
    ".yml",
    ".json",
    ".toml",
    ".ini",
    ".cfg",
    ".sh",
    ".bash",
    ".zsh",
    ".ipynb",
}
_IGNORE_DIRS = {
    ".git",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "build",
    "dist",
    "node_modules",
    ".idea",
    ".vscode",
}
_PROTOCOL_PATTERN = re.compile(
    r"\b(train|training|evaluate|evaluation|dataset|preprocess|seed|config|checkpoint|metric|artifact|"
    r"epoch|batch|optimizer|adamw|weight_decay|learning_rate|lr|epsilon|attack|pgd|apgd|autoattack|"
    r"iterations?|precision|fp16|fp32|sampling|manifest|command|python|bash|wandb|save|load)\b",
    re.I,
)
_TOKEN_RE = re.compile(r"[A-Za-z0-9_./\-]{2,}")
_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "into",
    "that",
    "this",
    "using",
    "use",
    "based",
    "experiment",
    "method",
    "model",
    "task",
    "data",
    "dataset",
    "metric",
    "train",
    "training",
    "eval",
    "evaluation",
    "implementation",
    "paper",
    "code",
    "agent",
    "agents",
    "state",
    "states",
    "step",
    "steps",
    "reward",
    "rewards",
    "policy",
    "policies",
    "action",
    "actions",
    "random",
    "current",
    "trajectory",
    "trajectories",
    "score",
    "scores",
    "environment",
    "environments",
    "env",
    "rl",
}
_GENERIC_SYMBOL_NAMES = {
    "__init__",
    "__module__",
    "main",
    "train",
    "test",
    "step",
    "reset",
    "forward",
}
_SURFACE_HINTS: dict[str, tuple[str, ...]] = {
    "entrypoint": ("main", "cli", "run", "launch", "entry"),
    "data_pipeline": ("data", "dataset", "loader", "preprocess", "buffer"),
    "model_or_method": ("model", "policy", "agent", "method", "algorithm", "objective", "loss"),
    "training_loop": ("train", "update", "learn", "optimiz", "epoch", "rollout"),
    "evaluation": ("eval", "evaluate", "metric", "score", "validate", "test"),
    "baseline_or_ablation": ("baseline", "ablation", "compare"),
    "artifact_writer": ("save", "write", "log", "result", "report", "checkpoint"),
    "config": ("config", "args", "argparse", "yaml", "json", "toml"),
    "environment": ("env", "environment", "simulator", "simulation", "task", "world"),
    "tests": ("test", "pytest", "assert"),
}


def _get_reference_repo_preparation(state: PaperBenchReproState) -> dict[str, Any]:
    """Return normalized reference repo preparation payload from start node."""
    payload = state.temp_data.get("reference_repo_preparation")
    if isinstance(payload, dict):
        return payload
    return {
        "clone_root": "",
        "requested_repositories": [],
        "prepared_repositories": [],
        "failed_repositories": [],
    }


def _get_reference_repo_surveys(
    state: PaperBenchReproState,
    boundary_output: BoundaryRequirementsOutput | None = None,
) -> list[PreparedReferenceRepositorySurvey]:
    """Return cached local repo surveys or build them on demand."""
    if state.reference_repo_surveys:
        return state.reference_repo_surveys

    cached_payload = state.temp_data.get("reference_repo_surveys")
    if isinstance(cached_payload, list):
        try:
            cached_surveys = [
                PreparedReferenceRepositorySurvey.model_validate(item)
                for item in cached_payload
                if isinstance(item, dict)
            ]
            preparation = _get_reference_repo_preparation(state)
            prepared = preparation.get("prepared_repositories")
            prepared_count = len(prepared) if isinstance(prepared, list) else 0
            if cached_surveys or prepared_count <= 0:
                state.reference_repo_surveys = cached_surveys
                return state.reference_repo_surveys
        except Exception:
            state.temp_data.pop("reference_repo_surveys", None)

    state.reference_repo_surveys = _build_reference_repo_surveys(state, boundary_output)
    state.temp_data["reference_repo_surveys"] = [
        item.model_dump(mode="json")
        for item in state.reference_repo_surveys
    ]
    return state.reference_repo_surveys


def _build_reference_repo_surveys(
    state: PaperBenchReproState,
    boundary_output: BoundaryRequirementsOutput | None = None,
) -> list[PreparedReferenceRepositorySurvey]:
    """Build evidence-oriented repo surveys from prepared local repositories."""
    preparation = _get_reference_repo_preparation(state)
    prepared = preparation.get("prepared_repositories")
    if not isinstance(prepared, list):
        return []

    units = list(state.unit_extraction.units if state.unit_extraction else [])
    surveys: list[PreparedReferenceRepositorySurvey] = []
    for item in prepared:
        if not isinstance(item, dict):
            continue
        local_repo_path = str(item.get("local_repo_path", "")).strip()
        if not local_repo_path:
            continue
        repo_path = Path(local_repo_path)
        ref_id = str(item.get("ref_id", "")).strip()
        symbol_evidence = _survey_symbol_evidence(
            repo_path,
            ref_id=ref_id,
            units=units,
            boundary_output=boundary_output,
        )
        requirement_coverage = _aggregate_requirement_coverage(
            boundary_output,
            symbol_evidence,
            units=units,
        )
        surveys.append(
            PreparedReferenceRepositorySurvey(
                ref_id=ref_id,
                title=str(item.get("title", "")).strip(),
                paper_path=str(item.get("paper_path", "")).strip(),
                paper_url=str(item.get("paper_url", "")).strip(),
                repository_url=str(item.get("repository_url", "")).strip(),
                repository_origin=str(item.get("repository_origin", "")).strip(),
                repository_type=str(item.get("repository_type", "")).strip(),
                reference_role=str(item.get("reference_role", "")).strip(),
                local_repo_path=local_repo_path,
                default_branch=str(item.get("default_branch", "")).strip(),
                status=str(item.get("status", "")).strip(),
                readme_summary=_readme_summary(repo_path),
                file_tree_summary=_file_tree_summary(repo_path),
                top_level_files=_top_level_files(repo_path),
                top_python_files=_top_python_files(repo_path),
                likely_reusable_files=_likely_reusable_files(repo_path, evidence=symbol_evidence),
                protocol_clues=_protocol_clues(repo_path),
                source_file_count=_count_source_files(repo_path),
                requirement_coverage=requirement_coverage,
                symbol_evidence=symbol_evidence,
            )
        )
    return surveys


def _readme_summary(repo_path: Path, max_lines: int = 24, max_chars: int = 1400) -> str:
    """Read a compact README summary from a local repository."""
    for name in ("README.md", "README.rst", "README.txt", "README"):
        candidate = repo_path / name
        if not candidate.exists() or not candidate.is_file():
            continue
        text = _safe_read_text(candidate)
        if not text:
            continue
        cleaned_lines: list[str] = []
        for raw_line in text.splitlines():
            line = re.sub(r"[#>*`]", " ", raw_line).strip()
            line = re.sub(r"\s+", " ", line)
            if not line:
                continue
            cleaned_lines.append(line)
            if len(cleaned_lines) >= max_lines:
                break
        summary = "\n".join(cleaned_lines)
        if len(summary) > max_chars:
            return summary[:max_chars].rstrip() + "..."
        return summary
    return ""


def _file_tree_summary(repo_path: Path, max_entries: int = 40) -> str:
    """Build a compact shallow file tree summary."""
    if not repo_path.exists():
        return ""
    entries: list[str] = []
    try:
        top_level = sorted(repo_path.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))
    except OSError:
        return ""

    for item in top_level:
        if item.name in _IGNORE_DIRS:
            continue
        if item.is_dir():
            entries.append(f"{item.name}/")
            child_names = []
            try:
                for child in sorted(item.iterdir(), key=lambda entry: (not entry.is_dir(), entry.name.lower())):
                    if child.name in _IGNORE_DIRS:
                        continue
                    suffix = "/" if child.is_dir() else ""
                    child_names.append(f"  - {item.name}/{child.name}{suffix}")
                    if len(child_names) >= 4:
                        break
            except OSError:
                child_names = []
            entries.extend(child_names)
        else:
            entries.append(item.name)
        if len(entries) >= max_entries:
            break
    return "\n".join(entries[:max_entries])


def _top_level_files(repo_path: Path, max_entries: int = 20) -> list[str]:
    """Return a small list of informative top-level files and directories."""
    if not repo_path.exists():
        return []
    items: list[str] = []
    try:
        for entry in sorted(repo_path.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower())):
            if entry.name in _IGNORE_DIRS:
                continue
            items.append(f"{entry.name}/" if entry.is_dir() else entry.name)
            if len(items) >= max_entries:
                break
    except OSError:
        return []
    return items


def _top_python_files(repo_path: Path, max_entries: int = 10) -> list[str]:
    """Return the most informative Python-like source files in the repository."""
    scored: list[tuple[int, str]] = []
    for path in _iter_repo_files(repo_path):
        if path.suffix.lower() != ".py":
            continue
        relative = str(path.relative_to(repo_path))
        lowered = relative.lower()
        score = 0
        if any(token in lowered for token in ("main", "train", "eval", "dataset", "model", "pipeline", "experiment")):
            score += 3
        if "test" not in lowered:
            score += 1
        text = _safe_read_text(path, max_bytes=48_000).lower()
        if "def main" in text or "if __name__ ==" in text:
            score += 2
        if "argparse" in text or "click.command" in text:
            score += 1
        scored.append((score, relative))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [item[1] for item in scored[:max_entries] if item[0] > 0]


def _count_source_files(repo_path: Path) -> int:
    """Count source-like files for a coarse repo size signal."""
    count = 0
    for path in _iter_repo_files(repo_path):
        if path.suffix.lower() in _SOURCE_EXTENSIONS:
            count += 1
    return count


def _likely_reusable_files(
    repo_path: Path,
    *,
    evidence: list[ReferenceSymbolEvidence],
    max_items: int = 8,
) -> list[str]:
    """Rank repo files that look most reusable for downstream adaptation."""
    scored: dict[str, float] = {}
    for item in evidence:
        relative = str(item.file_path or "").strip()
        if not relative:
            continue
        scored[relative] = max(float(item.score or 0.0), scored.get(relative, 0.0))
    if scored:
        return [
            path
            for path, _score in sorted(scored.items(), key=lambda item: (-item[1], item[0]))[:max_items]
        ]

    fallback: list[tuple[int, str]] = []
    for path in _iter_repo_files(repo_path):
        if path.suffix.lower() not in _SOURCE_EXTENSIONS:
            continue
        relative = str(path.relative_to(repo_path))
        lowered = relative.lower()
        score = 0
        if any(token in lowered for token in ("pipeline", "model", "trainer", "dataset", "eval", "experiment", "metric")):
            score += 3
        text = _safe_read_text(path, max_bytes=24_000).lower()
        if "def main" in text or "argparse" in text:
            score += 2
        if "dataset" in text or "dataloader" in text:
            score += 1
        if "train" in text or "evaluate" in text or "metric" in text:
            score += 1
        if "class " in text or "def " in text:
            score += 1
        fallback.append((score, relative))
    fallback.sort(key=lambda item: (-item[0], item[1]))
    return [item[1] for item in fallback[:max_items] if item[0] > 0]


def _protocol_clues(repo_path: Path, max_items: int = 10) -> list[str]:
    """Extract compact protocol clues from README and representative source files."""
    clues: list[str] = []
    readme = _readme_summary(repo_path)
    for line in readme.splitlines():
        if _PROTOCOL_PATTERN.search(line):
            clues.append(line.strip())
        if len(clues) >= max_items:
            return clues[:max_items]

    for path in _iter_repo_files(repo_path):
        if path.suffix.lower() not in _TEXT_EXTENSIONS:
            continue
        text = _safe_read_text(path, max_bytes=12_000)
        if not text:
            continue
        for raw_line in text.splitlines():
            stripped = raw_line.strip()
            if not stripped or not _PROTOCOL_PATTERN.search(stripped):
                continue
            clues.append(f"{path.relative_to(repo_path)}: {stripped[:180]}")
            if len(clues) >= max_items:
                return clues[:max_items]
    return clues[:max_items]


def _survey_symbol_evidence(
    repo_path: Path,
    *,
    ref_id: str,
    units: list[ExtractedUnit],
    boundary_output: BoundaryRequirementsOutput | None,
    max_items: int = 36,
) -> list[ReferenceSymbolEvidence]:
    """Extract reference evidence from code plus protocol/config files and bind it to units."""
    requirement_ids_by_unit = _requirement_ids_by_unit(boundary_output)
    evidence_rows: list[ReferenceSymbolEvidence] = []
    counter = 1

    for path in _iter_repo_files(repo_path):
        suffix = path.suffix.lower()
        if suffix == ".py":
            text = _safe_read_text(path, max_bytes=160_000)
            if not text:
                continue
            relative = str(path.relative_to(repo_path))
            candidates = _extract_python_symbol_candidates(relative, text)
        elif suffix in _PROTOCOL_SNIPPET_EXTENSIONS:
            text = _safe_read_text(path, max_bytes=96_000)
            if not text:
                continue
            relative = str(path.relative_to(repo_path))
            candidates = _extract_protocol_snippet_candidates(relative, text)
        else:
            continue

        for candidate in candidates:
            matched = _match_candidate_to_units(candidate, units, requirement_ids_by_unit)
            if not matched:
                continue
            matched_unit_ids = [item["unit_id"] for item in matched]
            matched_requirement_ids = _dedupe(
                [
                    requirement_id
                    for item in matched
                    for requirement_id in item["requirement_ids"]
                    if requirement_id
                ]
            )
            matched_surfaces = _dedupe(
                [
                    surface
                    for item in matched
                    for surface in item["matched_surfaces"]
                    if surface
                ]
            )
            matched_artifacts = _dedupe(
                [
                    artifact
                    for item in matched
                    for artifact in item["matched_artifacts"]
                    if artifact
                ]
            )
            matched_keywords = _dedupe(
                [
                    keyword
                    for item in matched
                    for keyword in item["matched_keywords"]
                    if keyword
                ]
            )[:12]
            max_score = max(float(item["score"]) for item in matched)
            relevance_reason = (
                f"{candidate['symbol_kind']} `{candidate['symbol_name']}` in `{relative}` aligns with units "
                f"{', '.join(matched_unit_ids[:4])} via {', '.join(matched_keywords[:4]) or 'unit-driven retrieval'}."
            )
            evidence_rows.append(
                ReferenceSymbolEvidence(
                    evidence_id=f"{ref_id}_sym_{counter:03d}",
                    ref_id=ref_id,
                    file_path=relative,
                    symbol_name=str(candidate["symbol_name"]),
                    symbol_kind=str(candidate["symbol_kind"]),
                    start_line=int(candidate["start_line"]),
                    end_line=int(candidate["end_line"]),
                    snippet=str(candidate["snippet"])[:1400],
                    matched_unit_ids=matched_unit_ids,
                    matched_requirement_ids=matched_requirement_ids,
                    matched_surfaces=matched_surfaces,
                    matched_artifacts=matched_artifacts,
                    matched_keywords=matched_keywords,
                    relevance_reason=relevance_reason,
                    score=round(max_score, 4),
                )
            )
            counter += 1

    evidence_rows.sort(
        key=lambda item: (
            -float(item.score or 0.0),
            item.file_path,
            item.start_line,
            item.symbol_name,
        )
    )
    return _select_diverse_evidence_rows(evidence_rows, max_items=max_items)


def _aggregate_requirement_coverage(
    boundary_output: BoundaryRequirementsOutput | None,
    symbol_evidence: list[ReferenceSymbolEvidence],
    *,
    units: list[ExtractedUnit],
) -> list[ReferenceRequirementCoverage]:
    """Aggregate unit-driven symbol evidence back into requirement coverage rows."""
    if boundary_output is None:
        return []

    unit_ids = {str(unit.unit_id or "").strip() for unit in units if str(unit.unit_id or "").strip()}
    coverage_rows: list[ReferenceRequirementCoverage] = []
    for requirement in boundary_output.boundary_requirements:
        requirement_id = str(requirement.requirement_id or "").strip()
        if not requirement_id:
            continue
        source_unit_ids = _dedupe(
            [
                str(unit_id or "").strip()
                for unit_id in list(requirement.source_unit_ids or [])
                if str(unit_id or "").strip() in unit_ids
            ]
        )
        matched = [
            evidence
            for evidence in symbol_evidence
            if requirement_id in list(evidence.matched_requirement_ids or [])
            or (
                source_unit_ids
                and set(source_unit_ids).intersection(str(unit_id or "").strip() for unit_id in list(evidence.matched_unit_ids or []))
            )
        ]
        if not matched:
            requirement_terms = _tokens(
                requirement.title,
                requirement.scope,
                requirement.description,
                *list(requirement.acceptance_criteria or []),
            )
            ranked: list[tuple[int, ReferenceSymbolEvidence]] = []
            for evidence in symbol_evidence:
                overlap = len(requirement_terms.intersection(set(evidence.matched_keywords or [])))
                if overlap > 0:
                    ranked.append((overlap, evidence))
            ranked.sort(key=lambda item: (-item[0], -float(item[1].score or 0.0), item[1].file_path))
            matched = [item[1] for item in ranked[:3]]

        matched_keywords = _dedupe(
            [
                keyword
                for evidence in matched
                for keyword in list(evidence.matched_keywords or [])
                if keyword
            ]
        )[:12]
        coverage_rows.append(
            ReferenceRequirementCoverage(
                requirement_id=requirement_id,
                title=str(requirement.title or "").strip(),
                scope=str(requirement.scope or "").strip(),
                source_unit_ids=source_unit_ids,
                keyword_hits=len(matched),
                matched_keywords=matched_keywords,
                matched_files=_dedupe([str(item.file_path or "").strip() for item in matched if str(item.file_path or "").strip()])[:6],
                match_locations=_dedupe(
                    [
                        f"{item.file_path}:{int(item.start_line or 0)}:{item.symbol_kind} {item.symbol_name}".strip(":")
                        for item in matched
                        if str(item.file_path or "").strip()
                    ]
                )[:6],
                code_snippets=[str(item.snippet or "") for item in matched[:2] if str(item.snippet or "").strip()],
            )
        )
    return coverage_rows


def _extract_python_symbol_candidates(relative_path: str, text: str) -> list[dict[str, Any]]:
    """Extract functions/classes and a module fallback from a Python file."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return [_module_candidate(relative_path, text)]

    lines = text.splitlines()
    candidates: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        start_line = int(getattr(node, "lineno", 1) or 1)
        end_line = int(getattr(node, "end_lineno", start_line) or start_line)
        start_index = max(0, start_line - 1)
        end_index = min(len(lines), max(end_line, start_line) + 10)
        snippet = "\n".join(
            f"{relative_path}:{line_no}: {lines[line_no - 1]}"
            for line_no in range(start_line, min(end_index, start_line + 24))
            if 1 <= line_no <= len(lines)
        ).strip()
        if not snippet:
            continue
        candidates.append(
            {
                "symbol_name": str(getattr(node, "name", "") or "__anonymous__"),
                "symbol_kind": "class" if isinstance(node, ast.ClassDef) else "function",
                "start_line": start_line,
                "end_line": end_line,
                "snippet": snippet,
                "search_text": " ".join(
                    [
                        relative_path,
                        str(getattr(node, "name", "") or ""),
                        ast.get_docstring(node, clean=False) or "",
                        snippet,
                    ]
                ),
            }
        )

    if not candidates:
        return [_module_candidate(relative_path, text)]
    return candidates


def _module_candidate(relative_path: str, text: str) -> dict[str, Any]:
    lines = text.splitlines()
    snippet = "\n".join(
        f"{relative_path}:{index + 1}: {line}"
        for index, line in enumerate(lines[:24])
    ).strip()
    return {
        "symbol_name": "__module__",
        "symbol_kind": "module",
        "start_line": 1,
        "end_line": min(len(lines), 24),
        "snippet": snippet,
        "search_text": " ".join([relative_path, snippet]),
    }


def _extract_protocol_snippet_candidates(relative_path: str, text: str) -> list[dict[str, Any]]:
    """Extract reusable protocol/config/command windows from non-Python reference files."""
    lines = text.splitlines()
    if not lines:
        return []
    lowered_path = relative_path.lower()
    path_terms = _tokens(relative_path)
    candidate_lines: list[int] = []
    for index, raw_line in enumerate(lines, start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        lowered = stripped.lower()
        has_protocol = bool(_PROTOCOL_PATTERN.search(stripped))
        has_config_value = bool(re.search(r"(^|[\s_./-])(lr|learning_rate|weight_decay|batch_size|epochs?|epsilon|alpha|beta|gamma|seed|steps?|iterations?)\s*[:=]", lowered))
        has_command = lowered.startswith(("python ", "python3 ", "bash ", "sh ", "torchrun ", "accelerate ", "CUDA_VISIBLE_DEVICES".lower()))
        if has_protocol or has_config_value or has_command or path_terms.intersection(_tokens(stripped)):
            candidate_lines.append(index)
    if not candidate_lines:
        return []

    windows: list[tuple[int, int]] = []
    for line_no in candidate_lines:
        start = max(1, line_no - 3)
        end = min(len(lines), line_no + 8)
        if windows and start <= windows[-1][1] + 2:
            windows[-1] = (windows[-1][0], max(windows[-1][1], end))
        else:
            windows.append((start, end))

    kind = _protocol_candidate_kind(relative_path)
    candidates: list[dict[str, Any]] = []
    for start, end in windows[:8]:
        snippet = "\n".join(
            f"{relative_path}:{line_no}: {lines[line_no - 1]}"
            for line_no in range(start, end + 1)
            if 1 <= line_no <= len(lines)
        ).strip()
        if not snippet:
            continue
        symbol_name = f"{Path(relative_path).stem or 'reference'}_{kind}_{start}"
        candidates.append(
            {
                "symbol_name": symbol_name,
                "symbol_kind": kind,
                "start_line": start,
                "end_line": end,
                "snippet": snippet,
                "search_text": " ".join([relative_path, lowered_path, snippet]),
            }
        )
    return candidates


def _protocol_candidate_kind(relative_path: str) -> str:
    suffix = Path(relative_path).suffix.lower()
    lowered = relative_path.lower()
    if suffix in {".yaml", ".yml", ".json", ".toml", ".ini", ".cfg"}:
        return "config_protocol"
    if suffix in {".sh", ".bash", ".zsh"}:
        return "command_protocol"
    if "readme" in lowered or suffix in {".md", ".rst", ".txt"}:
        return "doc_protocol"
    if suffix == ".ipynb":
        return "notebook_protocol"
    return "protocol"


def _select_diverse_evidence_rows(
    evidence_rows: list[ReferenceSymbolEvidence],
    *,
    max_items: int,
) -> list[ReferenceSymbolEvidence]:
    """Retain high-scoring code evidence while reserving budget for protocol/config snippets."""
    if len(evidence_rows) <= max_items:
        return evidence_rows
    protocol_kinds = {"config_protocol", "command_protocol", "doc_protocol", "notebook_protocol", "protocol"}
    protocol_rows = [item for item in evidence_rows if str(item.symbol_kind or "") in protocol_kinds]
    selected: list[ReferenceSymbolEvidence] = []
    seen_ids: set[str] = set()
    per_file: dict[str, int] = {}

    def add_rows(rows: list[ReferenceSymbolEvidence], limit: int, per_file_limit: int) -> None:
        for item in rows:
            if len(selected) >= limit:
                return
            if item.evidence_id in seen_ids:
                continue
            file_path = str(item.file_path or "")
            if per_file.get(file_path, 0) >= per_file_limit:
                continue
            selected.append(item)
            seen_ids.add(item.evidence_id)
            per_file[file_path] = per_file.get(file_path, 0) + 1

    protocol_budget = min(max(len(protocol_rows), 0), max(6, max_items // 3))
    add_rows(protocol_rows, protocol_budget, per_file_limit=2)
    add_rows(evidence_rows, max_items, per_file_limit=4)
    if len(selected) < max_items:
        add_rows(evidence_rows, max_items, per_file_limit=999)
    return selected[:max_items]


def _match_candidate_to_units(
    candidate: dict[str, Any],
    units: list[ExtractedUnit],
    requirement_ids_by_unit: dict[str, list[str]],
) -> list[dict[str, Any]]:
    candidate_text = str(candidate.get("search_text", "") or "")
    candidate_tokens = _tokens(candidate_text)
    candidate_lower = candidate_text.lower()
    matched: list[dict[str, Any]] = []
    for unit in units:
        unit_id = str(unit.unit_id or "").strip()
        if not unit_id:
            continue
        unit_tokens = _unit_tokens(unit)
        overlap = sorted(candidate_tokens.intersection(unit_tokens))
        surface_hits = [
            surface
            for surface in list(unit.implementation_surfaces or [])
            if any(hint in candidate_lower for hint in _SURFACE_HINTS.get(str(surface), ()))
        ]
        artifact_hits = [
            artifact
            for artifact in list(unit.expected_artifacts or [])
            if any(token in candidate_lower for token in _tokens(artifact))
        ]
        score = float(len(overlap)) * 1.5
        score += float(len(surface_hits)) * 1.75
        score += float(len(artifact_hits)) * 1.25
        symbol_name = str(candidate.get("symbol_name", "") or "").lower()
        if symbol_name and symbol_name not in _GENERIC_SYMBOL_NAMES and symbol_name in candidate_lower:
            score += 0.25
        if not overlap and not surface_hits and not artifact_hits:
            continue
        if symbol_name in _GENERIC_SYMBOL_NAMES and len(overlap) <= 0 and len(surface_hits) < 2 and len(artifact_hits) <= 0:
            continue
        if not surface_hits and len(overlap) < 2 and len(artifact_hits) <= 0:
            continue
        if score < 2.5:
            continue
        matched.append(
            {
                "unit_id": unit_id,
                "score": score,
                "requirement_ids": list(requirement_ids_by_unit.get(unit_id, [])),
                "matched_surfaces": _dedupe(surface_hits),
                "matched_artifacts": _dedupe(artifact_hits),
                "matched_keywords": overlap[:10],
            }
        )
    matched.sort(key=lambda item: (-float(item["score"]), item["unit_id"]))
    return matched[:4]


def _requirement_ids_by_unit(boundary_output: BoundaryRequirementsOutput | None) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    if boundary_output is None:
        return mapping
    for requirement in boundary_output.boundary_requirements:
        requirement_id = str(requirement.requirement_id or "").strip()
        if not requirement_id:
            continue
        for unit_id in list(requirement.source_unit_ids or []):
            normalized = str(unit_id or "").strip()
            if not normalized:
                continue
            mapping.setdefault(normalized, [])
            if requirement_id not in mapping[normalized]:
                mapping[normalized].append(requirement_id)
    return mapping


def _unit_tokens(unit: ExtractedUnit) -> set[str]:
    return _tokens(
        unit.statement,
        *list(unit.implementation_surfaces or []),
        *list(unit.code_obligations or []),
        *list(unit.runtime_interfaces or []),
        *list(unit.expected_artifacts or []),
        *list(unit.suggested_module_kinds or []),
    )


def _tokens(*parts: str) -> set[str]:
    values: set[str] = set()
    for part in parts:
        text = str(part or "").strip().lower()
        if not text:
            continue
        for token in _TOKEN_RE.findall(text):
            normalized = token.strip("._-/")
            if len(normalized) < 2 or normalized in _STOPWORDS:
                continue
            values.add(normalized)
            if "/" in normalized:
                values.update(fragment for fragment in normalized.split("/") if len(fragment) >= 2 and fragment not in _STOPWORDS)
            if "_" in normalized:
                values.update(fragment for fragment in normalized.split("_") if len(fragment) >= 2 and fragment not in _STOPWORDS)
            if "-" in normalized:
                values.update(fragment for fragment in normalized.split("-") if len(fragment) >= 2 and fragment not in _STOPWORDS)
    return values


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


def _iter_repo_files(repo_path: Path):
    """Yield repo files while skipping heavy irrelevant directories."""
    if not repo_path.exists():
        return
    for path in repo_path.rglob("*"):
        if not path.is_file():
            continue
        relative_parts = path.relative_to(repo_path).parts
        if any(part in _IGNORE_DIRS for part in relative_parts):
            continue
        yield path


def _safe_read_text(path: Path, max_bytes: int = 256_000) -> str:
    """Read a text file defensively with size and decoding guards."""
    try:
        if path.stat().st_size > max_bytes:
            return ""
        data = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""
    return data.strip()
