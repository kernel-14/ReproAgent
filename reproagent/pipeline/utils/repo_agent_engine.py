"""Repo-level agent execution helpers for reproagent generation and repair."""

from __future__ import annotations

import json
import logging
import re
import shutil
import os
import ast
from pathlib import Path
from typing import Any

from reproagent.pipeline.schemas import PaperBenchReproState
from reproagent.pipeline.tools import load_project_files, merge_project_files, save_project_files
from reproagent.pipeline.config import semantic_anchor_disabled
from reproagent.pipeline.utils.claude_sdk_wrapper import ClaudeSDKWrapper
from reproagent.pipeline.utils.codex_wrapper import CodexWrapper, aggregate_codex_usage, normalize_codex_usage
from reproagent.pipeline.utils.llm_agent import LLMWorkflowAgent
from reproagent.pipeline.utils.contract_sanitizer import sanitize_task_contract

from . import memory as run_memory

logger = logging.getLogger(__name__)


def _looks_like_non_python_file(content: str, target_file: str) -> bool:
    """Return True when extracted text is plausible content for a non-Python target."""
    stripped = str(content or "").strip()
    if not stripped:
        return False
    path = Path(str(target_file or ""))
    name = path.name.lower()
    suffix = path.suffix.lower()
    if name in {"dockerfile", "makefile"}:
        return True
    if suffix in {".md", ".rst", ".txt", ".cfg", ".ini", ".csv", ".tsv", ".sh", ".bash"}:
        return True
    if suffix == ".toml":
        return "[" in stripped or "=" in stripped
    if suffix in {".yaml", ".yml"}:
        return ":" in stripped or stripped.startswith("-")
    if suffix == ".json":
        return stripped.startswith("{") or stripped.startswith("[")
    if not suffix:
        return "\n" in stripped
    return True


class RepoAgentEngine:
    """Small repo-driven engine used by reproagent generate/repair stages."""

    def __init__(self, codegen_config: Any, workflow_config: Any, sandbox_provider: Any):
        self.codegen_config = codegen_config
        self.workflow_config = workflow_config
        self.sandbox_provider = sandbox_provider
        self.language = "zh"
        self.output_dir: Path | None = None
        self.checkpoint_path = ""
        self.node_artifact_dir: Path | None = None

    def _node_dir(self) -> Path:
        if self.node_artifact_dir is not None:
            return self.node_artifact_dir
        return (self.output_dir or Path(".")) / "nodes" / "generate"

    def _repo_dir(self) -> Path:
        return (self.output_dir or Path(".")) / "repo"

    def _safe_task_segment(self, value: str) -> str:
        sanitized = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(value or "").strip())
        return sanitized.strip("._") or "task"

    def _task_dir(self, iteration: int, task_id: str) -> Path:
        del iteration, task_id
        return self._node_dir()

    def _agent_trace_dir(self) -> Path:
        return self._node_dir() / "agent_traces"

    def _save_agent_trace(
        self,
        *,
        iteration: int,
        task_id: str,
        attempt_index: int,
        prompt: str,
        result: dict[str, Any],
        project_files_before: dict[str, str],
        project_files_after: dict[str, str],
    ) -> None:
        if self.output_dir is None:
            return
        trace_dir = self._agent_trace_dir()
        trace_dir.mkdir(parents=True, exist_ok=True)
        task_segment = self._safe_task_segment(task_id)
        prefix = f"{int(iteration):03d}_{task_segment}_attempt_{int(attempt_index) + 1:02d}"
        usage = normalize_codex_usage(result.get("usage"))
        summary = {
            "iteration": int(iteration),
            "task_id": str(task_id or ""),
            "attempt": int(attempt_index) + 1,
            "success": bool(result.get("success")),
            "timed_out": bool(result.get("timed_out", False)),
            "exit_code": int(result.get("exit_code", 0) or 0),
            "error": str(result.get("error", "") or "")[:4000],
            "session_id": str(usage.get("session_id", "") or ""),
            "usage": usage,
            "prompt_chars": len(str(prompt or "")),
            "prompt_estimated_tokens": self._estimate_text_tokens(prompt),
            "raw_output_chars": len(str(result.get("raw_output", "") or "")),
            "stderr_chars": len(str(result.get("stderr", "") or "")),
            "files_before": sorted(project_files_before.keys()),
            "files_after": sorted(project_files_after.keys()),
            "changed_files": sorted(self._diff_project_files(project_files_before, project_files_after).keys()),
        }
        (trace_dir / f"{prefix}.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        for suffix, content in {
            "prompt.md": str(prompt or ""),
            "stdout.txt": str(result.get("raw_output", "") or result.get("output", "") or ""),
            "stderr.txt": str(result.get("stderr", "") or ""),
        }.items():
            (trace_dir / f"{prefix}.{suffix}").write_text(content, encoding="utf-8")

    def _diff_project_files(
        self,
        previous_files: dict[str, str],
        project_files: dict[str, str],
    ) -> dict[str, str]:
        return {
            path: content
            for path, content in project_files.items()
            if previous_files.get(path) != content
        }

    def _public_python_symbols(self, content: str) -> set[str]:
        try:
            tree = ast.parse(str(content or ""))
        except SyntaxError:
            return set()
        symbols: set[str] = set()
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if not node.name.startswith("_"):
                    symbols.add(node.name)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = list(node.targets) if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name) and not target.id.startswith("_"):
                        symbols.add(target.id)
                    elif isinstance(target, (ast.Tuple, ast.List)):
                        for item in target.elts:
                            if isinstance(item, ast.Name) and not item.id.startswith("_"):
                                symbols.add(item.id)
        return symbols

    def _semantic_anchor_terms(self, content: str) -> set[str]:
        """Extract paper/domain-looking terms without paper-specific hard-coding."""
        text = str(content or "")
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9_'-]{1,}|\d+(?:\.\d+)?[kKmM]?|\d+-[A-Za-z0-9_'-]+", text)
        stopwords = {
            "about", "above", "after", "again", "against", "allow", "also", "and", "any",
            "artifact", "artifacts", "before", "being", "between", "bool", "call", "class",
            "code", "config", "data", "default", "dict", "does", "each", "else", "false",
            "file", "files", "float", "from", "function", "generated", "import", "inside",
            "list", "main", "make", "method", "mode", "none", "only", "output", "paper",
            "path", "return", "results", "route", "self", "smoke", "state", "string",
            "that", "the", "their", "these", "this", "true", "use", "used", "using",
            "value", "when", "with", "without",
        }
        anchors: set[str] = set()
        for token in tokens:
            normalized = token.strip("_'\"")
            if not normalized:
                continue
            lowered = normalized.lower()
            if lowered in stopwords:
                continue
            if len(normalized) <= 1:
                continue
            has_digit = any(ch.isdigit() for ch in normalized)
            has_hyphen = "-" in normalized
            has_apostrophe = "'" in normalized
            is_upper_acronym = normalized.isupper() and 2 <= len(normalized) <= 10
            is_camel = bool(re.search(r"[a-z][A-Z]|[A-Z][a-z]+[A-Z]", normalized))
            is_domain_word = len(normalized) >= 7 and normalized.isalpha()
            if has_digit or has_hyphen or has_apostrophe or is_upper_acronym or is_camel or is_domain_word:
                anchors.add(lowered)
        return anchors

    def _syntax_report_for_python(self, path: str, content: str) -> str:
        if not (str(path).endswith(".py") or str(path).endswith("/__init__.py")):
            return ""
        try:
            ast.parse(str(content or ""))
        except SyntaxError as exc:
            return f"{path}: SyntaxError line {exc.lineno}: {exc.msg}"
        return ""

    def _repair_safety_thresholds(self) -> dict[str, Any]:
        return {
            "enabled": bool(getattr(self.workflow_config, "repair_regression_guard_enabled", True)),
            "allow_large_rewrites": bool(getattr(self.workflow_config, "repair_guard_allow_large_rewrites", False)),
            "large_file_min_lines": int(getattr(self.workflow_config, "repair_guard_large_file_min_lines", 120) or 120),
            "large_file_min_bytes": int(getattr(self.workflow_config, "repair_guard_large_file_min_bytes", 6000) or 6000),
            "min_line_retention": float(getattr(self.workflow_config, "repair_guard_min_line_retention", 0.45) or 0.45),
            "min_byte_retention": float(getattr(self.workflow_config, "repair_guard_min_byte_retention", 0.45) or 0.45),
            "min_symbol_retention": float(getattr(self.workflow_config, "repair_guard_min_symbol_retention", 0.70) or 0.70),
            "min_anchor_retention": float(getattr(self.workflow_config, "repair_guard_min_anchor_retention", 0.70) or 0.70),
        }

    def _guard_repo_repair_update(
        self,
        *,
        previous_files: dict[str, str],
        project_files: dict[str, str],
        updated_files: dict[str, str],
        recommended_surfaces: list[str],
    ) -> dict[str, Any]:
        thresholds = self._repair_safety_thresholds()
        if not thresholds["enabled"]:
            return {"passed": True, "disabled": True, "issues": [], "file_reports": [], "thresholds": thresholds}
        issues: list[str] = []
        file_reports: list[dict[str, Any]] = []
        recommended = {str(item).strip() for item in recommended_surfaces if str(item).strip()}
        for deleted_path in sorted(set(previous_files) - set(project_files)):
            if deleted_path.startswith("__pycache__/"):
                continue
            issues.append(f"{deleted_path}: file deletion rejected during repair")
            file_reports.append(
                {
                    "path": deleted_path,
                    "status": "deleted_existing_file",
                    "recommended_surface": deleted_path in recommended,
                    "old_lines": len(str(previous_files.get(deleted_path, "")).splitlines()),
                    "new_lines": 0,
                    "old_bytes": len(str(previous_files.get(deleted_path, "")).encode("utf-8")),
                    "new_bytes": 0,
                }
            )
        for path, new_content in sorted(updated_files.items()):
            path = str(path).strip()
            old_content = previous_files.get(path)
            if old_content is None:
                syntax_issue = self._syntax_report_for_python(path, new_content)
                if syntax_issue:
                    issues.append(syntax_issue)
                file_reports.append(
                    {
                        "path": path,
                        "status": "new_file",
                        "syntax_issue": syntax_issue,
                        "recommended_surface": path in recommended,
                    }
                )
                continue

            old_lines = len(str(old_content).splitlines())
            new_lines = len(str(new_content).splitlines())
            old_bytes = len(str(old_content).encode("utf-8"))
            new_bytes = len(str(new_content).encode("utf-8"))
            line_retention = (new_lines / old_lines) if old_lines else 1.0
            byte_retention = (new_bytes / old_bytes) if old_bytes else 1.0
            syntax_issue = self._syntax_report_for_python(path, new_content)
            if syntax_issue:
                issues.append(syntax_issue)

            large_existing_file = (
                old_lines >= thresholds["large_file_min_lines"]
                or old_bytes >= thresholds["large_file_min_bytes"]
            )
            if (
                large_existing_file
                and not thresholds["allow_large_rewrites"]
                and (
                    line_retention < thresholds["min_line_retention"]
                    or byte_retention < thresholds["min_byte_retention"]
                )
            ):
                issues.append(
                    f"{path}: destructive shrink rejected "
                    f"({old_lines}->{new_lines} lines, {old_bytes}->{new_bytes} bytes)"
                )

            old_symbols = self._public_python_symbols(old_content) if path.endswith(".py") else set()
            new_symbols = self._public_python_symbols(new_content) if path.endswith(".py") else set()
            symbol_retention = (
                len(old_symbols & new_symbols) / len(old_symbols)
                if old_symbols
                else 1.0
            )
            if (
                large_existing_file
                and path.endswith(".py")
                and len(old_symbols) >= 8
                and symbol_retention < thresholds["min_symbol_retention"]
            ):
                missing = sorted(old_symbols - new_symbols)[:12]
                issues.append(
                    f"{path}: public symbol regression rejected "
                    f"({len(old_symbols & new_symbols)}/{len(old_symbols)} retained; missing={missing})"
                )

            old_anchors = self._semantic_anchor_terms(old_content)
            new_anchors = self._semantic_anchor_terms(new_content)
            anchor_retention = (
                len(old_anchors & new_anchors) / len(old_anchors)
                if old_anchors
                else 1.0
            )
            if (
                large_existing_file
                and len(old_anchors) >= 12
                and anchor_retention < thresholds["min_anchor_retention"]
            ):
                missing_anchors = sorted(old_anchors - new_anchors)[:20]
                issues.append(
                    f"{path}: semantic anchor regression rejected "
                    f"({len(old_anchors & new_anchors)}/{len(old_anchors)} retained; missing={missing_anchors})"
                )

            file_reports.append(
                {
                    "path": path,
                    "status": "changed_existing_file",
                    "recommended_surface": path in recommended,
                    "old_lines": old_lines,
                    "new_lines": new_lines,
                    "old_bytes": old_bytes,
                    "new_bytes": new_bytes,
                    "line_retention": line_retention,
                    "byte_retention": byte_retention,
                    "old_public_symbol_count": len(old_symbols),
                    "new_public_symbol_count": len(new_symbols),
                    "symbol_retention": symbol_retention,
                    "old_anchor_count": len(old_anchors),
                    "new_anchor_count": len(new_anchors),
                    "anchor_retention": anchor_retention,
                    "syntax_issue": syntax_issue,
                    "large_existing_file": large_existing_file,
                }
            )
        return {
            "passed": not issues,
            "issues": issues,
            "file_reports": file_reports,
            "thresholds": thresholds,
        }

    def _write_repair_safety_report(self, report: dict[str, Any]) -> None:
        if self.output_dir is None:
            return
        node_dir = self._node_dir()
        node_dir.mkdir(parents=True, exist_ok=True)
        (node_dir / "repair_safety_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _persist_project_files(self, project_files: dict[str, str]) -> Path:
        repo_dir = self._repo_dir()
        repo_dir.mkdir(parents=True, exist_ok=True)
        existing_paths = {
            path
            for path in repo_dir.rglob("*")
            if path.is_file()
        }
        save_project_files(project_files, repo_dir)
        desired_paths = {
            (repo_dir / relative_path).resolve()
            for relative_path in project_files
        }
        for path in existing_paths:
            resolved = path.resolve()
            if resolved in desired_paths:
                continue
            try:
                os.remove(resolved)
            except FileNotFoundError:
                continue
        for path in sorted({item.parent for item in existing_paths}, reverse=True):
            if path == repo_dir:
                continue
            try:
                path.rmdir()
            except OSError:
                continue
        return repo_dir

    def _project_entrypoints(self, project_plan: dict[str, Any]) -> dict[str, str]:
        entrypoints = project_plan.get("entrypoints", {}) if isinstance(project_plan, dict) else {}
        main_path = str(entrypoints.get("main", "main.py") or "main.py").strip() or "main.py"
        runtime_smoke = str(entrypoints.get("runtime_smoke", "") or "").strip() or f"python {main_path}"
        docker_validate = str(entrypoints.get("docker_validate", "") or "").strip() or runtime_smoke
        return {
            "main": main_path,
            "runtime_smoke": runtime_smoke,
            "docker_validate": docker_validate,
        }

    def _artifact_instruction_block(self) -> str:
        return (
            "Artifact requirements:\n"
            "- Persist declared runtime artifacts under the repository output paths.\n"
            "- Use os.environ['PAPERBENCH_REPRO_ARTIFACT_DIR'] for auxiliary artifact output when available.\n"
            "- The repository may expose dry-run readiness/manifest writers for smoke validation, but paper-visible tables, figures, metrics, predictions, and reports must not be replaced by schema-only result shells.\n"
            "- Dry-run artifacts must be explicitly labeled as readiness/schema/contract artifacts and must not claim benchmark scores, trained-model performance, or completed experiments.\n"
            "- For declared table/CSV/JSON/JSONL/figure paths, create parent directories during smoke validation, but only write benchmark-visible content when it is computed by the bounded implementation route; otherwise record readiness/full-mode requirements in auxiliary manifests.\n"
            "- Write `readiness.json` and `evaluation_result.json` from smoke validation so downstream validation can confirm the command exercised artifact closure.\n"
            "- Satisfy the plan with faithful implementation modules, real entrypoints, and code-backed contract files.\n"
            "- If a required file cannot be implemented faithfully, leave it missing and state that in the summary.\n"
            "- Keep the repository importable in a minimal code-only smoke environment: optional heavy packages may be listed in requirements, but importing project modules must not fail solely because such packages are absent.\n"
            "- Use lazy imports, small protocol-style fallbacks, or clear runtime errors inside training-only methods for unavailable optional dependencies; do not put optional RL/vision/simulator imports at module top level when a lightweight import smoke will touch the file.\n"
        )

    def _estimate_context_usage(self, prompt: str) -> dict[str, Any]:
        window_tokens = max(1, int(getattr(self.workflow_config, "repair_context_window_tokens", 32000) or 32000))
        estimated_tokens = max(1, len(prompt) // 4)
        return {
            "estimated_tokens": estimated_tokens,
            "window_tokens": window_tokens,
            "ratio": estimated_tokens / window_tokens,
        }

    def _estimate_text_tokens(self, text: str) -> int:
        rendered = str(text or "")
        return max(1, len(rendered) // 4) if rendered.strip() else 0

    def _build_task_usage(self, prompt: str, agent_result: dict[str, Any]) -> dict[str, Any]:
        raw_output = str(agent_result.get("raw_output", "") or agent_result.get("output", "") or "")
        estimated_input_tokens = self._estimate_text_tokens(prompt)
        estimated_output_tokens = self._estimate_text_tokens(raw_output)
        estimated_total_tokens = estimated_input_tokens + estimated_output_tokens
        usage = normalize_codex_usage(agent_result.get("usage"))
        window_tokens = max(1, int(getattr(self.workflow_config, "repair_context_window_tokens", 32000) or 32000))
        return {
            "estimated_input_tokens": estimated_input_tokens,
            "estimated_output_tokens": estimated_output_tokens,
            "estimated_total_tokens": estimated_total_tokens,
            "actual_input_tokens": int(usage.get("input_tokens", 0) or 0),
            "actual_output_tokens": int(usage.get("output_tokens", 0) or 0),
            "actual_total_tokens": int(usage.get("total_tokens", 0) or 0),
            "usage_found": bool(usage.get("usage_found")),
            "usage_source": str(usage.get("usage_source", "") or ""),
            "session_id": str(usage.get("session_id", "") or ""),
            "matched_lines": list(usage.get("matched_lines", []) or []),
            "window_tokens": window_tokens,
            "ratio": (estimated_input_tokens / window_tokens) if window_tokens else 0.0,
        }

    def _summarize_task_usages(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        normalized_items = [dict(item or {}) for item in items if isinstance(item, dict)]
        usage_summary = aggregate_codex_usage(normalized_items)
        usage_sources: list[str] = []
        for item in normalized_items:
            source = str(item.get("usage_source", "") or "")
            if source and source not in usage_sources:
                usage_sources.append(source)
        return {
            "calls": len(normalized_items),
            "calls_with_usage": int(usage_summary.get("calls_with_usage", 0) or 0),
            "input_tokens": int(usage_summary.get("input_tokens", 0) or 0),
            "output_tokens": int(usage_summary.get("output_tokens", 0) or 0),
            "total_tokens": int(usage_summary.get("total_tokens", 0) or 0),
            "session_ids": list(usage_summary.get("session_ids", []) or []),
            "usage_sources": usage_sources or list(usage_summary.get("usage_sources", []) or []),
            "estimated_input_tokens": sum(int(item.get("estimated_input_tokens", 0) or 0) for item in normalized_items),
            "estimated_output_tokens": sum(int(item.get("estimated_output_tokens", 0) or 0) for item in normalized_items),
            "estimated_total_tokens": sum(int(item.get("estimated_total_tokens", 0) or 0) for item in normalized_items),
        }

    def _render_project_files_for_prompt(self, project_files: dict[str, str]) -> str:
        if not project_files:
            return ""
        chunks: list[str] = []
        for index, (path, content) in enumerate(sorted(project_files.items())):
            if index >= 6:
                break
            chunks.append(f"[{path}]\n{content[:1800]}")
        return "\n\n".join(chunks)

    def _task_scope_instruction_block(self, current_task_input: dict[str, Any], current_file_plan: dict[str, Any]) -> str:
        target_file = str(current_task_input.get("file_path", "") or "").strip()
        writes_artifacts = [
            str(item or "").strip()
            for item in [
                *list(current_task_input.get("writes_artifacts", []) or []),
                *list(current_file_plan.get("writes_artifacts", []) or []),
            ]
            if str(item or "").strip()
        ]
        reads_artifacts = [
            str(item or "").strip()
            for item in [
                *list(current_task_input.get("reads_artifacts", []) or []),
                *list(current_file_plan.get("reads_artifacts", []) or []),
            ]
            if str(item or "").strip()
        ]
        lower_target = target_file.lower()
        is_entrypoint = lower_target in {"main.py", "run.py"} or lower_target.endswith("/main.py")
        is_test = lower_target.startswith("tests/") or "/tests/" in lower_target
        is_doc_or_config = lower_target.endswith((".md", ".yaml", ".yml", ".toml", ".json"))
        owns_runtime_artifacts = bool(writes_artifacts) or any(
            token in lower_target
            for token in ("artifact", "report", "metric", "evaluation", "main")
        )
        if is_test:
            verification = (
                "After editing, run the narrowest relevant test or syntax check for this test file. "
                "Do not run the full test suite unless this task explicitly asks for it."
            )
        elif is_doc_or_config:
            verification = (
                "After editing, run a lightweight parse/read check if useful. "
                "Do not execute canonical routes just because this file names artifacts."
            )
        else:
            verification = (
                "After editing, run at most a syntax check and a dependency-light import/symbol smoke for the target module. "
                "Do not execute canonical routes, artifact writers, training loops, dataset downloads, or full package smoke from this task."
            )
        if is_entrypoint or owns_runtime_artifacts:
            verification += (
                " During generate, this is still a materialization task: create the target file and verify it imports/parses. "
                "Defer full artifact-route execution and package-wide smoke to work-package review, repo validation, or repair."
            )
        context = (
            "Read only the target file, direct dependency files listed in `dependency_files`, and small focused snippets needed to preserve imports/call sites. "
            "Do not re-read large generated modules or repository-wide artifacts unless the current file directly imports them and a focused snippet is insufficient."
        )
        return "\n".join(
            [
                "Task scope and speed discipline:",
                f"- Target file: `{target_file}`.",
                f"- Declared writes_artifacts: {writes_artifacts[:8] if writes_artifacts else 'none'}.",
                f"- Declared reads_artifacts: {reads_artifacts[:8] if reads_artifacts else 'none'}.",
                f"- Context discipline: {context}",
                f"- Verification discipline: {verification}",
                "- Summarize only after the target file exists on disk and the lightweight check passes or the failure is explicitly reported.",
            ]
        )

    def _save_iteration_checkpoint(
        self,
        iteration: int,
        execution_history: list[dict[str, Any]],
        generated_files: dict[str, str],
        termination_reason: str,
        latest_stage: str,
        latest_status: str,
    ) -> dict[str, Any]:
        payload = {
            "current_iteration": iteration,
            "best_round": iteration,
            "termination_reason": termination_reason,
            "latest_stage": latest_stage,
            "latest_status": latest_status,
            "generated_files": sorted(generated_files.keys()),
            "execution_history_count": len(execution_history),
        }
        if self.checkpoint_path:
            checkpoint_path = Path(self.checkpoint_path)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        return payload

    def _save_attempt(
        self,
        iteration: int,
        code: str,
        result: dict[str, Any],
        suggestions: list[str],
        repair_trace: list[Any],
        context_usage: dict[str, Any],
        project_files: dict[str, str],
        project_root: str,
        project_manifest: dict[str, Any],
        *,
        checkpoint_payload: dict[str, Any] | None = None,
        task_id: str = "",
        changed_files: dict[str, str] | None = None,
        **_: Any,
    ) -> None:
        if self.output_dir is None:
            return
        changed_files = dict(changed_files or {})
        payload = {
            "iteration": iteration,
            "task_id": task_id,
            "code": code,
            "execution_result": result,
            "suggestions": suggestions,
            "repair_trace": repair_trace,
            "context_usage": context_usage,
            "changed_files": sorted(changed_files.keys()),
            "project_root": project_root,
            "project_manifest": project_manifest,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_payload": checkpoint_payload or {},
        }
        node_dir = self._node_dir()
        node_dir.mkdir(parents=True, exist_ok=True)
        (node_dir / "last_attempt.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        if changed_files:
            changed_root = node_dir / "changed_files"
            save_project_files(changed_files, changed_root)

    def _targets_met(self, experiment_results: dict[str, Any], raw_metrics: dict[str, Any] | None = None) -> bool:
        raw = raw_metrics or {}
        for key in ("all_targets_met", "target_hit", "target_met", "meets_target"):
            if raw.get(key) is True:
                return True
        summary = dict(experiment_results.get("summary", {}) or {})
        return any(summary.get(key) is True for key in ("all_targets_met", "target_hit", "target_met"))

    def _trim_prompt_text(self, value: Any, max_chars: int) -> str:
        text = str(value or "")
        if max_chars <= 0 or len(text) <= max_chars:
            return text
        return text[:max_chars].rstrip() + "\n...[truncated for generation prompt budget]"

    def _compact_sequence(self, items: Any, *, limit: int, item_max_chars: int = 500) -> list[Any]:
        compacted: list[Any] = []
        for item in list(items or [])[: max(0, limit)]:
            if isinstance(item, str):
                compacted.append(self._trim_prompt_text(item, item_max_chars))
            elif isinstance(item, dict):
                compacted.append(self._compact_json_for_prompt(item, max_text_chars=item_max_chars))
            else:
                compacted.append(item)
        return compacted

    def _high_signal_sequence(self, items: Any, *, limit: int, item_max_chars: int = 520) -> list[Any]:
        """Preserve paper-specific obligations more aggressively than generic lists."""
        compacted: list[Any] = []
        for item in list(items or []):
            if len(compacted) >= max(0, limit):
                break
            if isinstance(item, str):
                rendered = self._trim_prompt_text(item, item_max_chars)
            elif isinstance(item, dict):
                rendered = self._compact_json_for_prompt(item, max_text_chars=item_max_chars)
            else:
                rendered = item
            if rendered not in compacted:
                compacted.append(rendered)
        return compacted

    def _compact_file_index_plan(self, item: Any) -> dict[str, Any]:
        """A very small repo index row for cross-file orientation."""
        plan = dict(item or {}) if isinstance(item, dict) else {}
        return {
            "target_file": str(plan.get("target_file", "") or plan.get("path", "") or ""),
            "task_id": str(plan.get("task_id", "") or ""),
            "work_package_id": str(plan.get("work_package_id", "") or ""),
            "defines_symbols": self._compact_sequence(plan.get("defines_symbols", []), limit=4, item_max_chars=80),
            "writes_artifacts": self._compact_sequence(plan.get("writes_artifacts", []), limit=4, item_max_chars=100),
        }

    def _compact_formula_algorithm_anchor(self, item: Any) -> dict[str, Any]:
        anchor = dict(item or {}) if isinstance(item, dict) else {}
        return {
            "source_id": str(anchor.get("source_id", "") or ""),
            "section_title": str(anchor.get("section_title", "") or ""),
            "formula_or_algorithm_excerpts": self._compact_sequence(
                anchor.get("formula_or_algorithm_excerpts", []),
                limit=2,
                item_max_chars=450,
            ),
            "required_symbols": self._compact_sequence(anchor.get("required_symbols", []), limit=12, item_max_chars=80),
            "required_numeric_values": self._compact_sequence(anchor.get("required_numeric_values", []), limit=8, item_max_chars=80),
            "algorithm_terms": self._compact_sequence(anchor.get("algorithm_terms", []), limit=10, item_max_chars=80),
            "algorithm_steps": self._compact_sequence(anchor.get("algorithm_steps", []), limit=5, item_max_chars=180),
            "implementation_obligation": self._trim_prompt_text(anchor.get("implementation_obligation", ""), 600),
        }

    def _compact_formula_algorithm_contract(self, payload: Any) -> dict[str, Any]:
        contract = dict(payload or {}) if isinstance(payload, dict) else {}
        anchors = [
            self._compact_formula_algorithm_anchor(item)
            for item in list(contract.get("anchors", []) or [])[:5]
            if isinstance(item, dict)
        ]
        return {
            "source": str(contract.get("source", "") or ""),
            "anchor_count": int(contract.get("anchor_count", len(anchors)) or len(anchors)),
            "anchors": anchors,
        }

    def _compact_paper_evidence_contract_for_codegen(self, payload: Any) -> dict[str, Any]:
        evidence = dict(payload or {}) if isinstance(payload, dict) else {}
        contract = dict(evidence.get("contract", {}) or {})
        if not contract and any(key in evidence for key in ("methods", "datasets", "metrics")):
            contract = evidence
        return {
            "required_claim_inventory": self._compact_json_for_prompt(
                evidence.get("required_claim_inventory", contract),
                max_text_chars=500,
            ),
            "formula_algorithm_contract": self._compact_formula_algorithm_contract(
                evidence.get("formula_algorithm_contract")
                or contract.get("formula_algorithm_contract")
                or {}
            ),
            "closure_items": self._compact_sequence(
                evidence.get("closure_items", []),
                limit=40,
                item_max_chars=140,
            ),
            "closure_rules": self._compact_sequence(
                evidence.get("closure_rules", []),
                limit=4,
                item_max_chars=360,
            ),
            "prepare_gate_summary": self._compact_json_for_prompt(
                evidence.get("prepare_gate_summary", {}),
                max_text_chars=500,
            ),
        }

    def _compact_paper_claim_inventory(self, payload: Any, *, max_items_per_category: int = 80) -> dict[str, list[str]]:
        if not isinstance(payload, dict):
            return {}
        compacted: dict[str, list[str]] = {}
        for key, values in payload.items():
            category = str(key or "").strip()
            if not category or not isinstance(values, list):
                continue
            compacted[category] = self._compact_sequence(
                values,
                limit=max_items_per_category,
                item_max_chars=180,
            )
        return compacted

    def _compact_json_for_prompt(self, payload: Any, *, max_text_chars: int = 800) -> Any:
        if isinstance(payload, str):
            return self._trim_prompt_text(payload, max_text_chars)
        if isinstance(payload, list):
            return self._compact_sequence(payload, limit=12, item_max_chars=max_text_chars)
        if not isinstance(payload, dict):
            return payload
        return {
            str(key): self._compact_json_for_prompt(value, max_text_chars=max_text_chars)
            for key, value in payload.items()
            if key not in {"resource_manifest", "reference_repo_surveys", "paper_chunks", "paper_text", "proposal_text"}
        }

    def _compact_generation_context_for_prompt(self, payload: Any) -> dict[str, Any]:
        """Compact the task-local prepare evidence without dropping its anchor keys."""
        if not isinstance(payload, dict):
            return {}
        compacted: dict[str, Any] = {}
        paper_chunks = payload.get("paper_chunks", [])
        if isinstance(paper_chunks, list):
            compacted["paper_chunks"] = self._compact_sequence(
                paper_chunks,
                limit=3,
                item_max_chars=700,
            )
        reference_surveys = payload.get("reference_surveys", [])
        if isinstance(reference_surveys, list):
            compacted["reference_surveys"] = self._compact_sequence(
                reference_surveys,
                limit=2,
                item_max_chars=500,
            )
        symbol_evidence = payload.get("symbol_evidence", [])
        if isinstance(symbol_evidence, list):
            compacted["symbol_evidence"] = self._compact_sequence(
                symbol_evidence,
                limit=5,
                item_max_chars=240,
            )
        resource_manifest = payload.get("resource_manifest", {})
        if isinstance(resource_manifest, dict):
            compacted["resource_manifest_summary"] = {
                "code_only": bool(resource_manifest.get("code_only", False)),
                "resource_keys": [
                    str(key)
                    for key in list(resource_manifest.keys())[:12]
                    if str(key).strip()
                ],
            }
        return {key: value for key, value in compacted.items() if value}

    def _brief_task_input_for_bundle(self, task: dict[str, Any]) -> dict[str, Any]:
        return {
            "task_id": str(task.get("task_id", "") or ""),
            "file_path": str(task.get("file_path", "") or ""),
            "work_package_id": str(task.get("work_package_id", "") or ""),
            "implementation_surfaces": self._compact_sequence(task.get("implementation_surfaces", []), limit=8, item_max_chars=140),
            "method_obligations": self._high_signal_sequence(task.get("method_obligations", []), limit=14, item_max_chars=260),
            "defines_symbols": self._compact_sequence(task.get("defines_symbols", []), limit=10, item_max_chars=100),
            "calls_symbols": self._compact_sequence(task.get("calls_symbols", []), limit=10, item_max_chars=100),
            "writes_artifacts": self._compact_sequence(task.get("writes_artifacts", []), limit=10, item_max_chars=120),
            "review_points": self._high_signal_sequence(task.get("review_points", []), limit=10, item_max_chars=220),
            "paper_evidence_contract": self._compact_paper_evidence_contract_for_codegen(
                task.get("paper_evidence_contract", {})
            ),
        }

    def _compact_snippet_candidate(self, item: Any) -> dict[str, Any]:
        candidate = dict(item or {}) if isinstance(item, dict) else {}
        return {
            "ref_id": str(candidate.get("ref_id", "") or ""),
            "repository_url": str(candidate.get("repository_url", "") or ""),
            "reusable_module": str(candidate.get("reusable_module", "") or ""),
            "insight": self._trim_prompt_text(candidate.get("insight", ""), 420),
            "code_snippet": self._trim_prompt_text(candidate.get("code_snippet", ""), 1200),
            "supported_file_paths": self._compact_sequence(candidate.get("supported_file_paths", []), limit=3),
        }

    def _compact_snippet_candidates(self, items: Any, *, limit: int = 2) -> list[dict[str, Any]]:
        compacted: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for item in list(items or []):
            candidate = self._compact_snippet_candidate(item)
            key = (
                str(candidate.get("ref_id", "") or ""),
                str(candidate.get("reusable_module", "") or ""),
                str(candidate.get("code_snippet", "") or "")[:120],
            )
            if key in seen:
                continue
            seen.add(key)
            compacted.append(candidate)
            if len(compacted) >= max(0, limit):
                break
        return compacted

    def _compact_snippet_candidate_refs(self, items: Any, *, limit: int = 3) -> list[dict[str, Any]]:
        refs: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for item in list(items or []):
            if not isinstance(item, dict):
                continue
            ref = {
                "ref_id": str(item.get("ref_id", "") or ""),
                "repository_url": str(item.get("repository_url", "") or ""),
                "reusable_module": str(item.get("reusable_module", "") or ""),
                "insight": self._trim_prompt_text(item.get("insight", ""), 260),
            }
            key = (ref["ref_id"], ref["reusable_module"])
            if key in seen:
                continue
            seen.add(key)
            refs.append(ref)
            if len(refs) >= max(0, limit):
                break
        return refs

    def _compact_canonical_route(self, route: Any) -> dict[str, Any]:
        payload = dict(route or {}) if isinstance(route, dict) else {}
        return {
            "entry_surface": str(payload.get("entry_surface", "") or ""),
            "stage_sequence": self._compact_sequence(payload.get("stage_sequence", []), limit=6, item_max_chars=120),
            "expected_outputs": self._compact_sequence(payload.get("expected_outputs", []), limit=6, item_max_chars=160),
            "example_invocation": str(payload.get("example_invocation", "") or ""),
        }

    def _compact_task_input(self, task_input: Any) -> dict[str, Any]:
        task = sanitize_task_contract(dict(task_input or {}) if isinstance(task_input, dict) else {})
        return {
            "task_id": str(task.get("task_id", "") or ""),
            "file_path": str(task.get("file_path", "") or ""),
            "work_package_id": str(task.get("work_package_id", "") or ""),
            "dependency_files": self._compact_sequence(task.get("dependency_files", []), limit=8),
            "reference_ids": self._compact_sequence(task.get("reference_ids", []), limit=8),
            "interface_contract": self._high_signal_sequence(task.get("interface_contract", []), limit=16, item_max_chars=360),
            "implementation_surfaces": self._compact_sequence(task.get("implementation_surfaces", []), limit=8, item_max_chars=180),
            "method_obligations": self._high_signal_sequence(task.get("method_obligations", []), limit=28, item_max_chars=520),
            "defines_symbols": self._compact_sequence(task.get("defines_symbols", []), limit=12, item_max_chars=160),
            "calls_symbols": self._compact_sequence(task.get("calls_symbols", []), limit=12, item_max_chars=160),
            "generation_prompt": self._trim_prompt_text(task.get("generation_prompt", ""), 3000),
            "evidence_summary": self._high_signal_sequence(task.get("evidence_summary", []), limit=12, item_max_chars=360),
            "snippet_candidates": self._compact_snippet_candidate_refs(task.get("snippet_candidates", []), limit=3),
            "review_points": self._high_signal_sequence(task.get("review_points", []), limit=12, item_max_chars=260),
            "writes_artifacts": self._compact_sequence(task.get("writes_artifacts", []), limit=18),
            "reads_artifacts": self._compact_sequence(task.get("reads_artifacts", []), limit=6),
            "allowed_scope": self._compact_json_for_prompt(task.get("allowed_scope", {}), max_text_chars=400),
            "canonical_route": self._compact_canonical_route(task.get("canonical_route", {})),
            "work_package_required_files": self._compact_sequence(task.get("work_package_required_files", []), limit=10),
            "work_package_smoke": self._compact_json_for_prompt(task.get("work_package_smoke", {}), max_text_chars=300),
            "paper_claim_inventory": self._compact_paper_claim_inventory(
                task.get("paper_claim_inventory", {}),
                max_items_per_category=24,
            ),
            "paper_claim_closure_items": self._compact_sequence(
                task.get("paper_claim_closure_items", []),
                limit=16,
                item_max_chars=140,
            ),
            "paper_claim_closure_rules": self._compact_sequence(
                task.get("paper_claim_closure_rules", []),
                limit=6,
                item_max_chars=360,
            ),
            "paper_evidence_contract": self._compact_paper_evidence_contract_for_codegen(
                task.get("paper_evidence_contract", {})
            ),
            "prepare_quality_gate_summary": self._compact_json_for_prompt(
                task.get("prepare_quality_gate_summary", {}),
                max_text_chars=800,
            ),
            "generation_context": self._compact_generation_context_for_prompt(task.get("generation_context", {})),
            "critical_grounding_warning": bool(task.get("critical_grounding_warning", False)),
        }

    def _compact_file_plan(self, item: Any) -> dict[str, Any]:
        plan = dict(item or {}) if isinstance(item, dict) else {}
        return {
            "task_id": str(plan.get("task_id", "") or plan.get("target_file", "") or ""),
            "target_file": str(plan.get("target_file", "") or plan.get("path", "") or ""),
            "purpose": self._trim_prompt_text(plan.get("purpose", ""), 400),
            "work_package_id": str(plan.get("work_package_id", "") or ""),
            "depends_on": self._compact_sequence(plan.get("depends_on", []), limit=12),
            "blocking_dependencies": self._compact_sequence(plan.get("blocking_dependencies", []), limit=12),
            "interface_contract": self._high_signal_sequence(plan.get("interface_contract", []), limit=18, item_max_chars=360),
            "implementation_surfaces": self._compact_sequence(plan.get("implementation_surfaces", []), limit=12),
            "method_obligations": self._high_signal_sequence(plan.get("method_obligations", []), limit=30, item_max_chars=520),
            "defines_symbols": self._compact_sequence(plan.get("defines_symbols", []), limit=12),
            "calls_symbols": self._compact_sequence(plan.get("calls_symbols", []), limit=12),
            "writes_artifacts": self._compact_sequence(plan.get("writes_artifacts", []), limit=18),
            "reads_artifacts": self._compact_sequence(plan.get("reads_artifacts", []), limit=8),
            "review_points": self._high_signal_sequence(plan.get("review_points", []), limit=18, item_max_chars=360),
            "generation_prompt": self._trim_prompt_text(plan.get("generation_prompt", ""), 3000),
        }

    def _compact_global_contract(self, global_contract: Any, current_work_package_id: str) -> dict[str, Any]:
        contract = dict(global_contract or {}) if isinstance(global_contract, dict) else {}
        all_package_contracts = [
            item
            for item in list(contract.get("work_package_contracts", []) or [])
            if isinstance(item, dict)
        ]
        package_contracts = [
            item
            for item in all_package_contracts
            if (
                not current_work_package_id
                or str(item.get("work_package_id", "") or "") == current_work_package_id
            )
        ]
        if not package_contracts:
            package_contracts = all_package_contracts[:3]
        cross_package_summary = [
            {
                "work_package_id": str(item.get("work_package_id", "") or ""),
                "produces": self._compact_sequence(item.get("produces", []), limit=5, item_max_chars=180),
                "implementation_surfaces": self._compact_sequence(
                    list(dict(item.get("inventories", {}) or {}).get("implementation_surface_inventory", []) or []),
                    limit=8,
                    item_max_chars=120,
                ),
                "key_obligations": self._high_signal_sequence(
                    item.get("method_obligations", []),
                    limit=4,
                    item_max_chars=240,
                ),
                "artifact_paths": self._compact_sequence(item.get("artifact_paths", []), limit=5, item_max_chars=160),
            }
            for item in all_package_contracts[:20]
        ]
        return {
            "global_invariants": self._compact_sequence(contract.get("global_invariants", []), limit=20),
            "result_targets": self._compact_json_for_prompt(
                list(contract.get("result_targets", []) or [])[:40],
                max_text_chars=500,
            ),
            "work_package_contracts": self._compact_json_for_prompt(package_contracts[:8], max_text_chars=1200),
            "cross_package_contract_summary": self._compact_json_for_prompt(cross_package_summary, max_text_chars=600),
            "inventories": self._compact_json_for_prompt(contract.get("inventories", {}), max_text_chars=2800),
            "inventory_owners": self._compact_json_for_prompt(contract.get("inventory_owners", {}), max_text_chars=1800),
        }

    def _compact_project_plan(self, project_plan: dict[str, Any], generation_manifest: dict[str, Any]) -> dict[str, Any]:
        manifest = generation_manifest if isinstance(generation_manifest, dict) else {}
        current_task = self._compact_task_input(manifest.get("current_task_input", {}))
        current_work_package_id = str(current_task.get("work_package_id", "") or "")
        runtime_contract = project_plan.get("runtime_contract", {}) if isinstance(project_plan, dict) else {}
        current_file_plan = runtime_contract.get("current_task_file_plan", {})
        neighbor_file_plans = list(runtime_contract.get("current_task_neighbor_file_plans", []) or [])[:6]
        repo_file_index = [
            self._compact_file_plan(item)
            for item in list(runtime_contract.get("repo_plan_files", []) or [])[:80]
            if isinstance(item, dict)
        ]
        return {
            "project_type": str(project_plan.get("project_type", "") or ""),
            "summary": self._trim_prompt_text(project_plan.get("summary", ""), 1200),
            "entrypoints": self._compact_json_for_prompt(project_plan.get("entrypoints", {}), max_text_chars=400),
            "artifact_contract": self._compact_json_for_prompt(project_plan.get("artifact_contract", {}), max_text_chars=500),
            "file_specs": self._compact_json_for_prompt(list(project_plan.get("file_specs", []) or [])[:8], max_text_chars=240),
            "runtime_contract": {
                "output_dir": str(runtime_contract.get("output_dir", "") or ""),
                "artifacts_dir": str(runtime_contract.get("artifacts_dir", "") or ""),
                "canonical_route": self._compact_json_for_prompt(runtime_contract.get("canonical_route", {}), max_text_chars=500),
                "stage_public_surfaces": self._compact_json_for_prompt(
                    list(runtime_contract.get("stage_public_surfaces", []) or [])[:8],
                    max_text_chars=500,
                ),
                "artifact_contract": self._compact_json_for_prompt(
                    list(runtime_contract.get("artifact_contract", []) or [])[:8],
                    max_text_chars=500,
                ),
                "result_targets": self._compact_json_for_prompt(
                    list(runtime_contract.get("result_targets", []) or [])[:8],
                    max_text_chars=600,
                ),
                "global_contract": self._compact_global_contract(
                    runtime_contract.get("global_contract", {}),
                    current_work_package_id,
                ),
                "current_task_file_plan": self._compact_file_plan(current_file_plan),
                "current_task_neighbor_file_plans": [
                    self._compact_file_plan(item)
                    for item in neighbor_file_plans
                ],
                "repo_file_index": repo_file_index,
                "generation_tasks": [current_task],
            },
        }

    def _compact_result_target(self, item: Any) -> dict[str, Any]:
        target = dict(item or {}) if isinstance(item, dict) else {}
        return {
            "target_id": str(target.get("target_id", "") or ""),
            "kind": str(target.get("kind", "") or ""),
            "name": self._trim_prompt_text(target.get("name", ""), 180),
            "owner_work_packages": self._compact_sequence(target.get("owner_work_packages", []), limit=4),
            "artifact_paths": self._compact_sequence(target.get("artifact_paths", []), limit=6),
        }

    def _compact_neighbor_file_plan(self, item: Any) -> dict[str, Any]:
        plan = dict(item or {}) if isinstance(item, dict) else {}
        return {
            "task_id": str(plan.get("task_id", "") or plan.get("target_file", "") or ""),
            "target_file": str(plan.get("target_file", "") or plan.get("path", "") or ""),
            "work_package_id": str(plan.get("work_package_id", "") or ""),
            "purpose": self._trim_prompt_text(plan.get("purpose", ""), 160),
            "writes_artifacts": self._compact_sequence(plan.get("writes_artifacts", []), limit=4),
        }

    def _generation_bundle(self, project_plan: dict[str, Any], generation_manifest: dict[str, Any]) -> dict[str, Any]:
        manifest = generation_manifest if isinstance(generation_manifest, dict) else {}
        runtime_contract = project_plan.get("runtime_contract", {}) if isinstance(project_plan, dict) else {}
        current_task_input = self._compact_task_input(manifest.get("current_task_input", {}))
        neighbor_file_plans = list(runtime_contract.get("current_task_neighbor_file_plans", []) or [])[:6]
        global_contract = dict(runtime_contract.get("global_contract", {}) or {})
        repo_file_index = [
            self._compact_file_index_plan(item)
            for item in list(runtime_contract.get("repo_plan_files", []) or [])[:80]
            if isinstance(item, dict)
        ]
        return {
            "project_summary": {
                "project_type": str(project_plan.get("project_type", "") or ""),
                "summary": self._trim_prompt_text(project_plan.get("summary", ""), 500),
                "entrypoints": self._compact_json_for_prompt(project_plan.get("entrypoints", {}), max_text_chars=300),
            },
            "global_contract": self._compact_global_contract(
                global_contract,
                str(current_task_input.get("work_package_id", "") or ""),
            ),
            "result_targets": [
                self._compact_result_target(item)
                for item in list(runtime_contract.get("result_targets", []) or [])[:12]
                if isinstance(item, dict)
            ],
            "current_task_input": current_task_input,
            "current_task_file_plan": {
                "target_file": str(
                    dict(runtime_contract.get("current_task_file_plan", {}) or {}).get("target_file", "")
                    or current_task_input.get("file_path", "")
                    or ""
                ),
                "task_id": str(current_task_input.get("task_id", "") or ""),
                "work_package_id": str(current_task_input.get("work_package_id", "") or ""),
            },
            "current_task_neighbor_file_plans": [
                self._compact_neighbor_file_plan(item)
                for item in neighbor_file_plans[:6]
            ],
            "repo_file_index": repo_file_index,
            "paper_claim_closure": {
                "items": self._compact_sequence(
                    current_task_input.get("paper_claim_closure_items", []),
                    limit=24,
                    item_max_chars=140,
                ),
                "rules": self._compact_sequence(
                    current_task_input.get("paper_claim_closure_rules", []),
                    limit=6,
                    item_max_chars=360,
                ),
            },
            "review_points": manifest.get("review_points", [])[:16],
        }

    def _repo_edit_generation_bundle(self, project_plan: dict[str, Any], generation_manifest: dict[str, Any]) -> dict[str, Any]:
        manifest = generation_manifest if isinstance(generation_manifest, dict) else {}
        runtime_contract = project_plan.get("runtime_contract", {}) if isinstance(project_plan, dict) else {}
        current_task_input = self._compact_task_input(manifest.get("current_task_input", {}))
        global_contract = dict(runtime_contract.get("global_contract", {}) or {})
        inventories = dict(global_contract.get("inventories", {}) or {})
        current_file_plan = self._compact_file_plan(runtime_contract.get("current_task_file_plan", {}))
        neighbor_file_plans = [
            self._compact_neighbor_file_plan(item)
            for item in list(runtime_contract.get("current_task_neighbor_file_plans", []) or [])[:4]
            if isinstance(item, dict)
        ]
        repo_plan_files = [
            self._compact_neighbor_file_plan(item)
            for item in list(runtime_contract.get("repo_plan_files", []) or [])[:40]
            if isinstance(item, dict)
        ]
        return {
            "project_summary": {
                "project_type": str(project_plan.get("project_type", "") or ""),
                "summary": self._trim_prompt_text(project_plan.get("summary", ""), 700),
                "entrypoints": self._compact_json_for_prompt(project_plan.get("entrypoints", {}), max_text_chars=400),
            },
            "target_file": str(current_task_input.get("file_path", "") or ""),
            "current_task": self._brief_task_input_for_bundle(current_task_input),
            "current_file_plan": current_file_plan,
            "neighbor_file_plans": neighbor_file_plans,
            "repo_file_index": repo_plan_files,
            "canonical_route": self._compact_json_for_prompt(runtime_contract.get("canonical_route", {}), max_text_chars=900),
            "stage_public_surfaces": self._compact_json_for_prompt(
                list(runtime_contract.get("stage_public_surfaces", []) or [])[:8],
                max_text_chars=500,
            ),
            "result_targets": [
                self._compact_result_target(item)
                for item in list(runtime_contract.get("result_targets", []) or [])[:12]
                if isinstance(item, dict)
            ],
            "global_invariants": self._compact_sequence(
                global_contract.get("global_invariants", []),
                limit=14,
                item_max_chars=220,
            ),
            "contract_inventories": {
                "methods": self._compact_sequence(
                    inventories.get("methods", []),
                    limit=20,
                    item_max_chars=120,
                ),
                "datasets": self._compact_sequence(
                    inventories.get("datasets", []),
                    limit=16,
                    item_max_chars=120,
                ),
                "metrics": self._compact_sequence(
                    inventories.get("metrics", []),
                    limit=16,
                    item_max_chars=120,
                ),
                "artifacts": self._compact_sequence(
                    inventories.get("artifacts", []),
                    limit=24,
                    item_max_chars=160,
                ),
            },
            "ordered_tasks": self._compact_sequence(manifest.get("ordered_tasks", []), limit=40, item_max_chars=120),
            "review_points": self._high_signal_sequence(manifest.get("review_points", []), limit=12, item_max_chars=300),
        }

    def _build_repo_generation_prompt(
        self,
        plan: str,
        target: str,
        project_plan: dict[str, Any],
        generation_manifest: dict[str, Any],
        iteration_context: dict[str, Any] | None,
        *,
        repo_edit_mode: bool,
        run_memory_text: str = "",
    ) -> str:
        current_task_input = self._compact_task_input(generation_manifest.get("current_task_input", {}))
        current_file_plan = dict(
            (project_plan.get("runtime_contract", {}) if isinstance(project_plan, dict) else {}).get("current_task_file_plan", {}) or {}
        )
        current_file_path = str(current_task_input.get("file_path", "") or "").strip()
        bundle_payload = (
            self._repo_edit_generation_bundle(project_plan, generation_manifest)
            if repo_edit_mode
            else self._generation_bundle(project_plan, generation_manifest)
        )
        bundle_json = json.dumps(
            bundle_payload,
            ensure_ascii=False,
            indent=2,
        )
        return_mode_instruction = (
            "Edit files in-place under the current working directory. The filesystem is the source of truth."
            if repo_edit_mode
            else (
                f"Return the full content for `{current_file_path}`. If this task is an active route closure and "
                "a dependent helper file must also be touched, return JSON with an `updated_files` object. "
                "Otherwise return one fenced code block for the current file. Do not include explanatory prose."
            )
        )
        prompt_parts = [
            "You are implementing a faithful, complete, judgeable reproduction repository for reproagent.",
            return_mode_instruction,
            "Work inside one evolving canonical repository. Treat the current task as faithful completion work for a judgeable repo.",
            "Implement the paper-owned route as concrete code/config/artifact writers with bounded execution defaults.",
            "Close the current file's package obligations in a way that keeps the whole repo moving toward runnable canonical-route closure.",
            "Treat `implementation_surfaces` and `method_obligations` as mandatory code requirements for the current file, not as optional commentary.",
            "If any legacy task contract asks for schema-complete outputs, registry/declaration hooks, dry-run-only artifacts, or config echoes, reinterpret that as a requirement for executable measured routes: constants/defaults, factories/selectors, metric functions, artifact writer functions, and entrypoint calls.",
            "When `snippet_candidates` are present, adapt the relevant reference evidence into real code/config/registry logic or document an incompatibility in the file. Preserve a machine-readable grounding marker such as `reference_grounding: <ref_id> <source_path>` near the adapted implementation or registry entry.",
            "For reproduction papers, implement the paper-derived method, environment/data interfaces, policy/model adapters, training or pretraining loops, evaluation metrics, ablation/refinement variants, config, and artifact-writing surfaces that appear in the current task contract.",
            "Paper-derived obligations must land in executable or importable code/config paths: dataset obtain/prepare/validate functions, model loader/factory functions, metric formulas/aggregation, attack/adaptation algorithms, training/evaluation loops, per-sample bookkeeping, and table/figure artifact writers as applicable.",
            "If the current task names formulas, numeric constants, sweep values, hyperparameter defaults, algorithm steps, section/table/figure routes, or metric definitions, implement them as concrete constants/dataclasses/functions/classes and call them from the canonical route. Mentions in comments, README text, JSON schemas, registry rows, or report templates are not sufficient.",
            "Do not invent public function/class names from generic contract prose such as `must be implemented`, `selector set must include`, or `executable-route acceptance`. Public symbols should be paper/method nouns or stable route names that the entrypoint actually imports or calls.",
            "A README table, manifest, or protocol schema is useful only as an index; it is not enough unless the referenced implementation path exists and the canonical runner/config can reach it.",
            "Satisfy algorithmic and paper-visible obligations with concrete modules, measured values, implemented metrics, code-backed outputs, and the same data/model/method/metric/artifact path that full mode would use.",
            "If this file owns a route, explicitly wire calls to its dependencies and artifact writers. If this file defines a helper, ensure an entrypoint, experiment runner, training/evaluation route, or figure/table writer imports or calls it through the plan's `calls_symbols` contract.",
            "Use `generation_context.paper_chunks`, `generation_context.reference_surveys`, `generation_context.symbol_evidence`, and `generation_context.resource_manifest` as task-local source evidence. These are compact anchors from prepare; preserve referenced external backend hooks unless the task explicitly marks the backend unavailable, and even then keep the lazy import/factory hook.",
            "For external simulators, datasets, model checkpoints, or libraries named in the task context, implement a lazy import/load factory plus an availability check and full-mode route. A synthetic local fixture may only exercise smoke mode through the same interface; it must not be the sole implementation.",
            "Implement experiments as hypothesis- and decision-value-driven code paths. Use the task contract and scope_boundary.preserve to choose the required implementation routes; represent repeated variants through executable selectors when they are paper-visible.",
            "Make the selected experiment set explicit in executable code/config: core contribution hypothesis, decisive comparison, decisive metric function, and positive implementation-scope boundary for bounded execution.",
            "When repeated variants are benchmark-visible, expose them through selectors backed by executable config defaults/constants; only execute the bounded smoke/default subset unless full mode is explicitly requested.",
            "If an obligation is heavy to execute, still provide importable code, configuration, and runnable orchestration entrypoints; do not claim results were run.",
            "If faithful implementation is impossible with the provided evidence, leave the affected file or route clearly incomplete through a validation-facing failure.",
            "Make benchmark-visible contracts code-first: constants/defaults, callable task/config selectors, named experiment routes, method/baseline factories, metric formula functions, artifact writer functions, and canonical route call sites. Docs should only index those code paths.",
            "For repo-edit mode, create or modify the target file directly before summarizing. A summary without an on-disk changed target file is a failed attempt.",
            "Do not use TDD or pre-implementation failing-test workflows for this generation task. Implement the target file first, then run only lightweight compile/import/smoke checks after editing.",
            "Do not read local development-process skills or methodology documents unless the target file itself depends on them; focus on paper evidence and repository code.",
            self._task_scope_instruction_block(current_task_input, current_file_plan),
            "Default commands must be safe smoke/dry-run paths that validate wiring by calling the real implementation surfaces with bounded inputs; full training/evaluation should require an explicit mode or config.",
            "The default smoke/dry-run path may create readiness/manifests plus `readiness.json` and `evaluation_result.json`; it must not create benchmark-visible table/figure/metric/prediction/report files as schema-only result shells. Paper-visible outputs must come from bounded measured code paths or explicit full-mode commands.",
            "Dry-run readiness manifests are allowed only as auxiliary smoke artifacts. Method, environment, policy/model, training, refinement, baseline, metric, and evaluation obligations need code-backed implementation routes.",
            "For Python files, do not import optional simulator, RL, GPU, plotting, or dataset packages at module top level. Put imports such as gym/gymnasium, torch, stable_baselines3, metadrive, CybORG, pandas, matplotlib, and sklearn inside the functions/classes that need them, or guard them with importlib-based availability checks plus lightweight local fallbacks.",
            "Keep optional simulator, RL, GPU, or dataset dependencies behind lazy imports and availability checks so static import and smoke review still work in a minimal environment.",
            "Every planned file should contain faithful implementation content or a validation-facing incomplete route when evidence is insufficient.",
            "Do not bypass unresolved contract failures by deleting surfaces, shrinking interfaces, or silently dropping declared outputs.",
            f"Target:\n{target}",
            f"Plan excerpt:\n{plan[:1800]}",
            "Current task contract:\n" + json.dumps(current_task_input, ensure_ascii=False, indent=2),
            "Current file plan prompt:\n" + str(current_file_plan.get("generation_prompt", "") or ""),
            f"Structured generation bundle:\n{bundle_json}",
            self._artifact_instruction_block(),
        ]
        if iteration_context:
            runtime_first_blockers = [
                item
                for item in list(iteration_context.get("runtime_first_blockers", []) or [])
                if isinstance(item, dict)
            ]
            if runtime_first_blockers:
                prompt_parts.insert(
                    3,
                    "Runtime-first blockers for this repair round:\n"
                    + json.dumps(runtime_first_blockers, ensure_ascii=False, indent=2)
                    + "\nThese exact startup/import/runtime blockers are mandatory for this task when the "
                    "current file or its direct dependencies appear in file_refs. Repair the producer/consumer "
                    "mismatch first, then preserve the broader paper semantics."
                )
            score_feedback = dict(iteration_context.get("score_feedback", {}) or {})
            if score_feedback:
                prompt_parts.insert(
                    4,
                    "Judge score feedback for this repair round:\n"
                    + json.dumps(score_feedback, ensure_ascii=False, indent=2)
                    + "\nPrioritize these valid-score low/zero rubric leaves when they touch the current file or its package. "
                    "Invalid-score leaves are excluded because they are judge/extraction failures, not reliable repair targets. "
                    "Repair them with executable code/config/constants/call sites, not README prose or manifest-only declarations."
                )
            initial_runtime_execution_result = dict(
                iteration_context.get("initial_runtime_execution_result", {}) or {}
            )
            if initial_runtime_execution_result and initial_runtime_execution_result != dict(
                iteration_context.get("execution_result", {}) or {}
            ):
                prompt_parts.append(
                    "Original repo-validation execution result for this repair round:\n"
                    + json.dumps(
                        {
                            "success": bool(initial_runtime_execution_result.get("success", False)),
                            "exit_code": initial_runtime_execution_result.get("exit_code", 1),
                            "error": str(initial_runtime_execution_result.get("error", "") or "")[:3000],
                            "output": str(initial_runtime_execution_result.get("output", "") or "")[-1200:],
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
            previous_files = dict(iteration_context.get("previous_files", {}) or {})
            if previous_files:
                prompt_parts.append("Current repository snapshot:\n" + self._render_project_files_for_prompt(previous_files))
            execution_result = dict(iteration_context.get("execution_result", {}) or {})
            if execution_result:
                stderr = str(execution_result.get("error", "") or "").strip()
                stdout = str(execution_result.get("output", "") or "").strip()
                if stderr:
                    prompt_parts.append(f"Previous stderr:\n{stderr[:3000]}")
                if stdout:
                    prompt_parts.append(f"Previous stdout:\n{stdout[:3000]}")
            suggestions = [str(item).strip() for item in iteration_context.get("suggestions", []) or [] if str(item).strip()]
            if suggestions:
                prompt_parts.append("Follow-up suggestions:\n" + "\n".join(f"- {item}" for item in suggestions[:10]))
        if run_memory_text:
            prompt_parts.append(run_memory_text)
        if repo_edit_mode:
            prompt_parts.append("End with a concise summary of the files you changed and why.")
        else:
            prompt_parts.append(
                f"Return `{current_file_path}` as a complete file, or JSON `updated_files` if route closure requires touching directly related helper files."
            )
        return "\n\n".join(part for part in prompt_parts if part)

    def _extract_code_block(self, text: str, target_file: str = "") -> str:
        cleaned = text.strip()
        if not cleaned:
            return ""
        target_suffix = Path(target_file).suffix.lower()
        is_python_target = target_suffix == ".py" or target_file.endswith("/__init__.py")
        if "```" not in cleaned:
            if is_python_target:
                try:
                    ast.parse(cleaned)
                    return cleaned
                except SyntaxError:
                    return ""
            return cleaned
        parts = cleaned.split("```")
        for part in parts:
            candidate = part.strip()
            first_line, _, rest = candidate.partition("\n")
            if rest and re.fullmatch(r"[A-Za-z0-9_+.-]+", first_line.strip()):
                candidate = rest.lstrip()
            elif candidate.startswith("python"):
                candidate = candidate[len("python"):].lstrip()
            if not candidate:
                continue
            if is_python_target:
                try:
                    ast.parse(candidate)
                    return candidate
                except SyntaxError:
                    continue
            if _looks_like_non_python_file(candidate, target_file):
                return candidate
        return ""

    def _parse_updated_files(
        self,
        response: str,
        project_plan: dict[str, Any],
        task_manifest: dict[str, Any],
    ) -> dict[str, str]:
        cleaned = str(response or "").strip()
        if not cleaned:
            return {}
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if len(lines) >= 3:
                cleaned = "\n".join(lines[1:-1]).strip()

        def _files_from_payload(payload: Any) -> dict[str, str]:
            if isinstance(payload, dict):
                updated_files = payload.get("updated_files") or payload.get("project_files")
                if isinstance(updated_files, dict):
                    parsed_files = {
                        str(path).strip(): str(content)
                        for path, content in updated_files.items()
                        if str(path).strip() and isinstance(content, str)
                    }
                    if parsed_files:
                        return parsed_files
                file_path = str(
                    payload.get("file_path")
                    or payload.get("path")
                    or payload.get("target_file")
                    or ""
                ).strip()
                content = payload.get("content")
                if file_path and isinstance(content, str):
                    return {file_path: content}
            if isinstance(payload, list):
                collected: dict[str, str] = {}
                for item in payload:
                    collected.update(_files_from_payload(item))
                return collected
            return {}

        def _try_json_payload(text: str) -> dict[str, str]:
            try:
                return _files_from_payload(json.loads(text))
            except Exception:
                return {}

        parsed = _try_json_payload(cleaned)
        if parsed:
            return parsed

        for match in re.finditer(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE):
            parsed = _try_json_payload(match.group(1).strip())
            if parsed:
                return parsed

        decoder = json.JSONDecoder()
        for index, char in enumerate(cleaned):
            if char not in "[{":
                continue
            try:
                payload, _ = decoder.raw_decode(cleaned[index:])
            except Exception:
                continue
            parsed = _files_from_payload(payload)
            if parsed:
                return parsed

        task_inputs = list(task_manifest.get("task_inputs", []) or [])
        target_file = ""
        if task_inputs and isinstance(task_inputs[0], dict):
            target_file = str(task_inputs[0].get("file_path", "") or "").strip()
        if not target_file:
            target_file = self._project_entrypoints(project_plan)["main"]
        code = self._extract_code_block(cleaned, target_file)
        return {target_file: code} if code else {}

    def _build_repo_agent_context(self, repo_dir: Path) -> dict[str, Any]:
        artifact_dir = self._node_dir() / "artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        return {
            "cwd": str(repo_dir.resolve()),
            "env": {
                "PAPERBENCH_REPRO_ARTIFACT_DIR": str(artifact_dir.resolve()),
            },
        }

    def _build_generation_agent(self) -> Any:
        mode = str(getattr(self.codegen_config, "mode", "llm_agent") or "llm_agent")
        if mode in {"codex_cli_collaborative", "collaborative"}:
            return CodexWrapper(
                self.codegen_config.codex_cli_path,
                model=getattr(self.codegen_config, "codex_model", None),
                model_provider=getattr(self.codegen_config, "codex_model_provider", None),
                base_url=getattr(self.codegen_config, "codex_base_url", None),
                reasoning_effort=getattr(self.codegen_config, "codex_reasoning_effort", None),
            )
        if mode == "claude_only":
            return ClaudeSDKWrapper(
                self.codegen_config.claude_cli_path,
                model=getattr(self.codegen_config, "claude_model", None),
                effort=getattr(self.codegen_config, "claude_effort", None),
            )
        return LLMWorkflowAgent(node_name="file_generation")

    def _build_repair_agent(self) -> Any:
        mode = str(
            getattr(self.codegen_config, "repair_mode", "") or getattr(self.codegen_config, "mode", "llm_agent") or "llm_agent"
        )
        if mode in {"codex_cli_collaborative", "collaborative"}:
            return CodexWrapper(
                self.codegen_config.codex_cli_path,
                model=getattr(self.codegen_config, "codex_model", None),
                model_provider=getattr(self.codegen_config, "codex_model_provider", None),
                base_url=getattr(self.codegen_config, "codex_base_url", None),
                reasoning_effort=getattr(self.codegen_config, "codex_reasoning_effort", None),
            )
        if mode == "claude_only":
            return ClaudeSDKWrapper(
                self.codegen_config.claude_cli_path,
                model=getattr(self.codegen_config, "claude_model", None),
                effort=getattr(self.codegen_config, "claude_effort", None),
            )
        return LLMWorkflowAgent(node_name="file_generation")

    def _generate_project_from_plan(
        self,
        plan: str,
        target: str,
        project_plan: dict[str, Any],
        task_manifest: dict[str, Any],
        iteration_context: dict[str, Any] | None,
        iteration: int = 0,
        attempt_index: int = 0,
    ) -> dict[str, Any]:
        previous_files = dict((iteration_context or {}).get("previous_files", {}) or {})
        task_inputs = list(task_manifest.get("task_inputs", []) or [])
        task_id = ""
        if task_inputs and isinstance(task_inputs[0], dict):
            task_id = str(task_inputs[0].get("task_id", "") or "").strip()
        if not task_id:
            task_id = f"task_{iteration}"
        task_dir = self._task_dir(iteration, task_id)
        task_dir.mkdir(parents=True, exist_ok=True)

        repo_dir = self._repo_dir()
        repo_dir.mkdir(parents=True, exist_ok=True)
        if previous_files and not any(repo_dir.iterdir()):
            save_project_files(previous_files, repo_dir)

        agent = self._build_generation_agent()
        repo_edit_mode = isinstance(agent, CodexWrapper)
        memory_state = iteration_context.get("state") if isinstance(iteration_context, dict) else None
        prompt = self._build_repo_generation_prompt(
            plan,
            target,
            project_plan,
            task_manifest,
            iteration_context,
            repo_edit_mode=repo_edit_mode,
            run_memory_text=(
                run_memory.get_run_memory_prompt(memory_state, max_chars=4000)
                if isinstance(memory_state, PaperBenchReproState)
                else ""
            ),
        )

        if self.output_dir is not None:
            (task_dir / "current_prompt.md").write_text(prompt, encoding="utf-8")

        if isinstance(agent, CodexWrapper):
            result = agent.execute_best_effort(
                prompt,
                self._build_repo_agent_context(repo_dir),
                output_mode="text",
                timeout=int(getattr(self.workflow_config, "docker_validate_timeout", 1800) or 1800),
            )
            project_files = load_project_files(repo_dir)
            if not project_files:
                fallback_updates = self._parse_updated_files(
                    result.get("output", "") or result.get("raw_output", ""),
                    project_plan,
                    task_manifest,
                )
                if fallback_updates:
                    project_files = merge_project_files(previous_files, fallback_updates)
                    save_project_files(project_files, repo_dir)
            self._save_agent_trace(
                iteration=iteration,
                task_id=task_id,
                attempt_index=attempt_index,
                prompt=prompt,
                result=result,
                project_files_before=previous_files,
                project_files_after=project_files,
            )
            updated_files = self._diff_project_files(previous_files, project_files)
            context_usage = self._build_task_usage(prompt, result)
            return {
                "project_files": project_files,
                "updated_files": updated_files,
                "agent_result": result,
                "task_id": task_id,
                "context_usage": context_usage,
            }

        response = agent.execute(prompt, self._build_repo_agent_context(repo_dir), output_mode="text")
        updated_files = self._parse_updated_files(response, project_plan, task_manifest)
        agent_result = {
            "success": bool(updated_files),
            "output": response,
            "raw_output": response,
            "error": "" if updated_files else "No files were produced by the generator.",
            "usage": {},
        }
        project_files = merge_project_files(previous_files, updated_files)
        save_project_files(project_files, repo_dir)
        return {
            "project_files": project_files,
            "updated_files": updated_files,
            "agent_result": agent_result,
            "task_id": task_id,
            "context_usage": self._build_task_usage(prompt, agent_result),
        }

    def _build_repo_repair_prompt(
        self,
        *,
        target: str,
        requirement_anchor: dict[str, Any],
        repair_eval_report: dict[str, Any],
        repair_plan: dict[str, Any],
        iteration_context: dict[str, Any] | None,
        run_memory_text: str = "",
    ) -> str:
        repo_dir = self._repo_dir().resolve()
        anchor_goal = str(requirement_anchor.get("goal", "") or target or "").strip()
        anchor_summary = str(requirement_anchor.get("summary", "") or "").strip()
        semantic_invariants = [
            str(item).strip()
            for item in list(requirement_anchor.get("semantic_invariants", []) or [])
            if str(item).strip()
        ][:8]
        runtime_invariants = [
            str(item).strip()
            for item in list(requirement_anchor.get("runtime_invariants", []) or [])
            if str(item).strip()
        ][:8]
        acceptance_signals = [
            str(item).strip()
            for item in list(requirement_anchor.get("acceptance_signals", []) or [])
            if str(item).strip()
        ][:8]
        top_findings = []
        for item in list(repair_eval_report.get("findings", []) or [])[:6]:
            if not isinstance(item, dict):
                continue
            top_findings.append(
                {
                    "finding_id": str(item.get("finding_id", "") or ""),
                    "category": str(item.get("category", "") or ""),
                    "severity": str(item.get("severity", "") or ""),
                    "summary": str(item.get("summary", "") or ""),
                    "affected_surfaces": list(item.get("affected_surfaces", []) or [])[:6],
                    "suggested_focus": list(item.get("suggested_focus", []) or [])[:6],
                    "assertion_ids": list(item.get("assertion_ids", []) or [])[:6],
                    "failure_layer": str(item.get("failure_layer", "") or ""),
                    "fix_hint": str(item.get("fix_hint", "") or ""),
                }
            )
        recommended_surfaces = [
            str(item).strip()
            for item in list(repair_plan.get("recommended_surfaces", []) or [])
            if str(item).strip()
        ][:16]
        repair_guidance = [
            str(item).strip()
            for item in list(repair_plan.get("repair_guidance", []) or [])
            if str(item).strip()
        ][:10]
        review_points = [
            str(item).strip()
            for item in list(repair_plan.get("review_points", []) or [])
            if str(item).strip()
        ][:10]
        forbidden_shortcuts = [
            str(item).strip()
            for item in list(repair_plan.get("forbidden_shortcuts", []) or [])
            if str(item).strip()
        ][:8]
        repo_file_index = []
        if iteration_context:
            previous_files = dict(iteration_context.get("previous_files", {}) or {})
            repo_file_index = sorted(previous_files.keys())[:60]
        prompt_parts = [
            "You are running one repo-wide repair pass for reproagent.",
            "Edit files in-place under the current working directory. The filesystem is the source of truth.",
            "You are repairing an already materialized faithful reproduction repository.",
            "You may inspect, search, and modify any repository files needed to fix the current repo-level problems.",
            "Preserve the existing implementation by default. Make the smallest direct patch that fixes the reported failure.",
            "Do not replace large, high-information modules with compact rewrites, smoke-only scripts, or simplified reimplementations.",
            "Do not delete paper-specific constants, method classes, environment routes, metric formulas, artifact writers, or public exports that are unrelated to the exact failure.",
            "Preserve or restore real runnable closure with code-backed artifacts and retained interfaces.",
            "When validation reports missing paper implementation paths, repair the executable code/config surface itself: dataset prepare/validate, model loader/factory, metric formula, attack/adaptation, training/evaluation loop, per-sample bookkeeping, and artifact writer paths as applicable.",
            "When validation reports missing formulas, numeric constants, default hyperparameters, algorithm steps, or active routes, repair concrete constants/dataclasses/functions/classes and their canonical-route call sites.",
            "Do not create public functions/classes from generic validation prose such as `must be implemented`, `selector set must include`, or `executable-route acceptance`; use paper/method names and wire them through the canonical route.",
            "Reference-grounded repair should adapt the relevant snippet/protocol pattern into local code and keep a `reference_grounding: <ref_id> <source_path>` marker near that implementation.",
            f"Repository root: {repo_dir}",
            f"Target:\n{target}",
            "Repair problem index:\n"
            + json.dumps(
                {
                    "summary": str(repair_eval_report.get("summary", "") or ""),
                    "semantic_status": str(repair_eval_report.get("semantic_status", "") or ""),
                    "runtime_status": str(repair_eval_report.get("runtime_status", "") or ""),
                    "anti_shortcut_status": str(repair_eval_report.get("anti_shortcut_status", "") or ""),
                    "top_findings": top_findings,
                    "recommended_surfaces": recommended_surfaces,
                    "repair_guidance": repair_guidance,
                    "review_points": review_points,
                },
                ensure_ascii=False,
                indent=2,
            ),
            "Focused repo-wide repair rule:\n"
            + (
                "The recommended_surfaces list is the required patch set for this repair round. "
                "Inspect and update those surfaces together, including directly coupled helper files only when needed for imports/runtime closure. "
                "Patch only the failing functions, imports, exports, call sites, or artifact-writer paths. "
                "Do not spend this round rewriting unrelated configuration or dependency files unless they are necessary for the recommended surfaces to run. "
                "Whole-file replacement is allowed only for tiny files or genuinely missing files; otherwise preserve existing symbols and paper-specific anchors. "
                "If the execution environment cannot persist edits directly, return JSON with an `updated_files` object containing every changed file."
            ),
            "Repository file index:\n" + json.dumps(repo_file_index, ensure_ascii=False, indent=2),
            self._artifact_instruction_block(),
        ]
        if not semantic_anchor_disabled():
            prompt_parts.insert(
                4,
                "Use the requirement anchor as the stable semantic baseline. Do not drift away from the original task semantics.",
            )
            prompt_parts.insert(
                16,
                "Requirement anchor:\n"
                + json.dumps(
                    {
                        "goal": anchor_goal,
                        "summary": anchor_summary,
                        "semantic_invariants": semantic_invariants,
                        "runtime_invariants": runtime_invariants,
                        "acceptance_signals": acceptance_signals,
                        "forbidden_shortcuts": forbidden_shortcuts
                        or list(requirement_anchor.get("forbidden_shortcuts", []) or [])[:8],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            )
        if iteration_context:
            previous_review = dict(iteration_context.get("previous_review", {}) or {})
            if previous_review:
                prompt_parts.append(
                    "Previous repo-level review summary:\n"
                    + json.dumps(
                        {
                            "passed": bool(previous_review.get("passed", False)),
                            "repo_level_review": previous_review.get("repo_level_review", {}),
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
            validation_report = dict(iteration_context.get("validation_report", {}) or {})
            if validation_report:
                prompt_parts.append(
                    "Current validation snapshot:\n"
                    + json.dumps(
                        {
                            "passed": bool(validation_report.get("passed", False)),
                            "overall_status": str(validation_report.get("overall_status", "") or ""),
                            "failure_categories": list(validation_report.get("failure_categories", []) or [])[:6],
                            "blocked_reasons": list(validation_report.get("blocked_reasons", []) or [])[:6],
                            "repair_recommendations": list(validation_report.get("repair_recommendations", []) or [])[:6],
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
            execution_result = dict(iteration_context.get("execution_result", {}) or {})
            runtime_first_blockers = [
                item
                for item in list(iteration_context.get("runtime_first_blockers", []) or [])
                if isinstance(item, dict)
            ]
            if runtime_first_blockers:
                prompt_parts.append(
                    "Runtime-first blockers for this round:\n"
                    + json.dumps(
                        runtime_first_blockers,
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\nRepair these exact startup/import/syntax blockers before broad semantic or artifact work."
                )
            score_feedback = dict(iteration_context.get("score_feedback", {}) or {})
            if score_feedback:
                prompt_parts.append(
                    "Judge score feedback for this round:\n"
                    + json.dumps(score_feedback, ensure_ascii=False, indent=2)
                    + "\nTreat these valid-score low/zero rubric leaves as high-priority semantic repair targets. "
                    "Do not chase invalid-score judge failures as implementation requirements. "
                    "Patch the concrete implementation surfaces that can satisfy them; do not spend the pass only polishing route declarations."
                )
            elif execution_result:
                prompt_parts.append(
                    "Current execution result excerpt:\n"
                    + json.dumps(
                        {
                            "success": bool(execution_result.get("success", False)),
                            "exit_code": execution_result.get("exit_code", 1),
                            "error": str(execution_result.get("error", "") or "")[:1800],
                            "output": str(execution_result.get("output", "") or "")[-1800:],
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
            repair_ticket = dict(iteration_context.get("repair_ticket", {}) or {})
            if repair_ticket:
                prompt_parts.append(
                    "Current repair ticket:\n"
                    + json.dumps(
                        {
                            "failure_type": str(repair_ticket.get("failure_type", "") or ""),
                            "reason": str(repair_ticket.get("reason", "") or ""),
                            "trigger_signals": list(repair_ticket.get("trigger_signals", []) or [])[:6],
                            "required_fix_targets": list(repair_ticket.get("required_fix_targets", []) or [])[:10],
                            "next_fix_scope": list(repair_ticket.get("next_fix_scope", []) or [])[:10],
                            "evidence": dict(repair_ticket.get("evidence", {}) or {}),
                            "forbidden_changes": list(repair_ticket.get("forbidden_changes", []) or [])[:8],
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
            suggestions = [
                str(item).strip()
                for item in iteration_context.get("suggestions", []) or []
                if str(item).strip()
            ]
            if suggestions:
                prompt_parts.append("Repair notes:\n" + "\n".join(f"- {item}" for item in suggestions[:8]))
        if run_memory_text:
            prompt_parts.append(run_memory_text)
        prompt_parts.append(
            "Search the repository yourself before editing. Prefer targeted inspection of the recommended surfaces and their dependencies over blind rewriting."
        )
        prompt_parts.append(
            "Finish with a concise summary of the files you changed, the repo-level problems addressed, and any remaining risks."
        )
        return "\n\n".join(part for part in prompt_parts if part)

    def run_repo_repair_pass(
        self,
        *,
        state: PaperBenchReproState,
        target: str,
        requirement_anchor: dict[str, Any],
        repair_eval_report: dict[str, Any],
        repair_plan: dict[str, Any],
        iteration: int,
        iteration_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        previous_files = dict((iteration_context or {}).get("previous_files", {}) or {})
        repo_dir = self._repo_dir()
        repo_dir.mkdir(parents=True, exist_ok=True)
        if previous_files and not any(repo_dir.iterdir()):
            save_project_files(previous_files, repo_dir)

        agent = self._build_repair_agent()
        prompt = self._build_repo_repair_prompt(
            target=target,
            requirement_anchor=requirement_anchor,
            repair_eval_report=repair_eval_report,
            repair_plan=repair_plan,
            iteration_context=iteration_context,
            run_memory_text=run_memory.get_run_memory_prompt(state, max_chars=4000),
        )
        task_dir = self._node_dir()
        task_dir.mkdir(parents=True, exist_ok=True)
        if self.output_dir is not None:
            (task_dir / "current_prompt.md").write_text(prompt, encoding="utf-8")

        if hasattr(agent, "execute_best_effort"):
            result = agent.execute_best_effort(
                prompt,
                self._build_repo_agent_context(repo_dir),
                output_mode="text",
                timeout=int(getattr(self.workflow_config, "docker_validate_timeout", 1800) or 1800),
            )
            project_files = load_project_files(repo_dir)
            fallback_updates = self._parse_updated_files(
                result.get("output", "") or result.get("raw_output", ""),
                {},
                {},
            )
            if fallback_updates:
                project_files = merge_project_files(project_files or previous_files, fallback_updates)
                save_project_files(project_files, repo_dir)
            updated_files = self._diff_project_files(previous_files, project_files)
            safety_report = self._guard_repo_repair_update(
                previous_files=previous_files,
                project_files=project_files,
                updated_files=updated_files,
                recommended_surfaces=[
                    str(item).strip()
                    for item in list(repair_plan.get("recommended_surfaces", []) or [])
                    if str(item).strip()
                ],
            )
            self._write_repair_safety_report(safety_report)
            if not bool(safety_report.get("passed", False)):
                project_files = dict(previous_files)
                self._persist_project_files(project_files)
                updated_files = {}
                result = {
                    **dict(result or {}),
                    "success": False,
                    "error": (
                        str((result or {}).get("error", "") or "").strip()
                        + "\nrepair safety guard rejected repo-wide update: "
                        + "; ".join(str(item) for item in list(safety_report.get("issues", []) or [])[:8])
                    ).strip(),
                    "safety_report": safety_report,
                }
            context_usage = self._build_task_usage(prompt, result)
            checkpoint_payload = self._save_iteration_checkpoint(
                iteration=iteration,
                execution_history=list(state.execution_history or []),
                generated_files=project_files,
                termination_reason="Completed repo-wide repair pass",
                latest_stage="repo_repair",
                latest_status=(
                    "rejected_by_safety_guard"
                    if not bool(safety_report.get("passed", False))
                    else "passed" if bool(updated_files) else "no_changes"
                ),
            )
            self._save_attempt(
                iteration,
                project_files.get(state.project_plan.entrypoints.get("main", "main.py"), ""),
                {
                    "success": bool(result.get("success")),
                    "output": str(result.get("output", "") or ""),
                    "error": str(result.get("error", "") or ""),
                    "exit_code": int(result.get("exit_code", 0) or 0),
                    "checks": [],
                    "artifacts": [],
                    "artifact_summary": {},
                    "safety_report": safety_report,
                },
                list(repair_plan.get("review_points", []) or []),
                [],
                context_usage,
                project_files,
                str(repo_dir.resolve()),
                {
                    "materialization_mode": "repair_repo_wide",
                    "recommended_surfaces": list(repair_plan.get("recommended_surfaces", []) or []),
                },
                checkpoint_payload=checkpoint_payload,
                task_id=f"repair_round_{iteration:03d}",
                changed_files=updated_files,
            )
            return {
                "project_files": project_files,
                "updated_files": updated_files,
                "agent_result": result,
                "context_usage": context_usage,
                "checkpoint_payload": checkpoint_payload,
                "safety_report": safety_report,
            }

        response = agent.execute(prompt, self._build_repo_agent_context(repo_dir), output_mode="text")
        updated_files = self._parse_updated_files(response, {}, {})
        project_files = merge_project_files(previous_files, updated_files)
        safety_report = self._guard_repo_repair_update(
            previous_files=previous_files,
            project_files=project_files,
            updated_files=updated_files,
            recommended_surfaces=[
                str(item).strip()
                for item in list(repair_plan.get("recommended_surfaces", []) or [])
                if str(item).strip()
            ],
        )
        self._write_repair_safety_report(safety_report)
        if bool(safety_report.get("passed", False)):
            save_project_files(project_files, repo_dir)
        else:
            project_files = dict(previous_files)
            updated_files = {}
            self._persist_project_files(project_files)
        agent_result = {
            "success": bool(updated_files) and bool(safety_report.get("passed", False)),
            "output": response,
            "raw_output": response,
            "error": (
                ""
                if updated_files and bool(safety_report.get("passed", False))
                else (
                    "repair safety guard rejected repo-wide update: "
                    + "; ".join(str(item) for item in list(safety_report.get("issues", []) or [])[:8])
                    if not bool(safety_report.get("passed", False))
                    else "No files were produced by the repo-wide repair agent."
                )
            ),
            "usage": {},
            "safety_report": safety_report,
        }
        context_usage = self._build_task_usage(prompt, agent_result)
        checkpoint_payload = self._save_iteration_checkpoint(
            iteration=iteration,
            execution_history=list(state.execution_history or []),
            generated_files=project_files,
            termination_reason="Completed repo-wide repair pass",
            latest_stage="repo_repair",
            latest_status=(
                "rejected_by_safety_guard"
                if not bool(safety_report.get("passed", False))
                else "passed" if bool(updated_files) else "no_changes"
            ),
        )
        self._save_attempt(
            iteration,
            project_files.get(state.project_plan.entrypoints.get("main", "main.py"), ""),
            {
                "success": bool(agent_result.get("success")),
                "output": response,
                "error": str(agent_result.get("error", "") or ""),
                "exit_code": 0 if bool(agent_result.get("success")) else 1,
                "checks": [],
                "artifacts": [],
                "artifact_summary": {},
                "safety_report": safety_report,
            },
            list(repair_plan.get("review_points", []) or []),
            [],
            context_usage,
            project_files,
            str(repo_dir.resolve()),
            {
                "materialization_mode": "repair_repo_wide",
                "recommended_surfaces": list(repair_plan.get("recommended_surfaces", []) or []),
            },
            checkpoint_payload=checkpoint_payload,
            task_id=f"repair_round_{iteration:03d}",
            changed_files=updated_files,
        )
        return {
            "project_files": project_files,
            "updated_files": updated_files,
            "agent_result": agent_result,
            "context_usage": context_usage,
            "checkpoint_payload": checkpoint_payload,
            "safety_report": safety_report,
        }
