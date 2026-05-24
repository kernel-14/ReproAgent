"""Static preflight checks for generated reproagent projects."""
import ast
import sys
from pathlib import Path


_OPTIONAL_HEAVY_IMPORT_ROOTS = {
    "CybORG",
    "cv2",
    "datasets",
    "gym",
    "gymnasium",
    "matplotlib",
    "metadrive",
    "numpy",
    "pandas",
    "PIL",
    "sklearn",
    "stable_baselines3",
    "tensorflow",
    "torch",
    "torchvision",
    "transformers",
}


class PreflightChecker:
    """Run lightweight structural checks before executing a generated project."""

    def run(self, project_root: Path, project_plan: dict, project_files: dict[str, str]) -> dict:
        """Run all preflight checks and return a normalized payload."""
        checks: list[dict] = []
        blocking_failures: list[str] = []
        warning_messages: list[str] = []

        self._append_check_results(
            checks,
            self._check_required_files(project_root, project_plan, project_files),
            blocking_failures,
            warning_messages,
        )
        self._append_check_results(
            checks,
            self._check_runtime_artifacts(project_root, project_plan),
            blocking_failures,
            warning_messages,
        )
        self._append_check_results(
            checks,
            self._check_artifact_hygiene(project_root, project_files),
            blocking_failures,
            warning_messages,
        )
        self._append_check_results(
            checks,
            self._check_declared_artifact_writers(project_plan, project_files),
            blocking_failures,
            warning_messages,
        )
        self._append_check_results(
            checks,
            self._check_entrypoints(project_root, project_plan),
            blocking_failures,
            warning_messages,
        )
        self._append_check_results(
            checks,
            self._check_python_syntax(project_root, project_files),
            blocking_failures,
            warning_messages,
        )
        self._append_check_results(
            checks,
            self._check_import_resolution(project_root, project_files),
            blocking_failures,
            warning_messages,
        )
        self._append_check_results(
            checks,
            self._check_top_level_optional_imports(project_files),
            blocking_failures,
            warning_messages,
        )

        if blocking_failures:
            status = "failed"
        elif warning_messages:
            status = "warnings"
        else:
            status = "passed"

        return {
            "status": status,
            "checks": checks,
            "blocking_failures": blocking_failures,
            "warning_messages": warning_messages,
            "suggested_fixes": blocking_failures + warning_messages,
        }

    def _append_check_results(self, sink: list[dict], results: list[dict], blocking: list[str], warnings: list[str]) -> None:
        """Merge one check batch into the overall preflight result."""
        for item in results:
            sink.append(item)
            message = item.get("message", "")
            severity = item.get("severity", "warning")
            passed = item.get("passed", False)
            if passed:
                continue
            if severity == "blocking":
                blocking.append(message)
            else:
                warnings.append(message)

    def _check_required_files(self, project_root: Path, project_plan: dict, project_files: dict[str, str]) -> list[dict]:
        """Ensure required files declared by the plan are present."""
        file_specs = project_plan.get("file_specs", []) if isinstance(project_plan, dict) else []
        artifact_contract = project_plan.get("artifact_contract", {}) if isinstance(project_plan, dict) else {}
        runtime_artifacts = {
            item
            for item in artifact_contract.get("required_files", [])
            if isinstance(item, str) and item
        }
        metrics_path = artifact_contract.get("metrics_path")
        if isinstance(metrics_path, str) and metrics_path:
            runtime_artifacts.add(metrics_path)

        required_files = [
            item.get("path")
            for item in file_specs
            if item.get("required", False) and item.get("path") and item.get("path") not in runtime_artifacts
        ]
        results = []
        for relative_path in required_files:
            exists = (project_root / relative_path).exists() or relative_path in project_files
            results.append({
                "name": "required_file",
                "path": relative_path,
                "passed": exists,
                "severity": "blocking",
                "message": f"Missing required file: {relative_path}" if not exists else f"Required file exists: {relative_path}",
            })
        return results

    def _check_artifact_hygiene(self, project_root: Path, project_files: dict[str, str]) -> list[dict]:
        """Block generated repos that already contain runtime result artifacts or caches."""
        del project_root
        runtime_roots = ("results/", "outputs/", "artifacts/", "reports/", "figures/", "plots/", "metrics/")
        suspicious_suffixes = (".csv", ".json", ".jsonl", ".tsv", ".parquet", ".png", ".jpg", ".jpeg", ".pdf")
        allowed_source_artifacts = {
            "configs/default.json",
            "package.json",
            "tsconfig.json",
        }
        suspicious_paths: list[str] = []
        for raw_path in sorted(project_files):
            path = str(raw_path or "").replace("\\", "/").strip("/")
            lowered = path.lower()
            if "__pycache__/" in lowered or lowered.endswith(".pyc"):
                suspicious_paths.append(path)
                continue
            if lowered in allowed_source_artifacts:
                continue
            if lowered.startswith(runtime_roots) and lowered.endswith(suspicious_suffixes):
                suspicious_paths.append(path)
        if suspicious_paths:
            return [
                {
                    "name": "artifact_hygiene",
                    "path": ", ".join(suspicious_paths[:8]),
                    "passed": False,
                    "severity": "blocking",
                    "message": (
                        "Generated repository snapshot contains runtime result artifacts or cache files. "
                        "Keep artifact writers in source code, but do not commit precomputed outputs: "
                        + ", ".join(suspicious_paths[:8])
                    ),
                }
            ]
        return [
            {
                "name": "artifact_hygiene",
                "path": "",
                "passed": True,
                "severity": "blocking",
                "message": "Generated repository snapshot is clean of runtime result artifacts and cache files.",
            }
        ]

    def _check_runtime_artifacts(self, project_root: Path, project_plan: dict) -> list[dict]:
        """Report runtime artifacts without failing static preflight."""
        artifact_contract = project_plan.get("artifact_contract", {}) if isinstance(project_plan, dict) else {}
        runtime_paths = {
            item
            for item in artifact_contract.get("required_files", [])
            if isinstance(item, str) and item
        }
        metrics_path = artifact_contract.get("metrics_path")
        if isinstance(metrics_path, str) and metrics_path:
            runtime_paths.add(metrics_path)

        results = []
        for relative_path in sorted(runtime_paths):
            exists = (project_root / relative_path).exists()
            results.append({
                "name": "runtime_artifact",
                "path": relative_path,
                "passed": exists,
                "severity": "warning",
                "message": (
                    f"Runtime artifact already exists: {relative_path}"
                    if exists
                    else f"Runtime artifact pending until execution: {relative_path}"
                ),
            })
        return results

    def _declared_runtime_artifacts(self, project_plan: dict) -> list[str]:
        artifact_contract = project_plan.get("artifact_contract", {}) if isinstance(project_plan, dict) else {}
        paths = [
            item
            for item in artifact_contract.get("required_files", [])
            if isinstance(item, str) and item
        ]
        metrics_path = artifact_contract.get("metrics_path")
        if isinstance(metrics_path, str) and metrics_path:
            paths.append(metrics_path)
        seen: set[str] = set()
        ordered: list[str] = []
        for path in paths:
            normalized = str(path).replace("\\", "/").strip("/")
            if normalized and normalized not in seen:
                seen.add(normalized)
                ordered.append(normalized)
        return ordered

    def _source_files_for_artifact_writers(self, project_files: dict[str, str]) -> dict[str, str]:
        runtime_roots = ("results/", "outputs/", "artifacts/", "reports/", "figures/", "plots/", "metrics/")
        return {
            path: content
            for path, content in project_files.items()
            if path.endswith((".py", ".sh", ".toml", ".yaml", ".yml", ".md"))
            and not str(path).replace("\\", "/").lower().startswith(runtime_roots)
        }

    def _check_declared_artifact_writers(self, project_plan: dict, project_files: dict[str, str]) -> list[dict]:
        """Require declared runtime artifacts to have source-level producers."""
        artifact_paths = self._declared_runtime_artifacts(project_plan)
        if not artifact_paths:
            return [
                {
                    "name": "declared_artifact_writers",
                    "path": "",
                    "passed": True,
                    "severity": "blocking",
                    "message": "No required runtime artifacts declared.",
                }
            ]

        source_files = self._source_files_for_artifact_writers(project_files)
        combined_source = "\n".join(source_files.values()).lower()
        writer_markers = ("write_json", "json.dump", "open(", ".write_text", "to_csv", "writerow", "savefig")
        missing: list[str] = []
        for artifact_path in artifact_paths:
            artifact_key = artifact_path.lower()
            basename = Path(artifact_path).name.lower()
            path_mentioned = artifact_key in combined_source or basename in combined_source
            writer_present = any(marker in combined_source for marker in writer_markers)
            if not (path_mentioned and writer_present):
                missing.append(artifact_path)

        if missing:
            return [
                {
                    "name": "declared_artifact_writers",
                    "path": ", ".join(missing[:8]),
                    "passed": False,
                    "severity": "blocking",
                    "message": (
                        "Declared runtime artifacts lack source-level producer code. "
                        "Artifact contracts must be backed by executable writer functions, not only by manifest declarations: "
                        + ", ".join(missing[:8])
                    ),
                }
            ]
        return [
            {
                "name": "declared_artifact_writers",
                "path": ", ".join(artifact_paths[:8]),
                "passed": True,
                "severity": "blocking",
                "message": "Declared runtime artifacts have discoverable source-level producers.",
            }
        ]

    def _check_entrypoints(self, project_root: Path, project_plan: dict) -> list[dict]:
        """Ensure the main entrypoint exists and staged commands are declared."""
        entrypoints = project_plan.get("entrypoints", {}) if isinstance(project_plan, dict) else {}
        main_path = entrypoints.get("main", "main.py")
        main_exists = (project_root / main_path).exists()
        results = [
            {
                "name": "main_entrypoint",
                "path": main_path,
                "passed": main_exists,
                "severity": "blocking",
                "message": f"Main entrypoint missing: {main_path}" if not main_exists else f"Main entrypoint exists: {main_path}",
            }
        ]
        for key in ("runtime_smoke", "docker_validate"):
            command = entrypoints.get(key, "")
            results.append({
                "name": key,
                "path": command,
                "passed": bool(command),
                "severity": "warning",
                "message": f"Entrypoint command missing for {key}" if not command else f"Entrypoint command declared for {key}",
            })
        return results

    def _check_python_syntax(self, project_root: Path, project_files: dict[str, str]) -> list[dict]:
        """Parse all Python files to catch syntax errors early."""
        results = []
        candidates = sorted(path for path in project_files.keys() if path.endswith(".py"))
        for relative_path in candidates:
            content = project_files[relative_path]
            try:
                ast.parse(content)
                results.append({
                    "name": "python_syntax",
                    "path": relative_path,
                    "passed": True,
                    "severity": "blocking",
                    "message": f"Python syntax OK: {relative_path}",
                })
            except SyntaxError as exc:
                results.append({
                    "name": "python_syntax",
                    "path": relative_path,
                    "passed": False,
                    "severity": "blocking",
                    "message": f"Python syntax error in {relative_path}: {exc}",
                })
        if not results:
            main_py = project_root / "main.py"
            results.append({
                "name": "python_syntax",
                "path": str(main_py),
                "passed": main_py.exists(),
                "severity": "warning",
                "message": "No Python files found for syntax check" if not main_py.exists() else "Fallback main.py exists for syntax checks",
            })
        return results

    def _check_import_resolution(self, project_root: Path, project_files: dict[str, str]) -> list[dict]:
        """Run basic static import checks for local modules."""
        module_stems = {Path(path).stem for path in project_files.keys() if path.endswith(".py")}
        results = []
        for relative_path, content in project_files.items():
            if not relative_path.endswith(".py"):
                continue
            try:
                tree = ast.parse(content)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        root_name = alias.name.split(".")[0]
                        if root_name in {"os", "sys", "json", "math", "pathlib", "typing", "collections", "itertools"}:
                            continue
                        if root_name in module_stems:
                            continue
                        results.append({
                            "name": "import_resolution",
                            "path": relative_path,
                            "passed": True,
                            "severity": "warning",
                            "message": f"Import present in {relative_path}: {alias.name}",
                        })
        if not results:
            results.append({
                "name": "import_resolution",
                "path": str(project_root),
                "passed": True,
                "severity": "warning",
                "message": "No additional import resolution issues detected",
            })
        return results

    def _check_top_level_optional_imports(self, project_files: dict[str, str]) -> list[dict]:
        """Warn when optional heavy dependencies are imported at module top level."""
        local_roots = {
            Path(path).parts[0].removesuffix(".py")
            for path in project_files
            if path.endswith(".py") and Path(path).parts
        }
        local_roots.update(Path(path).stem for path in project_files if path.endswith(".py"))
        violations: list[str] = []
        for relative_path, content in sorted(project_files.items()):
            if not relative_path.endswith(".py"):
                continue
            try:
                tree = ast.parse(content)
            except SyntaxError:
                continue
            for statement in tree.body:
                roots: list[str] = []
                if isinstance(statement, ast.Import):
                    roots.extend(str(alias.name or "").split(".", 1)[0] for alias in statement.names)
                elif isinstance(statement, ast.ImportFrom):
                    if int(statement.level or 0) > 0:
                        continue
                    roots.append(str(statement.module or "").split(".", 1)[0])
                for root in roots:
                    if not root or root in local_roots or root in sys.builtin_module_names:
                        continue
                    if root in _OPTIONAL_HEAVY_IMPORT_ROOTS:
                        violations.append(f"{relative_path}:{root}")
        if not violations:
            return [
                {
                    "name": "import_light_optional_dependencies",
                    "path": "",
                    "passed": True,
                    "severity": "warning",
                    "message": "No optional heavy dependencies are imported at module top level.",
                }
            ]
        return [
            {
                "name": "import_light_optional_dependencies",
                "path": ", ".join(violations[:12]),
                "passed": False,
                "severity": "warning",
                "message": (
                    "Optional heavy dependencies are imported at module top level. "
                    "Move them behind lazy imports or availability-checked factories so minimal smoke review can import the repo: "
                    + ", ".join(violations[:12])
                ),
            }
        ]
