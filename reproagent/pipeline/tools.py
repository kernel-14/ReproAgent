"""Tools for reproagent workflow."""
import json
import re
import shlex
import subprocess
import tempfile
import uuid
from pathlib import Path
from reproagent.sandbox import get_sandbox_provider
from reproagent.pipeline.config import build_github_repo_config
from reproagent.pipeline.utils.ref_repo_search_tool import (
    search_github_repositories,
    search_reference_repository,
)


def _cleanup_sandbox_file(sandbox, path: str) -> None:
    """Best-effort cleanup for temporary sandbox files."""
    try:
        sandbox.execute_command(f"rm -f {shlex.quote(path)}")
    except Exception:
        pass


def _format_env_assignments(extra_env: dict[str, str]) -> str:
    """Render validated env assignments for shell-based sandbox execution."""
    assignments: list[str] = []
    for key, value in extra_env.items():
        rendered_key = str(key or "").strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", rendered_key):
            raise ValueError(f"Invalid environment variable name: {rendered_key!r}")
        assignments.append(f"{rendered_key}={shlex.quote(str(value))}")
    return " ".join(assignments)


def execute_sandbox_commands(commands: list[str], sandbox) -> list[dict]:
    """Execute shell commands inside an already-acquired sandbox."""
    results = []
    for command in commands:
        result = sandbox.execute_command(command)
        results.append({
            "command": command,
            "success": result["exit_code"] == 0,
            "output": result["stdout"],
            "error": result["stderr"],
            "exit_code": result["exit_code"],
        })
    return results


def execute_python_code(
    code: str,
    timeout: int = 300,
    soft_timeout: int = 180,
    sandbox_provider=None,
    sandbox=None,
    script_path: str | None = None,
    extra_env: dict[str, str] | None = None,
) -> dict:
    """Execute Python code in sandbox with soft/hard timeout monitoring.

    Args:
        code: Python code to execute
        timeout: Hard timeout in seconds (kills process)
        soft_timeout: Soft timeout in seconds (warning only)
        sandbox_provider: Optional sandbox provider (defaults to get_sandbox_provider())
        sandbox: Optional already-acquired sandbox instance for reuse
        script_path: Optional fixed script path inside sandbox
        extra_env: Optional environment variables for the Python process

    Returns:
        Dict with success, output, error, exit_code, warnings
    """
    del timeout, soft_timeout  # Managed by sandbox runtime.

    warnings = []
    generated_script_path = script_path is None
    provider = sandbox_provider or get_sandbox_provider()
    active_sandbox = sandbox
    owned_sandbox_id: str | None = None
    if active_sandbox is None:
        owned_sandbox_id = provider.acquire()
        active_sandbox = provider.get(owned_sandbox_id)

    active_script_path = script_path or f"/tmp/reproagent_{uuid.uuid4().hex}.py"
    try:
        if active_sandbox is None:
            return {
                "success": False,
                "output": "",
                "error": "Failed to acquire sandbox",
                "exit_code": -1,
                "warnings": warnings
            }
        active_sandbox.write_file(active_script_path, code)
        if extra_env:
            env_args = _format_env_assignments(extra_env)
            command = f"env {env_args} python {shlex.quote(active_script_path)}"
        else:
            command = f"python {shlex.quote(active_script_path)}"
        result = active_sandbox.execute_command(command)

        return {
            "success": result["exit_code"] == 0,
            "output": result["stdout"],
            "error": result["stderr"],
            "exit_code": result["exit_code"],
            "warnings": warnings
        }
    except Exception as e:
        return {
            "success": False,
            "output": "",
            "error": str(e),
            "exit_code": -1,
            "warnings": warnings
        }
    finally:
        if generated_script_path and active_sandbox is not None:
            _cleanup_sandbox_file(active_sandbox, active_script_path)
        if owned_sandbox_id is not None:
            provider.release(owned_sandbox_id)


def save_code_artifact(code: str, output_dir: Path, filename: str = "generated_code.py") -> Path:
    """Save generated code to file.

    Args:
        code: Code content
        output_dir: Output directory
        filename: Output filename

    Returns:
        Path to saved file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / filename
    filepath.write_text(code, encoding="utf-8")
    return filepath


def save_project_files(project_files: dict[str, str], output_dir: Path) -> list[Path]:
    """Persist a generated multi-file project layout."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    for relative_path, content in project_files.items():
        filepath = output_dir / relative_path
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(content, encoding="utf-8")
        saved_paths.append(filepath)
    return saved_paths


def merge_project_files(project_files: dict[str, str], updated_files: dict[str, str]) -> dict[str, str]:
    """Merge changed files into the current project snapshot."""
    merged = dict(project_files)
    for relative_path, content in updated_files.items():
        if not relative_path or not isinstance(content, str):
            continue
        merged[relative_path] = content
    return merged


_IGNORED_PROJECT_DIRS = {
    "__pycache__",
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "node_modules",
    ".mypy_cache",
    ".pytest_cache",
}

_IGNORED_PROJECT_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".so",
    ".dylib",
    ".dll",
    ".pkl",
    ".pickle",
    ".npy",
    ".npz",
    ".pt",
    ".pth",
    ".ckpt",
}


def _should_load_project_file(project_root: Path, path: Path) -> bool:
    """Return True when a repo file should be treated as editable text state."""
    relative_parts = path.relative_to(project_root).parts
    if any(part in _IGNORED_PROJECT_DIRS for part in relative_parts[:-1]):
        return False
    if path.suffix.lower() in _IGNORED_PROJECT_SUFFIXES:
        return False
    return True


def load_project_files(project_root: Path) -> dict[str, str]:
    """Load all files in a generated project directory."""
    if not project_root.exists():
        return {}

    files: dict[str, str] = {}
    for path in sorted(project_root.rglob("*")):
        if not path.is_file():
            continue
        if not _should_load_project_file(project_root, path):
            continue
        try:
            files[str(path.relative_to(project_root))] = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
    return files


def list_project_files(project_root: Path) -> list[str]:
    """List relative file paths inside a generated project directory."""
    return sorted(load_project_files(project_root).keys())


def write_json_artifact(payload: dict, output_path: Path) -> Path:
    """Write a JSON artifact with utf-8 encoding."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_path


def run_project_entrypoint(
    command: str,
    project_root: Path,
    sandbox,
    extra_env: dict[str, str] | None = None,
    timeout: int | None = None,
) -> dict:
    """Execute a project command inside the sandbox from the project root."""
    env_prefix = ""
    if extra_env:
        env_args = " ".join(f"{key}={shlex.quote(str(value))}" for key, value in extra_env.items())
        env_prefix = f"env {env_args} "
    inner_command = f"cd {shlex.quote(str(project_root))} && {env_prefix}{command}"
    if timeout and timeout > 0:
        full_command = f"timeout {int(timeout)}s bash -lc {shlex.quote(inner_command)}"
    else:
        full_command = inner_command
    result = sandbox.execute_command(full_command)
    stderr = result["stderr"]
    if timeout and result["exit_code"] == 124 and not stderr:
        stderr = f"Command timed out after {timeout}s"
    return {
        "success": result["exit_code"] == 0,
        "output": result["stdout"],
        "error": stderr,
        "exit_code": result["exit_code"],
        "command": full_command,
    }


def run_runtime_smoke_entrypoint(
    entrypoints: dict[str, str],
    project_root: Path,
    sandbox,
    extra_env: dict[str, str] | None = None,
    timeout: int | None = None,
) -> dict:
    """Run the declared runtime-smoke command for a generated project."""
    command = entrypoints.get("runtime_smoke") or "python main.py"
    return run_project_entrypoint(command, project_root, sandbox, extra_env=extra_env, timeout=timeout)


def run_docker_validate_entrypoint(
    entrypoints: dict[str, str],
    project_root: Path,
    sandbox,
    extra_env: dict[str, str] | None = None,
    timeout: int | None = None,
) -> dict:
    """Run the declared docker-validation command for a generated project."""
    command = entrypoints.get("docker_validate") or "python main.py"
    return run_project_entrypoint(command, project_root, sandbox, extra_env=extra_env, timeout=timeout)


def read_metrics_json(metrics_path: Path) -> dict:
    """Read and parse a metrics.json artifact."""
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def validate_metrics_contract(payload: dict, required_keys: list[str] | None = None) -> dict:
    """Validate the minimal experiment metrics contract."""
    required = required_keys or ["main_results", "ablation_results", "training_log"]
    if isinstance(payload, dict):
        summary_source = payload
        nested_test_metrics = payload.get("test_metrics")
        if isinstance(nested_test_metrics, dict):
            summary_source = nested_test_metrics

        summary_score = summary_source.get("score")
        if not isinstance(summary_score, (int, float)):
            summary_score = summary_source.get("macro_f1")
        if not isinstance(summary_score, (int, float)):
            summary_score = summary_source.get("accuracy")

        if isinstance(summary_score, (int, float)):
            normalized_main_result = {
                "score": float(summary_score),
            }
            for key in (
                "accuracy",
                "macro_f1",
                "all_targets_met",
                "target_met",
                "meets_target",
                "meets_both_targets",
                "target_hit",
                "meets_accuracy_target",
                "meets_macro_f1_target",
                "n_test_samples",
                "accuracy_threshold",
                "macro_f1_threshold",
                "target_accuracy",
                "target_macro_f1",
            ):
                if key in summary_source:
                    normalized_main_result[key] = summary_source[key]

            if "target_met" in normalized_main_result and "all_targets_met" not in normalized_main_result:
                normalized_main_result["all_targets_met"] = bool(normalized_main_result["target_met"])
            if "meets_target" in normalized_main_result and "all_targets_met" not in normalized_main_result:
                normalized_main_result["all_targets_met"] = bool(normalized_main_result["meets_target"])
            if "meets_both_targets" in normalized_main_result and "all_targets_met" not in normalized_main_result:
                normalized_main_result["all_targets_met"] = bool(normalized_main_result["meets_both_targets"])
            if "target_hit" in normalized_main_result and "all_targets_met" not in normalized_main_result:
                normalized_main_result["all_targets_met"] = bool(normalized_main_result["target_hit"])
            if (
                "all_targets_met" not in normalized_main_result
                and normalized_main_result.get("meets_accuracy_target") is True
                and normalized_main_result.get("meets_macro_f1_target") is True
            ):
                normalized_main_result["all_targets_met"] = True

            return {
                "passed": True,
                "required_keys": list(required),
                "missing_keys": [],
                "invalid_keys": [],
                "normalized": {
                    "main_results": [normalized_main_result],
                    "ablation_results": [],
                    "training_log": [],
                },
            }

        if all(key in payload for key in required):
            normalized = {}
            invalid = []
            for key in required:
                section = payload.get(key, [])
                if isinstance(section, list):
                    normalized[key] = section
                    continue
                if key == "main_results" and isinstance(section, dict):
                    normalized[key] = [section]
                    continue
                invalid.append(key)
                normalized[key] = []

            return {
                "passed": not invalid,
                "required_keys": list(required),
                "missing_keys": [],
                "invalid_keys": invalid,
                "normalized": normalized,
            }

    missing = [key for key in required if key not in payload]
    invalid = [key for key in required if key in payload and not isinstance(payload[key], list)]
    return {
        "passed": not missing and not invalid,
        "required_keys": list(required),
        "missing_keys": missing,
        "invalid_keys": invalid,
        "normalized": {
            key: payload.get(key, []) if isinstance(payload.get(key, []), list) else []
            for key in required
        },
    }


def run_acceptance_checks(code: str, checks: list[str], timeout: int) -> list[dict]:
    """Run acceptance checks on code.

    Args:
        code: Python code to check
        checks: List of check commands with {code_file} placeholder
        timeout: Timeout per check in seconds

    Returns:
        List of CheckResult dicts
    """
    results = []
    with tempfile.NamedTemporaryFile("w", suffix=".py", encoding="utf-8", delete=False) as handle:
        handle.write(code)
        temp_file = Path(handle.name)

    try:
        for check_cmd in checks:
            cmd = check_cmd.format(code_file=str(temp_file))
            try:
                cmd_args = shlex.split(cmd)
                if not cmd_args:
                    raise ValueError("Acceptance check command is empty")
                result = subprocess.run(
                    cmd_args,
                    shell=False,
                    capture_output=True,
                    timeout=timeout,
                    text=True
                )
                results.append({
                    "command": cmd,
                    "passed": result.returncode == 0,
                    "output": result.stdout,
                    "error": result.stderr
                })
            except subprocess.TimeoutExpired:
                results.append({
                    "command": cmd,
                    "passed": False,
                    "output": "",
                    "error": f"Timeout after {timeout}s"
                })
            except Exception as e:
                results.append({
                    "command": cmd,
                    "passed": False,
                    "output": "",
                    "error": str(e)
                })
    finally:
        temp_file.unlink(missing_ok=True)
    return results


def parse_training_metrics(output: str) -> dict[str, float]:
    """Parse training metrics from execution output."""
    import re
    metrics = {}

    patterns = [
        (r"(?:top-?1|top1|accuracy)[\s:=]+(\d+\.?\d*)%?", "top-1"),
        (r"(?:top-?5|top5)[\s:=]+(\d+\.?\d*)%?", "top-5"),
        (r"(?:loss|train_loss)[\s:=]+(\d+\.?\d*)", "loss"),
        (r"(?:val_acc|val_accuracy)[\s:=]+(\d+\.?\d*)%?", "val_accuracy"),
    ]

    for pattern, key in patterns:
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            metrics[key] = float(match.group(1))

    return metrics


def repo_search_ref_repository(
    ref_id: str,
    paper_path: str,
    paper_title: str = "",
    paper_url: str = "",
    max_results: int = 5,
    github_config: dict | None = None,
) -> dict:
    """Resolve a paper markdown path to a trusted reference repository."""
    effective_github_config = build_github_repo_config(github_config)
    return search_reference_repository(
        ref_id=ref_id,
        paper_path=paper_path,
        paper_title=paper_title,
        paper_url=paper_url,
        max_results=max_results,
        github_config=effective_github_config,
    )


def search_reference_github_repositories(
    title: str,
    title_aliases: list[str] | None = None,
    arxiv_id: str = "",
    max_results: int = 5,
    github_config: dict | None = None,
) -> list[dict]:
    """Search GitHub repositories directly for paper-aligned repo candidates."""
    effective_github_config = build_github_repo_config(github_config)
    return search_github_repositories(
        title=title,
        title_aliases=title_aliases,
        arxiv_id=arxiv_id,
        max_results=max_results,
        github_config=effective_github_config,
    )
