"""Git clone helper for reproagent reference repositories."""

from __future__ import annotations

import os
import shlex
import re
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlparse


def clone_reference_repository(
    repository_url: str,
    target_root: str | Path,
    repo_name: str | None = None,
    github_config: dict | None = None,
) -> dict:
    """Clone or reuse a reference repository under the given root."""
    repository_identity = _repository_identity(repository_url, github_config=github_config)
    if repository_identity is None:
        return {
            "repository_url": repository_url,
            "status": "failed",
            "local_repo_path": "",
            "default_branch": "",
            "reason": "Invalid GitHub repository URL",
        }
    normalized_url = _https_repository_url(repository_identity)
    clone_urls = _clone_repository_urls(repository_identity, github_config=github_config)

    root = Path(target_root)
    root.mkdir(parents=True, exist_ok=True)
    directory_name = repo_name.strip() if repo_name else _derive_repo_dir_name(normalized_url)
    local_repo_path = root / directory_name

    if local_repo_path.exists() and any(local_repo_path.iterdir()):
        remote_url = _get_origin_remote(local_repo_path)
        if _repository_identity(remote_url, github_config=github_config) == repository_identity:
            return {
                "repository_url": normalized_url,
                "status": "reused",
                "local_repo_path": str(local_repo_path.resolve()),
                "default_branch": _get_default_branch(local_repo_path),
                "reason": "",
            }
        shutil.rmtree(local_repo_path, ignore_errors=True)

    last_error = ""
    used_transport = ""
    clone_timeout = _clone_timeout_seconds(github_config)
    clone_attempts = _clone_attempt_count(github_config)
    for clone_url in clone_urls:
        for attempt in range(1, clone_attempts + 1):
            if local_repo_path.exists() and not any(local_repo_path.iterdir()):
                local_repo_path.rmdir()
            elif local_repo_path.exists():
                shutil.rmtree(local_repo_path, ignore_errors=True)
            try:
                result = subprocess.run(
                    [
                        "git",
                        "-c",
                        "http.version=HTTP/1.1",
                        "-c",
                        "http.lowSpeedLimit=1000",
                        "-c",
                        "http.lowSpeedTime=30",
                        "clone",
                        "--depth",
                        "1",
                        "--filter=blob:none",
                        "--no-tags",
                        "--single-branch",
                        clone_url,
                        str(local_repo_path),
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=clone_timeout,
                    env=_git_clone_env(github_config, clone_url),
                )
            except subprocess.TimeoutExpired:
                last_error = f"git clone timed out after {clone_timeout} seconds"
                continue
            except Exception as exc:
                last_error = str(exc)
                continue

            if result.returncode == 0:
                used_transport = "ssh" if clone_url.startswith("git@") else "https"
                break

            last_error = (result.stderr or result.stdout).strip()
            if attempt < clone_attempts and _is_transient_clone_error(last_error):
                shutil.rmtree(local_repo_path, ignore_errors=True)
                continue
            break
        if used_transport:
            break
    else:
        return {
            "repository_url": normalized_url,
            "status": "failed",
            "local_repo_path": "",
            "default_branch": "",
            "reason": last_error,
        }

    return {
        "repository_url": normalized_url,
        "status": "cloned",
        "local_repo_path": str(local_repo_path.resolve()),
        "default_branch": _get_default_branch(local_repo_path),
        "transport": used_transport,
        "reason": "",
    }


def _repository_identity(
    repository_url: str,
    *,
    github_config: dict | None = None,
) -> tuple[str, str] | None:
    raw = (repository_url or "").strip()
    if not raw:
        return None
    ssh_match = re.match(r"git@([^:]+):([^/]+)/(.+?)(?:\.git)?$", raw)
    if ssh_match:
        host = ssh_match.group(1).lower()
        if host != _github_ssh_host(github_config).lower():
            return None
        owner = ssh_match.group(2)
        repo = ssh_match.group(3)
    else:
        parsed = urlparse(raw)
        if parsed.netloc.lower() != "github.com":
            return None
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) < 2:
            return None
        owner, repo = parts[0], parts[1]
    repo = repo.removesuffix(".git")
    if not owner or not repo:
        return None
    return owner, repo


def _https_repository_url(repository_identity: tuple[str, str]) -> str:
    owner, repo = repository_identity
    return f"https://github.com/{owner}/{repo}"


def _clone_repository_urls(repository_identity: tuple[str, str], github_config: dict | None = None) -> list[str]:
    owner, repo = repository_identity
    https_url = _https_repository_url(repository_identity)
    if _ssh_clone_enabled(github_config):
        ssh_url = f"git@{_github_ssh_host(github_config)}:{owner}/{repo}.git"
        return [ssh_url, https_url]
    return [https_url]


def _ssh_clone_enabled(github_config: dict | None = None) -> bool:
    config_value = (github_config or {}).get("ssh_enabled")
    if isinstance(config_value, bool):
        return config_value
    if config_value is not None:
        return str(config_value).strip().lower() in {"1", "true", "yes"}
    explicit_enabled = (
        os.getenv("PAPERBENCH_REPRO_GITHUB_SSH_ENABLED", "").strip()
        or os.getenv("EXP_GEN_GITHUB_SSH_ENABLED", "").strip()
    ).lower() in {"1", "true", "yes"}
    return explicit_enabled and bool(_github_ssh_command(github_config))


def _github_ssh_host(github_config: dict | None = None) -> str:
    config_value = str((github_config or {}).get("ssh_host", "")).strip()
    if config_value:
        return config_value
    return (
        os.getenv("PAPERBENCH_REPRO_GITHUB_SSH_HOST", "").strip()
        or os.getenv("EXP_GEN_GITHUB_SSH_HOST", "github.com").strip()
        or "github.com"
    )


def _git_clone_env(github_config: dict | None, clone_url: str) -> dict[str, str]:
    env = os.environ.copy()
    env["GIT_LFS_SKIP_SMUDGE"] = "1"
    env["GIT_TERMINAL_PROMPT"] = "0"

    if clone_url.startswith("git@"):
        ssh_command = _github_ssh_command(github_config)
        if ssh_command:
            env["GIT_SSH_COMMAND"] = ssh_command

    return env


def _github_ssh_command(github_config: dict | None = None) -> str:
    configured_command = str((github_config or {}).get("ssh_command", "")).strip()
    if configured_command:
        return configured_command

    key_path = str((github_config or {}).get("ssh_key_path", "")).strip()
    if key_path:
        return (
            f"ssh -i {shlex.quote(key_path)} "
            "-o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new"
        )

    env_command = (
        os.getenv("PAPERBENCH_REPRO_GITHUB_SSH_COMMAND", "").strip()
        or os.getenv("EXP_GEN_GITHUB_SSH_COMMAND", "").strip()
    )
    if env_command:
        return env_command

    env_key_path = (
        os.getenv("PAPERBENCH_REPRO_GITHUB_SSH_KEY_PATH", "").strip()
        or os.getenv("EXP_GEN_GITHUB_SSH_KEY_PATH", "").strip()
    )
    if env_key_path:
        return (
            f"ssh -i {shlex.quote(env_key_path)} "
            "-o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new"
        )

    return ""


def _clone_timeout_seconds(github_config: dict | None = None) -> int:
    value = (github_config or {}).get("clone_timeout_seconds")
    if value is None:
        value = os.getenv("PAPERBENCH_REPRO_GITHUB_CLONE_TIMEOUT", "").strip()
    try:
        timeout = int(value or 120)
    except (TypeError, ValueError):
        timeout = 120
    return max(10, timeout)


def _clone_attempt_count(github_config: dict | None = None) -> int:
    value = (github_config or {}).get("clone_attempts")
    if value is None:
        value = os.getenv("PAPERBENCH_REPRO_GITHUB_CLONE_ATTEMPTS", "").strip()
    try:
        attempts = int(value or 2)
    except (TypeError, ValueError):
        attempts = 2
    return max(1, min(attempts, 4))


def _is_transient_clone_error(message: str) -> bool:
    lowered = (message or "").lower()
    return any(
        token in lowered
        for token in (
            "gnutls",
            "tls connection",
            "connection reset",
            "connection timed out",
            "early eof",
            "the remote end hung up",
            "network is unreachable",
            "failed to connect",
            "operation timed out",
        )
    )


def _derive_repo_dir_name(repository_url: str) -> str:
    parsed = urlparse(repository_url)
    parts = [part for part in parsed.path.split("/") if part]
    owner = parts[0]
    repo = parts[1].removesuffix(".git")
    return f"{owner}__{repo}"


def _get_origin_remote(repo_path: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_path), "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except Exception:
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _get_default_branch(repo_path: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_path), "branch", "--show-current"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except Exception:
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip()
