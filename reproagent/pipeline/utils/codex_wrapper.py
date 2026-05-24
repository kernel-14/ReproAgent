"""Codex CLI wrapper via subprocess."""
import subprocess
import os
import re
from pathlib import Path
from typing import Any

from .agent_executor import AgentExecutor


def _coerce_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    digits = re.sub(r"[^0-9]", "", str(value))
    return int(digits) if digits else None


def normalize_codex_usage(usage: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = dict(usage or {})
    session_id = str(payload.get("session_id", "") or "").strip()
    input_tokens = _coerce_int(payload.get("input_tokens"))
    output_tokens = _coerce_int(payload.get("output_tokens"))
    total_tokens = _coerce_int(payload.get("total_tokens"))
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    matched_lines: list[str] = []
    for line in list(payload.get("matched_lines", []) or []):
        rendered = str(line or "").strip()
        if rendered and rendered not in matched_lines:
            matched_lines.append(rendered)

    usage_found = any(value is not None for value in (input_tokens, output_tokens, total_tokens))
    if payload.get("usage_found") and not usage_found:
        usage_found = bool(session_id)

    return {
        "session_id": session_id,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "usage_found": usage_found,
        "usage_source": str(payload.get("usage_source", "") or "").strip(),
        "matched_lines": matched_lines[:6],
    }


def _extract_codex_session_id(text: str) -> str:
    if not text:
        return ""
    match = re.search(r"(?im)^\s*session id:\s*([A-Za-z0-9-]+)\s*$", text)
    return str(match.group(1)).strip() if match else ""


def _extract_codex_token_count(text: str, patterns: list[str]) -> int | None:
    if not text:
        return None
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if not match:
            continue
        return _coerce_int(match.group(1))
    return None


def _collect_codex_usage_lines(text: str) -> list[str]:
    if not text:
        return []
    captured: list[str] = []
    token_line_pattern = re.compile(
        r"(?i)(session id:|token usage|input[_ ]tokens?|output[_ ]tokens?|total[_ ]tokens?|prompt[_ ]tokens?|completion[_ ]tokens?)"
    )
    for line in text.splitlines():
        rendered = line.strip()
        if rendered and token_line_pattern.search(rendered) and rendered not in captured:
            captured.append(rendered)
    return captured[:6]


def merge_codex_usage(stdout: str, stderr: str) -> dict[str, Any]:
    combined = "\n".join(part for part in [stderr, stdout] if part)
    usage = {
        "session_id": _extract_codex_session_id(combined),
        "input_tokens": _extract_codex_token_count(
            combined,
            [
                r"\binput[_ ]tokens?\b[^0-9]{0,20}([0-9][0-9,]*)",
                r"\bprompt[_ ]tokens?\b[^0-9]{0,20}([0-9][0-9,]*)",
            ],
        ),
        "output_tokens": _extract_codex_token_count(
            combined,
            [
                r"\boutput[_ ]tokens?\b[^0-9]{0,20}([0-9][0-9,]*)",
                r"\bcompletion[_ ]tokens?\b[^0-9]{0,20}([0-9][0-9,]*)",
            ],
        ),
        "total_tokens": _extract_codex_token_count(
            combined,
            [
                r"\btotal[_ ]tokens?\b[^0-9]{0,20}([0-9][0-9,]*)",
            ],
        ),
        "usage_source": "codex_cli",
        "matched_lines": _collect_codex_usage_lines(combined),
    }
    return normalize_codex_usage(usage)


def aggregate_codex_usage(items: list[dict[str, Any] | None]) -> dict[str, Any]:
    normalized_items = [normalize_codex_usage(item) for item in items]
    session_ids: list[str] = []
    usage_sources: list[str] = []
    for item in normalized_items:
        session_id = str(item.get("session_id", "") or "")
        if session_id and session_id not in session_ids:
            session_ids.append(session_id)
        usage_source = str(item.get("usage_source", "") or "")
        if usage_source and usage_source not in usage_sources:
            usage_sources.append(usage_source)

    return {
        "calls": len(normalized_items),
        "calls_with_usage": sum(1 for item in normalized_items if bool(item.get("usage_found"))),
        "calls_with_session_id": sum(1 for item in normalized_items if str(item.get("session_id", "") or "")),
        "input_tokens": sum(int(item.get("input_tokens", 0) or 0) for item in normalized_items),
        "output_tokens": sum(int(item.get("output_tokens", 0) or 0) for item in normalized_items),
        "total_tokens": sum(int(item.get("total_tokens", 0) or 0) for item in normalized_items),
        "session_ids": session_ids,
        "usage_sources": usage_sources,
    }


class CodexWrapper(AgentExecutor):
    """Wrapper for Codex via codeagent-wrapper."""

    def __init__(
        self,
        cli_path: str = "/root/.claude/bin/codeagent-wrapper",
        model: str | None = None,
        model_provider: str | None = None,
        base_url: str | None = None,
        reasoning_effort: str | None = None,
    ):
        self.cli_path = cli_path
        self.model = model
        self.model_provider = model_provider
        self.base_url = base_url
        self.reasoning_effort = reasoning_effort
        self.last_result: dict[str, Any] = {}

    def _build_env(self, base_env: dict[str, str] | None = None) -> dict[str, str]:
        env = dict(os.environ)
        if base_env:
            env.update(base_env)
        slot = os.getenv("PAPERBENCH_REPRO_PROVIDER_SLOT", "").strip()
        slot_suffix = f"_{slot}" if slot and not slot.startswith("_") else slot
        api_key = str(
            os.getenv("PAPERAGENT_PAPERBENCH_REPRO_CODEX_API_KEY", "")
            or os.getenv("PAPERAGENT_EXP_GEN_CODEX_API_KEY", "")
            or (os.getenv(f"OPENAI_API_KEY{slot_suffix}", "") if slot_suffix else "")
            or ""
        ).strip()
        if api_key:
            env["OPENAI_API_KEY"] = api_key
        return env

    def _build_codex_exec_cmd(self, cwd: str) -> list[str]:
        """Prefer direct `codex exec` when explicit provider settings are required."""
        provider = self.model_provider
        cmd = [
            os.getenv("PAPERBENCH_REPRO_CODEX_DIRECT_CLI_PATH")
            or os.getenv("EXP_GEN_CODEX_DIRECT_CLI_PATH", "codex"),
            "exec",
            "--ignore-user-config",
            "--dangerously-bypass-approvals-and-sandbox",
            "--skip-git-repo-check",
            "-C",
            cwd,
        ]
        if self.model:
            cmd.extend(["-m", self.model])
        if self.reasoning_effort:
            cmd.extend(["-c", f'model_reasoning_effort="{self.reasoning_effort}"'])
        if provider:
            cmd.extend(["-c", f'model_provider="{provider}"'])
            cmd.extend(["-c", f'model_providers.{provider}.name="{provider}"'])
        if provider and self.base_url:
            cmd.extend(["-c", f'model_providers.{provider}.base_url="{self.base_url}"'])
            cmd.extend(["-c", f'model_providers.{provider}.wire_api="responses"'])
            cmd.extend(["-c", f'model_providers.{provider}.env_key="OPENAI_API_KEY"'])
            cmd.extend(["-c", f"model_providers.{provider}.requires_openai_auth=true"])
        cmd.append("-")
        return cmd

    def _normalize_output(self, output: str, output_mode: str) -> str:
        """Normalize CLI output for either raw text or Python code."""
        cleaned = output.strip()
        if output_mode != "code":
            return cleaned

        match = re.search(r'```(?:python)?\n(.*?)\n```', cleaned, re.DOTALL)
        return match.group(1).strip() if match else cleaned

    def execute_best_effort(self, prompt: str, context: dict, output_mode: str = "code", timeout: int = 600) -> dict:
        """Execute via Codex CLI and return structured best-effort metadata.

        This variant is intended for repo-driven workflows where filesystem state is
        the source of truth. Timeout or non-zero exit should not discard partial file
        edits already written by Codex under the working directory.
        """
        env = self._build_env(context.get("env") if isinstance(context.get("env"), dict) else None)
        original_cwd = os.getcwd()
        if cwd := context.get("cwd"):
            os.chdir(cwd)

        try:
            if self.model or self.model_provider or self.base_url or self.reasoning_effort:
                cmd = self._build_codex_exec_cmd(str(Path.cwd()))
            else:
                cmd = [self.cli_path, "--backend", "codex", "-", str(Path.cwd())]

            try:
                result = subprocess.run(
                    cmd,
                    input=prompt,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    env=env,
                )
                stdout = result.stdout or ""
                stderr = result.stderr or ""
                success = result.returncode == 0
                payload = {
                    "success": success,
                    "timed_out": False,
                    "output": self._normalize_output(stdout, output_mode),
                    "raw_output": stdout,
                    "error": "" if success else f"Codex CLI failed: {stderr or stdout}".strip(),
                    "stderr": stderr,
                    "exit_code": result.returncode,
                    "usage": merge_codex_usage(stdout, stderr),
                }
                self.last_result = payload
                return payload
            except subprocess.TimeoutExpired as exc:
                stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
                stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
                payload = {
                    "success": False,
                    "timed_out": True,
                    "output": self._normalize_output(stdout, output_mode),
                    "raw_output": stdout,
                    "error": str(exc),
                    "stderr": stderr,
                    "exit_code": -1,
                    "usage": merge_codex_usage(stdout, stderr),
                }
                self.last_result = payload
                return payload
            except Exception as exc:
                payload = {
                    "success": False,
                    "timed_out": False,
                    "output": "",
                    "raw_output": "",
                    "error": str(exc),
                    "stderr": "",
                    "exit_code": -1,
                    "usage": normalize_codex_usage({"usage_source": "codex_cli"}),
                }
                self.last_result = payload
                return payload
        finally:
            os.chdir(original_cwd)

    def execute(self, prompt: str, context: dict, output_mode: str = "code") -> str:
        """Execute via Codex CLI."""
        result = self.execute_best_effort(prompt, context, output_mode=output_mode, timeout=600)
        if result["success"]:
            return result["output"]
        if result.get("timed_out"):
            raise TimeoutError(result["error"])
        raise RuntimeError(result["error"])
