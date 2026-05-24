"""Claude Code CLI wrapper via subprocess."""
import os
import subprocess
import tempfile
from .agent_executor import AgentExecutor


class ClaudeSDKWrapper(AgentExecutor):
    """Wrapper for Claude Code CLI."""

    def __init__(self, cli_path: str = "claude", model: str | None = None, effort: str | None = None):
        self.cli_path = cli_path
        self.model = model
        self.effort = effort
        self.session_id: str | None = None
        self.last_result: dict = {}

    def _normalize_output(self, output: str, output_mode: str) -> str:
        """Normalize CLI output for either raw text or Python code."""
        import re

        cleaned = output.strip()
        if output_mode != "code":
            return cleaned

        match = re.search(r'```(?:python)?\n(.*?)\n```', cleaned, re.DOTALL)
        return match.group(1).strip() if match else cleaned

    def _build_env(self, base_env: dict[str, str] | None = None) -> dict:
        env = os.environ.copy()
        if base_env:
            env.update(base_env)
        api_key = str(
            os.getenv("PAPERAGENT_PAPERBENCH_REPRO_CC_API_KEY", "")
            or os.getenv("PAPERAGENT_EXP_GEN_CC_API_KEY", "")
            or ""
        ).strip()
        if api_key:
            env["ANTHROPIC_API_KEY"] = api_key
            env["ANTHROPIC_AUTH_TOKEN"] = api_key
        configured_base_url = str(
            os.getenv("PAPERAGENT_PAPERBENCH_REPRO_CC_BASE_URL", "")
            or os.getenv("PAPERAGENT_EXP_GEN_CC_BASE_URL", "")
            or ""
        ).strip()
        if configured_base_url:
            env["ANTHROPIC_BASE_URL"] = configured_base_url
            env.setdefault("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1")
            env.setdefault("API_TIMEOUT_MS", "3000000")
            env.setdefault("CLAUDE_CODE_MAX_OUTPUT_TOKENS", "50000")
            env.setdefault("CLAUDE_BASH_NO_LOGIN", "1")
        env.pop("CLAUDECODE", None)
        if "ANTHROPIC_AUTH_TOKEN" not in env and "ANTHROPIC_API_KEY" in env:
            env["ANTHROPIC_AUTH_TOKEN"] = env["ANTHROPIC_API_KEY"]
        proxy_base_url = str(env.get("PAPERBENCH_REPRO_CLAUDE_BASE_URL", "") or env.get("EXP_GEN_CLAUDE_BASE_URL", "") or "").strip()
        if proxy_base_url and not configured_base_url:
            env["ANTHROPIC_BASE_URL"] = proxy_base_url
            env.setdefault("ANTHROPIC_AUTH_TOKEN", "proxy")
            env.setdefault("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1")
            env.setdefault("API_TIMEOUT_MS", "3000000")
            env.setdefault("CLAUDE_CODE_MAX_OUTPUT_TOKENS", "50000")
            env.setdefault("CLAUDE_BASH_NO_LOGIN", "1")
        return env

    def _build_cmd(self) -> list[str]:
        cmd = [self.cli_path, "--setting-sources", "project,local", "--print", "--input-format", "text"]
        if self.model:
            cmd[1:1] = ["--model", self.model]
        if self.effort:
            cmd[1:1] = ["--effort", self.effort]
        return cmd

    def _run_cli(self, prompt: str, *, timeout: int, env: dict) -> subprocess.CompletedProcess[str]:
        cmd = self._build_cmd()
        try:
            return subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
        except OSError as exc:
            if getattr(exc, "errno", None) != 7:
                raise
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
                handle.write(prompt)
                temp_path = handle.name
            try:
                fallback_cmd = self._build_cmd() + [f"@{temp_path}"]
                return subprocess.run(
                    fallback_cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    env=env,
                )
            finally:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

    def execute(self, prompt: str, context: dict, output_mode: str = "code") -> str:
        """Execute via Claude Code CLI."""
        env = self._build_env(context.get("env") if isinstance(context.get("env"), dict) else None)
        original_cwd = os.getcwd()
        if cwd := context.get("cwd"):
            os.chdir(cwd)
        try:
            result = self._run_cli(prompt, timeout=300, env=env)
            if result.returncode != 0:
                stderr = (result.stderr or "").strip()
                stdout = (result.stdout or "").strip()
                detail = stderr or stdout or f"exit_code={result.returncode}"
                raise RuntimeError(f"Claude Code CLI failed: {detail}")

            return self._normalize_output(result.stdout, output_mode)
        finally:
            os.chdir(original_cwd)

    def execute_best_effort(self, prompt: str, context: dict, output_mode: str = "code", timeout: int = 600) -> dict:
        """Execute via Claude Code CLI without discarding partial file edits on failure."""
        env = self._build_env(context.get("env") if isinstance(context.get("env"), dict) else None)
        original_cwd = os.getcwd()
        if cwd := context.get("cwd"):
            os.chdir(cwd)
        try:
            try:
                result = self._run_cli(prompt, timeout=timeout, env=env)
                stdout = result.stdout or ""
                stderr = result.stderr or ""
                payload = {
                    "success": result.returncode == 0,
                    "timed_out": False,
                    "output": self._normalize_output(stdout, output_mode),
                    "raw_output": stdout,
                    "error": "" if result.returncode == 0 else f"Claude Code CLI failed: {stderr or stdout}".strip(),
                    "stderr": stderr,
                    "exit_code": result.returncode,
                    "usage": {},
                }
                self.last_result = payload
                return payload
            except subprocess.TimeoutExpired as exc:
                stdout = exc.stdout or ""
                stderr = exc.stderr or ""
                if isinstance(stdout, bytes):
                    stdout = stdout.decode("utf-8", errors="replace")
                if isinstance(stderr, bytes):
                    stderr = stderr.decode("utf-8", errors="replace")
                payload = {
                    "success": False,
                    "timed_out": True,
                    "output": self._normalize_output(stdout, output_mode),
                    "raw_output": stdout,
                    "error": str(exc),
                    "stderr": stderr,
                    "exit_code": -1,
                    "usage": {},
                }
                self.last_result = payload
                return payload
        finally:
            os.chdir(original_cwd)
