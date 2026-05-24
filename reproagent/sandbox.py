"""Local sandbox provider used by the standalone reproduction pipeline."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any


class LocalSandbox:
    """Small local sandbox compatible with reproagent's expected interface."""

    def __init__(self, root: Path | None = None):
        self.root = root or Path(tempfile.mkdtemp(prefix="emnlp_repro_sandbox_"))

    def write_file(self, path: str, content: str) -> None:
        target = Path(path)
        if not target.is_absolute():
            target = self.root / target
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    def execute_command(self, command: str) -> dict[str, Any]:
        result = subprocess.run(
            command,
            shell=True,
            cwd=self.root,
            capture_output=True,
            text=True,
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.returncode,
        }


class LocalSandboxProvider:
    """In-process sandbox provider compatible with backend reproagent."""

    def __init__(self):
        self._sandboxes: dict[str, LocalSandbox] = {}

    def acquire(self) -> str:
        sandbox_id = f"local_{len(self._sandboxes) + 1}"
        self._sandboxes[sandbox_id] = LocalSandbox()
        return sandbox_id

    def get(self, sandbox_id: str) -> LocalSandbox | None:
        return self._sandboxes.get(sandbox_id)

    def release(self, sandbox_id: str) -> None:
        self._sandboxes.pop(sandbox_id, None)


_PROVIDER = LocalSandboxProvider()


def get_sandbox_provider() -> LocalSandboxProvider:
    """Return the standalone local sandbox provider."""
    return _PROVIDER
