"""Run-scoped path, serialization, and agent-context helpers for reproagent."""

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from reproagent.pipeline.schemas import PaperBenchReproState
from reproagent.pipeline.skills_config import get_skills_dir

def _new_run_id() -> str:
    """Generate unique run_id with microsecond precision + UUID."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    return f"{ts}_{uuid4().hex[:8]}"


def _get_output_dir(state: PaperBenchReproState) -> Path:
    """Get output directory for this run."""
    base = Path(__file__).resolve().parents[3] / "output" / "reproagent"
    return base / state.run_id


def _json_default(value):
    """Best-effort serializer for workflow state."""
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return str(value)


def _build_agent_context() -> dict:
    """Build CLI agent execution context for reproagent."""
    project_root = Path(__file__).resolve().parents[3]
    return {
        "cwd": str(project_root),
        "skills_dir": str(get_skills_dir()),
    }
