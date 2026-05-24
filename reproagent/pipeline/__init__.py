"""Standalone EMNLP reproduction pipeline based on reproagent."""
from reproagent.pipeline.paperbench_loader import build_repro_input_from_paperbench, load_paperbench_case
from reproagent.pipeline.schemas import PaperBenchReproInput, PaperBenchReproState


def build_workflow(*args, **kwargs):
    """Lazily import and build the reproagent workflow."""
    from reproagent.pipeline.workflow import build_workflow as _build_workflow

    return _build_workflow(*args, **kwargs)


def build_repo_handoff_payload(*args, **kwargs):
    """Lazily import the repository handoff payload builder."""
    from reproagent.pipeline.workflow import build_repo_handoff_payload as _build_repo_handoff_payload

    return _build_repo_handoff_payload(*args, **kwargs)


__all__ = [
    "build_workflow",
    "build_repo_handoff_payload",
    "build_repro_input_from_paperbench",
    "load_paperbench_case",
    "PaperBenchReproInput",
    "PaperBenchReproState",
]
