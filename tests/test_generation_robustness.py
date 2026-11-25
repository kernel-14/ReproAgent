"""Offline tests for generation retry diagnostics, the paper spec sheet and repair gating."""

from pathlib import Path

from reproagent.pipeline import paperbench_cli
from reproagent.pipeline.schemas import PaperBenchReproInput, PaperBenchReproState
from reproagent.pipeline.utils import repo_agent_engine as engine_mod


def _engine(tmp_path: Path):
    engine = engine_mod.RepoAgentEngine.__new__(engine_mod.RepoAgentEngine)
    engine.output_dir = tmp_path
    engine.node_artifact_dir = None
    return engine


def test_diagnostic_reports_syntax_error_and_truncation(tmp_path):
    engine = _engine(tmp_path)
    manifest = {"task_inputs": [{"file_path": "src/model.py"}]}
    msg = engine._diagnose_unusable_output("```python\nx = (1,\ny = 2\n", manifest)
    assert "src/model.py" in msg and "SyntaxError at line 1" in msg and "cut off" in msg
    assert "empty" in engine._diagnose_unusable_output("", manifest)


def test_spec_sheet_is_extracted_once_and_injected_before_target(tmp_path, monkeypatch):
    calls = []

    def fake_execute(self, prompt, context, output_mode="text"):
        calls.append(prompt)
        return 'sheet:\n{"hyperparameters": [{"name": "lr", "value": "3e-4", "applies_to": "Adam", "quote": "lr 3e-4"}]}'

    monkeypatch.setattr(engine_mod.LLMWorkflowAgent, "execute", fake_execute)
    engine = _engine(tmp_path)
    state = PaperBenchReproState(input=PaperBenchReproInput(target="t", paper_text="We use lr 3e-4."))
    first = engine._with_paper_spec("Header\n\nTarget:\nX", {"state": state})
    second = engine._with_paper_spec("Header\n\nTarget:\nX", {"state": state})
    assert len(calls) == 1 and first == second
    assert first.index("Paper specification sheet") < first.index("Target:")
    assert '"3e-4"' in first


def test_truncated_spec_sheet_is_kept(tmp_path, monkeypatch):
    def truncated(self, prompt, context, output_mode="text"):
        return '{"hyperparameters": [{"name": "lr", "value": "3e-4"}, {"name": "batch", "val'

    monkeypatch.setattr(engine_mod.LLMWorkflowAgent, "execute", truncated)
    engine = _engine(tmp_path)
    state = PaperBenchReproState(input=PaperBenchReproInput(target="t", paper_text="paper"))
    out = engine._with_paper_spec("Header\n\nTarget:\nX", {"state": state})
    assert "Paper specification sheet" in out and '"3e-4"' in out


def test_spec_sheet_failure_does_not_block_generation(tmp_path, monkeypatch):
    def boom(self, prompt, context, output_mode="text"):
        raise RuntimeError("upstream down")

    monkeypatch.setattr(engine_mod.LLMWorkflowAgent, "execute", boom)
    engine = _engine(tmp_path)
    state = PaperBenchReproState(input=PaperBenchReproInput(target="t", paper_text="paper"))
    out = engine._with_paper_spec("Header\n\nTarget:\nX", {"state": state})
    assert "Implementation priority" in out and "Paper specification sheet" not in out


def test_repair_runs_after_completed_with_risk_but_not_after_failure():
    state = PaperBenchReproState(input=PaperBenchReproInput(target="t"))
    for status in ("completed_with_runtime_risk", "completed_with_degraded_contract", "completed_unverified"):
        state.status = status
        assert paperbench_cli._should_stop_pipeline(state)
        assert not paperbench_cli._should_skip_repair(state)
    state.status = "failed"
    assert paperbench_cli._should_skip_repair(state)
