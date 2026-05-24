"""CLI entrypoint for PaperBench reproduction runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from reproagent.pipeline.paperbench_loader import (
    build_repro_input_from_paperbench,
    default_paperbench_data_root,
)
from reproagent.pipeline.schemas import PaperBenchReproInput, PaperBenchReproState
from reproagent.pipeline.utils.quality_status import refresh_quality_status


_PLAN_RESUME_STAGES = {
    "topic_profile_synthesis",
    "work_package_planning",
    "package_evidence_grounding",
    "reference_selection",
    "pipeline_plan",
    "global_contract_synthesis",
    "architecture_planning",
    "package_file_planning",
    "canonical_ir_synthesis",
}
_REPAIR_RESUME_STAGES = {"repair_validation"}
_RESUME_STAGE_ALIASES = {
    "plan": "topic_profile_synthesis",
    "generate": "local_file_generation",
    "codegen-only": "local_file_generation",
    "repair": "repair_validation",
}


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return str(value)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the standalone reproagent-based PaperBench code reproduction pipeline.",
    )
    parser.add_argument(
        "case",
        help="PaperBench case id under paperbench_data, or an explicit case directory.",
    )
    parser.add_argument(
        "--data-root",
        default=str(default_paperbench_data_root()),
        help="PaperBench data root. Defaults to ./paperbench_data in this workspace.",
    )
    parser.add_argument("--run-id", default="", help="Optional run id for output/reproagent/<run-id>.")
    parser.add_argument(
        "--fork-from-run-id",
        default="",
        help="Copy reusable artifacts from an existing run into this new run before executing.",
    )
    parser.add_argument(
        "--resume-from-run-id",
        default="",
        help="Reuse an existing run id/output directory instead of creating a fresh run.",
    )
    parser.add_argument(
        "--resume-in-place",
        action="store_true",
        help="When used with --resume-from-run-id, invalidate downstream state from --resume-start-stage.",
    )
    parser.add_argument(
        "--resume-start-stage",
        default="",
        help="Stage name to restart when --resume-in-place is set.",
    )
    parser.add_argument("--language", default="zh", help="Prompt/output language for planning stages.")
    parser.add_argument("--chunk-max-chars", type=int, default=6000, help="Maximum paper chunk size.")
    parser.add_argument("--max-iterations", type=int, default=30, help="Generation iteration budget.")
    parser.add_argument(
        "--clone-references",
        dest="clone_references",
        action="store_true",
        default=True,
        help="Clone non-blacklisted GitHub references detected in paper/addendum. This is the default.",
    )
    parser.add_argument(
        "--no-clone-references",
        dest="clone_references",
        action="store_false",
        help="Do not clone detected GitHub references; keep them as summaries only.",
    )
    parser.add_argument(
        "--stage",
        choices=["start", "prepare", "plan", "codegen-only", "generate", "repair"],
        default="generate",
        help=(
            "start: deterministic input/chunk/materialization only; prepare: add LLM normalization/units; "
            "plan: prepare plus a complete executable reproduction contract; "
            "codegen-only: diagnostic local materialization that is not judge-eligible unless handoff validation passes; "
            "generate: faithful complete judgeable reproduction repo with repo-level validation before repair; "
            "repair: bounded post-generate patching."
        ),
    )
    parser.add_argument(
        "--print-input",
        action="store_true",
        help="Print the PaperBenchReproInput JSON and exit without importing/running workflow nodes.",
    )
    return parser


def _summary(state: PaperBenchReproState, output_dir: Path | None = None) -> dict[str, Any]:
    quality = refresh_quality_status(state).model_dump(mode="json")
    return {
        "run_id": state.run_id,
        "status": state.status,
        "terminal_outcome": state.terminal_outcome,
        "terminal_outcome_reason": state.terminal_outcome_reason,
        "quality_status": quality,
        "ready_for_open_source": bool(quality.get("handoff_ready", False)),
        "failed_node": state.failed_node,
        "error_message": state.error_message,
        "project_root": state.project_root,
        "generated_files": list(state.generated_files),
        "paper_chunk_count": len(state.paper_chunks),
        "output_dir": str(output_dir) if output_dir else "",
    }


def _should_stop_pipeline(state: PaperBenchReproState) -> bool:
    return str(getattr(state, "status", "") or "") in {
        "failed",
        "completed_unverified",
        "completed_with_degraded_contract",
        "completed_with_runtime_risk",
    }


def _normalize_resume_start_stage(stage_name: str) -> str:
    normalized = str(stage_name or "").strip()
    return _RESUME_STAGE_ALIASES.get(normalized, normalized)


def run_stage(repro_input: PaperBenchReproInput, *, stage: str, run_id: str = "") -> PaperBenchReproState:
    """Run a selected PaperBench pipeline stage."""
    from reproagent.pipeline import workflow

    state = PaperBenchReproState(input=repro_input, run_id=run_id)
    resume_stage = _normalize_resume_start_stage(getattr(repro_input, "resume_start_stage", ""))
    if resume_stage != str(getattr(repro_input, "resume_start_stage", "") or "").strip():
        repro_input.resume_start_stage = resume_stage
    if bool(getattr(repro_input, "resume_in_place", False)) and resume_stage:
        state = workflow._hydrate_state_for_in_place_resume(state)
    if stage == "start":
        state = workflow._start_impl(state)
        if state.status == "pending":
            state.status = "completed"
        return state
    if stage == "prepare":
        return workflow.prepare_node(state)
    if stage == "plan":
        if not (bool(getattr(repro_input, "resume_in_place", False)) and resume_stage in _PLAN_RESUME_STAGES):
            state = workflow.prepare_node(state)
        state = workflow.plan_node(state)
        return state
    if stage in {"codegen-only", "generate"}:
        if bool(getattr(repro_input, "resume_in_place", False)) and resume_stage == "local_file_generation":
            state = workflow.normalization_gate_node(state)
            if _should_stop_pipeline(state):
                return state
            state = workflow.generate_node(state)
            return state
        if not (bool(getattr(repro_input, "resume_in_place", False)) and resume_stage in _PLAN_RESUME_STAGES):
            state = workflow.prepare_node(state)
            if _should_stop_pipeline(state):
                return state
        state = workflow.plan_node(state)
        if _should_stop_pipeline(state):
            return state
        state = workflow.normalization_gate_node(state)
        if _should_stop_pipeline(state):
            return state
        state = workflow.generate_node(state)
        return state
    if stage == "repair":
        if bool(getattr(repro_input, "resume_in_place", False)) and resume_stage in _REPAIR_RESUME_STAGES:
            state = workflow.repair_node(state)
            return state
        if not (bool(getattr(repro_input, "resume_in_place", False)) and resume_stage in _PLAN_RESUME_STAGES):
            state = workflow.prepare_node(state)
            if _should_stop_pipeline(state):
                return state
        state = workflow.plan_node(state)
        if _should_stop_pipeline(state):
            return state
        state = workflow.normalization_gate_node(state)
        if _should_stop_pipeline(state):
            return state
        state = workflow.generate_node(state)
        if _should_stop_pipeline(state) or not str(getattr(state, "project_root", "") or "").strip():
            return state
        state = workflow.repair_node(state)
        return state
    raise ValueError(f"unsupported stage: {stage}")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    repro_input = build_repro_input_from_paperbench(
        args.case,
        data_root=args.data_root,
        language=args.language,
        chunk_max_chars=args.chunk_max_chars,
        clone_references=args.clone_references,
        max_iterations=args.max_iterations,
    )
    repro_input.fork_from_run_id = str(args.fork_from_run_id or "").strip()
    repro_input.resume_from_run_id = str(args.resume_from_run_id or "").strip()
    repro_input.resume_in_place = bool(args.resume_in_place)
    repro_input.resume_start_stage = _normalize_resume_start_stage(args.resume_start_stage)
    if args.print_input:
        print(json.dumps(repro_input.model_dump(mode="json"), indent=2, ensure_ascii=False, default=_json_default))
        return 0

    state = run_stage(repro_input, stage=args.stage, run_id=args.run_id)
    output_dir = None
    try:
        from reproagent.pipeline import workflow

        output_dir = workflow._get_output_dir(state) if state.run_id else None
    except Exception:
        output_dir = None
    print(json.dumps(_summary(state, output_dir), indent=2, ensure_ascii=False, default=_json_default))
    if _should_stop_pipeline(state):
        return 1
    if str(getattr(state, "status", "") or "") not in {"completed", "pending"}:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
