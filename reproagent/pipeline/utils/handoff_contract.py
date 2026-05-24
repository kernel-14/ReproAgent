"""Helpers for the PaperBench Repro repository handoff contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from reproagent.pipeline.schemas import PaperBenchReproState


def _dedupe_text(items: list[object]) -> list[str]:
    values: list[str] = []
    for item in items:
        text = str(item or "").strip()
        if text and text not in values:
            values.append(text)
    return values


def canonical_main_entry(state: PaperBenchReproState) -> str:
    """Return the generated repo's canonical main entry surface."""

    if state.repo_plan is not None:
        candidate = str(state.repo_plan.canonical_route.entry_surface or "").strip()
        if candidate:
            return candidate
    if state.project_plan is not None:
        return str(state.project_plan.entrypoints.get("main", "main.py") or "main.py").strip()
    return "main.py"


def _benchmark_flag_for_name(name: str) -> str | None:
    normalized = str(name or "").strip().lower().replace("-", "_")
    if not normalized:
        return None
    if "autoscholar" in normalized:
        return "autoscholarquery"
    if "spar_bench" in normalized or normalized == "spar":
        return "spar_bench"
    return None


def _build_evaluate_command(state: PaperBenchReproState, main_entry: str) -> str:
    benchmark_items = list((state.input.experiment_design or {}).get("benchmarks", []) or [])
    selected: list[str] = []
    autoscholar_path = ""
    spar_bench_path = ""
    for item in benchmark_items:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "") or item.get("benchmark") or "")
        path = str(item.get("path", "") or item.get("benchmark_path", "") or "").strip()
        flag = _benchmark_flag_for_name(name)
        if flag and flag not in selected:
            selected.append(flag)
        if flag == "autoscholarquery" and path and not autoscholar_path:
            autoscholar_path = path
        if flag == "spar_bench" and path and not spar_bench_path:
            spar_bench_path = path

    benchmark_value = "both"
    if selected == ["autoscholarquery"]:
        benchmark_value = "autoscholarquery"
    elif selected == ["spar_bench"]:
        benchmark_value = "spar_bench"
    elif selected:
        benchmark_value = "both"

    if main_entry.endswith("src/spar/cli.py") or main_entry == "src/spar/cli.py":
        command = f"python {main_entry} --mode evaluate --benchmark {benchmark_value}"
        if autoscholar_path:
            command += f" --autoscholar-path {autoscholar_path}"
        if spar_bench_path:
            command += f" --spar-bench-path {spar_bench_path}"
        return command
    return f"python {main_entry}"


def build_stage1_repo_contract(
    state: PaperBenchReproState,
    *,
    repo_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build the explicit Stage1 repository contract for downstream consumers."""

    main_entry = canonical_main_entry(state)
    project_plan = state.project_plan
    entrypoints = dict(project_plan.entrypoints if project_plan is not None else {})
    runtime_contract = dict(project_plan.runtime_contract if project_plan is not None else {})

    smoke_command = str(entrypoints.get("runtime_smoke", "") or "").strip()
    if not smoke_command:
        smoke_command = f"python {main_entry}" if main_entry else ""

    evaluate_command = _build_evaluate_command(state, main_entry)

    baseline_command = str(
        runtime_contract.get("baseline_command")
        or entrypoints.get("baseline_command")
        or evaluate_command
    ).strip()
    idea_command = str(
        runtime_contract.get("idea_command")
        or entrypoints.get("idea_command")
        or evaluate_command
    ).strip()
    variant_command = str(
        runtime_contract.get("variant_command")
        or entrypoints.get("variant_command")
        or evaluate_command
    ).strip()

    editable_paths = (
        [item.target_file for item in state.repo_plan.files]
        if state.repo_plan is not None and state.repo_plan.files
        else list(state.generated_files)
    )
    editable_paths = _dedupe_text(editable_paths)

    artifact_contract = project_plan.artifact_contract if project_plan is not None else None
    required_files = list(artifact_contract.required_files if artifact_contract is not None else [])
    optional_files = list(artifact_contract.optional_files if artifact_contract is not None else [])
    metric_paths = _dedupe_text(required_files or [getattr(artifact_contract, "metrics_path", "results/metrics.json")])
    if not metric_paths:
        metric_paths = ["results/metrics.json"]

    protected_paths: list[object] = []
    if state.repo_plan is not None:
        protected_paths.extend(path for path in state.repo_plan.artifact_paths if path)
    protected_paths.extend(required_files)
    protected_paths.extend(optional_files)
    protected_paths.append("results")

    install_command = ""
    if "requirements.txt" in state.generated_files:
        install_command = "pip install -r requirements.txt"

    return {
        "repo_path": str(repo_path or state.project_root or "").strip(),
        "repo_source": "reproagent_repo_validated_repo",
        "entrypoint_hint": main_entry,
        "install_command": install_command,
        "variant_mode": "unknown",
        "baseline_command": baseline_command,
        "idea_command": idea_command,
        "variant_command": variant_command,
        "smoke_command": smoke_command,
        "command_contract": {
            "type": "bash",
            "validated_by_reproagent": True,
            "exprun_must_not_append_cli_args": True,
            "supports_variant_placeholder": "{variant}" in variant_command,
            "supports_seed_placeholder": "{seed}" in " ".join([baseline_command, idea_command, variant_command]),
            "supports_subset_cap_placeholder": "{subset_cap}" in " ".join([baseline_command, idea_command, variant_command]),
        },
        "metric_paths": metric_paths,
        "editable_paths": editable_paths,
        "protected_paths": _dedupe_text(protected_paths),
    }
