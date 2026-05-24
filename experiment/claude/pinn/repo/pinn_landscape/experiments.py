"""Experiment orchestration wrappers for the PINN reproduction package."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from pinn_landscape import models, reporting, training
from pinn_landscape.config import ExperimentConfig, expand_experiment_matrix, resolve_config
from pinn_landscape.problems import make_problem
from src import artifact_contract
from src.method_registry import build_experiment_registry_payload, expand_experiment_registry, write_smoke_artifacts


def experiment_registry(
    mode: str = "runtime_smoke",
    max_experiments: Optional[int] = 6,
) -> List[Any]:
    return expand_experiment_registry(mode=mode, max_experiments=max_experiments)


def build_registry_payload(mode: str = "runtime_smoke", max_experiments: Optional[int] = 6) -> Dict[str, Any]:
    return build_experiment_registry_payload(mode=mode, max_experiments=max_experiments)


def run_experiment(
    problem: str = "convection",
    optimizer: str = "Adam",
    width: int = 200,
    seed: int = 0,
    mode: str = "runtime_smoke",
    output_root: str | Path = "results",
    **overrides: Any,
) -> Dict[str, Any]:
    cfg = resolve_config(
        {
            "problem": problem,
            "optimizer": optimizer,
            "width": width,
            "seed": seed,
            "mode": mode,
            **overrides,
        },
        mode=mode,
    )
    model = models.build_model(problem=problem, width=width, seed=seed, prefer_torch=True)
    problem_obj = make_problem(problem, cfg)
    result = training.train(model, problem_obj.name, optimizer, cfg)
    if mode in {"runtime_smoke", "docker_validate", "dry_run"}:
        artifact_contract.write_dry_run_artifacts(output_root=output_root, mode=mode)
    return {
        "config": cfg,
        "result": result,
        "problem": problem,
        "optimizer": optimizer,
        "width": width,
        "seed": seed,
        "mode": mode,
    }


def run_experiments(
    mode: str = "runtime_smoke",
    max_experiments: Optional[int] = 6,
    output_root: str | Path = "results",
) -> List[Dict[str, Any]]:
    rows = expand_experiment_registry(mode=mode, max_experiments=max_experiments)
    results: List[Dict[str, Any]] = []
    for row in rows:
        results.append(
            run_experiment(
                problem=row.problem,
                optimizer=row.optimizer,
                width=row.width,
                seed=row.seed,
                mode=mode,
                output_root=output_root,
                max_steps=row.iteration_count,
            )
        )
    return results


def run_best_checkpoint_refinement_workflows(
    mode: str = "runtime_smoke",
    problems: Sequence[str] = ("convection", "reaction", "wave"),
    width: int = 200,
    seed: int = 0,
) -> Dict[str, Any]:
    """Run problem-scoped Adam+L-BFGS lowest-L2RE checkpoint GD/NNCG workflows."""

    workflows: Dict[str, Any] = {}
    for problem in problems:
        workflows[problem] = {
            "gradient_descent_after_best_adam_lbfgs": training.resume_best_adam_lbfgs_with_gradient_descent(
                problem,
                width=width,
                seed=seed,
                mode=mode,
                max_steps=4,
                steps=2,
            ),
            "nncg_after_best_adam_lbfgs": training.resume_best_adam_lbfgs_with_nncg(
                problem,
                width=width,
                seed=seed,
                mode=mode,
                max_steps=4,
                steps=1,
                rank=4,
            ),
        }
    return {
        "mode": mode,
        "selection_rule": "filter Section 2.2 Adam+L-BFGS sweep by problem, choose min final L2RE, reload checkpoint, resume",
        "workflows": workflows,
    }


def run_phasewise_pointwise_error_workflows(
    mode: str = "runtime_smoke",
    problems: Sequence[str] = ("convection", "reaction", "wave"),
    width: int = 200,
    seed: int = 0,
) -> Dict[str, Any]:
    """Compute pointwise absolute errors at Adam/L-BFGS/GD/NNCG phase ends."""

    return {
        "mode": mode,
        "metric": "phase-wise pointwise absolute error",
        "results": {
            problem: training.phasewise_pointwise_absolute_error_workflow(
                problem,
                width=width,
                seed=seed,
                mode=mode,
            )
            for problem in problems
        },
    }


def materialize_experiments(
    output_root: str | Path = "results",
    mode: str = "runtime_smoke",
    max_experiments: int = 6,
) -> Dict[str, Any]:
    return write_smoke_artifacts(output_root=output_root, mode=mode, max_experiments=max_experiments)


def experiment_manifest(mode: str = "runtime_smoke") -> Dict[str, Any]:
    return {
        "mode": mode,
        "registered_experiments": [asdict(row) for row in expand_experiment_registry(mode=mode, max_experiments=6)],
        "registry_payload": build_registry_payload(mode=mode),
    }


__all__ = [
    "experiment_registry",
    "build_registry_payload",
    "run_experiment",
    "run_experiments",
    "run_best_checkpoint_refinement_workflows",
    "run_phasewise_pointwise_error_workflows",
    "materialize_experiments",
    "experiment_manifest",
]
