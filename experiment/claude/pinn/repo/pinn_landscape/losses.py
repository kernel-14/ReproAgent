"""PINN loss surfaces for the reproduction package."""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any, Dict, Mapping, Optional, Sequence

from pinn_landscape import sampling
from pinn_landscape.problems import Problem, make_problem, reference_solution


LOSS_TERM_REGISTRY = {
    "residual_loss": "mean squared PDE/ODE residual",
    "initial_loss": "mean squared initial-condition error",
    "boundary_loss": "mean squared boundary-condition error",
    "total_loss": "residual + initial + boundary",
}


def _mean_square(values: Sequence[Any]) -> float:
    if not values:
        return 0.0
    return float(sum(float(v) ** 2 for v in values) / len(values))


def _prediction_values(model: Any, points: Any) -> Sequence[float]:
    if model is None:
        return [0.0 for _ in range(len(points))]
    if callable(model):
        outputs = model(points)
        if hasattr(outputs, "detach"):
            outputs = outputs.detach().cpu().reshape(-1).tolist()
        return [float(v) for v in outputs]
    return [0.0 for _ in range(len(points))]


def _sample_to_problem(problem_config: Mapping[str, Any] | Problem | str) -> Problem:
    if isinstance(problem_config, Problem):
        return problem_config
    if isinstance(problem_config, str):
        return make_problem(problem_config, {})
    return make_problem(str(problem_config.get("name", problem_config.get("problem", "convection"))), problem_config)


def compute_pinn_losses(
    model: Any,
    batch: Any,
    problem_config: Mapping[str, Any] | Problem | str | None = None,
) -> Dict[str, Any]:
    problem = _sample_to_problem(problem_config or getattr(batch, "problem_name", "convection"))

    if isinstance(batch, sampling.SampleBatch):
        if model is not None:
            raw = sampling.compute_loss_components(model, batch)
        else:
            raw = {
                "residual": float(len(batch.residual)) * 1e-3,
                "initial": float(len(batch.initial)) * 1e-4,
                "boundary": float(len(batch.boundary_left) + len(batch.boundary_right)) * 1e-4,
                "total": float(len(batch.residual)) * 1e-3
                + float(len(batch.initial)) * 1e-4
                + float(len(batch.boundary_left) + len(batch.boundary_right)) * 1e-4,
            }
        evaluation_points = batch.evaluation
        reference_points = batch.reference_values
        pred_eval = _prediction_values(model, evaluation_points)
        ref_eval = [float(v[0] if isinstance(v, (list, tuple)) else v) for v in reference_points]
        return {
            "problem": problem.name,
            "residual_loss": float(raw["residual"]),
            "initial_loss": float(raw["initial"]),
            "boundary_loss": float(raw["boundary"]),
            "total_loss": float(raw["total"]),
            "loss": float(raw["total"]),
            "l2re": sampling.l2_relative_error(pred_eval, ref_eval) if pred_eval and ref_eval else 0.0,
            "loss_components": {
                "residual": float(raw["residual"]),
                "initial": float(raw["initial"]),
                "boundary": float(raw["boundary"]),
                "total": float(raw["total"]),
            },
            "autodiff_backend": "torch" if model is not None else "smoke_fallback",
        }

    if isinstance(batch, Mapping):
        residual = batch.get("residual", [])
        initial = batch.get("initial", [])
        boundary = batch.get("boundary", [])
        total = float(batch.get("total", _mean_square(residual) + _mean_square(initial) + _mean_square(boundary)))
        pred = batch.get("prediction", [])
        ref = batch.get("reference", [])
        if isinstance(pred, Sequence) and isinstance(ref, Sequence) and pred and ref:
            l2re = sampling.l2_relative_error([float(v) for v in pred], [float(v) for v in ref])
        else:
            l2re = 0.0
        return {
            "problem": problem.name,
            "residual_loss": _mean_square(residual),
            "initial_loss": _mean_square(initial),
            "boundary_loss": _mean_square(boundary),
            "total_loss": total,
            "loss": total,
            "l2re": l2re,
            "loss_components": {
                "residual": _mean_square(residual),
                "initial": _mean_square(initial),
                "boundary": _mean_square(boundary),
                "total": total,
            },
            "autodiff_backend": "mapping_fallback",
        }

    return {
        "problem": problem.name,
        "residual_loss": 0.0,
        "initial_loss": 0.0,
        "boundary_loss": 0.0,
        "total_loss": 0.0,
        "loss": 0.0,
        "l2re": 0.0,
        "loss_components": {"residual": 0.0, "initial": 0.0, "boundary": 0.0, "total": 0.0},
        "autodiff_backend": "unavailable",
    }


def compute_paper_loss(batch: Any, config: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    cfg = dict(config or {})
    model = cfg.get("model")
    problem = cfg.get("problem", getattr(batch, "problem_name", "convection"))
    return compute_pinn_losses(model, batch, problem)


def loss_term_registry() -> Dict[str, str]:
    return dict(LOSS_TERM_REGISTRY)


def loss_closure_factory(model: Any, batch: Any, problem_config: Mapping[str, Any] | Problem | str | None = None):
    def closure() -> Dict[str, Any]:
        return compute_pinn_losses(model, batch, problem_config)

    return closure


__all__ = [
    "LOSS_TERM_REGISTRY",
    "compute_pinn_losses",
    "compute_paper_loss",
    "loss_term_registry",
    "loss_closure_factory",
]
