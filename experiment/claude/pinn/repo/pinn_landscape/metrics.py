"""Metric helpers for the PINN reproduction package."""

from __future__ import annotations

import statistics
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from pinn_landscape.losses import compute_pinn_losses
from pinn_landscape.sampling import l2_relative_error as sampling_l2re
from src.method_registry import aggregate_lowest_l2re_by_sample


def l2_relative_error(prediction: Sequence[float], reference: Sequence[float], eps: float = 1e-12) -> float:
    return sampling_l2re(prediction, reference, eps=eps)


def evaluate_model(model: Any, batch: Any, problem_config: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    return compute_pinn_losses(model, batch, problem_config)


def aggregate_seed_statistics(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[Tuple[str, str, int], List[Mapping[str, Any]]] = {}
    for row in records:
        key = (str(row.get("problem")), str(row.get("optimizer")), int(row.get("width", 0)))
        grouped.setdefault(key, []).append(row)
    summary: List[Dict[str, Any]] = []
    for (problem, optimizer, width), rows in sorted(grouped.items()):
        losses = [float(row.get("loss", row.get("total_loss", 0.0))) for row in rows]
        l2res = [float(row.get("L2RE", row.get("l2re", 0.0))) for row in rows]
        gnorms = [float(row.get("gradient_norm", 0.0)) for row in rows]
        summary.append(
            {
                "problem": problem,
                "optimizer": optimizer,
                "width": width,
                "n_seeds": len(rows),
                "mean_loss": statistics.fmean(losses) if losses else 0.0,
                "mean_L2RE": statistics.fmean(l2res) if l2res else 0.0,
                "mean_gradient_norm": statistics.fmean(gnorms) if gnorms else 0.0,
                "min_loss": min(losses) if losses else 0.0,
                "min_L2RE": min(l2res) if l2res else 0.0,
            }
        )
    return {"summary": summary, "grouped": grouped}


def summarize_lowest_l2re(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return aggregate_lowest_l2re_by_sample(records)


def metric_schema() -> Dict[str, Any]:
    return {
        "loss": "total PINN objective",
        "residual_loss": "mean squared residual component",
        "initial_loss": "mean squared initial-condition component",
        "boundary_loss": "mean squared boundary-condition component",
        "L2RE": "relative L2 error",
        "gradient_norm": "Euclidean gradient norm or proxy",
        "condition_number": "Hessian condition proxy",
        "training_time": "seconds per optimizer step",
        "fidelity_score": "artifact and trend-completeness score",
    }


__all__ = [
    "l2_relative_error",
    "evaluate_model",
    "aggregate_seed_statistics",
    "summarize_lowest_l2re",
    "metric_schema",
]
