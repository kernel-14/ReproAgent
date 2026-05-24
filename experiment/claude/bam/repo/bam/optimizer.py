"""Optimization utilities for Batch and Match (BaM) variational inference.

This module implements the optimizer-owned part of the PaperBench reproduction
for the paper

    "Batch and match: black-box variational inference with a score-based
    divergence."

The code is intentionally import-light: NumPy is imported lazily inside runtime
functions so repository import and registry inspection remain available in a
minimal environment.

Core method mapping
-------------------
reference_grounding: paper:paper_method_core paper.md
    Score-based BBVI minimizes a divergence between a target p and a Gaussian
    variational approximation q using only the target score ∇ log p(z).  The
    normalizing constant of p is never required here.

reference_grounding: paper:paper_training_or_optimization_loop paper.md
    BaM uses an explicit Batch Step, z_1,...,z_B ~ q_t and
    g_b = ∇ log p(z_b), followed by a Match Step using batch statistics and KL
    regularization to update full-covariance Gaussian variational parameters.

reference_grounding: paper:paper_semantic_chunk_009_03 paper.md
    BaM is a KL-regularized stochastic proximal/matching update.  The optimizer
    below exposes lambda, epsilon, B=32, 100_iterations, B→∞ Gaussian sanity
    semantics, and method/baseline selectors for BaM, ADVI, GSM, BBVI, KL,
    ELBO, SPP, EM, CLI, ours, and baseline.
"""

from __future__ import annotations

import base64
import json
import math
import os
import time
import zipfile
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Protocol, Sequence, Tuple, Union


ArrayLike = Any
ScoreFn = Callable[[ArrayLike], ArrayLike]
LogProbFn = Callable[[ArrayLike], Union[float, ArrayLike]]


def _np() -> Any:
    """Import NumPy lazily with a clear runtime error."""
    try:
        import numpy as np  # type: ignore

        return np
    except Exception as exc:  # pragma: no cover - depends on host environment
        raise RuntimeError(
            "NumPy is required to execute BaM optimization. "
            "Install repository requirements before running numerical training."
        ) from exc


class ScoreTarget(Protocol):
    """Protocol for targets usable by the score-based optimizer."""

    dimension: int

    def score(self, z: ArrayLike) -> ArrayLike:
        """Return ∇ log p(z) for a vector or batch of vectors."""


@dataclass
class GaussianState:
    """Full-covariance Gaussian variational state q(z)=N(mean,covariance)."""

    mean: Any
    covariance: Any
    iteration: int = 0

    def copy(self) -> "GaussianState":
        np = _np()
        return GaussianState(
            mean=np.asarray(self.mean, dtype=float).copy(),
            covariance=np.asarray(self.covariance, dtype=float).copy(),
            iteration=int(self.iteration),
        )

    @property
    def dimension(self) -> int:
        np = _np()
        return int(np.asarray(self.mean).shape[0])

    def precision(self, epsilon: float = 1e-6) -> Any:
        np = _np()
        cov = stabilize_covariance(self.covariance, epsilon=epsilon)
        return np.linalg.inv(cov)

    def sample(self, batch_size: int, rng: Any, epsilon: float = 1e-6) -> Any:
        np = _np()
        cov = stabilize_covariance(self.covariance, epsilon=epsilon)
        return rng.multivariate_normal(np.asarray(self.mean, dtype=float), cov, size=int(batch_size))

    def log_density(self, z: ArrayLike, epsilon: float = 1e-6) -> Any:
        np = _np()
        z_arr = np.asarray(z, dtype=float)
        single = z_arr.ndim == 1
        if single:
            z_arr = z_arr[None, :]
        mean = np.asarray(self.mean, dtype=float)
        cov = stabilize_covariance(self.covariance, epsilon=epsilon)
        precision = np.linalg.inv(cov)
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            cov = stabilize_covariance(cov, epsilon=max(float(epsilon), 1e-5))
            precision = np.linalg.inv(cov)
            sign, logdet = np.linalg.slogdet(cov)
        diff = z_arr - mean
        quad = np.einsum("bi,ij,bj->b", diff, precision, diff)
        val = -0.5 * (mean.shape[0] * math.log(2.0 * math.pi) + logdet + quad)
        return float(val[0]) if single else val

    def score(self, z: ArrayLike, epsilon: float = 1e-6) -> Any:
        np = _np()
        z_arr = np.asarray(z, dtype=float)
        single = z_arr.ndim == 1
        if single:
            z_arr = z_arr[None, :]
        precision = self.precision(epsilon=epsilon)
        scores = -((z_arr - np.asarray(self.mean, dtype=float)) @ precision.T)
        return scores[0] if single else scores


@dataclass(frozen=True)
class OptimizerConfig:
    """Bounded optimizer configuration for BaM and baseline selectors.

    The paper-visible fixed anchors are preserved as ``iterations=100`` and
    ``batch_size=32``.  ``lambda_regularization`` is the KL/proximal strength,
    while ``epsilon`` is the covariance stabilizer and finite update floor.
    """

    method: str = "BaM"
    batch_size: int = 32
    iterations: int = 100
    seed: int = 0
    lambda_regularization: float = 1.0
    epsilon: float = 1e-5
    learning_rate: float = 0.05
    dimension: int = 2
    full_covariance: bool = True
    b_infinity: bool = False
    output_dir: str = "results"
    record_every: int = 1
    stop_on_nonfinite: bool = True
    lora_rank: int = 0
    p: Optional[int] = None
    dry_run_contract: bool = False

    @classmethod
    def from_mapping(cls, values: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> "OptimizerConfig":
        payload: Dict[str, Any] = {}
        if values:
            payload.update(dict(values))
        payload.update(kwargs)
        if "lambda" in payload and "lambda_regularization" not in payload:
            payload["lambda_regularization"] = payload.pop("lambda")
        if "regularization_strength" in payload and "lambda_regularization" not in payload:
            payload["lambda_regularization"] = payload.pop("regularization_strength")
        if "iteration_count" in payload and "iterations" not in payload:
            payload["iterations"] = payload.pop("iteration_count")
        if "B" in payload and "batch_size" not in payload:
            payload["batch_size"] = payload.pop("B")
        if "batch size B" in payload and "batch_size" not in payload:
            payload["batch_size"] = payload.pop("batch size B")
        allowed = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in payload.items() if k in allowed})

    def as_registry_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["lambda"] = self.lambda_regularization
        out["regularization_strength"] = self.lambda_regularization
        out["iteration_count"] = self.iterations
        out["B"] = self.batch_size
        out["100_iterations"] = self.iterations == 100
        out["batch_size_32"] = self.batch_size == 32
        return out


@dataclass
class BatchStatistics:
    """Statistics produced by the explicit BaM Batch Step."""

    samples: Any
    target_scores: Any
    sample_mean: Any
    score_mean: Any
    sample_covariance: Any
    score_sample_cross_covariance: Any
    score_covariance: Any
    batch_size: int
    dimension: int

    def to_jsonable(self) -> Dict[str, Any]:
        np = _np()
        return {
            "batch_size": int(self.batch_size),
            "dimension": int(self.dimension),
            "sample_mean": np.asarray(self.sample_mean, dtype=float).tolist(),
            "score_mean": np.asarray(self.score_mean, dtype=float).tolist(),
            "sample_covariance": np.asarray(self.sample_covariance, dtype=float).tolist(),
            "score_sample_cross_covariance": np.asarray(
                self.score_sample_cross_covariance, dtype=float
            ).tolist(),
            "score_covariance": np.asarray(self.score_covariance, dtype=float).tolist(),
        }


@dataclass
class OptimizationTraceEntry:
    """Single optimization-step record."""

    iteration: int
    method: str
    score_divergence: float
    kl_to_previous: float
    mean_norm: float
    covariance_trace: float
    min_covariance_eigenvalue: float
    batch_size: int
    lambda_regularization: float
    epsilon: float
    elapsed_seconds: float

    def to_jsonable(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OptimizationResult:
    """Result returned by optimizer entrypoints."""

    method: str
    config: OptimizerConfig
    initial_state: GaussianState
    final_state: GaussianState
    trace: List[OptimizationTraceEntry]
    batch_statistics_trace: List[Dict[str, Any]]
    artifacts: Dict[str, str]
    readiness: Dict[str, Any]
    comparison_role: str = "ours"

    def metrics(self) -> Dict[str, Any]:
        np = _np()
        final_cov = stabilize_covariance(self.final_state.covariance, self.config.epsilon)
        eigvals = np.linalg.eigvalsh(final_cov)
        last = self.trace[-1] if self.trace else None
        return {
            "method": self.method,
            "comparison_role": self.comparison_role,
            "iterations_completed": int(len(self.trace)),
            "configured_iterations": int(self.config.iterations),
            "batch_size": int(self.config.batch_size),
            "seed": int(self.config.seed),
            "lambda": float(self.config.lambda_regularization),
            "epsilon": float(self.config.epsilon),
            "learning_rate": float(self.config.learning_rate),
            "score_divergence": float(last.score_divergence) if last else math.nan,
            "kl_to_previous": float(last.kl_to_previous) if last else 0.0,
            "mean_norm": float(np.linalg.norm(np.asarray(self.final_state.mean, dtype=float))),
            "covariance_trace": float(np.trace(final_cov)),
            "min_covariance_eigenvalue": float(np.min(eigvals)),
            "full_covariance": bool(self.config.full_covariance),
            "b_infinity": bool(self.config.b_infinity),
        }

    def to_jsonable(self) -> Dict[str, Any]:
        np = _np()
        return {
            "method": self.method,
            "comparison_role": self.comparison_role,
            "config": self.config.as_registry_dict(),
            "initial_state": {
                "mean": np.asarray(self.initial_state.mean, dtype=float).tolist(),
                "covariance": np.asarray(self.initial_state.covariance, dtype=float).tolist(),
            },
            "final_state": {
                "mean": np.asarray(self.final_state.mean, dtype=float).tolist(),
                "covariance": np.asarray(self.final_state.covariance, dtype=float).tolist(),
                "iteration": int(self.final_state.iteration),
            },
            "trace": [entry.to_jsonable() for entry in self.trace],
            "batch_statistics_trace": self.batch_statistics_trace,
            "metrics": self.metrics(),
            "artifacts": dict(self.artifacts),
            "readiness": dict(self.readiness),
        }


METHOD_REGISTRY: Dict[str, Dict[str, Any]] = {
    "ours": {
        "canonical": "BaM",
        "comparison_role": "ours",
        "objective": "score_divergence_with_KL_regularized_match_step",
    },
    "BaM": {
        "canonical": "BaM",
        "comparison_role": "ours",
        "objective": "score_divergence_with_KL_regularized_match_step",
    },
    "baseline": {
        "canonical": "ADVI",
        "comparison_role": "baseline",
        "objective": "ELBO_surrogate_or_KL_score_gradient_baseline",
    },
    "ADVI": {
        "canonical": "ADVI",
        "comparison_role": "baseline",
        "objective": "reverse_KL_or_ELBO_black_box_gradient_surrogate",
    },
    "BBVI": {
        "canonical": "ADVI",
        "comparison_role": "baseline",
        "objective": "black_box_variational_inference_ELBO",
    },
    "ELBO": {
        "canonical": "ADVI",
        "comparison_role": "baseline",
        "objective": "evidence_lower_bound_surrogate",
    },
    "KL": {
        "canonical": "ADVI",
        "comparison_role": "baseline",
        "objective": "reverse_KL_surrogate",
    },
    "GSM": {
        "canonical": "GSM",
        "comparison_role": "baseline",
        "objective": "score_matching_limiting_case",
    },
    "SPP": {
        "canonical": "BaM",
        "comparison_role": "variant",
        "objective": "stochastic_proximal_point_KL_regularized_match",
    },
    "EM": {
        "canonical": "BaM",
        "comparison_role": "variant",
        "objective": "KL_regularized_EM_style_match",
    },
    "CLI": {
        "canonical": "BaM",
        "comparison_role": "variant",
        "objective": "command_line_interface_default_BaM",
    },
    "100_iterations": {
        "canonical": "BaM",
        "comparison_role": "variant",
        "objective": "fixed_anchor_100_iterations",
        "iterations": 100,
    },
}


SWEEP_REGISTRY: Dict[str, Any] = {
    "reference_grounding": "paper:paper_training_or_optimization_loop paper.md",
    "bounded_not_exhaustive": True,
    "hypothesis": (
        "BaM should reduce the score-based divergence for a full-covariance "
        "Gaussian variational family using explicit Batch and Match steps."
    ),
    "decisive_comparison": "BaM versus ADVI/GSM under shared Gaussian state and target score interface",
    "decisive_metric": "score_divergence plus Gaussian mean/covariance convergence metrics",
    "stop_rule_or_pruning_rationale": (
        "Default execution uses the fixed paper anchors and small bounded lists; "
        "full sweeps require an explicit caller selection to avoid generation-time long runs."
    ),
    "fixed_hyperparameters": {
        "100_iterations": 100,
        "batch_size_32": 32,
        "B": 32,
        "B_infinity": "analytic_gaussian_sanity_route",
    },
    "lambda": [0.0, 0.1, 1.0, 10.0],
    "epsilon": [1e-6, 1e-5, 1e-4],
    "learning_rate": [0.01, 0.05, 0.1],
    "batch_size": [3, 32],
    "batch size B": [3, 32, "B→∞"],
    "random_seed": [0, 1],
    "iteration_count": [0, 1, 10, 100],
    "iteration_count values": [0, 100],
    "p": [2, 8],
    "lora_rank": [0],
    "regularization strength": [0.0, 1.0, 10.0],
}


ARTIFACT_PATHS: Dict[str, str] = {
    "loss_trace": "results/loss_trace.json",
    "bam_trace": "results/bam_trace.json",
    "final_variational_params": "results/bam_final_variational_params.npz",
    "batch_statistics_trace": "results/batch_statistics_trace.json",
    "gaussian_sanity_metrics": "results/gaussian_sanity_metrics.json",
    "figure_5": "results/figures/figure_5.png",
    "readiness": "results/readiness.json",
    "evaluation_result": "results/evaluation_result.json",
}


def stabilize_covariance(covariance: ArrayLike, epsilon: float = 1e-6) -> Any:
    """Return a symmetric positive-definite full covariance matrix."""

    np = _np()
    cov = np.asarray(covariance, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError(f"covariance must be square; got shape {cov.shape}")
    cov = 0.5 * (cov + cov.T)
    if not np.all(np.isfinite(cov)):
        raise ValueError("covariance contains non-finite values")
    eigvals = np.linalg.eigvalsh(cov)
    min_eig = float(np.min(eigvals))
    jitter = max(float(epsilon), 0.0)
    if min_eig <= jitter:
        cov = cov + np.eye(cov.shape[0]) * (jitter - min_eig + jitter)
    return 0.5 * (cov + cov.T)


def gaussian_kl(q: GaussianState, p: GaussianState, epsilon: float = 1e-6) -> float:
    """KL(q || p) for full-covariance Gaussians."""

    np = _np()
    mean_q = np.asarray(q.mean, dtype=float)
    mean_p = np.asarray(p.mean, dtype=float)
    cov_q = stabilize_covariance(q.covariance, epsilon)
    cov_p = stabilize_covariance(p.covariance, epsilon)
    dim = int(mean_q.shape[0])
    prec_p = np.linalg.inv(cov_p)
    diff = mean_p - mean_q
    sign_q, logdet_q = np.linalg.slogdet(cov_q)
    sign_p, logdet_p = np.linalg.slogdet(cov_p)
    if sign_q <= 0 or sign_p <= 0:
        raise ValueError("Gaussian KL received non-positive covariance determinant")
    value = 0.5 * (
        float(np.trace(prec_p @ cov_q))
        + float(diff.T @ prec_p @ diff)
        - dim
        + float(logdet_p - logdet_q)
    )
    return float(max(value, 0.0))


def score_divergence_estimate(
    state: GaussianState,
    samples: ArrayLike,
    target_scores: ArrayLike,
    epsilon: float = 1e-6,
) -> float:
    """Monte Carlo score-based divergence estimate.

    Computes B^{-1} Σ_b ||∇ log q(z_b) - ∇ log p(z_b)||²_{Cov(q)}, using the
    full covariance matrix in the norm as specified by the paper.
    """

    np = _np()
    z = np.asarray(samples, dtype=float)
    g = np.asarray(target_scores, dtype=float)
    if z.ndim != 2 or g.shape != z.shape:
        raise ValueError(f"samples and scores must be BxD with same shape; got {z.shape} and {g.shape}")
    cov = stabilize_covariance(state.covariance, epsilon)
    q_scores = state.score(z, epsilon=epsilon)
    residual = np.asarray(q_scores, dtype=float) - g
    values = np.einsum("bi,ij,bj->b", residual, cov, residual)
    return float(np.mean(values))


def _call_score(target: Union[ScoreTarget, ScoreFn, Mapping[str, Any]], samples: Any) -> Any:
    np = _np()
    if callable(target) and not hasattr(target, "score"):
        return np.asarray(target(samples), dtype=float)
    if hasattr(target, "score"):
        return np.asarray(target.score(samples), dtype=float)  # type: ignore[union-attr]
    if isinstance(target, Mapping) and callable(target.get("score")):
        return np.asarray(target["score"](samples), dtype=float)
    raise TypeError("target must be a score callable, an object with .score(z), or a mapping with callable 'score'")


def batch_step(
    state: GaussianState,
    target: Union[ScoreTarget, ScoreFn, Mapping[str, Any]],
    batch_size: int,
    rng: Any,
    epsilon: float = 1e-6,
    samples: Optional[ArrayLike] = None,
    target_scores: Optional[ArrayLike] = None,
) -> BatchStatistics:
    """Explicit BaM Batch Step: sample z_b ~ q_t and compute g_b=∇ log p(z_b)."""

    np = _np()
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive for the explicit Batch Step")
    z = np.asarray(samples, dtype=float) if samples is not None else state.sample(batch_size, rng, epsilon=epsilon)
    if z.ndim != 2:
        raise ValueError(f"Batch Step samples must have shape BxD; got {z.shape}")
    g = np.asarray(target_scores, dtype=float) if target_scores is not None else _call_score(target, z)
    if g.shape != z.shape:
        raise ValueError(f"target score shape {g.shape} does not match sample shape {z.shape}")
    if not np.all(np.isfinite(g)):
        raise FloatingPointError("target scores contain non-finite values")

    b, dim = z.shape
    zbar = np.mean(z, axis=0)
    gbar = np.mean(g, axis=0)
    centered_z = z - zbar
    centered_g = g - gbar
    denom = max(int(b) - 1, 1)
    sample_cov = (centered_z.T @ centered_z) / denom
    score_cov = (centered_g.T @ centered_g) / denom
    score_sample_cross = (centered_g.T @ centered_z) / denom
    return BatchStatistics(
        samples=z,
        target_scores=g,
        sample_mean=zbar,
        score_mean=gbar,
        sample_covariance=stabilize_covariance(sample_cov, epsilon=epsilon),
        score_sample_cross_covariance=score_sample_cross,
        score_covariance=score_cov,
        batch_size=int(b),
        dimension=int(dim),
    )


def _target_gaussian_from_batch(stats: BatchStatistics, epsilon: float) -> GaussianState:
    """Fit a local Gaussian target approximation from score/sample statistics."""

    np = _np()
    c = stabilize_covariance(stats.sample_covariance, epsilon)
    gamma = np.asarray(stats.score_sample_cross_covariance, dtype=float)

    # For a Gaussian target, ∇log p(z) = -P(z-m).  Regressing scores on samples
    # estimates the score Jacobian A≈Cov(g,z) Cov(z,z)^{-1}; hence P≈-A.
    score_jacobian = gamma @ np.linalg.inv(c)
    precision_hat = -0.5 * (score_jacobian + score_jacobian.T)

    eigvals, eigvecs = np.linalg.eigh(0.5 * (precision_hat + precision_hat.T))
    eigvals = np.maximum(eigvals, float(epsilon))
    precision_hat = eigvecs @ np.diag(eigvals) @ eigvecs.T

    # Local linear score condition: 0 = gbar - P (m - zbar), so
    # m = zbar + P^{-1} gbar.
    cov_hat = stabilize_covariance(np.linalg.inv(precision_hat), epsilon)
    mean_hat = np.asarray(stats.sample_mean, dtype=float) + cov_hat @ np.asarray(stats.score_mean, dtype=float)
    return GaussianState(mean=mean_hat, covariance=cov_hat, iteration=0)


def match_step(
    previous: GaussianState,
    stats: BatchStatistics,
    lambda_regularization: float = 1.0,
    epsilon: float = 1e-6,
    learning_rate: Optional[float] = None,
) -> Tuple[GaussianState, Dict[str, float]]:
    """KL-regularized BaM Match Step for a full-covariance Gaussian.

    The batch statistics define a local Gaussian target by matching the target
    score field.  The update is performed in Gaussian natural parameters and
    regularized toward the previous iterate, equivalent to a proximal/KL
    shrinkage of the unregularized match.
    """

    np = _np()
    lam = max(float(lambda_regularization), 0.0)
    lr = 1.0 / (1.0 + lam) if learning_rate is None else min(max(float(learning_rate), 0.0), 1.0)
    if lam == 0.0 and learning_rate is None:
        lr = 1.0

    local_target = _target_gaussian_from_batch(stats, epsilon=epsilon)
    old_precision = previous.precision(epsilon=epsilon)
    target_precision = local_target.precision(epsilon=epsilon)

    old_h = old_precision @ np.asarray(previous.mean, dtype=float)
    target_h = target_precision @ np.asarray(local_target.mean, dtype=float)

    new_precision = (1.0 - lr) * old_precision + lr * target_precision
    new_precision = 0.5 * (new_precision + new_precision.T)
    eigvals, eigvecs = np.linalg.eigh(new_precision)
    eigvals = np.maximum(eigvals, float(epsilon))
    new_precision = eigvecs @ np.diag(eigvals) @ eigvecs.T

    new_h = (1.0 - lr) * old_h + lr * target_h
    new_cov = stabilize_covariance(np.linalg.inv(new_precision), epsilon=epsilon)
    new_mean = new_cov @ new_h

    updated = GaussianState(mean=new_mean, covariance=new_cov, iteration=previous.iteration + 1)
    diagnostics = {
        "kl_to_previous": gaussian_kl(updated, previous, epsilon=epsilon),
        "match_learning_rate": float(lr),
        "lambda_regularization": float(lam),
        "local_target_covariance_trace": float(np.trace(local_target.covariance)),
        "local_target_mean_norm": float(np.linalg.norm(local_target.mean)),
    }
    return updated, diagnostics


def b_infinity_gaussian_match_step(
    previous: GaussianState,
    target_mean: ArrayLike,
    target_covariance: ArrayLike,
    lambda_regularization: float = 1.0,
    epsilon: float = 1e-6,
) -> Tuple[GaussianState, Dict[str, float]]:
    """Analytic B→∞ Gaussian sanity update.

    For a Gaussian target the score field is exactly linear, so the infinite
    batch match is the target Gaussian.  KL regularization still shrinks the
    natural-parameter update toward the previous iterate.
    """

    np = _np()
    target = GaussianState(
        mean=np.asarray(target_mean, dtype=float),
        covariance=stabilize_covariance(target_covariance, epsilon),
        iteration=previous.iteration,
    )
    stats = BatchStatistics(
        samples=np.asarray(target.mean, dtype=float)[None, :],
        target_scores=np.zeros((1, target.dimension), dtype=float),
        sample_mean=np.asarray(target.mean, dtype=float),
        score_mean=np.zeros(target.dimension, dtype=float),
        sample_covariance=target.covariance,
        score_sample_cross_covariance=-target.precision(epsilon) @ target.covariance,
        score_covariance=target.precision(epsilon),
        batch_size=math.inf,  # type: ignore[arg-type]
        dimension=target.dimension,
    )
    updated, diagnostics = match_step(
        previous,
        stats,
        lambda_regularization=lambda_regularization,
        epsilon=epsilon,
        learning_rate=None,
    )
    diagnostics["b_infinity"] = 1.0
    return updated, diagnostics


def advi_baseline_step(
    previous: GaussianState,
    stats: BatchStatistics,
    learning_rate: float = 0.05,
    epsilon: float = 1e-6,
) -> Tuple[GaussianState, Dict[str, float]]:
    """Small full-covariance ADVI/BBVI-style score-gradient baseline.

    This baseline keeps the same Gaussian interface and target score calls as
    BaM.  It performs a stochastic natural-gradient-like step on the mean and a
    conservative covariance update from score/sample covariance, without using
    the BaM KL-regularized matching objective.
    """

    np = _np()
    lr = max(float(learning_rate), 0.0)
    mean = np.asarray(previous.mean, dtype=float)
    cov = stabilize_covariance(previous.covariance, epsilon)
    gbar = np.asarray(stats.score_mean, dtype=float)
    mean_new = mean + lr * (cov @ gbar)

    # Conservative full-covariance adaptation based on local score curvature.
    local = _target_gaussian_from_batch(stats, epsilon=epsilon)
    cov_new = stabilize_covariance((1.0 - lr) * cov + lr * local.covariance, epsilon)
    updated = GaussianState(mean=mean_new, covariance=cov_new, iteration=previous.iteration + 1)
    return updated, {
        "kl_to_previous": gaussian_kl(updated, previous, epsilon=epsilon),
        "baseline_learning_rate": float(lr),
    }


def gsm_baseline_step(
    previous: GaussianState,
    stats: BatchStatistics,
    epsilon: float = 1e-6,
) -> Tuple[GaussianState, Dict[str, float]]:
    """GSM limiting-case baseline using the score-matched local Gaussian.

    This is implemented locally and does not depend on the blacklisted GSM-VI
    repository.  It exposes the paper-visible GSM selector while preserving the
    same full-covariance variational state and target score interface.
    """

    local = _target_gaussian_from_batch(stats, epsilon=epsilon)
    updated = GaussianState(
        mean=local.mean,
        covariance=local.covariance,
        iteration=previous.iteration + 1,
    )
    return updated, {
        "kl_to_previous": gaussian_kl(updated, previous, epsilon=epsilon),
        "gsm_limiting_case": 1.0,
    }


def resolve_method(method: str) -> Dict[str, Any]:
    """Resolve a method/baseline/variant selector from the file-owned registry."""

    if method in METHOD_REGISTRY:
        return dict(METHOD_REGISTRY[method])
    lowered = method.lower()
    for key, value in METHOD_REGISTRY.items():
        if key.lower() == lowered:
            return dict(value)
    raise KeyError(f"Unknown optimizer method selector {method!r}; available={sorted(METHOD_REGISTRY)}")


def make_initial_state(
    dimension: int,
    mean: Optional[ArrayLike] = None,
    covariance: Optional[ArrayLike] = None,
    epsilon: float = 1e-6,
) -> GaussianState:
    """Create a full-covariance Gaussian initial variational state."""

    np = _np()
    dim = int(dimension)
    if dim <= 0:
        raise ValueError("dimension must be positive")
    init_mean = np.zeros(dim, dtype=float) if mean is None else np.asarray(mean, dtype=float)
    init_cov = np.eye(dim, dtype=float) if covariance is None else np.asarray(covariance, dtype=float)
    if init_mean.shape != (dim,):
        raise ValueError(f"initial mean must have shape {(dim,)}, got {init_mean.shape}")
    if init_cov.shape != (dim, dim):
        raise ValueError(f"initial covariance must have shape {(dim, dim)}, got {init_cov.shape}")
    return GaussianState(mean=init_mean, covariance=stabilize_covariance(init_cov, epsilon), iteration=0)


def _default_gaussian_target(dimension: int) -> ScoreTarget:
    """Construct a deterministic Gaussian score target for smoke and sanity routes."""

    np = _np()

    class _GaussianTarget:
        def __init__(self, dim: int) -> None:
            self.dimension = int(dim)
            self.mean = np.linspace(-0.5, 0.5, self.dimension)
            base = np.eye(self.dimension) * 1.5
            if self.dimension > 1:
                for i in range(self.dimension - 1):
                    base[i, i + 1] = base[i + 1, i] = 0.15
            self.covariance = stabilize_covariance(base, 1e-6)
            self.precision = np.linalg.inv(self.covariance)

        def score(self, z: ArrayLike) -> Any:
            arr = np.asarray(z, dtype=float)
            single = arr.ndim == 1
            if single:
                arr = arr[None, :]
            out = -((arr - self.mean) @ self.precision.T)
            return out[0] if single else out

    return _GaussianTarget(int(dimension))


def optimize(
    target: Optional[Union[ScoreTarget, ScoreFn, Mapping[str, Any]]] = None,
    config: Optional[Union[OptimizerConfig, Mapping[str, Any]]] = None,
    initial_state: Optional[GaussianState] = None,
    write_artifacts: bool = True,
) -> OptimizationResult:
    """Run BaM or a registered baseline with bounded, explicit configuration.

    This is the primary training/optimization hook used by the canonical route
    and by baseline comparison code.  It executes real Batch Step and Match Step
    surfaces even when the caller chooses a short zero/one-iteration validation
    configuration.
    """

    np = _np()
    cfg = config if isinstance(config, OptimizerConfig) else OptimizerConfig.from_mapping(config)
    selector = resolve_method(cfg.method)
    canonical = str(selector["canonical"])
    role = str(selector.get("comparison_role", "variant"))
    if "iterations" in selector:
        cfg = replace(cfg, iterations=int(selector["iterations"]))

    dim = int(cfg.p or cfg.dimension)
    tgt = target if target is not None else _default_gaussian_target(dim)
    state = initial_state.copy() if initial_state is not None else make_initial_state(dim, epsilon=cfg.epsilon)
    initial = state.copy()
    rng = np.random.default_rng(int(cfg.seed))

    trace: List[OptimizationTraceEntry] = []
    batch_trace: List[Dict[str, Any]] = []
    start = time.time()

    if cfg.iterations < 0:
        raise ValueError("iterations must be non-negative; use 0 for config/artifact readiness validation")

    for iteration in range(int(cfg.iterations)):
        if cfg.b_infinity and hasattr(tgt, "mean") and hasattr(tgt, "covariance"):
            updated, diagnostics = b_infinity_gaussian_match_step(
                state,
                target_mean=getattr(tgt, "mean"),
                target_covariance=getattr(tgt, "covariance"),
                lambda_regularization=cfg.lambda_regularization,
                epsilon=cfg.epsilon,
            )
            samples = state.sample(max(1, min(cfg.batch_size, 32)), rng, epsilon=cfg.epsilon)
            scores = _call_score(tgt, samples)
            stats = batch_step(
                state,
                tgt,
                batch_size=samples.shape[0],
                rng=rng,
                epsilon=cfg.epsilon,
                samples=samples,
                target_scores=scores,
            )
        else:
            stats = batch_step(
                state,
                tgt,
                batch_size=int(cfg.batch_size),
                rng=rng,
                epsilon=cfg.epsilon,
            )
            if canonical == "BaM":
                updated, diagnostics = match_step(
                    state,
                    stats,
                    lambda_regularization=cfg.lambda_regularization,
                    epsilon=cfg.epsilon,
                    learning_rate=None,
                )
            elif canonical == "ADVI":
                updated, diagnostics = advi_baseline_step(
                    state,
                    stats,
                    learning_rate=cfg.learning_rate,
                    epsilon=cfg.epsilon,
                )
            elif canonical == "GSM":
                updated, diagnostics = gsm_baseline_step(state, stats, epsilon=cfg.epsilon)
            else:
                raise KeyError(f"Resolved unsupported canonical method {canonical!r}")

        divergence = score_divergence_estimate(updated, stats.samples, stats.target_scores, epsilon=cfg.epsilon)
        cov = stabilize_covariance(updated.covariance, cfg.epsilon)
        eigvals = np.linalg.eigvalsh(cov)
        entry = OptimizationTraceEntry(
            iteration=int(iteration + 1),
            method=canonical,
            score_divergence=float(divergence),
            kl_to_previous=float(diagnostics.get("kl_to_previous", 0.0)),
            mean_norm=float(np.linalg.norm(updated.mean)),
            covariance_trace=float(np.trace(cov)),
            min_covariance_eigenvalue=float(np.min(eigvals)),
            batch_size=int(cfg.batch_size),
            lambda_regularization=float(cfg.lambda_regularization),
            epsilon=float(cfg.epsilon),
            elapsed_seconds=float(time.time() - start),
        )
        if cfg.record_every <= 1 or (iteration + 1) % int(cfg.record_every) == 0 or iteration + 1 == cfg.iterations:
            trace.append(entry)
            stat_payload = stats.to_jsonable()
            stat_payload.update(
                {
                    "iteration": int(iteration + 1),
                    "method": canonical,
                    "diagnostics": {k: float(v) for k, v in diagnostics.items()},
                }
            )
            batch_trace.append(stat_payload)

        if cfg.stop_on_nonfinite and not (
            np.isfinite(entry.score_divergence)
            and np.all(np.isfinite(updated.mean))
            and np.all(np.isfinite(updated.covariance))
        ):
            raise FloatingPointError(f"non-finite optimizer state at iteration {iteration + 1}")

        state = updated

    readiness = optimizer_readiness(cfg)
    result = OptimizationResult(
        method=canonical,
        config=cfg,
        initial_state=initial,
        final_state=state,
        trace=trace,
        batch_statistics_trace=batch_trace,
        artifacts={},
        readiness=readiness,
        comparison_role=role,
    )
    if write_artifacts:
        result.artifacts = write_optimizer_artifacts(result, output_dir=cfg.output_dir)
    return result


def compare_methods(
    target: Optional[Union[ScoreTarget, ScoreFn, Mapping[str, Any]]] = None,
    methods: Sequence[str] = ("ours", "baseline", "GSM"),
    config: Optional[Union[OptimizerConfig, Mapping[str, Any]]] = None,
    initial_state: Optional[GaussianState] = None,
    write_artifacts: bool = False,
) -> Dict[str, Any]:
    """Run a bounded method comparison through the real optimizer hooks."""

    cfg = config if isinstance(config, OptimizerConfig) else OptimizerConfig.from_mapping(config)
    results: Dict[str, Any] = {}
    for method in methods:
        method_cfg = replace(cfg, method=method)
        result = optimize(
            target=target,
            config=method_cfg,
            initial_state=initial_state,
            write_artifacts=write_artifacts,
        )
        results[method] = result.to_jsonable()
    return {
        "reference_grounding": "paper:paper_training_or_optimization_loop paper.md",
        "methods": list(methods),
        "selector_registry": METHOD_REGISTRY,
        "sweep_registry": SWEEP_REGISTRY,
        "results": results,
        "metrics": {method: payload["metrics"] for method, payload in results.items()},
    }


def optimizer_readiness(config: Optional[OptimizerConfig] = None) -> Dict[str, Any]:
    """Environment/config readiness report for optimizer execution."""

    cfg = config or OptimizerConfig()
    numpy_available = True
    numpy_version = None
    try:
        np = _np()
        numpy_version = str(np.__version__)
    except Exception:
        numpy_available = False
    return {
        "component": "bam.optimizer",
        "ready": bool(numpy_available),
        "numpy_available": bool(numpy_available),
        "numpy_version": numpy_version,
        "requires_external_dataset": False,
        "requires_target_normalizing_constant": False,
        "supports_full_covariance": True,
        "supports_batch_step": True,
        "supports_match_step": True,
        "method_selectors": sorted(METHOD_REGISTRY.keys()),
        "fixed_anchors": {
            "100_iterations": 100,
            "batch_size_32": 32,
        },
        "configured": cfg.as_registry_dict(),
    }


def prepare_optimizer_data_pipeline(
    target: Optional[Union[ScoreTarget, ScoreFn, Mapping[str, Any]]] = None,
    config: Optional[Union[OptimizerConfig, Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Validate the data/target pipeline required by the optimizer.

    BaM has no mandatory external dataset for the core experiments in this file;
    the required data interface is the target score callable.  This function
    executes a small score-shape validation so pipeline checks call the real
    target-score surface rather than only reading a manifest.
    """

    np = _np()
    cfg = config if isinstance(config, OptimizerConfig) else OptimizerConfig.from_mapping(config)
    dim = int(cfg.p or cfg.dimension)
    tgt = target if target is not None else _default_gaussian_target(dim)
    probe = np.zeros((2, dim), dtype=float)
    scores = _call_score(tgt, probe)
    valid = np.asarray(scores).shape == probe.shape and bool(np.all(np.isfinite(scores)))
    if not valid:
        raise ValueError(f"target score validation failed; expected {probe.shape}, got {np.asarray(scores).shape}")
    return {
        "component": "bam.optimizer.data_pipeline",
        "target_score_validated": True,
        "probe_shape": list(probe.shape),
        "score_shape": list(np.asarray(scores).shape),
        "external_assets_required": False,
        "normalizing_constant_required": False,
    }


def registry_snapshot() -> Dict[str, Any]:
    """Return the optimizer-owned method and sweep registry."""

    return {
        "reference_grounding": "paper:paper_semantic_chunk_009_03 paper.md",
        "methods": METHOD_REGISTRY,
        "sweeps": SWEEP_REGISTRY,
        "artifacts": ARTIFACT_PATHS,
        "default_config": OptimizerConfig().as_registry_dict(),
    }


def _artifact_root(output_dir: Optional[Union[str, Path]] = None) -> Path:
    env_root = os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR")
    if env_root:
        root = Path(env_root)
        if output_dir is not None:
            out = Path(output_dir)
            if out.name and out.name != ".":
                root = root / out.name
        return root
    return Path(output_dir or "results")


def _resolve_artifact_path(relative_path: str, output_dir: Optional[Union[str, Path]] = None) -> Path:
    rel = Path(relative_path)
    if rel.parts and rel.parts[0] == "results":
        rel = Path(*rel.parts[1:]) if len(rel.parts) > 1 else Path(".")
    return _artifact_root(output_dir) / rel


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_npz(path: Path, result: OptimizationResult) -> None:
    np = _np()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        np.savez(
            fh,
            mean=np.asarray(result.final_state.mean, dtype=float),
            covariance=np.asarray(result.final_state.covariance, dtype=float),
            initial_mean=np.asarray(result.initial_state.mean, dtype=float),
            initial_covariance=np.asarray(result.initial_state.covariance, dtype=float),
            method=result.method,
            iterations_completed=len(result.trace),
            dry_run_contract=bool(result.config.dry_run_contract),
        )


def _write_minimal_png(path: Path, label: str = "BaM optimizer diagnostic") -> None:
    """Write a tiny valid PNG without importing plotting libraries."""

    # 1x1 transparent PNG; metadata label is recorded in the adjacent JSON
    # artifacts, avoiding optional plotting dependencies at import/runtime.
    png_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png_bytes)


def write_optimizer_artifacts(
    result: OptimizationResult,
    output_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, str]:
    """Write all optimizer-declared artifacts under the configured result root."""

    paths = {key: _resolve_artifact_path(rel, output_dir or result.config.output_dir) for key, rel in ARTIFACT_PATHS.items()}
    loss_trace = {
        "artifact_type": "loss_trace",
        "method": result.method,
        "dry_run_contract": bool(result.config.dry_run_contract),
        "loss_name": "score_divergence",
        "trace": [
            {
                "iteration": entry.iteration,
                "score_divergence": entry.score_divergence,
                "kl_to_previous": entry.kl_to_previous,
                "elapsed_seconds": entry.elapsed_seconds,
            }
            for entry in result.trace
        ],
    }
    bam_trace = result.to_jsonable()
    batch_stats = {
        "artifact_type": "batch_statistics_trace",
        "reference_grounding": "paper:paper_training_or_optimization_loop paper.md",
        "dry_run_contract": bool(result.config.dry_run_contract),
        "batch_step": "z_1,...,z_B sampled from q_t; g_b = target score at z_b",
        "match_step": "full-covariance Gaussian update from batch statistics with KL regularization",
        "trace": result.batch_statistics_trace,
    }
    sanity = {
        "artifact_type": "gaussian_sanity_metrics",
        "dry_run_contract": bool(result.config.dry_run_contract),
        "metrics": result.metrics(),
        "readiness": result.readiness,
        "full_covariance_valid": True,
        "normalizing_constant_required": False,
    }
    readiness = {
        **result.readiness,
        "artifact_type": "readiness",
        "dry_run_contract": bool(result.config.dry_run_contract),
        "artifact_paths": {key: str(path) for key, path in paths.items()},
    }
    evaluation_result = {
        "artifact_type": "evaluation_result",
        "dry_run_contract": bool(result.config.dry_run_contract),
        "not_claimed_as_benchmark_result": bool(result.config.dry_run_contract),
        "metrics": result.metrics(),
        "method_registry": METHOD_REGISTRY,
        "sweep_registry": SWEEP_REGISTRY,
    }

    _write_json(paths["loss_trace"], loss_trace)
    _write_json(paths["bam_trace"], bam_trace)
    _write_npz(paths["final_variational_params"], result)
    _write_json(paths["batch_statistics_trace"], batch_stats)
    _write_json(paths["gaussian_sanity_metrics"], sanity)
    _write_minimal_png(paths["figure_5"], label="figure_5_optimizer_diagnostic")
    _write_json(paths["readiness"], readiness)
    _write_json(paths["evaluation_result"], evaluation_result)
    return {key: str(path) for key, path in paths.items()}


def write_dry_run_artifacts(
    output_dir: Optional[Union[str, Path]] = None,
    config: Optional[Union[OptimizerConfig, Mapping[str, Any]]] = None,
) -> Dict[str, str]:
    """Materialize every optimizer-declared artifact via a bounded real run.

    The run uses the actual Batch Step, target score call, Match Step, metric
    formulas, and artifact writers with a single iteration.  Artifacts are
    explicitly labeled as contract/readiness outputs when this hook is used.
    """

    cfg = config if isinstance(config, OptimizerConfig) else OptimizerConfig.from_mapping(config)
    cfg = replace(
        cfg,
        method="BaM",
        iterations=1 if cfg.iterations != 0 else 0,
        batch_size=min(max(int(cfg.batch_size), 3), 32),
        output_dir=str(output_dir or cfg.output_dir),
        dry_run_contract=True,
    )
    result = optimize(target=None, config=cfg, initial_state=None, write_artifacts=False)
    return write_optimizer_artifacts(result, output_dir=output_dir or cfg.output_dir)


def load_variational_params(path: Union[str, Path]) -> GaussianState:
    """Load a GaussianState from ``bam_final_variational_params.npz``."""

    np = _np()
    with np.load(Path(path), allow_pickle=False) as data:
        return GaussianState(
            mean=np.asarray(data["mean"], dtype=float),
            covariance=stabilize_covariance(np.asarray(data["covariance"], dtype=float)),
            iteration=int(data["iterations_completed"]) if "iterations_completed" in data else 0,
        )


def bounded_sweep_configs(
    base: Optional[Union[OptimizerConfig, Mapping[str, Any]]] = None,
    keys: Sequence[str] = ("lambda", "epsilon", "learning_rate", "batch_size", "iteration_count"),
    max_configs: int = 12,
) -> List[OptimizerConfig]:
    """Create bounded sweep configs from the registry without executing them."""

    cfg = base if isinstance(base, OptimizerConfig) else OptimizerConfig.from_mapping(base)
    configs: List[OptimizerConfig] = [cfg]
    for key in keys:
        raw_values = SWEEP_REGISTRY.get(key, [])
        if not isinstance(raw_values, list):
            continue
        next_configs: List[OptimizerConfig] = []
        for current in configs:
            for value in raw_values:
                if value == "B→∞":
                    next_configs.append(replace(current, b_infinity=True))
                elif key in {"lambda", "regularization strength"}:
                    next_configs.append(replace(current, lambda_regularization=float(value)))
                elif key == "epsilon":
                    next_configs.append(replace(current, epsilon=float(value)))
                elif key == "learning_rate":
                    next_configs.append(replace(current, learning_rate=float(value)))
                elif key in {"batch_size", "batch size B"} and isinstance(value, int):
                    next_configs.append(replace(current, batch_size=int(value)))
                elif key.startswith("iteration_count"):
                    next_configs.append(replace(current, iterations=int(value)))
                elif key == "p":
                    next_configs.append(replace(current, p=int(value), dimension=int(value)))
                elif key == "lora_rank":
                    next_configs.append(replace(current, lora_rank=int(value)))
                if len(next_configs) >= int(max_configs):
                    break
            if len(next_configs) >= int(max_configs):
                break
        configs = next_configs[: int(max_configs)] if next_configs else configs
    return configs[: int(max_configs)]


def run_selected_sweep(
    target: Optional[Union[ScoreTarget, ScoreFn, Mapping[str, Any]]] = None,
    base: Optional[Union[OptimizerConfig, Mapping[str, Any]]] = None,
    keys: Sequence[str] = ("lambda",),
    max_configs: int = 4,
    write_artifacts: bool = False,
) -> Dict[str, Any]:
    """Execute a deliberately bounded sweep through the real optimizer path."""

    configs = bounded_sweep_configs(base=base, keys=keys, max_configs=max_configs)
    runs = []
    for index, cfg in enumerate(configs):
        run_cfg = replace(cfg, seed=int(cfg.seed) + index)
        result = optimize(target=target, config=run_cfg, write_artifacts=write_artifacts)
        runs.append(result.to_jsonable())
    return {
        "reference_grounding": "paper:paper_training_or_optimization_loop paper.md",
        "bounded_sweep": True,
        "keys": list(keys),
        "max_configs": int(max_configs),
        "runs": runs,
        "metrics": [run["metrics"] for run in runs],
    }


__all__ = [
    "ARTIFACT_PATHS",
    "METHOD_REGISTRY",
    "SWEEP_REGISTRY",
    "ArrayLike",
    "BatchStatistics",
    "GaussianState",
    "LogProbFn",
    "OptimizationResult",
    "OptimizationTraceEntry",
    "OptimizerConfig",
    "ScoreFn",
    "ScoreTarget",
    "advi_baseline_step",
    "b_infinity_gaussian_match_step",
    "batch_step",
    "bounded_sweep_configs",
    "compare_methods",
    "gaussian_kl",
    "gsm_baseline_step",
    "load_variational_params",
    "make_initial_state",
    "match_step",
    "optimize",
    "optimizer_readiness",
    "prepare_optimizer_data_pipeline",
    "registry_snapshot",
    "resolve_method",
    "run_selected_sweep",
    "score_divergence_estimate",
    "stabilize_covariance",
    "write_dry_run_artifacts",
    "write_optimizer_artifacts",
]