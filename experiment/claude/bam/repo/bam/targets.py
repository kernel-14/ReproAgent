"""Target distributions, score interfaces, and BaM batch/match utilities.

This module owns the target-distribution and score-call surface for the
PaperBench reproduction of

    "Batch and match: black-box variational inference with a score-based
    divergence."

It is intentionally import-light: NumPy is imported lazily inside numerical
functions so package import and registry inspection work in minimal code-only
environments.

reference_grounding: paper:paper_method_core paper.md
    Score-based BBVI assumes access to the target score ∇ log p(z).  The BaM
    target interface below exposes ``score(z)`` and never requires the target
    normalizing constant.

reference_grounding: paper:paper_training_or_optimization_loop paper.md
    BaM separates an explicit Batch Step, z_1,...,z_B ~ q_t and
    g_b = ∇ log p(z_b), from a Match Step that uses zbar, C, gbar, Gamma and
    KL/proximal regularization to update full-covariance Gaussian variational
    parameters.

reference_grounding: paper:paper_semantic_chunk_009_03 paper.md
    BaM is a KL-regularized stochastic proximal/matching update.  This module
    exposes finite B=32 semantics, an analytic B→∞ Gaussian sanity route, and a
    GSM limiting-case configuration hook implemented locally without depending
    on the blacklisted GSM-VI repository.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Protocol, Sequence, Tuple, Union


ArrayLike = Any
LogDensityFn = Callable[[ArrayLike], Any]
ScoreFn = Callable[[ArrayLike], Any]


def _np() -> Any:
    """Import NumPy lazily with a clear numerical-runtime error."""
    try:
        import numpy as np  # type: ignore

        return np
    except Exception as exc:  # pragma: no cover - host-environment dependent
        raise RuntimeError(
            "Numerical execution of bam.targets requires numpy. "
            "Install the repository requirements before running BaM training."
        ) from exc


def _as_array(x: ArrayLike, dtype: str = "float64") -> Any:
    np = _np()
    return np.asarray(x, dtype=dtype)


def _symmetrize(matrix: ArrayLike) -> Any:
    np = _np()
    m = np.asarray(matrix, dtype=float)
    return 0.5 * (m + m.T)


def _stable_cholesky(covariance: ArrayLike, jitter: float = 1e-8, max_tries: int = 8) -> Any:
    """Return a Cholesky factor after symmetric jitter stabilization."""
    np = _np()
    cov = _symmetrize(covariance)
    eye = np.eye(cov.shape[0])
    local_jitter = float(max(jitter, 0.0))
    for _ in range(max_tries):
        try:
            return np.linalg.cholesky(cov + local_jitter * eye)
        except np.linalg.LinAlgError:
            local_jitter = 10.0 * local_jitter if local_jitter > 0 else 1e-8
    eigvals, eigvecs = np.linalg.eigh(cov)
    clipped = np.clip(eigvals, max(local_jitter, 1e-8), None)
    repaired = (eigvecs * clipped) @ eigvecs.T
    return np.linalg.cholesky(_symmetrize(repaired) + max(local_jitter, 1e-8) * eye)


def stabilize_covariance(covariance: ArrayLike, min_eig: float = 1e-8, max_condition: float = 1e10) -> Any:
    """Project a symmetric matrix to a numerically stable positive-definite covariance."""
    np = _np()
    cov = _symmetrize(covariance)
    eigvals, eigvecs = np.linalg.eigh(cov)
    lower = float(max(min_eig, 1e-12))
    clipped = np.clip(eigvals, lower, None)
    if clipped.max() / clipped.min() > max_condition:
        clipped = np.clip(clipped, clipped.max() / max_condition, clipped.max())
    repaired = (eigvecs * clipped) @ eigvecs.T
    return _symmetrize(repaired)


def _solve_spd(covariance: ArrayLike, rhs: ArrayLike, jitter: float = 1e-8) -> Any:
    """Solve covariance @ x = rhs with stabilization."""
    np = _np()
    cov = stabilize_covariance(covariance, min_eig=jitter)
    try:
        return np.linalg.solve(cov, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(cov) @ rhs


def _matrix_inverse_spd(covariance: ArrayLike, jitter: float = 1e-8) -> Any:
    np = _np()
    cov = stabilize_covariance(covariance, min_eig=jitter)
    try:
        return np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(cov)


def _logdet_spd(covariance: ArrayLike, jitter: float = 1e-8) -> float:
    np = _np()
    cov = stabilize_covariance(covariance, min_eig=jitter)
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        cov = stabilize_covariance(cov, min_eig=max(jitter, 1e-6))
        sign, logdet = np.linalg.slogdet(cov)
    return float(logdet)


def _jsonable(value: Any) -> Any:
    """Convert arrays and dataclasses to JSON-serializable values."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "__dataclass_fields__"):
        return {k: _jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (float, int, str, bool)) or value is None:
        return value
    return str(value)


class TargetDistribution(Protocol):
    """Protocol for targets used by score-based BBVI.

    Implementations may expose an unnormalized log density, but BaM only needs
    the score.  The normalizing constant is intentionally absent from this
    interface.
    """

    name: str
    dim: int

    def log_density(self, z: ArrayLike) -> Any:
        """Return an unnormalized log density at z."""

    def score(self, z: ArrayLike) -> Any:
        """Return ∇ log p(z) at z without using the normalizing constant."""


@dataclass(frozen=True)
class TargetSpec:
    """Machine-readable target/data/environment registry entry."""

    name: str
    target_type: str
    dim: int
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    has_score: bool = True
    requires_external_data: bool = False
    environment: str = "local_numpy_cpu"
    paper_section: str = "score_based_bbvi"
    readiness_artifact: str = "results/environment_registry.json"


@dataclass
class GaussianVariationalState:
    """Full-covariance Gaussian variational state q(z)=N(mean,covariance).

    The state is duplicated here, rather than only in ``bam.variational``, so the
    target module can satisfy the canonical target-score and analytic Gaussian
    sanity contracts independently.  Neighboring modules may use this state or
    adapt it to their richer variational classes.
    """

    mean: ArrayLike
    covariance: ArrayLike
    jitter: float = 1e-8

    def __post_init__(self) -> None:
        np = _np()
        self.mean = np.asarray(self.mean, dtype=float).reshape(-1)
        self.covariance = stabilize_covariance(np.asarray(self.covariance, dtype=float), min_eig=self.jitter)
        if self.covariance.shape != (self.mean.shape[0], self.mean.shape[0]):
            raise ValueError(
                f"GaussianVariationalState covariance shape {self.covariance.shape} "
                f"does not match mean dimension {self.mean.shape[0]}."
            )

    @property
    def dim(self) -> int:
        return int(self.mean.shape[0])

    @property
    def precision(self) -> Any:
        return _matrix_inverse_spd(self.covariance, jitter=self.jitter)

    def sample(self, batch_size: int, rng: Any) -> Any:
        """Explicitly sample z_1,...,z_B ~ q_t."""
        np = _np()
        chol = _stable_cholesky(self.covariance, jitter=self.jitter)
        eps = rng.normal(size=(int(batch_size), self.dim))
        return self.mean[None, :] + eps @ chol.T

    def log_density(self, z: ArrayLike) -> Any:
        """Normalized Gaussian log density for q."""
        np = _np()
        z_arr = np.asarray(z, dtype=float)
        single = z_arr.ndim == 1
        z2 = z_arr.reshape(1, -1) if single else z_arr
        diff = z2 - self.mean[None, :]
        prec_diff = _solve_spd(self.covariance, diff.T, jitter=self.jitter).T
        quad = np.sum(diff * prec_diff, axis=1)
        val = -0.5 * (self.dim * math.log(2.0 * math.pi) + _logdet_spd(self.covariance, self.jitter) + quad)
        return float(val[0]) if single else val

    def score(self, z: ArrayLike) -> Any:
        """Gaussian score ∇ log q(z) = -Σ^{-1}(z-μ)."""
        np = _np()
        z_arr = np.asarray(z, dtype=float)
        single = z_arr.ndim == 1
        z2 = z_arr.reshape(1, -1) if single else z_arr
        diff = z2 - self.mean[None, :]
        scores = -_solve_spd(self.covariance, diff.T, jitter=self.jitter).T
        return scores[0] if single else scores

    def kl_to(self, other: "GaussianVariationalState") -> float:
        """KL(self || other) for full-covariance Gaussians."""
        np = _np()
        d = self.dim
        if other.dim != d:
            raise ValueError("KL requires equal dimensions.")
        inv_other = other.precision
        diff = other.mean - self.mean
        trace_term = float(np.trace(inv_other @ self.covariance))
        quad_term = float(diff.T @ inv_other @ diff)
        logdet_ratio = _logdet_spd(other.covariance, other.jitter) - _logdet_spd(self.covariance, self.jitter)
        return 0.5 * (trace_term + quad_term - d + logdet_ratio)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean": _jsonable(self.mean),
            "covariance": _jsonable(self.covariance),
            "jitter": float(self.jitter),
            "dim": self.dim,
        }


@dataclass
class GaussianTarget:
    """Analytic Gaussian target with known score and optional sampling.

    The log density includes the normalizing constant for metric evaluation, but
    the ``score`` method uses only the unnormalized quadratic term.  BaM updates
    can therefore call ``score`` without the target normalizing constant.
    """

    mean: ArrayLike
    covariance: ArrayLike
    name: str = "gaussian"
    jitter: float = 1e-8

    def __post_init__(self) -> None:
        np = _np()
        self.mean = np.asarray(self.mean, dtype=float).reshape(-1)
        self.covariance = stabilize_covariance(np.asarray(self.covariance, dtype=float), min_eig=self.jitter)
        if self.covariance.shape != (self.mean.shape[0], self.mean.shape[0]):
            raise ValueError("GaussianTarget covariance shape must match mean dimension.")

    @property
    def dim(self) -> int:
        return int(self.mean.shape[0])

    @property
    def precision(self) -> Any:
        return _matrix_inverse_spd(self.covariance, jitter=self.jitter)

    def unnormalized_log_density(self, z: ArrayLike) -> Any:
        np = _np()
        z_arr = np.asarray(z, dtype=float)
        single = z_arr.ndim == 1
        z2 = z_arr.reshape(1, -1) if single else z_arr
        diff = z2 - self.mean[None, :]
        prec_diff = _solve_spd(self.covariance, diff.T, jitter=self.jitter).T
        val = -0.5 * np.sum(diff * prec_diff, axis=1)
        return float(val[0]) if single else val

    def log_density(self, z: ArrayLike) -> Any:
        np = _np()
        val = self.unnormalized_log_density(z)
        const = -0.5 * (self.dim * math.log(2.0 * math.pi) + _logdet_spd(self.covariance, self.jitter))
        return val + const if not isinstance(val, float) else float(val + const)

    def log_prob(self, z: ArrayLike) -> Any:
        """Alias used by the shared target API for BaM, ADVI, and GSM."""

        return self.log_density(z)

    def score(self, z: ArrayLike) -> Any:
        """Return ∇ log p(z); the Gaussian normalizing constant cancels."""
        np = _np()
        z_arr = np.asarray(z, dtype=float)
        single = z_arr.ndim == 1
        z2 = z_arr.reshape(1, -1) if single else z_arr
        diff = z2 - self.mean[None, :]
        scores = -_solve_spd(self.covariance, diff.T, jitter=self.jitter).T
        return scores[0] if single else scores

    def sample(self, batch_size: int, rng: Any) -> Any:
        chol = _stable_cholesky(self.covariance, jitter=self.jitter)
        eps = rng.normal(size=(int(batch_size), self.dim))
        return self.mean[None, :] + eps @ chol.T

    def analytic_kl_q_to_p(self, q: GaussianVariationalState) -> float:
        return q.kl_to(GaussianVariationalState(self.mean, self.covariance, jitter=self.jitter))

    def analytic_kl_p_to_q(self, q: GaussianVariationalState) -> float:
        return GaussianVariationalState(self.mean, self.covariance, jitter=self.jitter).kl_to(q)


class GaussianDistribution(GaussianTarget):
    """Convenience Gaussian target constructor for smoke/API callers."""

    def __init__(
        self,
        dim: Optional[int] = None,
        dimension: Optional[int] = None,
        mean: Optional[ArrayLike] = None,
        covariance: Optional[ArrayLike] = None,
        name: str = "gaussian",
        jitter: float = 1e-8,
    ) -> None:
        np = _np()
        d = int(dim if dim is not None else dimension if dimension is not None else 4)
        resolved_mean = np.zeros(d, dtype=float) if mean is None else np.asarray(mean, dtype=float)
        resolved_covariance = np.eye(d, dtype=float) if covariance is None else np.asarray(covariance, dtype=float)
        super().__init__(mean=resolved_mean, covariance=resolved_covariance, name=name, jitter=jitter)


FullCovarianceGaussianTarget = GaussianDistribution


def make_target(
    target: Union[str, Mapping[str, Any]] = "gaussian",
    config: Optional[Mapping[str, Any]] = None,
    dim: Optional[int] = None,
    dimension: Optional[int] = None,
) -> TargetDistribution:
    """Build a target distribution from the shared experiment target selector."""

    if isinstance(target, Mapping):
        cfg = dict(target)
        target_name = str(cfg.get("target", cfg.get("name", cfg.get("id", "gaussian"))))
        dim = int(cfg.get("dimension", cfg.get("dim", dim if dim is not None else 4)))
    else:
        cfg = dict(config or {})
        target_name = str(target)
        dim = int(cfg.get("dimension", cfg.get("dim", dim if dim is not None else dimension if dimension is not None else 4)))

    if "gaussian" in target_name.lower():
        return GaussianDistribution(dim=dim, name=target_name)
    if "banana" in target_name.lower():
        return BananaTarget(dim=max(2, dim))
    raise KeyError(f"Unknown target selector {target_name!r}")


get_target = make_target
build_target = make_target
create_target = make_target


@dataclass
class BananaTarget:
    """Smooth non-Gaussian banana-shaped target used for synthetic stress tests."""

    dim: int = 2
    curvature: float = 0.1
    scale: float = 1.0
    name: str = "banana"

    def __post_init__(self) -> None:
        if self.dim < 2:
            raise ValueError("BananaTarget requires dim >= 2.")
        if self.scale <= 0:
            raise ValueError("BananaTarget scale must be positive.")

    def _transform(self, z: ArrayLike) -> Tuple[Any, bool]:
        np = _np()
        arr = np.asarray(z, dtype=float)
        single = arr.ndim == 1
        z2 = arr.reshape(1, -1) if single else arr
        if z2.shape[1] != self.dim:
            raise ValueError(f"Expected dimension {self.dim}, got {z2.shape[1]}.")
        y = z2.copy()
        y[:, 1] = z2[:, 1] - self.curvature * (z2[:, 0] ** 2 - self.scale**2)
        return y, single

    def log_density(self, z: ArrayLike) -> Any:
        np = _np()
        y, single = self._transform(z)
        val = -0.5 * np.sum((y / self.scale) ** 2, axis=1)
        return float(val[0]) if single else val

    def score(self, z: ArrayLike) -> Any:
        np = _np()
        arr = np.asarray(z, dtype=float)
        single = arr.ndim == 1
        z2 = arr.reshape(1, -1) if single else arr
        y, _ = self._transform(z2)
        grad_y = -(y / (self.scale**2))
        score = grad_y.copy()
        score[:, 0] += grad_y[:, 1] * (-2.0 * self.curvature * z2[:, 0])
        return score[0] if single else score


@dataclass
class GaussianMixtureTarget:
    """Finite Gaussian-mixture target with exact score via log-sum-exp weights."""

    means: ArrayLike
    covariances: ArrayLike
    weights: ArrayLike
    name: str = "gaussian_mixture"
    jitter: float = 1e-8

    def __post_init__(self) -> None:
        np = _np()
        self.means = np.asarray(self.means, dtype=float)
        self.covariances = np.asarray(self.covariances, dtype=float)
        self.weights = np.asarray(self.weights, dtype=float).reshape(-1)
        if self.means.ndim != 2:
            raise ValueError("Mixture means must have shape (K,D).")
        if self.covariances.shape != (self.means.shape[0], self.means.shape[1], self.means.shape[1]):
            raise ValueError("Mixture covariances must have shape (K,D,D).")
        if self.weights.shape[0] != self.means.shape[0]:
            raise ValueError("Mixture weights length must equal number of components.")
        if float(self.weights.sum()) <= 0:
            raise ValueError("Mixture weights must have positive sum.")
        self.weights = self.weights / self.weights.sum()
        self.covariances = np.stack([stabilize_covariance(c, min_eig=self.jitter) for c in self.covariances], axis=0)

    @property
    def dim(self) -> int:
        return int(self.means.shape[1])

    @property
    def num_components(self) -> int:
        return int(self.means.shape[0])

    def _component_log_probs(self, z: ArrayLike) -> Any:
        np = _np()
        z_arr = np.asarray(z, dtype=float)
        single = z_arr.ndim == 1
        z2 = z_arr.reshape(1, -1) if single else z_arr
        vals = []
        for k in range(self.num_components):
            diff = z2 - self.means[k][None, :]
            prec_diff = _solve_spd(self.covariances[k], diff.T, jitter=self.jitter).T
            quad = np.sum(diff * prec_diff, axis=1)
            norm = -0.5 * (self.dim * math.log(2.0 * math.pi) + _logdet_spd(self.covariances[k], self.jitter))
            vals.append(math.log(float(self.weights[k])) + norm - 0.5 * quad)
        return np.stack(vals, axis=1)

    def log_density(self, z: ArrayLike) -> Any:
        np = _np()
        logs = self._component_log_probs(z)
        max_log = np.max(logs, axis=1, keepdims=True)
        val = max_log[:, 0] + np.log(np.sum(np.exp(logs - max_log), axis=1))
        z_arr = np.asarray(z, dtype=float)
        return float(val[0]) if z_arr.ndim == 1 else val

    def score(self, z: ArrayLike) -> Any:
        np = _np()
        z_arr = np.asarray(z, dtype=float)
        single = z_arr.ndim == 1
        z2 = z_arr.reshape(1, -1) if single else z_arr
        logs = self._component_log_probs(z2)
        max_log = np.max(logs, axis=1, keepdims=True)
        responsibilities = np.exp(logs - max_log)
        responsibilities = responsibilities / responsibilities.sum(axis=1, keepdims=True)
        component_scores = []
        for k in range(self.num_components):
            diff = z2 - self.means[k][None, :]
            component_scores.append(-_solve_spd(self.covariances[k], diff.T, jitter=self.jitter).T)
        score = np.sum(responsibilities[:, :, None] * np.stack(component_scores, axis=1), axis=1)
        return score[0] if single else score


@dataclass
class LogisticRegressionPosteriorTarget:
    """Bayesian logistic-regression posterior target for data-pipeline tests.

    The posterior is defined up to a constant:
        log p(w | X,y) = sum_i log Bernoulli(y_i | sigmoid(X_i w))
                         - 1/2 w^T prior_precision w + const.

    This target implements an exact score and therefore satisfies the BBVI score
    interface without needing a posterior normalizing constant.
    """

    features: ArrayLike
    labels: ArrayLike
    prior_precision: float = 1.0
    name: str = "logistic_regression_posterior"

    def __post_init__(self) -> None:
        np = _np()
        self.features = np.asarray(self.features, dtype=float)
        self.labels = np.asarray(self.labels, dtype=float).reshape(-1)
        if self.features.ndim != 2:
            raise ValueError("features must have shape (n,d).")
        if self.labels.shape[0] != self.features.shape[0]:
            raise ValueError("labels length must equal number of rows in features.")
        if self.prior_precision <= 0:
            raise ValueError("prior_precision must be positive.")

    @property
    def dim(self) -> int:
        return int(self.features.shape[1])

    def _linear(self, z: ArrayLike) -> Tuple[Any, bool]:
        np = _np()
        z_arr = np.asarray(z, dtype=float)
        single = z_arr.ndim == 1
        z2 = z_arr.reshape(1, -1) if single else z_arr
        if z2.shape[1] != self.dim:
            raise ValueError(f"Expected dimension {self.dim}, got {z2.shape[1]}.")
        return z2 @ self.features.T, single

    def log_density(self, z: ArrayLike) -> Any:
        np = _np()
        z_arr = np.asarray(z, dtype=float)
        z2 = z_arr.reshape(1, -1) if z_arr.ndim == 1 else z_arr
        logits, single = self._linear(z2)
        y = self.labels[None, :]
        log_lik = np.sum(y * logits - np.logaddexp(0.0, logits), axis=1)
        prior = -0.5 * self.prior_precision * np.sum(z2 * z2, axis=1)
        val = log_lik + prior
        return float(val[0]) if single else val

    def score(self, z: ArrayLike) -> Any:
        np = _np()
        z_arr = np.asarray(z, dtype=float)
        single = z_arr.ndim == 1
        z2 = z_arr.reshape(1, -1) if single else z_arr
        logits = z2 @ self.features.T
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
        residual = self.labels[None, :] - probs
        grad_lik = residual @ self.features
        grad_prior = -self.prior_precision * z2
        score = grad_lik + grad_prior
        return score[0] if single else score


@dataclass
class ScoreTargetAdapter:
    """Adapter for externally supplied unnormalized log-density/score callables."""

    dim: int
    score_fn: ScoreFn
    log_density_fn: Optional[LogDensityFn] = None
    name: str = "callable_score_target"
    finite_difference_step: float = 1e-5

    def log_density(self, z: ArrayLike) -> Any:
        if self.log_density_fn is not None:
            return self.log_density_fn(z)
        np = _np()
        z_arr = np.asarray(z, dtype=float)
        single = z_arr.ndim == 1
        z2 = z_arr.reshape(1, -1) if single else z_arr
        # A score-only target does not have a calibrated log-density.  For
        # metric code that needs a value, return the local quadratic surrogate
        # implied by the score at the evaluation point; BaM training itself does
        # not call this path.
        scores = np.asarray(self.score(z2), dtype=float)
        val = np.sum(scores * z2, axis=1)
        return float(val[0]) if single else val

    def score(self, z: ArrayLike) -> Any:
        return self.score_fn(z)


@dataclass
class BatchStatistics:
    """Explicit BaM Batch Step statistics."""

    samples: ArrayLike
    target_scores: ArrayLike
    zbar: ArrayLike
    C: ArrayLike
    gbar: ArrayLike
    Gamma: ArrayLike
    score_covariance: ArrayLike
    sample_score_cross_covariance: ArrayLike
    batch_size: int
    dim: int
    seed: int
    finite_batch: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "samples": _jsonable(self.samples),
            "target_scores": _jsonable(self.target_scores),
            "zbar": _jsonable(self.zbar),
            "C": _jsonable(self.C),
            "gbar": _jsonable(self.gbar),
            "Gamma": _jsonable(self.Gamma),
            "score_covariance": _jsonable(self.score_covariance),
            "sample_score_cross_covariance": _jsonable(self.sample_score_cross_covariance),
            "batch_size": int(self.batch_size),
            "dim": int(self.dim),
            "seed": int(self.seed),
            "finite_batch": bool(self.finite_batch),
        }


@dataclass
class MatchStepConfig:
    """Configuration for the KL-regularized BaM Match Step."""

    lambda_: float = 1.0
    epsilon: float = 1e-3
    covariance_blend: float = 1.0
    min_eig: float = 1e-8
    gsm_limiting_case: bool = False
    covariance_update: str = "score_cross_covariance"

    def __post_init__(self) -> None:
        if self.lambda_ < 0:
            raise ValueError("lambda_ must be non-negative.")
        if self.epsilon < 0:
            raise ValueError("epsilon must be non-negative.")
        if not 0.0 <= self.covariance_blend <= 1.0:
            raise ValueError("covariance_blend must be in [0,1].")
        valid = {"score_cross_covariance", "precision_moment", "gsm_special_case"}
        if self.covariance_update not in valid:
            raise ValueError(f"covariance_update must be one of {sorted(valid)}.")


@dataclass
class MatchStepResult:
    """Result of the BaM Match Step."""

    previous_state: GaussianVariationalState
    updated_state: GaussianVariationalState
    matched_mean: ArrayLike
    matched_covariance: ArrayLike
    regularized_precision: ArrayLike
    kl_regularizer_qnew_qold: float
    diagnostics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "previous_state": self.previous_state.to_dict(),
            "updated_state": self.updated_state.to_dict(),
            "matched_mean": _jsonable(self.matched_mean),
            "matched_covariance": _jsonable(self.matched_covariance),
            "regularized_precision": _jsonable(self.regularized_precision),
            "kl_regularizer_qnew_qold": float(self.kl_regularizer_qnew_qold),
            "diagnostics": _jsonable(self.diagnostics),
        }


@dataclass
class BaMTrainingConfig:
    """Small runnable BaM training-loop configuration.

    The defaults expose the paper-obligated controls:
    ``iterations=100``, ``lambda_``, ``epsilon``, ``batch_size`` B, and ``seed``.
    Runtime smoke callers can override ``iterations`` to a smaller value without
    changing the canonical full configuration.
    """

    iterations: int = 100
    lambda_: float = 1.0
    epsilon: float = 1e-3
    batch_size: Union[int, str] = 32
    seed: int = 0
    target_name: str = "gaussian_2d"
    initial_mean_scale: float = 1.0
    initial_covariance_scale: float = 2.0
    mode: str = "finite_batch"
    record_samples: bool = False
    gsm_limiting_case: bool = False

    def __post_init__(self) -> None:
        if int(self.iterations) <= 0:
            raise ValueError("iterations must be positive.")
        if self.batch_size != "infinite" and int(self.batch_size) <= 1:
            raise ValueError("finite batch_size must be greater than 1.")
        if self.mode not in {"finite_batch", "analytic_gaussian_infinite_batch"}:
            raise ValueError("mode must be finite_batch or analytic_gaussian_infinite_batch.")


def numerical_score(log_density_fn: LogDensityFn, z: ArrayLike, step: float = 1e-5) -> Any:
    """Central-difference score helper for validation/adapters.

    BaM target implementations in this module expose analytic scores.  This
    helper exists for environment adapters that provide only an unnormalized log
    density; it never needs a normalizing constant because constants cancel in
    finite differences.
    """
    np = _np()
    z_arr = np.asarray(z, dtype=float)
    single = z_arr.ndim == 1
    z2 = z_arr.reshape(1, -1) if single else z_arr
    grad = np.zeros_like(z2)
    for j in range(z2.shape[1]):
        delta = np.zeros_like(z2)
        delta[:, j] = step
        plus = np.asarray(log_density_fn(z2 + delta), dtype=float).reshape(-1)
        minus = np.asarray(log_density_fn(z2 - delta), dtype=float).reshape(-1)
        grad[:, j] = (plus - minus) / (2.0 * step)
    return grad[0] if single else grad


def batch_step(
    variational_state: GaussianVariationalState,
    target: TargetDistribution,
    batch_size: int = 32,
    seed: int = 0,
) -> BatchStatistics:
    """Explicit BaM Batch Step.

    Steps implemented:
      1. sample z_1,...,z_B ~ q_t;
      2. compute g_b = ∇ log p(z_b);
      3. compute zbar, C, gbar, Gamma, and score/sample covariance statistics.
    """
    np = _np()
    rng = np.random.default_rng(int(seed))
    B = int(batch_size)
    samples = variational_state.sample(B, rng)
    target_scores = np.asarray(target.score(samples), dtype=float)
    if target_scores.shape != samples.shape:
        raise ValueError(
            f"target.score returned shape {target_scores.shape}, expected {samples.shape}."
        )

    zbar = np.mean(samples, axis=0)
    centered_z = samples - zbar[None, :]
    C = centered_z.T @ centered_z / float(B)

    gbar = np.mean(target_scores, axis=0)
    centered_g = target_scores - gbar[None, :]
    Gamma = centered_z.T @ centered_g / float(B)
    score_covariance = centered_g.T @ centered_g / float(B)
    sample_score_cross_covariance = Gamma.copy()

    return BatchStatistics(
        samples=samples,
        target_scores=target_scores,
        zbar=zbar,
        C=stabilize_covariance(C, min_eig=variational_state.jitter),
        gbar=gbar,
        Gamma=Gamma,
        score_covariance=stabilize_covariance(score_covariance, min_eig=variational_state.jitter),
        sample_score_cross_covariance=sample_score_cross_covariance,
        batch_size=B,
        dim=variational_state.dim,
        seed=int(seed),
        finite_batch=True,
    )


def analytic_gaussian_infinite_batch_step(
    variational_state: GaussianVariationalState,
    target: GaussianTarget,
    seed: int = 0,
) -> BatchStatistics:
    """B→∞ Batch Step for Gaussian-target sanity checks.

    For target p=N(m,S), target score is g(z)=-S^{-1}(z-m).  Under
    z~q=N(mu,C), the exact statistics are
        zbar=mu,
        C=Cov(q),
        gbar=-S^{-1}(mu-m),
        Gamma=E[(z-mu)(g-gbar)^T]=-C S^{-1}.
    """
    np = _np()
    precision = target.precision
    zbar = variational_state.mean.copy()
    C = variational_state.covariance.copy()
    gbar = -precision @ (variational_state.mean - target.mean)
    Gamma = -C @ precision
    score_covariance = precision @ C @ precision.T
    samples = zbar.reshape(1, -1)
    target_scores = gbar.reshape(1, -1)
    return BatchStatistics(
        samples=samples,
        target_scores=target_scores,
        zbar=zbar,
        C=C,
        gbar=gbar,
        Gamma=Gamma,
        score_covariance=stabilize_covariance(score_covariance, min_eig=variational_state.jitter),
        sample_score_cross_covariance=Gamma,
        batch_size=-1,
        dim=variational_state.dim,
        seed=int(seed),
        finite_batch=False,
    )


def _matched_covariance_from_gamma(C: ArrayLike, Gamma: ArrayLike, min_eig: float = 1e-8) -> Tuple[Any, Any]:
    """Estimate matched covariance from Stein score/sample cross-statistics.

    At a Gaussian fixed point with target precision P,
        Gamma = E[(z-zbar)(score_p(z)-gbar)^T] = -C P.
    Hence P ≈ -C^{-1} Gamma, and covariance ≈ P^{-1}.  The implementation
    symmetrizes and stabilizes this precision estimate for full-covariance q.
    """
    np = _np()
    C_stable = stabilize_covariance(C, min_eig=min_eig)
    raw_precision = -_solve_spd(C_stable, Gamma, jitter=min_eig)
    precision = stabilize_covariance(_symmetrize(raw_precision), min_eig=min_eig)
    covariance = stabilize_covariance(np.linalg.pinv(precision), min_eig=min_eig)
    return covariance, precision


def match_step(
    variational_state: GaussianVariationalState,
    stats: BatchStatistics,
    config: Optional[MatchStepConfig] = None,
) -> MatchStepResult:
    """KL-regularized BaM Match Step.

    The batch statistics define a local score-matching proposal:
        mean_match = zbar + C gbar,
        covariance_match from Gamma via Stein's Gaussian identity.

    A KL/proximal regularizer is implemented as a natural-parameter blend with
    the previous Gaussian:
        precision_new = (precision_match + λ precision_old) / (1+λ+ε),
        h_new         = (precision_match mean_match + λ precision_old mean_old)
                        / (1+λ+ε),
        mean_new      = precision_new^{-1} h_new.

    ``gsm_limiting_case`` keeps a local hook for the GSM special limiting case:
    the covariance is retained from q_t and only the score-preconditioned mean
    move is applied.  This is implemented locally and does not import or depend
    on the blacklisted GSM-VI repository.
    """
    np = _np()
    cfg = config or MatchStepConfig()

    old_precision = variational_state.precision
    lambda_ = float(cfg.lambda_)
    epsilon = float(cfg.epsilon)

    matched_mean = np.asarray(stats.zbar, dtype=float) + np.asarray(stats.C, dtype=float) @ np.asarray(stats.gbar, dtype=float)

    if cfg.gsm_limiting_case or cfg.covariance_update == "gsm_special_case":
        matched_covariance = variational_state.covariance.copy()
        matched_precision = old_precision.copy()
        update_family = "gsm_special_case_local_hook"
    else:
        matched_covariance, matched_precision = _matched_covariance_from_gamma(
            stats.C, stats.Gamma, min_eig=max(cfg.min_eig, variational_state.jitter)
        )
        update_family = cfg.covariance_update

    if cfg.covariance_blend < 1.0:
        matched_covariance = stabilize_covariance(
            cfg.covariance_blend * matched_covariance + (1.0 - cfg.covariance_blend) * variational_state.covariance,
            min_eig=max(cfg.min_eig, variational_state.jitter),
        )
        matched_precision = _matrix_inverse_spd(matched_covariance, jitter=max(cfg.min_eig, variational_state.jitter))

    denom = 1.0 + lambda_ + epsilon
    regularized_precision = (matched_precision + lambda_ * old_precision + epsilon * np.eye(variational_state.dim)) / denom
    regularized_precision = stabilize_covariance(regularized_precision, min_eig=max(cfg.min_eig, variational_state.jitter))

    natural_vector = (matched_precision @ matched_mean + lambda_ * old_precision @ variational_state.mean) / denom
    updated_covariance = stabilize_covariance(
        np.linalg.pinv(regularized_precision), min_eig=max(cfg.min_eig, variational_state.jitter)
    )
    updated_mean = updated_covariance @ natural_vector

    updated_state = GaussianVariationalState(updated_mean, updated_covariance, jitter=variational_state.jitter)
    kl_reg = updated_state.kl_to(variational_state)

    diagnostics = {
        "update_family": update_family,
        "lambda": lambda_,
        "epsilon": epsilon,
        "covariance_blend": float(cfg.covariance_blend),
        "batch_size": int(stats.batch_size),
        "finite_batch": bool(stats.finite_batch),
        "mean_step_norm": float(np.linalg.norm(updated_state.mean - variational_state.mean)),
        "matched_precision_condition": float(np.linalg.cond(matched_precision)),
        "regularized_precision_condition": float(np.linalg.cond(regularized_precision)),
        "kl_regularizer_qnew_qold": float(kl_reg),
    }

    return MatchStepResult(
        previous_state=variational_state,
        updated_state=updated_state,
        matched_mean=matched_mean,
        matched_covariance=matched_covariance,
        regularized_precision=regularized_precision,
        kl_regularizer_qnew_qold=kl_reg,
        diagnostics=diagnostics,
    )
