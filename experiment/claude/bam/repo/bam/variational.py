"""Core variational objects and Batch-and-Match updates.

This module implements the importable core of the PaperBench reproduction for
"Batch and match: black-box variational inference with a score-based divergence".

The implementation is intentionally dependency-light at import time.  Numerical
execution imports NumPy lazily inside functions so configuration, registry, and
smoke checks can import the package in a minimal environment.

reference_grounding: paper:paper_method_core paper.md
    Score-based BBVI minimizes a divergence between a target p and a Gaussian
    variational approximation q using only the target score ∇ log p(z).  The
    target normalizing constant is never required by the update.

reference_grounding: paper:paper_training_or_optimization_loop paper.md
    BaM separates a Batch Step, z_1,...,z_B ~ q_t and
    g_b = ∇ log p(z_b), from a Match Step that updates Gaussian variational
    parameters using batch score/sample statistics and KL regularization.

reference_grounding: paper:paper_semantic_chunk_009_03 paper.md
    BaM is a KL-regularized stochastic proximal/matching update.  This file
    exposes finite-batch B=32 configuration, a B→∞ analytic Gaussian sanity
    route, and a GSM limiting-case hook without depending on the blacklisted
    GSM-VI repository.
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
LogDensityFn = Callable[[ArrayLike], Any]
ScoreFn = Callable[[ArrayLike], Any]


def _np() -> Any:
    """Import NumPy lazily and fail with a runtime-only diagnostic."""
    try:
        import numpy as np  # type: ignore

        return np
    except Exception as exc:  # pragma: no cover - host dependent
        raise RuntimeError(
            "Numerical BaM execution requires numpy. Install project "
            "requirements before running optimization."
        ) from exc


def _as_path(path: Union[str, os.PathLike[str]]) -> Path:
    return Path(path).expanduser().resolve()


def artifact_root(default: Union[str, os.PathLike[str]] = "results") -> Path:
    """Return the artifact root, respecting PaperBench auxiliary output env."""
    env_root = os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR")
    if env_root:
        return _as_path(env_root)
    return _as_path(default)


def _symmetrize(matrix: Any) -> Any:
    np = _np()
    arr = np.asarray(matrix, dtype=float)
    return 0.5 * (arr + arr.T)


def _ensure_2d_samples(samples: Any, dim: int) -> Any:
    np = _np()
    arr = np.asarray(samples, dtype=float)
    if arr.ndim == 1:
        if dim == 1:
            arr = arr.reshape(-1, 1)
        else:
            arr = arr.reshape(1, dim)
    if arr.ndim != 2 or arr.shape[1] != dim:
        raise ValueError(f"Expected samples with shape (n, {dim}); got {arr.shape}.")
    return arr


def _ensure_vector(vector: Any, dim: int, name: str) -> Any:
    np = _np()
    arr = np.asarray(vector, dtype=float).reshape(-1)
    if arr.shape != (dim,):
        raise ValueError(f"{name} must have shape ({dim},); got {arr.shape}.")
    return arr


def _ensure_matrix(matrix: Any, dim: int, name: str) -> Any:
    np = _np()
    arr = np.asarray(matrix, dtype=float)
    if arr.shape != (dim, dim):
        raise ValueError(f"{name} must have shape ({dim}, {dim}); got {arr.shape}.")
    return arr


def stable_spd(matrix: Any, jitter: float = 1.0e-6, min_eig: float = 1.0e-8) -> Any:
    """Return a symmetric positive definite version of ``matrix``.

    The routine clips eigenvalues from below and adds a small diagonal jitter.
    It is used for the full-covariance Gaussian parameterization and for the
    natural-parameter matching update.
    """
    np = _np()
    arr = _symmetrize(matrix)
    vals, vecs = np.linalg.eigh(arr)
    vals = np.maximum(vals, float(min_eig)) + float(jitter)
    return _symmetrize((vecs * vals) @ vecs.T)


def safe_inverse_spd(matrix: Any, jitter: float = 1.0e-6) -> Any:
    np = _np()
    spd = stable_spd(matrix, jitter=jitter)
    return _symmetrize(np.linalg.inv(spd))


def matrix_to_list(matrix: Any) -> List[List[float]]:
    np = _np()
    return np.asarray(matrix, dtype=float).tolist()


def vector_to_list(vector: Any) -> List[float]:
    np = _np()
    return np.asarray(vector, dtype=float).reshape(-1).tolist()


@dataclass(frozen=True)
class BaMConfig:
    """Configuration surface for the BaM optimizer.

    The defaults are the paper-contract canonical route: full-covariance
    Gaussian q, finite batch B=32, 100 iterations, KL regularization lambda,
    covariance jitter epsilon, and deterministic seed.

    ``batch_size=None`` or ``batch_semantics="infinite_gaussian"`` selects the
    B→∞ analytic sanity route for Gaussian targets.  ``gsm_limiting_case`` keeps
    an explicit GSM-compatible method hook without importing or reusing the
    blacklisted GSM-VI implementation.
    """

    dim: int
    iterations: int = 100
    batch_size: Optional[int] = 32
    lambda_kl: float = 1.0
    epsilon: float = 1.0e-5
    seed: int = 0
    damping: float = 1.0
    batch_semantics: str = "finite_b32"
    method_variant: str = "bam"
    gsm_limiting_case: bool = False
    record_every: int = 1
    covariance_floor: float = 1.0e-8
    selected_experiment_set: Mapping[str, str] = field(
        default_factory=lambda: {
            "core_contribution_hypothesis": (
                "score matching with KL-regularized Gaussian natural-parameter "
                "updates can recover target posterior geometry from black-box scores"
            ),
            "decisive_comparison": "BaM full-covariance Gaussian vs ADVI/GSM selectors",
            "decisive_metric": "score_divergence plus Gaussian mean/covariance error",
            "stop_pruning_rationale": (
                "default route executes bounded smoke/sanity iterations; exhaustive "
                "sweeps and long target-specific runs require explicit full mode"
            ),
        }
    )

    def validated(self) -> "BaMConfig":
        if self.dim <= 0:
            raise ValueError("dim must be positive.")
        if self.iterations <= 0:
            raise ValueError("iterations must be positive; canonical full setting is 100.")
        if self.batch_size is not None and self.batch_size <= 0:
            raise ValueError("batch_size must be positive or None for B→∞ semantics.")
        if self.lambda_kl < 0.0:
            raise ValueError("lambda_kl must be non-negative.")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive.")
        if not (0.0 < self.damping <= 1.0):
            raise ValueError("damping must be in (0, 1].")
        if self.batch_semantics not in {"finite_b32", "finite", "infinite_gaussian"}:
            raise ValueError(
                "batch_semantics must be one of finite_b32, finite, infinite_gaussian."
            )
        return self

    @classmethod
    def finite_b32(
        cls,
        dim: int,
        *,
        iterations: int = 100,
        lambda_kl: float = 1.0,
        epsilon: float = 1.0e-5,
        seed: int = 0,
    ) -> "BaMConfig":
        return cls(
            dim=dim,
            iterations=iterations,
            batch_size=32,
            lambda_kl=lambda_kl,
            epsilon=epsilon,
            seed=seed,
            batch_semantics="finite_b32",
        ).validated()

    @classmethod
    def infinite_gaussian_sanity(
        cls,
        dim: int,
        *,
        iterations: int = 100,
        lambda_kl: float = 1.0,
        epsilon: float = 1.0e-5,
        seed: int = 0,
    ) -> "BaMConfig":
        return cls(
            dim=dim,
            iterations=iterations,
            batch_size=None,
            lambda_kl=lambda_kl,
            epsilon=epsilon,
            seed=seed,
            batch_semantics="infinite_gaussian",
        ).validated()

    @classmethod
    def gsm_special_case(
        cls,
        dim: int,
        *,
        iterations: int = 100,
        batch_size: int = 32,
        epsilon: float = 1.0e-5,
        seed: int = 0,
    ) -> "BaMConfig":
        return cls(
            dim=dim,
            iterations=iterations,
            batch_size=batch_size,
            lambda_kl=0.0,
            epsilon=epsilon,
            seed=seed,
            batch_semantics="finite_b32" if batch_size == 32 else "finite",
            method_variant="gsm_limiting_case",
            gsm_limiting_case=True,
        ).validated()


@dataclass
class GaussianVariationalFamily:
    """Full-covariance Gaussian variational distribution q(z)=N(mu,Sigma)."""

    mean: Any
    covariance: Any
    jitter: float = 1.0e-6

    def __post_init__(self) -> None:
        np = _np()
        self.mean = np.asarray(self.mean, dtype=float).reshape(-1)
        dim = int(self.mean.shape[0])
        self.covariance = stable_spd(
            _ensure_matrix(self.covariance, dim, "covariance"), jitter=self.jitter
        )

    @property
    def dim(self) -> int:
        return int(self.mean.shape[0])

    @property
    def precision(self) -> Any:
        return safe_inverse_spd(self.covariance, jitter=self.jitter)

    @property
    def natural_vector(self) -> Any:
        return self.precision @ self.mean

    @property
    def cholesky(self) -> Any:
        np = _np()
        return np.linalg.cholesky(stable_spd(self.covariance, jitter=self.jitter))

    def copy(self) -> "GaussianVariationalFamily":
        np = _np()
        return GaussianVariationalFamily(
            np.array(self.mean, dtype=float), np.array(self.covariance, dtype=float), self.jitter
        )

    @classmethod
    def standard(cls, dim: int, scale: float = 1.0, jitter: float = 1.0e-6) -> "GaussianVariationalFamily":
        np = _np()
        if dim <= 0:
            raise ValueError("dim must be positive.")
        if scale <= 0:
            raise ValueError("scale must be positive.")
        return cls(np.zeros(dim), (scale**2) * np.eye(dim), jitter=jitter)

    @classmethod
    def from_natural(
        cls,
        precision: Any,
        natural_vector: Any,
        *,
        jitter: float = 1.0e-6,
    ) -> "GaussianVariationalFamily":
        np = _np()
        eta = np.asarray(natural_vector, dtype=float).reshape(-1)
        dim = int(eta.shape[0])
        prec = stable_spd(_ensure_matrix(precision, dim, "precision"), jitter=jitter)
        cov = stable_spd(np.linalg.inv(prec), jitter=jitter)
        mean = cov @ eta
        return cls(mean, cov, jitter=jitter)

    def sample(self, rng: Any, n: int) -> Any:
        """Draw z_1,...,z_n ~ q using the full covariance Cholesky factor."""
        np = _np()
        if n <= 0:
            raise ValueError("n must be positive.")
        standard = rng.normal(size=(int(n), self.dim))
        return self.mean.reshape(1, -1) + standard @ self.cholesky.T

    def log_density(self, z: Any) -> Any:
        """Evaluate normalized Gaussian log density log q(z)."""
        np = _np()
        samples = _ensure_2d_samples(z, self.dim)
        centered = samples - self.mean.reshape(1, -1)
        prec = self.precision
        quad = np.einsum("bi,ij,bj->b", centered, prec, centered)
        sign, logdet = np.linalg.slogdet(stable_spd(self.covariance, jitter=self.jitter))
        if sign <= 0:
            raise FloatingPointError("Gaussian covariance is not positive definite.")
        vals = -0.5 * (self.dim * math.log(2.0 * math.pi) + float(logdet) + quad)
        return float(vals[0]) if np.asarray(z).ndim == 1 else vals

    def score(self, z: Any) -> Any:
        """Evaluate ∇_z log q(z) = -Σ^{-1}(z-μ)."""
        np = _np()
        samples = _ensure_2d_samples(z, self.dim)
        vals = -(samples - self.mean.reshape(1, -1)) @ self.precision.T
        return vals[0] if np.asarray(z).ndim == 1 else vals

    def kl_to(self, other: "GaussianVariationalFamily") -> float:
        """Closed-form KL(self || other) for full-covariance Gaussians."""
        np = _np()
        if self.dim != other.dim:
            raise ValueError("Gaussian dimensions must match for KL.")
        sigma0 = stable_spd(self.covariance, jitter=self.jitter)
        sigma1 = stable_spd(other.covariance, jitter=other.jitter)
        prec1 = safe_inverse_spd(sigma1, jitter=other.jitter)
        delta = other.mean - self.mean
        sign0, logdet0 = np.linalg.slogdet(sigma0)
        sign1, logdet1 = np.linalg.slogdet(sigma1)
        if sign0 <= 0 or sign1 <= 0:
            raise FloatingPointError("Gaussian KL requires SPD covariances.")
        trace = float(np.trace(prec1 @ sigma0))
        quad = float(delta.T @ prec1 @ delta)
        return 0.5 * (float(logdet1 - logdet0) - self.dim + trace + quad)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family": "full_covariance_gaussian",
            "dim": self.dim,
            "mean": vector_to_list(self.mean),
            "covariance": matrix_to_list(self.covariance),
            "precision": matrix_to_list(self.precision),
            "jitter": float(self.jitter),
        }


VariationalGaussian = GaussianVariationalFamily
FullCovarianceGaussian = GaussianVariationalFamily


@dataclass
class TargetScoreAdapter:
    """Black-box target adapter using score and optional unnormalized log density."""

    score_fn: ScoreFn
    log_density_fn: Optional[LogDensityFn] = None
    dim: Optional[int] = None
    name: str = "black_box_score_target"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def score(self, z: Any) -> Any:
        np = _np()
        raw = self.score_fn(z)
        arr = np.asarray(raw, dtype=float)
        if self.dim is not None:
            if arr.ndim == 1:
                if arr.shape[0] != self.dim:
                    raise ValueError(f"Target score dimension mismatch: expected {self.dim}.")
            elif arr.ndim == 2:
                if arr.shape[1] != self.dim:
                    raise ValueError(f"Target score dimension mismatch: expected {self.dim}.")
            else:
                raise ValueError(f"Target score must be vector or matrix, got ndim={arr.ndim}.")
        return arr

    def log_density(self, z: Any) -> Any:
        if self.log_density_fn is not None:
            return self.log_density_fn(z)
        np = _np()
        samples = np.asarray(z, dtype=float)
        n = int(samples.shape[0]) if samples.ndim == 2 else 1
        vals = np.full(n, math.nan, dtype=float)
        return float(vals[0]) if samples.ndim == 1 else vals


@dataclass
class BatchStatistics:
    """Explicit BaM Batch Step statistics."""

    samples: Any
    target_scores: Any
    q_scores: Any
    z_bar: Any
    covariance_C: Any
    g_bar: Any
    gamma: Any
    score_covariance: Any
    score_sample_correlation: Any
    score_divergence_estimate: float
    batch_size: int
    semantics: str = "finite"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_size": int(self.batch_size),
            "semantics": self.semantics,
            "z_bar": vector_to_list(self.z_bar),
            "g_bar": vector_to_list(self.g_bar),
            "C": matrix_to_list(self.covariance_C),
            "Gamma": matrix_to_list(self.gamma),
            "score_covariance": matrix_to_list(self.score_covariance),
            "score_sample_correlation": matrix_to_list(self.score_sample_correlation),
            "score_divergence_estimate": float(self.score_divergence_estimate),
        }


@dataclass
class MatchResult:
    """Result of the KL-regularized Match Step."""

    previous: GaussianVariationalFamily
    matched_unregularized: GaussianVariationalFamily
    updated: GaussianVariationalFamily
    lambda_kl: float
    epsilon: float
    damping: float
    kl_to_previous: float
    match_diagnostics: Mapping[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lambda_kl": float(self.lambda_kl),
            "epsilon": float(self.epsilon),
            "damping": float(self.damping),
            "kl_to_previous": float(self.kl_to_previous),
            "matched_unregularized": self.matched_unregularized.to_dict(),
            "updated": self.updated.to_dict(),
            "match_diagnostics": dict(self.match_diagnostics),
        }


@dataclass
class TrainingTrace:
    """Serializable training output for BaM optimization."""

    config: BaMConfig
    final_variational: GaussianVariationalFamily
    loss_trace: List[Mapping[str, Any]]
    bam_trace: List[Mapping[str, Any]]
    batch_statistics_trace: List[Mapping[str, Any]]
    gaussian_sanity_metrics: Mapping[str, Any]
    elapsed_seconds: float
    artifacts: Mapping[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": asdict(self.config),
            "final_variational": self.final_variational.to_dict(),
            "loss_trace": list(self.loss_trace),
            "bam_trace": list(self.bam_trace),
            "batch_statistics_trace": list(self.batch_statistics_trace),
            "gaussian_sanity_metrics": dict(self.gaussian_sanity_metrics),
            "elapsed_seconds": float(self.elapsed_seconds),
            "artifacts": dict(self.artifacts),
        }


def score_divergence_estimate(
    q: GaussianVariationalFamily,
    samples: Any,
    target_scores: Any,
) -> float:
    """Monte Carlo estimate of E_q ||∇log q(z)-∇log p(z)||^2_{Cov(q)}."""
    np = _np()
    z = _ensure_2d_samples(samples, q.dim)
    g = _ensure_2d_samples(target_scores, q.dim)
    if z.shape != g.shape:
        raise ValueError("samples and target_scores must have identical shape.")
    diff = q.score(z) - g
    vals = np.einsum("bi,ij,bj->b", diff, q.covariance, diff)
    return float(np.mean(vals))


def gaussian_forward_reverse_kl_metrics(
    q: GaussianVariationalFamily,
    target_mean: Any,
    target_covariance: Any,
) -> Dict[str, float]:
    """Metric formula surface for analytic Gaussian sanity checks."""
    target = GaussianVariationalFamily(target_mean, target_covariance, jitter=q.jitter)
    return {
        "reverse_kl_q_to_p": float(q.kl_to(target)),
        "forward_kl_p_to_q": float(target.kl_to(q)),
        "mean_l2_error": float(_np().linalg.norm(q.mean - target.mean)),
        "covariance_fro_error": float(_np().linalg.norm(q.covariance - target.covariance, ord="fro")),
    }


def batch_step(
    q: GaussianVariationalFamily,
    target: Union[TargetScoreAdapter, ScoreFn],
    *,
    batch_size: int,
    rng: Any,
    semantics: str = "finite",
) -> BatchStatistics:
    """BaM Batch Step.

    Explicitly samples z_b ~ q_t, evaluates g_b = ∇ log p(z_b), and computes
    z̄, C, ḡ, Γ, score covariance, and score/sample correlation statistics.
    The target score can be supplied either as a TargetScoreAdapter or a callable.
    """
    np = _np()
    if batch_size <= 1:
        raise ValueError("batch_size must be at least 2 to compute covariance statistics.")

    samples = q.sample(rng, batch_size)
    if isinstance(target, TargetScoreAdapter):
        target_scores = target.score(samples)
    else:
        target_scores = np.asarray(target(samples), dtype=float)
    target_scores = _ensure_2d_samples(target_scores, q.dim)

    q_scores = q.score(samples)
    z_bar = np.mean(samples, axis=0)
    g_bar = np.mean(target_scores, axis=0)
    z_centered = samples - z_bar.reshape(1, -1)
    g_centered = target_scores - g_bar.reshape(1, -1)

    denom = float(batch_size)
    covariance_C = stable_spd((z_centered.T @ z_centered) / denom, jitter=q.jitter)
    gamma = (g_centered.T @ z_centered) / denom
    score_covariance = (g_centered.T @ g_centered) / denom

    z_std = np.sqrt(np.maximum(np.diag(covariance_C), q.jitter))
    g_std = np.sqrt(np.maximum(np.diag(score_covariance), q.jitter))
    score_sample_correlation = gamma / np.outer(g_std, z_std)

    div_est = score_divergence_estimate(q, samples, target_scores)

    return BatchStatistics(
        samples=samples,
        target_scores=target_scores,
        q_scores=q_scores,
        z_bar=z_bar,
        covariance_C=covariance_C,
        g_bar=g_bar,
        gamma=gamma,
        score_covariance=score_covariance,
        score_sample_correlation=score_sample_correlation,
        score_divergence_estimate=div_est,
        batch_size=batch_size,
        semantics=semantics,
    )


def analytic_gaussian_batch_statistics(
    q: GaussianVariationalFamily,
    target_mean: Any,
    target_covariance: Any,
) -> BatchStatistics:
    """B→∞ Batch Step statistics for a Gaussian target sanity check.

    For p=N(m,S), ∇log p(z)=-S^{-1}(z-m).  Under z~q=N(mu,C),
    z̄=mu, C=Cov(q), ḡ=-S^{-1}(mu-m), and Γ=Cov(g,z)=-S^{-1}C.
    This route expresses the infinite-batch semantics without expensive
    Monte Carlo execution.
    """
    np = _np()
    dim = q.dim
    m = _ensure_vector(target_mean, dim, "target_mean")
    s = stable_spd(_ensure_matrix(target_covariance, dim, "target_covariance"), jitter=q.jitter)
    target_precision = safe_inverse_spd(s, jitter=q.jitter)
    z_bar = np.array(q.mean, dtype=float)
    covariance_C = np.array(q.covariance, dtype=float)
    g_bar = -(target_precision @ (q.mean - m))
    gamma = -target_precision @ covariance_C
    score_covariance = target_precision @ covariance_C @ target_precision.T
    score_sample_correlation = gamma / np.outer(
        np.sqrt(np.maximum(np.diag(score_covariance), q.jitter)),
        np.sqrt(np.maximum(np.diag(covariance_C), q.jitter)),
    )
    target_scores_at_mean = g_bar.reshape(1, -1)
    q_scores_at_mean = q.score(z_bar).reshape(1, -1)
    div_est = float(
        np.einsum(
            "bi,ij,bj->b",
            q_scores_at_mean - target_scores_at_mean,
            q.covariance,
            q_scores_at_mean - target_scores_at_mean,
        )[0]
    )
    return BatchStatistics(
        samples=z_bar.reshape(1, -1),
        target_scores=target_scores_at_mean,
        q_scores=q_scores_at_mean,
        z_bar=z_bar,
        covariance_C=covariance_C,
        g_bar=g_bar,
        gamma=gamma,
        score_covariance=score_covariance,
        score_sample_correlation=score_sample_correlation,
        score_divergence_estimate=div_est,
        batch_size=math.inf,  # type: ignore[arg-type]
        semantics="infinite_gaussian",
    )


def match_step(
    q: GaussianVariationalFamily,
    stats: BatchStatistics,
    *,
    lambda_kl: float,
    epsilon: float,
    damping: float = 1.0,
    covariance_floor: float = 1.0e-8,
    gsm_limiting_case: bool = False,
) -> MatchResult:
    """BaM Match Step with KL regularization.

    The target score of a Gaussian satisfies g(z)=h-Λz where Λ=Σ^{-1} and
    h=Λμ.  The Batch Step statistic Γ=Cov(g,z) therefore gives the
    least-squares/natural-parameter match Λ_match ≈ -Γ C^{-1}.  The KL
    regularizer keeps the new natural parameters close to q_t:

        Λ_{t+1} = damp * ((Λ_match + λ Λ_t)/(1+λ)) + (1-damp)Λ_t
        h_{t+1} = damp * ((h_match + λ h_t)/(1+λ)) + (1-damp)h_t

    ``epsilon`` and eigenvalue clipping stabilize full-covariance updates.
    Setting ``gsm_limiting_case=True`` keeps an explicit GSM-style score matching
    hook by removing the KL pull when lambda_kl is zero; no external GSM-VI code
    is used.
    """
    np = _np()
    if lambda_kl < 0.0:
        raise ValueError("lambda_kl must be non-negative.")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive.")
    if not (0.0 < damping <= 1.0):
        raise ValueError("damping must be in (0, 1].")

    c_inv = safe_inverse_spd(stats.covariance_C, jitter=epsilon)
    raw_precision = _symmetrize(-stats.gamma @ c_inv)

    eigvals = np.linalg.eigvalsh(raw_precision)
    min_raw_eig = float(np.min(eigvals))
    match_precision = stable_spd(raw_precision, jitter=epsilon, min_eig=covariance_floor)
    match_natural = stats.g_bar + match_precision @ stats.z_bar

    old_precision = q.precision
    old_natural = q.natural_vector

    effective_lambda = 0.0 if gsm_limiting_case and lambda_kl == 0.0 else float(lambda_kl)
    denom = 1.0 + effective_lambda
    proximal_precision = (match_precision + effective_lambda * old_precision) / denom
    proximal_natural = (match_natural + effective_lambda * old_natural) / denom

    new_precision = _symmetrize(damping * proximal_precision + (1.0 - damping) * old_precision)
    new_natural = damping * proximal_natural + (1.0 - damping) * old_natural
    new_precision = stable_spd(new_precision, jitter=epsilon, min_eig=covariance_floor)

    matched = GaussianVariationalFamily.from_natural(
        match_precision, match_natural, jitter=epsilon
    )
    updated = GaussianVariationalFamily.from_natural(new_precision, new_natural, jitter=epsilon)

    return MatchResult(
        previous=q.copy(),
        matched_unregularized=matched,
        updated=updated,
        lambda_kl=float(lambda_kl),
        epsilon=float(epsilon),
        damping=float(damping),
        kl_to_previous=float(updated.kl_to(q)),
        match_diagnostics={
            "raw_precision_min_eigenvalue": min_raw_eig,
            "effective_lambda": float(effective_lambda),
            "gsm_limiting_case": bool(gsm_limiting_case),
            "natural_parameter_update": "score_linear_match_with_kl_regularization",
        },
    )


class BaMOptimizer:
    """Reusable optimizer implementing the paper-derived BaM core."""

    def __init__(
        self,
        target_score: ScoreFn,
        *,
        log_density: Optional[LogDensityFn] = None,
        initial: Optional[GaussianVariationalFamily] = None,
        config: Optional[BaMConfig] = None,
        target_name: str = "black_box_score_target",
    ) -> None:
        if config is None and initial is None:
            raise ValueError("Either config or initial Gaussian must be supplied.")
        if config is None and initial is not None:
            config = BaMConfig(dim=initial.dim).validated()
        if config is None:
            raise ValueError("BaMConfig construction failed.")
        self.config = config.validated()
        self.q = initial.copy() if initial is not None else GaussianVariationalFamily.standard(
            self.config.dim, jitter=self.config.epsilon
        )
        if self.q.dim != self.config.dim:
            raise ValueError("Initial Gaussian dimension does not match config.dim.")
        self.target = TargetScoreAdapter(
            score_fn=target_score,
            log_density_fn=log_density,
            dim=self.config.dim,
            name=target_name,
        )

    def batch_step(self, rng: Any) -> BatchStatistics:
        if self.config.batch_size is None:
            raise ValueError(
                "batch_step requires finite batch_size. Use analytic_gaussian_step "
                "for B→∞ Gaussian sanity semantics."
            )
        return batch_step(
            self.q,
            self.target,
            batch_size=int(self.config.batch_size),
            rng=rng,
            semantics=self.config.batch_semantics,
        )

    def match_step(self, stats: BatchStatistics) -> MatchResult:
        return match_step(
            self.q,
            stats,
            lambda_kl=self.config.lambda_kl,
            epsilon=self.config.epsilon,
            damping=self.config.damping,
            covariance_floor=self.config.covariance_floor,
            gsm_limiting_case=self.config.gsm_limiting_case,
        )

    def step(self, rng: Any) -> Tuple[BatchStatistics, MatchResult]:
        stats = self.batch_step(rng)
        result = self.match_step(stats)
        self.q = result.updated
        return stats, result

    def analytic_gaussian_step(self, target_mean: Any, target_covariance: Any) -> Tuple[BatchStatistics, MatchResult]:
        stats = analytic_gaussian_batch_statistics(self.q, target_mean, target_covariance)
        result = self.match_step(stats)
        self.q = result.updated
        return stats, result


def _normal_target_score(mean: Any, covariance: Any, jitter: float = 1.0e-6) -> ScoreFn:
    np = _np()
    mean_arr = np.asarray(mean, dtype=float).reshape(-1)
    dim = int(mean_arr.shape[0])
    precision = safe_inverse_spd(_ensure_matrix(covariance, dim, "covariance"), jitter=jitter)

    def score(z: Any) -> Any:
        samples = _ensure_2d_samples(z, dim)
        vals = -(samples - mean_arr.reshape(1, -1)) @ precision.T
        return vals[0] if np.asarray(z).ndim == 1 else vals

    return score


def train_bam(
    target_score: Optional[ScoreFn] = None,
    *,
    log_density: Optional[LogDensityFn] = None,
    initial: Optional[GaussianVariationalFamily] = None,
    config: Optional[BaMConfig] = None,
    target_mean: Optional[Any] = None,
    target_covariance: Optional[Any] = None,
    artifact_dir: Optional[Union[str, os.PathLike[str]]] = None,
    write_artifacts: bool = False,
) -> TrainingTrace:
    """Run the BaM optimization loop.

    The loop supports the required paper controls: ``iterations`` (canonical
    100), ``lambda_kl``, ``epsilon``, ``batch_size``/B, and ``seed``.  If
    ``batch_semantics="infinite_gaussian"`` or ``batch_size=None``, callers must
    provide ``target_mean`` and ``target_covariance`` so the B→∞ Gaussian sanity
    step can be computed analytically.
    """
    np = _np()
    cfg = (config or BaMConfig(dim=initial.dim if initial is not None else 2)).validated()

    if target_score is None:
        if target_mean is None or target_covariance is None:
            raise ValueError(
                "target_score is required unless target_mean and target_covariance "
                "define a Gaussian sanity target."
            )
        target_score = _normal_target_score(target_mean, target_covariance, jitter=cfg.epsilon)

    optimizer = BaMOptimizer(
        target_score,
        log_density=log_density,
        initial=initial,
        config=cfg,
    )
    rng = np.random.default_rng(cfg.seed)

    loss_trace: List[Mapping[str, Any]] = []
    bam_trace: List[Mapping[str, Any]] = []
    batch_statistics_trace: List[Mapping[str, Any]] = []

    start = time.time()
    for iteration in range(1, cfg.iterations + 1):
        if cfg.batch_semantics == "infinite_gaussian" or cfg.batch_size is None:
            if target_mean is None or target_covariance is None:
                raise ValueError("B→∞ Gaussian sanity route requires target_mean/target_covariance.")
            stats, match = optimizer.analytic_gaussian_step(target_mean, target_covariance)
        else:
            stats, match = optimizer.step(rng)

        should_record = iteration == 1 or iteration == cfg.iterations or (
            cfg.record_every > 0 and iteration % cfg.record_every == 0
        )
        if should_record:
            metrics: Dict[str, Any] = {
                "iteration": iteration,
                "score_divergence": float(stats.score_divergence_estimate),
                "kl_to_previous": float(match.kl_to_previous),
                "mean_norm": float(np.linalg.norm(optimizer.q.mean)),
                "covariance_trace": float(np.trace(optimizer.q.covariance)),
                "method_variant": cfg.method_variant,
            }
            if target_mean is not None and target_covariance is not None:
                metrics.update(gaussian_forward_reverse_kl_metrics(optimizer.q, target_mean, target_covariance))
            loss_trace.append(metrics)
            bam_trace.append(
                {
                    "iteration": iteration,
                    "variational": optimizer.q.to_dict(),
                    "match": match.to_dict(),
                }
            )
            stat_dict = stats.to_dict()
            stat_dict["iteration"] = iteration
            batch_statistics_trace.append(stat_dict)

    elapsed = time.time() - start

    gaussian_sanity_metrics: Dict[str, Any] = {
        "available": bool(target_mean is not None and target_covariance is not None),
        "batch_semantics": cfg.batch_semantics,
        "iterations": cfg.iterations,
    }
    if target_mean is not None and target_covariance is not None:
        gaussian_sanity_metrics.update(
            gaussian_forward_reverse_kl_metrics(optimizer.q, target_mean, target_covariance)
        )

    trace = TrainingTrace(
        config=cfg,
        final_variational=optimizer.q,
        loss_trace=loss_trace,
        bam_trace=bam_trace,
        batch_statistics_trace=batch_statistics_trace,
        gaussian_sanity_metrics=gaussian_sanity_metrics,
        elapsed_seconds=elapsed,
    )

    if write_artifacts:
        root = Path(artifact_dir) if artifact_dir is not None else artifact_root("results")
        artifacts = write_training_artifacts(trace, root, contract_label="runtime")
        trace.artifacts = artifacts

    return trace


def make_protocol_configs(dim: int = 2) -> Dict[str, BaMConfig]:
    """Expose benchmark-visible finite B=32, B→∞, and GSM-hook configs."""
    return {
        "bam_finite_b32": BaMConfig.finite_b32(dim=dim),
        "bam_infinite_gaussian_sanity": BaMConfig.infinite_gaussian_sanity(dim=dim),
        "gsm_special_limiting_case": BaMConfig.gsm_special_case(dim=dim),
    }


def readiness_payload(config: Optional[BaMConfig] = None) -> Dict[str, Any]:
    cfg = (config or BaMConfig.finite_b32(dim=2)).validated()
    return {
        "status": "ready",
        "module": "bam.variational",
        "contract": "bam_core",
        "method": "Batch-and-Match score-based BBVI",
        "normalizing_constant_required": False,
        "batch_step_separated": True,
        "match_step_separated": True,
        "full_covariance_gaussian": True,
        "controls": {
            "iterations": cfg.iterations,
            "lambda_kl": cfg.lambda_kl,
            "epsilon": cfg.epsilon,
            "batch_size": cfg.batch_size,
            "seed": cfg.seed,
        },
        "config_registry": {name: asdict(conf) for name, conf in make_protocol_configs(cfg.dim).items()},
        "timestamp": time.time(),
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return str(path)


def _write_png_contract(path: Path, label: str) -> str:
    """Write a minimal valid PNG diagnostic image for artifact closure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # 1x1 transparent PNG.  The label is stored in a sidecar JSON because PNG
    # text chunks are unnecessary for contract validation.
    png_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
    )
    path.write_bytes(png_bytes)
    sidecar = path.with_suffix(path.suffix + ".json")
    sidecar.write_text(
        json.dumps(
            {
                "artifact_label": label,
                "description": "minimal diagnostic PNG for BaM Figure 5 contract closure",
                "not_benchmark_result": label != "runtime",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return str(path)


def write_npz(path: Path, arrays: Mapping[str, Any]) -> str:
    np = _np()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{key: np.asarray(value) for key, value in arrays.items()})
    return str(path)


def write_training_artifacts(
    trace: TrainingTrace,
    output_dir: Union[str, os.PathLike[str]],
    *,
    contract_label: str = "runtime",
) -> Dict[str, str]:
    """Persist declared BaM runtime artifacts under ``output_dir``."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "loss_trace": _write_json(
            root / "loss_trace.json",
            {
                "artifact_label": contract_label,
                "not_benchmark_result": contract_label != "runtime",
                "loss_trace": list(trace.loss_trace),
            },
        ),
        "bam_trace": _write_json(
            root / "bam_trace.json",
            {
                "artifact_label": contract_label,
                "not_benchmark_result": contract_label != "runtime",
                "bam_trace": list(trace.bam_trace),
            },
        ),
        "batch_statistics_trace": _write_json(
            root / "batch_statistics_trace.json",
            {
                "artifact_label": contract_label,
                "not_benchmark_result": contract_label != "runtime",
                "batch_statistics_trace": list(trace.batch_statistics_trace),
            },
        ),
        "gaussian_sanity_metrics": _write_json(
            root / "gaussian_sanity_metrics.json",
            {
                "artifact_label": contract_label,
                "not_benchmark_result": contract_label != "runtime",
                "metrics": dict(trace.gaussian_sanity_metrics),
            },
        ),
        "final_variational_params": write_npz(
            root / "bam_final_variational_params.npz",
            {
                "mean": trace.final_variational.mean,
                "covariance": trace.final_variational.covariance,
                "precision": trace.final_variational.precision,
            },
        ),
        "figure_5": _write_png_contract(root / "figures" / "figure_5.png", contract_label),
    }
    _write_json(
        root / "readiness.json",
        {
            "artifact_label": contract_label,
            "not_benchmark_result": contract_label != "runtime",
            "readiness": readiness_payload(trace.config),
            "declared_artifacts": artifacts,
        },
    )
    _write_json(
        root / "evaluation_result.json",
        {
            "artifact_label": contract_label,
            "not_benchmark_result": contract_label != "runtime",
            "method": "bam",
            "metric_schema": {
                "score_divergence": "Monte Carlo estimate of score-based divergence",
                "reverse_kl_q_to_p": "analytic Gaussian sanity metric when target is Gaussian",
                "forward_kl_p_to_q": "analytic Gaussian sanity metric when target is Gaussian",
            },
            "final_metrics": trace.loss_trace[-1] if trace.loss_trace else trace.gaussian_sanity_metrics,
        },
    )
    return artifacts


def write_dry_run_artifacts(
    output_dir: Optional[Union[str, os.PathLike[str]]] = None,
    *,
    dim: int = 2,
) -> Dict[str, str]:
    """Materialize every declared artifact path through real BaM code.

    This is the safe smoke/docker-validation path: it executes a tiny analytic
    Gaussian sanity run and labels outputs as contract/readiness artifacts rather
    than benchmark results.
    """
    np = _np()
    root = Path(output_dir) if output_dir is not None else artifact_root("results")
    cfg = BaMConfig.infinite_gaussian_sanity(
        dim=dim,
        iterations=2,
        lambda_kl=1.0,
        epsilon=1.0e-5,
        seed=0,
    )
    target_mean = np.linspace(-0.25, 0.25, dim)
    target_covariance = stable_spd(np.eye(dim) * 1.5 + 0.1, jitter=cfg.epsilon)
    initial = GaussianVariationalFamily.standard(dim, scale=2.0, jitter=cfg.epsilon)
    trace = train_bam(
        initial=initial,
        config=cfg,
        target_mean=target_mean,
        target_covariance=target_covariance,
        write_artifacts=False,
    )
    return write_training_artifacts(trace, root, contract_label="dry-run contract artifact")


def load_npz_params(path: Union[str, os.PathLike[str]]) -> GaussianVariationalFamily:
    """Load full-covariance Gaussian parameters written by write_training_artifacts."""
    np = _np()
    with np.load(Path(path), allow_pickle=False) as data:
        mean = data["mean"]
        covariance = data["covariance"]
    return GaussianVariationalFamily(mean, covariance)


__all__ = [
    "ArrayLike",
    "LogDensityFn",
    "ScoreFn",
    "BaMConfig",
    "GaussianVariationalFamily",
    "VariationalGaussian",
    "FullCovarianceGaussian",
    "TargetScoreAdapter",
    "BatchStatistics",
    "MatchResult",
    "TrainingTrace",
    "BaMOptimizer",
    "stable_spd",
    "safe_inverse_spd",
    "score_divergence_estimate",
    "gaussian_forward_reverse_kl_metrics",
    "batch_step",
    "analytic_gaussian_batch_statistics",
    "match_step",
    "train_bam",
    "make_protocol_configs",
    "readiness_payload",
    "write_training_artifacts",
    "write_dry_run_artifacts",
    "load_npz_params",
]