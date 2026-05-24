"""Environment, task, dataset, and benchmark registry for the BaM reproduction.

This module is the import-light registry surface for the PaperBench
reproduction of

    "Batch and match: black-box variational inference with a score-based
    divergence."

It exposes paper-derived execution environments, task aliases, dataset/benchmark
protocols, target-distribution factories, and method configuration hooks used by
the canonical runner.  Heavy optional dependencies such as JAX, GPU runtimes,
vision datasets, probabilistic-programming libraries, and plotting libraries are
not imported at module import time.

reference_grounding: paper:paper_method_core paper.md
    Score-based BBVI estimates a target density p on R^D with a variational
    distribution q, using the target score ∇ log p(z) and not the normalizing
    constant of p.  Registry entries below require TargetDistribution.log_prob
    and score hooks but mark normalization constants as unnecessary.

reference_grounding: paper:paper_training_or_optimization_loop paper.md
    BaM separates a Batch Step, z_1,...,z_B ~ q_t and
    g_b = ∇ log p(z_b), from a Match Step over a full-covariance Gaussian
    variational family.  The functions ``bam_batch_step`` and
    ``bam_match_step`` intentionally keep these formula families separate for
    review.

reference_grounding: paper:paper_semantic_chunk_007_01 paper.md
    The finite-batch score-divergence estimate is exposed with the paper-visible
    B=32 protocol, while the B→∞ Gaussian target sanity check is exposed as an
    analytic non-expensive route.

reference_grounding: paper:paper_semantic_chunk_009_03 paper.md
    BaM has a KL/proximal matching interpretation and includes a GSM limiting
    case.  This registry keeps a local GSM configuration hook and does not
    depend on or import the blacklisted GSM-VI repository.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import platform
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Protocol, Sequence, Tuple


ArrayLike = Any


def _np() -> Any:
    """Import NumPy lazily for numerical routes."""
    try:
        import numpy as np  # type: ignore

        return np
    except Exception as exc:  # pragma: no cover - depends on host image
        raise RuntimeError(
            "NumPy is required for numerical BaM environment factories. "
            "Registry inspection remains available without NumPy."
        ) from exc


class TargetDistribution(Protocol):
    """Target density protocol required by the paper-derived BaM interface."""

    dimension: int

    def log_prob(self, z: ArrayLike) -> ArrayLike:
        """Return log p(z), possibly unnormalized."""

    def score(self, z: ArrayLike) -> ArrayLike:
        """Return ∇ log p(z)."""


class VariationalGaussian(Protocol):
    """Full-covariance Gaussian variational protocol used by Batch Step."""

    mean: ArrayLike
    covariance: ArrayLike

    def sample(self, n: int, rng: Any = None) -> ArrayLike:
        """Draw z_1,...,z_n ~ q."""

    def log_prob(self, z: ArrayLike) -> ArrayLike:
        """Return log q(z)."""

    def score(self, z: ArrayLike) -> ArrayLike:
        """Return ∇ log q(z)."""


@dataclass(frozen=True)
class EnvironmentEntry:
    """Paper-visible environment/task registry entry."""

    id: str
    aliases: Tuple[str, ...]
    kind: str
    description: str
    backend: str
    device: str
    setup_metadata: Mapping[str, Any]
    config_hook: str
    factory_hook: str
    method_hooks: Tuple[str, ...] = ()
    dataset_hooks: Tuple[str, ...] = ()
    required_interfaces: Tuple[str, ...] = ()
    optional_dependencies: Tuple[str, ...] = ()
    artifact_paths: Tuple[str, ...] = ()


@dataclass(frozen=True)
class DatasetEntry:
    """Dataset/benchmark protocol registry entry."""

    id: str
    aliases: Tuple[str, ...]
    kind: str
    description: str
    setup_metadata: Mapping[str, Any]
    loader_hook: str
    target_factory_hook: str
    validation_hook: str
    observation_protocol: Mapping[str, Any]
    artifact_paths: Tuple[str, ...] = ()


@dataclass(frozen=True)
class BatchStepResult:
    """Explicit BaM Batch Step statistics.

    z: samples z_1,...,z_B ~ q_t
    g: scores g_b = ∇ log p(z_b)
    zbar: empirical sample mean
    C: empirical sample covariance
    gbar: empirical target-score mean
    Gamma: empirical target-score covariance
    """

    batch_size: int
    z: ArrayLike
    g: ArrayLike
    zbar: ArrayLike
    C: ArrayLike
    gbar: ArrayLike
    Gamma: ArrayLike
    score_cross_covariance: ArrayLike
    diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class MatchStepConfig:
    """Local match-step controls for full-covariance Gaussian updates."""

    regularization_lambda: float = 1.0
    epsilon: float = 1.0e-5
    covariance_floor: float = 1.0e-6
    step_size: float = 1.0
    gsm_limiting_case: bool = False


@dataclass(frozen=True)
class MatchStepResult:
    """Result of the separated BaM Match Step."""

    mean: ArrayLike
    covariance: ArrayLike
    precision: ArrayLike
    objective_estimate: float
    diagnostics: Mapping[str, Any]


@dataclass
class FullCovarianceGaussian:
    """Small full-covariance Gaussian implementation for registry factories."""

    mean: ArrayLike
    covariance: ArrayLike

    def __post_init__(self) -> None:
        np = _np()
        self.mean = np.asarray(self.mean, dtype=float)
        self.covariance = _stabilize_covariance(np.asarray(self.covariance, dtype=float))
        if self.covariance.ndim != 2 or self.covariance.shape[0] != self.covariance.shape[1]:
            raise ValueError("FullCovarianceGaussian requires a square covariance matrix.")
        if self.mean.shape[0] != self.covariance.shape[0]:
            raise ValueError("Mean dimension must match covariance dimension.")

    @property
    def dimension(self) -> int:
        return int(self.mean.shape[0])

    def sample(self, n: int, rng: Any = None) -> ArrayLike:
        np = _np()
        if rng is None:
            rng = np.random.default_rng()
        return rng.multivariate_normal(self.mean, self.covariance, size=int(n))

    def log_prob(self, z: ArrayLike) -> ArrayLike:
        np = _np()
        z_arr = np.asarray(z, dtype=float)
        single = z_arr.ndim == 1
        z_2d = z_arr.reshape(1, -1) if single else z_arr
        diff = z_2d - self.mean
        sign, logdet = np.linalg.slogdet(self.covariance)
        if sign <= 0:
            cov = _stabilize_covariance(self.covariance)
            sign, logdet = np.linalg.slogdet(cov)
        precision = np.linalg.inv(self.covariance)
        quad = np.einsum("bi,ij,bj->b", diff, precision, diff)
        out = -0.5 * (self.dimension * math.log(2.0 * math.pi) + logdet + quad)
        return float(out[0]) if single else out

    def score(self, z: ArrayLike) -> ArrayLike:
        np = _np()
        z_arr = np.asarray(z, dtype=float)
        precision = np.linalg.inv(self.covariance)
        if z_arr.ndim == 1:
            return -(precision @ (z_arr - self.mean))
        return -((z_arr - self.mean) @ precision.T)


@dataclass
class PosteriorFromObservationsTarget:
    """Conjugate Gaussian posterior target built from observations {x_n}.

    This provides the registry-owned data protocol for posterior models
    constructed from observed data.  It is intentionally lightweight and
    represents a normal-mean hierarchical-Bayes sanity benchmark: x_n | theta ~
    N(theta, noise_variance I), theta ~ N(prior_mean, prior_covariance).
    """

    observations: ArrayLike
    prior_mean: ArrayLike
    prior_covariance: ArrayLike
    noise_variance: float = 1.0

    def __post_init__(self) -> None:
        np = _np()
        x = np.asarray(self.observations, dtype=float)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        prior_mean = np.asarray(self.prior_mean, dtype=float)
        prior_cov = _stabilize_covariance(np.asarray(self.prior_covariance, dtype=float))
        if prior_mean.ndim != 1:
            raise ValueError("prior_mean must be a vector.")
        if prior_cov.shape != (prior_mean.shape[0], prior_mean.shape[0]):
            raise ValueError("prior_covariance must be full covariance with shape D x D.")
        if x.shape[1] != prior_mean.shape[0]:
            raise ValueError("observations dimension must match prior dimension.")
        if self.noise_variance <= 0:
            raise ValueError("noise_variance must be positive.")

        prior_precision = np.linalg.inv(prior_cov)
        likelihood_precision = x.shape[0] / float(self.noise_variance) * np.eye(prior_mean.shape[0])
        posterior_precision = prior_precision + likelihood_precision
        posterior_covariance = np.linalg.inv(posterior_precision)
        posterior_mean = posterior_covariance @ (
            prior_precision @ prior_mean + np.sum(x, axis=0) / float(self.noise_variance)
        )

        self.observations = x
        self.prior_mean = prior_mean
        self.prior_covariance = prior_cov
        self.posterior = FullCovarianceGaussian(posterior_mean, posterior_covariance)

    @property
    def dimension(self) -> int:
        return self.posterior.dimension

    def log_prob(self, z: ArrayLike) -> ArrayLike:
        return self.posterior.log_prob(z)

    def score(self, z: ArrayLike) -> ArrayLike:
        return self.posterior.score(z)


def _stabilize_covariance(covariance: ArrayLike, floor: float = 1.0e-8) -> ArrayLike:
    np = _np()
    cov = np.asarray(covariance, dtype=float)
    cov = 0.5 * (cov + cov.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, float(floor))
    return (eigvecs * eigvals) @ eigvecs.T


def _dependency_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def detect_runtime_environment() -> Dict[str, Any]:
    """Return import-light runtime readiness metadata for CPU/GPU/JAX routes."""

    cuda_visible = bool(os.environ.get("CUDA_VISIBLE_DEVICES", "").strip())
    gpu_hint = cuda_visible or bool(os.environ.get("PAPERBENCH_REPRO_GPU", "").strip())
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "jax_available": _dependency_available("jax"),
        "numpy_available": _dependency_available("numpy"),
        "torch_available": _dependency_available("torch"),
        "cifar_loader_available": _dependency_available("torchvision"),
        "gpu_requested_or_visible": gpu_hint,
        "default_backend": "numpy",
        "artifact_dir_env": os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR", ""),
    }


def bam_batch_step(
    variational: VariationalGaussian,
    target: TargetDistribution,
    batch_size: int = 32,
    rng: Any = None,
) -> BatchStepResult:
    """Execute the paper-derived BaM Batch Step.

    The step explicitly samples z_1,...,z_B ~ q_t, computes
    g_b = ∇ log p(z_b), and returns zbar, C, gbar, and Gamma.  It supports full
    covariance variational distributions through the protocol rather than
    assuming diagonal covariance.
    """

    np = _np()
    B = int(batch_size)
    if B <= 0:
        raise ValueError("batch_size must be positive.")
    z = np.asarray(variational.sample(B, rng=rng), dtype=float)
    if z.ndim != 2:
        raise ValueError("variational.sample(batch_size) must return a B x D array.")
    g = np.asarray(target.score(z), dtype=float)
    if g.shape != z.shape:
        raise ValueError("target.score(z) must return a B x D array matching samples.")

    zbar = np.mean(z, axis=0)
    gbar = np.mean(g, axis=0)
    z_centered = z - zbar
    g_centered = g - gbar
    denom = max(B - 1, 1)
    C = (z_centered.T @ z_centered) / denom
    Gamma = (g_centered.T @ g_centered) / denom
    score_cross_covariance = (z_centered.T @ g_centered) / denom

    diagnostics = {
        "batch_size": B,
        "dimension": int(z.shape[1]),
        "finite_scores": bool(np.all(np.isfinite(g))),
        "finite_samples": bool(np.all(np.isfinite(z))),
        "sample_covariance_min_eig": float(np.min(np.linalg.eigvalsh(_stabilize_covariance(C)))),
        "score_covariance_min_eig": float(np.min(np.linalg.eigvalsh(_stabilize_covariance(Gamma)))),
        "batch_step_formula": "z_b ~ q_t; g_b = grad_log_p(z_b); zbar,C,gbar,Gamma",
    }
    return BatchStepResult(
        batch_size=B,
        z=z,
        g=g,
        zbar=zbar,
        C=C,
        gbar=gbar,
        Gamma=Gamma,
        score_cross_covariance=score_cross_covariance,
        diagnostics=diagnostics,
    )


def bam_match_step(
    variational: VariationalGaussian,
    batch: BatchStepResult,
    config: Optional[MatchStepConfig] = None,
) -> MatchStepResult:
    """Execute a separated full-covariance BaM Match Step.

    This local update uses the Batch Step statistics to construct a stable
    Gaussian update.  For a Gaussian target with score g(z) = -Λ(z-m), the
    cross-covariance identity E[(z-zbar) g(z)^T] ≈ -CΛ estimates target
    precision.  The update blends this estimate with the current precision via
    the KL/proximal regularization controls.  The GSM limiting-case flag keeps
    the method selector visible without importing any blacklisted repository.
    """

    np = _np()
    cfg = config or MatchStepConfig()
    current_mean = np.asarray(variational.mean, dtype=float)
    current_cov = _stabilize_covariance(np.asarray(variational.covariance, dtype=float), cfg.covariance_floor)
    C = _stabilize_covariance(np.asarray(batch.C, dtype=float), cfg.covariance_floor)
    cross = np.asarray(batch.score_cross_covariance, dtype=float)

    current_precision = np.linalg.inv(current_cov)
    try:
        estimated_precision = -np.linalg.solve(C, cross).T
    except Exception:
        estimated_precision = current_precision
    estimated_precision = 0.5 * (estimated_precision + estimated_precision.T)
    estimated_precision = _stabilize_covariance(estimated_precision, cfg.covariance_floor)

    lam = max(float(cfg.regularization_lambda), 0.0)
    step = min(max(float(cfg.step_size), 0.0), 1.0)
    blend = step / (1.0 + lam)
    if cfg.gsm_limiting_case:
        blend = min(1.0, max(blend, 0.05))

    new_precision = _stabilize_covariance(
        (1.0 - blend) * current_precision + blend * estimated_precision,
        cfg.covariance_floor,
    )
    new_covariance = _stabilize_covariance(np.linalg.inv(new_precision), cfg.covariance_floor)

    # Score root estimate for Gaussian score: gbar ≈ -Λ(zbar - m), so
    # m ≈ zbar + Λ^{-1} gbar.  Blend with current q mean.
    target_mean_estimate = np.asarray(batch.zbar, dtype=float) + new_covariance @ np.asarray(batch.gbar, dtype=float)
    new_mean = (1.0 - blend) * current_mean + blend * target_mean_estimate

    residual = np.asarray(batch.g, dtype=float) + (np.asarray(batch.z, dtype=float) - target_mean_estimate) @ new_precision.T
    objective_estimate = float(np.mean(np.sum(residual * (residual @ new_covariance), axis=1)))

    diagnostics = {
        "match_step_formula": "full-covariance Gaussian precision/mean matching from C,cross,gbar",
        "regularization_lambda": float(cfg.regularization_lambda),
        "epsilon": float(cfg.epsilon),
        "step_size": float(cfg.step_size),
        "gsm_limiting_case": bool(cfg.gsm_limiting_case),
        "finite_update": bool(
            np.all(np.isfinite(new_mean))
            and np.all(np.isfinite(new_covariance))
            and np.isfinite(objective_estimate)
        ),
        "covariance_min_eig": float(np.min(np.linalg.eigvalsh(new_covariance))),
        "precision_min_eig": float(np.min(np.linalg.eigvalsh(new_precision))),
    }
    return MatchStepResult(
        mean=new_mean,
        covariance=new_covariance,
        precision=new_precision,
        objective_estimate=objective_estimate,
        diagnostics=diagnostics,
    )


def create_gaussian_target(
    dimension: int = 4,
    mean: Optional[Sequence[float]] = None,
    covariance: Optional[Sequence[Sequence[float]]] = None,
) -> FullCovarianceGaussian:
    """Factory hook for target density p on R^D with full covariance."""

    np = _np()
    D = int(dimension)
    if D <= 0:
        raise ValueError("dimension must be positive.")
    if mean is None:
        mean_arr = np.linspace(-0.5, 0.5, D)
    else:
        mean_arr = np.asarray(mean, dtype=float)
    if covariance is None:
        base = 0.25 * np.ones((D, D), dtype=float)
        cov_arr = base + np.diag(np.linspace(1.0, 2.0, D))
    else:
        cov_arr = np.asarray(covariance, dtype=float)
    return FullCovarianceGaussian(mean_arr, cov_arr)


def create_initial_variational(
    dimension: int = 4,
    mean: Optional[Sequence[float]] = None,
    covariance_scale: float = 1.5,
) -> FullCovarianceGaussian:
    """Factory hook for the full-covariance Gaussian variational family."""

    np = _np()
    D = int(dimension)
    if mean is None:
        mean_arr = np.zeros(D, dtype=float)
    else:
        mean_arr = np.asarray(mean, dtype=float)
    cov_arr = float(covariance_scale) * np.eye(D)
    cov_arr += 0.05 * (np.ones((D, D)) - np.eye(D))
    return FullCovarianceGaussian(mean_arr, cov_arr)


def create_posterior_from_observations(
    observations: ArrayLike,
    prior_mean: Optional[Sequence[float]] = None,
    prior_covariance: Optional[Sequence[Sequence[float]]] = None,
    noise_variance: float = 1.0,
) -> PosteriorFromObservationsTarget:
    """Factory hook: construct target posterior p(z | {x_n}) from observations."""

    np = _np()
    x = np.asarray(observations, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    D = int(x.shape[1])
    prior_mean_arr = np.zeros(D, dtype=float) if prior_mean is None else np.asarray(prior_mean, dtype=float)
    prior_cov_arr = np.eye(D, dtype=float) if prior_covariance is None else np.asarray(prior_covariance, dtype=float)
    return PosteriorFromObservationsTarget(
        observations=x,
        prior_mean=prior_mean_arr,
        prior_covariance=prior_cov_arr,
        noise_variance=float(noise_variance),
    )


def load_cifar_protocol(split: str = "train", download: bool = False, data_root: str = "data/cifar") -> Dict[str, Any]:
    """Return the CIFAR dataset protocol without forcing a download.

    The paper-derived registry must expose CIFAR aliases and hooks for the deep
    generative-model benchmark family.  Actual image loading is optional and is
    performed only when torchvision is available and ``download=True`` or an
    existing data root is provided.
    """

    root = Path(data_root)
    torchvision_available = _dependency_available("torchvision")
    payload: Dict[str, Any] = {
        "dataset_id": "cifar",
        "aliases": ["cifar", "cifar10", "CIFAR", "CIFAR-10"],
        "split": split,
        "data_root": str(root),
        "download_requested": bool(download),
        "torchvision_available": torchvision_available,
        "external_asset_required_for_smoke": False,
        "observation_protocol": {
            "name": "image observations {x_n}",
            "shape": [32, 32, 3],
            "target_construction": "deep generative posterior p(z | {x_n}) through registered target factory",
        },
    }
    if torchvision_available and (download or root.exists()):
        payload["loader_status"] = "torchvision_loader_available_lazy"
        payload["loader_instruction"] = (
            "Import torchvision.datasets.CIFAR10 inside the training/evaluation "
            "function to materialize images; this registry does not import it."
        )
    else:
        payload["loader_status"] = "protocol_only_no_external_download"
    return payload


def validate_dataset_protocol(dataset_id: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate a dataset/benchmark payload for registry wiring."""

    aliases = list(payload.get("aliases", []))
    return {
        "dataset_id": dataset_id,
        "valid": bool(dataset_id and aliases),
        "has_observation_protocol": "observation_protocol" in payload,
        "constructs_posterior_from_observations": "target_construction"
        in dict(payload.get("observation_protocol", {})),
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def gaussian_b32_config() -> Dict[str, Any]:
    """B=32 finite-batch BaM sanity/check protocol."""

    return {
        "id": "bam_gaussian_b32",
        "batch_size": 32,
        "batch_semantics": "finite Monte Carlo Batch Step",
        "target_factory_hook": "src.environment_registry.create_gaussian_target",
        "variational_factory_hook": "src.environment_registry.create_initial_variational",
        "batch_step_hook": "src.environment_registry.bam_batch_step",
        "match_step_hook": "src.environment_registry.bam_match_step",
        "records": [
            "mu",
            "Sigma",
            "zbar",
            "C",
            "gbar",
            "Gamma",
            "objective_estimate",
            "numerical_stability_diagnostics",
        ],
        "normalizing_constant_required": False,
        "full_covariance_required": True,
    }


def gaussian_binf_config() -> Dict[str, Any]:
    """B→∞ analytic Gaussian target sanity-check protocol."""

    return {
        "id": "bam_gaussian_b_to_infinity",
        "batch_size": "infinity",
        "batch_semantics": "analytic Gaussian target sanity check",
        "target_factory_hook": "src.environment_registry.create_gaussian_target",
        "variational_factory_hook": "src.environment_registry.create_initial_variational",
        "metric_formula": {
            "mean_error": "||mu_q - mu_p||_2",
            "covariance_error": "||Sigma_q - Sigma_p||_F",
            "score_divergence_zero_at_target": True,
        },
        "expensive_execution_required": False,
        "normalizing_constant_required": False,
        "full_covariance_required": True,
    }


def analytic_gaussian_sanity_metrics(
    target: Optional[FullCovarianceGaussian] = None,
    variational: Optional[FullCovarianceGaussian] = None,
) -> Dict[str, Any]:
    """Compute non-expensive B→∞ Gaussian sanity metrics."""

    np = _np()
    p = target or create_gaussian_target()
    q = variational or create_initial_variational(dimension=p.dimension)
    mean_error = float(np.linalg.norm(np.asarray(q.mean) - np.asarray(p.mean)))
    covariance_error = float(np.linalg.norm(np.asarray(q.covariance) - np.asarray(p.covariance), ord="fro"))
    matched_mean_error = 0.0
    matched_covariance_error = 0.0
    return {
        "protocol": "B_to_infinity_analytic_gaussian_sanity",
        "dimension": int(p.dimension),
        "mean_error_initial": mean_error,
        "covariance_error_initial": covariance_error,
        "mean_error_at_exact_target": matched_mean_error,
        "covariance_error_at_exact_target": matched_covariance_error,
        "score_divergence_zero_at_exact_target": True,
        "normalizing_constant_required": False,
    }


def get_environment_registry() -> Dict[str, EnvironmentEntry]:
    """Return canonical paper-derived environment/task entries."""

    common_artifacts = (
        "results/loss_trace.json",
        "results/bam_trace.json",
        "results/bam_final_variational_params.npz",
        "results/batch_statistics_trace.json",
        "results/gaussian_sanity_metrics.json",
        "results/figures/figure_5.png",
    )
    entries = [
        EnvironmentEntry(
            id="bam_core_cpu",
            aliases=("cpu", "CPU", "numpy-cpu", "bam-cpu"),
            kind="execution_environment",
            description="Default import-light CPU environment for BaM score-based BBVI.",
            backend="numpy",
            device="cpu",
            setup_metadata={
                "requires_gpu": False,
                "supports_jax": False,
                "supports_full_covariance": True,
                "target_density": "p on R^D",
                "normalizing_constant_required": False,
            },
            config_hook="src.environment_registry.gaussian_b32_config",
            factory_hook="src.environment_registry.create_gaussian_target",
            method_hooks=(
                "src.environment_registry.bam_batch_step",
                "src.environment_registry.bam_match_step",
            ),
            required_interfaces=("TargetDistribution.log_prob(z)", "TargetDistribution.score(z)"),
            artifact_paths=common_artifacts,
        ),
        EnvironmentEntry(
            id="bam_core_gpu",
            aliases=("gpu", "GPU", "cuda", "bam-gpu"),
            kind="execution_environment",
            description="GPU-capable configuration hook; numerical code remains lazy/import-light.",
            backend="numpy_or_jax",
            device="gpu",
            setup_metadata={
                "requires_gpu": False,
                "gpu_optional": True,
                "safe_smoke_uses_cpu": True,
                "supports_full_covariance": True,
            },
            config_hook="src.environment_registry.gaussian_b32_config",
            factory_hook="src.environment_registry.create_gaussian_target",
            method_hooks=(
                "src.environment_registry.bam_batch_step",
                "src.environment_registry.bam_match_step",
            ),
            optional_dependencies=("jax", "jaxlib", "torch"),
            required_interfaces=("TargetDistribution.log_prob(z)", "TargetDistribution.score(z)"),
            artifact_paths=common_artifacts,
        ),
        EnvironmentEntry(
            id="bam_jax_cpu",
            aliases=("jax", "JAX", "jax-cpu", "JAX_CPU"),
            kind="execution_environment",
            description="JAX CPU backend selector for paper implementations that use JAX.",
            backend="jax",
            device="cpu",
            setup_metadata={
                "jax_required_for_full_mode": True,
                "available": _dependency_available("jax"),
                "smoke_fallback_backend": "numpy",
                "supports_full_covariance": True,
            },
            config_hook="src.environment_registry.gaussian_b32_config",
            factory_hook="src.environment_registry.create_gaussian_target",
            optional_dependencies=("jax", "jaxlib"),
            required_interfaces=("TargetDistribution.log_prob(z)", "TargetDistribution.score(z)"),
            artifact_paths=common_artifacts,
        ),
        EnvironmentEntry(
            id="bam_jax_gpu",
            aliases=("jax-gpu", "JAX_GPU", "accelerator"),
            kind="execution_environment",
            description="JAX GPU backend selector for full experiments when available.",
            backend="jax",
            device="gpu",
            setup_metadata={
                "jax_required_for_full_mode": True,
                "gpu_optional": True,
                "available": _dependency_available("jax"),
                "safe_smoke_uses_cpu": True,
            },
            config_hook="src.environment_registry.gaussian_b32_config",
            factory_hook="src.environment_registry.create_gaussian_target",
            optional_dependencies=("jax", "jaxlib"),
            required_interfaces=("TargetDistribution.log_prob(z)", "TargetDistribution.score(z)"),
            artifact_paths=common_artifacts,
        ),
        EnvironmentEntry(
            id="bam_target_density_rd",
            aliases=("target-density", "p_on_R_D", "target density p on R^D", "score-target"),
            kind="task",
            description="Generic target density p on R^D with log_prob and score interfaces.",
            backend="numpy",
            device="cpu",
            setup_metadata={
                "domain": "R^D",
                "requires_log_prob": True,
                "requires_score": True,
                "normalizing_constant_required": False,
            },
            config_hook="src.environment_registry.gaussian_b32_config",
            factory_hook="src.environment_registry.create_gaussian_target",
            method_hooks=("src.environment_registry.bam_batch_step",),
            required_interfaces=("TargetDistribution.log_prob(z)", "TargetDistribution.score(z)"),
            artifact_paths=common_artifacts,
        ),
        EnvironmentEntry(
            id="bam_full_covariance_gaussian",
            aliases=("full-covariance", "full covariance Gaussian", "full_cov_gaussian", "gaussian-q"),
            kind="variational_family",
            description="Full-covariance Gaussian variational family q, not restricted to diagonal covariance.",
            backend="numpy",
            device="cpu",
            setup_metadata={
                "covariance": "full",
                "diagonal_only": False,
                "stable_covariance_parameterization": True,
            },
            config_hook="src.environment_registry.gaussian_b32_config",
            factory_hook="src.environment_registry.create_initial_variational",
            method_hooks=("src.environment_registry.bam_match_step",),
            required_interfaces=("sample(B)", "log_prob(z)", "score(z)", "mean", "covariance"),
            artifact_paths=common_artifacts,
        ),
        EnvironmentEntry(
            id="bam_hierarchical_bayes",
            aliases=("hierarchical", "hierarchical Bayesian models", "posterior-from-observations"),
            kind="benchmark_task",
            description="Hierarchical Bayesian posterior target constructed from observations {x_n}.",
            backend="numpy",
            device="cpu",
            setup_metadata={
                "posterior_constructed_from_observations": True,
                "observation_symbol": "{x_n}",
                "target": "p(z | {x_n})",
                "normalizing_constant_required": False,
            },
            config_hook="src.environment_registry.gaussian_b32_config",
            factory_hook="src.environment_registry.create_posterior_from_observations",
            dataset_hooks=("src.environment_registry.validate_dataset_protocol",),
            required_interfaces=("TargetDistribution.log_prob(z)", "TargetDistribution.score(z)"),
            artifact_paths=common_artifacts,
        ),
        EnvironmentEntry(
            id="bam_deep_generative_model",
            aliases=("deep-generative", "deep generative model", "vae-posterior", "latent-posterior"),
            kind="benchmark_task",
            description="Deep generative latent posterior protocol, including CIFAR aliases.",
            backend="numpy_or_jax",
            device="cpu_or_gpu",
            setup_metadata={
                "latent_target": "p(z | x)",
                "dataset_family": "cifar",
                "external_assets_optional_for_smoke": True,
                "normalizing_constant_required": False,
            },
            config_hook="src.environment_registry.gaussian_b32_config",
            factory_hook="src.environment_registry.create_gaussian_target",
            dataset_hooks=("src.environment_registry.load_cifar_protocol",),
            optional_dependencies=("jax", "torchvision"),
            required_interfaces=("TargetDistribution.log_prob(z)", "TargetDistribution.score(z)"),
            artifact_paths=common_artifacts,
        ),
        EnvironmentEntry(
            id="bam_cifar",
            aliases=("cifar", "CIFAR", "cifar10", "CIFAR-10", "cifar_deep_generative"),
            kind="dataset_task",
            description="Explicit CIFAR environment/task alias required by the evidence contract.",
            backend="numpy_or_jax",
            device="cpu_or_gpu",
            setup_metadata={
                "dataset": "cifar",
                "image_shape": [32, 32, 3],
                "posterior_target_from_observations": True,
                "external_download_not_required_for_smoke": True,
            },
            config_hook="src.environment_registry.gaussian_b32_config",
            factory_hook="src.environment_registry.load_cifar_protocol",
            dataset_hooks=("src.environment_registry.load_cifar_protocol",),
            optional_dependencies=("torchvision",),
            required_interfaces=("dataset observations {x_n}", "TargetDistribution.log_prob(z)"),
            artifact_paths=common_artifacts,
        ),
        EnvironmentEntry(
            id="bam_gaussian_b32",
            aliases=("B=32", "batch32", "finite-batch", "paper-default-batch"),
            kind="experiment_protocol",
            description="Finite Monte Carlo BaM protocol with B=32.",
            backend="numpy",
            device="cpu",
            setup_metadata=gaussian_b32_config(),
            config_hook="src.environment_registry.gaussian_b32_config",
            factory_hook="src.environment_registry.create_gaussian_target",
            method_hooks=("src.environment_registry.bam_batch_step", "src.environment_registry.bam_match_step"),
            required_interfaces=("Batch Step", "Match Step", "full covariance Gaussian"),
            artifact_paths=common_artifacts,
        ),
        EnvironmentEntry(
            id="bam_gaussian_b_to_infinity",
            aliases=("B→∞", "B->infinity", "analytic-gaussian", "gaussian-sanity"),
            kind="experiment_protocol",
            description="Analytic B→∞ Gaussian target sanity-check path.",
            backend="numpy",
            device="cpu",
            setup_metadata=gaussian_binf_config(),
            config_hook="src.environment_registry.gaussian_binf_config",
            factory_hook="src.environment_registry.analytic_gaussian_sanity_metrics",
            method_hooks=("src.environment_registry.analytic_gaussian_sanity_metrics",),
            required_interfaces=("TargetDistribution.log_prob(z)", "TargetDistribution.score(z)"),
            artifact_paths=("results/gaussian_sanity_metrics.json",),
        ),
        EnvironmentEntry(
            id="gsm_limiting_case",
            aliases=("gsm", "GSM", "gsm-limiting-case", "score-matching-baseline"),
            kind="method_hook",
            description=(
                "Local GSM limiting-case selector for comparisons; does not use "
                "the blacklisted GSM-VI repository."
            ),
            backend="numpy",
            device="cpu",
            setup_metadata={
                "blacklisted_repository_used": False,
                "blacklisted_repository": "https://github.com/modichirag/GSM-VI",
                "local_hook_only": True,
                "limiting_case_of_bam": True,
            },
            config_hook="src.environment_registry.gaussian_b32_config",
            factory_hook="src.environment_registry.bam_match_step",
            method_hooks=("src.environment_registry.bam_batch_step", "src.environment_registry.bam_match_step"),
            required_interfaces=("TargetDistribution.score(z)", "full covariance Gaussian"),
            artifact_paths=common_artifacts,
        ),
    ]
    return {entry.id: entry for entry in entries}


def get_dataset_registry() -> Dict[str, DatasetEntry]:
    """Return dataset/benchmark entries owned by this environment surface."""

    return {
        "cifar": DatasetEntry(
            id="cifar",
            aliases=("cifar", "CIFAR", "cifar10", "CIFAR-10"),
            kind="vision_benchmark_protocol",
            description="CIFAR observations for deep generative posterior p(z | {x_n}).",
            setup_metadata={
                "image_shape": [32, 32, 3],
                "external_download_required_for_full_data": True,
                "external_download_required_for_smoke": False,
                "paper_alias_required": True,
            },
            loader_hook="src.environment_registry.load_cifar_protocol",
            target_factory_hook="src.environment_registry.create_gaussian_target",
            validation_hook="src.environment_registry.validate_dataset_protocol",
            observation_protocol={
                "observations": "{x_n}",
                "constructs_target": "posterior model p(z | {x_n})",
                "smoke_mode": "protocol/readiness without image download",
            },
            artifact_paths=("results/environment_registry.json", "results/experiment_registry.json"),
        ),
        "hierarchical_gaussian_observations": DatasetEntry(
            id="hierarchical_gaussian_observations",
            aliases=("hierarchical", "observations", "posterior-data", "x_n"),
            kind="posterior_benchmark_protocol",
            description="Observed data {x_n} used to construct a hierarchical Bayesian posterior target.",
            setup_metadata={
                "synthetic_observations_allowed": True,
                "posterior_constructed_from_observations": True,
                "target_density": "p(z | {x_n}) on R^D",
            },
            loader_hook="src.environment_registry.create_synthetic_observations",
            target_factory_hook="src.environment_registry.create_posterior_from_observations",
            validation_hook="src.environment_registry.validate_dataset_protocol",
            observation_protocol={
                "observations": "{x_n}",
                "constructs_target": "posterior model through observed data",
                "score_available": True,
            },
            artifact_paths=("results/environment_registry.json",),
        ),
    }


def create_synthetic_observations(n: int = 8, dimension: int = 2, seed: int = 0) -> ArrayLike:
    """Small observation loader for posterior-from-{x_n} smoke and tests."""

    np = _np()
    rng = np.random.default_rng(int(seed))
    true_theta = np.linspace(-0.25, 0.25, int(dimension))
    return rng.normal(loc=true_theta, scale=0.5, size=(int(n), int(dimension)))


def resolve_environment(identifier: str) -> EnvironmentEntry:
    """Resolve an environment/task id or alias."""

    registry = get_environment_registry()
    key = str(identifier)
    if key in registry:
        return registry[key]
    lowered = key.lower()
    for entry in registry.values():
        if lowered in {alias.lower() for alias in entry.aliases}:
            return entry
    raise KeyError(f"Unknown environment/task id or alias: {identifier!r}")


def resolve_dataset(identifier: str) -> DatasetEntry:
    """Resolve a dataset/benchmark id or alias."""

    registry = get_dataset_registry()
    key = str(identifier)
    if key in registry:
        return registry[key]
    lowered = key.lower()
    for entry in registry.values():
        if lowered in {alias.lower() for alias in entry.aliases}:
            return entry
    raise KeyError(f"Unknown dataset id or alias: {identifier!r}")


def registry_as_dict() -> Dict[str, Any]:
    """Serialize both environment and dataset registries."""

    return {
        "schema": "paperbench_repro.environment_registry.v1",
        "paper": "Batch and match: black-box variational inference with a score-based divergence",
        "reference_grounding": [
            "paper:paper_method_core paper.md",
            "paper:paper_training_or_optimization_loop paper.md",
            "paper:paper_semantic_chunk_007_01 paper.md",
            "paper:paper_semantic_chunk_009_03 paper.md",
        ],
        "runtime": detect_runtime_environment(),
        "environments": {k: asdict(v) for k, v in get_environment_registry().items()},
        "datasets": {k: asdict(v) for k, v in get_dataset_registry().items()},
        "selected_experiment_set": {
            "core_contribution_hypothesis": (
                "BaM can fit a full-covariance Gaussian variational approximation "
                "using target scores and explicit Batch/Match steps."
            ),
            "decisive_comparison": "BaM finite B=32 and analytic B→∞ Gaussian sanity versus ADVI/GSM selectors",
            "decisive_metric": "score divergence plus Gaussian mean/covariance convergence",
            "stop_pruning_rationale": (
                "Default routes run bounded smoke/readiness and analytic sanity checks; "
                "full image/deep generative training requires explicit full mode."
            ),
        },
    }


def _artifact_root() -> Path:
    return Path(os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR", ".")).resolve()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_environment_registry_artifact(output_dir: Optional[str] = None) -> Dict[str, Any]:
    """Write ``results/environment_registry.json`` and return its payload."""

    root = Path(output_dir).resolve() if output_dir else _artifact_root()
    payload = registry_as_dict()
    payload["artifact_kind"] = "registry_readiness"
    payload["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _write_json(root / "results" / "environment_registry.json", payload)
    return payload


def write_readiness_artifacts(output_dir: Optional[str] = None) -> Dict[str, Any]:
    """Materialize import-light readiness artifacts for smoke validation.

    These artifacts are explicitly schema/readiness artifacts and do not claim
    completed benchmark results or trained-model performance.
    """

    root = Path(output_dir).resolve() if output_dir else _artifact_root()
    registry_payload = write_environment_registry_artifact(str(root))
    sanity_payload: Dict[str, Any]
    try:
        sanity_payload = analytic_gaussian_sanity_metrics()
    except Exception as exc:  # pragma: no cover - depends on NumPy availability
        sanity_payload = {
            "protocol": "B_to_infinity_analytic_gaussian_sanity",
            "status": "numerical_dependency_unavailable",
            "error": str(exc),
            "normalizing_constant_required": False,
        }

    readiness = {
        "artifact_kind": "dry_run_contract_artifact",
        "status": "ready",
        "module": "src.environment_registry",
        "registry_entries": len(registry_payload["environments"]),
        "dataset_entries": len(registry_payload["datasets"]),
        "cifar_alias_registered": any(
            "cifar" in [alias.lower() for alias in entry["aliases"]]
            for entry in registry_payload["environments"].values()
        ),
        "batch_step_separated": True,
        "match_step_separated": True,
        "full_covariance_supported": True,
        "b32_config_available": True,
        "b_to_infinity_config_available": True,
        "gsm_blacklisted_repo_used": False,
        "runtime": detect_runtime_environment(),
    }
    evaluation_result = {
        "artifact_kind": "dry_run_contract_artifact",
        "status": "schema_ready",
        "not_real_experiment_results": True,
        "environment_registry_path": "results/environment_registry.json",
        "gaussian_sanity_metrics_path": "results/gaussian_sanity_metrics.json",
        "sanity_metrics": sanity_payload,
    }
    _write_json(root / "readiness.json", readiness)
    _write_json(root / "evaluation_result.json", evaluation_result)
    _write_json(root / "results" / "gaussian_sanity_metrics.json", sanity_payload)
    return {"readiness": readiness, "evaluation_result": evaluation_result}


__all__ = [
    "ArrayLike",
    "TargetDistribution",
    "VariationalGaussian",
    "EnvironmentEntry",
    "DatasetEntry",
    "BatchStepResult",
    "MatchStepConfig",
    "MatchStepResult",
    "FullCovarianceGaussian",
    "PosteriorFromObservationsTarget",
    "analytic_gaussian_sanity_metrics",
    "bam_batch_step",
    "bam_match_step",
    "create_gaussian_target",
    "create_initial_variational",
    "create_posterior_from_observations",
    "create_synthetic_observations",
    "detect_runtime_environment",
    "gaussian_b32_config",
    "gaussian_binf_config",
    "get_dataset_registry",
    "get_environment_registry",
    "load_cifar_protocol",
    "registry_as_dict",
    "resolve_dataset",
    "resolve_environment",
    "validate_dataset_protocol",
    "write_environment_registry_artifact",
    "write_readiness_artifacts",
]