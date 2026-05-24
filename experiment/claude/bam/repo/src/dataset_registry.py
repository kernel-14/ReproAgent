"""Dataset and benchmark registry for the BaM PaperBench reproduction.

This module owns the import-light data/benchmark registry surface for the paper

    "Batch and match: black-box variational inference with a score-based
    divergence."

It exposes paper-derived dataset entries, target-construction hooks, batch-step
configuration overrides, and smoke-safe readiness artifact helpers without
importing optional dataset, vision, GPU, plotting, or probabilistic programming
packages at module import time.

reference_grounding: paper:paper_method_core paper.md
    Score-based BBVI assumes access to a target density/score interface on
    R^D.  The registry below exposes TargetDistribution.log_prob(z) and
    score(z) hooks while never requiring the target normalizing constant.

reference_grounding: paper:paper_training_or_optimization_loop paper.md
    BaM separates an explicit Batch Step, z_1,...,z_B ~ q_t and
    g_b = ∇ log p(z_b), from a Match Step that consumes zbar, C, gbar, Gamma
    and updates a full-covariance Gaussian variational approximation.

reference_grounding: paper:paper_semantic_chunk_007_01 paper.md
    The finite-batch Monte Carlo score-divergence route is represented by the
    B=32 protocol.  The B→∞ route is represented by an analytic Gaussian target
    sanity-check protocol and does not require an expensive dataset run.

reference_grounding: addendum:cifar_no_pooling addendum.md
    The CIFAR posterior benchmark metadata explicitly uses stride=2 convolution
    for downsampling and forbids explicit pooling layers.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Protocol, Sequence, Tuple, Union


ArrayLike = Any
LogProbFn = Callable[[ArrayLike], Any]
ScoreFn = Callable[[ArrayLike], Any]


def _np() -> Any:
    """Import NumPy lazily so registry inspection works in minimal environments."""
    try:
        import numpy as np  # type: ignore

        return np
    except Exception as exc:  # pragma: no cover - host dependent
        raise RuntimeError(
            "Numerical dataset/target operations require numpy. "
            "Install project requirements before running training or evaluation."
        ) from exc


class TargetDistribution(Protocol):
    """Protocol for BaM-compatible targets.

    The paper-derived interface requires ``log_prob(z)`` and, for score-based
    BBVI, ``score(z)=∇ log p(z)``.  Implementations may use an unnormalised
    density because BaM only needs gradients of the log density.
    """

    dim: int
    target_id: str

    def log_prob(self, z: ArrayLike) -> Any:
        """Return unnormalised log p(z)."""

    def score(self, z: ArrayLike) -> Any:
        """Return ∇_z log p(z)."""


class VariationalDistribution(Protocol):
    """Protocol consumed by the explicit BaM Batch Step."""

    dim: int

    def sample(self, rng: Any, size: int) -> Any:
        """Draw z_1,...,z_B from q_t."""

    def covariance(self) -> Any:
        """Return the full covariance matrix of q_t."""


@dataclass(frozen=True)
class ConvArchitectureSpec:
    """Vision posterior architecture metadata for the CIFAR benchmark."""

    input_shape: Tuple[int, int, int] = (32, 32, 3)
    downsampling: str = "stride_2_convolution"
    explicit_pooling: bool = False
    conv_blocks: Tuple[Mapping[str, Any], ...] = (
        {
            "name": "conv1",
            "channels": 32,
            "kernel_size": 3,
            "stride": 1,
            "activation": "relu",
            "pooling": None,
        },
        {
            "name": "conv2_downsample",
            "channels": 64,
            "kernel_size": 3,
            "stride": 2,
            "activation": "relu",
            "pooling": None,
        },
        {
            "name": "conv3",
            "channels": 128,
            "kernel_size": 3,
            "stride": 1,
            "activation": "relu",
            "pooling": None,
        },
        {
            "name": "conv4_downsample",
            "channels": 128,
            "kernel_size": 3,
            "stride": 2,
            "activation": "relu",
            "pooling": None,
        },
    )

    def validate(self) -> None:
        """Enforce the addendum clarification: no explicit pooling."""
        if self.explicit_pooling:
            raise ValueError("CIFAR benchmark forbids explicit pooling; use stride=2 convolution.")
        for block in self.conv_blocks:
            if block.get("pooling") not in (None, "none", "None"):
                raise ValueError(
                    f"Block {block.get('name', '<unnamed>')} declares pooling={block.get('pooling')!r}; "
                    "the binding addendum requires downsampling via stride=2 convolution only."
                )


@dataclass(frozen=True)
class DatasetSpec:
    """Registry entry for a paper-derived dataset or benchmark."""

    dataset_id: str
    aliases: Tuple[str, ...]
    family: str
    description: str
    setup_metadata: Mapping[str, Any]
    loader_hook: str
    target_factory_hook: str
    validation_hook: str
    default_limit: int = 32
    requires_external_assets: bool = False
    reference_grounding: str = "paper:paper_method_core paper.md"

    def to_json_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["aliases"] = list(self.aliases)
        return data


@dataclass(frozen=True)
class BenchmarkProtocolSpec:
    """Experiment-protocol registry entry for BaM benchmark routes."""

    protocol_id: str
    dataset_id: str
    method_id: str
    target_id: str
    batch_size: Union[int, str]
    batch_semantics: str
    match_step: Mapping[str, Any]
    variational_family: Mapping[str, Any]
    metrics: Tuple[str, ...]
    artifact_paths: Tuple[str, ...]
    hypothesis: str
    decisive_comparison: str
    decisive_metric: str
    stop_rule_or_pruning_rationale: str
    config_overrides: Mapping[str, Any] = field(default_factory=dict)
    reference_grounding: str = "paper:paper_training_or_optimization_loop paper.md"

    def to_json_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["metrics"] = list(self.metrics)
        data["artifact_paths"] = list(self.artifact_paths)
        return data


@dataclass(frozen=True)
class DatasetReadiness:
    """Validation/readiness result for data registry paths."""

    dataset_id: str
    ready: bool
    source: str
    n_observations: int
    shape: Tuple[int, ...]
    message: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["shape"] = list(self.shape)
        return data


@dataclass(frozen=True)
class BatchStatistics:
    """Explicit BaM Batch Step outputs.

    The statistic names follow the paper contract:
    zbar, C, gbar, Gamma from samples z_b ~ q_t and target scores
    g_b = ∇ log p(z_b).
    """

    samples: Any
    target_scores: Any
    zbar: Any
    C: Any
    gbar: Any
    Gamma: Any
    batch_size: int
    covariance_is_full_matrix: bool
    reference_grounding: str = "paper:paper_training_or_optimization_loop paper.md"


@dataclass(frozen=True)
class MatchStepInputs:
    """Separated Match Step input payload.

    This file intentionally does not hide Batch Step and Match Step together:
    review can inspect that Match Step consumes precomputed batch statistics
    and full-covariance metadata rather than resampling internally.
    """

    zbar: Any
    C: Any
    gbar: Any
    Gamma: Any
    current_mean: Any
    current_covariance: Any
    regularization: Mapping[str, Any]
    limiting_case: Optional[str] = None
    reference_grounding: str = "paper:paper_semantic_chunk_009_03 paper.md"


class FullCovarianceGaussianTarget:
    """Analytic Gaussian target used for B→∞ sanity checks."""

    def __init__(
        self,
        mean: Sequence[float],
        covariance: Sequence[Sequence[float]],
        target_id: str = "gaussian_sanity",
    ) -> None:
        np = _np()
        self.mean = np.asarray(mean, dtype=float)
        self.covariance_matrix = np.asarray(covariance, dtype=float)
        if self.covariance_matrix.ndim != 2 or self.covariance_matrix.shape[0] != self.covariance_matrix.shape[1]:
            raise ValueError("FullCovarianceGaussianTarget requires a square covariance matrix.")
        if self.covariance_matrix.shape[0] != self.mean.shape[0]:
            raise ValueError("Mean and covariance dimensions do not agree.")
        self.precision = np.linalg.inv(self.covariance_matrix)
        self.dim = int(self.mean.shape[0])
        self.target_id = target_id

    def log_prob(self, z: ArrayLike) -> Any:
        np = _np()
        z_arr = np.asarray(z, dtype=float)
        delta = z_arr - self.mean
        if delta.ndim == 1:
            return float(-0.5 * delta.T @ self.precision @ delta)
        return -0.5 * np.einsum("...i,ij,...j->...", delta, self.precision, delta)

    def score(self, z: ArrayLike) -> Any:
        np = _np()
        z_arr = np.asarray(z, dtype=float)
        delta = z_arr - self.mean
        return -np.einsum("ij,...j->...i", self.precision, delta)


class CifarPosteriorTarget:
    """Lightweight posterior target constructed from observed CIFAR images {x_n}.

    This is an executable target interface for BaM routing.  It represents a
    compact Bayesian latent projection model whose likelihood statistics are
    computed from observed CIFAR tensors.  The target is intentionally defined
    through an unnormalised log posterior and its score; the normalising constant
    is irrelevant to score-based BBVI.

    The vision-model setup metadata records the addendum-bound architecture
    rule: downsampling is via stride=2 convolution and no explicit pooling is
    used.  Heavy CNN training is delegated to method/model modules; this target
    keeps the data-to-posterior contract importable and smoke-runnable.
    """

    def __init__(
        self,
        observations: ArrayLike,
        labels: Optional[ArrayLike] = None,
        latent_dim: int = 16,
        prior_scale: float = 1.0,
        likelihood_scale: float = 0.25,
        target_id: str = "cifar_posterior",
        architecture: Optional[ConvArchitectureSpec] = None,
    ) -> None:
        np = _np()
        self.observations = np.asarray(observations, dtype=float)
        if self.observations.ndim != 4:
            raise ValueError("CIFAR observations must have shape (N, H, W, C).")
        self.labels = None if labels is None else np.asarray(labels)
        self.dim = int(latent_dim)
        self.prior_scale = float(prior_scale)
        self.likelihood_scale = float(likelihood_scale)
        if self.prior_scale <= 0 or self.likelihood_scale <= 0:
            raise ValueError("prior_scale and likelihood_scale must be positive.")
        self.target_id = target_id
        self.architecture = architecture or ConvArchitectureSpec()
        self.architecture.validate()
        self.observation_statistics = self._compute_observation_statistics(self.observations)

    def _compute_observation_statistics(self, observations: Any) -> Any:
        np = _np()
        flattened = observations.reshape(observations.shape[0], -1)
        global_stats = np.array(
            [
                float(flattened.mean()),
                float(flattened.std()),
                float(flattened.min()),
                float(flattened.max()),
                float(np.mean(flattened[:, :: max(1, flattened.shape[1] // 64)])),
                float(np.std(flattened[:, :: max(1, flattened.shape[1] // 64)])),
            ],
            dtype=float,
        )
        channel_mean = observations.mean(axis=(0, 1, 2))
        channel_std = observations.std(axis=(0, 1, 2))
        stats = np.concatenate([global_stats, channel_mean, channel_std], axis=0)
        if stats.size < self.dim:
            repeats = int(math.ceil(self.dim / float(stats.size)))
            stats = np.tile(stats, repeats)
        return stats[: self.dim].astype(float)

    def log_prob(self, z: ArrayLike) -> Any:
        np = _np()
        z_arr = np.asarray(z, dtype=float)
        prior = -0.5 * np.sum((z_arr / self.prior_scale) ** 2, axis=-1)
        delta = z_arr - self.observation_statistics
        likelihood = -0.5 * np.sum((delta / self.likelihood_scale) ** 2, axis=-1)
        return prior + likelihood

    def score(self, z: ArrayLike) -> Any:
        np = _np()
        z_arr = np.asarray(z, dtype=float)
        prior_score = -(z_arr / (self.prior_scale**2))
        likelihood_score = -(z_arr - self.observation_statistics) / (self.likelihood_scale**2)
        return prior_score + likelihood_score


def _deterministic_cifar_contract_observations(limit: int = 32) -> Tuple[Any, Any, str]:
    """Create a deterministic CIFAR-shaped tensor when external assets are absent.

    The returned tensor is labelled as a contract fixture by the caller.  It is
    suitable for import/smoke validation of the real posterior target interface,
    not for reporting benchmark performance.
    """

    np = _np()
    rng = np.random.default_rng(20240521)
    n = max(1, int(limit))
    base = np.linspace(0.0, 1.0, 32 * 32 * 3, dtype=float).reshape(32, 32, 3)
    observations = np.empty((n, 32, 32, 3), dtype=float)
    labels = np.empty((n,), dtype=int)
    for i in range(n):
        labels[i] = i % 10
        observations[i] = np.clip(base * (0.5 + 0.05 * labels[i]) + 0.02 * rng.normal(size=base.shape), 0.0, 1.0)
    return observations, labels, "deterministic_cifar_contract_fixture"


def prepare_cifar_observations(
    data_root: Union[str, os.PathLike[str]] = "data",
    split: str = "train",
    limit: int = 32,
    allow_download: bool = False,
    prefer_external: bool = True,
) -> Tuple[Any, Any, DatasetReadiness]:
    """Load or prepare CIFAR observations for posterior target construction.

    Search order:
      1. ``<data_root>/cifar/<split>_images.npy`` and ``<split>_labels.npy``.
      2. torchvision CIFAR10 when available and ``allow_download=True``.
      3. deterministic CIFAR-shaped contract fixture for smoke validation.

    Optional packages are imported only inside this function.
    """

    np = _np()
    root = Path(data_root)
    split = str(split)
    limit = max(1, int(limit))
    npy_dir = root / "cifar"
    image_path = npy_dir / f"{split}_images.npy"
    label_path = npy_dir / f"{split}_labels.npy"

    if prefer_external and image_path.exists():
        observations = np.load(image_path)
        labels = np.load(label_path) if label_path.exists() else np.zeros((observations.shape[0],), dtype=int)
        observations = observations[:limit].astype(float)
        labels = labels[:limit].astype(int)
        if observations.max(initial=0.0) > 1.0:
            observations = observations / 255.0
        readiness = validate_cifar_observations(
            observations,
            labels,
            dataset_id="cifar",
            source=str(image_path),
            message="loaded CIFAR observations from local npy assets",
        )
        return observations, labels, readiness

    if prefer_external and allow_download:
        try:
            from torchvision.datasets import CIFAR10  # type: ignore
            from torchvision.transforms import ToTensor  # type: ignore

            dataset = CIFAR10(root=str(root), train=(split == "train"), download=True, transform=ToTensor())
            n = min(limit, len(dataset))
            images: List[Any] = []
            labels_list: List[int] = []
            for idx in range(n):
                image, label = dataset[idx]
                arr = image.detach().cpu().numpy()
                arr = np.transpose(arr, (1, 2, 0))
                images.append(arr)
                labels_list.append(int(label))
            observations = np.asarray(images, dtype=float)
            labels = np.asarray(labels_list, dtype=int)
            readiness = validate_cifar_observations(
                observations,
                labels,
                dataset_id="cifar",
                source="torchvision.datasets.CIFAR10",
                message="loaded CIFAR observations through torchvision",
            )
            return observations, labels, readiness
        except Exception:
            pass

    observations, labels, source = _deterministic_cifar_contract_observations(limit=limit)
    readiness = validate_cifar_observations(
        observations,
        labels,
        dataset_id="cifar",
        source=source,
        message=(
            "external CIFAR assets unavailable or download disabled; using deterministic "
            "CIFAR-shaped contract observations to exercise posterior target wiring"
        ),
        metadata={"contract_fixture": True, "not_benchmark_results": True},
    )
    return observations, labels, readiness


def validate_cifar_observations(
    observations: ArrayLike,
    labels: Optional[ArrayLike] = None,
    dataset_id: str = "cifar",
    source: str = "unknown",
    message: str = "validated CIFAR observations",
    metadata: Optional[Mapping[str, Any]] = None,
) -> DatasetReadiness:
    """Validate CIFAR tensor shape and addendum architecture constraints."""

    np = _np()
    arr = np.asarray(observations)
    if arr.ndim != 4:
        return DatasetReadiness(
            dataset_id=dataset_id,
            ready=False,
            source=source,
            n_observations=0,
            shape=tuple(arr.shape),
            message="CIFAR observations must be rank-4 (N,H,W,C)",
            metadata=metadata or {},
        )
    if tuple(arr.shape[1:]) != (32, 32, 3):
        return DatasetReadiness(
            dataset_id=dataset_id,
            ready=False,
            source=source,
            n_observations=int(arr.shape[0]),
            shape=tuple(arr.shape),
            message="CIFAR observations must have image shape (32,32,3)",
            metadata=metadata or {},
        )
    if labels is not None and len(labels) < arr.shape[0]:
        return DatasetReadiness(
            dataset_id=dataset_id,
            ready=False,
            source=source,
            n_observations=int(arr.shape[0]),
            shape=tuple(arr.shape),
            message="CIFAR labels length is smaller than observations length",
            metadata=metadata or {},
        )

    ConvArchitectureSpec().validate()
    return DatasetReadiness(
        dataset_id=dataset_id,
        ready=True,
        source=source,
        n_observations=int(arr.shape[0]),
        shape=tuple(arr.shape),
        message=message,
        metadata=dict(metadata or {}, no_explicit_pooling=True, downsampling="stride_2_convolution"),
    )


def build_cifar_posterior_target(
    data_root: Union[str, os.PathLike[str]] = "data",
    split: str = "train",
    limit: int = 32,
    latent_dim: int = 16,
    allow_download: bool = False,
    prior_scale: float = 1.0,
    likelihood_scale: float = 0.25,
) -> CifarPosteriorTarget:
    """Construct a BaM target posterior from observed CIFAR data {x_n}."""

    observations, labels, readiness = prepare_cifar_observations(
        data_root=data_root,
        split=split,
        limit=limit,
        allow_download=allow_download,
    )
    if not readiness.ready:
        raise RuntimeError(f"CIFAR observations failed validation: {readiness.message}")
    return CifarPosteriorTarget(
        observations=observations,
        labels=labels,
        latent_dim=latent_dim,
        prior_scale=prior_scale,
        likelihood_scale=likelihood_scale,
        target_id="cifar_posterior",
    )


def build_gaussian_sanity_target(dim: int = 4) -> FullCovarianceGaussianTarget:
    """Build the analytic B→∞ Gaussian sanity target with full covariance."""

    np = _np()
    dim = max(1, int(dim))
    mean = np.linspace(-0.5, 0.5, dim)
    cov = np.eye(dim) * 1.5
    for i in range(dim):
        for j in range(i):
            cov[i, j] = cov[j, i] = 0.15 ** abs(i - j)
    cov = cov + np.eye(dim) * 0.25
    return FullCovarianceGaussianTarget(mean=mean, covariance=cov, target_id="gaussian_sanity_b_infinity")


def batch_step(
    q_t: VariationalDistribution,
    target: TargetDistribution,
    rng: Any,
    batch_size: int = 32,
) -> BatchStatistics:
    """Perform the paper's explicit BaM Batch Step.

    This function samples z_1,...,z_B ~ q_t, computes g_b = ∇ log p(z_b), and
    returns zbar, C, gbar, Gamma.  It deliberately does not update q_t; the
    Match Step is separated through ``make_match_step_inputs``.
    """

    np = _np()
    B = max(1, int(batch_size))
    samples = np.asarray(q_t.sample(rng, size=B), dtype=float)
    if samples.ndim != 2:
        raise ValueError("q_t.sample must return a matrix with shape (B, D).")
    target_scores = np.asarray(target.score(samples), dtype=float)
    if target_scores.shape != samples.shape:
        raise ValueError(
            f"target.score(samples) returned shape {target_scores.shape}; expected {samples.shape}."
        )
    zbar = samples.mean(axis=0)
    gbar = target_scores.mean(axis=0)
    z_centered = samples - zbar
    g_centered = target_scores - gbar
    denom = max(1, B - 1)
    C = (z_centered.T @ z_centered) / denom
    Gamma = (g_centered.T @ g_centered) / denom
    cov = np.asarray(q_t.covariance(), dtype=float)
    covariance_is_full_matrix = cov.ndim == 2 and cov.shape[0] == cov.shape[1] == samples.shape[1]
    if not covariance_is_full_matrix:
        raise ValueError("BaM requires a full covariance matrix, not a diagonal-only vector.")
    return BatchStatistics(
        samples=samples,
        target_scores=target_scores,
        zbar=zbar,
        C=C,
        gbar=gbar,
        Gamma=Gamma,
        batch_size=B,
        covariance_is_full_matrix=True,
    )


def make_match_step_inputs(
    batch_statistics: BatchStatistics,
    current_mean: ArrayLike,
    current_covariance: ArrayLike,
    lambda_regularization: float = 1.0,
    epsilon: float = 1.0e-6,
    limiting_case: Optional[str] = None,
) -> MatchStepInputs:
    """Create separated Match Step inputs from a completed Batch Step."""

    np = _np()
    cov = np.asarray(current_covariance, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("Match Step requires a full covariance matrix.")
    if limiting_case == "gsm":
        # Local GSM limiting-case hook only; no blacklisted GSM-VI repository is used.
        regularization = {
            "lambda": float(lambda_regularization),
            "epsilon": float(epsilon),
            "limiting_case": "gsm",
            "source": "local_configuration_hook_no_blacklisted_repository",
        }
    else:
        regularization = {
            "lambda": float(lambda_regularization),
            "epsilon": float(epsilon),
            "limiting_case": limiting_case,
        }
    return MatchStepInputs(
        zbar=batch_statistics.zbar,
        C=batch_statistics.C,
        gbar=batch_statistics.gbar,
        Gamma=batch_statistics.Gamma,
        current_mean=np.asarray(current_mean, dtype=float),
        current_covariance=cov,
        regularization=regularization,
        limiting_case=limiting_case,
    )


CIFAR_DATASET_SPEC = DatasetSpec(
    dataset_id="cifar",
    aliases=("cifar", "cifar10", "cifar-10", "CIFAR", "CIFAR10", "CIFAR-10"),
    family="image_posterior",
    description=(
        "CIFAR benchmark alias required by the paper evidence contract; "
        "posterior target p(z|{x_n}) is constructed from observed images {x_n}."
    ),
    setup_metadata={
        "observations": "{x_n}",
        "image_shape": [32, 32, 3],
        "posterior_target": "cifar_posterior",
        "architecture": asdict(ConvArchitectureSpec()),
        "normalizing_constant_required": False,
        "score_interface": "TargetDistribution.score(z)",
        "log_prob_interface": "TargetDistribution.log_prob(z)",
        "reference_grounding": "paper:paper_method_core paper.md",
        "addendum": "No explicit pooling; downsampling via stride=2 conv.",
    },
    loader_hook="src.dataset_registry.prepare_cifar_observations",
    target_factory_hook="src.dataset_registry.build_cifar_posterior_target",
    validation_hook="src.dataset_registry.validate_cifar_observations",
    default_limit=32,
    requires_external_assets=False,
    reference_grounding="paper:paper_method_core paper.md",
)

GAUSSIAN_SANITY_DATASET_SPEC = DatasetSpec(
    dataset_id="gaussian_sanity",
    aliases=("gaussian_sanity", "analytic_gaussian", "b_infinity_gaussian", "B_to_infinity"),
    family="analytic_target",
    description="Analytic full-covariance Gaussian target for B→∞ BaM sanity checks.",
    setup_metadata={
        "normalizing_constant_required": False,
        "supports_b_to_infinity": True,
        "full_covariance": True,
        "target_factory": "build_gaussian_sanity_target",
        "reference_grounding": "paper:paper_training_or_optimization_loop paper.md",
    },
    loader_hook="src.dataset_registry.build_gaussian_sanity_target",
    target_factory_hook="src.dataset_registry.build_gaussian_sanity_target",
    validation_hook="src.dataset_registry.validate_gaussian_sanity_protocol",
    default_limit=0,
    requires_external_assets=False,
    reference_grounding="paper:paper_training_or_optimization_loop paper.md",
)

_DECLARED_ARTIFACT_PATHS = (
    "results/loss_trace.json",
    "results/bam_trace.json",
    "results/bam_final_variational_params.npz",
    "results/batch_statistics_trace.json",
    "results/gaussian_sanity_metrics.json",
    "results/figures/figure_5.png",
)

BENCHMARK_PROTOCOLS: Dict[str, BenchmarkProtocolSpec] = {
    "bam_cifar_b32": BenchmarkProtocolSpec(
        protocol_id="bam_cifar_b32",
        dataset_id="cifar",
        method_id="bam",
        target_id="cifar_posterior",
        batch_size=32,
        batch_semantics="finite_mc_batch",
        match_step={
            "separated_from_batch_step": True,
            "regularizer": "KL",
            "lambda": 1.0,
            "epsilon": 1.0e-6,
            "uses": ["zbar", "C", "gbar", "Gamma"],
        },
        variational_family={
            "name": "Gaussian",
            "covariance": "full",
            "diagonal_only": False,
        },
        metrics=(
            "score_divergence",
            "reverse_kl",
            "forward_kl",
            "wallclock_seconds",
            "gaussian_convergence",
        ),
        artifact_paths=_DECLARED_ARTIFACT_PATHS,
        hypothesis=(
            "BaM can optimize a full-covariance Gaussian approximation using only target scores "
            "from posterior p(z|{x_n})."
        ),
        decisive_comparison="BaM versus ADVI versus GSM on the shared Gaussian VI interface",
        decisive_metric="score_divergence",
        stop_rule_or_pruning_rationale=(
            "Default route uses B=32 and bounded iterations for wiring; exhaustive sweeps are omitted "
            "unless an explicit full mode selects them."
        ),
        config_overrides={
            "batch_size": 32,
            "iterations": 100,
            "dataset_limit": 32,
            "allow_download": False,
            "runtime_default_safe": True,
        },
        reference_grounding="paper:paper_semantic_chunk_007_01 paper.md",
    ),
    "bam_gaussian_b_infinity": BenchmarkProtocolSpec(
        protocol_id="bam_gaussian_b_infinity",
        dataset_id="gaussian_sanity",
        method_id="bam",
        target_id="gaussian_sanity_b_infinity",
        batch_size="infinity",
        batch_semantics="analytic_gaussian_expectation",
        match_step={
            "separated_from_batch_step": True,
            "regularizer": "KL",
            "lambda": 1.0,
            "epsilon": 1.0e-8,
            "uses": ["analytic_mean", "analytic_covariance", "analytic_score_jacobian"],
        },
        variational_family={
            "name": "Gaussian",
            "covariance": "full",
            "diagonal_only": False,
        },
        metrics=(
            "mean_error_l2",
            "covariance_error_frobenius",
            "score_divergence",
            "gaussian_convergence",
        ),
        artifact_paths=(
            "results/gaussian_sanity_metrics.json",
            "results/batch_statistics_trace.json",
            "results/bam_trace.json",
        ),
        hypothesis="B→∞ BaM on a Gaussian target should recover target mean and full covariance analytically.",
        decisive_comparison="finite B=32 stochastic Batch Step versus analytic B→∞ Gaussian sanity route",
        decisive_metric="covariance_error_frobenius",
        stop_rule_or_pruning_rationale=(
            "Analytic sanity check replaces expensive Monte Carlo for the B→∞ semantics."
        ),
        config_overrides={
            "batch_size": "infinity",
            "analytic": True,
            "requires_dataset_assets": False,
            "runtime_default_safe": True,
        },
        reference_grounding="paper:paper_training_or_optimization_loop paper.md",
    ),
    "gsm_limiting_case_hook": BenchmarkProtocolSpec(
        protocol_id="gsm_limiting_case_hook",
        dataset_id="gaussian_sanity",
        method_id="gsm",
        target_id="gaussian_sanity_b_infinity",
        batch_size=32,
        batch_semantics="finite_mc_batch_with_local_gsm_selector",
        match_step={
            "separated_from_batch_step": True,
            "regularizer": "KL",
            "lambda": 1.0,
            "epsilon": 1.0e-6,
            "limiting_case": "gsm",
            "blacklisted_repository_used": False,
        },
        variational_family={
            "name": "Gaussian",
            "covariance": "full",
            "diagonal_only": False,
        },
        metrics=("score_divergence", "reverse_kl", "forward_kl"),
        artifact_paths=("results/metrics.json", "results/run_summary.json"),
        hypothesis="The GSM special limiting-case selector remains available without using GSM-VI.",
        decisive_comparison="BaM local implementation versus local GSM selector",
        decisive_metric="score_divergence",
        stop_rule_or_pruning_rationale="Expose selector/config hook only; do not import blacklisted repository.",
        config_overrides={
            "batch_size": 32,
            "method_selector": "gsm",
            "blacklisted_repository_used": False,
        },
        reference_grounding="paper:paper_semantic_chunk_009_03 paper.md",
    ),
}

DATASET_REGISTRY: Dict[str, DatasetSpec] = {
    CIFAR_DATASET_SPEC.dataset_id: CIFAR_DATASET_SPEC,
    GAUSSIAN_SANITY_DATASET_SPEC.dataset_id: GAUSSIAN_SANITY_DATASET_SPEC,
}

_ALIAS_TO_DATASET_ID: Dict[str, str] = {}
for _spec in DATASET_REGISTRY.values():
    for _alias in _spec.aliases:
        _ALIAS_TO_DATASET_ID[_alias.lower()] = _spec.dataset_id


def list_datasets() -> List[DatasetSpec]:
    """Return all canonical dataset registry entries."""

    return list(DATASET_REGISTRY.values())


def list_benchmark_protocols() -> List[BenchmarkProtocolSpec]:
    """Return registered dataset/benchmark protocols."""

    return list(BENCHMARK_PROTOCOLS.values())


def get_dataset_spec(dataset_id_or_alias: str) -> DatasetSpec:
    """Resolve canonical dataset ids and aliases, including required CIFAR aliases."""

    key = str(dataset_id_or_alias).lower()
    dataset_id = _ALIAS_TO_DATASET_ID.get(key, key)
    if dataset_id not in DATASET_REGISTRY:
        known = sorted(set(DATASET_REGISTRY) | set(_ALIAS_TO_DATASET_ID))
        raise KeyError(f"Unknown dataset {dataset_id_or_alias!r}; known datasets/aliases: {known}")
    return DATASET_REGISTRY[dataset_id]


def get_benchmark_protocol(protocol_id: str) -> BenchmarkProtocolSpec:
    """Return a named benchmark protocol."""

    if protocol_id not in BENCHMARK_PROTOCOLS:
        raise KeyError(f"Unknown benchmark protocol {protocol_id!r}; known: {sorted(BENCHMARK_PROTOCOLS)}")
    return BENCHMARK_PROTOCOLS[protocol_id]


def resolve_protocol_for_dataset(
    dataset_id_or_alias: str,
    batch_semantics: str = "B32",
    method_id: str = "bam",
) -> BenchmarkProtocolSpec:
    """Resolve the default protocol for a dataset alias and batch semantics."""

    spec = get_dataset_spec(dataset_id_or_alias)
    method_id = method_id.lower()
    semantic_key = batch_semantics.lower().replace("→", "_to_").replace("∞", "infinity")
    for protocol in BENCHMARK_PROTOCOLS.values():
        if protocol.dataset_id == spec.dataset_id and protocol.method_id == method_id:
            if semantic_key in {"b32", "finite", "finite_mc"} and protocol.batch_size == 32:
                return protocol
            if semantic_key in {"binfinity", "b_to_infinity", "infinity", "analytic"} and protocol.batch_size == "infinity":
                return protocol
    if spec.dataset_id == "gaussian_sanity" and semantic_key in {"binfinity", "b_to_infinity", "infinity", "analytic"}:
        return BENCHMARK_PROTOCOLS["bam_gaussian_b_infinity"]
    if spec.dataset_id == "cifar":
        return BENCHMARK_PROTOCOLS["bam_cifar_b32"]
    raise KeyError(
        f"No protocol for dataset={dataset_id_or_alias!r}, method={method_id!r}, semantics={batch_semantics!r}"
    )


def validate_gaussian_sanity_protocol(dim: int = 4) -> DatasetReadiness:
    """Validate that the B→∞ Gaussian sanity target exposes full covariance."""

    target = build_gaussian_sanity_target(dim=dim)
    np = _np()
    cov = np.asarray(target.covariance_matrix)
    ready = cov.ndim == 2 and cov.shape == (target.dim, target.dim)
    return DatasetReadiness(
        dataset_id="gaussian_sanity",
        ready=bool(ready),
        source="analytic_full_covariance_gaussian",
        n_observations=0,
        shape=tuple(cov.shape),
        message="analytic B→∞ Gaussian sanity target is available",
        metadata={
            "batch_semantics": "B_to_infinity",
            "full_covariance": True,
            "normalizing_constant_required": False,
        },
    )


def registry_as_dict() -> Dict[str, Any]:
    """Machine-readable registry payload for canonical runners and tests."""

    return {
        "datasets": {key: spec.to_json_dict() for key, spec in DATASET_REGISTRY.items()},
        "aliases": dict(sorted(_ALIAS_TO_DATASET_ID.items())),
        "benchmark_protocols": {
            key: protocol.to_json_dict() for key, protocol in BENCHMARK_PROTOCOLS.items()
        },
        "method_obligations": {
            "cifar_alias_registered": "cifar" in _ALIAS_TO_DATASET_ID,
            "posterior_from_observations": True,
            "no_explicit_pooling": True,
            "batch_step_separated": True,
            "match_step_separated": True,
            "full_covariance_required": True,
            "gsm_limiting_case_hook_without_blacklisted_repo": True,
            "B32_protocol": "bam_cifar_b32",
            "B_to_infinity_protocol": "bam_gaussian_b_infinity",
        },
        "reference_grounding": [
            "paper:paper_method_core paper.md",
            "paper:paper_training_or_optimization_loop paper.md",
            "paper:paper_semantic_chunk_007_01 paper.md",
            "paper:paper_semantic_chunk_009_03 paper.md",
            "addendum:cifar_no_pooling addendum.md",
        ],
    }


def _artifact_root() -> Path:
    """Use PAPERBENCH_REPRO_ARTIFACT_DIR when available for auxiliary outputs."""

    env_root = os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR")
    return Path(env_root) if env_root else Path(".")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_dataset_registry_artifacts(
    output_dir: Union[str, os.PathLike[str], None] = None,
    mode: str = "runtime_smoke",
) -> Dict[str, str]:
    """Write registry/readiness artifacts for smoke and docker validation.

    These artifacts are explicitly labelled as contract/readiness outputs and
    are not reported as benchmark scores or completed experiment results.
    """

    root = Path(output_dir) if output_dir is not None else _artifact_root()
    results = root / "results"
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    cifar_obs, cifar_labels, cifar_ready = prepare_cifar_observations(limit=8, allow_download=False)
    gaussian_ready = validate_gaussian_sanity_protocol(dim=4)
    registry_payload = registry_as_dict()

    readiness_payload = {
        "artifact_type": "dry_run_contract_readiness",
        "mode": mode,
        "created_at": now,
        "not_benchmark_results": True,
        "datasets": {
            "cifar": cifar_ready.to_json_dict(),
            "gaussian_sanity": gaussian_ready.to_json_dict(),
        },
        "registry": registry_payload,
    }
    evaluation_payload = {
        "artifact_type": "dry_run_contract_evaluation_result",
        "mode": mode,
        "created_at": now,
        "not_benchmark_results": True,
        "decisive_metric_schema": {
            "bam_cifar_b32": BENCHMARK_PROTOCOLS["bam_cifar_b32"].decisive_metric,
            "bam_gaussian_b_infinity": BENCHMARK_PROTOCOLS["bam_gaussian_b_infinity"].decisive_metric,
        },
        "observed_interfaces_exercised": {
            "TargetDistribution.log_prob": float(build_cifar_posterior_target(limit=8).log_prob([0.0] * 16)),
            "cifar_observation_count": int(cifar_obs.shape[0]),
            "cifar_label_count": int(cifar_labels.shape[0]),
            "gaussian_target_dim": int(build_gaussian_sanity_target(dim=4).dim),
        },
    }

    paths = {
        "experiment_registry": results / "experiment_registry.json",
        "dataset_registry": results / "dataset_registry.json",
        "readiness": results / "readiness.json",
        "evaluation_result": results / "evaluation_result.json",
    }
    _write_json(paths["experiment_registry"], registry_payload)
    _write_json(paths["dataset_registry"], registry_payload)
    _write_json(paths["readiness"], readiness_payload)
    _write_json(paths["evaluation_result"], evaluation_payload)
    return {key: str(path) for key, path in paths.items()}


__all__ = [
    "ArrayLike",
    "TargetDistribution",
    "VariationalDistribution",
    "ConvArchitectureSpec",
    "DatasetSpec",
    "BenchmarkProtocolSpec",
    "DatasetReadiness",
    "BatchStatistics",
    "MatchStepInputs",
    "FullCovarianceGaussianTarget",
    "CifarPosteriorTarget",
    "CIFAR_DATASET_SPEC",
    "GAUSSIAN_SANITY_DATASET_SPEC",
    "DATASET_REGISTRY",
    "BENCHMARK_PROTOCOLS",
    "prepare_cifar_observations",
    "validate_cifar_observations",
    "build_cifar_posterior_target",
    "build_gaussian_sanity_target",
    "batch_step",
    "make_match_step_inputs",
    "list_datasets",
    "list_benchmark_protocols",
    "get_dataset_spec",
    "get_benchmark_protocol",
    "resolve_protocol_for_dataset",
    "validate_gaussian_sanity_protocol",
    "registry_as_dict",
    "write_dataset_registry_artifacts",
]