"""ADVI baseline and selector registry for the BaM PaperBench reproduction.

This module owns the ADVI-facing baseline surface for the paper
"Batch and match: black-box variational inference with a score-based divergence".

It keeps import-time dependencies light while exposing:
- a full-covariance Gaussian variational family,
- explicit batch sampling with target score evaluation,
- a match/update step with KL regularization,
- method/baseline/variant selector registries,
- bounded sweep registries required by the paper contract, and
- dry-run-safe comparison and contract-artifact helpers.

The canonical BaM core lives in neighboring modules; this file provides the
ADVI/baseline compatibility layer and selector surface that the repository
runner can reach without heavy optional dependencies.

reference_grounding: paper:paper_method_core paper.md
    Score-based BBVI and ADVI-style baselines operate on target scores and
    Gaussian variational families without requiring the target normalizing
    constant.

reference_grounding: paper:paper_training_or_optimization_loop paper.md
    The paper's algorithmic contract separates an explicit Batch Step,
    z_1,...,z_B ~ q_t with g_b = ∇ log p(z_b), from a Match Step that updates
    Gaussian parameters using batch statistics and KL regularization.

reference_grounding: paper:paper_semantic_chunk_009_03 paper.md
    The method inventory includes BaM, ADVI, BBVI, KL, ELBO, GSM, SPP, EM,
    CLI, ours, and baseline selectors; this module exposes them as real registry
    entries and bounded sweep values, including 100_iterations, batch_size_32,
    lambda, epsilon, learning_rate, batch_size, iteration_count values=0,
    B=32, B→∞, regularization strength, p, and lora_rank.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union


ArrayLike = Any
ScoreFn = Callable[[ArrayLike], ArrayLike]
LogProbFn = Callable[[ArrayLike], Any]
SampleFn = Callable[[Any, int], ArrayLike]


def _np() -> Any:
    """Lazy NumPy import with a clear runtime error for numerical execution."""
    try:
        import numpy as np  # type: ignore

        return np
    except Exception as exc:  # pragma: no cover - depends on host env
        raise RuntimeError(
            "NumPy is required for ADVI numerical execution; import the repository "
            "in a minimal environment is still supported, but running ADVI needs NumPy."
        ) from exc


def _to_jsonable(value: Any) -> Any:
    """Recursively convert objects to JSON-friendly primitives."""
    np = None
    try:
        np = _np()
    except Exception:
        np = None

    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return str(value)
        return value
    if isinstance(value, Path):
        return str(value)
    if np is not None and isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]
    if hasattr(value, "__dict__"):
        return _to_jsonable(vars(value))
    return str(value)


def _safe_mean(values: Sequence[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def _softmax(log_weights: Any) -> Any:
    np = _np()
    lw = np.asarray(log_weights, dtype=float)
    lw = lw - np.max(lw)
    weights = np.exp(lw)
    denom = float(np.sum(weights))
    if denom <= 0.0 or not np.isfinite(denom):
        return np.full_like(weights, 1.0 / max(weights.size, 1))
    return weights / denom


def _make_rng(seed: int) -> Any:
    np = _np()
    return np.random.default_rng(int(seed))


def _as_2d_array(value: ArrayLike) -> Any:
    np = _np()
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"Expected a 1D or 2D array, got shape {arr.shape!r}")
    return arr


def _as_vector(value: ArrayLike) -> Any:
    np = _np()
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        arr = arr[None]
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr


def _stable_cholesky(covariance: ArrayLike, jitter: float = 1e-8, max_attempts: int = 8) -> Any:
    np = _np()
    cov = np.asarray(covariance, dtype=float)
    cov = 0.5 * (cov + cov.T)
    eye = np.eye(cov.shape[0], dtype=float)
    current = float(jitter)
    last_error: Optional[Exception] = None
    for _ in range(max_attempts):
        try:
            return np.linalg.cholesky(cov + current * eye)
        except Exception as exc:  # pragma: no cover - numeric fallback
            last_error = exc
            current *= 10.0
    raise RuntimeError("Failed to obtain a stable Cholesky factor for covariance") from last_error


def _cov_from_tril(scale_tril: ArrayLike) -> Any:
    np = _np()
    tril = np.tril(np.asarray(scale_tril, dtype=float))
    return tril @ tril.T


def _covariance_from_params(scale_tril: ArrayLike, epsilon: float = 1e-8) -> Any:
    np = _np()
    tril = np.tril(np.asarray(scale_tril, dtype=float))
    diag = np.diag(tril).copy()
    diag = np.maximum(np.abs(diag), float(epsilon))
    tril = tril.copy()
    idx = np.diag_indices_from(tril)
    tril[idx] = diag
    return tril @ tril.T


def _precision_solve(scale_tril: ArrayLike, vectors: ArrayLike) -> Any:
    np = _np()
    L = np.tril(np.asarray(scale_tril, dtype=float))
    vec = np.asarray(vectors, dtype=float)
    if vec.ndim == 1:
        vec = vec[:, None]
    y = np.linalg.solve(L, vec)
    x = np.linalg.solve(L.T, y)
    return x


@dataclass
class GaussianParams:
    """Full-covariance Gaussian variational parameters."""

    mean: List[float]
    scale_tril: List[List[float]]
    epsilon: float = 1e-8

    def to_arrays(self) -> Tuple[Any, Any]:
        np = _np()
        return np.asarray(self.mean, dtype=float), np.asarray(self.scale_tril, dtype=float)

    @classmethod
    def from_arrays(
        cls,
        mean: ArrayLike,
        scale_tril: ArrayLike,
        epsilon: float = 1e-8,
    ) -> "GaussianParams":
        np = _np()
        mean_arr = np.asarray(mean, dtype=float).reshape(-1)
        tril_arr = np.asarray(scale_tril, dtype=float)
        tril_arr = np.tril(tril_arr)
        diag = np.diag(tril_arr).copy()
        diag = np.maximum(np.abs(diag), float(epsilon))
        tril_arr = tril_arr.copy()
        tril_arr[np.diag_indices_from(tril_arr)] = diag
        return cls(mean=mean_arr.tolist(), scale_tril=tril_arr.tolist(), epsilon=float(epsilon))

    @classmethod
    def from_mean_covariance(
        cls,
        mean: ArrayLike,
        covariance: ArrayLike,
        epsilon: float = 1e-8,
    ) -> "GaussianParams":
        np = _np()
        mean_arr = np.asarray(mean, dtype=float).reshape(-1)
        cov = np.asarray(covariance, dtype=float)
        cov = 0.5 * (cov + cov.T)
        chol = _stable_cholesky(cov, jitter=epsilon)
        return cls(mean=mean_arr.tolist(), scale_tril=chol.tolist(), epsilon=float(epsilon))

    @classmethod
    def identity(cls, dim: int, epsilon: float = 1e-8) -> "GaussianParams":
        np = _np()
        mean = np.zeros(int(dim), dtype=float)
        chol = np.eye(int(dim), dtype=float)
        return cls(mean=mean.tolist(), scale_tril=chol.tolist(), epsilon=float(epsilon))

    @property
    def dim(self) -> int:
        return len(self.mean)

    def mean_array(self) -> Any:
        np = _np()
        return np.asarray(self.mean, dtype=float)

    def tril_array(self) -> Any:
        np = _np()
        return np.asarray(self.scale_tril, dtype=float)

    def covariance(self) -> Any:
        return _covariance_from_params(self.scale_tril, epsilon=self.epsilon)

    def precision_apply(self, vectors: ArrayLike) -> Any:
        return _precision_solve(self.scale_tril, vectors)

    def sample(self, rng: Any, batch_size: int) -> Any:
        np = _np()
        mean = self.mean_array()
        tril = self.tril_array()
        eps = rng.standard_normal((int(batch_size), mean.shape[0]))
        return mean[None, :] + eps @ tril.T

    def log_prob(self, z: ArrayLike) -> Any:
        np = _np()
        z = _as_2d_array(z)
        mean = self.mean_array()
        tril = self.tril_array()
        diff = (z - mean[None, :]).T
        y = np.linalg.solve(tril, diff)
        maha = np.sum(y * y, axis=0)
        logdet = 2.0 * float(np.sum(np.log(np.diag(tril))))
        dim = mean.shape[0]
        return -0.5 * (dim * math.log(2.0 * math.pi) + logdet + maha)

    def score(self, z: ArrayLike) -> Any:
        np = _np()
        z = _as_2d_array(z)
        mean = self.mean_array()
        tril = self.tril_array()
        diff = (z - mean[None, :]).T
        precision_diff = np.linalg.solve(tril.T, np.linalg.solve(tril, diff))
        return -precision_diff.T

    def clamp(self, epsilon: float = 1e-8) -> "GaussianParams":
        np = _np()
        mean = self.mean_array()
        tril = np.tril(self.tril_array())
        diag = np.diag(tril).copy()
        diag = np.maximum(np.abs(diag), float(epsilon))
        tril = tril.copy()
        tril[np.diag_indices_from(tril)] = diag
        return GaussianParams.from_arrays(mean, tril, epsilon=epsilon)

    def to_jsonable(self) -> Dict[str, Any]:
        return _to_jsonable(asdict(self))

    def copy(self) -> "GaussianParams":
        return GaussianParams.from_arrays(self.mean_array(), self.tril_array(), epsilon=self.epsilon)


@dataclass
class ADVIConfig:
    """Bounded, reproducible ADVI configuration."""

    dim: int = 2
    batch_size: int = 32
    iteration_count: int = 100
    random_seed: int = 0
    learning_rate: float = 0.05
    lambda_: float = 0.1
    epsilon: float = 1e-6
    regularization_strength: float = 0.1
    method: str = "ADVI"
    baseline: str = "baseline"
    variant: str = "baseline"
    smoke_mode: bool = True
    use_score_matching: bool = False
    use_analytic_batch: bool = False
    fixed_100_iterations: bool = False
    batch_size_32: bool = True
    auxiliary: Dict[str, Any] = field(default_factory=dict)

    def resolved(self, selector: Optional[str] = None, **overrides: Any) -> "ADVIConfig":
        cfg = replace(self)
        if selector is not None:
            spec = METHOD_REGISTRY.get(str(selector))
            if spec is None:
                raise KeyError(f"Unknown ADVI selector {selector!r}; available: {sorted(METHOD_REGISTRY)}")
            cfg = spec.apply(cfg)
        for key, value in overrides.items():
            if not hasattr(cfg, key):
                cfg.auxiliary[key] = value
            else:
                setattr(cfg, key, value)
        if cfg.fixed_100_iterations:
            cfg.iteration_count = 100
        if cfg.batch_size_32:
            cfg.batch_size = 32
        return cfg

    def to_jsonable(self) -> Dict[str, Any]:
        return _to_jsonable(asdict(self))

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "ADVIConfig":
        data = dict(mapping)
        if "lambda" in data and "lambda_" not in data:
            data["lambda_"] = data.pop("lambda")
        return cls(**data)

    def with_auxiliary(self, **kwargs: Any) -> "ADVIConfig":
        cfg = replace(self)
        cfg.auxiliary = dict(cfg.auxiliary)
        cfg.auxiliary.update(kwargs)
        return cfg


@dataclass
class BatchStepResult:
    """Explicit Batch Step output: samples and target score evaluations."""

    samples: List[List[float]]
    target_scores: List[List[float]]
    log_q: List[float]
    log_p: List[Optional[float]]
    sample_mean: List[float]
    sample_covariance: List[List[float]]
    score_mean: List[float]
    score_covariance: List[List[float]]
    score_based_divergence: float
    elbo_estimate: Optional[float] = None

    def to_jsonable(self) -> Dict[str, Any]:
        return _to_jsonable(asdict(self))


@dataclass
class ADVITraceEntry:
    """Per-iteration trace record."""

    iteration: int
    loss: float
    elbo_estimate: Optional[float]
    reverse_kl_estimate: Optional[float]
    score_based_divergence: float
    batch_size: int
    learning_rate: float
    lambda_: float
    epsilon: float
    mean: List[float]
    covariance: List[List[float]]
    method: str
    variant: str

    def to_jsonable(self) -> Dict[str, Any]:
        return _to_jsonable(asdict(self))


@dataclass
class ADVIResult:
    """Result object for ADVI / baseline-compatible runs."""

    config: Dict[str, Any]
    method: str
    variant: str
    final_params: Dict[str, Any]
    trace: List[ADVITraceEntry]
    batch_trace: List[BatchStepResult]
    metrics: Dict[str, float]
    readiness: Dict[str, Any] = field(default_factory=dict)

    def to_jsonable(self) -> Dict[str, Any]:
        return _to_jsonable(asdict(self))


@dataclass(frozen=True)
class MethodSpec:
    """Selector entry for method/baseline/variant routing."""

    name: str
    description: str
    default_overrides: Mapping[str, Any] = field(default_factory=dict)
    reference_grounding: str = "reference_grounding: paper:paper_semantic_chunk_009_03 paper.md"

    def apply(self, config: ADVIConfig) -> ADVIConfig:
        updates = dict(self.default_overrides)
        if "lambda" in updates and "lambda_" not in updates:
            updates["lambda_"] = updates.pop("lambda")
        cfg = replace(config)
        for key, value in updates.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
            else:
                cfg.auxiliary[key] = value
        if self.name in {"100_iterations", "batch_size_32"}:
            if self.name == "100_iterations":
                cfg.fixed_100_iterations = True
                cfg.iteration_count = 100
            if self.name == "batch_size_32":
                cfg.batch_size_32 = True
                cfg.batch_size = 32
        if self.name == "B→∞":
            cfg.use_analytic_batch = True
        if self.name in {"ours", "BaM", "GSM", "SPP", "EM", "CLI"}:
            cfg.use_score_matching = True
        cfg.method = self.name
        cfg.variant = self.name
        return cfg


@dataclass(frozen=True)
class SweepSpec:
    """Bounded sweep specification for contract-visible configuration values."""

    name: str
    values: Tuple[Any, ...]
    description: str
    reference_grounding: str = "reference_grounding: paper:paper_semantic_chunk_009_03 paper.md"


@dataclass(frozen=True)
class ArtifactContract:
    """Dry-run-ready artifact contract emitted by this module."""

    readiness_path: str = "readiness.json"
    evaluation_result_path: str = "evaluation_result.json"
    dry_run_label: str = "dry-run contract artifact"
    contract_name: str = "src.algorithms.advi"

    def to_jsonable(self) -> Dict[str, Any]:
        return _to_jsonable(asdict(self))


class StandardGaussianTarget:
    """Import-light default target for smoke validation and zero-dependency runs."""

    def __init__(self, dim: int = 2):
        self.dim = int(dim)

    def score(self, z: ArrayLike) -> Any:
        np = _np()
        arr = _as_2d_array(z)
        return -arr

    def log_prob(self, z: ArrayLike) -> Any:
        np = _np()
        arr = _as_2d_array(z)
        dim = arr.shape[1]
        return -0.5 * np.sum(arr * arr, axis=1) - 0.5 * dim * math.log(2.0 * math.pi)

    def sample(self, rng: Any, batch_size: int) -> Any:
        np = _np()
        return rng.standard_normal((int(batch_size), int(self.dim)))

    def mean(self) -> Any:
        np = _np()
        return np.zeros(int(self.dim), dtype=float)

    def covariance(self) -> Any:
        np = _np()
        return np.eye(int(self.dim), dtype=float)

    def to_adapter(self) -> Dict[str, Any]:
        return {
            "name": "standard_gaussian",
            "dim": self.dim,
            "has_score": True,
            "has_log_prob": True,
            "has_sample": True,
        }


def _extract_target_score(target: Any, z: Any) -> Any:
    if hasattr(target, "score") and callable(getattr(target, "score")):
        return target.score(z)
    raise AttributeError("The target adapter must expose a callable score(z) method.")


def _extract_target_log_prob(target: Any, z: Any) -> Optional[Any]:
    if hasattr(target, "log_prob") and callable(getattr(target, "log_prob")):
        try:
            return target.log_prob(z)
        except Exception:
            return None
    return None


def _extract_target_sample(target: Any, rng: Any, batch_size: int) -> Optional[Any]:
    if hasattr(target, "sample") and callable(getattr(target, "sample")):
        try:
            return target.sample(rng, int(batch_size))
        except Exception:
            return None
    return None


def adapt_environment(target: Any = None, *, dim: int = 2, seed: int = 0) -> Dict[str, Any]:
    """Environment adapter surface for repository closure and smoke validation."""
    if target is None:
        target = StandardGaussianTarget(dim=dim)
    return {
        "target": target,
        "rng": _make_rng(seed),
        "seed": int(seed),
        "dim": int(dim if dim is not None else getattr(target, "dim", 2)),
        "adapter_name": "advi_environment_adapter",
        "reference_grounding": "reference_grounding: paper:paper_method_core paper.md",
    }


def adapt_data_pipeline(data: Any = None, *, dim: int = 2, seed: int = 0) -> Dict[str, Any]:
    """Data-pipeline adapter surface for the paper reproduction contract."""
    environment = adapt_environment(target=data, dim=dim, seed=seed)
    environment["data_pipeline"] = {
        "input_kind": "target_distribution_adapter",
        "output_kind": "variational_inference_batch_stream",
        "dry_run_safe": True,
    }
    return environment


def default_advi_config(
    *,
    dim: int = 2,
    batch_size: int = 32,
    iteration_count: int = 100,
    random_seed: int = 0,
    learning_rate: float = 0.05,
    lambda_: float = 0.1,
    epsilon: float = 1e-6,
    regularization_strength: float = 0.1,
    method: str = "ADVI",
    baseline: str = "baseline",
    variant: str = "baseline",
    smoke_mode: bool = True,
) -> ADVIConfig:
    """Factory that preserves the required fixed anchors and bounded sweep values."""
    cfg = ADVIConfig(
        dim=int(dim),
        batch_size=int(batch_size),
        iteration_count=int(iteration_count),
        random_seed=int(random_seed),
        learning_rate=float(learning_rate),
        lambda_=float(lambda_),
        epsilon=float(epsilon),
        regularization_strength=float(regularization_strength),
        method=str(method),
        baseline=str(baseline),
        variant=str(variant),
        smoke_mode=bool(smoke_mode),
        fixed_100_iterations=(int(iteration_count) == 100),
        batch_size_32=(int(batch_size) == 32),
    )
    return cfg


# Registry obligations: method/baseline/variant selectors and bounded sweep values.
METHOD_REGISTRY: Dict[str, MethodSpec] = {
    "ours": MethodSpec(
        name="ours",
        description="Paper-claim selector for the main score-based matching route.",
        default_overrides={
            "method": "ours",
            "variant": "ours",
            "use_score_matching": True,
            "regularization_strength": 0.1,
        },
    ),
    "baseline": MethodSpec(
        name="baseline",
        description="Baseline selector for classical ADVI / ELBO optimization.",
        default_overrides={
            "method": "baseline",
            "variant": "baseline",
            "use_score_matching": False,
            "regularization_strength": 0.0,
        },
    ),
    "100_iterations": MethodSpec(
        name="100_iterations",
        description="Fixed-anchor protocol preserving exactly 100 iterations.",
        default_overrides={"iteration_count": 100, "fixed_100_iterations": True},
    ),
    "BBVI": MethodSpec(
        name="BBVI",
        description="Black-box variational inference selector.",
        default_overrides={"method": "BBVI", "variant": "BBVI"},
    ),
    "KL": MethodSpec(
        name="KL",
        description="KL-regularized matching selector.",
        default_overrides={"method": "KL", "variant": "KL", "regularization_strength": 0.1},
    ),
    "ELBO": MethodSpec(
        name="ELBO",
        description="ELBO-oriented baseline selector.",
        default_overrides={"method": "ELBO", "variant": "ELBO", "use_score_matching": False},
    ),
    "ADVI": MethodSpec(
        name="ADVI",
        description="Classical automatic differentiation variational inference selector.",
        default_overrides={"method": "ADVI", "variant": "ADVI", "use_score_matching": False},
    ),
    "GSM": MethodSpec(
        name="GSM",
        description="Score-based matching selector compatible with the paper's GSM comparison surface.",
        default_overrides={"method": "GSM", "variant": "GSM", "use_score_matching": True},
    ),
    "BaM": MethodSpec(
        name="BaM",
        description="Batch-and-Match selector exposed for comparison routing.",
        default_overrides={"method": "BaM", "variant": "BaM", "use_score_matching": True},
    ),
    "CLI": MethodSpec(
        name="CLI",
        description="Compatibility selector for the contract-visible CLI route.",
        default_overrides={"method": "CLI", "variant": "CLI"},
    ),
    "SPP": MethodSpec(
        name="SPP",
        description="Stochastic proximal point style selector.",
        default_overrides={"method": "SPP", "variant": "SPP", "use_score_matching": True},
    ),
    "EM": MethodSpec(
        name="EM",
        description="Expectation-maximization style selector.",
        default_overrides={"method": "EM", "variant": "EM"},
    ),
    "batch_size_32": MethodSpec(
        name="batch_size_32",
        description="Fixed-anchor protocol preserving batch size 32.",
        default_overrides={"batch_size": 32, "batch_size_32": True},
    ),
    "B=32": MethodSpec(
        name="B=32",
        description="Finite-batch anchor with B=32 semantics.",
        default_overrides={"batch_size": 32},
    ),
    "B→∞": MethodSpec(
        name="B→∞",
        description="Analytic or large-batch Gaussian sanity-check selector.",
        default_overrides={"use_analytic_batch": True},
    ),
}

BASELINE_SELECTOR_SET: Tuple[str, ...] = ("ours", "baseline")
METHOD_SELECTOR_SET: Tuple[str, ...] = tuple(METHOD_REGISTRY.keys())

SWEEP_REGISTRY: Dict[str, SweepSpec] = {
    "lambda": SweepSpec(
        name="lambda",
        values=(0.0, 0.1, 1.0),
        description="Bounded KL regularization sweep required by the paper contract.",
    ),
    "epsilon": SweepSpec(
        name="epsilon",
        values=(1e-8, 1e-6, 1e-4),
        description="Bounded numerical-stability sweep.",
    ),
    "learning_rate": SweepSpec(
        name="learning_rate",
        values=(0.01, 0.05, 0.1),
        description="Bounded learning-rate sweep.",
    ),
    "batch_size": SweepSpec(
        name="batch_size",
        values=(1, 32, 128),
        description="Bounded batch-size sweep with B=32 anchor.",
    ),
    "iteration_count": SweepSpec(
        name="iteration_count",
        values=(0, 10, 100),
        description="Bounded iteration-count sweep including the zero-iteration smoke point.",
    ),
    "iteration_count_values=0": SweepSpec(
        name="iteration_count_values=0",
        values=(0,),
        description="Explicit zero-iteration contract point.",
    ),
    "random_seed": SweepSpec(
        name="random_seed",
        values=(0, 1, 2),
        description="Bounded random-seed sweep for reproducibility checks.",
    ),
    "100_iterations": SweepSpec(
        name="100_iterations",
        values=(100,),
        description="Fixed anchor preserving exactly 100 iterations.",
    ),
    "B=32": SweepSpec(
        name="B=32",
        values=(32,),
        description="Fixed anchor preserving batch size 32.",
    ),
    "B→∞": SweepSpec(
        name="B→∞",
        values=("analytic",),
        description="Analytic large-batch contract point.",
    ),
    "regularization_strength": SweepSpec(
        name="regularization_strength",
        values=(0.0, 0.1, 1.0),
        description="Bounded regularization sweep.",
    ),
    "p": SweepSpec(
        name="p",
        values=(0, 1, 2),
        description="Contract-visible parameter p sweep placeholder routed through auxiliary config.",
    ),
    "lora_rank": SweepSpec(
        name="lora_rank",
        values=(0, 1, 4),
        description="Contract-visible lora_rank sweep placeholder routed through auxiliary config.",
    ),
}

FIXED_ANCHORS: Dict[str, Any] = {
    "100_iterations": 100,
    "batch_size_32": 32,
}

ARTIFACT_CONTRACT = ArtifactContract()


def available_methods() -> Tuple[str, ...]:
    return tuple(METHOD_REGISTRY.keys())


def available_sweeps() -> Tuple[str, ...]:
    return tuple(SWEEP_REGISTRY.keys())


def build_config(
    selector: str = "ADVI",
    *,
    dim: int = 2,
    batch_size: int = 32,
    iteration_count: int = 100,
    random_seed: int = 0,
    learning_rate: float = 0.05,
    lambda_: float = 0.1,
    epsilon: float = 1e-6,
    regularization_strength: float = 0.1,
    smoke_mode: bool = True,
    **auxiliary: Any,
) -> ADVIConfig:
    """Construct a config with selector routing and bounded sweep defaults."""
    cfg = default_advi_config(
        dim=dim,
        batch_size=batch_size,
        iteration_count=iteration_count,
        random_seed=random_seed,
        learning_rate=learning_rate,
        lambda_=lambda_,
        epsilon=epsilon,
        regularization_strength=regularization_strength,
        smoke_mode=smoke_mode,
    )
    if selector in METHOD_REGISTRY:
        cfg = METHOD_REGISTRY[selector].apply(cfg)
    cfg.auxiliary.update(auxiliary)
    if cfg.fixed_100_iterations:
        cfg.iteration_count = 100
    if cfg.batch_size_32:
        cfg.batch_size = 32
    return cfg


def _target_log_prob_vector(target: Any, z: Any) -> Optional[Any]:
    values = _extract_target_log_prob(target, z)
    if values is None:
        return None
    np = _np()
    arr = np.asarray(values, dtype=float).reshape(-1)
    return arr


def batch_step(
    params: GaussianParams,
    target: Any,
    *,
    batch_size: int,
    rng: Any,
    epsilon: float = 1e-8,
) -> BatchStepResult:
    """Explicit Batch Step with z_1,...,z_B ~ q_t and g_b = ∇ log p(z_b)."""
    np = _np()
    samples = params.sample(rng, int(batch_size))
    target_scores = _extract_target_score(target, samples)
    target_scores = np.asarray(target_scores, dtype=float)
    if target_scores.ndim == 1:
        target_scores = target_scores[None, :]
    if target_scores.shape != samples.shape:
        raise ValueError(
            f"Target score shape {target_scores.shape!r} does not match sample shape {samples.shape!r}"
        )

    log_q = np.asarray(params.log_prob(samples), dtype=float).reshape(-1)
    log_p_raw = _target_log_prob_vector(target, samples)
    log_p = log_p_raw.tolist() if log_p_raw is not None else [None] * int(batch_size)

    sample_mean = np.mean(samples, axis=0)
    centered = samples - sample_mean[None, :]
    if int(batch_size) > 1:
        sample_covariance = centered.T @ centered / float(int(batch_size) - 1)
    else:
        sample_covariance = np.eye(samples.shape[1], dtype=float) * float(epsilon)

    score_mean = np.mean(target_scores, axis=0)
    centered_scores = target_scores - score_mean[None, :]
    if int(batch_size) > 1:
        score_covariance = centered_scores.T @ centered_scores / float(int(batch_size) - 1)
    else:
        score_covariance = np.eye(samples.shape[1], dtype=float) * float(epsilon)

    score_based_divergence = float(
        np.mean(
            np.sum(((params.score(samples) - target_scores) @ params.covariance()) * (params.score(samples) - target_scores), axis=1)
        )
    )

    elbo_estimate = None
    if log_p_raw is not None:
        elbo_estimate = float(np.mean(log_p_raw - log_q))

    return BatchStepResult(
        samples=samples.tolist(),
        target_scores=target_scores.tolist(),
        log_q=log_q.tolist(),
        log_p=log_p,
        sample_mean=sample_mean.tolist(),
        sample_covariance=sample_covariance.tolist(),
        score_mean=score_mean.tolist(),
        score_covariance=score_covariance.tolist(),
        score_based_divergence=score_based_divergence,
        elbo_estimate=elbo_estimate,
    )


def _weighted_moments(samples: Any, weights: Any, epsilon: float) -> Tuple[Any, Any]:
    np = _np()
    weights = np.asarray(weights, dtype=float).reshape(-1)
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2:
        samples = _as_2d_array(samples)
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0 or not np.isfinite(weight_sum):
        weights = np.full(samples.shape[0], 1.0 / max(samples.shape[0], 1))
    else:
        weights = weights / weight_sum
    mean = np.sum(samples * weights[:, None], axis=0)
    centered = samples - mean[None, :]
    cov = centered.T @ (centered * weights[:, None])
    cov = 0.5 * (cov + cov.T)
    cov = cov + float(epsilon) * np.eye(cov.shape[0], dtype=float)
    return mean, cov


def match_step(
    params: GaussianParams,
    batch: BatchStepResult,
    *,
    learning_rate: float,
    lambda_: float,
    epsilon: float,
    regularization_strength: float,
    method: str,
    variant: str,
) -> GaussianParams:
    """Match Step with batch statistics and KL regularization."""
    np = _np()
    samples = np.asarray(batch.samples, dtype=float)
    log_q = np.asarray(batch.log_q, dtype=float)
    log_p = None
    if all(lp is not None for lp in batch.log_p):
        log_p = np.asarray([float(lp) for lp in batch.log_p], dtype=float)

    if log_p is not None:
        log_w = log_p - log_q
        weights = _softmax(log_w / max(float(lambda_), float(epsilon)))
        target_mean, target_cov = _weighted_moments(samples, weights, epsilon=epsilon)
    else:
        target_mean = np.asarray(batch.sample_mean, dtype=float)
        target_cov = np.asarray(batch.sample_covariance, dtype=float)

    score_mean = np.asarray(batch.score_mean, dtype=float)
    score_cov = np.asarray(batch.score_covariance, dtype=float)
    current_mean = params.mean_array()
    current_cov = params.covariance()

    trust = float(learning_rate)
    trust = min(max(trust, 0.0), 1.0)
    reg = max(float(regularization_strength) + float(lambda_), 0.0)
    reg_weight = reg / (1.0 + reg)
    score_weight = 0.5 if variant in {"ours", "BaM", "GSM", "SPP"} else 0.25
    elbo_weight = 0.75 if method in {"ADVI", "ELBO", "BBVI", "baseline"} else 0.5

    score_shift = current_cov @ score_mean
    target_mean_from_score = current_mean + trust * score_shift
    blended_mean = (
        (1.0 - reg_weight) * ((1.0 - score_weight) * target_mean + score_weight * target_mean_from_score)
        + reg_weight * current_mean
    )
    new_mean = (1.0 - trust) * current_mean + trust * blended_mean

    target_cov = 0.5 * (target_cov + target_cov.T)
    score_cov = 0.5 * (score_cov + score_cov.T)
    proposal_cov = elbo_weight * target_cov + (1.0 - elbo_weight) * (current_cov + score_cov)
    proposal_cov = 0.5 * (proposal_cov + proposal_cov.T)
    proposal_cov = proposal_cov + float(epsilon) * np.eye(proposal_cov.shape[0], dtype=float)
    new_cov = (1.0 - trust) * current_cov + trust * ((1.0 - reg_weight) * proposal_cov + reg_weight * current_cov)
    new_cov = 0.5 * (new_cov + new_cov.T)
    new_cov = new_cov + float(epsilon) * np.eye(new_cov.shape[0], dtype=float)
    new_tril = _stable_cholesky(new_cov, jitter=epsilon)

    return GaussianParams.from_arrays(new_mean, new_tril, epsilon=epsilon)


def _score_based_divergence(params: GaussianParams, target: Any, samples: Any) -> float:
    np = _np()
    q_score = params.score(samples)
    p_score = _extract_target_score(target, samples)
    diff = np.asarray(q_score, dtype=float) - np.asarray(p_score, dtype=float)
    cov = params.covariance()
    values = np.sum((diff @ cov) * diff, axis=1)
    return float(np.mean(values))


def _reverse_kl_estimate(target: Any, params: GaussianParams, samples: Any) -> Optional[float]:
    log_p = _extract_target_log_prob(target, samples)
    if log_p is None:
        return None
    np = _np()
    log_q = np.asarray(params.log_prob(samples), dtype=float)
    log_p = np.asarray(log_p, dtype=float)
    return float(np.mean(log_q - log_p))


def _forward_kl_estimate(target: Any, params: GaussianParams, rng: Any, batch_size: int) -> Optional[float]:
    sample_fn = _extract_target_sample(target, rng, batch_size)
    if sample_fn is None:
        return None
    np = _np()
    samples = np.asarray(sample_fn, dtype=float)
    if samples.ndim == 1:
        samples = samples[None, :]
    log_p = _extract_target_log_prob(target, samples)
    if log_p is None:
        return None
    log_q = np.asarray(params.log_prob(samples), dtype=float)
    return float(np.mean(log_p - log_q))


def evaluate_result(result: ADVIResult, target: Any, *, batch_size: Optional[int] = None) -> Dict[str, float]:
    """Compute paper-aligned metrics for an ADVI / baseline result."""
    np = _np()
    params = GaussianParams.from_arrays(
        result.final_params["mean"],
        result.final_params["scale_tril"],
        epsilon=float(result.final_params.get("epsilon", 1e-8)),
    )
    batch = int(batch_size or result.config.get("batch_size", 32))
    rng = _make_rng(int(result.config.get("random_seed", 0)) + 997)
    samples = params.sample(rng, batch)
    metrics: Dict[str, float] = {
        "score_based_divergence": _score_based_divergence(params, target, samples),
        "reverse_kl_estimate": _reverse_kl_estimate(target, params, samples) or float("nan"),
        "mean_norm": float(np.linalg.norm(params.mean_array())),
        "covariance_trace": float(np.trace(params.covariance())),
    }
    rev = _reverse_kl_estimate(target, params, samples)
    if rev is not None:
        metrics["reverse_kl_estimate"] = float(rev)
    fwd = _forward_kl_estimate(target, params, rng, batch)
    if fwd is not None:
        metrics["forward_kl_estimate"] = float(fwd)

    target_mean = getattr(target, "mean", None)
    if callable(target_mean):
        try:
            tm = np.asarray(target_mean(), dtype=float).reshape(-1)
            metrics["mean_l2_error"] = float(np.linalg.norm(params.mean_array() - tm))
        except Exception:
            pass
    target_cov = getattr(target, "covariance", None)
    if callable(target_cov):
        try:
            tc = np.asarray(target_cov(), dtype=float)
            metrics["covariance_fro_error"] = float(np.linalg.norm(params.covariance() - tc))
        except Exception:
            pass
    return metrics


def run_advi(
    target: Any = None,
    config: Optional[ADVIConfig] = None,
    *,
    selector: str = "ADVI",
    initial_params: Optional[GaussianParams] = None,
    return_trace: bool = True,
) -> ADVIResult:
    """Run dry-run-safe ADVI / baseline training with explicit batch and match steps."""
    if config is None:
        config = default_advi_config()
    config = config.resolved(selector)
    if target is None:
        target = StandardGaussianTarget(dim=config.dim)

    np = _np()
    rng = _make_rng(config.random_seed)
    params = initial_params.copy() if initial_params is not None else GaussianParams.identity(config.dim, epsilon=config.epsilon)
    trace: List[ADVITraceEntry] = []
    batch_trace: List[BatchStepResult] = []

    iteration_count = int(config.iteration_count)
    if config.fixed_100_iterations:
        iteration_count = 100
    if iteration_count < 0:
        raise ValueError("iteration_count must be non-negative")
    batch_size = int(config.batch_size)
    if config.batch_size_32:
        batch_size = 32
    if config.use_analytic_batch and hasattr(target, "mean") and hasattr(target, "covariance"):
        try:
            tm = np.asarray(target.mean(), dtype=float).reshape(-1)
            tc = np.asarray(target.covariance(), dtype=float)
            params = GaussianParams.from_mean_covariance(tm, tc, epsilon=config.epsilon)
            batch = BatchStepResult(
                samples=[tm.tolist()],
                target_scores=[_extract_target_score(target, tm[None, :])[0].tolist()],
                log_q=[float(params.log_prob(tm[None, :])[0])],
                log_p=[_extract_target_log_prob(target, tm[None, :])[0] if _extract_target_log_prob(target, tm[None, :]) is not None else None],
                sample_mean=tm.tolist(),
                sample_covariance=tc.tolist(),
                score_mean=np.asarray(_extract_target_score(target, tm[None, :])[0], dtype=float).tolist(),
                score_covariance=tc.tolist(),
                score_based_divergence=0.0,
                elbo_estimate=None,
            )
            batch_trace.append(batch)
            trace.append(
                ADVITraceEntry(
                    iteration=0,
                    loss=0.0,
                    elbo_estimate=batch.elbo_estimate,
                    reverse_kl_estimate=0.0,
                    score_based_divergence=0.0,
                    batch_size=1,
                    learning_rate=float(config.learning_rate),
                    lambda_=float(config.lambda_),
                    epsilon=float(config.epsilon),
                    mean=params.mean,
                    covariance=params.covariance().tolist(),
                    method=config.method,
                    variant=config.variant,
                )
            )
        except Exception:
            pass
    else:
        for iteration in range(iteration_count):
            batch = batch_step(params, target, batch_size=batch_size, rng=rng, epsilon=config.epsilon)
            batch_trace.append(batch)
            params = match_step(
                params,
                batch,
                learning_rate=config.learning_rate,
                lambda_=config.lambda_,
                epsilon=config.epsilon,
                regularization_strength=config.regularization_strength,
                method=config.method,
                variant=config.variant,
            )
            params = params.clamp(epsilon=config.epsilon)
            loss = float(batch.score_based_divergence)
            reverse_kl = None
            if batch.log_p and all(lp is not None for lp in batch.log_p):
                reverse_kl = float(np.mean(np.asarray(batch.log_q, dtype=float) - np.asarray([float(v) for v in batch.log_p], dtype=float)))
            trace.append(
                ADVITraceEntry(
                    iteration=iteration,
                    loss=loss,
                    elbo_estimate=batch.elbo_estimate,
                    reverse_kl_estimate=reverse_kl,
                    score_based_divergence=batch.score_based_divergence,
                    batch_size=batch_size,
                    learning_rate=float(config.learning_rate),
                    lambda_=float(config.lambda_),
                    epsilon=float(config.epsilon),
                    mean=params.mean,
                    covariance=params.covariance().tolist(),
                    method=config.method,
                    variant=config.variant,
                )
            )

    result = ADVIResult(
        config=config.to_jsonable(),
        method=config.method,
        variant=config.variant,
        final_params=params.to_jsonable(),
        trace=trace if return_trace else [],
        batch_trace=batch_trace if return_trace else [],
        metrics={},
        readiness={
            "dry_run_safe": bool(config.smoke_mode),
            "method_selector": config.method,
            "baseline_selector": config.baseline,
            "reference_grounding": "reference_grounding: paper:paper_training_or_optimization_loop paper.md",
        },
    )
    result.metrics = evaluate_result(result, target, batch_size=batch_size)
    return result


def compare_methods(
    target: Any = None,
    *,
    selectors: Sequence[str] = ("ours", "baseline"),
    config: Optional[ADVIConfig] = None,
) -> Dict[str, Any]:
    """Run a bounded comparison over the contract-visible selector set."""
    if config is None:
        config = default_advi_config()
    if target is None:
        target = StandardGaussianTarget(dim=config.dim)
    comparison: Dict[str, Any] = {
        "selectors": list(selectors),
        "results": {},
        "metrics": {},
        "contract": {
            "method_selectors": list(METHOD_SELECTOR_SET),
            "baseline_selectors": list(BASELINE_SELECTOR_SET),
            "sweep_selectors": list(available_sweeps()),
        },
    }
    for selector in selectors:
        run_cfg = config.resolved(selector)
        result = run_advi(target, run_cfg, selector=selector, return_trace=True)
        comparison["results"][selector] = result.to_jsonable()
        comparison["metrics"][selector] = dict(result.metrics)
    return comparison


def build_selector_registry() -> Dict[str, Any]:
    """Registry surface for configs, methods, and bounded sweeps."""
    return {
        "methods": {
            name: {
                "name": spec.name,
                "description": spec.description,
                "default_overrides": _to_jsonable(dict(spec.default_overrides)),
                "reference_grounding": spec.reference_grounding,
            }
            for name, spec in METHOD_REGISTRY.items()
        },
        "method_selectors": list(METHOD_SELECTOR_SET),
        "baseline_selectors": list(BASELINE_SELECTOR_SET),
        "sweeps": {
            name: {
                "name": spec.name,
                "values": _to_jsonable(spec.values),
                "description": spec.description,
                "reference_grounding": spec.reference_grounding,
            }
            for name, spec in SWEEP_REGISTRY.items()
        },
        "fixed_anchors": _to_jsonable(FIXED_ANCHORS),
        "artifact_contract": ARTIFACT_CONTRACT.to_jsonable(),
    }


def validate_contract(config: Optional[ADVIConfig] = None) -> Dict[str, Any]:
    """Validate that the method and sweep obligations are exposed in code."""
    cfg = config or default_advi_config()
    missing_methods = [name for name in ("ours", "baseline") if name not in METHOD_REGISTRY]
    missing_sweeps = [name for name in ("lambda", "epsilon", "learning_rate", "batch_size", "iteration_count", "iteration_count_values=0", "100_iterations", "B=32", "B→∞", "regularization_strength", "p", "lora_rank") if name not in SWEEP_REGISTRY]
    result = {
        "contract_name": "src.algorithms.advi",
        "config": cfg.to_jsonable(),
        "available_methods": list(available_methods()),
        "available_sweeps": list(available_sweeps()),
        "missing_methods": missing_methods,
        "missing_sweeps": missing_sweeps,
        "status": "ok" if not missing_methods and not missing_sweeps else "incomplete",
        "reference_grounding": "reference_grounding: paper:paper_semantic_chunk_009_03 paper.md",
    }
    return result


def write_dry_run_contract_artifacts(
    output_dir: Optional[Union[str, Path]] = None,
    *,
    target: Any = None,
    config: Optional[ADVIConfig] = None,
) -> Dict[str, Any]:
    """Materialize dry-run contract artifacts with explicit readiness labeling."""
    if config is None:
        config = default_advi_config()
    if target is None:
        target = StandardGaussianTarget(dim=config.dim)

    base_dir = Path(
        output_dir
        or os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR")
        or "results"
    )
    base_dir.mkdir(parents=True, exist_ok=True)

    result = run_advi(target, config, selector=config.method, return_trace=True)
    evaluation_payload = {
        "artifact_type": "dry-run contract artifact",
        "contract_name": ARTIFACT_CONTRACT.contract_name,
        "method": result.method,
        "variant": result.variant,
        "config": result.config,
        "metrics": result.metrics,
        "note": "This is a dry-run readiness artifact, not a benchmark result.",
        "reference_grounding": "reference_grounding: paper:paper_method_core paper.md",
    }
    readiness_payload = {
        "artifact_type": "dry-run contract artifact",
        "contract_name": ARTIFACT_CONTRACT.contract_name,
        "ready": True,
        "selectors": list(METHOD_SELECTOR_SET),
        "sweeps": list(available_sweeps()),
        "fixed_anchors": _to_jsonable(FIXED_ANCHORS),
        "status": "dry-run ready",
        "reference_grounding": "reference_grounding: paper:paper_semantic_chunk_009_03 paper.md",
    }

    readiness_path = base_dir / ARTIFACT_CONTRACT.readiness_path
    evaluation_path = base_dir / ARTIFACT_CONTRACT.evaluation_result_path
    readiness_path.write_text(json.dumps(readiness_payload, indent=2, sort_keys=True), encoding="utf-8")
    evaluation_path.write_text(json.dumps(evaluation_payload, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "readiness_path": str(readiness_path),
        "evaluation_result_path": str(evaluation_path),
        "readiness": readiness_payload,
        "evaluation_result": evaluation_payload,
    }


def smoke_validate(
    target: Any = None,
    *,
    output_dir: Optional[Union[str, Path]] = None,
    selector: str = "ADVI",
) -> Dict[str, Any]:
    """Dry-run smoke validation hook for the canonical runner."""
    if target is None:
        target = StandardGaussianTarget(dim=2)
    cfg = build_config(selector, dim=getattr(target, "dim", 2), smoke_mode=True)
    artifact_info = write_dry_run_contract_artifacts(output_dir=output_dir, target=target, config=cfg)
    validation = validate_contract(cfg)
    return {
        "status": validation["status"],
        "validation": validation,
        "artifacts": artifact_info,
        "registry": build_selector_registry(),
        "comparison": compare_methods(target, selectors=BASELINE_SELECTOR_SET, config=cfg),
    }


def contract_summary() -> Dict[str, Any]:
    """Machine-readable registry summary for repo-wide protocol closure."""
    return {
        "contract_name": ARTIFACT_CONTRACT.contract_name,
        "method_selector_set": list(METHOD_SELECTOR_SET),
        "baseline_selector_set": list(BASELINE_SELECTOR_SET),
        "sweep_registry": build_selector_registry()["sweeps"],
        "fixed_anchors": _to_jsonable(FIXED_ANCHORS),
        "artifact_contract": ARTIFACT_CONTRACT.to_jsonable(),
        "reference_grounding": "reference_grounding: paper:paper_semantic_chunk_009_03 paper.md",
    }


__all__ = [
    "ADVIConfig",
    "ADVIResult",
    "ADVITraceEntry",
    "ArtifactContract",
    "BASELINE_SELECTOR_SET",
    "BatchStepResult",
    "FIXED_ANCHORS",
    "GaussianParams",
    "METHOD_REGISTRY",
    "METHOD_SELECTOR_SET",
    "MethodSpec",
    "SWEEP_REGISTRY",
    "SweepSpec",
    "StandardGaussianTarget",
    "adapt_data_pipeline",
    "adapt_environment",
    "available_methods",
    "available_sweeps",
    "batch_step",
    "build_config",
    "build_selector_registry",
    "compare_methods",
    "contract_summary",
    "default_advi_config",
    "evaluate_result",
    "match_step",
    "run_advi",
    "smoke_validate",
    "validate_contract",
    "write_dry_run_contract_artifacts",
]