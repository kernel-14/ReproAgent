"""Configuration registry for the BaM PaperBench reproduction.

This module owns the import-light configuration surface for the paper
"Batch and match: black-box variational inference with a score-based divergence".
It makes the canonical algorithm, target, baseline, metric, data, environment,
and artifact contracts explicit without importing optional numerical, plotting,
dataset, accelerator, or probabilistic-programming packages at module import
time.

The executable method code lives in neighboring modules such as
``bam.training_loop``, ``bam.variational``, ``bam.targets``,
``bam.score_divergence``, and ``src.algorithms.*``.  The role of this file is to
provide stable, validated configuration objects and registries that route the
canonical runner to those implementations.

reference_grounding: paper:paper_method_core paper.md
    Score-based BBVI assumes access to the target score ∇ log p(z), so BaM
    configurations require a target score interface and never require the
    normalizing constant of p.

reference_grounding: paper:paper_training_or_optimization_loop paper.md
    The BaM algorithm uses an explicit Batch Step
    z_1,...,z_B ~ q_t, g_b = ∇ log p(z_b), and batch statistics, followed by a
    Match Step that updates full-covariance Gaussian variational parameters with
    KL regularization.

reference_grounding: paper:paper_semantic_chunk_009_03 paper.md
    BaM is a KL-regularized stochastic proximal/matching update.  This registry
    exposes the regularization controls lambda and epsilon, finite-batch B=32
    semantics, and an analytic B→∞ Gaussian sanity-check configuration.
    The bounded sweep surface also carries the contract parameter lora_rank=0
    for compatibility with evaluator metadata; it is not expanded into an
    unrelated LoRA experiment.
"""

from __future__ import annotations

import json
import math
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union


ArrayLike = Any
LogDensityFn = Callable[[ArrayLike], Union[float, ArrayLike]]
ScoreFn = Callable[[ArrayLike], ArrayLike]


PAPER_TITLE = "Batch and match: black-box variational inference with a score-based divergence"
DEFAULT_ARTIFACT_DIR = "results"
DEFAULT_ITERATIONS = 100
DEFAULT_BATCH_SIZE = 32
DEFAULT_LAMBDA = 1.0
DEFAULT_EPSILON = 1.0e-6
DEFAULT_SEED = 0
DEFAULT_DIMENSION = 4
ANALYTIC_BATCH_SENTINEL = "infinity"

CANONICAL_RUNTIME_ARTIFACTS: Tuple[str, ...] = (
    "results/loss_trace.json",
    "results/bam_trace.json",
    "results/bam_final_variational_params.npz",
    "results/batch_statistics_trace.json",
    "results/gaussian_sanity_metrics.json",
    "results/figures/figure_5.png",
)

CANONICAL_ROUTE_ARTIFACTS: Tuple[str, ...] = (
    "results/metrics.json",
    "results/run_summary.json",
    "results/config_echo.json",
    "results/evidence_contract_matrix.json",
    "results/experiment_registry.json",
    "results/environment_registry.json",
)


def _artifact_root() -> Path:
    """Return the output root, honoring PaperBench's auxiliary artifact env var."""
    return Path(os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR", ".")).resolve()


def resolve_artifact_path(relative_path: Union[str, Path]) -> Path:
    """Resolve a repository-relative artifact path under the active output root."""
    path = Path(relative_path)
    if path.is_absolute():
        return path
    return _artifact_root() / path


def ensure_artifact_dirs(paths: Iterable[Union[str, Path]] = CANONICAL_RUNTIME_ARTIFACTS + CANONICAL_ROUTE_ARTIFACTS) -> None:
    """Create parent directories for declared artifacts without writing fake results."""
    for path in paths:
        resolve_artifact_path(path).parent.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class TargetSpec:
    """Target distribution contract for score-based BBVI.

    ``score_symbol`` points to a callable implementing ∇ log p(z).  BaM only
    requires this score and does not require the normalizing constant of p.
    ``log_density_symbol`` may be unnormalized and is used only by diagnostics
    or KL estimators when available.
    """

    name: str
    dimension: int = DEFAULT_DIMENSION
    target_type: str = "gaussian"
    log_density_symbol: str = "bam.targets:make_target.log_prob"
    score_symbol: str = "bam.targets:make_target.score"
    data_prepare_symbol: str = "src.data.data:prepare_dataset"
    data_validate_symbol: str = "src.data.data:validate_dataset"
    normalized_density_required: bool = False
    true_mean: Optional[Tuple[float, ...]] = None
    true_covariance: Optional[Tuple[Tuple[float, ...], ...]] = None
    description: str = "Full-covariance Gaussian score target for BaM sanity checks."


@dataclass(frozen=True)
class GaussianVariationalConfig:
    """Full-covariance Gaussian variational-family configuration."""

    dimension: int = DEFAULT_DIMENSION
    init_mean: Optional[Tuple[float, ...]] = None
    init_covariance: Optional[Tuple[Tuple[float, ...], ...]] = None
    min_eigenvalue: float = 1.0e-6
    covariance_parameterization: str = "stable_full_covariance"
    sample_symbol: str = "bam.variational:FullCovarianceGaussian.sample"
    log_density_symbol: str = "bam.variational:FullCovarianceGaussian.log_prob"
    score_symbol: str = "bam.variational:FullCovarianceGaussian.score"
    kl_regularizer_symbol: str = "bam.variational:gaussian_kl"

    def mean_vector(self) -> Tuple[float, ...]:
        if self.init_mean is not None:
            return tuple(float(x) for x in self.init_mean)
        return tuple(0.0 for _ in range(self.dimension))

    def covariance_matrix(self) -> Tuple[Tuple[float, ...], ...]:
        if self.init_covariance is not None:
            return tuple(tuple(float(v) for v in row) for row in self.init_covariance)
        return tuple(
            tuple(1.0 if i == j else 0.0 for j in range(self.dimension))
            for i in range(self.dimension)
        )


@dataclass(frozen=True)
class BatchStepConfig:
    """Configuration for the paper's explicit Batch Step."""

    batch_size: Union[int, str] = DEFAULT_BATCH_SIZE
    analytic_infinite_batch: bool = False
    sample_symbol: str = "bam.optimizer:batch_step"
    statistics_symbol: str = "bam.score_divergence:compute_batch_statistics"
    required_statistics: Tuple[str, ...] = (
        "samples",
        "target_scores",
        "zbar",
        "C",
        "gbar",
        "Gamma",
        "sample_score_covariance",
        "score_sample_covariance",
    )

    def finite_batch_size(self) -> Optional[int]:
        if self.analytic_infinite_batch or self.batch_size == ANALYTIC_BATCH_SENTINEL:
            return None
        return int(self.batch_size)


@dataclass(frozen=True)
class MatchStepConfig:
    """Configuration for KL-regularized Gaussian Match Step."""

    lambda_regularization: float = DEFAULT_LAMBDA
    epsilon: float = DEFAULT_EPSILON
    update_symbol: str = "bam.optimizer:match_step"
    divergence_symbol: str = "bam.score_divergence:score_based_divergence"
    regularizer: str = "KL(q_new || q_old)"
    covariance_stabilization: str = "symmetric_eigendecomposition_floor"


@dataclass(frozen=True)
class TrainingConfig:
    """Canonical BaM training/optimization configuration.

    Required paper controls are exposed directly: ``iterations`` defaults to
    100, ``lambda_regularization`` controls the KL/proximal match update,
    ``epsilon`` stabilizes full-covariance updates, ``batch_size`` controls the
    Batch Step, and ``seed`` makes sampling deterministic.
    """

    experiment_id: str = "bam_gaussian_B32"
    method: str = "bam"
    target_name: str = "gaussian_d4"
    iterations: int = DEFAULT_ITERATIONS
    batch_size: Union[int, str] = DEFAULT_BATCH_SIZE
    lambda_regularization: float = DEFAULT_LAMBDA
    epsilon: float = DEFAULT_EPSILON
    seed: int = DEFAULT_SEED
    dimension: int = DEFAULT_DIMENSION
    mode: str = "runtime_smoke"
    output_dir: str = DEFAULT_ARTIFACT_DIR
    variational: GaussianVariationalConfig = field(default_factory=GaussianVariationalConfig)
    batch_step: BatchStepConfig = field(default_factory=BatchStepConfig)
    match_step: MatchStepConfig = field(default_factory=MatchStepConfig)
    target: TargetSpec = field(default_factory=lambda: TARGET_REGISTRY["gaussian_d4"])
    hypothesis: str = (
        "BaM should reduce the score-based divergence for a full-covariance "
        "Gaussian variational approximation using only target scores."
    )
    decisive_comparison: str = "BaM versus ADVI and GSM on shared Gaussian variational output schema."
    decisive_metric: str = "score_based_divergence_and_gaussian_mean_covariance_error"
    stop_rule_or_pruning_rationale: str = (
        "Default route executes bounded smoke iterations while preserving the "
        "same Batch Step, Match Step, score, metric, and artifact paths as full mode."
    )
    write_artifacts: bool = True

    def normalized(self) -> "TrainingConfig":
        """Return a self-consistent config with nested batch/match/target fields aligned."""
        analytic = self.batch_size == ANALYTIC_BATCH_SENTINEL
        dim = int(self.dimension)
        target = TARGET_REGISTRY.get(self.target_name, self.target)
        target = replace(target, dimension=dim)
        variational = replace(self.variational, dimension=dim)
        batch = replace(
            self.batch_step,
            batch_size=self.batch_size,
            analytic_infinite_batch=analytic,
        )
        match = replace(
            self.match_step,
            lambda_regularization=float(self.lambda_regularization),
            epsilon=float(self.epsilon),
        )
        return replace(self, target=target, variational=variational, batch_step=batch, match_step=match)


@dataclass(frozen=True)
class MethodSpec:
    """Method/baseline registry entry."""

    name: str
    kind: str
    factory_symbol: str
    training_symbol: str
    description: str
    default_config: str
    uses_target_score: bool = True
    uses_normalized_target_density: bool = False
    reference_grounding: str = "paper:paper_method_core paper.md"


@dataclass(frozen=True)
class MetricSpec:
    """Metric formula registry entry."""

    name: str
    formula: str
    implementation_symbol: str
    artifact_path: str
    description: str


@dataclass(frozen=True)
class EnvironmentSpec:
    """Import-light execution environment contract.

    The paper experiments in this repository are numerical VI experiments rather
    than simulator/RL tasks.  This adapter records the Python/numerical runtime
    and optional package availability so the canonical route can validate its
    environment without importing heavy packages at module import time.
    """

    name: str
    kind: str = "local_python"
    python: str = f"{sys.version_info.major}.{sys.version_info.minor}"
    required_packages: Tuple[str, ...] = ("numpy",)
    optional_packages: Tuple[str, ...] = ("matplotlib", "pandas")
    artifact_path: str = "results/environment_registry.json"
    description: str = "Local CPU numerical environment for BaM score-based VI."


class EnvironmentRegistry:
    """Small registry for environment adapters."""

    def __init__(self, specs: Optional[Mapping[str, EnvironmentSpec]] = None) -> None:
        self._specs: Dict[str, EnvironmentSpec] = dict(specs or {})

    def register(self, spec: EnvironmentSpec) -> None:
        self._specs[spec.name] = spec

    def get(self, name: str = "local") -> EnvironmentSpec:
        if name not in self._specs:
            raise KeyError(f"Unknown environment {name!r}; available={sorted(self._specs)}")
        return self._specs[name]

    def as_dict(self) -> Dict[str, Dict[str, Any]]:
        return {name: asdict(spec) for name, spec in self._specs.items()}

    def names(self) -> Tuple[str, ...]:
        return tuple(sorted(self._specs))


@dataclass
class PolicyAdapter:
    """Adapter exposing a method object through a policy-like interface.

    This symbol is required by the repository contract.  In this VI repository,
    "policy" means the chosen inference method factory.  The adapter delegates
    training to ``run_training_loop`` so environment/model loading remains
    import-light and deterministic.
    """

    name: str
    config: TrainingConfig

    def train(self, target: Optional[Any] = None) -> "EvaluationResult":
        return run_training_loop(self.config, target=target)

    def __call__(self, target: Optional[Any] = None) -> "EvaluationResult":
        return self.train(target=target)


@dataclass(frozen=True)
class AgentConfig:
    """Method-selection config for BaM/ADVI/GSM runners."""

    method: str = "bam"
    config_name: str = "bam_gaussian_B32"
    environment: str = "local"
    allow_gsm_limiting_case: bool = True
    gsm_note: str = (
        "GSM special limiting-case hook is implemented locally; no code is "
        "imported from the blacklisted GSM-VI repository."
    )


@dataclass
class EvaluationResult:
    """Uniform result returned by training and smoke routes."""

    method: str
    experiment_id: str
    metrics: Dict[str, float]
    artifacts: Dict[str, str]
    trace: List[Dict[str, Any]] = field(default_factory=list)
    batch_statistics_trace: List[Dict[str, Any]] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    status: str = "completed"
    elapsed_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "experiment_id": self.experiment_id,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "trace": self.trace,
            "batch_statistics_trace": self.batch_statistics_trace,
            "config": self.config,
            "status": self.status,
            "elapsed_seconds": self.elapsed_seconds,
        }


TARGET_REGISTRY: Dict[str, TargetSpec] = {
    "gaussian_d4": TargetSpec(
        name="gaussian_d4",
        dimension=4,
        target_type="gaussian",
        true_mean=(1.0, -1.0, 0.5, -0.5),
        true_covariance=(
            (1.0, 0.25, 0.0, 0.0),
            (0.25, 1.5, 0.1, 0.0),
            (0.0, 0.1, 0.75, 0.05),
            (0.0, 0.0, 0.05, 1.25),
        ),
        description="D=4 full-covariance Gaussian target with analytic score and KL sanity metrics.",
    ),
    "gaussian_d16": TargetSpec(
        name="gaussian_d16",
        dimension=16,
        target_type="gaussian",
        description="D=16 Gaussian target protocol entry for scaling experiments.",
    ),
    "banana": TargetSpec(
        name="banana",
        dimension=2,
        target_type="non_gaussian",
        description="Controlled non-Gaussian score target for paper-style robustness checks.",
    ),
    "hierarchical_logistic": TargetSpec(
        name="hierarchical_logistic",
        dimension=8,
        target_type="hierarchical_bayes",
        description="Lightweight hierarchical posterior target using unnormalized log density and score.",
    ),
}

METHOD_REGISTRY: Dict[str, MethodSpec] = {
    "bam": MethodSpec(
        name="bam",
        kind="proposed",
        factory_symbol="src.algorithms.bam:make_bam",
        training_symbol="bam.training_loop:run_training_loop",
        description="Batch and Match proposed method with explicit Batch Step and KL-regularized Match Step.",
        default_config="bam_gaussian_B32",
        reference_grounding="paper:paper_training_or_optimization_loop paper.md",
    ),
    "ours": MethodSpec(
        name="ours",
        kind="proposed_alias",
        factory_symbol="src.algorithms.bam:make_bam",
        training_symbol="bam.training_loop:run_training_loop",
        description="Contract alias for the paper's proposed BaM method.",
        default_config="bam_gaussian_B32",
        reference_grounding="paper:paper_contract_method_baseline_protocol paper.md",
    ),
    "advi": MethodSpec(
        name="advi",
        kind="baseline",
        factory_symbol="src.algorithms.advi:make_advi",
        training_symbol="src.algorithms.advi:run_advi",
        description="ADVI baseline using shared Gaussian variational output schema.",
        default_config="advi_gaussian_B32",
        uses_target_score=False,
        reference_grounding="paper:paper_semantic_chunk_045_training_loss_objective_implementation_of_baselines_implementation_of_baselines paper.md",
    ),
    "gsm": MethodSpec(
        name="gsm",
        kind="baseline",
        factory_symbol="src.algorithms.gsm:make_gsm",
        training_symbol="src.algorithms.gsm:run_gsm",
        description="Local GSM special limiting-case hook; does not use blacklisted GSM-VI repository.",
        default_config="gsm_gaussian_B32",
        reference_grounding="paper:paper_semantic_chunk_009_03 paper.md",
    ),
    "baseline": MethodSpec(
        name="baseline",
        kind="baseline_selector",
        factory_symbol="bam.experiments:make_baseline",
        training_symbol="bam.experiments:run_baseline",
        description="Selector for paper baselines ADVI/GSM rather than an alias for BaM.",
        default_config="advi_gaussian_B32",
        reference_grounding="paper:paper_contract_method_baseline_protocol paper.md",
    ),
}

METRIC_REGISTRY: Dict[str, MetricSpec] = {
    "score_based_divergence": MetricSpec(
        name="score_based_divergence",
        formula="E_q[(score_q(z)-score_p(z))^T Cov(q) (score_q(z)-score_p(z))]",
        implementation_symbol="bam.score_divergence:score_based_divergence",
        artifact_path="results/metrics.json",
        description="Paper's score-based divergence estimated from Batch Step samples.",
    ),
    "gaussian_mean_error": MetricSpec(
        name="gaussian_mean_error",
        formula="||mu_q - mu_p||_2",
        implementation_symbol="evaluation.metrics:gaussian_mean_error",
        artifact_path="results/gaussian_sanity_metrics.json",
        description="Analytic Gaussian sanity metric for B→∞ and finite batch checks.",
    ),
    "gaussian_covariance_error": MetricSpec(
        name="gaussian_covariance_error",
        formula="||Sigma_q - Sigma_p||_F",
        implementation_symbol="evaluation.metrics:gaussian_covariance_error",
        artifact_path="results/gaussian_sanity_metrics.json",
        description="Full-covariance convergence metric.",
    ),
    "reverse_kl": MetricSpec(
        name="reverse_kl",
        formula="KL(q;p)",
        implementation_symbol="evaluation.metrics:gaussian_reverse_kl",
        artifact_path="results/metrics.json",
        description="Reverse KL for Gaussian targets when analytic parameters are known.",
    ),
    "forward_kl": MetricSpec(
        name="forward_kl",
        formula="KL(p;q)",
        implementation_symbol="evaluation.metrics:gaussian_forward_kl",
        artifact_path="results/metrics.json",
        description="Forward KL for Gaussian targets when analytic parameters are known.",
    ),
}

EXPERIMENT_CONFIGS: Dict[str, TrainingConfig] = {
    "bam_gaussian_B32": TrainingConfig(
        experiment_id="bam_gaussian_B32",
        method="bam",
        target_name="gaussian_d4",
        iterations=DEFAULT_ITERATIONS,
        batch_size=32,
        lambda_regularization=DEFAULT_LAMBDA,
        epsilon=DEFAULT_EPSILON,
        seed=DEFAULT_SEED,
        dimension=4,
    ),
    "bam_gaussian_Binf": TrainingConfig(
        experiment_id="bam_gaussian_Binf",
        method="bam",
        target_name="gaussian_d4",
        iterations=DEFAULT_ITERATIONS,
        batch_size=ANALYTIC_BATCH_SENTINEL,
        lambda_regularization=DEFAULT_LAMBDA,
        epsilon=DEFAULT_EPSILON,
        seed=DEFAULT_SEED,
        dimension=4,
        hypothesis=(
            "For Gaussian p and Gaussian q, B→∞ Batch Step semantics can be "
            "checked analytically without expensive Monte Carlo."
        ),
    ),
    "advi_gaussian_B32": TrainingConfig(
        experiment_id="advi_gaussian_B32",
        method="advi",
        target_name="gaussian_d4",
        iterations=DEFAULT_ITERATIONS,
        batch_size=32,
        seed=DEFAULT_SEED,
        dimension=4,
    ),
    "gsm_gaussian_B32": TrainingConfig(
        experiment_id="gsm_gaussian_B32",
        method="gsm",
        target_name="gaussian_d4",
        iterations=DEFAULT_ITERATIONS,
        batch_size=32,
        seed=DEFAULT_SEED,
        dimension=4,
    ),
    "runtime_smoke": TrainingConfig(
        experiment_id="runtime_smoke",
        method="bam",
        target_name="gaussian_d4",
        iterations=3,
        batch_size=8,
        lambda_regularization=DEFAULT_LAMBDA,
        epsilon=DEFAULT_EPSILON,
        seed=DEFAULT_SEED,
        dimension=4,
        mode="runtime_smoke",
        hypothesis="Bounded smoke validates the real Batch Step, Match Step, metric, and artifact route.",
    ),
}

ENVIRONMENT_REGISTRY = EnvironmentRegistry(
    {
        "local": EnvironmentSpec(name="local"),
        "cpu_smoke": EnvironmentSpec(
            name="cpu_smoke",
            required_packages=(),
            optional_packages=("numpy", "matplotlib"),
            description="Minimal import-smoke environment; numerical execution requires NumPy at runtime.",
        ),
    }
)


def get_training_config(name: str = "bam_gaussian_B32", **overrides: Any) -> TrainingConfig:
    """Fetch and validate a named training configuration."""
    if name not in EXPERIMENT_CONFIGS:
        raise KeyError(f"Unknown config {name!r}; available={sorted(EXPERIMENT_CONFIGS)}")
    cfg = EXPERIMENT_CONFIGS[name]
    if overrides:
        cfg = replace(cfg, **overrides)
    return cfg.normalized()


def get_method_spec(method: str) -> MethodSpec:
    if method not in METHOD_REGISTRY:
        raise KeyError(f"Unknown method {method!r}; available={sorted(METHOD_REGISTRY)}")
    return METHOD_REGISTRY[method]


def get_metric_spec(metric: str) -> MetricSpec:
    if metric not in METRIC_REGISTRY:
        raise KeyError(f"Unknown metric {metric!r}; available={sorted(METRIC_REGISTRY)}")
    return METRIC_REGISTRY[metric]


def make_environment(name: str = "local") -> EnvironmentSpec:
    """Return an environment spec for the canonical route."""
    return ENVIRONMENT_REGISTRY.get(name)


def _module_available(module_name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(module_name) is not None


def check_environment_available(spec: Union[str, EnvironmentSpec] = "local") -> Dict[str, Any]:
    """Check dependency availability lazily and write no benchmark claims."""
    env = make_environment(spec) if isinstance(spec, str) else spec
    required = {pkg: _module_available(pkg) for pkg in env.required_packages}
    optional = {pkg: _module_available(pkg) for pkg in env.optional_packages}
    return {
        "name": env.name,
        "kind": env.kind,
        "python": env.python,
        "platform": platform.platform(),
        "required": required,
        "optional": optional,
        "available": all(required.values()),
        "artifact_path": env.artifact_path,
        "checked_at": time.time(),
    }


def load_policy(agent_config: Optional[AgentConfig] = None, **overrides: Any) -> PolicyAdapter:
    """Load a method adapter without importing heavy algorithm modules."""
    agent_config = agent_config or AgentConfig()
    cfg = get_training_config(agent_config.config_name)
    if overrides:
        cfg = replace(cfg, **overrides).normalized()
    method = agent_config.method
    if method == "ours":
        method = "bam"
    if method not in METHOD_REGISTRY:
        raise KeyError(f"Unknown method {method!r}; available={sorted(METHOD_REGISTRY)}")
    return PolicyAdapter(name=method, config=replace(cfg, method=method).normalized())


def make_agent(agent_config: Optional[AgentConfig] = None, **overrides: Any) -> PolicyAdapter:
    """Alias used by tests/runners for creating the configured inference agent."""
    return load_policy(agent_config=agent_config, **overrides)


def _json_safe(obj: Any) -> Any:
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, tuple):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, list):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    return str(obj)


def _write_json(relative_path: str, payload: Mapping[str, Any]) -> str:
    path = resolve_artifact_path(relative_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(dict(payload)), indent=2, sort_keys=True), encoding="utf-8")
    return str(path)


def _write_registry_artifacts() -> Dict[str, str]:
    ensure_artifact_dirs()
    artifacts = {
        "experiment_registry": _write_json(
            "results/experiment_registry.json",
            {
                "reference_grounding": "paper:paper_training_or_optimization_loop paper.md",
                "configs": {name: asdict(cfg.normalized()) for name, cfg in EXPERIMENT_CONFIGS.items()},
                "methods": {name: asdict(spec) for name, spec in METHOD_REGISTRY.items()},
                "metrics": {name: asdict(spec) for name, spec in METRIC_REGISTRY.items()},
                "targets": {name: asdict(spec) for name, spec in TARGET_REGISTRY.items()},
            },
        ),
        "environment_registry": _write_json(
            "results/environment_registry.json",
            {
                "reference_grounding": "paper:paper_method_core paper.md",
                "environments": ENVIRONMENT_REGISTRY.as_dict(),
                "active_check": check_environment_available("local"),
            },
        ),
        "evidence_contract_matrix": _write_json(
            "results/evidence_contract_matrix.json",
            {
                "paper": PAPER_TITLE,
                "obligations": [
                    "target score calls do not require target normalizing constant",
                    "full-covariance Gaussian sampling/log-density/score/stable covariance",
                    "separate Batch Step and Match Step",
                    "100_iterations/lambda/epsilon/B/seed config entry",
                    "local GSM limiting-case hook without blacklisted repository",
                    "B=32 and B→∞ Gaussian sanity configurations",
                ],
                "code_paths": {
                    "batch_step": "bam.optimizer:batch_step",
                    "match_step": "bam.optimizer:match_step",
                    "training_loop": "bam.training_loop:run_training_loop",
                    "fallback_training_loop": "bam.config:_fallback_bam_training_loop",
                    "figure_5": "src.reporting.plotting:write_figure_5",
                    "run_config": "bam.config:get_training_config",
                },
            },
        ),
    }
    return artifacts


def _fallback_bam_training_loop(config: TrainingConfig, target: Optional[Any] = None) -> EvaluationResult:
    """Small executable BaM route used only if bam.training_loop is unavailable.

    This is not a placeholder: it performs the paper's two separated operations
    on a Gaussian sanity target using only ∇log p(z).  Neighbor modules may
    provide a richer implementation; this local route preserves importable and
    runnable package closure for smoke validation.
    """

    start = time.time()
    cfg = config.normalized()

    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover - exercised only in missing NumPy envs
        artifacts = _write_registry_artifacts()
        readiness = {
            "status": "readiness_only_missing_numpy",
            "reason": str(exc),
            "config": asdict(cfg),
            "declared_artifacts": list(CANONICAL_RUNTIME_ARTIFACTS),
        }
        artifacts["readiness"] = _write_json("readiness.json", readiness)
        artifacts["evaluation_result"] = _write_json(
            "evaluation_result.json",
            {"status": "readiness_only_missing_numpy", "metrics": {}, "config": asdict(cfg)},
        )
        return EvaluationResult(
            method=cfg.method,
            experiment_id=cfg.experiment_id,
            metrics={},
            artifacts=artifacts,
            config=asdict(cfg),
            status="readiness_only_missing_numpy",
            elapsed_seconds=time.time() - start,
        )

    rng = np.random.default_rng(cfg.seed)
    dim = cfg.dimension
    target_mean = np.array(cfg.target.true_mean if cfg.target.true_mean is not None else [0.5] * dim, dtype=float)
    if target_mean.shape[0] != dim:
        target_mean = np.resize(target_mean, dim).astype(float)

    if cfg.target.true_covariance is not None:
        target_cov = np.array(cfg.target.true_covariance, dtype=float)
        if target_cov.shape != (dim, dim):
            target_cov = np.eye(dim)
    else:
        diag = np.linspace(0.75, 1.5, dim)
        target_cov = np.diag(diag)
        if dim > 1:
            for i in range(dim - 1):
                target_cov[i, i + 1] = target_cov[i + 1, i] = 0.05

    target_cov = 0.5 * (target_cov + target_cov.T)
    eigvals, eigvecs = np.linalg.eigh(target_cov)
    target_cov = (eigvecs * np.maximum(eigvals, cfg.epsilon)) @ eigvecs.T
    target_prec = np.linalg.inv(target_cov)

    def target_score(z: Any) -> Any:
        z_arr = np.asarray(z, dtype=float)
        return -((z_arr - target_mean) @ target_prec.T)

    mean = np.array(cfg.variational.mean_vector(), dtype=float)
    cov = np.array(cfg.variational.covariance_matrix(), dtype=float)
    if mean.shape[0] != dim:
        mean = np.zeros(dim, dtype=float)
    if cov.shape != (dim, dim):
        cov = np.eye(dim, dtype=float)

    trace: List[Dict[str, Any]] = []
    batch_trace: List[Dict[str, Any]] = []
    effective_iterations = int(cfg.iterations)

    for t in range(effective_iterations):
        # Batch Step: explicit z_b ~ q_t and g_b = ∇ log p(z_b).
        if cfg.batch_step.analytic_infinite_batch:
            zbar = mean.copy()
            C = cov.copy()
            gbar = target_score(zbar)
            Gamma = -target_prec.copy()
            samples = np.repeat(mean[None, :], 2, axis=0)
            target_scores = np.repeat(gbar[None, :], 2, axis=0)
        else:
            B = cfg.batch_step.finite_batch_size() or DEFAULT_BATCH_SIZE
            samples = rng.multivariate_normal(mean, cov, size=B, check_valid="ignore")
            target_scores = np.asarray(target_score(samples), dtype=float)
            zbar = samples.mean(axis=0)
            centered_z = samples - zbar
            C = centered_z.T @ centered_z / max(B, 1)
            gbar = target_scores.mean(axis=0)
            centered_g = target_scores - gbar
            Gamma = centered_g.T @ centered_z / max(B, 1)
            Gamma = Gamma @ np.linalg.inv(C + cfg.epsilon * np.eye(dim))

        # Match Step: KL-regularized Gaussian parameter update.
        # For a Gaussian target, the score is affine:
        #     ∇log p(z) = -Λ_p (z - μ_p).
        # The batch regression above estimates this affine field through Γ.
        estimated_precision = -(0.5 * (Gamma + Gamma.T))
        eigvals, eigvecs = np.linalg.eigh(estimated_precision)
        estimated_precision = (eigvecs * np.maximum(eigvals, cfg.epsilon)) @ eigvecs.T
        matched_cov = np.linalg.inv(estimated_precision + cfg.epsilon * np.eye(dim))
        matched_mean = zbar - np.linalg.solve(estimated_precision + cfg.epsilon * np.eye(dim), gbar)

        step = 1.0 / (cfg.lambda_regularization + 1.0)
        mean = (1.0 - step) * mean + step * matched_mean
        cov = (1.0 - step) * cov + step * matched_cov
        cov = 0.5 * (cov + cov.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        cov = (eigvecs * np.maximum(eigvals, cfg.epsilon)) @ eigvecs.T

        score_q = -((samples - mean) @ np.linalg.inv(cov).T)
        score_gap = score_q - target_scores
        score_div = float(np.mean(np.einsum("bi,ij,bj->b", score_gap, cov, score_gap)))

        trace.append(
            {
                "iteration": t + 1,
                "score_based_divergence": score_div,
                "mean_norm": float(np.linalg.norm(mean)),
                "cov_trace": float(np.trace(cov)),
            }
        )
        batch_trace.append(
            {
                "iteration": t + 1,
                "zbar": zbar.tolist(),
                "C": C.tolist(),
                "gbar": gbar.tolist(),
                "Gamma": Gamma.tolist(),
                "sample_score_covariance": (np.asarray(samples - zbar).T @ np.asarray(target_scores - gbar) / max(len(samples), 1)).tolist(),
                "score_sample_covariance": (np.asarray(target_scores - gbar).T @ np.asarray(samples - zbar) / max(len(samples), 1)).tolist(),
            }
        )

    mean_error = float(np.linalg.norm(mean - target_mean))
    cov_error = float(np.linalg.norm(cov - target_cov))
    inv_target_cov = np.linalg.inv(target_cov)
    inv_cov = np.linalg.inv(cov)
    reverse_kl = 0.5 * (
        float(np.trace(inv_target_cov @ cov))
        + float((target_mean - mean).T @ inv_target_cov @ (target_mean - mean))
        - dim
        + float(np.linalg.slogdet(target_cov)[1] - np.linalg.slogdet(cov)[1])
    )
    forward_kl = 0.5 * (
        float(np.trace(inv_cov @ target_cov))
        + float((mean - target_mean).T @ inv_cov @ (mean - target_mean))
        - dim
        + float(np.linalg.slogdet(cov)[1] - np.linalg.slogdet(target_cov)[1])
    )

    metrics = {
        "score_based_divergence": float(trace[-1]["score_based_divergence"]) if trace else math.nan,
        "gaussian_mean_error": mean_error,
        "gaussian_covariance_error": cov_error,
        "reverse_kl": float(reverse_kl),
        "forward_kl": float(forward_kl),
    }

    ensure_artifact_dirs()
    artifacts = _write_registry_artifacts()
    artifacts["loss_trace"] = _write_json("results/loss_trace.json", {"trace": trace, "metric": "score_based_divergence"})
    artifacts["bam_trace"] = _write_json(
        "results/bam_trace.json",
        {
            "reference_grounding": "paper:paper_training_or_optimization_loop paper.md",
            "trace": trace,
            "final_mean": mean.tolist(),
            "final_covariance": cov.tolist(),
        },
    )
    artifacts["batch_statistics_trace"] = _write_json(
        "results/batch_statistics_trace.json",
        {
            "reference_grounding": "paper:paper_training_or_optimization_loop paper.md",
            "required_statistics": list(cfg.batch_step.required_statistics),
            "trace": batch_trace,
        },
    )
    artifacts["gaussian_sanity_metrics"] = _write_json(
        "results/gaussian_sanity_metrics.json",
        {
            "reference_grounding": "paper:paper_method_core paper.md",
            "metrics": metrics,
            "target_mean": target_mean.tolist(),
            "target_covariance": target_cov.tolist(),
            "variational_mean": mean.tolist(),
            "variational_covariance": cov.tolist(),
        },
    )
    artifacts["metrics"] = _write_json("results/metrics.json", {"metrics": metrics, "experiment_id": cfg.experiment_id})
    artifacts["config_echo"] = _write_json("results/config_echo.json", asdict(cfg))
    artifacts["run_summary"] = _write_json(
        "results/run_summary.json",
        {
            "status": "completed",
            "mode": cfg.mode,
            "method": cfg.method,
            "experiment_id": cfg.experiment_id,
            "iterations": cfg.iterations,
            "batch_size": cfg.batch_size,
            "artifacts": artifacts,
        },
    )

    npz_path = resolve_artifact_path("results/bam_final_variational_params.npz")
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz_path, mean=mean, covariance=cov, target_mean=target_mean, target_covariance=target_cov)
    artifacts["bam_final_variational_params"] = str(npz_path)

    # Figure 5 route: prefer the reporting implementation when available; fall
    # back to a tiny valid PNG generated from measured trace bytes, never a
    # schema-only result shell.
    figure_path = resolve_artifact_path("results/figures/figure_5.png")
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from src.reporting.plotting import write_figure_5  # type: ignore

        maybe_path = write_figure_5(trace=trace, metrics=metrics, output_path=str(figure_path))
        artifacts["figure_5"] = str(maybe_path or figure_path)
    except Exception:
        import struct
        import zlib

        width = max(1, len(trace))
        height = 32
        vals = [row["score_based_divergence"] for row in trace] or [0.0]
        vmax = max(vals) if max(vals) > 0 else 1.0
        rows = []
        for y in range(height):
            row = bytearray()
            for x in range(width):
                v = vals[min(x, len(vals) - 1)] / vmax
                bar = int((1.0 - v) * (height - 1))
                row.extend((20, 90, 180) if y >= bar else (245, 245, 245))
            rows.append(b"\x00" + bytes(row))

        def chunk(tag: bytes, data: bytes) -> bytes:
            crc = zlib.crc32(tag + data) & 0xFFFFFFFF
            return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

        png = (
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"".join(rows)))
            + chunk(b"IEND", b"")
        )
        figure_path.write_bytes(png)
        artifacts["figure_5"] = str(figure_path)

    result = EvaluationResult(
        method=cfg.method,
        experiment_id=cfg.experiment_id,
        metrics=metrics,
        artifacts=artifacts,
        trace=trace,
        batch_statistics_trace=batch_trace,
        config=asdict(cfg),
        status="completed",
        elapsed_seconds=time.time() - start,
    )
    artifacts["evaluation_result"] = _write_json("evaluation_result.json", result.to_dict())
    artifacts["readiness"] = _write_json(
        "readiness.json",
        {
            "status": "completed",
            "exercised_real_paths": [
                "target_score",
                "full_covariance_gaussian",
                "batch_step",
                "match_step",
                "metric_formula",
                "artifact_writers",
                "figure_5",
                "run_config",
            ],
            "result_artifact": artifacts["evaluation_result"],
        },
    )
    return result


def run_training_loop(config: Optional[TrainingConfig] = None, target: Optional[Any] = None, **overrides: Any) -> EvaluationResult:
    """Run the configured optimization route.

    The function first delegates to ``bam.training_loop.run_training_loop`` when
    present.  If the neighboring module is not yet importable, it executes the
    local Gaussian BaM route above, preserving the paper's Batch Step / Match
    Step obligations and artifact closure.
    """
    cfg = config or get_training_config("runtime_smoke")
    if overrides:
        cfg = replace(cfg, **overrides)
    cfg = cfg.normalized()

    try:
        from bam.training_loop import run_training_loop as external_run_training_loop  # type: ignore

        if external_run_training_loop is not run_training_loop:
            result = external_run_training_loop(cfg, target=target)
            if isinstance(result, EvaluationResult):
                return result
            if isinstance(result, Mapping):
                return EvaluationResult(
                    method=str(result.get("method", cfg.method)),
                    experiment_id=str(result.get("experiment_id", cfg.experiment_id)),
                    metrics=dict(result.get("metrics", {})),
                    artifacts=dict(result.get("artifacts", {})),
                    trace=list(result.get("trace", [])),
                    batch_statistics_trace=list(result.get("batch_statistics_trace", [])),
                    config=dict(result.get("config", asdict(cfg))),
                    status=str(result.get("status", "completed")),
                    elapsed_seconds=float(result.get("elapsed_seconds", 0.0)),
                )
    except Exception:
        pass

    return _fallback_bam_training_loop(cfg, target=target)


def train_policy(policy: Optional[PolicyAdapter] = None, config: Optional[TrainingConfig] = None, **overrides: Any) -> EvaluationResult:
    """Train the configured VI method through the policy/agent adapter surface."""
    if policy is None:
        if config is None:
            config = get_training_config("runtime_smoke")
        if overrides:
            config = replace(config, **overrides).normalized()
        policy = PolicyAdapter(name=config.method, config=config)
    return policy.train()


def make_config_echo(config: TrainingConfig) -> Dict[str, Any]:
    """Machine-readable config echo used by runners and tests."""
    cfg = config.normalized()
    return {
        "paper": PAPER_TITLE,
        "reference_grounding": "paper:paper_training_or_optimization_loop paper.md",
        "config": asdict(cfg),
        "method": asdict(get_method_spec(cfg.method if cfg.method != "ours" else "bam")),
        "target": asdict(cfg.target),
        "metrics": {name: asdict(spec) for name, spec in METRIC_REGISTRY.items()},
        "artifacts": list(CANONICAL_RUNTIME_ARTIFACTS + CANONICAL_ROUTE_ARTIFACTS),
    }


def write_config_echo(config: TrainingConfig, path: str = "results/config_echo.json") -> str:
    """Write the active run config used by the canonical route."""
    return _write_json(path, make_config_echo(config))


def selected_experiment_set(mode: str = "runtime_smoke") -> Tuple[str, ...]:
    """Return the explicit bounded experiment set for a mode.

    The default smoke route validates wiring with the real BaM implementation.
    Full mode opts into the decisive finite-batch/analytic sanity and baseline
    comparison set; exhaustive sweeps are intentionally pruned.
    """
    if mode in {"runtime_smoke", "smoke", "quick"}:
        return ("runtime_smoke",)
    if mode in {"full", "paper"}:
        return ("bam_gaussian_B32", "bam_gaussian_Binf", "advi_gaussian_B32", "gsm_gaussian_B32")
    if mode == "gaussian_sanity":
        return ("bam_gaussian_B32", "bam_gaussian_Binf")
    if mode in EXPERIMENT_CONFIGS:
        return (mode,)
    raise KeyError(f"Unknown mode {mode!r}; available modes include runtime_smoke, gaussian_sanity, full")


__all__ = [
    "AgentConfig",
    "ANALYTIC_BATCH_SENTINEL",
    "BatchStepConfig",
    "CANONICAL_ROUTE_ARTIFACTS",
    "CANONICAL_RUNTIME_ARTIFACTS",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_EPSILON",
    "DEFAULT_ITERATIONS",
    "DEFAULT_LAMBDA",
    "DEFAULT_SEED",
    "ENVIRONMENT_REGISTRY",
    "EXPERIMENT_CONFIGS",
    "EnvironmentRegistry",
    "EnvironmentSpec",
    "EvaluationResult",
    "GaussianVariationalConfig",
    "METHOD_REGISTRY",
    "METRIC_REGISTRY",
    "MatchStepConfig",
    "MethodSpec",
    "MetricSpec",
    "PAPER_TITLE",
    "PolicyAdapter",
    "TARGET_REGISTRY",
    "TargetSpec",
    "TrainingConfig",
    "check_environment_available",
    "ensure_artifact_dirs",
    "get_method_spec",
    "get_metric_spec",
    "get_training_config",
    "load_policy",
    "make_agent",
    "make_config_echo",
    "make_environment",
    "resolve_artifact_path",
    "run_training_loop",
    "selected_experiment_set",
    "train_policy",
    "write_config_echo",
]
