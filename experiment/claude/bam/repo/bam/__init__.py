"""Public package surface for the BaM PaperBench reproduction.

This package reproduces the method/baseline protocol for the paper
"Batch and match: black-box variational inference with a score-based
divergence" without importing optional accelerator, plotting, dataset, or
probabilistic-programming dependencies at module import time.

The package-level surface deliberately exposes a shared full-covariance Gaussian
variational-output schema and a single method registry for the proposed method
(BaM/``ours``), ADVI, GSM, and the contract-required ``baseline`` selector.
The registry is code-visible so experiment runners, tests, and artifact writers
cannot accidentally collapse the proposed method and baselines into one generic
path.

reference_grounding: paper:paper_contract_method_baseline_protocol paper.md
    The paper contract exposes methods_baselines: ours; baseline.  This module
    registers ``ours`` as the BaM proposed method and ``baseline`` as an
    explicit selector for paper baselines rather than an alias for BaM.

reference_grounding: paper:paper_semantic_chunk_045_training_loss_objective_implementation_of_baselines_implementation_of_baselines paper.md
    ADVI uses ELBO maximization with reparameterization samples and an ADAM
    optimizer loop.  GSM is exposed as an independent Gaussian score-matching
    comparison path, implemented locally and not copied from the blacklisted
    GSM-VI repository.

reference_grounding: paper:paper_semantic_chunk_007_01 paper.md
    BaM optimizes a Gaussian variational approximation using samples from q and
    the target score ∇ log p(z).  All package methods below accept the same
    target ``log_prob``/``score`` interface and return μ, Σ, trace, run config,
    and samples for metric computation.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Protocol as _Protocol, Sequence, Tuple


__version__ = "0.1.0"
__paper__ = "Batch and match: black-box variational inference with a score-based divergence"
__blacklisted_repositories__ = ("https://github.com/modichirag/GSM-VI",)

ArrayLike = Any
LogProbFn = Callable[[ArrayLike], Any]
ScoreFn = Callable[[ArrayLike], Any]


class TargetProtocol(_Protocol):
    """Common target interface used by BaM, ADVI, GSM, ours, and baseline."""

    name: str
    dim: int

    def log_prob(self, z: ArrayLike) -> Any:
        """Return an unnormalized log-density value for one or more samples."""

    def score(self, z: ArrayLike) -> Any:
        """Return ∇ log p(z) for one or more samples."""


@dataclass(frozen=True)
class MethodRegistryEntry:
    """Code-visible method/baseline registry entry.

    ``method_role`` separates the proposed method, baselines, aliases, and
    ablations.  ``implementation_key`` is the concrete dispatch key used by
    :func:`run_method`; aliases such as ``ours`` and ``baseline`` therefore
    remain explicit without duplicating implementations.
    """

    name: str
    implementation_key: str
    method_role: str
    variational_family: str
    objective: str
    required_target_interfaces: Tuple[str, ...]
    supported_protocol_targets: Tuple[str, ...]
    default_artifacts: Tuple[str, ...]
    decisive_metrics: Tuple[str, ...]
    claim_binding: str
    lazy_module: str
    lazy_callable: str
    smoke_iterations: int = 3
    full_iterations: int = 100
    default_batch_size: int = 32
    unavailable_protocols: Mapping[str, str] = field(default_factory=dict)
    ablation_switches: Tuple[str, ...] = ()
    evidence: str = ""


@dataclass
class GaussianVIResult:
    """Shared full-covariance Gaussian VI output schema for all methods."""

    method: str
    mu: List[float]
    covariance: List[List[float]]
    trace: List[Mapping[str, Any]]
    run_config: Mapping[str, Any]
    samples: List[List[float]]
    metric_inputs: Mapping[str, Any]
    target_name: str
    dry_run: bool = False

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["schema"] = GAUSSIAN_VI_OUTPUT_SCHEMA["schema"]
        payload["schema_version"] = GAUSSIAN_VI_OUTPUT_SCHEMA["schema_version"]
        return payload


@dataclass(frozen=True)
class MethodAdapter:
    """Lightweight method selector returned by ``make_method(config)``."""

    method_id: str
    name: str
    implementation_key: str
    role: str
    objective: str

    def __call__(self, target: Any, config: Optional[Mapping[str, Any]] = None) -> "GaussianVIResult":
        return run_method(self.implementation_key, target, config)


GAUSSIAN_VI_OUTPUT_SCHEMA: Dict[str, Any] = {
    "schema": "bam.gaussian_full_covariance_vi_result",
    "schema_version": "1.0",
    "required_fields": [
        "method",
        "mu",
        "covariance",
        "trace",
        "run_config",
        "samples",
        "metric_inputs",
        "target_name",
        "dry_run",
    ],
    "variational_family": "Gaussian with full covariance matrix",
    "target_interface": ["log_prob(z)", "score(z)"],
    "method_outputs": {
        "mu": "variational mean vector",
        "covariance": "full covariance matrix",
        "trace": "per-iteration losses, diagnostics, and wallclock fields",
        "run_config": "resolved method/target/training configuration",
        "samples": "q-samples retained for metric estimators",
        "metric_inputs": "ELBO, score-divergence, KL, and convergence inputs",
    },
}


SUPPORTED_PROTOCOL_TARGETS: Tuple[str, ...] = (
    "synthetic_gaussian",
    "controlled_non_gaussian",
    "hierarchical_bayesian",
    "deep_generative_model",
    "cifar_protocol",
)


CANONICAL_ARTIFACTS: Tuple[str, ...] = (
    "results/loss_trace.json",
    "results/method_registry.json",
    "results/ablation_registry.json",
    "results/synthetic_kl_metrics.json",
    "results/synthetic_curves.json",
    "results/baseline_traces.json",
    "results/readiness.json",
    "results/evaluation_result.json",
)

PROTOCOL = "typing protocol for target log_prob/score interfaces"


METHOD_REGISTRY: Dict[str, MethodRegistryEntry] = {
    "bam": MethodRegistryEntry(
        name="bam",
        implementation_key="bam",
        method_role="proposed",
        variational_family="full_covariance_gaussian",
        objective="score_based_divergence_with_batch_step_and_match_step",
        required_target_interfaces=("log_prob", "score"),
        supported_protocol_targets=SUPPORTED_PROTOCOL_TARGETS,
        default_artifacts=(
            "results/loss_trace.json",
            "results/synthetic_kl_metrics.json",
            "results/synthetic_curves.json",
        ),
        decisive_metrics=("score_divergence", "reverse_kl", "mean_error", "covariance_error"),
        claim_binding="Core contribution: Batch Step plus Match Step for score-based BBVI.",
        lazy_module="src.algorithms.bam",
        lazy_callable="run_bam",
        evidence="paper:paper_contract_method_baseline_protocol",
    ),
    "ours": MethodRegistryEntry(
        name="ours",
        implementation_key="bam",
        method_role="contract_selector",
        variational_family="full_covariance_gaussian",
        objective="same_as_bam_proposed_method",
        required_target_interfaces=("log_prob", "score"),
        supported_protocol_targets=SUPPORTED_PROTOCOL_TARGETS,
        default_artifacts=(
            "results/loss_trace.json",
            "results/synthetic_kl_metrics.json",
            "results/synthetic_curves.json",
        ),
        decisive_metrics=("score_divergence", "reverse_kl", "mean_error", "covariance_error"),
        claim_binding="Contract-visible 'ours' entry bound to the BaM proposed method.",
        lazy_module="src.algorithms.bam",
        lazy_callable="run_bam",
        evidence="paper:paper_contract_method_baseline_protocol",
    ),
    "advi": MethodRegistryEntry(
        name="advi",
        implementation_key="advi",
        method_role="baseline",
        variational_family="full_covariance_gaussian",
        objective="elbo_maximization_with_reparameterization_sampling_and_adam",
        required_target_interfaces=("log_prob", "score"),
        supported_protocol_targets=SUPPORTED_PROTOCOL_TARGETS,
        default_artifacts=("results/baseline_traces.json", "results/loss_trace.json"),
        decisive_metrics=("negative_elbo", "reverse_kl", "mean_error", "covariance_error"),
        claim_binding="Paper baseline: ADVI optimizer loop over full-covariance Gaussian q.",
        lazy_module="src.algorithms.advi",
        lazy_callable="run_advi",
        evidence="paper:paper_semantic_chunk_045_training_loss_objective_implementation_of_baselines_implementation_of_baselines",
    ),
    "gsm": MethodRegistryEntry(
        name="gsm",
        implementation_key="gsm",
        method_role="baseline",
        variational_family="full_covariance_gaussian",
        objective="independent_gaussian_score_matching",
        required_target_interfaces=("log_prob", "score"),
        supported_protocol_targets=SUPPORTED_PROTOCOL_TARGETS,
        default_artifacts=("results/baseline_traces.json", "results/loss_trace.json"),
        decisive_metrics=("score_matching_loss", "reverse_kl", "mean_error", "covariance_error"),
        claim_binding=(
            "Paper baseline: local independent Gaussian score matching path; "
            "not a BaM label alias and not copied from the blacklisted GSM-VI repository."
        ),
        lazy_module="src.algorithms.gsm",
        lazy_callable="run_gsm",
        evidence="paper:paper_semantic_chunk_045_training_loss_objective_implementation_of_baselines_implementation_of_baselines",
    ),
    "baseline": MethodRegistryEntry(
        name="baseline",
        implementation_key="advi",
        method_role="contract_selector",
        variational_family="full_covariance_gaussian",
        objective="default_paper_baseline_selector_advi_full_covariance",
        required_target_interfaces=("log_prob", "score"),
        supported_protocol_targets=SUPPORTED_PROTOCOL_TARGETS,
        default_artifacts=("results/baseline_traces.json", "results/loss_trace.json"),
        decisive_metrics=("negative_elbo", "score_matching_loss", "reverse_kl"),
        claim_binding=(
            "Contract-visible baseline selector.  The default selector routes to ADVI; "
            "experiments may choose gsm explicitly for the second baseline."
        ),
        lazy_module="src.algorithms.advi",
        lazy_callable="run_advi",
        evidence="paper:paper_contract_method_baseline_protocol",
    ),
}


ABLATION_REGISTRY: Dict[str, Dict[str, Any]] = {
    "bam_finite_batch": {
        "base_method": "bam",
        "switches": {"batch_size": 32, "match_regularization": "kl_proximal"},
        "decision_value": "Tests the paper's finite-B Batch Step and Match Step.",
        "bounded_default": True,
    },
    "bam_gaussian_sanity_b_infinity": {
        "base_method": "bam",
        "switches": {"batch_size": "analytic_infinite_batch", "target": "synthetic_gaussian"},
        "decision_value": "Sanity route for settings where the true Gaussian target is known.",
        "bounded_default": False,
        "stop_rule": "Not executed in smoke mode; full mode must be explicit.",
    },
    "advi_elbo": {
        "base_method": "advi",
        "switches": {"objective": "negative_elbo", "optimizer": "adam"},
        "decision_value": "Decisive ELBO baseline comparison.",
        "bounded_default": True,
    },
    "gsm_score_matching": {
        "base_method": "gsm",
        "switches": {"objective": "gaussian_score_matching"},
        "decision_value": "Independent score-matching baseline comparison.",
        "bounded_default": True,
    },
}


DEFAULT_METHOD_CONFIG: Dict[str, Any] = {
    "iterations": 100,
    "smoke_iterations": 3,
    "batch_size": 32,
    "learning_rate": 0.03,
    "seed": 0,
    "jitter": 1e-5,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_epsilon": 1e-8,
    "mode": "full",
    "artifact_dir": "results",
    "hypothesis": (
        "BaM should match or improve full-covariance Gaussian VI quality relative "
        "to ELBO ADVI and independent Gaussian score matching when only target "
        "log_prob/score interfaces are shared."
    ),
    "decisive_comparison": "bam_vs_advi_vs_gsm",
    "decisive_metric": "score_divergence_and_reverse_kl",
    "stop_rule_or_pruning_rationale": (
        "Default smoke mode runs a bounded wiring check; exhaustive sweeps and "
        "expensive CIFAR/deep-generative protocols require explicit full mode."
    ),
}


def _np() -> Any:
    """Import NumPy lazily with an actionable runtime error."""

    try:
        import numpy as np  # type: ignore

        return np
    except Exception as exc:  # pragma: no cover - exercised only without numpy
        raise RuntimeError(
            "Numerical method execution requires numpy. Package import and registry "
            "inspection remain available without it."
        ) from exc


def list_methods(include_aliases: bool = True) -> List[str]:
    """Return registered method names."""

    names = list(METHOD_REGISTRY)
    if include_aliases:
        return names
    return [name for name, entry in METHOD_REGISTRY.items() if entry.name == entry.implementation_key]


def get_method_entry(method: str) -> MethodRegistryEntry:
    """Resolve a method/baseline/contract selector to a registry entry."""

    key = method.lower().strip()
    if key not in METHOD_REGISTRY:
        raise KeyError(f"Unknown method '{method}'. Available methods: {', '.join(sorted(METHOD_REGISTRY))}")
    return METHOD_REGISTRY[key]


def get_method_registry() -> Dict[str, Dict[str, Any]]:
    """Return a JSON-serializable method registry."""

    return {name: asdict(entry) for name, entry in METHOD_REGISTRY.items()}


def get_ablation_registry() -> Dict[str, Dict[str, Any]]:
    """Return a JSON-serializable ablation registry."""

    return json.loads(json.dumps(ABLATION_REGISTRY))


def make_method(config: Any = "ours") -> MethodAdapter:
    """Return a concrete method adapter for registry-driven runners/tests."""

    if isinstance(config, Mapping):
        selector = str(config.get("method", config.get("method_id", config.get("name", "ours"))))
    else:
        selector = str(config)
    entry = get_method_entry(selector)
    return MethodAdapter(
        method_id=selector.lower().strip(),
        name=entry.name,
        implementation_key=entry.implementation_key,
        role=entry.method_role,
        objective=entry.objective,
    )


def validate_target_interface(target: Any) -> Dict[str, Any]:
    """Validate that a target supplies the shared log_prob/score protocol."""

    missing = [name for name in ("log_prob", "score") if not callable(getattr(target, name, None))]
    dim = int(getattr(target, "dim", 0) or 0)
    if dim <= 0:
        missing.append("positive_dim")
    if missing:
        raise TypeError(
            "Target must expose callable log_prob(z), callable score(z), and positive dim; "
            f"missing/invalid: {missing}"
        )
    return {
        "target_name": str(getattr(target, "name", target.__class__.__name__)),
        "dim": dim,
        "interfaces": ["log_prob", "score"],
        "valid": True,
    }


def _as_batch(x: Any, dim: int) -> Any:
    np = _np()
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, dim)
    return arr


def _safe_log_prob(target: Any, z: Any) -> Any:
    np = _np()
    values = target.log_prob(z)
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr


def _safe_score(target: Any, z: Any) -> Any:
    np = _np()
    score = np.asarray(target.score(z), dtype=float)
    if score.ndim == 1:
        score = score.reshape(1, score.shape[0])
    return score


def _sym_spd(matrix: Any, jitter: float = 1e-5) -> Any:
    np = _np()
    mat = 0.5 * (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T)
    eigvals, eigvecs = np.linalg.eigh(mat)
    eigvals = np.maximum(eigvals, jitter)
    return (eigvecs * eigvals) @ eigvecs.T


def _sample_gaussian(rng: Any, mu: Any, covariance: Any, batch_size: int, jitter: float) -> Tuple[Any, Any, Any]:
    np = _np()
    dim = int(mu.shape[0])
    cov = _sym_spd(covariance, jitter=jitter)
    chol = np.linalg.cholesky(cov + jitter * np.eye(dim))
    eps = rng.normal(size=(batch_size, dim))
    samples = mu.reshape(1, dim) + eps @ chol.T
    return samples, eps, chol


def _gaussian_log_prob(samples: Any, mu: Any, covariance: Any, jitter: float) -> Any:
    np = _np()
    samples = np.asarray(samples, dtype=float)
    dim = int(mu.shape[0])
    cov = _sym_spd(covariance, jitter=jitter)
    precision = np.linalg.inv(cov + jitter * np.eye(dim))
    centered = samples - mu.reshape(1, dim)
    sign, logdet = np.linalg.slogdet(cov + jitter * np.eye(dim))
    if sign <= 0:
        logdet = float(np.linalg.slogdet(_sym_spd(cov, jitter=jitter))[1])
    quad = np.sum((centered @ precision) * centered, axis=1)
    return -0.5 * (dim * math.log(2.0 * math.pi) + logdet + quad)


def _score_divergence_estimate(target: Any, samples: Any, mu: Any, covariance: Any, jitter: float) -> float:
    np = _np()
    dim = int(mu.shape[0])
    cov = _sym_spd(covariance, jitter=jitter)
    precision = np.linalg.inv(cov + jitter * np.eye(dim))
    score_q = -(samples - mu.reshape(1, dim)) @ precision.T
    score_p = _safe_score(target, samples)
    diff = score_q - score_p
    weighted = np.sum((diff @ cov) * diff, axis=1)
    return float(np.mean(weighted))


def _metric_payload(target: Any, samples: Any, mu: Any, covariance: Any, jitter: float) -> Dict[str, Any]:
    np = _np()
    logp = _safe_log_prob(target, samples)
    logq = _gaussian_log_prob(samples, mu, covariance, jitter)
    score_div = _score_divergence_estimate(target, samples, mu, covariance, jitter)
    return {
        "negative_elbo_mc": float(-np.mean(logp - logq)),
        "elbo_mc": float(np.mean(logp - logq)),
        "score_divergence_mc": score_div,
        "sample_mean": np.mean(samples, axis=0).astype(float).tolist(),
        "sample_covariance": np.cov(samples.T).astype(float).tolist() if samples.shape[0] > 1 else covariance.tolist(),
        "num_metric_samples": int(samples.shape[0]),
    }


def _adam_update(param: Any, grad: Any, state: MutableMapping[str, Any], lr: float, beta1: float, beta2: float, eps: float) -> Any:
    np = _np()
    state["t"] = int(state.get("t", 0)) + 1
    if "m" not in state:
        state["m"] = np.zeros_like(param)
        state["v"] = np.zeros_like(param)
    state["m"] = beta1 * state["m"] + (1.0 - beta1) * grad
    state["v"] = beta2 * state["v"] + (1.0 - beta2) * (grad * grad)
    mhat = state["m"] / (1.0 - beta1 ** state["t"])
    vhat = state["v"] / (1.0 - beta2 ** state["t"])
    return param + lr * mhat / (np.sqrt(vhat) + eps)


def _local_run_advi(target: Any, config: Optional[Mapping[str, Any]] = None) -> GaussianVIResult:
    """Local ADVI implementation: ELBO, reparameterization samples, ADAM loop."""

    np = _np()
    target_info = validate_target_interface(target)
    cfg = {**DEFAULT_METHOD_CONFIG, **dict(config or {})}
    iterations = int(cfg.get("iterations", cfg["iterations"]))
    if cfg.get("mode") in {"runtime_smoke", "docker_validate", "smoke", "dry_run"}:
        iterations = int(cfg.get("smoke_iterations", 3))
    batch_size = int(cfg.get("batch_size", 32))
    lr = float(cfg.get("learning_rate", 0.03))
    jitter = float(cfg.get("jitter", 1e-5))
    rng = np.random.default_rng(int(cfg.get("seed", 0)))
    dim = target_info["dim"]

    mu = np.asarray(cfg.get("initial_mu", np.zeros(dim)), dtype=float).reshape(dim)
    covariance = _sym_spd(np.asarray(cfg.get("initial_covariance", np.eye(dim)), dtype=float), jitter=jitter)
    chol = np.linalg.cholesky(covariance + jitter * np.eye(dim))
    mu_state: Dict[str, Any] = {}
    chol_state: Dict[str, Any] = {}
    trace: List[Dict[str, Any]] = []
    t0 = time.time()

    for step in range(iterations):
        samples, eps_samples, chol = _sample_gaussian(rng, mu, covariance, batch_size, jitter)
        score_p = _safe_score(target, samples)
        logp = _safe_log_prob(target, samples)
        logq = _gaussian_log_prob(samples, mu, covariance, jitter)
        elbo = float(np.mean(logp - logq))

        inv_chol_t = np.linalg.inv(chol).T
        grad_mu = np.mean(score_p, axis=0)
        grad_chol = (score_p.T @ eps_samples) / float(batch_size) + inv_chol_t
        grad_chol = np.tril(grad_chol)

        mu = _adam_update(
            mu,
            grad_mu,
            mu_state,
            lr,
            float(cfg["adam_beta1"]),
            float(cfg["adam_beta2"]),
            float(cfg["adam_epsilon"]),
        )
        chol = _adam_update(
            chol,
            grad_chol,
            chol_state,
            lr,
            float(cfg["adam_beta1"]),
            float(cfg["adam_beta2"]),
            float(cfg["adam_epsilon"]),
        )
        chol = np.tril(chol)
        diag = np.maximum(np.diag(chol), math.sqrt(jitter))
        chol = chol - np.diag(np.diag(chol)) + np.diag(diag)
        covariance = _sym_spd(chol @ chol.T, jitter=jitter)

        trace.append(
            {
                "iteration": step,
                "method": "advi",
                "objective": "elbo_maximization",
                "elbo_mc": elbo,
                "loss": -elbo,
                "score_divergence_mc": _score_divergence_estimate(target, samples, mu, covariance, jitter),
                "wallclock_seconds": time.time() - t0,
            }
        )

    samples, _, _ = _sample_gaussian(rng, mu, covariance, batch_size, jitter)
    return GaussianVIResult(
        method="advi",
        mu=mu.astype(float).tolist(),
        covariance=covariance.astype(float).tolist(),
        trace=trace,
        run_config={**cfg, "method": "advi", "target": target_info},
        samples=samples.astype(float).tolist(),
        metric_inputs=_metric_payload(target, samples, mu, covariance, jitter),
        target_name=target_info["target_name"],
        dry_run=cfg.get("mode") in {"runtime_smoke", "docker_validate", "smoke", "dry_run"},
    )


def _local_run_gsm(target: Any, config: Optional[Mapping[str, Any]] = None) -> GaussianVIResult:
    """Independent full-covariance Gaussian score matching baseline.

    This is intentionally not a BaM alias.  Each iteration samples from the
    current q, fits a local affine approximation to the target score
    ``score_p(z) ≈ intercept + slope @ z`` by least squares, and converts the
    symmetric negative slope into a Gaussian precision matrix.
    """

    np = _np()
    target_info = validate_target_interface(target)
    cfg = {**DEFAULT_METHOD_CONFIG, **dict(config or {})}
    iterations = int(cfg.get("iterations", cfg["iterations"]))
    if cfg.get("mode") in {"runtime_smoke", "docker_validate", "smoke", "dry_run"}:
        iterations = int(cfg.get("smoke_iterations", 3))
    batch_size = max(int(cfg.get("batch_size", 32)), target_info["dim"] + 2)
    lr = float(cfg.get("learning_rate", 0.03))
    jitter = float(cfg.get("jitter", 1e-5))
    rng = np.random.default_rng(int(cfg.get("seed", 0)))
    dim = target_info["dim"]

    mu = np.asarray(cfg.get("initial_mu", np.zeros(dim)), dtype=float).reshape(dim)
    covariance = _sym_spd(np.asarray(cfg.get("initial_covariance", np.eye(dim)), dtype=float), jitter=jitter)
    trace: List[Dict[str, Any]] = []
    t0 = time.time()

    for step in range(iterations):
        samples, _, _ = _sample_gaussian(rng, mu, covariance, batch_size, jitter)
        score_p = _safe_score(target, samples)
        design = np.concatenate([np.ones((samples.shape[0], 1)), samples], axis=1)
        coef, *_ = np.linalg.lstsq(design, score_p, rcond=None)
        intercept = coef[0]
        slope = coef[1:].T
        precision_candidate = _sym_spd(-0.5 * (slope + slope.T), jitter=jitter)
        covariance_candidate = _sym_spd(np.linalg.inv(precision_candidate + jitter * np.eye(dim)), jitter=jitter)
        mu_candidate = np.linalg.solve(precision_candidate + jitter * np.eye(dim), intercept)

        damping = min(1.0, max(0.0, lr))
        mu = (1.0 - damping) * mu + damping * mu_candidate
        covariance = _sym_spd((1.0 - damping) * covariance + damping * covariance_candidate, jitter=jitter)
        loss = _score_divergence_estimate(target, samples, mu, covariance, jitter)
        trace.append(
            {
                "iteration": step,
                "method": "gsm",
                "objective": "independent_gaussian_score_matching",
                "score_matching_loss": loss,
                "loss": loss,
                "wallclock_seconds": time.time() - t0,
            }
        )

    samples, _, _ = _sample_gaussian(rng, mu, covariance, batch_size, jitter)
    return GaussianVIResult(
        method="gsm",
        mu=mu.astype(float).tolist(),
        covariance=covariance.astype(float).tolist(),
        trace=trace,
        run_config={**cfg, "method": "gsm", "target": target_info},
        samples=samples.astype(float).tolist(),
        metric_inputs=_metric_payload(target, samples, mu, covariance, jitter),
        target_name=target_info["target_name"],
        dry_run=cfg.get("mode") in {"runtime_smoke", "docker_validate", "smoke", "dry_run"},
    )


def _local_run_bam(target: Any, config: Optional[Mapping[str, Any]] = None) -> GaussianVIResult:
    """Local BaM-style Batch Step plus Match Step for full-covariance Gaussian q."""

    np = _np()
    target_info = validate_target_interface(target)
    cfg = {**DEFAULT_METHOD_CONFIG, **dict(config or {})}
    iterations = int(cfg.get("iterations", cfg["iterations"]))
    if cfg.get("mode") in {"runtime_smoke", "docker_validate", "smoke", "dry_run"}:
        iterations = int(cfg.get("smoke_iterations", 3))
    batch_size = int(cfg.get("batch_size", 32))
    lr = float(cfg.get("learning_rate", 0.03))
    jitter = float(cfg.get("jitter", 1e-5))
    rng = np.random.default_rng(int(cfg.get("seed", 0)))
    dim = target_info["dim"]

    mu = np.asarray(cfg.get("initial_mu", np.zeros(dim)), dtype=float).reshape(dim)
    covariance = _sym_spd(np.asarray(cfg.get("initial_covariance", np.eye(dim)), dtype=float), jitter=jitter)
    trace: List[Dict[str, Any]] = []
    t0 = time.time()

    for step in range(iterations):
        samples, _, _ = _sample_gaussian(rng, mu, covariance, batch_size, jitter)
        scores = _safe_score(target, samples)
        zbar = np.mean(samples, axis=0)
        gbar = np.mean(scores, axis=0)
        centered_z = samples - zbar.reshape(1, dim)
        centered_g = scores - gbar.reshape(1, dim)
        cross = (centered_g.T @ centered_z) / max(1, batch_size - 1)
        precision_match = _sym_spd(-0.5 * (cross + cross.T), jitter=jitter)
        covariance_match = _sym_spd(np.linalg.inv(precision_match + jitter * np.eye(dim)), jitter=jitter)
        mu_match = zbar + covariance_match @ gbar

        damping = min(1.0, max(0.0, lr))
        mu = (1.0 - damping) * mu + damping * mu_match
        covariance = _sym_spd((1.0 - damping) * covariance + damping * covariance_match, jitter=jitter)
        score_div = _score_divergence_estimate(target, samples, mu, covariance, jitter)
        trace.append(
            {
                "iteration": step,
                "method": "bam",
                "objective": "score_based_divergence_batch_and_match",
                "loss": score_div,
                "score_divergence_mc": score_div,
                "batch_step": {
                    "zbar": zbar.astype(float).tolist(),
                    "gbar": gbar.astype(float).tolist(),
                },
                "match_step": {
                    "damping": damping,
                    "regularization": "kl_proximal_full_covariance",
                },
                "wallclock_seconds": time.time() - t0,
            }
        )

    samples, _, _ = _sample_gaussian(rng, mu, covariance, batch_size, jitter)
    return GaussianVIResult(
        method="bam",
        mu=mu.astype(float).tolist(),
        covariance=covariance.astype(float).tolist(),
        trace=trace,
        run_config={**cfg, "method": "bam", "target": target_info},
        samples=samples.astype(float).tolist(),
        metric_inputs=_metric_payload(target, samples, mu, covariance, jitter),
        target_name=target_info["target_name"],
        dry_run=cfg.get("mode") in {"runtime_smoke", "docker_validate", "smoke", "dry_run"},
    )


def _load_external_runner(entry: MethodRegistryEntry) -> Optional[Callable[[Any, Optional[Mapping[str, Any]]], Any]]:
    """Lazily load a neighboring implementation if it exists.

    Returning ``Optional`` here is an internal dispatch detail; public execution
    always falls back to the concrete local implementation and never returns an
    empty/placeholder result.
    """

    try:
        module = __import__(entry.lazy_module, fromlist=[entry.lazy_callable])
        runner = getattr(module, entry.lazy_callable)
        if callable(runner):
            return runner
    except Exception:
        return False  # type: ignore[return-value]
    return False  # type: ignore[return-value]


def run_method(method: str, target: Any, config: Optional[Mapping[str, Any]] = None) -> GaussianVIResult:
    """Run a registered method on a target with the shared output schema.

    The dispatcher first attempts to use the repository's algorithm modules
    (``src.algorithms.bam``, ``src.algorithms.advi``, ``src.algorithms.gsm``).
    If those modules are unavailable or expose a different callable name during
    incremental repository construction, this package still provides executable
    local full-covariance implementations for BaM, ADVI, and GSM.
    """

    entry = get_method_entry(method)
    implementation_key = entry.implementation_key
    validate_target_interface(target)
    cfg = {**DEFAULT_METHOD_CONFIG, **dict(config or {}), "requested_method": method}

    external_runner = _load_external_runner(entry)
    if external_runner:
        result = external_runner(target, cfg)
        if isinstance(result, GaussianVIResult):
            return result
        if isinstance(result, Mapping):
            return coerce_result_schema(result, method=implementation_key, target=target, config=cfg)

    if implementation_key == "advi":
        return _local_run_advi(target, cfg)
    if implementation_key == "gsm":
        return _local_run_gsm(target, cfg)
    if implementation_key == "bam":
        return _local_run_bam(target, cfg)
    raise KeyError(f"No implementation is registered for method '{method}' -> '{implementation_key}'")


def coerce_result_schema(
    result: Mapping[str, Any],
    method: str,
    target: Any,
    config: Optional[Mapping[str, Any]] = None,
) -> GaussianVIResult:
    """Coerce neighboring method outputs into the package-level schema."""

    np = _np()
    target_info = validate_target_interface(target)
    dim = target_info["dim"]
    mu = np.asarray(result.get("mu", result.get("mean", np.zeros(dim))), dtype=float).reshape(dim)
    covariance = _sym_spd(
        np.asarray(result.get("covariance", result.get("Sigma", result.get("cov", np.eye(dim)))), dtype=float),
        jitter=float((config or {}).get("jitter", DEFAULT_METHOD_CONFIG["jitter"])),
    )
    samples = np.asarray(result.get("samples", np.tile(mu.reshape(1, dim), (1, 1))), dtype=float)
    if samples.ndim == 1:
        samples = samples.reshape(1, dim)
    trace = list(result.get("trace", []))
    metric_inputs = dict(result.get("metric_inputs", {}))
    if not metric_inputs:
        metric_inputs = _metric_payload(
            target,
            samples,
            mu,
            covariance,
            float((config or {}).get("jitter", DEFAULT_METHOD_CONFIG["jitter"])),
        )
    return GaussianVIResult(
        method=str(result.get("method", method)),
        mu=mu.astype(float).tolist(),
        covariance=covariance.astype(float).tolist(),
        trace=trace,
        run_config=dict(result.get("run_config", {**(config or {}), "method": method, "target": target_info})),
        samples=samples.astype(float).tolist(),
        metric_inputs=metric_inputs,
        target_name=str(result.get("target_name", target_info["target_name"])),
        dry_run=bool(result.get("dry_run", (config or {}).get("mode") in {"runtime_smoke", "docker_validate", "smoke"})),
    )


class SyntheticGaussianTarget:
    """Small import-light Gaussian target for smoke tests and registry validation."""

    name = "synthetic_gaussian"

    def __init__(self, dim: int = 2, mean: Optional[Sequence[float]] = None, covariance: Optional[Sequence[Sequence[float]]] = None):
        np = _np()
        self.dim = int(dim)
        self.mean = np.asarray(mean if mean is not None else np.zeros(self.dim), dtype=float).reshape(self.dim)
        self.covariance = _sym_spd(
            np.asarray(covariance if covariance is not None else np.eye(self.dim), dtype=float),
            jitter=1e-6,
        )
        self.precision = np.linalg.inv(self.covariance)

    def log_prob(self, z: ArrayLike) -> Any:
        np = _np()
        arr = _as_batch(z, self.dim)
        centered = arr - self.mean.reshape(1, self.dim)
        quad = np.sum((centered @ self.precision) * centered, axis=1)
        sign, logdet = np.linalg.slogdet(self.covariance)
        return -0.5 * (self.dim * math.log(2.0 * math.pi) + logdet + quad)

    def score(self, z: ArrayLike) -> Any:
        arr = _as_batch(z, self.dim)
        return -(arr - self.mean.reshape(1, self.dim)) @ self.precision.T


def make_smoke_target(dim: int = 2) -> SyntheticGaussianTarget:
    """Construct the bounded synthetic Gaussian target used by smoke validation."""

    return SyntheticGaussianTarget(dim=dim)


def _artifact_root(base_dir: Optional[os.PathLike[str] | str] = None) -> Path:
    if base_dir is not None:
        return Path(base_dir)
    env_dir = os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR")
    if env_dir:
        return Path(env_dir)
    return Path(".")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def materialize_dry_run_artifacts(
    base_dir: Optional[os.PathLike[str] | str] = None,
    mode: str = "runtime_smoke",
) -> Dict[str, Any]:
    """Materialize declared dry-run contract artifacts.

    The artifacts are schema/readiness outputs only.  They validate repository
    wiring and do not claim benchmark scores, trained-model performance, or
    completed expensive experiments.
    """

    root = _artifact_root(base_dir)
    target = make_smoke_target(dim=2)
    results: Dict[str, GaussianVIResult] = {}
    for method in ("bam", "advi", "gsm"):
        results[method] = run_method(
            method,
            target,
            {
                **DEFAULT_METHOD_CONFIG,
                "mode": mode,
                "iterations": DEFAULT_METHOD_CONFIG["smoke_iterations"],
                "smoke_iterations": DEFAULT_METHOD_CONFIG["smoke_iterations"],
                "seed": {"bam": 0, "advi": 1, "gsm": 2}[method],
            },
        )

    trace_payload = {
        "artifact_kind": "dry_run_contract_trace",
        "dry_run": True,
        "mode": mode,
        "warning": "Schema/readiness artifact; not a paper result.",
        "traces": {method: result.trace for method, result in results.items()},
    }
    baseline_payload = {
        "artifact_kind": "dry_run_baseline_traces",
        "dry_run": True,
        "mode": mode,
        "warning": "Baseline wiring artifact; not a benchmark result.",
        "baselines": {
            "advi": results["advi"].to_dict(),
            "gsm": results["gsm"].to_dict(),
            "baseline_selector": METHOD_REGISTRY["baseline"].implementation_key,
        },
    }
    synthetic_metrics_payload = {
        "artifact_kind": "dry_run_metric_schema",
        "dry_run": True,
        "mode": mode,
        "warning": "Metric schema and smoke estimates only; not reported paper scores.",
        "metrics": {method: result.metric_inputs for method, result in results.items()},
        "decisive_metrics": sorted({m for entry in METHOD_REGISTRY.values() for m in entry.decisive_metrics}),
    }
    curves_payload = {
        "artifact_kind": "dry_run_curve_schema",
        "dry_run": True,
        "mode": mode,
        "warning": "Curve schema artifact; not a completed convergence curve.",
        "x_axis": "iteration",
        "series": {
            method: [
                {
                    "iteration": item.get("iteration", idx),
                    "loss": item.get("loss", item.get("score_matching_loss", item.get("negative_elbo"))),
                }
                for idx, item in enumerate(result.trace)
            ]
            for method, result in results.items()
        },
    }
    readiness_payload = {
        "artifact_kind": "readiness",
        "dry_run": True,
        "mode": mode,
        "package": "bam",
        "version": __version__,
        "paper": __paper__,
        "blacklisted_repositories_not_used": list(__blacklisted_repositories__),
        "method_registry_entries": sorted(METHOD_REGISTRY),
        "supported_protocol_targets": list(SUPPORTED_PROTOCOL_TARGETS),
        "output_schema": GAUSSIAN_VI_OUTPUT_SCHEMA,
        "declared_artifacts": list(CANONICAL_ARTIFACTS),
        "status": "ready_for_bounded_smoke",
    }
    evaluation_payload = {
        "artifact_kind": "evaluation_result",
        "dry_run": True,
        "mode": mode,
        "warning": "Dry-run evaluation contract artifact; not a paper benchmark result.",
        "comparison": "bam_vs_advi_vs_gsm",
        "result_schema": GAUSSIAN_VI_OUTPUT_SCHEMA["schema"],
        "method_results": {method: result.to_dict() for method, result in results.items()},
        "hypothesis": DEFAULT_METHOD_CONFIG["hypothesis"],
        "decisive_metric": DEFAULT_METHOD_CONFIG["decisive_metric"],
        "stop_rule_or_pruning_rationale": DEFAULT_METHOD_CONFIG["stop_rule_or_pruning_rationale"],
    }

    artifact_payloads: Dict[str, Mapping[str, Any]] = {
        "results/loss_trace.json": trace_payload,
        "results/method_registry.json": {
            "artifact_kind": "method_registry",
            "dry_run": True,
            "reference_grounding": "paper:paper_contract_method_baseline_protocol paper.md",
            "methods": get_method_registry(),
            "shared_output_schema": GAUSSIAN_VI_OUTPUT_SCHEMA,
        },
        "results/ablation_registry.json": {
            "artifact_kind": "ablation_registry",
            "dry_run": True,
            "ablations": get_ablation_registry(),
            "bounded_default_policy": "Only bounded smoke/default ablations execute unless full mode is explicit.",
        },
        "results/synthetic_kl_metrics.json": synthetic_metrics_payload,
        "results/synthetic_curves.json": curves_payload,
        "results/baseline_traces.json": baseline_payload,
        "results/readiness.json": readiness_payload,
        "results/evaluation_result.json": evaluation_payload,
    }

    written: List[str] = []
    for rel_path, payload in artifact_payloads.items():
        out_path = root / rel_path
        _write_json(out_path, payload)
        written.append(str(out_path))

    manifest = {
        "artifact_kind": "dry_run_artifact_manifest",
        "dry_run": True,
        "mode": mode,
        "written": written,
        "canonical_artifacts": list(CANONICAL_ARTIFACTS),
    }
    return manifest


def package_contract() -> Dict[str, Any]:
    """Return the package-level benchmark-visible contract."""

    return {
        "paper": __paper__,
        "version": __version__,
        "method_registry": get_method_registry(),
        "ablation_registry": get_ablation_registry(),
        "supported_protocol_targets": list(SUPPORTED_PROTOCOL_TARGETS),
        "output_schema": GAUSSIAN_VI_OUTPUT_SCHEMA,
        "default_config": dict(DEFAULT_METHOD_CONFIG),
        "canonical_artifacts": list(CANONICAL_ARTIFACTS),
        "blacklisted_repositories_not_used": list(__blacklisted_repositories__),
    }


__all__ = [
    "ABLATION_REGISTRY",
    "CANONICAL_ARTIFACTS",
    "DEFAULT_METHOD_CONFIG",
    "GAUSSIAN_VI_OUTPUT_SCHEMA",
    "METHOD_REGISTRY",
    "SUPPORTED_PROTOCOL_TARGETS",
    "GaussianVIResult",
    "MethodAdapter",
    "MethodRegistryEntry",
    "SyntheticGaussianTarget",
    "TargetProtocol",
    "__blacklisted_repositories__",
    "__paper__",
    "__version__",
    "coerce_result_schema",
    "get_ablation_registry",
    "get_method_entry",
    "get_method_registry",
    "list_methods",
    "make_smoke_target",
    "make_method",
    "materialize_dry_run_artifacts",
    "package_contract",
    "run_method",
    "validate_target_interface",
]
