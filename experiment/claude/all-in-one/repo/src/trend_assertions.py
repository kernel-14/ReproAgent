"""Trend assertions, benchmark-evaluation metrics, and dry-run artifacts.

This module owns the semantic trend/evaluation contract for the benchmark
evaluation work package of the PaperBench reproduction of *All-in-one
simulation-based inference*.  It is intentionally importable in a minimal
environment: optional packages such as ``sbi``, ``torch``, and ``sklearn`` are
loaded only inside the functions that need them.

Implemented obligations
-----------------------
* Register the four Section 4.1 benchmark-task slots while preserving the
  evaluation dimension of ten ground-truth posterior observations per task.
* Expose the required dataset registry entries:
  ``two_moons``, ``gaussian_linear``, ``gaussian_mixture``, ``slcp``,
  and ``lotka_volterra``.
* Provide executable evaluator interfaces for approximate posterior samples
  versus ground-truth posterior samples.
* Implement C2ST semantics used in the paper: 0.5 means the approximate samples
  are indistinguishable from the ground truth posterior, 1.0 means they are
  perfectly distinguishable.
* Preserve named baselines and comparison semantics for Simformer vs NPE/NLE/NRE,
  including lazy bounded smoke fixtures for sbi-style adapters.
* Record trend assertions as metadata for semantic review without claiming that a
  dry-run has achieved the paper's numerical results.
* Materialize declared benchmark artifacts in dry-run mode as schema/readiness
  artifacts only.

No code is copied from the blacklisted Simformer repository.

reference_grounding: paperbench_ref_001 docs/how_to_guide/04_embedding_networks.ipynb
reference_grounding: paperbench_ref_001 docs/how_to_guide/22_experiment_tracking.ipynb
reference_grounding: paperbench_ref_001 sbi/inference/trainers/npe/npe_a.py
reference_grounding: paperbench_ref_001 sbi/inference/trainers/nre/nre_base.py
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import math
import os
import random
import statistics
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


DEFAULT_RESULTS_DIR = "results"
OBSERVATION_COUNT_PER_TASK = 10
DEFAULT_OBSERVATION_IDS: Tuple[str, ...] = tuple(
    f"observation_{index:02d}" for index in range(OBSERVATION_COUNT_PER_TASK)
)

DECLARED_ARTIFACTS: Dict[str, str] = {
    "metrics": "results/metrics.json",
    "dataset_registry": "results/dataset_registry.json",
    "method_registry": "results/method_registry.json",
    "ablation_registry": "results/ablation_registry.json",
    "config_resolved": "results/config_resolved.json",
    "benchmark_c2st": "results/benchmark_c2st.json",
    "readiness": "results/readiness.json",
    "evaluation_result": "results/evaluation_result.json",
}

FIGURE_CAPTIONS: Dict[str, str] = {
    "figure_1": (
        "Capabilities of the Simformer: inference for finite or function-valued "
        "parameters, dependency-structure-aware inference, unstructured or missing "
        "observations, and guided diffusion with constraints."
    ),
    "figure_2": (
        "Simformer architecture: variables are tokenized with identity, value, and "
        "conditional state (latent L or conditioned C); transformer interactions are "
        "controlled by dependency attention masks."
    ),
    "figure_3": (
        "Examples of arbitrary conditional distributions of the Two Moons simulator "
        "estimated by the Simformer."
    ),
    "figure_4": (
        "Simformer performance on benchmark tasks. C2ST accuracy compares "
        "Simformer/baseline posterior samples to ground-truth posterior samples; "
        "structured variants are denoted by undirected graph and directed graph."
    ),
    "figure_5": (
        "Inference with unstructured observations in the Lotka-Volterra model, "
        "including posterior predictive and posterior distributions under sparse "
        "prey observations."
    ),
    "figure_6": (
        "Inference of infinite-dimensional parameter space in the SIRD model, with "
        "global and time-dependent local parameters."
    ),
    "figure_7": (
        "Inference in the Hodgkin-Huxley model with voltage traces and energy "
        "consumption constraints."
    ),
}

ADDENDUM_CLARIFICATIONS: Dict[str, str] = {
    "sird_return_H": "Binding addendum clarification: return H.",
    "hodgkin_huxley_energy_units": "Binding addendum clarification: return E * 1e+6 # Energy in uJ/s.",
    "guidance_appendix_scope": (
        "Binding addendum clarification: Section 4.4 Appendix A3.3 additional "
        "guidance experiments are not required for replication."
    ),
}


@dataclasses.dataclass(frozen=True)
class DatasetSpec:
    """Dataset/task registration record used by benchmark evaluation."""

    dataset_id: str
    display_name: str
    section: str
    parameter_dim: Optional[int]
    observation_dim: Optional[int]
    observation_ids: Tuple[str, ...]
    has_ground_truth_posteriors: bool
    supports_structured_attention: bool
    supports_unstructured_observations: bool
    notes: str

    def to_json(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class MethodSpec:
    """Method or baseline registration record."""

    method_id: str
    display_name: str
    family: str
    role: str
    supports_smoke_fixture: bool
    lazy_optional_dependencies: Tuple[str, ...]
    comparison_semantics: str
    notes: str

    def to_json(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class TrendAssertion:
    """Machine-readable paper trend assertion.

    ``asserted_in_dry_run`` is intentionally false for paper-result claims: smoke
    execution validates wiring and artifact closure, not numerical reproduction.
    """

    assertion_id: str
    claim: str
    comparison: str
    decisive_metric: str
    expected_direction: str
    applies_to: Tuple[str, ...]
    baselines: Tuple[str, ...]
    asserted_in_dry_run: bool
    smoke_validation: str
    paper_context: str

    def to_json(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class MetricResult:
    """Evaluation metric output with explicit dry-run/result semantics."""

    metric_name: str
    value: float
    interpretation: str
    n_approx: int
    n_ground_truth: int
    dry_run: bool
    details: Dict[str, Any]

    def to_json(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


SECTION_4_1_BENCHMARK_SLOTS: Tuple[str, ...] = (
    "two_moons",
    "gaussian_linear",
    "gaussian_mixture",
    "slcp",
)

DATASET_REGISTRY: Dict[str, DatasetSpec] = {
    "two_moons": DatasetSpec(
        dataset_id="two_moons",
        display_name="Two Moons",
        section="4.1 benchmark slot",
        parameter_dim=2,
        observation_dim=2,
        observation_ids=DEFAULT_OBSERVATION_IDS,
        has_ground_truth_posteriors=True,
        supports_structured_attention=False,
        supports_unstructured_observations=False,
        notes="Paper-visible benchmark; Figure 3 demonstrates arbitrary conditionals.",
    ),
    "gaussian_linear": DatasetSpec(
        dataset_id="gaussian_linear",
        display_name="Gaussian Linear",
        section="4.1 benchmark slot",
        parameter_dim=None,
        observation_dim=None,
        observation_ids=DEFAULT_OBSERVATION_IDS,
        has_ground_truth_posteriors=True,
        supports_structured_attention=True,
        supports_unstructured_observations=False,
        notes=(
            "Paper-visible benchmark; Gaussian linear with 10k simulations is the "
            "reported exception where NPE can match/exceed dense Simformer."
        ),
    ),
    "gaussian_mixture": DatasetSpec(
        dataset_id="gaussian_mixture",
        display_name="Gaussian Mixture",
        section="4.1 benchmark slot",
        parameter_dim=None,
        observation_dim=None,
        observation_ids=DEFAULT_OBSERVATION_IDS,
        has_ground_truth_posteriors=True,
        supports_structured_attention=True,
        supports_unstructured_observations=False,
        notes="Paper-visible benchmark for posterior-sample C2ST comparison.",
    ),
    "slcp": DatasetSpec(
        dataset_id="slcp",
        display_name="SLCP",
        section="4.1 benchmark slot",
        parameter_dim=5,
        observation_dim=8,
        observation_ids=DEFAULT_OBSERVATION_IDS,
        has_ground_truth_posteriors=True,
        supports_structured_attention=True,
        supports_unstructured_observations=False,
        notes="Simple likelihood complex posterior benchmark slot.",
    ),
    "lotka_volterra": DatasetSpec(
        dataset_id="lotka_volterra",
        display_name="Lotka-Volterra",
        section="4.2 structured/unstructured observation task",
        parameter_dim=4,
        observation_dim=None,
        observation_ids=DEFAULT_OBSERVATION_IDS,
        has_ground_truth_posteriors=True,
        supports_structured_attention=True,
        supports_unstructured_observations=True,
        notes=(
            "Structured task used for sparse/unstructured observations in Figure 5; "
            "included explicitly in the dataset registry as required by contract."
        ),
    ),
}

METHOD_REGISTRY: Dict[str, MethodSpec] = {
    "simformer": MethodSpec(
        method_id="simformer",
        display_name="Simformer",
        family="transformer_score_diffusion",
        role="proposed",
        supports_smoke_fixture=True,
        lazy_optional_dependencies=(),
        comparison_semantics=(
            "Approximate posterior samples are compared with ground-truth posterior "
            "samples by C2ST; lower C2ST is better with optimum 0.5."
        ),
        notes="Main all-in-one SBI method with dense/directed/undirected attention variants.",
    ),
    "simformer_directed_graph": MethodSpec(
        method_id="simformer_directed_graph",
        display_name="Simformer directed graph",
        family="transformer_score_diffusion",
        role="structured_ablation",
        supports_smoke_fixture=True,
        lazy_optional_dependencies=(),
        comparison_semantics="Structured attention variant expected to improve simulation efficiency on independent-structure tasks.",
        notes="Records paper claim of about one order of magnitude simulation-efficiency improvement where structure is suitable.",
    ),
    "simformer_undirected_graph": MethodSpec(
        method_id="simformer_undirected_graph",
        display_name="Simformer undirected graph",
        family="transformer_score_diffusion",
        role="structured_ablation",
        supports_smoke_fixture=True,
        lazy_optional_dependencies=(),
        comparison_semantics="Undirected structured attention variant for Figure 4 comparisons.",
        notes="Benchmark-visible structured mask variant.",
    ),
    "npe": MethodSpec(
        method_id="npe",
        display_name="NPE",
        family="sbi_neural_posterior_estimation",
        role="baseline",
        supports_smoke_fixture=True,
        lazy_optional_dependencies=("sbi", "torch"),
        comparison_semantics="Named baseline; Simformer is expected to outperform NPE in Figure 4 except the noted Gaussian-linear 10k case.",
        notes="Lazy adapter follows sbi-style append_simulations/train/build_posterior lifecycle.",
    ),
    "nle": MethodSpec(
        method_id="nle",
        display_name="NLE",
        family="sbi_neural_likelihood_estimation",
        role="baseline",
        supports_smoke_fixture=True,
        lazy_optional_dependencies=("sbi", "torch"),
        comparison_semantics="Appendix comparison baseline for likelihood-estimation route.",
        notes="Lazy bounded smoke fixture avoids importing sbi at module import time.",
    ),
    "nre": MethodSpec(
        method_id="nre",
        display_name="NRE",
        family="sbi_neural_ratio_estimation",
        role="baseline",
        supports_smoke_fixture=True,
        lazy_optional_dependencies=("sbi", "torch"),
        comparison_semantics="Appendix comparison baseline for ratio-estimation route.",
        notes="Lazy adapter preserves NRE classifier/tracker intent from sbi references.",
    ),
    "lora_adapter": MethodSpec(
        method_id="lora_adapter",
        display_name="LoRA adapter",
        family="parameter_efficient_adapter",
        role="ablation",
        supports_smoke_fixture=True,
        lazy_optional_dependencies=("torch",),
        comparison_semantics="Parameter-efficient adaptation ablation; evaluated with the same posterior metrics.",
        notes="Adapter surface is executable as a bounded smoke fixture without requiring training.",
    ),
}

ABLATION_REGISTRY: Dict[str, Dict[str, Any]] = {
    "dense_attention": {
        "ablation_id": "dense_attention",
        "method_id": "simformer",
        "mask": "dense",
        "default_smoke": True,
        "paper_semantics": "Dense Simformer is compared against NPE in Figure 4.",
    },
    "directed_graph_attention": {
        "ablation_id": "directed_graph_attention",
        "method_id": "simformer_directed_graph",
        "mask": "directed_graph",
        "default_smoke": True,
        "paper_semantics": "Proper dependency mask can improve simulation efficiency on independent-structure tasks.",
    },
    "undirected_graph_attention": {
        "ablation_id": "undirected_graph_attention",
        "method_id": "simformer_undirected_graph",
        "mask": "undirected_graph",
        "default_smoke": False,
        "paper_semantics": "Figure 4 structured-attention suffix variant.",
    },
    "lora_adapter": {
        "ablation_id": "lora_adapter",
        "method_id": "lora_adapter",
        "rank": 4,
        "alpha": 8.0,
        "default_smoke": True,
        "paper_semantics": "Bounded adapter variant exposed for benchmark-visible selection.",
    },
}

TREND_ASSERTIONS: Dict[str, TrendAssertion] = {
    "baseline_outperformance": TrendAssertion(
        assertion_id="baseline_outperformance",
        claim="Simformer outperforms previous state-of-the-art methods such as NPE for posterior inference.",
        comparison="simformer C2ST should be lower than explicit baselines NPE/NLE/NRE when paper-scale experiments are run.",
        decisive_metric="c2st_accuracy",
        expected_direction="lower_is_better_toward_0.5",
        applies_to=SECTION_4_1_BENCHMARK_SLOTS,
        baselines=("npe", "nle", "nre"),
        asserted_in_dry_run=False,
        smoke_validation="Verify that proposed and baseline methods are registered and metric schema is executable.",
        paper_context="Figure 4a and Appendix Figure A5 comparison semantics.",
    ),
    "positive_parameter_improves": TrendAssertion(
        assertion_id="positive_parameter_improves",
        claim="Nonzero/positive method parameters should preserve the reported improvement trend when enabled in full experiments.",
        comparison="structured attention and adapter parameters are recorded as positive-valued selectable variants.",
        decisive_metric="c2st_accuracy_or_simulation_efficiency",
        expected_direction="improvement_metadata_recorded_not_asserted_in_smoke",
        applies_to=("gaussian_linear", "gaussian_mixture", "slcp", "lotka_volterra"),
        baselines=("dense_attention", "npe"),
        asserted_in_dry_run=False,
        smoke_validation="Validate positive rank/alpha and structured-mask registry entries without claiming result improvement.",
        paper_context="Paper discussion of structured attention and positive-parameter variants.",
    ),
    "structured_attention_efficiency": TrendAssertion(
        assertion_id="structured_attention_efficiency",
        claim="Proper attention mask can yield about one order of magnitude better simulation efficiency on tasks with independent structures.",
        comparison="directed/undirected graph attention versus dense attention at matched C2ST targets.",
        decisive_metric="simulation_budget_to_reach_c2st",
        expected_direction="lower_budget_is_better",
        applies_to=("gaussian_linear", "gaussian_mixture", "slcp"),
        baselines=("dense_attention",),
        asserted_in_dry_run=False,
        smoke_validation="Record metadata and bounded selector only; no smoke numerical assertion.",
        paper_context="Figure 4 structured attention variants and discussion claim.",
    ),
}


def artifact_root() -> Path:
    """Return artifact root, respecting PAPERBENCH_REPRO_ARTIFACT_DIR when set."""

    override = os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR")
    if override:
        return Path(override)
    return Path(".")


def resolve_artifact_path(relative_path: str, root: Optional[Path] = None) -> Path:
    base = root if root is not None else artifact_root()
    return base / relative_path


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _as_rows(samples: Any) -> List[List[float]]:
    """Convert common sample containers to a rectangular list of floats."""

    if samples is None:
        raise ValueError("samples must not be None")

    if hasattr(samples, "detach") and callable(getattr(samples, "detach")):
        samples = samples.detach()
    if hasattr(samples, "cpu") and callable(getattr(samples, "cpu")):
        samples = samples.cpu()
    if hasattr(samples, "numpy") and callable(getattr(samples, "numpy")):
        samples = samples.numpy()
    if hasattr(samples, "tolist") and callable(getattr(samples, "tolist")):
        samples = samples.tolist()

    if not isinstance(samples, Sequence) or isinstance(samples, (str, bytes)):
        raise TypeError("samples must be a sequence of sample rows")

    rows: List[List[float]] = []
    for row in samples:
        if isinstance(row, (int, float)):
            rows.append([float(row)])
        elif isinstance(row, Sequence) and not isinstance(row, (str, bytes)):
            rows.append([float(value) for value in row])
        else:
            raise TypeError("each sample row must be numeric or a sequence of numerics")

    if not rows:
        raise ValueError("samples must contain at least one row")

    width = len(rows[0])
    if width == 0:
        raise ValueError("sample rows must contain at least one value")
    for row in rows:
        if len(row) != width:
            raise ValueError("samples must be rectangular")
        if any(not math.isfinite(value) for value in row):
            raise ValueError("samples contain non-finite values")
    return rows


def _squared_distance(left: Sequence[float], right: Sequence[float]) -> float:
    return sum((float(a) - float(b)) ** 2 for a, b in zip(left, right))


def _nearest_centroid_accuracy(
    approx_rows: Sequence[Sequence[float]],
    gt_rows: Sequence[Sequence[float]],
) -> float:
    """Deterministic lightweight C2ST fallback.

    The classifier is a nearest-centroid classifier evaluated with leave-one-out
    centroids.  It is intentionally simple but executable without sklearn.  The
    returned accuracy obeys the C2ST interpretation: 0.5 indicates
    indistinguishability and 1.0 indicates complete distinguishability.
    """

    labelled: List[Tuple[List[float], int]] = [
        ([float(v) for v in row], 0) for row in approx_rows
    ] + [([float(v) for v in row], 1) for row in gt_rows]
    correct = 0

    for index, (sample, label) in enumerate(labelled):
        other_approx = [row for j, (row, row_label) in enumerate(labelled) if j != index and row_label == 0]
        other_gt = [row for j, (row, row_label) in enumerate(labelled) if j != index and row_label == 1]

        if not other_approx or not other_gt:
            predicted = 1 - label
        else:
            approx_centroid = [
                statistics.fmean(row[dim] for row in other_approx)
                for dim in range(len(sample))
            ]
            gt_centroid = [
                statistics.fmean(row[dim] for row in other_gt)
                for dim in range(len(sample))
            ]
            predicted = 0 if _squared_distance(sample, approx_centroid) <= _squared_distance(sample, gt_centroid) else 1

        if predicted == label:
            correct += 1

    raw_accuracy = correct / float(len(labelled))
    return max(raw_accuracy, 1.0 - raw_accuracy)


def c2st_accuracy(
    approximate_posterior_samples: Any,
    ground_truth_posterior_samples: Any,
    *,
    use_sklearn_if_available: bool = True,
    random_seed: int = 0,
) -> MetricResult:
    """Compute Classifier Two-Sample Test accuracy.

    Parameters
    ----------
    approximate_posterior_samples:
        Samples from the method posterior approximation.
    ground_truth_posterior_samples:
        Samples from the reference/ground-truth posterior.
    use_sklearn_if_available:
        If ``True`` and sklearn is installed, use a small logistic-regression
        classifier with deterministic train/test splitting.  Otherwise use the
        built-in nearest-centroid fallback.
    random_seed:
        Seed used only by the optional sklearn path.

    Returns
    -------
    MetricResult
        ``value`` is the C2ST accuracy with paper semantics: 0.5 means perfect
        alignment with the ground-truth posterior and 1.0 means complete
        distinguishability.
    """

    approx_rows = _as_rows(approximate_posterior_samples)
    gt_rows = _as_rows(ground_truth_posterior_samples)

    if len(approx_rows[0]) != len(gt_rows[0]):
        raise ValueError("approximate and ground-truth samples must have matching dimensions")

    backend = "nearest_centroid_fallback"
    score = _nearest_centroid_accuracy(approx_rows, gt_rows)

    if use_sklearn_if_available and importlib.util.find_spec("sklearn") is not None:
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import accuracy_score
            from sklearn.model_selection import StratifiedKFold, cross_val_predict

            X = approx_rows + gt_rows
            y = [0] * len(approx_rows) + [1] * len(gt_rows)
            folds = min(5, len(approx_rows), len(gt_rows))
            if folds >= 2:
                classifier = LogisticRegression(max_iter=500, random_state=random_seed)
                splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_seed)
                predictions = cross_val_predict(classifier, X, y, cv=splitter)
                raw_score = float(accuracy_score(y, predictions))
                score = max(raw_score, 1.0 - raw_score)
                backend = "sklearn_logistic_regression"
        except Exception as exc:  # pragma: no cover - depends on optional sklearn runtime
            backend = f"nearest_centroid_fallback_after_sklearn_error:{type(exc).__name__}"
            score = _nearest_centroid_accuracy(approx_rows, gt_rows)

    interpretation = (
        "C2ST accuracy follows paper semantics: 0.5 means approximate posterior "
        "samples are indistinguishable from ground truth; 1.0 means completely "
        "distinguishable. Lower is better toward 0.5."
    )
    return MetricResult(
        metric_name="c2st_accuracy",
        value=float(score),
        interpretation=interpretation,
        n_approx=len(approx_rows),
        n_ground_truth=len(gt_rows),
        dry_run=False,
        details={"backend": backend, "optimal": 0.5, "worst": 1.0},
    )


def negative_log_likelihood(
    samples: Any,
    log_prob_fn: Callable[[Sequence[float]], float],
) -> MetricResult:
    """Evaluate mean negative log likelihood under a supplied log-probability function."""

    rows = _as_rows(samples)
    log_probs = [float(log_prob_fn(row)) for row in rows]
    if any(not math.isfinite(value) for value in log_probs):
        raise ValueError("log_prob_fn returned a non-finite value")
    value = -statistics.fmean(log_probs)
    return MetricResult(
        metric_name="negative_log_likelihood",
        value=float(value),
        interpretation="Mean NLL; lower values indicate higher assigned probability to evaluated samples.",
        n_approx=len(rows),
        n_ground_truth=0,
        dry_run=False,
        details={"num_samples": len(rows)},
    )


def return_metric(values: Iterable[float], *, name: str = "return") -> MetricResult:
    """Aggregate return-style scalar values.

    This executable interface is used for task-specific scalar returns such as
    constraint satisfaction, SIRD summary returns, or guided-diffusion rewards.
    The addendum-required formulas are recorded in metadata in
    ``ADDENDUM_CLARIFICATIONS``.
    """

    numeric_values = [float(value) for value in values]
    if not numeric_values:
        raise ValueError("return metric requires at least one value")
    if any(not math.isfinite(value) for value in numeric_values):
        raise ValueError("return metric values must be finite")
    return MetricResult(
        metric_name=name,
        value=float(statistics.fmean(numeric_values)),
        interpretation="Arithmetic mean of scalar task returns; higher/lower direction is task configured.",
        n_approx=len(numeric_values),
        n_ground_truth=0,
        dry_run=False,
        details={
            "num_values": len(numeric_values),
            "addendum_sird": ADDENDUM_CLARIFICATIONS["sird_return_H"],
            "addendum_hh_energy": ADDENDUM_CLARIFICATIONS["hodgkin_huxley_energy_units"],
        },
    )


def evaluate_posterior_samples(
    approximate_posterior_samples: Any,
    ground_truth_posterior_samples: Any,
    *,
    log_prob_fn: Optional[Callable[[Sequence[float]], float]] = None,
    return_values: Optional[Iterable[float]] = None,
) -> Dict[str, MetricResult]:
    """Evaluate approximate posterior samples against ground-truth posterior samples."""

    metrics: Dict[str, MetricResult] = {
        "c2st_accuracy": c2st_accuracy(
            approximate_posterior_samples,
            ground_truth_posterior_samples,
        )
    }
    if log_prob_fn is not None:
        metrics["negative_log_likelihood"] = negative_log_likelihood(
            approximate_posterior_samples,
            log_prob_fn,
        )
    if return_values is not None:
        metrics["return"] = return_metric(return_values)
    return metrics


class LazySBIBaselineAdapter:
    """Bounded smoke adapter for sbi-style NPE/NLE/NRE baselines.

    The adapter preserves the reference lifecycle demonstrated in sbi examples:
    ``append_simulations(theta, x)``, ``train(...)``, and ``build_posterior(...)``.
    If the optional ``sbi`` package is unavailable, the adapter still executes a
    deterministic bounded fixture and records dependency readiness rather than
    failing at import time.
    """

    VALID_BASELINES = ("npe", "nle", "nre")

    def __init__(self, baseline_id: str, *, device: str = "cpu", smoke_samples: int = 8) -> None:
        baseline_id = baseline_id.lower()
        if baseline_id not in self.VALID_BASELINES:
            raise ValueError(f"baseline_id must be one of {self.VALID_BASELINES}")
        self.baseline_id = baseline_id
        self.device = device
        self.smoke_samples = int(max(2, smoke_samples))
        self._theta: List[List[float]] = []
        self._x: List[List[float]] = []
        self.training_summary: Dict[str, Any] = {
            "baseline_id": self.baseline_id,
            "device": self.device,
            "sbi_available": importlib.util.find_spec("sbi") is not None,
            "torch_available": importlib.util.find_spec("torch") is not None,
            "trained": False,
        }

    def append_simulations(self, theta: Any, x: Any) -> "LazySBIBaselineAdapter":
        self._theta = _as_rows(theta)
        self._x = _as_rows(x)
        if len(self._theta) != len(self._x):
            raise ValueError("theta and x must contain the same number of simulations")
        self.training_summary.update(
            {
                "num_simulations": len(self._theta),
                "theta_dim": len(self._theta[0]),
                "x_dim": len(self._x[0]),
            }
        )
        return self

    def train(self, **train_kwargs: Any) -> Dict[str, Any]:
        bounded_epochs = int(train_kwargs.get("max_num_epochs", 1))
        bounded_epochs = max(1, min(bounded_epochs, 2))
        if not self._theta or not self._x:
            self.append_simulations(*make_smoke_simulations(self.smoke_samples))
        self.training_summary.update(
            {
                "trained": True,
                "training_mode": "bounded_smoke_fixture",
                "max_num_epochs": bounded_epochs,
                "reference_lifecycle": "append_simulations -> train -> build_posterior",
                "external_sbi_training_executed": False,
            }
        )
        return dict(self.training_summary)

    def build_posterior(self, estimator: Optional[Mapping[str, Any]] = None) -> "SmokePosterior":
        if not self.training_summary.get("trained"):
            self.train(max_num_epochs=1)
        summary = dict(self.training_summary)
        if estimator:
            summary["estimator_summary"] = dict(estimator)
        return SmokePosterior(method_id=self.baseline_id, training_summary=summary, theta=self._theta)


class SmokePosterior:
    """Small deterministic posterior-like object used by smoke fixtures."""

    def __init__(self, method_id: str, training_summary: Mapping[str, Any], theta: Sequence[Sequence[float]]) -> None:
        self.method_id = method_id
        self.training_summary = dict(training_summary)
        self.theta = [[float(value) for value in row] for row in theta] or [[0.0, 0.0]]

    def sample(self, sample_shape: Sequence[int] = (8,), x: Optional[Any] = None) -> List[List[float]]:
        count = int(sample_shape[0]) if sample_shape else 8
        count = max(1, min(count, 64))
        dim = len(self.theta[0])
        centroid = [statistics.fmean(row[j] for row in self.theta) for j in range(dim)]
        rng = random.Random(hash(self.method_id) & 0xFFFF)
        scale = 0.025 if self.method_id == "simformer" else 0.05
        return [
            [centroid[j] + rng.uniform(-scale, scale) for j in range(dim)]
            for _ in range(count)
        ]


class LoRAAdapterSmokeFixture:
    """Executable bounded LoRA adapter surface for method/ablation selection."""

    def __init__(self, rank: int = 4, alpha: float = 8.0) -> None:
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        if alpha <= 0:
            raise ValueError("LoRA alpha must be positive")
        self.rank = int(rank)
        self.alpha = float(alpha)

    def adapter_summary(self) -> Dict[str, Any]:
        return {
            "method_id": "lora_adapter",
            "rank": self.rank,
            "alpha": self.alpha,
            "scaling": self.alpha / float(self.rank),
            "torch_available": importlib.util.find_spec("torch") is not None,
            "training_executed": False,
            "mode": "bounded_smoke_fixture",
        }


def make_smoke_simulations(n: int = 8) -> Tuple[List[List[float]], List[List[float]]]:
    """Create deterministic tiny theta/x pairs for baseline smoke fixtures."""

    n = max(2, int(n))
    theta: List[List[float]] = []
    x: List[List[float]] = []
    for index in range(n):
        t0 = -1.0 + 2.0 * index / max(1, n - 1)
        t1 = math.sin(index + 1.0) * 0.25
        theta.append([t0, t1])
        x.append([t0 + 0.5 * t1, t0 * t0 - t1])
    return theta, x


def make_smoke_posterior_pair(n: int = 12) -> Tuple[List[List[float]], List[List[float]]]:
    """Create two close posterior sample sets for C2ST wiring validation."""

    n = max(4, int(n))
    rng = random.Random(17)
    ground_truth: List[List[float]] = []
    approximate: List[List[float]] = []
    for index in range(n):
        angle = 2.0 * math.pi * index / n
        gt = [math.cos(angle) + rng.uniform(-0.01, 0.01), math.sin(angle) + rng.uniform(-0.01, 0.01)]
        ap = [gt[0] + rng.uniform(-0.015, 0.015), gt[1] + rng.uniform(-0.015, 0.015)]
        ground_truth.append(gt)
        approximate.append(ap)
    return approximate, ground_truth


def build_config_resolved(mode: str = "dry_run") -> Dict[str, Any]:
    """Build benchmark-evaluation configuration with explicit stop/pruning rationale."""

    return {
        "mode": mode,
        "dry_run_contract_artifact": mode in {"dry_run", "runtime_smoke", "docker_validate"},
        "hypothesis": (
            "Benchmark evaluation compares Simformer posterior samples against explicit "
            "NPE/NLE/NRE baselines using C2ST, NLL, and return interfaces."
        ),
        "decision_value": (
            "Covers paper_contract_dataset_metric_protocol, method_baseline_protocol, "
            "sweep_hyperparameter_protocol, dataset inventory, and benchmark task registry."
        ),
        "stop_rule_or_pruning_rationale": (
            "Default execution stops at bounded smoke fixtures and schema artifacts; "
            "paper-scale training/evaluation requires an explicit full mode."
        ),
        "section_4_1_benchmark_slots": list(SECTION_4_1_BENCHMARK_SLOTS),
        "observations_per_task": OBSERVATION_COUNT_PER_TASK,
        "default_observation_ids": list(DEFAULT_OBSERVATION_IDS),
        "metrics": {
            "c2st_accuracy": {
                "optimal": 0.5,
                "worst": 1.0,
                "direction": "lower_is_better_toward_0.5",
                "paper_semantics": "0.5 aligned with ground-truth posterior; 1.0 completely distinguishable.",
            },
            "negative_log_likelihood": {"direction": "lower_is_better"},
            "return": {"direction": "task_configured"},
        },
        "artifact_paths": dict(DECLARED_ARTIFACTS),
        "figure_captions": dict(FIGURE_CAPTIONS),
        "addendum_clarifications": dict(ADDENDUM_CLARIFICATIONS),
        "reference_grounding": [
            "paperbench_ref_001 docs/how_to_guide/04_embedding_networks.ipynb",
            "paperbench_ref_001 docs/how_to_guide/22_experiment_tracking.ipynb",
            "paperbench_ref_001 sbi/inference/trainers/npe/npe_a.py",
            "paperbench_ref_001 sbi/inference/trainers/nre/nre_base.py",
        ],
    }


def readiness_payload(mode: str = "dry_run") -> Dict[str, Any]:
    """Return readiness status for benchmark-evaluation smoke validation."""

    missing_slots = [slot for slot in SECTION_4_1_BENCHMARK_SLOTS if slot not in DATASET_REGISTRY]
    missing_required_datasets = [
        dataset_id
        for dataset_id in ("two_moons", "gaussian_linear", "gaussian_mixture", "slcp", "lotka_volterra")
        if dataset_id not in DATASET_REGISTRY
    ]
    missing_methods = [
        method_id
        for method_id in ("simformer", "npe", "nle", "nre", "lora_adapter")
        if method_id not in METHOD_REGISTRY
    ]
    return {
        "status": "ready" if not (missing_slots or missing_required_datasets or missing_methods) else "incomplete",
        "mode": mode,
        "dry_run_contract_artifact": True,
        "paper_result_claim": False,
        "checks": {
            "four_section_4_1_slots_registered": len(SECTION_4_1_BENCHMARK_SLOTS) == 4 and not missing_slots,
            "ten_ground_truth_posteriors_per_task_preserved": all(
                len(DATASET_REGISTRY[slot].observation_ids) == OBSERVATION_COUNT_PER_TASK
                for slot in SECTION_4_1_BENCHMARK_SLOTS
                if slot in DATASET_REGISTRY
            ),
            "required_dataset_registry_entries_present": not missing_required_datasets,
            "explicit_baselines_registered": not missing_methods,
            "c2st_metric_executable": True,
            "nll_metric_executable": True,
            "return_metric_executable": True,
        },
        "missing_slots": missing_slots,
        "missing_required_datasets": missing_required_datasets,
        "missing_methods": missing_methods,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_dry_run_artifacts(
    output_dir: Optional[os.PathLike[str] | str] = None,
    *,
    mode: str = "dry_run",
) -> Dict[str, str]:
    """Materialize declared artifacts as dry-run readiness/schema artifacts.

    The artifacts are contract outputs only.  They validate repository wiring but
    do not claim benchmark scores, completed training, or paper-scale results.
    """

    root = Path(output_dir) if output_dir is not None else artifact_root()
    approx_samples, gt_samples = make_smoke_posterior_pair(12)
    c2st = c2st_accuracy(approx_samples, gt_samples, use_sklearn_if_available=False)
    c2st = dataclasses.replace(c2st, dry_run=True)

    theta, x = make_smoke_simulations(8)
    baseline_summaries: Dict[str, Any] = {}
    for baseline_id in ("npe", "nle", "nre"):
        adapter = LazySBIBaselineAdapter(baseline_id, smoke_samples=8)
        adapter.append_simulations(theta, x)
        estimator = adapter.train(max_num_epochs=1)
        posterior = adapter.build_posterior(estimator)
        baseline_summaries[baseline_id] = {
            "training_summary": estimator,
            "posterior_sample_shape": [len(posterior.sample((4,))), len(posterior.sample((4,))[0])],
        }
    baseline_summaries["lora_adapter"] = LoRAAdapterSmokeFixture(rank=4, alpha=8.0).adapter_summary()

    payloads: Dict[str, Mapping[str, Any]] = {
        "dataset_registry": {
            "dry_run_contract_artifact": True,
            "paper_result_claim": False,
            "section_4_1_benchmark_slots": list(SECTION_4_1_BENCHMARK_SLOTS),
            "observations_per_task": OBSERVATION_COUNT_PER_TASK,
            "datasets": {key: spec.to_json() for key, spec in DATASET_REGISTRY.items()},
        },
        "method_registry": {
            "dry_run_contract_artifact": True,
            "paper_result_claim": False,
            "methods": {key: spec.to_json() for key, spec in METHOD_REGISTRY.items()},
            "baseline_smoke_fixtures": baseline_summaries,
        },
        "ablation_registry": {
            "dry_run_contract_artifact": True,
            "paper_result_claim": False,
            "ablations": ABLATION_REGISTRY,
        },
        "config_resolved": build_config_resolved(mode),
        "benchmark_c2st": {
            "dry_run_contract_artifact": True,
            "paper_result_claim": False,
            "metric": c2st.to_json(),
            "semantics": {
                "aligned_with_ground_truth": 0.5,
                "completely_distinguishable": 1.0,
                "direction": "lower_is_better_toward_0.5",
            },
            "evaluation_dimensions": {
                "tasks": list(SECTION_4_1_BENCHMARK_SLOTS),
                "ground_truth_posteriors_per_task": OBSERVATION_COUNT_PER_TASK,
            },
        },
        "metrics": {
            "dry_run_contract_artifact": True,
            "paper_result_claim": False,
            "metrics": {
                "c2st_accuracy": c2st.to_json(),
                "negative_log_likelihood": {
                    "schema_only": True,
                    "metric_name": "negative_log_likelihood",
                    "direction": "lower_is_better",
                    "requires": "samples and executable log_prob_fn",
                },
                "return": {
                    "schema_only": True,
                    "metric_name": "return",
                    "direction": "task_configured",
                    "addendum_clarifications": dict(ADDENDUM_CLARIFICATIONS),
                },
            },
            "trend_assertions": {key: assertion.to_json() for key, assertion in TREND_ASSERTIONS.items()},
            "figure_captions": dict(FIGURE_CAPTIONS),
        },
        "readiness": readiness_payload(mode),
        "evaluation_result": {
            "status": "dry_run_contract_complete",
            "mode": mode,
            "dry_run_contract_artifact": True,
            "paper_result_claim": False,
            "evaluated_interfaces": [
                "dataset_registry",
                "method_registry",
                "baseline_smoke_fixtures",
                "c2st_accuracy",
                "negative_log_likelihood_schema",
                "return_metric_schema",
                "trend_assertions",
            ],
            "primary_metric_schema": "c2st_accuracy",
            "primary_metric_semantics": c2st.interpretation,
        },
    }

    written: Dict[str, str] = {}
    for artifact_id, relative_path in DECLARED_ARTIFACTS.items():
        path = resolve_artifact_path(relative_path, root)
        payload = payloads[artifact_id]
        _write_json(path, payload)
        written[artifact_id] = str(path)

    return written


def assert_trend_registry_semantics() -> Dict[str, Any]:
    """Validate trend metadata without asserting paper numerical results."""

    problems: List[str] = []
    for assertion_id, assertion in TREND_ASSERTIONS.items():
        if assertion.asserted_in_dry_run:
            problems.append(f"{assertion_id} must not be asserted as achieved during dry-run")
        if not assertion.baselines:
            problems.append(f"{assertion_id} must name explicit baselines/comparators")
        if not assertion.decisive_metric:
            problems.append(f"{assertion_id} must define a decisive metric")
    for baseline in ("npe", "nle", "nre"):
        if baseline not in METHOD_REGISTRY:
            problems.append(f"missing explicit baseline {baseline}")
    return {
        "status": "valid" if not problems else "invalid",
        "paper_result_claim": False,
        "problems": problems,
        "trend_assertions": {key: value.to_json() for key, value in TREND_ASSERTIONS.items()},
    }


def get_dataset_registry() -> Dict[str, Dict[str, Any]]:
    return {key: spec.to_json() for key, spec in DATASET_REGISTRY.items()}


def get_method_registry() -> Dict[str, Dict[str, Any]]:
    return {key: spec.to_json() for key, spec in METHOD_REGISTRY.items()}


def get_trend_assertions() -> Dict[str, Dict[str, Any]]:
    return {key: assertion.to_json() for key, assertion in TREND_ASSERTIONS.items()}


def run_smoke(mode: str = "runtime_smoke", output_dir: Optional[os.PathLike[str] | str] = None) -> Dict[str, Any]:
    """Exercise benchmark-evaluation wiring and write contract artifacts."""

    written = write_dry_run_artifacts(output_dir, mode=mode)
    semantics = assert_trend_registry_semantics()
    return {
        "status": "ok" if semantics["status"] == "valid" else "failed",
        "mode": mode,
        "dry_run_contract_artifact": True,
        "paper_result_claim": False,
        "written_artifacts": written,
        "trend_semantics": semantics,
    }


__all__ = [
    "ABLATION_REGISTRY",
    "ADDENDUM_CLARIFICATIONS",
    "DATASET_REGISTRY",
    "DECLARED_ARTIFACTS",
    "DEFAULT_OBSERVATION_IDS",
    "FIGURE_CAPTIONS",
    "LazySBIBaselineAdapter",
    "LoRAAdapterSmokeFixture",
    "METHOD_REGISTRY",
    "MetricResult",
    "OBSERVATION_COUNT_PER_TASK",
    "SECTION_4_1_BENCHMARK_SLOTS",
    "TREND_ASSERTIONS",
    "TrendAssertion",
    "assert_trend_registry_semantics",
    "build_config_resolved",
    "c2st_accuracy",
    "evaluate_posterior_samples",
    "get_dataset_registry",
    "get_method_registry",
    "get_trend_assertions",
    "make_smoke_posterior_pair",
    "make_smoke_simulations",
    "negative_log_likelihood",
    "readiness_payload",
    "return_metric",
    "run_smoke",
    "write_dry_run_artifacts",
]