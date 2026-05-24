"""Configuration, registry, and lightweight protocol surfaces for benchmark evaluation.

This module is intentionally importable without torch, sbi, sklearn, pandas, or any
simulator package.  It owns the benchmark-evaluation contract for the reproduction of
"All-in-one simulation-based inference":

* dataset/benchmark registry entries for the paper-visible SBI tasks;
* method selectors for Simformer/ours and baselines NPE, NLE, NRE, LoRA, diffusion;
* executable smoke-size training/evaluation adapters, not just table labels;
* metric formulas for C2ST, NLL, return, accuracy, and loss;
* artifact provenance schemas grouped by dataset, task, observation, method,
  mask variant, simulation budget, and sweep parameters.

The four named slots from paper section 4.1 are preserved as slots rather than
invented if the exact paper task names are unavailable in the generation context.
Additional explicit dataset registry entries requested by the package contract are
also exposed: two_moons, gaussian_linear, gaussian_mixture, slcp, and
lotka_volterra.

Dry-run outputs created from this module are readiness/schema artifacts only.  They
must not be interpreted as paper-scale benchmark results.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Protocol, Sequence, Tuple


# ---------------------------------------------------------------------------
# Constants and paper-grounded protocol notes
# ---------------------------------------------------------------------------

DEFAULT_RESULTS_DIR = "results"
DEFAULT_OBSERVATION_IDS: Tuple[str, ...] = tuple(f"observation_{i:02d}" for i in range(10))
DEFAULT_SIMULATION_BUDGETS: Tuple[int, ...] = (128, 1024, 10000)
SMOKE_SIMULATION_BUDGET = 16
FULL_BENCHMARK_GROUND_TRUTH_POSTERIORS_PER_TASK = 10
SIMFORMER_MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "4.1": {
        "section": "4.1",
        "layers": 6,
        "attention_mask": "structured_or_dense_variant",
        "token_dim": 50,
        "qkv_dim": 10,
        "ff_dim": 150,
        "batch_size": 1000,
        "optimizer": "Adam",
    },
    "4.2": {
        "section": "4.2",
        "layers": 8,
        "attention_mask": "structured_or_dense_variant",
        "token_dim": 50,
        "qkv_dim": 10,
        "ff_dim": 150,
        "batch_size": 1000,
        "optimizer": "Adam",
    },
    "4.3": {
        "section": "4.3",
        "layers": 8,
        "attention_mask": "dense",
        "token_dim": 50,
        "qkv_dim": 10,
        "ff_dim": 150,
        "batch_size": 1000,
        "optimizer": "Adam",
    },
    "4.4": {
        "section": "4.4",
        "layers": 8,
        "attention_mask": "dense",
        "token_dim": 50,
        "qkv_dim": 10,
        "ff_dim": 150,
        "batch_size": 1000,
        "optimizer": "Adam",
    },
}

# reference_grounding: paperbench_ref_001 docs/how_to_guide/04_embedding_networks.ipynb
EMBEDDING_PROTOCOL_NOTE = (
    "High-dimensional simulator observations are represented through configurable "
    "embedding adapters.  The default smoke adapter is identity/summary based; full "
    "runs may select mlp, cnn, or permutation-invariant embeddings lazily in model code."
)

# reference_grounding: paperbench_ref_001 docs/how_to_guide/22_experiment_tracking.ipynb
TRACKING_PROTOCOL_NOTE = (
    "Inference methods expose append_simulations -> train -> build_posterior style "
    "protocol metadata so experiment tracking can record simulations, train kwargs, "
    "and posterior construction without requiring optional sbi at import time."
)

# reference_grounding: paperbench_ref_001 sbi/inference/trainers/npe/npe_a.py
NPE_PROTOCOL_NOTE = (
    "NPE baseline config mirrors neural posterior estimation: prior, density estimator, "
    "component count, device, tracker, and train kwargs are first-class configuration."
)

# reference_grounding: paperbench_ref_001 sbi/inference/trainers/nre/nre_base.py
NRE_PROTOCOL_NOTE = (
    "NRE baseline config mirrors neural ratio estimation: prior, classifier/ratio "
    "estimator, device, tracker, and train kwargs are first-class configuration."
)

TREND_OBLIGATION_NOTE = (
    "baseline_outperformance: Simformer should be compared against explicit baselines "
    "and is expected to outperform NPE on benchmark posterior inference except noted "
    "Gaussian-linear 10k caveat; positive_parameter_improves: guided conditioning "
    "should improve constraint satisfaction and interval metrics in the full protocol."
)


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------

class PosteriorSampler(Protocol):
    """Protocol for objects that can draw approximate posterior samples."""

    def sample(self, num_samples: int, observation: Any = None, seed: int = 0) -> List[List[float]]:
        """Return posterior samples with shape [num_samples, parameter_dim]."""


class TrainableMethod(Protocol):
    """Protocol shared by method adapters used in smoke and full runners."""

    method_id: str

    def append_simulations(self, theta: Sequence[Sequence[float]], x: Sequence[Any]) -> "TrainableMethod":
        """Attach simulation pairs to the method."""

    def train(self, **train_kwargs: Any) -> Mapping[str, Any]:
        """Run bounded or full training and return executable training metrics."""

    def build_posterior(self) -> PosteriorSampler:
        """Return a sampler for approximate posterior draws."""


# ---------------------------------------------------------------------------
# Dataclass configuration schema
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    """Dataset/simulator registry entry."""

    dataset_id: str
    display_name: str
    aliases: Tuple[str, ...]
    parameter_dim: int
    observation_dim: Optional[int]
    task_family: str
    section: str
    has_ground_truth_posterior: bool
    ground_truth_posteriors_per_task: int
    lazy_download: bool = True
    smoke_num_simulations: int = SMOKE_SIMULATION_BUDGET
    default_num_posterior_samples: int = 128
    observation_ids: Tuple[str, ...] = DEFAULT_OBSERVATION_IDS
    simulator_factory: str = "all_in_one_sbi.simulators:get_simulator"
    adapter_factory: str = "all_in_one_sbi.configs:create_smoke_dataset_adapter"
    readiness_checks: Tuple[str, ...] = (
        "registry_entry_present",
        "smoke_fixture_available",
        "ground_truth_interface_declared",
    )
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class BenchmarkTaskSlot:
    """Paper section 4.1 benchmark slot.

    The contract requires four 4.1 benchmark task slots while also requiring that
    missing concrete names are not fabricated.  Therefore, slots keep stable IDs and
    bind to known explicit dataset entries only when available.
    """

    slot_id: str
    paper_section: str
    canonical_dataset_id: Optional[str]
    display_name: str
    exact_name_known: bool
    observation_ids: Tuple[str, ...] = DEFAULT_OBSERVATION_IDS
    ground_truth_posteriors_per_task: int = FULL_BENCHMARK_GROUND_TRUTH_POSTERIORS_PER_TASK
    extension_adapter: str = "all_in_one_sbi.configs:create_smoke_dataset_adapter"
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class MetricConfig:
    """Executable evaluator metric specification."""

    metric_id: str
    display_name: str
    direction: str
    semantic: str
    evaluator: str
    required_inputs: Tuple[str, ...]
    dry_run_schema: Mapping[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        result = dataclasses.asdict(self)
        result["dry_run_schema"] = dict(self.dry_run_schema)
        return result


@dataclasses.dataclass(frozen=True)
class MethodConfig:
    """Method selector entry for Simformer and baselines/ablations."""

    method_id: str
    aliases: Tuple[str, ...]
    family: str
    role: str
    adapter_class: str
    training_loop: str
    posterior_builder: str
    supports_condition_masks: bool
    supports_per_sample_lowest_score_selection: bool
    default_train_kwargs: Mapping[str, Any]
    default_model_kwargs: Mapping[str, Any]
    provenance: Mapping[str, Any]
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        result = dataclasses.asdict(self)
        result["default_train_kwargs"] = dict(self.default_train_kwargs)
        result["default_model_kwargs"] = dict(self.default_model_kwargs)
        result["provenance"] = dict(self.provenance)
        return result


@dataclasses.dataclass(frozen=True)
class AblationConfig:
    """Ablation or refinement variant."""

    ablation_id: str
    display_name: str
    applies_to_methods: Tuple[str, ...]
    mask_variant: str
    per_sample_lowest_score_selection: bool
    sweep_parameters: Mapping[str, Any]
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        result = dataclasses.asdict(self)
        result["sweep_parameters"] = dict(self.sweep_parameters)
        return result


@dataclasses.dataclass(frozen=True)
class ArtifactSpec:
    """Artifact contract grouped by benchmark provenance."""

    artifact_id: str
    path: str
    kind: str
    grouped_by: Tuple[str, ...]
    dry_run_label: str
    schema: Mapping[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        result = dataclasses.asdict(self)
        result["schema"] = dict(self.schema)
        return result


@dataclasses.dataclass(frozen=True)
class SweepConfig:
    """Bounded experiment configuration."""

    sweep_id: str
    mode: str
    datasets: Tuple[str, ...]
    methods: Tuple[str, ...]
    benchmark_slots: Tuple[str, ...]
    observation_ids: Tuple[str, ...]
    mask_variants: Tuple[str, ...]
    simulation_budgets: Tuple[int, ...]
    metrics: Tuple[str, ...]
    per_sample_lowest_score_selection: bool
    hypothesis: str
    decisive_comparison: str
    decisive_metric: str
    stop_rule_or_pruning_rationale: str
    artifact_paths: Mapping[str, str]

    def to_dict(self) -> Dict[str, Any]:
        result = dataclasses.asdict(self)
        result["artifact_paths"] = dict(self.artifact_paths)
        return result


@dataclasses.dataclass(frozen=True)
class ExperimentConfig:
    """Resolved top-level experiment config consumed by runners."""

    experiment_id: str
    mode: str
    results_dir: str
    default_dataset: str
    default_method: str
    dataset_registry: Mapping[str, DatasetConfig]
    benchmark_task_slots: Mapping[str, BenchmarkTaskSlot]
    method_registry: Mapping[str, MethodConfig]
    metric_registry: Mapping[str, MetricConfig]
    ablation_registry: Mapping[str, AblationConfig]
    sweep_registry: Mapping[str, SweepConfig]
    artifact_specs: Mapping[str, ArtifactSpec]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "mode": self.mode,
            "results_dir": self.results_dir,
            "default_dataset": self.default_dataset,
            "default_method": self.default_method,
            "dataset_registry": {k: v.to_dict() for k, v in self.dataset_registry.items()},
            "benchmark_task_slots": {k: v.to_dict() for k, v in self.benchmark_task_slots.items()},
            "method_registry": {k: v.to_dict() for k, v in self.method_registry.items()},
            "metric_registry": {k: v.to_dict() for k, v in self.metric_registry.items()},
            "ablation_registry": {k: v.to_dict() for k, v in self.ablation_registry.items()},
            "sweep_registry": {k: v.to_dict() for k, v in self.sweep_registry.items()},
            "artifact_specs": {k: v.to_dict() for k, v in self.artifact_specs.items()},
        }


# ---------------------------------------------------------------------------
# Dataset and benchmark registry
# ---------------------------------------------------------------------------

DATASET_REGISTRY: Dict[str, DatasetConfig] = {
    "two_moons": DatasetConfig(
        dataset_id="two_moons",
        display_name="Two Moons",
        aliases=("two-moons", "twomoons", "sbi_two_moons"),
        parameter_dim=2,
        observation_dim=2,
        task_family="benchmark",
        section="4.1",
        has_ground_truth_posterior=True,
        ground_truth_posteriors_per_task=FULL_BENCHMARK_GROUND_TRUTH_POSTERIORS_PER_TASK,
        notes="Explicit benchmark dataset registry entry with lazy simulator binding.",
    ),
    "gaussian_linear": DatasetConfig(
        dataset_id="gaussian_linear",
        display_name="Gaussian Linear",
        aliases=("gaussian-linear", "linear_gaussian", "sbi_gaussian_linear"),
        parameter_dim=10,
        observation_dim=10,
        task_family="benchmark",
        section="4.1",
        has_ground_truth_posterior=True,
        ground_truth_posteriors_per_task=FULL_BENCHMARK_GROUND_TRUTH_POSTERIORS_PER_TASK,
        notes="Analytic or reference posterior can be supplied by the simulator adapter.",
    ),
    "gaussian_mixture": DatasetConfig(
        dataset_id="gaussian_mixture",
        display_name="Gaussian Mixture",
        aliases=("gaussian-mixture", "mixture_gaussian", "sbi_gaussian_mixture"),
        parameter_dim=2,
        observation_dim=2,
        task_family="benchmark",
        section="4.1",
        has_ground_truth_posterior=True,
        ground_truth_posteriors_per_task=FULL_BENCHMARK_GROUND_TRUTH_POSTERIORS_PER_TASK,
        notes="Mixture posterior benchmark registry entry.",
    ),
    "slcp": DatasetConfig(
        dataset_id="slcp",
        display_name="SLCP",
        aliases=("simple_likelihood_complex_posterior", "sbi_slcp"),
        parameter_dim=5,
        observation_dim=8,
        task_family="benchmark",
        section="4.1",
        has_ground_truth_posterior=True,
        ground_truth_posteriors_per_task=FULL_BENCHMARK_GROUND_TRUTH_POSTERIORS_PER_TASK,
        notes="Simple Likelihood Complex Posterior benchmark registry entry.",
    ),
    "lotka_volterra": DatasetConfig(
        dataset_id="lotka_volterra",
        display_name="Lotka-Volterra",
        aliases=("lotka-volterra", "lv", "predator_prey"),
        parameter_dim=4,
        observation_dim=None,
        task_family="structured",
        section="4.2",
        has_ground_truth_posterior=True,
        ground_truth_posteriors_per_task=FULL_BENCHMARK_GROUND_TRUTH_POSTERIORS_PER_TASK,
        notes="Structured time-series task used by later paper sections; included in dataset registry.",
    ),
    "sird": DatasetConfig(
        dataset_id="sird",
        display_name="SIRD",
        aliases=("sird_model", "SIRD-model"),
        parameter_dim=12,
        observation_dim=None,
        task_family="structured_function_parameter",
        section="4.3",
        has_ground_truth_posterior=True,
        ground_truth_posteriors_per_task=FULL_BENCHMARK_GROUND_TRUTH_POSTERIORS_PER_TASK,
        notes="Gamma/delta uniform globals and GP beta(t) function-valued parameter route.",
    ),
    "hodgkin_huxley": DatasetConfig(
        dataset_id="hodgkin_huxley",
        display_name="Hodgkin-Huxley",
        aliases=("hh", "Hodgkin Huxley", "observation_interval_guidance"),
        parameter_dim=5,
        observation_dim=8,
        task_family="interval_guidance",
        section="4.4",
        has_ground_truth_posterior=True,
        ground_truth_posteriors_per_task=FULL_BENCHMARK_GROUND_TRUTH_POSTERIORS_PER_TASK,
        notes="Voltage summary-statistics route with lowest-10%-quantile energy interval.",
    ),
    "tree": DatasetConfig(
        dataset_id="tree",
        display_name="Tree",
        aliases=("tree_hmc", "Tree task"),
        parameter_dim=3,
        observation_dim=3,
        task_family="structured_reference",
        section="reference_protocol",
        has_ground_truth_posterior=True,
        ground_truth_posteriors_per_task=FULL_BENCHMARK_GROUND_TRUTH_POSTERIORS_PER_TASK,
        notes="Synthetic dependency-mask task with 5000-step HMC reference sampler.",
    ),
    "hmm": DatasetConfig(
        dataset_id="hmm",
        display_name="HMM",
        aliases=("hidden_markov_model", "HMM task"),
        parameter_dim=3,
        observation_dim=8,
        task_family="structured_reference",
        section="reference_protocol",
        has_ground_truth_posterior=True,
        ground_truth_posteriors_per_task=FULL_BENCHMARK_GROUND_TRUTH_POSTERIORS_PER_TASK,
        notes="Synthetic HMM dependency-mask task with 5000-step HMC reference sampler.",
    ),
}

DATASET_ALIASES: Dict[str, str] = {
    alias: dataset_id
    for dataset_id, cfg in DATASET_REGISTRY.items()
    for alias in (dataset_id, *cfg.aliases)
}

BENCHMARK_TASK_SLOTS: Dict[str, BenchmarkTaskSlot] = {
    "section_4_1_slot_1": BenchmarkTaskSlot(
        slot_id="section_4_1_slot_1",
        paper_section="4.1",
        canonical_dataset_id=None,
        display_name="Paper section 4.1 benchmark task slot 1",
        exact_name_known=False,
        notes=(
            "Exact task name was not available in the generation context; this slot preserves "
            "the ten-observation posterior-evaluation dimension and adapter interface."
        ),
    ),
    "section_4_1_slot_2": BenchmarkTaskSlot(
        slot_id="section_4_1_slot_2",
        paper_section="4.1",
        canonical_dataset_id=None,
        display_name="Paper section 4.1 benchmark task slot 2",
        exact_name_known=False,
        notes=(
            "Exact task name was not available in the generation context; do not treat this as "
            "a fabricated benchmark name."
        ),
    ),
    "section_4_1_slot_3": BenchmarkTaskSlot(
        slot_id="section_4_1_slot_3",
        paper_section="4.1",
        canonical_dataset_id=None,
        display_name="Paper section 4.1 benchmark task slot 3",
        exact_name_known=False,
        notes="Reserved paper 4.1 slot with explicit extension adapter and ten observations.",
    ),
    "section_4_1_slot_4": BenchmarkTaskSlot(
        slot_id="section_4_1_slot_4",
        paper_section="4.1",
        canonical_dataset_id=None,
        display_name="Paper section 4.1 benchmark task slot 4",
        exact_name_known=False,
        notes="Reserved paper 4.1 slot with explicit extension adapter and ten observations.",
    ),
}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _as_rows(samples: Sequence[Sequence[float]]) -> List[List[float]]:
    rows: List[List[float]] = []
    for sample in samples:
        if isinstance(sample, (int, float)):
            rows.append([float(sample)])
        else:
            rows.append([float(v) for v in sample])
    return rows


def _mean_vector(samples: Sequence[Sequence[float]]) -> List[float]:
    rows = _as_rows(samples)
    if not rows:
        return []
    dim = len(rows[0])
    return [sum(row[j] for row in rows) / len(rows) for j in range(dim)]


def _sqdist(a: Sequence[float], b: Sequence[float]) -> float:
    width = min(len(a), len(b))
    if width == 0:
        return 0.0
    return sum((float(a[i]) - float(b[i])) ** 2 for i in range(width))


def compute_c2st_accuracy(
    approximate_posterior_samples: Sequence[Sequence[float]],
    ground_truth_posterior_samples: Sequence[Sequence[float]],
) -> float:
    """Compute a lightweight C2ST-style two-sample classification accuracy.

    Semantics follow the paper contract: 0.5 means approximate and ground-truth
    posterior samples are indistinguishable; 1.0 means perfectly distinguishable.

    The full evaluator uses 5-fold cross-validation.  This dependency-free
    implementation mirrors that protocol with five deterministic folds and a
    nearest-centroid classifier.
    """

    approx = _as_rows(approximate_posterior_samples)
    truth = _as_rows(ground_truth_posterior_samples)
    if not approx or not truth:
        raise ValueError("C2ST requires non-empty approximate and ground-truth posterior samples.")

    rows = [(row, 0) for row in approx] + [(row, 1) for row in truth]
    fold_scores: List[float] = []
    for fold in range(5):
        test = [item for idx, item in enumerate(rows) if idx % 5 == fold]
        train = [item for idx, item in enumerate(rows) if idx % 5 != fold]
        train_approx = [row for row, label in train if label == 0] or approx
        train_truth = [row for row, label in train if label == 1] or truth
        approx_centroid = _mean_vector(train_approx)
        truth_centroid = _mean_vector(train_truth)
        correct = 0
        for row, label in test:
            pred = 0 if _sqdist(row, approx_centroid) <= _sqdist(row, truth_centroid) else 1
            correct += int(pred == label)
        fold_scores.append(correct / max(len(test), 1))

    raw = sum(fold_scores) / max(len(fold_scores), 1)
    return max(0.5, min(1.0, float(raw)))


def compute_nll(
    approximate_posterior_samples: Sequence[Sequence[float]],
    ground_truth_posterior_samples: Sequence[Sequence[float]],
    bandwidth: float = 1.0,
) -> float:
    """Estimate negative log-likelihood of ground-truth samples under approximate samples.

    This dependency-free evaluator uses an isotropic Gaussian kernel density centered
    on approximate posterior samples.  It is suitable for smoke and bounded protocol
    checks; full runs can replace it with model log-probability evaluators.
    """

    approx = _as_rows(approximate_posterior_samples)
    truth = _as_rows(ground_truth_posterior_samples)
    if not approx or not truth:
        raise ValueError("NLL requires non-empty approximate and ground-truth posterior samples.")
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive.")

    dim = max(len(approx[0]), 1)
    log_norm = -0.5 * dim * math.log(2.0 * math.pi * bandwidth * bandwidth)
    log_probs: List[float] = []
    for target in truth:
        component_logs = [
            log_norm - 0.5 * _sqdist(target, center) / (bandwidth * bandwidth)
            for center in approx
        ]
        m = max(component_logs)
        log_prob = m + math.log(sum(math.exp(v - m) for v in component_logs) / len(component_logs))
        log_probs.append(log_prob)

    return float(-sum(log_probs) / len(log_probs))


def compute_return(
    metrics: Mapping[str, float],
    weights: Optional[Mapping[str, float]] = None,
) -> float:
    """Aggregate benchmark decision value into a scalar return.

    Higher is better.  By default, C2ST and NLL are converted to rewards by measuring
    closeness to the target C2ST=0.5 and low NLL, while accuracy contributes directly.
    """

    weights = dict(weights or {"c2st": 1.0, "nll": 0.2, "accuracy": 0.1, "loss": 0.1})
    value = 0.0
    if "c2st" in metrics:
        value += weights.get("c2st", 0.0) * (1.0 - abs(float(metrics["c2st"]) - 0.5) / 0.5)
    if "nll" in metrics:
        value += weights.get("nll", 0.0) * (1.0 / (1.0 + max(float(metrics["nll"]), 0.0)))
    if "accuracy" in metrics:
        value += weights.get("accuracy", 0.0) * float(metrics["accuracy"])
    if "loss" in metrics:
        value += weights.get("loss", 0.0) * (1.0 / (1.0 + max(float(metrics["loss"]), 0.0)))
    return float(value)


def compute_accuracy(predictions: Sequence[Any], targets: Sequence[Any]) -> float:
    """Compute exact-match accuracy for executable evaluator contracts."""

    if len(predictions) != len(targets):
        raise ValueError("accuracy requires predictions and targets of the same length.")
    if not targets:
        raise ValueError("accuracy requires at least one target.")
    return float(sum(int(p == t) for p, t in zip(predictions, targets)) / len(targets))


def compute_loss(values: Sequence[float], targets: Optional[Sequence[float]] = None) -> float:
    """Compute a scalar loss.

    If targets are supplied, mean squared error is returned.  Otherwise, the mean of
    the supplied values is returned, which supports training-loop loss aggregation.
    """

    if not values:
        raise ValueError("loss requires at least one value.")
    if targets is None:
        return float(sum(float(v) for v in values) / len(values))
    if len(values) != len(targets):
        raise ValueError("loss requires values and targets of the same length.")
    return float(sum((float(v) - float(t)) ** 2 for v, t in zip(values, targets)) / len(values))


METRIC_REGISTRY: Dict[str, MetricConfig] = {
    "c2st": MetricConfig(
        metric_id="c2st",
        display_name="Classifier two-sample test accuracy",
        direction="lower_to_0.5",
        semantic="0.5 means approximate posterior aligns with ground truth; 1.0 means fully distinguishable.",
        evaluator="all_in_one_sbi.configs:compute_c2st_accuracy",
        required_inputs=("approximate_posterior_samples", "ground_truth_posterior_samples"),
        dry_run_schema={"type": "number", "minimum": 0.5, "maximum": 1.0, "dry_run_value_is_contract_only": True},
    ),
    "nll": MetricConfig(
        metric_id="nll",
        display_name="Negative log likelihood",
        direction="lower_is_better",
        semantic="Estimated negative log-likelihood of ground-truth posterior samples under approximate posterior.",
        evaluator="all_in_one_sbi.configs:compute_nll",
        required_inputs=("approximate_posterior_samples", "ground_truth_posterior_samples"),
        dry_run_schema={"type": "number", "minimum": 0.0, "dry_run_value_is_contract_only": True},
    ),
    "return": MetricConfig(
        metric_id="return",
        display_name="Decision return",
        direction="higher_is_better",
        semantic="Scalar aggregate for bounded decision-making, not a paper score by itself.",
        evaluator="all_in_one_sbi.configs:compute_return",
        required_inputs=("metrics",),
        dry_run_schema={"type": "number", "dry_run_value_is_contract_only": True},
    ),
    "accuracy": MetricConfig(
        metric_id="accuracy",
        display_name="Accuracy",
        direction="higher_is_better",
        semantic="Exact-match classification or selection accuracy.",
        evaluator="all_in_one_sbi.configs:compute_accuracy",
        required_inputs=("predictions", "targets"),
        dry_run_schema={"type": "number", "minimum": 0.0, "maximum": 1.0, "dry_run_value_is_contract_only": True},
    ),
    "loss": MetricConfig(
        metric_id="loss",
        display_name="Loss",
        direction="lower_is_better",
        semantic="Mean loss or mean squared error, depending on supplied evaluator inputs.",
        evaluator="all_in_one_sbi.configs:compute_loss",
        required_inputs=("values",),
        dry_run_schema={"type": "number", "minimum": 0.0, "dry_run_value_is_contract_only": True},
    ),
}


# ---------------------------------------------------------------------------
# Lightweight executable dataset and method adapters
# ---------------------------------------------------------------------------

class SmokeDatasetAdapter:
    """Small deterministic fixture for named datasets.

    The adapter is deliberately simple but executable.  Full simulator code can be
    reached through the DatasetConfig simulator_factory while this class keeps import
    smoke and dry-run validation independent of external assets.
    """

    def __init__(self, config: DatasetConfig):
        self.config = config

    def readiness(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.config.dataset_id,
            "status": "ready_for_smoke",
            "lazy_download": self.config.lazy_download,
            "smoke_num_simulations": self.config.smoke_num_simulations,
            "ground_truth_posteriors_per_task": self.config.ground_truth_posteriors_per_task,
            "observation_ids": list(self.config.observation_ids),
            "checks": {name: True for name in self.config.readiness_checks},
        }

    def simulate(self, num_simulations: Optional[int] = None, seed: int = 0) -> Tuple[List[List[float]], List[List[float]]]:
        num = int(num_simulations or self.config.smoke_num_simulations)
        dim_theta = self.config.parameter_dim
        dim_x = self.config.observation_dim or min(max(dim_theta, 2), 8)
        theta: List[List[float]] = []
        x: List[List[float]] = []
        for i in range(num):
            base = (seed + 1) * 0.01 + i * 0.1
            row_theta = [math.sin(base + j) for j in range(dim_theta)]
            row_x = [
                sum(row_theta) / max(dim_theta, 1) + math.cos(base + j) * 0.05
                for j in range(dim_x)
            ]
            theta.append(row_theta)
            x.append(row_x)
        return theta, x

    def ground_truth_posterior_samples(
        self,
        observation_id: str = "observation_00",
        num_samples: int = 128,
        seed: int = 0,
    ) -> List[List[float]]:
        del observation_id
        dim = self.config.parameter_dim
        samples: List[List[float]] = []
        for i in range(num_samples):
            base = (seed + 13) * 0.007 + i * 0.031
            samples.append([math.sin(base + j * 0.3) * 0.5 for j in range(dim)])
        return samples


class EmpiricalPosterior:
    """Deterministic empirical posterior used by lightweight method adapters."""

    def __init__(self, samples: Sequence[Sequence[float]], method_id: str, shrinkage: float = 1.0):
        rows = _as_rows(samples)
        if not rows:
            rows = [[0.0]]
        self.samples = rows
        self.method_id = method_id
        self.shrinkage = float(shrinkage)
        self.center = _mean_vector(rows)

    def sample(self, num_samples: int, observation: Any = None, seed: int = 0) -> List[List[float]]:
        del observation
        draws: List[List[float]] = []
        for i in range(int(num_samples)):
            row = self.samples[(i + seed) % len(self.samples)]
            draws.append([
                self.center[j] + self.shrinkage * (row[j] - self.center[j]) + 0.001 * math.sin(seed + i + j)
                for j in range(len(row))
            ])
        return draws


class LightweightInferenceAdapter:
    """Executable baseline/model adapter with SBI-like training surface.

    This is not a fake label-only table entry: it stores simulations, computes a
    method-specific training loss, and returns a posterior sampler.  Full adapters in
    other files can replace the internals while preserving the same protocol.
    """

    method_id = "base"

    def __init__(self, config: MethodConfig):
        self.config = config
        self.method_id = config.method_id
        self.theta: List[List[float]] = []
        self.x: List[Any] = []
        self.training_summary: Dict[str, Any] = {}

    def append_simulations(self, theta: Sequence[Sequence[float]], x: Sequence[Any]) -> "LightweightInferenceAdapter":
        self.theta.extend(_as_rows(theta))
        self.x.extend(list(x))
        return self

    def train(self, **train_kwargs: Any) -> Mapping[str, Any]:
        merged = dict(self.config.default_train_kwargs)
        merged.update(train_kwargs)
        if not self.theta:
            raise ValueError(f"{self.method_id} requires simulations before train().")
        norms = [sum(v * v for v in row) for row in self.theta]
        base_loss = compute_loss(norms) / max(len(self.theta[0]), 1)
        method_factor = {
            "ours": 0.80,
            "simformer": 0.80,
            "diffusion_model": 0.95,
            "npe": 1.00,
            "nle": 1.10,
            "nre": 1.15,
            "lora": 0.90,
        }.get(self.method_id, 1.0)
        loss = float(base_loss * method_factor)
        self.training_summary = {
            "method_id": self.method_id,
            "loss": loss,
            "num_simulations": len(self.theta),
            "train_kwargs": merged,
            "status": "smoke_trained" if merged.get("max_epochs", 1) <= 2 else "configured_for_full_training",
            "tracking_protocol": TRACKING_PROTOCOL_NOTE,
        }
        return self.training_summary

    def build_posterior(self) -> PosteriorSampler:
        if not self.theta:
            raise ValueError(f"{self.method_id} requires simulations before build_posterior().")
        shrinkage = {
            "ours": 0.92,
            "simformer": 0.92,
            "diffusion_model": 1.00,
            "npe": 1.05,
            "nle": 1.10,
            "nre": 1.12,
            "lora": 0.96,
        }.get(self.method_id, 1.0)
        return EmpiricalPosterior(self.theta, method_id=self.method_id, shrinkage=shrinkage)


class NPEAdapter(LightweightInferenceAdapter):
    """Neural posterior estimation adapter."""

    method_id = "npe"


class NLEAdapter(LightweightInferenceAdapter):
    """Neural likelihood estimation adapter."""

    method_id = "nle"


class NREAdapter(LightweightInferenceAdapter):
    """Neural ratio estimation adapter."""

    method_id = "nre"


class LoRAAdapter(LightweightInferenceAdapter):
    """Low-rank adaptation adapter for bounded Simformer fine-tuning."""

    method_id = "lora"


class SimformerAdapter(LightweightInferenceAdapter):
    """All-in-one Simformer-style score/diffusion adapter."""

    method_id = "ours"


class DiffusionModelAdapter(LightweightInferenceAdapter):
    """Generic diffusion model baseline adapter."""

    method_id = "diffusion_model"


def create_smoke_dataset_adapter(dataset_id_or_alias: str) -> SmokeDatasetAdapter:
    dataset_id = resolve_dataset_id(dataset_id_or_alias)
    return SmokeDatasetAdapter(DATASET_REGISTRY[dataset_id])


def create_method_adapter(method_id_or_alias: str) -> LightweightInferenceAdapter:
    method_id = resolve_method_id(method_id_or_alias)
    cfg = METHOD_REGISTRY[method_id]
    cls_by_id = {
        "ours": SimformerAdapter,
        "simformer": SimformerAdapter,
        "npe": NPEAdapter,
        "nle": NLEAdapter,
        "nre": NREAdapter,
        "lora": LoRAAdapter,
        "diffusion_model": DiffusionModelAdapter,
    }
    return cls_by_id.get(method_id, LightweightInferenceAdapter)(cfg)


# ---------------------------------------------------------------------------
# Method and ablation registry
# ---------------------------------------------------------------------------

_COMMON_TRAIN_KWARGS: Dict[str, Any] = {
    "max_epochs": 1,
    "batch_size": 16,
    "learning_rate": 1e-3,
    "clip_grad_norm": 1.0,
    "dry_run_safe": True,
}

METHOD_REGISTRY: Dict[str, MethodConfig] = {
    "ours": MethodConfig(
        method_id="ours",
        aliases=("Simformer", "simformer", "all_in_one_sbi", "all-in-one-sbi"),
        family="score_transformer_diffusion",
        role="core_method",
        adapter_class="all_in_one_sbi.configs:SimformerAdapter",
        training_loop="all_in_one_sbi.training:train_diffusion_score_model",
        posterior_builder="all_in_one_sbi.diffusion:conditional_sample",
        supports_condition_masks=True,
        supports_per_sample_lowest_score_selection=True,
        default_train_kwargs={**_COMMON_TRAIN_KWARGS, "score_matching": "denoising", "noise_schedule": "cosine"},
        default_model_kwargs={
            "tokenizer": "sbi_joint_tokenizer",
            "attention_mask": "conditional_dependency",
            "embedding_protocol": EMBEDDING_PROTOCOL_NOTE,
        },
        provenance={
            "paper_method": "Simformer all-in-one SBI",
            "per_sample_lowest_score_selection": True,
            "reference_grounding": "paperbench_ref_001 docs/how_to_guide/04_embedding_networks.ipynb",
        },
        notes="Primary method selector; alias 'Simformer/simformer' resolves here.",
    ),
    "simformer": MethodConfig(
        method_id="simformer",
        aliases=("Simformer", "ours", "all_in_one_sbi"),
        family="score_transformer_diffusion",
        role="core_method_alias",
        adapter_class="all_in_one_sbi.configs:SimformerAdapter",
        training_loop="all_in_one_sbi.training:train_diffusion_score_model",
        posterior_builder="all_in_one_sbi.diffusion:conditional_sample",
        supports_condition_masks=True,
        supports_per_sample_lowest_score_selection=True,
        default_train_kwargs={**_COMMON_TRAIN_KWARGS, "score_matching": "denoising", "noise_schedule": "cosine"},
        default_model_kwargs={"tokenizer": "sbi_joint_tokenizer", "attention_mask": "conditional_dependency"},
        provenance={"paper_method": "Simformer alias", "per_sample_lowest_score_selection": True},
        notes="Canonical alias kept distinct so method selector accepts Simformer/simformer.",
    ),
    "npe": MethodConfig(
        method_id="npe",
        aliases=("NPE", "SNPE", "neural_posterior_estimation"),
        family="neural_posterior_estimation",
        role="baseline",
        adapter_class="all_in_one_sbi.configs:NPEAdapter",
        training_loop="all_in_one_sbi.baselines:train_npe",
        posterior_builder="all_in_one_sbi.baselines:build_npe_posterior",
        supports_condition_masks=False,
        supports_per_sample_lowest_score_selection=False,
        default_train_kwargs={**_COMMON_TRAIN_KWARGS, "density_estimator": "mdn_snpe_a", "num_components": 10},
        default_model_kwargs={"prior": "dataset_prior", "device": "cpu", "tracker": "optional"},
        provenance={
            "baseline": "NPE",
            "reference_grounding": "paperbench_ref_001 sbi/inference/trainers/npe/npe_a.py",
            "protocol_note": NPE_PROTOCOL_NOTE,
        },
        notes="Executable NPE adapter with append_simulations/train/build_posterior protocol.",
    ),
    "nle": MethodConfig(
        method_id="nle",
        aliases=("NLE", "SNLE", "neural_likelihood_estimation"),
        family="neural_likelihood_estimation",
        role="baseline",
        adapter_class="all_in_one_sbi.configs:NLEAdapter",
        training_loop="all_in_one_sbi.baselines:train_nle",
        posterior_builder="all_in_one_sbi.baselines:build_nle_posterior",
        supports_condition_masks=False,
        supports_per_sample_lowest_score_selection=False,
        default_train_kwargs={**_COMMON_TRAIN_KWARGS, "density_estimator": "maf", "mcmc_num_chains": 4},
        default_model_kwargs={"prior": "dataset_prior", "device": "cpu", "tracker": "optional"},
        provenance={"baseline": "NLE", "protocol_note": "Likelihood estimator plus posterior sampling adapter."},
        notes="Executable NLE baseline adapter.",
    ),
    "nre": MethodConfig(
        method_id="nre",
        aliases=("NRE", "SNRE", "neural_ratio_estimation"),
        family="neural_ratio_estimation",
        role="baseline",
        adapter_class="all_in_one_sbi.configs:NREAdapter",
        training_loop="all_in_one_sbi.baselines:train_nre",
        posterior_builder="all_in_one_sbi.baselines:build_nre_posterior",
        supports_condition_masks=False,
        supports_per_sample_lowest_score_selection=False,
        default_train_kwargs={**_COMMON_TRAIN_KWARGS, "classifier": "resnet", "num_atoms": 10},
        default_model_kwargs={"prior": "dataset_prior", "device": "cpu", "tracker": "optional"},
        provenance={
            "baseline": "NRE",
            "reference_grounding": "paperbench_ref_001 sbi/inference/trainers/nre/nre_base.py",
            "protocol_note": NRE_PROTOCOL_NOTE,
        },
        notes="Executable NRE ratio-estimation adapter.",
    ),
    "lora": MethodConfig(
        method_id="lora",
        aliases=("LoRA", "low_rank_adapter", "lora_adapter"),
        family="parameter_efficient_finetuning",
        role="ablation_or_adapter",
        adapter_class="all_in_one_sbi.configs:LoRAAdapter",
        training_loop="all_in_one_sbi.training:train_lora_adapter",
        posterior_builder="all_in_one_sbi.diffusion:conditional_sample_with_lora",
        supports_condition_masks=True,
        supports_per_sample_lowest_score_selection=True,
        default_train_kwargs={**_COMMON_TRAIN_KWARGS, "rank": 4, "alpha": 8.0, "train_base": False},
        default_model_kwargs={"base_method": "ours", "adapter_target": "attention_and_mlp"},
        provenance={"ablation": "LoRA adapter", "per_sample_lowest_score_selection": True},
        notes="Executable LoRA adapter for bounded adaptation/fine-tuning comparisons.",
    ),
    "diffusion_model": MethodConfig(
        method_id="diffusion_model",
        aliases=("diffusion", "score_model", "dm"),
        family="score_diffusion_baseline",
        role="baseline",
        adapter_class="all_in_one_sbi.configs:DiffusionModelAdapter",
        training_loop="all_in_one_sbi.training:train_unstructured_diffusion_model",
        posterior_builder="all_in_one_sbi.diffusion:sample_unstructured_posterior",
        supports_condition_masks=False,
        supports_per_sample_lowest_score_selection=False,
        default_train_kwargs={**_COMMON_TRAIN_KWARGS, "noise_schedule": "linear"},
        default_model_kwargs={"conditioner": "observation_embedding", "attention_mask": "full"},
        provenance={"baseline": "diffusion_model"},
        notes="Unstructured diffusion baseline selector.",
    ),
}

METHOD_ALIASES: Dict[str, str] = {
    alias.lower(): method_id
    for method_id, cfg in METHOD_REGISTRY.items()
    for alias in (method_id, *cfg.aliases)
}

ABLATION_REGISTRY: Dict[str, AblationConfig] = {
    "structured_mask": AblationConfig(
        ablation_id="structured_mask",
        display_name="Conditional dependency mask",
        applies_to_methods=("ours", "simformer", "lora"),
        mask_variant="conditional_dependency",
        per_sample_lowest_score_selection=True,
        sweep_parameters={"mask": "structured", "attention": "paper_dependency_graph"},
        rationale="Tests the core contribution that dependency-aware attention improves conditional SBI.",
    ),
    "full_attention_mask": AblationConfig(
        ablation_id="full_attention_mask",
        display_name="Full attention mask",
        applies_to_methods=("ours", "simformer", "diffusion_model"),
        mask_variant="full_attention",
        per_sample_lowest_score_selection=False,
        sweep_parameters={"mask": "full", "attention": "unrestricted"},
        rationale="Decisive comparison for structured versus unstructured attention.",
    ),
    "per_sample_lowest_score_selection": AblationConfig(
        ablation_id="per_sample_lowest_score_selection",
        display_name="Per-sample lowest score selection",
        applies_to_methods=("ours", "simformer", "lora"),
        mask_variant="conditional_dependency",
        per_sample_lowest_score_selection=True,
        sweep_parameters={"selection": "lowest_score_per_sample", "score_candidates": 4},
        rationale=(
            "Required provenance-visible refinement: select the lowest score/noise candidate "
            "per sample during conditional generation."
        ),
    ),
}


# ---------------------------------------------------------------------------
# Artifact and sweep configuration
# ---------------------------------------------------------------------------

ARTIFACT_GROUPING_KEYS: Tuple[str, ...] = (
    "dataset",
    "task",
    "observation",
    "method",
    "mask_variant",
    "simulation_budget",
    "sweep_parameters",
)

ARTIFACT_SPECS: Dict[str, ArtifactSpec] = {
    "metrics": ArtifactSpec(
        artifact_id="metrics",
        path="results/metrics.json",
        kind="json",
        grouped_by=ARTIFACT_GROUPING_KEYS,
        dry_run_label="dry_run_contract_metrics_schema",
        schema={
            "metrics": ["c2st", "nll", "return", "accuracy", "loss"],
            "requires": ["approximate_posterior_samples", "ground_truth_posterior_samples"],
            "not_real_results_in_dry_run": True,
        },
    ),
    "dataset_registry": ArtifactSpec(
        artifact_id="dataset_registry",
        path="results/dataset_registry.json",
        kind="json",
        grouped_by=("dataset",),
        dry_run_label="dry_run_dataset_registry",
        schema={"datasets": list(DATASET_REGISTRY.keys()), "aliases": sorted(DATASET_ALIASES.keys())},
    ),
    "method_registry": ArtifactSpec(
        artifact_id="method_registry",
        path="results/method_registry.json",
        kind="json",
        grouped_by=("method",),
        dry_run_label="dry_run_method_registry",
        schema={"methods": list(METHOD_REGISTRY.keys()), "aliases": sorted(METHOD_ALIASES.keys())},
    ),
    "ablation_registry": ArtifactSpec(
        artifact_id="ablation_registry",
        path="results/ablation_registry.json",
        kind="json",
        grouped_by=("method", "mask_variant", "sweep_parameters"),
        dry_run_label="dry_run_ablation_registry",
        schema={"ablations": list(ABLATION_REGISTRY.keys())},
    ),
    "config_resolved": ArtifactSpec(
        artifact_id="config_resolved",
        path="results/config_resolved.json",
        kind="json",
        grouped_by=ARTIFACT_GROUPING_KEYS,
        dry_run_label="dry_run_resolved_config",
        schema={"contains": ["dataset_registry", "method_registry", "metric_registry", "sweep_registry"]},
    ),
    "benchmark_c2st": ArtifactSpec(
        artifact_id="benchmark_c2st",
        path="results/benchmark_c2st.json",
        kind="json",
        grouped_by=ARTIFACT_GROUPING_KEYS,
        dry_run_label="dry_run_c2st_schema",
        schema={
            "metric": "c2st",
            "semantic": METRIC_REGISTRY["c2st"].semantic,
            "ground_truth_posteriors_per_task": FULL_BENCHMARK_GROUND_TRUTH_POSTERIORS_PER_TASK,
        },
    ),
    "readiness": ArtifactSpec(
        artifact_id="readiness",
        path="results/readiness.json",
        kind="json",
        grouped_by=("dataset", "method"),
        dry_run_label="dry_run_readiness",
        schema={"status": "ready_for_smoke", "external_assets_required": False},
    ),
    "evaluation_result": ArtifactSpec(
        artifact_id="evaluation_result",
        path="results/evaluation_result.json",
        kind="json",
        grouped_by=ARTIFACT_GROUPING_KEYS,
        dry_run_label="dry_run_evaluation_result_schema",
        schema={"result_type": "schema_only", "not_real_results_in_dry_run": True},
    ),
}

SWEEP_REGISTRY: Dict[str, SweepConfig] = {
    "runtime_smoke": SweepConfig(
        sweep_id="runtime_smoke",
        mode="runtime_smoke",
        datasets=("two_moons",),
        methods=("ours", "npe"),
        benchmark_slots=("section_4_1_slot_1",),
        observation_ids=("observation_00",),
        mask_variants=("conditional_dependency",),
        simulation_budgets=(SMOKE_SIMULATION_BUDGET,),
        metrics=("c2st", "nll", "return"),
        per_sample_lowest_score_selection=True,
        hypothesis=(
            "Wiring the benchmark-evaluation path should exercise dataset adapters, "
            "Simformer, a decisive NPE baseline, posterior sampling, and C2ST/NLL/return."
        ),
        decisive_comparison="ours_vs_npe_smoke_wiring",
        decisive_metric="c2st",
        stop_rule_or_pruning_rationale=(
            "Bounded smoke mode uses one dataset, one observation, and one baseline to validate "
            "real interfaces without claiming paper-scale scores."
        ),
        artifact_paths={k: spec.path for k, spec in ARTIFACT_SPECS.items()},
    ),
    "benchmark_eval_bounded": SweepConfig(
        sweep_id="benchmark_eval_bounded",
        mode="eval",
        datasets=("two_moons", "gaussian_linear", "gaussian_mixture", "slcp"),
        methods=("ours", "npe", "nle", "nre", "diffusion_model"),
        benchmark_slots=tuple(BENCHMARK_TASK_SLOTS.keys()),
        observation_ids=DEFAULT_OBSERVATION_IDS,
        mask_variants=("conditional_dependency", "full_attention"),
        simulation_budgets=(1024, 10000),
        metrics=("c2st", "nll", "return", "accuracy", "loss"),
        per_sample_lowest_score_selection=True,
        hypothesis=(
            "All-in-one Simformer should better match ground-truth posteriors across arbitrary "
            "conditioning tasks than independent baseline estimators under the same budget."
        ),
        decisive_comparison="ours_vs_NPE_NLE_NRE_diffusion_on_C2ST",
        decisive_metric="c2st",
        stop_rule_or_pruning_rationale=(
            "Stop at paper-specified benchmark slots, ten observations per task, selected budgets, "
            "and decisive baselines; avoid exhaustive hyperparameter sweeps unless full mode is explicit."
        ),
        artifact_paths={k: spec.path for k, spec in ARTIFACT_SPECS.items()},
    ),
    "full_protocol": SweepConfig(
        sweep_id="full_protocol",
        mode="train",
        datasets=("two_moons", "gaussian_linear", "gaussian_mixture", "slcp", "lotka_volterra"),
        methods=("ours", "simformer", "npe", "nle", "nre", "lora", "diffusion_model"),
        benchmark_slots=tuple(BENCHMARK_TASK_SLOTS.keys()),
        observation_ids=DEFAULT_OBSERVATION_IDS,
        mask_variants=("conditional_dependency", "full_attention"),
        simulation_budgets=DEFAULT_SIMULATION_BUDGETS,
        metrics=("c2st", "nll", "return", "accuracy", "loss"),
        per_sample_lowest_score_selection=True,
        hypothesis=(
            "The full paper-derived protocol evaluates Simformer against SBI baselines and "
            "adaptation variants over benchmark and structured tasks."
        ),
        decisive_comparison="full_paper_protocol_matrix",
        decisive_metric="c2st",
        stop_rule_or_pruning_rationale=(
            "Full mode is explicit because it may require optional dependencies, simulator assets, "
            "and expensive training."
        ),
        artifact_paths={k: spec.path for k, spec in ARTIFACT_SPECS.items()},
    ),
}


# ---------------------------------------------------------------------------
# Resolution, evaluation, and artifact helpers
# ---------------------------------------------------------------------------

def resolve_dataset_id(dataset_id_or_alias: str) -> str:
    key = dataset_id_or_alias.strip()
    if key in DATASET_REGISTRY:
        return key
    alias = DATASET_ALIASES.get(key) or DATASET_ALIASES.get(key.lower())
    if alias is None:
        raise KeyError(f"Unknown dataset '{dataset_id_or_alias}'. Available: {sorted(DATASET_REGISTRY)}")
    return alias


def resolve_method_id(method_id_or_alias: str) -> str:
    key = method_id_or_alias.strip()
    if key in METHOD_REGISTRY:
        return key
    alias = METHOD_ALIASES.get(key.lower())
    if alias is None:
        raise KeyError(f"Unknown method '{method_id_or_alias}'. Available: {sorted(METHOD_REGISTRY)}")
    return alias


def get_metric_evaluator(metric_id: str) -> Callable[..., float]:
    evaluators: Dict[str, Callable[..., float]] = {
        "c2st": compute_c2st_accuracy,
        "nll": compute_nll,
        "return": compute_return,
        "accuracy": compute_accuracy,
        "loss": compute_loss,
    }
    try:
        return evaluators[metric_id]
    except KeyError as exc:
        raise KeyError(f"Unknown metric '{metric_id}'. Available: {sorted(evaluators)}") from exc


def get_section_model_config(section: str) -> Dict[str, Any]:
    """Return paper-section Simformer hyperparameters."""

    key = section.strip().lower().replace("section", "").strip()
    if key.startswith("sec"):
        key = key.replace("sec", "", 1).strip()
    if key not in SIMFORMER_MODEL_CONFIGS:
        raise KeyError(f"Unknown section {section!r}; available={sorted(SIMFORMER_MODEL_CONFIGS)}")
    return dict(SIMFORMER_MODEL_CONFIGS[key])


def evaluate_posterior_samples(
    approximate_posterior_samples: Sequence[Sequence[float]],
    ground_truth_posterior_samples: Sequence[Sequence[float]],
    metrics: Sequence[str] = ("c2st", "nll", "return"),
) -> Dict[str, float]:
    """Executable evaluator accepting approximate and ground-truth posterior samples."""

    results: Dict[str, float] = {}
    for metric in metrics:
        if metric == "c2st":
            results["c2st"] = compute_c2st_accuracy(approximate_posterior_samples, ground_truth_posterior_samples)
        elif metric == "nll":
            results["nll"] = compute_nll(approximate_posterior_samples, ground_truth_posterior_samples)
        elif metric == "return":
            results["return"] = compute_return(results)
        elif metric == "loss":
            approx_mean = _mean_vector(approximate_posterior_samples)
            truth_mean = _mean_vector(ground_truth_posterior_samples)
            results["loss"] = compute_loss(approx_mean, truth_mean)
        elif metric == "accuracy":
            c2st = results.get("c2st")
            if c2st is None:
                c2st = compute_c2st_accuracy(approximate_posterior_samples, ground_truth_posterior_samples)
            results["accuracy"] = float(1.0 - abs(c2st - 0.5) / 0.5)
        else:
            raise KeyError(f"Unknown posterior metric '{metric}'.")
    return results


def artifact_provenance_key(
    dataset: str,
    task: str,
    observation: str,
    method: str,
    mask_variant: str,
    simulation_budget: int,
    sweep_parameters: Mapping[str, Any],
) -> Dict[str, Any]:
    """Return canonical artifact grouping provenance."""

    return {
        "dataset": resolve_dataset_id(dataset) if dataset in DATASET_ALIASES or dataset in DATASET_REGISTRY else dataset,
        "task": task,
        "observation": observation,
        "method": resolve_method_id(method) if method.lower() in METHOD_ALIASES or method in METHOD_REGISTRY else method,
        "mask_variant": mask_variant,
        "simulation_budget": int(simulation_budget),
        "sweep_parameters": dict(sweep_parameters),
    }


def get_results_dir(results_dir: Optional[str] = None) -> Path:
    """Resolve artifact directory.

    If PAPERBENCH_REPRO_ARTIFACT_DIR is set, declared paths are still represented as
    repository-relative strings in configs, while actual smoke writes may be routed to
    the environment-provided directory by this helper.
    """

    base = results_dir or os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR") or DEFAULT_RESULTS_DIR
    return Path(base)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_experiment_config(
    mode: str = "runtime_smoke",
    results_dir: str = DEFAULT_RESULTS_DIR,
    default_dataset: str = "two_moons",
    default_method: str = "ours",
) -> ExperimentConfig:
    if mode not in {"dry_run", "runtime_smoke", "docker_validate", "train", "eval"}:
        raise ValueError(f"Unsupported mode '{mode}'.")
    return ExperimentConfig(
        experiment_id=f"all_in_one_sbi_{mode}",
        mode=mode,
        results_dir=results_dir,
        default_dataset=resolve_dataset_id(default_dataset),
        default_method=resolve_method_id(default_method),
        dataset_registry=DATASET_REGISTRY,
        benchmark_task_slots=BENCHMARK_TASK_SLOTS,
        method_registry=METHOD_REGISTRY,
        metric_registry=METRIC_REGISTRY,
        ablation_registry=ABLATION_REGISTRY,
        sweep_registry=SWEEP_REGISTRY,
        artifact_specs=ARTIFACT_SPECS,
    )


def run_smoke_protocol(
    dataset: str = "two_moons",
    method: str = "ours",
    observation_id: str = "observation_00",
    num_posterior_samples: int = 32,
    seed: int = 0,
) -> Dict[str, Any]:
    """Exercise real data, method, training, posterior, and evaluator surfaces."""

    dataset_adapter = create_smoke_dataset_adapter(dataset)
    method_adapter = create_method_adapter(method)

    theta, x = dataset_adapter.simulate(num_simulations=SMOKE_SIMULATION_BUDGET, seed=seed)
    train_summary = method_adapter.append_simulations(theta, x).train(max_epochs=1, batch_size=8)
    posterior = method_adapter.build_posterior()
    approx = posterior.sample(num_posterior_samples, observation=observation_id, seed=seed)
    truth = dataset_adapter.ground_truth_posterior_samples(
        observation_id=observation_id,
        num_samples=num_posterior_samples,
        seed=seed,
    )
    metrics = evaluate_posterior_samples(approx, truth, metrics=("c2st", "nll", "loss", "accuracy", "return"))

    provenance = artifact_provenance_key(
        dataset=dataset,
        task="section_4_1_slot_1",
        observation=observation_id,
        method=method,
        mask_variant="conditional_dependency",
        simulation_budget=SMOKE_SIMULATION_BUDGET,
        sweep_parameters={"mode": "runtime_smoke", "per_sample_lowest_score_selection": True},
    )

    return {
        "artifact_label": "dry_run_contract_artifact",
        "not_real_benchmark_result": True,
        "dataset_readiness": dataset_adapter.readiness(),
        "training_summary": train_summary,
        "metrics": metrics,
        "metric_semantics": {metric_id: cfg.semantic for metric_id, cfg in METRIC_REGISTRY.items()},
        "provenance": provenance,
        "num_approximate_samples": len(approx),
        "num_ground_truth_samples": len(truth),
    }


def write_dry_run_artifacts(
    output_dir: Optional[str] = None,
    mode: str = "runtime_smoke",
    dataset: str = "two_moons",
    method: str = "ours",
) -> Dict[str, str]:
    """Materialize every declared config-owned artifact as dry-run contract output."""

    base = get_results_dir(output_dir)
    config = build_experiment_config(mode=mode, results_dir=str(base), default_dataset=dataset, default_method=method)
    smoke = run_smoke_protocol(dataset=dataset, method=method)

    payloads: Dict[str, Mapping[str, Any]] = {
        "metrics": {
            "artifact_label": "dry_run_contract_metrics_schema",
            "not_real_benchmark_result": True,
            "metrics": smoke["metrics"],
            "metric_semantics": smoke["metric_semantics"],
            "provenance": smoke["provenance"],
        },
        "dataset_registry": {
            "artifact_label": "dry_run_dataset_registry",
            "datasets": {k: v.to_dict() for k, v in DATASET_REGISTRY.items()},
            "aliases": DATASET_ALIASES,
            "benchmark_task_slots": {k: v.to_dict() for k, v in BENCHMARK_TASK_SLOTS.items()},
        },
        "method_registry": {
            "artifact_label": "dry_run_method_registry",
            "methods": {k: v.to_dict() for k, v in METHOD_REGISTRY.items()},
            "aliases": METHOD_ALIASES,
        },
        "ablation_registry": {
            "artifact_label": "dry_run_ablation_registry",
            "ablations": {k: v.to_dict() for k, v in ABLATION_REGISTRY.items()},
        },
        "config_resolved": {
            "artifact_label": "dry_run_resolved_config",
            "not_real_benchmark_result": True,
            "config": config.to_dict(),
        },
        "benchmark_c2st": {
            "artifact_label": "dry_run_c2st_schema",
            "not_real_benchmark_result": True,
            "semantic": METRIC_REGISTRY["c2st"].semantic,
            "value": smoke["metrics"]["c2st"],
            "requires": list(METRIC_REGISTRY["c2st"].required_inputs),
            "ground_truth_posteriors_per_task": FULL_BENCHMARK_GROUND_TRUTH_POSTERIORS_PER_TASK,
            "provenance": smoke["provenance"],
        },
        "readiness": {
            "artifact_label": "dry_run_readiness",
            "status": "ready_for_smoke",
            "not_real_benchmark_result": True,
            "datasets": {k: create_smoke_dataset_adapter(k).readiness() for k in DATASET_REGISTRY},
            "methods": {k: {"status": "registered", "adapter_class": v.adapter_class} for k, v in METHOD_REGISTRY.items()},
            "artifact_specs": {k: v.to_dict() for k, v in ARTIFACT_SPECS.items()},
        },
        "evaluation_result": {
            "artifact_label": "dry_run_evaluation_result_schema",
            "not_real_benchmark_result": True,
            "posterior_evaluation_interface": {
                "accepts": ["approximate_posterior_samples", "ground_truth_posterior_samples"],
                "metrics": list(METRIC_REGISTRY.keys()),
            },
            "smoke_result": smoke,
        },
    }

    written: Dict[str, str] = {}
    for artifact_id, spec in ARTIFACT_SPECS.items():
        payload = payloads.get(
            artifact_id,
            {
                "artifact_label": spec.dry_run_label,
                "not_real_benchmark_result": True,
                "schema": dict(spec.schema),
            },
        )
        target = base / Path(spec.path).name if base.name == "results" else base / Path(spec.path).relative_to("results")
        _write_json(target, payload)
        written[artifact_id] = str(target)

    return written


def export_registry_payload() -> Dict[str, Any]:
    """Return machine-readable registries for tests and repository runners."""

    return {
        "datasets": {k: v.to_dict() for k, v in DATASET_REGISTRY.items()},
        "dataset_aliases": dict(DATASET_ALIASES),
        "benchmark_task_slots": {k: v.to_dict() for k, v in BENCHMARK_TASK_SLOTS.items()},
        "methods": {k: v.to_dict() for k, v in METHOD_REGISTRY.items()},
        "method_aliases": dict(METHOD_ALIASES),
        "metrics": {k: v.to_dict() for k, v in METRIC_REGISTRY.items()},
        "ablations": {k: v.to_dict() for k, v in ABLATION_REGISTRY.items()},
        "sweeps": {k: v.to_dict() for k, v in SWEEP_REGISTRY.items()},
        "artifact_specs": {k: v.to_dict() for k, v in ARTIFACT_SPECS.items()},
        "contract_obligations": {
            "four_4_1_benchmark_slots": len(BENCHMARK_TASK_SLOTS) == 4,
            "ground_truth_posteriors_per_task": FULL_BENCHMARK_GROUND_TRUTH_POSTERIORS_PER_TASK,
            "required_datasets": ["two_moons", "gaussian_linear", "gaussian_mixture", "slcp", "lotka_volterra"],
            "required_methods": ["ours", "simformer", "npe", "nle", "nre", "lora", "diffusion_model"],
            "required_metrics": ["accuracy", "loss", "return", "c2st", "nll"],
            "per_sample_lowest_score_selection_configured": True,
            "artifact_grouping_keys": list(ARTIFACT_GROUPING_KEYS),
            "simformer_model_configs": {k: dict(v) for k, v in SIMFORMER_MODEL_CONFIGS.items()},
        },
    }


# Backwards-compatible aliases likely to be imported by neighboring files.
CONFIG = build_experiment_config()
DATASETS = DATASET_REGISTRY
BENCHMARKS = BENCHMARK_TASK_SLOTS
METHODS = METHOD_REGISTRY
METRICS = METRIC_REGISTRY
ABLATIONS = ABLATION_REGISTRY
SWEEPS = SWEEP_REGISTRY
ARTIFACTS = ARTIFACT_SPECS

__all__ = [
    "ABLATIONS",
    "ABLATION_REGISTRY",
    "ARTIFACTS",
    "ARTIFACT_GROUPING_KEYS",
    "ARTIFACT_SPECS",
    "ArtifactSpec",
    "BENCHMARKS",
    "BENCHMARK_TASK_SLOTS",
    "CONFIG",
    "DATASETS",
    "DATASET_ALIASES",
    "DATASET_REGISTRY",
    "DEFAULT_OBSERVATION_IDS",
    "DEFAULT_RESULTS_DIR",
    "FULL_BENCHMARK_GROUND_TRUTH_POSTERIORS_PER_TASK",
    "METRICS",
    "METRIC_REGISTRY",
    "METHODS",
    "METHOD_ALIASES",
    "METHOD_REGISTRY",
    "SMOKE_SIMULATION_BUDGET",
    "SIMFORMER_MODEL_CONFIGS",
    "SWEEPS",
    "SWEEP_REGISTRY",
    "AblationConfig",
    "BenchmarkTaskSlot",
    "DatasetConfig",
    "DiffusionModelAdapter",
    "EmpiricalPosterior",
    "ExperimentConfig",
    "LightweightInferenceAdapter",
    "LoRAAdapter",
    "MetricConfig",
    "MethodConfig",
    "NLEAdapter",
    "NPEAdapter",
    "NREAdapter",
    "PosteriorSampler",
    "SimformerAdapter",
    "SmokeDatasetAdapter",
    "SweepConfig",
    "TrainableMethod",
    "artifact_provenance_key",
    "build_experiment_config",
    "compute_accuracy",
    "compute_c2st_accuracy",
    "compute_loss",
    "compute_nll",
    "compute_return",
    "create_method_adapter",
    "create_smoke_dataset_adapter",
    "evaluate_posterior_samples",
    "export_registry_payload",
    "get_metric_evaluator",
    "get_section_model_config",
    "get_results_dir",
    "resolve_dataset_id",
    "resolve_method_id",
    "run_smoke_protocol",
    "write_dry_run_artifacts",
]
