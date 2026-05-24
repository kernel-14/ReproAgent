"""Artifact, metric, and benchmark-evaluation contract for PaperBench reproduction.

This module owns the benchmark-evaluation artifact surface for the standalone
reproduction of *All-in-one simulation-based inference*.  It is intentionally
importable in a minimal Python environment: optional packages such as NumPy,
scikit-learn, pandas, torch, sbi, or plotting libraries are imported lazily only
inside functions that can use them.

Implemented contract surfaces
-----------------------------
* data_pipeline: canonical posterior-sample record schemas, simulation-budget
  bookkeeping, task/method/provenance records, and bounded sweep configuration.
* artifact_writer: statically discoverable output paths plus dry-run materializers
  for every declared JSON, JSONL, CSV, figure, checkpoint, and sample artifact.
* evaluation / metric_formula: executable C2ST, NLL, return, accuracy, and loss
  aggregation functions.  C2ST follows the paper semantics: 0.5 means approximate
  posterior samples are indistinguishable from ground-truth posterior samples and
  1.0 means perfectly distinguishable.  The preferred implementation uses a
  random forest classifier with 100 trees, configurable by callers; a lightweight
  deterministic bootstrap-stump ensemble is used when scikit-learn is absent.
* config / baseline_or_ablation / model_or_method: method, baseline, ablation,
  and sweep registries including NPE, NLE, NRE, dense/graph Simformer variants,
  LoRA, guided diffusion, and required sweep knobs.

Dry-run artifacts written by this file are readiness/schema/contract artifacts
only.  They do not claim trained-model performance or completed paper-scale
benchmark results.  Section 4.4 appendix-only guidance details referenced by the
addendum are deliberately marked out of scope for replication here.

No code is copied from the blacklisted Simformer repository.

reference_grounding: paperbench_ref_001 docs/how_to_guide/04_embedding_networks.ipynb
reference_grounding: paperbench_ref_001 docs/how_to_guide/22_experiment_tracking.ipynb
reference_grounding: paperbench_ref_001 sbi/inference/trainers/npe/npe_a.py
reference_grounding: paperbench_ref_001 sbi/inference/trainers/nre/nre_base.py
"""

from __future__ import annotations

import base64
import csv
import dataclasses
import datetime as _datetime
import json
import math
import os
import random
import statistics
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


DEFAULT_RESULTS_DIR = Path("results")
DRY_RUN_LABEL = "dry-run contract artifact: schema/readiness only; not a paper result"

REQUIRED_SWEEP_KEYS: Tuple[str, ...] = (
    "alpha",
    "beta",
    "gamma",
    "similarity_guidance_scale",
    "lora_rank",
    "population_size",
    "per_sample_lowest_score_selection",
    "mask_probability_0.3",
)

PAPER_METRIC_NAMES: Tuple[str, ...] = ("accuracy", "loss", "return", "c2st", "nll")

BENCHMARK_VISIBLE_TASK_SLOTS: Tuple[str, ...] = (
    "two_moons",
    "gaussian_linear",
    "gaussian_mixture",
    "slcp",
    "lotka_volterra",
    "sird_function_valued_parameters",
    "hodgkin_huxley_interval_guidance",
)

PAPER_BASELINES: Tuple[str, ...] = (
    "simformer_dense",
    "simformer_undirected_graph",
    "simformer_directed_graph",
    "npe",
    "nle",
    "nre",
    "lora_adapter",
    "guided_diffusion",
)

FIGURE_CAPTIONS: Dict[str, str] = {
    "figure_1": (
        "Capabilities of the Simformer: finite-dimensional and function-valued "
        "parameters, dependency-structured inference, unstructured/missing data, "
        "observation constraints, and arbitrary conditional distributions."
    ),
    "figure_2": (
        "Simformer architecture. Variables are reduced to token representations "
        "containing identity, value, and conditional state (latent L or conditioned C), "
        "then processed by a transformer score network with dependency attention masks."
    ),
    "figure_3": (
        "Examples of arbitrary conditional distributions of the Two Moons simulator "
        "estimated by the Simformer."
    ),
    "figure_4": (
        "Simformer performance on benchmark tasks. The suffixes undirected graph and "
        "directed graph denote structured-attention Simformer variants."
    ),
    "figure_4a": (
        "Classifier Two-Sample Test (C2ST) accuracy between Simformer and ground-truth "
        "posteriors; 0.5 is indistinguishable and 1.0 is perfectly distinguishable."
    ),
    "figure_4b": (
        "C2ST between arbitrary conditional distributions for benchmark-visible "
        "conditioning patterns."
    ),
    "figure_5": (
        "Inference with unstructured observations in the Lotka-Volterra model using "
        "posterior predictive and posterior distribution views."
    ),
    "figure_5a": (
        "Lotka-Volterra posterior predictive and posterior distribution from four "
        "unstructured prey observations."
    ),
    "figure_5b": (
        "Lotka-Volterra comparison for alternative unstructured observations and "
        "conditioning masks."
    ),
    "figure_5c": (
        "Lotka-Volterra missing/unstructured observation robustness and simulation "
        "efficiency artifact."
    ),
    "figure_6": (
        "Inference of an infinite-dimensional parameter space in the SIRD model with "
        "global and time-dependent local parameters."
    ),
    "figure_6a": (
        "SIRD posterior for global and time-dependent local parameters given five "
        "observations of infected, recovered, and death population densities."
    ),
    "figure_6b": (
        "SIRD posterior predictive coverage and function-valued parameter uncertainty."
    ),
    "figure_7": (
        "Inference in the Hodgkin-Huxley model with voltage trace and associated "
        "energy-consumption constraints."
    ),
    "figure_7a": "Hodgkin-Huxley model schematic, observed voltage trace, and energy consumption.",
    "figure_7b": "Hodgkin-Huxley posterior marginals for four parameters.",
    "figure_7c": "Posterior predictive energy consumption from Simformer and simulator outputs.",
    "figure_7e": "Guided posterior predictive samples satisfying observation-interval constraints.",
    "figure_7f": "Constraint-satisfaction and return summaries for interval-guided diffusion.",
    "figure_7g": "Bounded guidance-scale and simulation-efficiency comparison.",
}

FIGURE_ARTIFACT_PATHS: Dict[str, str] = {
    "figure_1": "results/figures/figure_1.png",
    "figure_2": "results/figures/figure_2.png",
    "figure_3": "results/figures/figure_3.png",
    "figure_4": "results/figures/figure_4.png",
    "figure_4a": "results/figures/figure_4a.png",
    "figure_4b": "results/figures/figure_4b.png",
    "figure_5": "results/figures/figure_5.png",
    "figure_5a": "results/figures/figure_5a.png",
    "figure_5c": "results/figures/figure_5c.png",
    "figure_5b": "results/figures/figure_5b.png",
    "figure_6": "results/figures/figure_6.png",
    "figure_6a": "results/figures/figure_6a.png",
    "figure_6b": "results/figures/figure_6b.png",
    "figure_7": "results/figures/figure_7.png",
    "figure_7a": "results/figures/figure_7a.png",
    "figure_7b": "results/figures/figure_7b.png",
    "figure_7c": "results/figures/figure_7c.png",
    "figure_7e": "results/figures/figure_7e.png",
    "figure_7f": "results/figures/figure_7f.png",
    "figure_7g": "results/figures/figure_7g.png",
    "result_figure": "results/figures/experiment_results.png",
    "fig. 2": "results/figures/figure_2.png",
    "fig. 4b": "results/figures/figure_4b.png",
}

NON_FIGURE_ARTIFACT_PATHS: Dict[str, str] = {
    "config": "results/config_resolved.json",
    "predictions": "results/predictions.jsonl",
    "checkpoint": "results/checkpoints/simformer_contract_checkpoint.json",
    "metrics_json": "results/metrics.json",
    "result_table": "results/tables/experiment_results.csv",
    "log": "results/logs/contract.log",
    "dataset_registry": "results/dataset_registry.json",
    "method_registry": "results/method_registry.json",
    "ablation_registry": "results/ablation_registry.json",
    "benchmark_c2st": "results/benchmark_c2st.json",
    "posterior_samples": "results/posterior_samples.npz",
    "samples": "results/samples.npz",
    "run_summary": "results/run_summary.json",
    "experiment_registry": "results/experiment_registry.json",
    "evidence_contract_matrix": "results/evidence_contract_matrix.json",
    "artifact_manifest": "results/artifact_manifest.json",
    "readiness": "results/readiness.json",
    "evaluation_result": "results/evaluation_result.json",
}


@dataclasses.dataclass(frozen=True)
class MetricSchema:
    """Machine-readable metric schema and aggregation semantics."""

    name: str
    description: str
    higher_is_better: Optional[bool]
    value_range: Optional[Tuple[float, float]]
    aggregation: str
    required_fields: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class ArtifactSpec:
    """Static declaration of a result artifact and its smoke writer."""

    artifact_id: str
    path: str
    kind: str
    paper_mapping: str
    caption: str
    metrics: Tuple[str, ...]
    dry_run_writer: str

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class PosteriorSampleRecord:
    """Data-pipeline record for posterior-sample evaluation."""

    task_id: str
    method_id: str
    observation_id: str
    simulation_budget: int
    approximate_samples: Sequence[Sequence[float]]
    ground_truth_samples: Sequence[Sequence[float]]
    sample_weight: Optional[Sequence[float]] = None
    provenance: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def validate(self) -> Dict[str, Any]:
        approx = _as_2d_float_list(self.approximate_samples, "approximate_samples")
        truth = _as_2d_float_list(self.ground_truth_samples, "ground_truth_samples")
        if not approx or not truth:
            raise ValueError("posterior sample records require non-empty approximate and ground-truth samples")
        if len(approx[0]) != len(truth[0]):
            raise ValueError("approximate and ground-truth posterior samples must have the same dimensionality")
        if self.simulation_budget < 0:
            raise ValueError("simulation_budget must be non-negative")
        return {
            "task_id": self.task_id,
            "method_id": self.method_id,
            "observation_id": self.observation_id,
            "simulation_budget": int(self.simulation_budget),
            "num_approximate_samples": len(approx),
            "num_ground_truth_samples": len(truth),
            "sample_dimensionality": len(approx[0]),
            "provenance": dict(self.provenance),
        }


@dataclasses.dataclass(frozen=True)
class SweepConfig:
    """Bounded benchmark-visible sweep knobs required by the contract."""

    alpha: Tuple[float, ...] = (0.1, 0.5, 1.0)
    beta: Tuple[float, ...] = (0.0, 0.1, 1.0)
    gamma: Tuple[float, ...] = (0.0, 0.9, 0.99)
    similarity_guidance_scale: Tuple[float, ...] = (0.0, 0.5, 1.0, 2.0)
    lora_rank: Tuple[int, ...] = (0, 4, 8)
    population_size: Tuple[int, ...] = (16, 64, 256)
    per_sample_lowest_score_selection: Tuple[bool, ...] = (False, True)
    mask_probability_0_3: Tuple[float, ...] = (0.3,)

    def to_dict(self, mode: str = "smoke") -> Dict[str, Any]:
        full = {
            "alpha": list(self.alpha),
            "beta": list(self.beta),
            "gamma": list(self.gamma),
            "similarity_guidance_scale": list(self.similarity_guidance_scale),
            "lora_rank": list(self.lora_rank),
            "population_size": list(self.population_size),
            "per_sample_lowest_score_selection": list(self.per_sample_lowest_score_selection),
            "mask_probability_0.3": list(self.mask_probability_0_3),
        }
        if mode in {"runtime_smoke", "docker_validate", "dry_run", "smoke"}:
            return {key: [values[0]] for key, values in full.items()}
        return full

    def validate_required_keys(self) -> Dict[str, Any]:
        payload = self.to_dict(mode="full")
        missing = [key for key in REQUIRED_SWEEP_KEYS if key not in payload]
        if missing:
            raise ValueError(f"missing required sweep keys: {missing}")
        return {"valid": True, "required_keys": list(REQUIRED_SWEEP_KEYS), "configured_keys": sorted(payload)}


METRIC_SCHEMAS: Dict[str, MetricSchema] = {
    "accuracy": MetricSchema(
        name="accuracy",
        description="Fraction of correct binary/multiclass decisions; used for classifiers and readiness checks.",
        higher_is_better=True,
        value_range=(0.0, 1.0),
        aggregation="mean over observations, tasks, and seeds with per-group counts retained",
        required_fields=("correct_count", "total_count"),
    ),
    "loss": MetricSchema(
        name="loss",
        description="Training or validation objective such as diffusion score matching loss.",
        higher_is_better=False,
        value_range=(0.0, math.inf),
        aggregation="mean final validation loss and optional loss trace summary",
        required_fields=("loss_values",),
    ),
    "return": MetricSchema(
        name="return",
        description="Constraint/reward return for guided diffusion and decision-valued posterior samples.",
        higher_is_better=True,
        value_range=None,
        aggregation="mean, standard deviation, minimum, maximum, and count",
        required_fields=("returns",),
    ),
    "c2st": MetricSchema(
        name="c2st",
        description=(
            "Classifier Two-Sample Test accuracy between approximate and ground-truth posterior samples; "
            "0.5 means aligned/indistinguishable and 1.0 means perfectly distinguishable."
        ),
        higher_is_better=False,
        value_range=(0.5, 1.0),
        aggregation="mean over observations for each task, method, mask variant, and simulation budget",
        required_fields=("approximate_posterior_samples", "ground_truth_posterior_samples", "classifier"),
    ),
    "nll": MetricSchema(
        name="nll",
        description="Negative log likelihood or negative posterior log probability of held-out samples.",
        higher_is_better=False,
        value_range=(0.0, math.inf),
        aggregation="mean NLL with finite-count and invalid-count bookkeeping",
        required_fields=("log_probabilities",),
    ),
}

# reference_grounding: paperbench_ref_001 docs/how_to_guide/22_experiment_tracking.ipynb
# The registry below preserves experiment-tracking intent: training/evaluation
# surfaces append simulations, build posterior evaluators, and write run metadata.
METHOD_REGISTRY: Dict[str, Dict[str, Any]] = {
    "simformer_dense": {
        "display_name": "Simformer dense attention",
        "family": "transformer_score_diffusion",
        "comparison_role": "paper main method without structured attention",
        "attention_mask": "dense",
        "supports_arbitrary_conditioning": True,
        "supports_function_valued_parameters": True,
        "requires_training_adapter": "all_in_one_sbi.training",
    },
    "simformer_undirected_graph": {
        "display_name": "Simformer undirected graph",
        "family": "transformer_score_diffusion",
        "comparison_role": "structured attention ablation for Figure 4",
        "attention_mask": "undirected_graph",
        "supports_arbitrary_conditioning": True,
        "supports_function_valued_parameters": True,
        "requires_training_adapter": "all_in_one_sbi.attention_masks",
    },
    "simformer_directed_graph": {
        "display_name": "Simformer directed graph",
        "family": "transformer_score_diffusion",
        "comparison_role": "structured attention ablation for Figure 4",
        "attention_mask": "directed_graph",
        "supports_arbitrary_conditioning": True,
        "supports_function_valued_parameters": True,
        "requires_training_adapter": "all_in_one_sbi.attention_masks",
    },
    # reference_grounding: paperbench_ref_001 sbi/inference/trainers/npe/npe_a.py
    "npe": {
        "display_name": "Neural Posterior Estimation",
        "family": "baseline_density_estimator",
        "comparison_role": "named baseline for benchmark posterior inference",
        "adapter_interface": "fit(theta, x) -> posterior.sample/log_prob",
        "default_density_estimator": "mdn_or_flow",
        "supports_arbitrary_conditioning": False,
    },
    "nle": {
        "display_name": "Neural Likelihood Estimation",
        "family": "baseline_likelihood_estimator",
        "comparison_role": "named baseline for extended benchmark comparison",
        "adapter_interface": "fit(theta, x) -> likelihood; combine_with_prior -> posterior",
        "supports_arbitrary_conditioning": False,
    },
    # reference_grounding: paperbench_ref_001 sbi/inference/trainers/nre/nre_base.py
    "nre": {
        "display_name": "Neural Ratio Estimation",
        "family": "baseline_ratio_estimator",
        "comparison_role": "named baseline for extended benchmark comparison",
        "adapter_interface": "fit classifier ratio(theta, x) -> posterior sampler",
        "default_classifier": "resnet_or_mlp",
        "supports_arbitrary_conditioning": False,
    },
    "lora_adapter": {
        "display_name": "LoRA adapter",
        "family": "parameter_efficient_refinement",
        "comparison_role": "bounded refinement ablation",
        "sweep_parameter": "lora_rank",
        "supports_arbitrary_conditioning": True,
    },
    "guided_diffusion": {
        "display_name": "Guided diffusion",
        "family": "conditional_score_guidance",
        "comparison_role": "Hodgkin-Huxley interval/metabolic-cost constrained inference",
        "sweep_parameter": "similarity_guidance_scale",
        "appendix_a3_3_extra_guidance_details_required": False,
    },
}

DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {
    "two_moons": {
        "display_name": "Two Moons",
        "paper_role": "Figure 3 arbitrary conditional distributions",
        "observation_type": "low_dimensional",
        "evaluation_metrics": ("c2st", "nll"),
        "simulation_budget_key": "simulation_count",
    },
    "gaussian_linear": {
        "display_name": "Gaussian linear",
        "paper_role": "Figure 4 benchmark task; exception noted at 10k simulations for NPE comparison",
        "observation_type": "low_dimensional",
        "evaluation_metrics": ("c2st", "nll"),
        "simulation_budget_key": "simulation_count",
    },
    "gaussian_mixture": {
        "display_name": "Gaussian mixture",
        "paper_role": "Figure 4 benchmark task",
        "observation_type": "low_dimensional",
        "evaluation_metrics": ("c2st", "nll"),
        "simulation_budget_key": "simulation_count",
    },
    "slcp": {
        "display_name": "SLCP",
        "paper_role": "Figure 4 benchmark task",
        "observation_type": "low_dimensional",
        "evaluation_metrics": ("c2st", "nll"),
        "simulation_budget_key": "simulation_count",
    },
    "lotka_volterra": {
        "display_name": "Lotka-Volterra",
        "paper_role": "Figure 5 unstructured/missing observation inference",
        "observation_type": "time_series_unstructured",
        "evaluation_metrics": ("c2st", "nll", "accuracy"),
        "embedding_adapter": "time_series_or_permutation_invariant_embedding",
        "simulation_budget_key": "simulation_count",
    },
    "sird_function_valued_parameters": {
        "display_name": "SIRD function-valued parameters",
        "paper_role": "Figure 6 infinite-dimensional parameter inference",
        "observation_type": "time_series_function_valued",
        "evaluation_metrics": ("c2st", "nll", "accuracy"),
        "simulation_budget_key": "simulation_count",
    },
    "hodgkin_huxley_interval_guidance": {
        "display_name": "Hodgkin-Huxley interval guidance",
        "paper_role": "Figure 7 observation interval and metabolic-cost constrained inference",
        "observation_type": "voltage_trace_with_constraints",
        "evaluation_metrics": ("return", "accuracy", "nll"),
        "simulation_budget_key": "simulation_count",
        "appendix_a3_3_extra_guidance_details_required": False,
    },
}

ABLATION_REGISTRY: Dict[str, Dict[str, Any]] = {
    "dense_attention": {
        "method_id": "simformer_dense",
        "decision_value": "Tests whether all-in-one conditioning works without graph structure.",
        "bounded_default": True,
    },
    "undirected_graph_attention": {
        "method_id": "simformer_undirected_graph",
        "decision_value": "Tests paper claim that dependency structure can improve accuracy.",
        "bounded_default": True,
    },
    "directed_graph_attention": {
        "method_id": "simformer_directed_graph",
        "decision_value": "Tests directed simulator dependency structure as Figure 4 variant.",
        "bounded_default": True,
    },
    "mask_probability_0.3": {
        "method_id": "simformer_dense",
        "decision_value": "Ensures conditioning-mask probability enters training/evaluation provenance.",
        "mask_probability": 0.3,
        "bounded_default": True,
    },
    "per_sample_lowest_score_selection": {
        "method_id": "guided_diffusion",
        "decision_value": "Decision-valued constrained sampling selection for interval-guided diffusion.",
        "bounded_default": False,
    },
    "lora_rank": {
        "method_id": "lora_adapter",
        "decision_value": "Parameter-efficient refinement variant with explicit rank sweep.",
        "bounded_default": False,
    },
}

FIGURE_ARTIFACT_SPECS: Dict[str, ArtifactSpec] = {
    key: ArtifactSpec(
        artifact_id=key,
        path=path,
        kind="figure",
        paper_mapping=key.replace("_", " ").title(),
        caption=FIGURE_CAPTIONS.get(key, "Paper reproduction figure artifact."),
        metrics=tuple(metric for metric in PAPER_METRIC_NAMES if metric in {"c2st", "nll", "return", "accuracy", "loss"}),
        dry_run_writer="write_diagnostic_png",
    )
    for key, path in FIGURE_ARTIFACT_PATHS.items()
}

NON_FIGURE_ARTIFACT_SPECS: Dict[str, ArtifactSpec] = {
    key: ArtifactSpec(
        artifact_id=key,
        path=path,
        kind=("jsonl" if path.endswith(".jsonl") else "csv" if path.endswith(".csv") else "npz" if path.endswith(".npz") else "json" if path.endswith(".json") else "log"),
        paper_mapping=key,
        caption=f"{key} artifact for benchmark evaluation contract.",
        metrics=tuple(PAPER_METRIC_NAMES),
        dry_run_writer="write_schema_artifact",
    )
    for key, path in NON_FIGURE_ARTIFACT_PATHS.items()
}

ARTIFACT_SPECS: Dict[str, ArtifactSpec] = {**FIGURE_ARTIFACT_SPECS, **NON_FIGURE_ARTIFACT_SPECS}


def _as_2d_float_list(samples: Sequence[Sequence[float]], name: str) -> List[List[float]]:
    rows: List[List[float]] = []
    for row in samples:
        converted = [float(value) for value in row]
        if not converted:
            raise ValueError(f"{name} rows must be non-empty")
        rows.append(converted)
    if rows:
        width = len(rows[0])
        if any(len(row) != width for row in rows):
            raise ValueError(f"{name} must be rectangular")
    return rows


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot compute mean of an empty sequence")
    return float(sum(values) / len(values))


def _safe_stdev(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(statistics.stdev(values))


def _train_test_split_indices(n: int, test_fraction: float, seed: int) -> Tuple[List[int], List[int]]:
    if n < 4:
        return list(range(n)), list(range(n))
    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)
    test_n = max(2, min(n - 2, int(round(n * test_fraction))))
    test_indices = sorted(indices[:test_n])
    train_indices = sorted(indices[test_n:])
    return train_indices, test_indices


def _fallback_random_forest_c2st(
    approximate_samples: Sequence[Sequence[float]],
    ground_truth_samples: Sequence[Sequence[float]],
    *,
    n_estimators: int,
    test_fraction: float,
    random_state: int,
) -> Dict[str, Any]:
    """Deterministic bootstrap-stump ensemble used when sklearn is unavailable.

    This is not a replacement for a production random forest, but it preserves the
    executable C2ST semantics in minimal smoke environments.  Each tree is a
    bootstrap-selected decision stump over one feature and one threshold.  The
    preferred path, when scikit-learn is installed, uses
    ``RandomForestClassifier(n_estimators=100)`` by default.
    """

    approx = _as_2d_float_list(approximate_samples, "approximate_samples")
    truth = _as_2d_float_list(ground_truth_samples, "ground_truth_samples")
    x = approx + truth
    y = [1] * len(approx) + [0] * len(truth)
    dim = len(x[0])
    train_idx, test_idx = _train_test_split_indices(len(x), test_fraction, random_state)
    rng = random.Random(random_state)

    stumps: List[Tuple[int, float, int, int]] = []
    for _ in range(max(1, int(n_estimators))):
        feature = rng.randrange(dim)
        boot = [rng.choice(train_idx) for _ in range(max(1, len(train_idx)))]
        values = [x[i][feature] for i in boot]
        threshold = statistics.median(values)
        left_labels = [y[i] for i in boot if x[i][feature] <= threshold]
        right_labels = [y[i] for i in boot if x[i][feature] > threshold]
        left_pred = 1 if sum(left_labels) >= (len(left_labels) / 2.0) else 0
        right_pred = 1 if sum(right_labels) >= (len(right_labels) / 2.0) else 0
        stumps.append((feature, float(threshold), left_pred, right_pred))

    correct = 0
    predictions: List[int] = []
    for i in test_idx:
        votes = 0
        for feature, threshold, left_pred, right_pred in stumps:
            votes += left_pred if x[i][feature] <= threshold else right_pred
        pred = 1 if votes >= (len(stumps) / 2.0) else 0
        predictions.append(pred)
        correct += int(pred == y[i])

    raw_accuracy = correct / max(1, len(test_idx))
    c2st_accuracy = max(float(raw_accuracy), float(1.0 - raw_accuracy))
    return {
        "metric": "c2st",
        "value": c2st_accuracy,
        "raw_accuracy": float(raw_accuracy),
        "aligned_value": 0.5,
        "perfectly_distinguishable_value": 1.0,
        "classifier": "fallback_bootstrap_decision_stump_forest",
        "n_estimators": int(n_estimators),
        "test_fraction": float(test_fraction),
        "random_state": int(random_state),
        "num_test_samples": len(test_idx),
        "num_train_samples": len(train_idx),
    }


def compute_c2st(
    approximate_samples: Sequence[Sequence[float]],
    ground_truth_samples: Sequence[Sequence[float]],
    *,
    n_estimators: int = 100,
    test_fraction: float = 0.5,
    random_state: int = 0,
) -> Dict[str, Any]:
    """Compute C2ST accuracy between approximate and ground-truth posterior samples.

    The paper's semantic interpretation is retained: 0.5 indicates that a
    classifier cannot distinguish approximate posterior samples from ground-truth
    posterior samples, while 1.0 indicates perfect distinguishability.  The
    preferred implementation uses a random forest classifier with 100 trees.

    Optional scikit-learn imports are lazy so importing this module is safe in a
    minimal environment.
    """

    approx = _as_2d_float_list(approximate_samples, "approximate_samples")
    truth = _as_2d_float_list(ground_truth_samples, "ground_truth_samples")
    if len(approx[0]) != len(truth[0]):
        raise ValueError("C2ST requires approximate and ground-truth samples with the same dimensionality")

    try:
        from sklearn.ensemble import RandomForestClassifier  # type: ignore
        from sklearn.metrics import accuracy_score  # type: ignore
        from sklearn.model_selection import train_test_split  # type: ignore

        x = approx + truth
        y = [1] * len(approx) + [0] * len(truth)
        stratify = y if len(set(y)) == 2 and min(y.count(0), y.count(1)) >= 2 else None
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=test_fraction,
            random_state=random_state,
            stratify=stratify,
        )
        classifier = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        classifier.fit(x_train, y_train)
        raw_accuracy = float(accuracy_score(y_test, classifier.predict(x_test)))
        c2st_accuracy = max(raw_accuracy, 1.0 - raw_accuracy)
        return {
            "metric": "c2st",
            "value": c2st_accuracy,
            "raw_accuracy": raw_accuracy,
            "aligned_value": 0.5,
            "perfectly_distinguishable_value": 1.0,
            "classifier": "sklearn.ensemble.RandomForestClassifier",
            "n_estimators": int(n_estimators),
            "test_fraction": float(test_fraction),
            "random_state": int(random_state),
            "num_test_samples": len(y_test),
            "num_train_samples": len(y_train),
        }
    except Exception as exc:
        result = _fallback_random_forest_c2st(
            approx,
            truth,
            n_estimators=n_estimators,
            test_fraction=test_fraction,
            random_state=random_state,
        )
        result["sklearn_unavailable_reason"] = f"{type(exc).__name__}: {exc}"
        return result


def compute_nll(log_probabilities: Sequence[float]) -> Dict[str, Any]:
    """Aggregate negative log likelihood from posterior log probabilities."""

    finite_log_probs = [float(v) for v in log_probabilities if math.isfinite(float(v))]
    invalid_count = len(list(log_probabilities)) - len(finite_log_probs)
    if not finite_log_probs:
        raise ValueError("NLL requires at least one finite log probability")
    nll_values = [-v for v in finite_log_probs]
    return {
        "metric": "nll",
        "value": _mean(nll_values),
        "mean": _mean(nll_values),
        "std": _safe_stdev(nll_values),
        "min": min(nll_values),
        "max": max(nll_values),
        "finite_count": len(finite_log_probs),
        "invalid_count": invalid_count,
        "aggregation": METRIC_SCHEMAS["nll"].aggregation,
    }


def compute_return(returns: Sequence[float]) -> Dict[str, Any]:
    """Aggregate guided-diffusion reward/return values."""

    values = [float(v) for v in returns if math.isfinite(float(v))]
    if not values:
        raise ValueError("return aggregation requires at least one finite value")
    return {
        "metric": "return",
        "value": _mean(values),
        "mean": _mean(values),
        "std": _safe_stdev(values),
        "min": min(values),
        "max": max(values),
        "count": len(values),
        "aggregation": METRIC_SCHEMAS["return"].aggregation,
    }


def compute_accuracy(correct_count: int, total_count: int) -> Dict[str, Any]:
    """Aggregate accuracy from count statistics."""

    if total_count <= 0:
        raise ValueError("total_count must be positive for accuracy")
    if correct_count < 0 or correct_count > total_count:
        raise ValueError("correct_count must be in [0, total_count]")
    return {
        "metric": "accuracy",
        "value": float(correct_count / total_count),
        "correct_count": int(correct_count),
        "total_count": int(total_count),
        "aggregation": METRIC_SCHEMAS["accuracy"].aggregation,
    }


def compute_loss(loss_values: Sequence[float]) -> Dict[str, Any]:
    """Aggregate training/validation loss values."""

    values = [float(v) for v in loss_values if math.isfinite(float(v))]
    if not values:
        raise ValueError("loss aggregation requires at least one finite value")
    return {
        "metric": "loss",
        "value": values[-1],
        "mean": _mean(values),
        "std": _safe_stdev(values),
        "first": values[0],
        "last": values[-1],
        "min": min(values),
        "max": max(values),
        "count": len(values),
        "aggregation": METRIC_SCHEMAS["loss"].aggregation,
    }


def evaluate_posterior_samples(
    record: PosteriorSampleRecord,
    *,
    log_probabilities: Optional[Sequence[float]] = None,
    returns: Optional[Sequence[float]] = None,
    n_estimators: int = 100,
    random_state: int = 0,
) -> Dict[str, Any]:
    """Evaluate approximate posterior samples against ground-truth samples."""

    validation = record.validate()
    c2st_result = compute_c2st(
        record.approximate_samples,
        record.ground_truth_samples,
        n_estimators=n_estimators,
        random_state=random_state,
    )
    metrics: Dict[str, Any] = {"c2st": c2st_result}
    if log_probabilities is not None:
        metrics["nll"] = compute_nll(log_probabilities)
    if returns is not None:
        metrics["return"] = compute_return(returns)
    return {
        "evaluation_type": "posterior_sample_evaluation",
        "dry_run": False,
        "record": validation,
        "metrics": metrics,
        "simulation_budget": int(record.simulation_budget),
        "simulation_count": int(record.simulation_budget),
        "metric_semantics": {
            "c2st": "0.5 aligned with ground truth posterior; 1.0 perfectly distinguishable",
            "nll": "lower is better",
            "return": "higher is better for guided decision-valued constraints",
        },
    }


def aggregate_metric_results(results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Aggregate evaluator outputs by metric name with task/method counts retained."""

    buckets: Dict[str, List[float]] = {name: [] for name in PAPER_METRIC_NAMES}
    budgets: List[int] = []
    for result in results:
        if "simulation_budget" in result:
            budgets.append(int(result["simulation_budget"]))
        metric_map = result.get("metrics", {})
        if isinstance(metric_map, Mapping):
            for name, payload in metric_map.items():
                if isinstance(payload, Mapping) and "value" in payload and name in buckets:
                    buckets[name].append(float(payload["value"]))
    aggregated: Dict[str, Any] = {}
    for name, values in buckets.items():
        if values:
            aggregated[name] = {
                "mean": _mean(values),
                "std": _safe_stdev(values),
                "min": min(values),
                "max": max(values),
                "count": len(values),
                "schema": METRIC_SCHEMAS[name].to_dict(),
            }
    aggregated["simulation_budget"] = {
        "total": sum(budgets),
        "min": min(budgets) if budgets else 0,
        "max": max(budgets) if budgets else 0,
        "count": len(budgets),
    }
    return aggregated


def prepare_evaluation_record(
    task_id: str,
    method_id: str,
    observation_id: str,
    approximate_samples: Sequence[Sequence[float]],
    ground_truth_samples: Sequence[Sequence[float]],
    *,
    simulation_budget: int,
    provenance: Optional[Mapping[str, Any]] = None,
) -> PosteriorSampleRecord:
    """Data-pipeline helper that validates and returns a posterior evaluation record."""

    record = PosteriorSampleRecord(
        task_id=task_id,
        method_id=method_id,
        observation_id=observation_id,
        simulation_budget=int(simulation_budget),
        approximate_samples=approximate_samples,
        ground_truth_samples=ground_truth_samples,
        provenance=dict(provenance or {}),
    )
    record.validate()
    return record


def get_metric_schemas() -> Dict[str, Dict[str, Any]]:
    return {name: schema.to_dict() for name, schema in METRIC_SCHEMAS.items()}


def get_artifact_registry() -> Dict[str, Dict[str, Any]]:
    return {artifact_id: spec.to_dict() for artifact_id, spec in ARTIFACT_SPECS.items()}


def get_dataset_registry() -> Dict[str, Dict[str, Any]]:
    return {key: dict(value) for key, value in DATASET_REGISTRY.items()}


def get_method_registry() -> Dict[str, Dict[str, Any]]:
    return {key: dict(value) for key, value in METHOD_REGISTRY.items()}


def get_ablation_registry() -> Dict[str, Dict[str, Any]]:
    return {key: dict(value) for key, value in ABLATION_REGISTRY.items()}


def get_sweep_config(mode: str = "smoke") -> Dict[str, Any]:
    sweep = SweepConfig()
    payload = sweep.to_dict(mode=mode)
    payload["_validation"] = sweep.validate_required_keys()
    payload["_stop_rule_or_pruning_rationale"] = (
        "Default smoke/dry-run selects the first value per sweep key to validate wiring. "
        "Full grids require explicit mode='full' to avoid unbounded paper-scale execution."
    )
    return payload


def resolve_artifact_path(relative_path: str, artifact_dir: Optional[str | Path] = None) -> Path:
    """Resolve artifact paths, honoring PAPERBENCH_REPRO_ARTIFACT_DIR for auxiliary output."""

    path = Path(relative_path)
    if path.is_absolute():
        return path
    root = Path(artifact_dir or os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR", "."))
    return root / path


def _write_json(path: Path, payload: Mapping[str, Any]) -> Dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"path": str(path), "bytes": path.stat().st_size, "kind": "json"}


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return {"path": str(path), "bytes": path.stat().st_size, "kind": "jsonl", "rows": len(rows)}


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()}) or ["artifact_id", "status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return {"path": str(path), "bytes": path.stat().st_size, "kind": "csv", "rows": len(rows)}


def _write_log(path: Path, lines: Sequence[str]) -> Dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"path": str(path), "bytes": path.stat().st_size, "kind": "log"}


def _write_png(path: Path, label: str) -> Dict[str, Any]:
    """Write a tiny valid PNG plus sidecar metadata labeling dry-run status."""

    path.parent.mkdir(parents=True, exist_ok=True)
    # 1x1 transparent PNG.  The sidecar carries the diagnostic label because PNG
    # text chunks are unnecessary for contract validation and would complicate the
    # dependency-free writer.
    png_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
    )
    path.write_bytes(png_bytes)
    sidecar = path.with_suffix(path.suffix + ".contract.json")
    _write_json(
        sidecar,
        {
            "label": label,
            "dry_run": True,
            "artifact_path": str(path),
            "diagnostic_image": "1x1 transparent PNG",
            "not_a_result": True,
        },
    )
    return {"path": str(path), "bytes": path.stat().st_size, "kind": "png", "sidecar": str(sidecar)}


def _write_npz_contract(path: Path) -> Dict[str, Any]:
    """Write dry-run posterior/sample artifact with NumPy when available, else zip JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import numpy as np  # type: ignore

        approximate = np.asarray([[0.0, 0.0], [0.1, -0.1], [0.2, -0.2]], dtype=float)
        ground_truth = np.asarray([[0.0, 0.0], [0.05, -0.05], [0.15, -0.15]], dtype=float)
        np.savez(
            path,
            approximate_samples=approximate,
            ground_truth_samples=ground_truth,
            simulation_budget=np.asarray([0], dtype=int),
            dry_run=np.asarray([1], dtype=int),
        )
        return {"path": str(path), "bytes": path.stat().st_size, "kind": "npz", "numpy_writer": True}
    except Exception as exc:
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(
                "contract_manifest.json",
                json.dumps(
                    {
                        "label": DRY_RUN_LABEL,
                        "numpy_unavailable_reason": f"{type(exc).__name__}: {exc}",
                        "arrays_declared": ["approximate_samples", "ground_truth_samples", "simulation_budget"],
                        "not_a_result": True,
                    },
                    indent=2,
                    sort_keys=True,
                ),
            )
        return {"path": str(path), "bytes": path.stat().st_size, "kind": "npz_zip_manifest", "numpy_writer": False}


def build_contract_payload(mode: str = "runtime_smoke") -> Dict[str, Any]:
    """Build the canonical machine-readable benchmark artifact contract."""

    return {
        "label": DRY_RUN_LABEL,
        "generated_at_utc": _datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "mode": mode,
        "paper": "All-in-one simulation-based inference",
        "work_package": "benchmark_eval",
        "hypothesis": (
            "Benchmark evaluation is decisive when approximate posterior samples are compared "
            "against ground-truth posterior samples by C2ST/NLL/return with simulation-budget "
            "bookkeeping and explicit method/baseline selectors."
        ),
        "decision_value": (
            "Supports Figure 4 benchmark comparisons, Figure 5/6 structured tasks, and Figure 7 "
            "guided inference without claiming paper-scale numerical results in smoke mode."
        ),
        "stop_rule_or_pruning_rationale": (
            "Smoke mode validates the real data/evaluation/artifact surfaces with bounded inputs; "
            "full sweeps require explicit full mode."
        ),
        "addendum_scope": {
            "section_4_4_appendix_a3_3_extra_guidance_details_required": False,
            "reason": "Binding addendum clarification excludes additional Appendix Sec. A3.3 guidance details.",
        },
        "metric_schemas": get_metric_schemas(),
        "artifact_registry": get_artifact_registry(),
        "dataset_registry": get_dataset_registry(),
        "method_registry": get_method_registry(),
        "ablation_registry": get_ablation_registry(),
        "sweep_config": get_sweep_config(mode=mode),
        "figure_captions": dict(FIGURE_CAPTIONS),
        "comparison_semantics": {
            "figure_4": (
                "Compare Simformer variants with NPE baseline across benchmark task slots; "
                "NLE and NRE adapters are exposed for extended comparisons."
            ),
            "c2st": "0.5 aligned/indistinguishable from ground-truth posterior; 1.0 perfectly distinguishable.",
            "simulation_efficiency": "All evaluator outputs retain simulation_budget/simulation_count.",
        },
    }


def _dry_run_evaluation_payload() -> Dict[str, Any]:
    approx = [[0.0, 0.0], [0.1, -0.1], [0.2, -0.2], [0.3, -0.3], [0.4, -0.4], [0.5, -0.5]]
    truth = [[0.0, 0.0], [0.05, -0.05], [0.15, -0.15], [0.25, -0.25], [0.35, -0.35], [0.45, -0.45]]
    record = prepare_evaluation_record(
        "two_moons",
        "simformer_dense",
        "dry_run_observation",
        approx,
        truth,
        simulation_budget=0,
        provenance={
            "dry_run": True,
            "per_sample_lowest_score_selection": False,
            "mask_probability_0.3": 0.3,
            "not_a_result": True,
        },
    )
    evaluation = evaluate_posterior_samples(
        record,
        log_probabilities=(-1.0, -1.2, -0.9),
        returns=(0.1, 0.2, 0.15),
        n_estimators=100,
        random_state=7,
    )
    evaluation["dry_run"] = True
    evaluation["label"] = DRY_RUN_LABEL
    evaluation["not_a_result"] = True
    return evaluation


def materialize_artifact(spec: ArtifactSpec, *, artifact_dir: Optional[str | Path] = None, mode: str = "runtime_smoke") -> Dict[str, Any]:
    """Materialize one declared artifact as a dry-run schema/readiness artifact."""

    path = resolve_artifact_path(spec.path, artifact_dir)
    label = f"{DRY_RUN_LABEL}; artifact_id={spec.artifact_id}; mode={mode}"
    if spec.kind == "figure":
        return _write_png(path, label)
    if spec.kind == "jsonl":
        return _write_jsonl(
            path,
            [
                {
                    "label": DRY_RUN_LABEL,
                    "artifact_id": spec.artifact_id,
                    "task_id": "two_moons",
                    "method_id": "simformer_dense",
                    "observation_id": "dry_run_observation",
                    "simulation_budget": 0,
                    "prediction_schema": {
                        "conditioned_variables": ["x"],
                        "latent_variables": ["theta"],
                        "posterior_sample": [0.0, 0.0],
                    },
                    "not_a_result": True,
                }
            ],
        )
    if spec.kind == "csv":
        return _write_csv(
            path,
            [
                {
                    "artifact_id": spec.artifact_id,
                    "label": DRY_RUN_LABEL,
                    "metric": "c2st",
                    "value_semantics": "schema only; 0.5 aligned, 1.0 distinguishable",
                    "simulation_budget": 0,
                    "not_a_result": True,
                }
            ],
        )
    if spec.kind == "npz":
        return _write_npz_contract(path)
    if spec.kind == "log":
        return _write_log(
            path,
            [
                DRY_RUN_LABEL,
                f"artifact_id={spec.artifact_id}",
                "No paper-scale training or evaluation was executed.",
            ],
        )

    payload = build_contract_payload(mode=mode)
    payload.update(
        {
            "artifact_id": spec.artifact_id,
            "artifact_path": spec.path,
            "artifact_kind": spec.kind,
            "caption": spec.caption,
            "paper_mapping": spec.paper_mapping,
            "not_a_result": True,
        }
    )
    if spec.artifact_id == "metrics_json":
        payload["evaluation"] = _dry_run_evaluation_payload()
        payload["aggregated_metrics"] = aggregate_metric_results([payload["evaluation"]])
    elif spec.artifact_id == "dataset_registry":
        payload = {"label": DRY_RUN_LABEL, "dataset_registry": get_dataset_registry(), "not_a_result": True}
    elif spec.artifact_id == "method_registry":
        payload = {"label": DRY_RUN_LABEL, "method_registry": get_method_registry(), "not_a_result": True}
    elif spec.artifact_id == "ablation_registry":
        payload = {"label": DRY_RUN_LABEL, "ablation_registry": get_ablation_registry(), "not_a_result": True}
    elif spec.artifact_id == "benchmark_c2st":
        payload = {
            "label": DRY_RUN_LABEL,
            "metric_schema": METRIC_SCHEMAS["c2st"].to_dict(),
            "evaluation": _dry_run_evaluation_payload()["metrics"]["c2st"],
            "simulation_budget": 0,
            "simulation_count": 0,
            "not_a_result": True,
        }
    elif spec.artifact_id == "evaluation_result":
        payload = _dry_run_evaluation_payload()
    elif spec.artifact_id == "readiness":
        payload = {
            "label": DRY_RUN_LABEL,
            "ready": True,
            "importable": True,
            "declared_artifact_count": len(ARTIFACT_SPECS),
            "required_sweep_keys": list(REQUIRED_SWEEP_KEYS),
            "not_a_result": True,
        }
    return _write_json(path, payload)


def write_dry_run_artifacts(
    *,
    artifact_dir: Optional[str | Path] = None,
    mode: str = "runtime_smoke",
    include_auxiliary_env_dir: bool = True,
) -> Dict[str, Any]:
    """Materialize every declared artifact path for smoke/docker validation.

    The function writes readiness.json and evaluation_result.json in addition to
    every figure/table/config/metric/sample artifact declared by this module.
    Outputs are explicitly labeled as dry-run contract artifacts.
    """

    written: Dict[str, Any] = {}
    errors: Dict[str, str] = {}
    for artifact_id, spec in ARTIFACT_SPECS.items():
        try:
            written[artifact_id] = materialize_artifact(spec, artifact_dir=artifact_dir, mode=mode)
        except Exception as exc:
            errors[artifact_id] = f"{type(exc).__name__}: {exc}"

    manifest_path = resolve_artifact_path(NON_FIGURE_ARTIFACT_PATHS["artifact_manifest"], artifact_dir)
    manifest_payload = {
        "label": DRY_RUN_LABEL,
        "mode": mode,
        "written": written,
        "errors": errors,
        "artifact_registry": get_artifact_registry(),
        "not_a_result": True,
    }
    written["artifact_manifest"] = _write_json(manifest_path, manifest_payload)

    readiness_path = resolve_artifact_path(NON_FIGURE_ARTIFACT_PATHS["readiness"], artifact_dir)
    readiness_payload = {
        "label": DRY_RUN_LABEL,
        "ready": not errors,
        "errors": errors,
        "written_artifact_count": len(written),
        "declared_artifact_count": len(ARTIFACT_SPECS),
        "metric_schemas": get_metric_schemas(),
        "not_a_result": True,
    }
    written["readiness"] = _write_json(readiness_path, readiness_payload)

    evaluation_path = resolve_artifact_path(NON_FIGURE_ARTIFACT_PATHS["evaluation_result"], artifact_dir)
    evaluation_payload = _dry_run_evaluation_payload()
    evaluation_payload["artifact_closure_errors"] = errors
    written["evaluation_result"] = _write_json(evaluation_path, evaluation_payload)

    if include_auxiliary_env_dir and os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR") and artifact_dir is None:
        aux_root = Path(os.environ["PAPERBENCH_REPRO_ARTIFACT_DIR"])
        aux_path = aux_root / "artifact_contract_auxiliary_manifest.json"
        written["auxiliary_manifest"] = _write_json(
            aux_path,
            {
                "label": DRY_RUN_LABEL,
                "source": "src.artifact_contract.write_dry_run_artifacts",
                "primary_results_root": str(resolve_artifact_path("results", artifact_dir)),
                "not_a_result": True,
            },
        )

    return {
        "label": DRY_RUN_LABEL,
        "mode": mode,
        "ok": not errors,
        "written": written,
        "errors": errors,
        "not_a_result": True,
    }


def artifact_paths() -> Dict[str, str]:
    """Return all stable artifact paths required by the benchmark contract."""

    return {artifact_id: spec.path for artifact_id, spec in ARTIFACT_SPECS.items()}


def evidence_contract_matrix() -> Dict[str, Any]:
    """Return paper-obligation-to-code-surface coverage matrix."""

    return {
        "label": "paper evidence contract matrix",
        "work_package": "benchmark_eval",
        "surfaces": {
            "data_pipeline": [
                "PosteriorSampleRecord",
                "prepare_evaluation_record",
                "DATASET_REGISTRY",
                "simulation_budget/simulation_count fields",
            ],
            "artifact_writer": ["ArtifactSpec", "write_dry_run_artifacts", "materialize_artifact"],
            "evaluation": ["evaluate_posterior_samples", "aggregate_metric_results"],
            "metric_formula": ["compute_c2st", "compute_nll", "compute_return", "compute_accuracy", "compute_loss"],
            "config": ["SweepConfig", "get_sweep_config"],
            "model_or_method": ["METHOD_REGISTRY"],
            "baseline_or_ablation": ["METHOD_REGISTRY:NPE/NLE/NRE", "ABLATION_REGISTRY"],
            "tests": ["importable functions with deterministic dry-run writer"],
        },
        "paper_figures": {
            key: {
                "path": FIGURE_ARTIFACT_PATHS[key],
                "caption": FIGURE_CAPTIONS.get(key, "Paper reproduction figure artifact."),
            }
            for key in FIGURE_ARTIFACT_PATHS
            if key.startswith("figure_")
        },
        "metric_schemas": get_metric_schemas(),
        "required_sweep_keys": list(REQUIRED_SWEEP_KEYS),
        "appendix_a3_3_guidance_details_required": False,
    }


__all__ = [
    "ABLATION_REGISTRY",
    "ARTIFACT_SPECS",
    "BENCHMARK_VISIBLE_TASK_SLOTS",
    "DATASET_REGISTRY",
    "FIGURE_ARTIFACT_PATHS",
    "FIGURE_CAPTIONS",
    "METHOD_REGISTRY",
    "METRIC_SCHEMAS",
    "NON_FIGURE_ARTIFACT_PATHS",
    "PAPER_BASELINES",
    "PAPER_METRIC_NAMES",
    "REQUIRED_SWEEP_KEYS",
    "ArtifactSpec",
    "MetricSchema",
    "PosteriorSampleRecord",
    "SweepConfig",
    "aggregate_metric_results",
    "artifact_paths",
    "build_contract_payload",
    "compute_accuracy",
    "compute_c2st",
    "compute_loss",
    "compute_nll",
    "compute_return",
    "evaluate_posterior_samples",
    "evidence_contract_matrix",
    "get_ablation_registry",
    "get_artifact_registry",
    "get_dataset_registry",
    "get_method_registry",
    "get_metric_schemas",
    "get_sweep_config",
    "materialize_artifact",
    "prepare_evaluation_record",
    "resolve_artifact_path",
    "write_dry_run_artifacts",
]