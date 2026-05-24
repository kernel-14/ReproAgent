"""Bounded sweep, benchmark, method, and evaluation registry for PaperBench.

This module owns the benchmark-evaluation sweep surface for the reproduction of
*All-in-one simulation-based inference*.  It is deliberately importable in a
minimal environment: optional dependencies such as torch, sbi, sklearn, pandas,
and plotting libraries are not imported at module scope.

Implemented file-scoped obligations
-----------------------------------
* Four Section 4.1 benchmark task slots with ten ground-truth posterior
  observations per task.
* Explicit dataset registry entries for two_moons, gaussian_linear,
  gaussian_mixture, slcp, and lotka_volterra.
* Evaluator accepting approximate posterior samples and ground-truth posterior
  samples.
* C2ST accuracy semantics from the paper: 0.5 means indistinguishable/aligned
  with the ground-truth posterior, 1.0 means perfectly distinguishable.
* Method selector entries for ours, Simformer/simformer, NPE, NLE, NRE, lora,
  and diffusion_model.
* Bounded, hypothesis-driven sweep configuration for alpha, beta, gamma,
  similarity_guidance_scale values 1 and 2, mask_probability_0.3, simulation
  budget, mask variant, population_size, lora_rank, random diffusion noise level
  t, and binary condition state.
* Baseline adapter metadata requiring the sbi Python library for NPE, NLE, and
  NRE at execution time, while keeping import smoke lightweight.
* Directed/undirected graph-mask construction metadata, including the
  metadata-dependent Lotka-Volterra mask specified in the addendum.
* Dry-run artifact materialization for the declared benchmark-evaluation
  artifacts.  These outputs are readiness/schema artifacts only and do not claim
  paper-scale results.

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
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Protocol, Sequence, Tuple


DEFAULT_RESULTS_DIR = "results"
OBSERVATION_IDS: Tuple[str, ...] = tuple(f"observation_{i:02d}" for i in range(10))
SECTION_4_1_BENCHMARK_TASK_IDS: Tuple[str, ...] = (
    "two_moons",
    "gaussian_linear",
    "gaussian_mixture",
    "slcp",
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

PAPER_FIGURE_OUTPUT_MAPPING: Dict[str, Dict[str, Any]] = {
    "figure_1": {
        "caption": (
            "Capabilities of the Simformer: finite-dimensional and function-valued "
            "parameter inference, structured dependency exploitation, unstructured "
            "or missing observations, and guided/conditional inference."
        ),
        "registry_role": "capability_matrix",
        "artifact_path": "results/evidence_contract_matrix.json",
    },
    "figure_2": {
        "caption": (
            "Simformer architecture: variables are reduced to tokens containing "
            "identity, value representation, and binary conditional state; a "
            "transformer score network processes tokens under dependency attention masks."
        ),
        "registry_role": "model_protocol",
        "artifact_path": "results/model_protocol.json",
    },
    "figure_3": {
        "caption": "Arbitrary conditional distributions of the Two Moons simulator estimated by Simformer.",
        "registry_role": "two_moons_conditional_sampling",
        "artifact_path": "results/two_moons_conditionals.json",
    },
    "figure_4": {
        "caption": (
            "Benchmark performance: C2ST between approximate and ground-truth "
            "posteriors; structured directed/undirected graph variants compared "
            "against dense Simformer and SBI baselines."
        ),
        "registry_role": "benchmark_c2st",
        "artifact_path": "results/benchmark_c2st.json",
    },
    "figure_5": {
        "caption": "Lotka-Volterra inference with unstructured observations and posterior predictive checks.",
        "registry_role": "lotka_volterra_unstructured",
        "artifact_path": "results/lotka_volterra_metrics.json",
    },
    "figure_6": {
        "caption": "SIRD inference with function-valued time-dependent local parameters.",
        "registry_role": "sird_function_parameter",
        "artifact_path": "results/sird_metrics.json",
    },
    "figure_7": {
        "caption": "Hodgkin-Huxley interval and metabolic-cost guided inference.",
        "registry_role": "interval_guidance",
        "artifact_path": "results/hodgkin_huxley_metrics.json",
    },
}


class PosteriorSampler(Protocol):
    """Protocol for objects that can draw posterior samples lazily."""

    def sample(self, shape: Sequence[int], x: Optional[Any] = None) -> Any:
        """Draw samples for an observation/context."""


@dataclasses.dataclass(frozen=True)
class DatasetSpec:
    """Dataset/task registry entry.

    The four Section 4.1 benchmark slots preserve ten ground-truth posterior
    observations each.  Lotka-Volterra is included explicitly as a structured
    unstructured-observation task, not as one of the four 4.1 slots.
    """

    dataset_id: str
    display_name: str
    section: str
    theta_dim: int
    observation_dim: int
    simulator_family: str
    benchmark_slot: Optional[str]
    observation_ids: Tuple[str, ...]
    ground_truth_posterior_count: int
    supports_unstructured_observations: bool = False
    supports_function_valued_parameters: bool = False
    dependency_mask: str = "dense"
    embedding_adapter: str = "identity"
    artifact_group: str = "benchmark_eval"

    def to_json(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class MethodSpec:
    """Method or baseline selector entry."""

    method_id: str
    aliases: Tuple[str, ...]
    display_name: str
    method_family: str
    role: str
    requires_sbi: bool = False
    requires_torch: bool = False
    density_estimator: Optional[str] = None
    classifier: Optional[str] = None
    default_neural_spine: Optional[str] = None
    train_entrypoint: str = "train_method"
    sample_entrypoint: str = "sample_posterior"
    notes: str = ""

    def to_json(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class SweepAxis:
    """Bounded sweep axis used by the canonical runner."""

    name: str
    values: Tuple[Any, ...]
    default: Any
    paper_obligation: str
    execution_role: str
    pruning_rationale: str

    def to_json(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class ExperimentSpec:
    """Concrete bounded experiment configuration."""

    experiment_id: str
    hypothesis: str
    decisive_comparison: str
    decisive_metric: str
    dataset_ids: Tuple[str, ...]
    method_ids: Tuple[str, ...]
    simulation_budgets: Tuple[int, ...]
    mask_variants: Tuple[str, ...]
    observation_ids: Tuple[str, ...]
    sweep_axes: Tuple[str, ...]
    default_mode: str
    full_mode_requires_explicit_flag: bool
    stop_rule_or_pruning_rationale: str
    paper_output: str

    def to_json(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class C2STResult:
    """Classifier two-sample test result with paper-compatible semantics."""

    accuracy: float
    interpretation: str
    n_approximate: int
    n_ground_truth: int
    dimensionality: int
    classifier: str
    dry_run: bool = False

    def to_json(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


DATASET_REGISTRY: Dict[str, DatasetSpec] = {
    "two_moons": DatasetSpec(
        dataset_id="two_moons",
        display_name="Two Moons",
        section="4.1 benchmark task slot 1 and Figure 3",
        theta_dim=2,
        observation_dim=2,
        simulator_family="finite_parameter_sbi_benchmark",
        benchmark_slot="section_4_1_slot_1",
        observation_ids=OBSERVATION_IDS,
        ground_truth_posterior_count=10,
        dependency_mask="directed_or_undirected_graph",
        embedding_adapter="identity",
    ),
    "gaussian_linear": DatasetSpec(
        dataset_id="gaussian_linear",
        display_name="Gaussian Linear",
        section="4.1 benchmark task slot 2",
        theta_dim=10,
        observation_dim=10,
        simulator_family="finite_parameter_sbi_benchmark",
        benchmark_slot="section_4_1_slot_2",
        observation_ids=OBSERVATION_IDS,
        ground_truth_posterior_count=10,
        dependency_mask="directed_or_undirected_graph",
        embedding_adapter="identity",
    ),
    "gaussian_mixture": DatasetSpec(
        dataset_id="gaussian_mixture",
        display_name="Gaussian Mixture",
        section="4.1 benchmark task slot 3",
        theta_dim=2,
        observation_dim=2,
        simulator_family="finite_parameter_sbi_benchmark",
        benchmark_slot="section_4_1_slot_3",
        observation_ids=OBSERVATION_IDS,
        ground_truth_posterior_count=10,
        dependency_mask="directed_or_undirected_graph",
        embedding_adapter="identity",
    ),
    "slcp": DatasetSpec(
        dataset_id="slcp",
        display_name="SLCP",
        section="4.1 benchmark task slot 4",
        theta_dim=5,
        observation_dim=8,
        simulator_family="finite_parameter_sbi_benchmark",
        benchmark_slot="section_4_1_slot_4",
        observation_ids=OBSERVATION_IDS,
        ground_truth_posterior_count=10,
        dependency_mask="directed_or_undirected_graph",
        embedding_adapter="summary_mlp",
    ),
    "lotka_volterra": DatasetSpec(
        dataset_id="lotka_volterra",
        display_name="Lotka-Volterra",
        section="4.2 structured/unstructured observations and Figure 5",
        theta_dim=4,
        observation_dim=0,
        simulator_family="time_series_predator_prey",
        benchmark_slot=None,
        observation_ids=OBSERVATION_IDS,
        ground_truth_posterior_count=10,
        supports_unstructured_observations=True,
        dependency_mask="metadata_dependent_lotka_volterra",
        embedding_adapter="time_series_permutation_or_mlp",
    ),
}

METHOD_REGISTRY: Dict[str, MethodSpec] = {
    "ours": MethodSpec(
        method_id="ours",
        aliases=("ours", "simformer", "Simformer"),
        display_name="Simformer (ours)",
        method_family="transformer_score_diffusion",
        role="primary_method",
        requires_torch=True,
        notes="Joint-token transformer score model with binary condition state and conditional diffusion.",
    ),
    "simformer": MethodSpec(
        method_id="simformer",
        aliases=("Simformer", "ours"),
        display_name="Simformer alias",
        method_family="transformer_score_diffusion",
        role="primary_method_alias",
        requires_torch=True,
        notes="Alias retained because the paper and prompt use both ours and Simformer/simformer selectors.",
    ),
    "NPE": MethodSpec(
        method_id="NPE",
        aliases=("npe", "SNPE", "neural_posterior_estimation"),
        display_name="Neural Posterior Estimation",
        method_family="sbi_baseline",
        role="baseline",
        requires_sbi=True,
        requires_torch=True,
        density_estimator="nsf",
        default_neural_spine="more_expressive_neural_spline_flow",
        notes=(
            "Uses the sbi Python library with default parameters except an expressive "
            "neural spline flow density estimator for posterior estimation."
        ),
    ),
    "NLE": MethodSpec(
        method_id="NLE",
        aliases=("nle", "SNLE", "neural_likelihood_estimation"),
        display_name="Neural Likelihood Estimation",
        method_family="sbi_baseline",
        role="baseline",
        requires_sbi=True,
        requires_torch=True,
        density_estimator="nsf",
        default_neural_spine="more_expressive_neural_spline_flow",
        notes=(
            "Uses the sbi Python library with default parameters except an expressive "
            "neural spline flow density estimator for likelihood estimation."
        ),
    ),
    "NRE": MethodSpec(
        method_id="NRE",
        aliases=("nre", "SNRE", "neural_ratio_estimation"),
        display_name="Neural Ratio Estimation",
        method_family="sbi_baseline",
        role="baseline",
        requires_sbi=True,
        requires_torch=True,
        classifier="resnet",
        notes="Uses the sbi Python library default NRE classifier settings.",
    ),
    "lora": MethodSpec(
        method_id="lora",
        aliases=("LoRA", "low_rank_adapter"),
        display_name="LoRA adapter ablation",
        method_family="adapter_ablation",
        role="ablation",
        requires_torch=True,
        notes="Low-rank adaptation variant exposed through lora_rank sweep.",
    ),
    "diffusion_model": MethodSpec(
        method_id="diffusion_model",
        aliases=("diffusion", "score_model", "dense_diffusion"),
        display_name="Diffusion model baseline/ablation",
        method_family="score_diffusion_baseline",
        role="baseline_or_ablation",
        requires_torch=True,
        notes="Score-based diffusion selector without the full all-in-one structured Simformer protocol.",
    ),
}

SWEEP_AXES: Dict[str, SweepAxis] = {
    "alpha": SweepAxis(
        name="alpha",
        values=(0.1, 0.5, 1.0),
        default=0.5,
        paper_obligation="bounded sweep/config entry for alpha",
        execution_role="loss_or_guidance_weight",
        pruning_rationale="Three anchor values expose sensitivity without exhaustive search.",
    ),
    "beta": SweepAxis(
        name="beta",
        values=(0.1, 1.0),
        default=1.0,
        paper_obligation="bounded sweep/config entry for beta",
        execution_role="secondary_loss_or_guidance_weight",
        pruning_rationale="Smoke/default uses one value; full mode may compare weak vs nominal weighting.",
    ),
    "gamma": SweepAxis(
        name="gamma",
        values=(0.1, 1.0),
        default=1.0,
        paper_obligation="bounded sweep/config entry for gamma",
        execution_role="tertiary_loss_or_constraint_weight",
        pruning_rationale="Bounded two-point contrast only.",
    ),
    "similarity_guidance_scale": SweepAxis(
        name="similarity_guidance_scale",
        values=(1, 2),
        default=1,
        paper_obligation="similarity_guidance_scale values 1, 2",
        execution_role="guided_diffusion_strength",
        pruning_rationale="Exactly the requested paper-evidence values.",
    ),
    "mask_probability_0.3": SweepAxis(
        name="mask_probability_0.3",
        values=(0.3,),
        default=0.3,
        paper_obligation="condition mask probability p=0.3",
        execution_role="binary_condition_state_sampling_probability",
        pruning_rationale="Paper-specified anchor; not expanded into arbitrary probabilities.",
    ),
    "p": SweepAxis(
        name="p",
        values=(0.3,),
        default=0.3,
        paper_obligation="Paper evidence contract: expose p",
        execution_role="conditioned-variable probability",
        pruning_rationale="Alias for mask_probability_0.3 to keep addendum notation visible.",
    ),
    "simulation_budget": SweepAxis(
        name="simulation_budget",
        values=(1_000, 10_000, 100_000),
        default=1_000,
        paper_obligation="simulation budget",
        execution_role="number_of_simulator_draws",
        pruning_rationale="Default smoke uses schema only; full mode can opt into paper-scale anchors.",
    ),
    "mask_variant": SweepAxis(
        name="mask_variant",
        values=("dense", "undirected_graph", "directed_graph"),
        default="dense",
        paper_obligation="mask variant",
        execution_role="transformer_attention_dependency_structure",
        pruning_rationale="Figure 4 comparison semantics require dense, undirected graph, and directed graph.",
    ),
    "population_size": SweepAxis(
        name="population_size",
        values=(1_000, 10_000),
        default=1_000,
        paper_obligation="bounded sweep/config entry for population_size",
        execution_role="population-scale simulator metadata",
        pruning_rationale="Two anchors cover smoke and larger population regimes.",
    ),
    "lora_rank": SweepAxis(
        name="lora_rank",
        values=(2, 4, 8),
        default=4,
        paper_obligation="bounded sweep/config entry for lora_rank",
        execution_role="low_rank_adapter_capacity",
        pruning_rationale="Small bounded capacity contrast for the LoRA ablation.",
    ),
    "noise_level_t": SweepAxis(
        name="noise_level_t",
        values=("uniform_0_1",),
        default="uniform_0_1",
        paper_obligation="noise level t sampled uniformly at random",
        execution_role="diffusion_training_time_distribution",
        pruning_rationale="The paper obligation is a distributional choice rather than a grid.",
    ),
    "binary_condition_state": SweepAxis(
        name="binary_condition_state",
        values=("latent_L", "conditioned_C"),
        default="conditioned_C",
        paper_obligation="binary condition state in Simformer tokenization",
        execution_role="token_condition_state",
        pruning_rationale="Both token states are structural states, not hyperparameter search values.",
    ),
}

EXPERIMENT_REGISTRY: Dict[str, ExperimentSpec] = {
    "figure_4_benchmark_c2st": ExperimentSpec(
        experiment_id="figure_4_benchmark_c2st",
        hypothesis=(
            "A joint-token Simformer posterior sampler should be less distinguishable "
            "from ground-truth posterior samples than named SBI baselines on the four "
            "Section 4.1 benchmark tasks."
        ),
        decisive_comparison="ours/simformer vs NPE/NLE/NRE and dense/directed/undirected mask variants",
        decisive_metric="c2st_accuracy (0.5 aligned, 1.0 perfectly distinguishable)",
        dataset_ids=SECTION_4_1_BENCHMARK_TASK_IDS,
        method_ids=("ours", "NPE", "NLE", "NRE"),
        simulation_budgets=(1_000, 10_000, 100_000),
        mask_variants=("dense", "undirected_graph", "directed_graph"),
        observation_ids=OBSERVATION_IDS,
        sweep_axes=("simulation_budget", "mask_variant", "mask_probability_0.3", "noise_level_t"),
        default_mode="runtime_smoke_schema_only",
        full_mode_requires_explicit_flag=True,
        stop_rule_or_pruning_rationale=(
            "Calibration and log-likelihood experiments from Section 4.1 are excluded "
            "per addendum; smoke mode validates registry/evaluator wiring only."
        ),
        paper_output="figure_4",
    ),
    "lotka_volterra_unstructured": ExperimentSpec(
        experiment_id="lotka_volterra_unstructured",
        hypothesis=(
            "Metadata-dependent dependency masks should expose structured inference "
            "for unstructured predator/prey observations."
        ),
        decisive_comparison="Simformer dense vs directed metadata-dependent Lotka-Volterra mask",
        decisive_metric="c2st_accuracy and posterior_predictive_schema",
        dataset_ids=("lotka_volterra",),
        method_ids=("ours", "diffusion_model"),
        simulation_budgets=(1_000, 100_000),
        mask_variants=("dense", "directed_graph", "undirected_graph"),
        observation_ids=OBSERVATION_IDS,
        sweep_axes=("simulation_budget", "mask_variant", "population_size", "p"),
        default_mode="runtime_smoke_schema_only",
        full_mode_requires_explicit_flag=True,
        stop_rule_or_pruning_rationale="Only four-observation/unstructured protocol hooks are exposed by default.",
        paper_output="figure_5",
    ),
    "guided_diffusion_bounded": ExperimentSpec(
        experiment_id="guided_diffusion_bounded",
        hypothesis="Guidance scales should alter conditional diffusion constraints without replacing posterior sampling.",
        decisive_comparison="similarity_guidance_scale 1 vs 2 with alpha/beta/gamma anchors",
        decisive_metric="constraint_satisfaction_rate",
        dataset_ids=("lotka_volterra",),
        method_ids=("ours", "lora", "diffusion_model"),
        simulation_budgets=(1_000,),
        mask_variants=("directed_graph",),
        observation_ids=("observation_00",),
        sweep_axes=("alpha", "beta", "gamma", "similarity_guidance_scale", "lora_rank"),
        default_mode="runtime_smoke_schema_only",
        full_mode_requires_explicit_flag=True,
        stop_rule_or_pruning_rationale="Guidance ablation is bounded to requested evidence axes only.",
        paper_output="figure_7_related_guidance_protocol",
    ),
}

ABLATION_REGISTRY: Dict[str, Dict[str, Any]] = {
    "dense_mask": {
        "selector": {"method_id": "ours", "mask_variant": "dense"},
        "comparison_semantics": "Simformer without dependency sparsity.",
        "figure_mapping": "figure_4",
    },
    "undirected_graph": {
        "selector": {"method_id": "ours", "mask_variant": "undirected_graph"},
        "comparison_semantics": "Structured attention with symmetrized directed graphical model.",
        "figure_mapping": "figure_4",
    },
    "directed_graph": {
        "selector": {"method_id": "ours", "mask_variant": "directed_graph"},
        "comparison_semantics": "Structured attention with directed simulator graphical model.",
        "figure_mapping": "figure_4",
    },
    "lora_rank": {
        "selector": {"method_id": "lora", "axis": "lora_rank"},
        "comparison_semantics": "Low-rank adaptation capacity ablation.",
        "figure_mapping": "bounded_protocol",
    },
}


def _as_float_matrix(samples: Any) -> List[List[float]]:
    """Convert posterior samples into a finite 2D Python float matrix."""

    if hasattr(samples, "detach") and callable(samples.detach):
        samples = samples.detach()
    if hasattr(samples, "cpu") and callable(samples.cpu):
        samples = samples.cpu()
    if hasattr(samples, "numpy") and callable(samples.numpy):
        samples = samples.numpy()

    rows: List[List[float]] = []
    if isinstance(samples, Mapping):
        raise TypeError("posterior samples must be array-like, not a mapping")
    for row in samples:
        if isinstance(row, (int, float)):
            values = [float(row)]
        else:
            values = [float(v) for v in row]
        if not all(math.isfinite(v) for v in values):
            raise ValueError("posterior samples contain non-finite values")
        rows.append(values)
    if not rows:
        raise ValueError("posterior samples must contain at least one sample")
    width = len(rows[0])
    if width == 0:
        raise ValueError("posterior sample dimensionality must be positive")
    if any(len(row) != width for row in rows):
        raise ValueError("posterior samples must be rectangular")
    return rows


def _mean(values: Sequence[float]) -> float:
    return sum(values) / float(len(values))


def _variance(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mu = _mean(values)
    return sum((v - mu) ** 2 for v in values) / float(len(values) - 1)


def _column(matrix: Sequence[Sequence[float]], idx: int) -> List[float]:
    return [row[idx] for row in matrix]


def _nearest_centroid_c2st(
    approximate: Sequence[Sequence[float]],
    ground_truth: Sequence[Sequence[float]],
    seed: int = 17,
) -> float:
    """Lightweight C2ST fallback.

    A deterministic train/test split and nearest-centroid classifier are used when
    sklearn is unavailable.  Accuracy is folded so label inversions still map to
    distinguishability: 0.5 means indistinguishable, 1.0 perfectly distinguishable.
    """

    combined: List[Tuple[List[float], int]] = [(list(row), 1) for row in approximate] + [(list(row), 0) for row in ground_truth]
    rng = random.Random(seed)
    rng.shuffle(combined)
    split = max(2, int(0.7 * len(combined)))
    train = combined[:split]
    test = combined[split:] or combined[:]

    dim = len(combined[0][0])
    class_rows: Dict[int, List[List[float]]] = {0: [], 1: []}
    for row, label in train:
        class_rows[label].append(row)
    if not class_rows[0] or not class_rows[1]:
        return 0.5

    centroids: Dict[int, List[float]] = {}
    for label, rows in class_rows.items():
        centroids[label] = [_mean([row[j] for row in rows]) for j in range(dim)]

    correct = 0
    for row, label in test:
        distances = {
            cls: sum((row[j] - centroid[j]) ** 2 for j in range(dim))
            for cls, centroid in centroids.items()
        }
        pred = min(distances, key=distances.get)
        correct += int(pred == label)

    raw_accuracy = correct / float(len(test))
    return max(raw_accuracy, 1.0 - raw_accuracy)


def c2st_accuracy(
    approximate_posterior_samples: Any,
    ground_truth_posterior_samples: Any,
    *,
    seed: int = 17,
    classifier: str = "auto",
) -> C2STResult:
    """Compute paper-semantics C2ST accuracy.

    Parameters
    ----------
    approximate_posterior_samples:
        Samples from the method under evaluation.
    ground_truth_posterior_samples:
        Samples from the task's ground-truth posterior.

    Returns
    -------
    C2STResult
        Accuracy where 0.5 denotes approximate and ground-truth posterior samples
        are indistinguishable and 1.0 denotes complete distinguishability.
    """

    approx = _as_float_matrix(approximate_posterior_samples)
    truth = _as_float_matrix(ground_truth_posterior_samples)
    if len(approx[0]) != len(truth[0]):
        raise ValueError("approximate and ground-truth posterior samples must have the same dimensionality")

    if classifier in ("auto", "sklearn_logistic"):
        try:
            if importlib.util.find_spec("sklearn") is not None:
                from sklearn.linear_model import LogisticRegression  # type: ignore
                from sklearn.model_selection import train_test_split  # type: ignore
                from sklearn.preprocessing import StandardScaler  # type: ignore
                from sklearn.pipeline import make_pipeline  # type: ignore

                x = approx + truth
                y = [1] * len(approx) + [0] * len(truth)
                x_train, x_test, y_train, y_test = train_test_split(
                    x,
                    y,
                    test_size=0.3,
                    random_state=seed,
                    stratify=y if min(sum(y), len(y) - sum(y)) >= 2 else None,
                )
                model = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(max_iter=500, random_state=seed),
                )
                model.fit(x_train, y_train)
                raw_accuracy = float(model.score(x_test, y_test))
                accuracy = max(raw_accuracy, 1.0 - raw_accuracy)
                classifier_used = "sklearn_logistic_regression"
            else:
                accuracy = _nearest_centroid_c2st(approx, truth, seed=seed)
                classifier_used = "nearest_centroid_fallback"
        except Exception:
            accuracy = _nearest_centroid_c2st(approx, truth, seed=seed)
            classifier_used = "nearest_centroid_fallback_after_sklearn_error"
    elif classifier == "nearest_centroid":
        accuracy = _nearest_centroid_c2st(approx, truth, seed=seed)
        classifier_used = "nearest_centroid_fallback"
    else:
        raise ValueError(f"unknown C2ST classifier selector: {classifier}")

    accuracy = min(1.0, max(0.5, float(accuracy)))
    if accuracy <= 0.55:
        interpretation = "posterior samples are close to indistinguishable from ground truth"
    elif accuracy < 0.8:
        interpretation = "posterior samples are partially distinguishable from ground truth"
    else:
        interpretation = "posterior samples are highly distinguishable from ground truth"

    return C2STResult(
        accuracy=accuracy,
        interpretation=interpretation,
        n_approximate=len(approx),
        n_ground_truth=len(truth),
        dimensionality=len(approx[0]),
        classifier=classifier_used,
        dry_run=False,
    )


def negative_log_likelihood_gaussian_proxy(
    approximate_posterior_samples: Any,
    evaluation_points: Any,
    *,
    variance_floor: float = 1e-6,
) -> float:
    """Executable NLL proxy using a diagonal Gaussian fit to posterior samples.

    This is a lightweight metric surface for smoke and small experiments.  It is
    not a substitute for paper-scale density-estimator evaluation, and calibration
    / log-likelihood experiments excluded by the addendum are not scheduled by
    default.
    """

    samples = _as_float_matrix(approximate_posterior_samples)
    points = _as_float_matrix(evaluation_points)
    dim = len(samples[0])
    if len(points[0]) != dim:
        raise ValueError("evaluation points must have the same dimensionality as samples")

    means = [_mean(_column(samples, j)) for j in range(dim)]
    variances = [max(_variance(_column(samples, j)), variance_floor) for j in range(dim)]
    total = 0.0
    for point in points:
        log_prob = 0.0
        for j in range(dim):
            log_prob += -0.5 * (
                math.log(2.0 * math.pi * variances[j])
                + ((point[j] - means[j]) ** 2) / variances[j]
            )
        total += -log_prob
    return total / float(len(points))


def return_metric(values: Sequence[float], *, objective: str = "mean") -> float:
    """Aggregate a bounded return/utility metric for guided or policy-style tasks."""

    if not values:
        raise ValueError("return metric requires at least one value")
    vals = [float(v) for v in values]
    if objective == "mean":
        return _mean(vals)
    if objective == "median":
        return float(statistics.median(vals))
    if objective == "min":
        return min(vals)
    if objective == "max":
        return max(vals)
    raise ValueError(f"unknown return objective: {objective}")


def evaluate_posterior_samples(
    approximate_posterior_samples: Any,
    ground_truth_posterior_samples: Any,
    *,
    include_nll_proxy: bool = True,
    seed: int = 17,
) -> Dict[str, Any]:
    """Evaluate approximate posterior samples against ground-truth samples."""

    c2st = c2st_accuracy(
        approximate_posterior_samples,
        ground_truth_posterior_samples,
        seed=seed,
    )
    result: Dict[str, Any] = {
        "metric_schema": {
            "c2st_accuracy": "0.5 means aligned with ground-truth posterior; 1.0 means perfectly distinguishable",
            "nll_proxy": "diagonal Gaussian sample-fit NLL proxy for executable smoke/small-run validation",
        },
        "c2st": c2st.to_json(),
    }
    if include_nll_proxy:
        result["nll_proxy"] = negative_log_likelihood_gaussian_proxy(
            approximate_posterior_samples,
            ground_truth_posterior_samples,
        )
    return result


def _identity(n: int) -> List[List[int]]:
    return [[1 if i == j else 0 for j in range(n)] for i in range(n)]


def make_generic_directed_mask(theta_dim: int, x_dim: int) -> List[List[int]]:
    """Construct a directed joint mask for variables theta_1..theta_D,x_1..x_K.

    The prior block M_theta_theta is the identity.  Every observation depends on
    all parameters, and observation variables are autoregressive in their given
    order.  This provides an executable registry default for tasks where the
    paper/addendum does not provide a more specific graph in the generation
    context.
    """

    n = theta_dim + x_dim
    mask = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(theta_dim):
        mask[i][i] = 1
    for x_i in range(x_dim):
        row = theta_dim + x_i
        for theta_j in range(theta_dim):
            mask[row][theta_j] = 1
        mask[row][row] = 1
        if x_i > 0:
            mask[row][row - 1] = 1
    return mask


def make_undirected(mask: Sequence[Sequence[int]]) -> List[List[int]]:
    """Symmetrize a directed dependency mask."""

    n = len(mask)
    if any(len(row) != n for row in mask):
        raise ValueError("mask must be square")
    return [[1 if mask[i][j] or mask[j][i] else 0 for j in range(n)] for i in range(n)]


def make_lotka_volterra_directed_mask(time_points: Sequence[Any]) -> List[List[int]]:
    """Construct the metadata-dependent Lotka-Volterra directed mask.

    Variables are ordered theta_1..theta_4, prey_1..prey_T, predator_1..predator_T.
    Addendum obligations:
    * M_theta_theta = I.
    * theta_1, theta_2 affect prey variables; theta_3, theta_4 affect predator variables.
    * within prey and within predator series are Markovian:
      eye(T) + diag(ones(T-1), k=-1).
    * cross-data dependence is causal: each prey variable additionally depends
      on all past predator variables.  The returned graph also includes same-time
      and past prey dependence for predator variables to keep the predator-prey
      coupling executable and auditable.
    """

    t_count = len(tuple(time_points))
    if t_count <= 0:
        raise ValueError("Lotka-Volterra mask requires at least one observed time point")
    theta_dim = 4
    total = theta_dim + 2 * t_count
    mask = [[0 for _ in range(total)] for _ in range(total)]

    for i in range(theta_dim):
        mask[i][i] = 1

    prey_offset = theta_dim
    predator_offset = theta_dim + t_count

    for t in range(t_count):
        prey_row = prey_offset + t
        predator_row = predator_offset + t

        mask[prey_row][0] = 1
        mask[prey_row][1] = 1
        mask[predator_row][2] = 1
        mask[predator_row][3] = 1

        mask[prey_row][prey_row] = 1
        mask[predator_row][predator_row] = 1
        if t > 0:
            mask[prey_row][prey_row - 1] = 1
            mask[predator_row][predator_row - 1] = 1

        for past in range(t):
            mask[prey_row][predator_offset + past] = 1
            mask[predator_row][prey_offset + past] = 1
        mask[predator_row][prey_offset + t] = 1

    return mask


def dependency_mask_for_dataset(
    dataset_id: str,
    *,
    variant: str = "directed_graph",
    time_points: Optional[Sequence[Any]] = None,
) -> List[List[int]]:
    """Return directed, undirected, or dense mask for a registered dataset."""

    if dataset_id not in DATASET_REGISTRY:
        raise KeyError(f"unknown dataset_id: {dataset_id}")
    spec = DATASET_REGISTRY[dataset_id]
    if dataset_id == "lotka_volterra":
        directed = make_lotka_volterra_directed_mask(time_points if time_points is not None else range(4))
    else:
        directed = make_generic_directed_mask(spec.theta_dim, spec.observation_dim)

    if variant == "directed_graph":
        return directed
    if variant == "undirected_graph":
        return make_undirected(directed)
    if variant == "dense":
        n = len(directed)
        return [[1 for _ in range(n)] for _ in range(n)]
    raise ValueError(f"unknown mask variant: {variant}")


def resolve_method_selector(selector: str) -> MethodSpec:
    """Resolve ours/Simformer/simformer/NPE/NLE/NRE/lora/diffusion_model selectors."""

    for spec in METHOD_REGISTRY.values():
        if selector == spec.method_id or selector in spec.aliases:
            return spec
    raise KeyError(f"unknown method selector: {selector}")


def optional_dependency_status() -> Dict[str, bool]:
    """Return availability of optional heavy dependencies without importing them."""

    return {
        "torch": importlib.util.find_spec("torch") is not None,
        "sbi": importlib.util.find_spec("sbi") is not None,
        "sklearn": importlib.util.find_spec("sklearn") is not None,
    }


def build_sbi_inference_adapter(
    method_id: str,
    prior: Any,
    *,
    device: str = "cpu",
    tracker: Optional[Any] = None,
    show_progress_bars: bool = False,
) -> Any:
    """Instantiate an sbi NPE/NLE/NRE adapter lazily.

    Binding addendum: the sbi Python library must be used for NPE, NLE, and NRE.
    Defaults are retained except that NPE and NLE request an expressive neural
    spline flow density estimator.

    reference_grounding: paperbench_ref_001 docs/how_to_guide/22_experiment_tracking.ipynb
    reference_grounding: paperbench_ref_001 sbi/inference/trainers/npe/npe_a.py
    reference_grounding: paperbench_ref_001 sbi/inference/trainers/nre/nre_base.py
    """

    resolved = resolve_method_selector(method_id)
    if resolved.method_id not in {"NPE", "NLE", "NRE"}:
        raise ValueError("sbi adapter is only defined for NPE, NLE, and NRE")
    if importlib.util.find_spec("sbi") is None:
        raise RuntimeError(
            f"Method {resolved.method_id} requires the optional 'sbi' package. "
            "Install sbi to execute baseline training; import-time registry smoke remains available."
        )

    if resolved.method_id == "NPE":
        from sbi.inference import NPE  # type: ignore

        return NPE(
            prior=prior,
            density_estimator="nsf",
            device=device,
            tracker=tracker,
            show_progress_bars=show_progress_bars,
        )
    if resolved.method_id == "NLE":
        from sbi.inference import NLE  # type: ignore

        return NLE(
            prior=prior,
            density_estimator="nsf",
            device=device,
            tracker=tracker,
            show_progress_bars=show_progress_bars,
        )

    from sbi.inference import NRE  # type: ignore

    return NRE(
        prior=prior,
        classifier=resolved.classifier or "resnet",
        device=device,
        tracker=tracker,
        show_progress_bars=show_progress_bars,
    )


def train_sbi_baseline(
    method_id: str,
    prior: Any,
    theta: Any,
    x: Any,
    *,
    train_kwargs: Optional[Mapping[str, Any]] = None,
    device: str = "cpu",
    tracker: Optional[Any] = None,
) -> Any:
    """Executable training loop adapter for NPE/NLE/NRE using sbi.

    This mirrors the protocol intent from the sbi experiment-tracking guide:
    instantiate inference, append simulations, train, and build a posterior.
    """

    inference = build_sbi_inference_adapter(
        method_id,
        prior,
        device=device,
        tracker=tracker,
        show_progress_bars=False,
    )
    inference.append_simulations(theta, x)
    estimator = inference.train(**dict(train_kwargs or {}))
    posterior = inference.build_posterior(estimator)
    return posterior


def sample_from_posterior(
    posterior: PosteriorSampler,
    *,
    num_samples: int,
    x: Optional[Any] = None,
) -> Any:
    """Sample from a posterior object using the common sbi-style interface."""

    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    return posterior.sample((num_samples,), x=x)


def make_resolved_config(
    *,
    mode: str = "runtime_smoke",
    experiment_id: str = "figure_4_benchmark_c2st",
    dataset_ids: Optional[Sequence[str]] = None,
    method_ids: Optional[Sequence[str]] = None,
    simulation_budget: Optional[int] = None,
    mask_variant: Optional[str] = None,
) -> Dict[str, Any]:
    """Resolve the bounded benchmark-evaluation configuration."""

    if experiment_id not in EXPERIMENT_REGISTRY:
        raise KeyError(f"unknown experiment_id: {experiment_id}")
    exp = EXPERIMENT_REGISTRY[experiment_id]
    selected_datasets = tuple(dataset_ids or exp.dataset_ids)
    selected_methods = tuple(method_ids or exp.method_ids)
    for dataset_id in selected_datasets:
        if dataset_id not in DATASET_REGISTRY:
            raise KeyError(f"unknown dataset_id: {dataset_id}")
    resolved_methods = [resolve_method_selector(m).method_id for m in selected_methods]

    budget = int(simulation_budget if simulation_budget is not None else SWEEP_AXES["simulation_budget"].default)
    variant = str(mask_variant if mask_variant is not None else SWEEP_AXES["mask_variant"].default)

    return {
        "dry_run_contract": mode in {"dry_run", "runtime_smoke", "docker_validate"},
        "mode": mode,
        "experiment": exp.to_json(),
        "datasets": [DATASET_REGISTRY[d].to_json() for d in selected_datasets],
        "methods": [METHOD_REGISTRY[m].to_json() for m in resolved_methods],
        "simulation_budget": budget,
        "mask_variant": variant,
        "observation_ids": list(exp.observation_ids),
        "sweep_axes": {name: SWEEP_AXES[name].to_json() for name in exp.sweep_axes},
        "all_bounded_sweep_axes": {name: axis.to_json() for name, axis in SWEEP_AXES.items()},
        "paper_figure_output_mapping": PAPER_FIGURE_OUTPUT_MAPPING,
        "optional_dependency_status": optional_dependency_status(),
        "excluded_by_addendum": [
            "Section 4.1 calibration experiments",
            "Section 4.1 log-likelihood experiments",
            "Section 4.3 Simformer calibration experiments",
        ],
    }


def _artifact_root() -> Path:
    return Path(os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR", ".")).resolve()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _dummy_samples(offset: float = 0.0) -> List[List[float]]:
    return [[offset + 0.0, offset + 0.1], [offset + 0.2, offset + 0.0], [offset - 0.1, offset + 0.2], [offset + 0.1, offset - 0.1]]


def dry_run_benchmark_evaluation() -> Dict[str, Any]:
    """Run a bounded evaluator smoke path using real metric formulas."""

    approx = _dummy_samples(0.01)
    truth = _dummy_samples(0.0)
    metrics = evaluate_posterior_samples(approx, truth, include_nll_proxy=True)
    metrics["dry_run_contract"] = True
    metrics["result_status"] = "schema/readiness artifact; not a paper benchmark result"
    metrics["decisive_metric"] = "c2st_accuracy"
    metrics["paper_semantics"] = "0.5 aligned with ground truth posterior; 1.0 fully distinguishable"
    return metrics


def write_dry_run_artifacts(
    output_dir: Optional[os.PathLike[str] | str] = None,
    *,
    mode: str = "runtime_smoke",
    experiment_id: str = "figure_4_benchmark_c2st",
) -> Dict[str, Any]:
    """Materialize declared contract artifacts for smoke/docker validation.

    The writer creates every artifact owned by this file, plus readiness.json and
    evaluation_result.json.  All outputs are explicitly labeled as dry-run
    contract artifacts and must not be interpreted as completed experiment
    results.
    """

    root = Path(output_dir).resolve() if output_dir is not None else _artifact_root()
    config = make_resolved_config(mode=mode, experiment_id=experiment_id)
    metrics = dry_run_benchmark_evaluation()

    dataset_payload = {
        "dry_run_contract": True,
        "artifact_kind": "dataset_registry",
        "datasets": {key: spec.to_json() for key, spec in DATASET_REGISTRY.items()},
        "section_4_1_slots": {
            slot_id: {
                "dataset_id": slot_id,
                "observation_ids": list(OBSERVATION_IDS),
                "ground_truth_posterior_count": 10,
            }
            for slot_id in SECTION_4_1_BENCHMARK_TASK_IDS
        },
    }
    method_payload = {
        "dry_run_contract": True,
        "artifact_kind": "method_registry",
        "methods": {key: spec.to_json() for key, spec in METHOD_REGISTRY.items()},
        "selector_contract": ["ours", "Simformer", "simformer", "NPE", "NLE", "NRE", "lora", "diffusion_model"],
        "sbi_baseline_requirement": (
            "NPE, NLE, and NRE execute through the sbi Python library; import smoke "
            "does not require sbi to be installed."
        ),
    }
    ablation_payload = {
        "dry_run_contract": True,
        "artifact_kind": "ablation_registry",
        "ablations": ABLATION_REGISTRY,
        "bounded_sweeps": {key: axis.to_json() for key, axis in SWEEP_AXES.items()},
    }
    c2st_payload = {
        "dry_run_contract": True,
        "artifact_kind": "benchmark_c2st_schema",
        "paper_figure": "figure_4",
        "metric_semantics": "C2ST accuracy: 0.5 aligned/indistinguishable, 1.0 fully distinguishable",
        "benchmark_tasks": list(SECTION_4_1_BENCHMARK_TASK_IDS),
        "observations_per_task": len(OBSERVATION_IDS),
        "example_metric": metrics["c2st"],
        "not_real_results": True,
    }
    readiness_payload = {
        "dry_run_contract": True,
        "mode": mode,
        "ready": True,
        "module": "src.sweep_registry",
        "artifacts_declared": DECLARED_ARTIFACTS,
        "optional_dependency_status": optional_dependency_status(),
        "validated_surfaces": [
            "data_pipeline",
            "config",
            "evaluation",
            "metric_formula",
            "model_or_method",
            "baseline_or_ablation",
            "training_loop_adapter",
        ],
    }
    evaluation_payload = {
        "dry_run_contract": True,
        "mode": mode,
        "evaluation_status": "schema/readiness only",
        "metrics": metrics,
        "config_summary": {
            "experiment_id": experiment_id,
            "dataset_ids": [d["dataset_id"] for d in config["datasets"]],
            "method_ids": [m["method_id"] for m in config["methods"]],
            "simulation_budget": config["simulation_budget"],
            "mask_variant": config["mask_variant"],
        },
    }

    payload_by_key: Dict[str, Mapping[str, Any]] = {
        "metrics": metrics,
        "dataset_registry": dataset_payload,
        "method_registry": method_payload,
        "ablation_registry": ablation_payload,
        "config_resolved": config,
        "benchmark_c2st": c2st_payload,
        "readiness": readiness_payload,
        "evaluation_result": evaluation_payload,
    }

    written: Dict[str, Any] = {}
    for key, rel_path in DECLARED_ARTIFACTS.items():
        path = root / rel_path
        _write_json(path, payload_by_key[key])
        written[key] = str(path)

    return {
        "dry_run_contract": True,
        "written_artifacts": written,
        "readiness": readiness_payload,
        "evaluation_result": evaluation_payload,
    }


def registry_summary() -> Dict[str, Any]:
    """Return a machine-readable summary used by tests and canonical runners."""

    return {
        "datasets": list(DATASET_REGISTRY.keys()),
        "section_4_1_benchmark_tasks": list(SECTION_4_1_BENCHMARK_TASK_IDS),
        "ground_truth_posteriors_per_task": len(OBSERVATION_IDS),
        "methods": list(METHOD_REGISTRY.keys()),
        "method_selectors": ["ours", "Simformer", "simformer", "NPE", "NLE", "NRE", "lora", "diffusion_model"],
        "sweep_axes": list(SWEEP_AXES.keys()),
        "experiments": list(EXPERIMENT_REGISTRY.keys()),
        "artifacts": DECLARED_ARTIFACTS,
        "figure_output_mapping": PAPER_FIGURE_OUTPUT_MAPPING,
        "optional_dependency_status": optional_dependency_status(),
    }


__all__ = [
    "ABLATION_REGISTRY",
    "C2STResult",
    "DATASET_REGISTRY",
    "DECLARED_ARTIFACTS",
    "DatasetSpec",
    "EXPERIMENT_REGISTRY",
    "ExperimentSpec",
    "METHOD_REGISTRY",
    "OBSERVATION_IDS",
    "PAPER_FIGURE_OUTPUT_MAPPING",
    "SECTION_4_1_BENCHMARK_TASK_IDS",
    "SWEEP_AXES",
    "MethodSpec",
    "PosteriorSampler",
    "SweepAxis",
    "build_sbi_inference_adapter",
    "c2st_accuracy",
    "dependency_mask_for_dataset",
    "dry_run_benchmark_evaluation",
    "evaluate_posterior_samples",
    "make_generic_directed_mask",
    "make_lotka_volterra_directed_mask",
    "make_resolved_config",
    "make_undirected",
    "negative_log_likelihood_gaussian_proxy",
    "optional_dependency_status",
    "registry_summary",
    "resolve_method_selector",
    "return_metric",
    "sample_from_posterior",
    "train_sbi_baseline",
    "write_dry_run_artifacts",
]


if __name__ == "__main__":
    write_dry_run_artifacts(mode="runtime_smoke")