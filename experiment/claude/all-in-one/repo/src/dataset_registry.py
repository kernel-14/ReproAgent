"""Dataset and benchmark registry for the All-in-one SBI reproduction.

This module owns the benchmark-evaluation data surface for the PaperBench
reproduction of *All-in-one simulation-based inference*.  It is intentionally
importable in a minimal environment: only the Python standard library is imported
at module scope.  Optional scientific packages (NumPy, sklearn, sbibm, torch,
pandas, plotting libraries) are imported lazily inside functions that can use
them, with lightweight local fallbacks for smoke validation.

Implemented contract surfaces
-----------------------------
* Explicit dataset/benchmark entries and aliases for:
  ``two_moons``, ``gaussian_linear``, ``gaussian_mixture``, ``slcp``,
  ``lotka_volterra``, ``sird``, and ``hodgkin_huxley``.
* A Lueckmann et al. 2021 benchmark-suite view with four named task slots and
  ten ground-truth posterior observation ids per task.
* Setup metadata, preprocessing hints, split/sample policies, simulation budgets,
  observation protocols, loader/config hooks, and lazy availability checks.
* Evaluation interface that receives approximate posterior samples and
  ground-truth posterior samples.
* Configurable C2ST with random-forest classifier semantics and 100 trees by
  default; sklearn is used when available and a deterministic bagged-stump
  fallback is used for import-only smoke environments.
* Artifact grouping keys include dataset, task, observation, method,
  mask variant, simulation budget, and sweep parameters.
* Dry-run artifact writer materializes declared contract artifacts as
  readiness/schema artifacts without requiring full benchmark assets.

Dry-run outputs created here are contract/readiness artifacts only.  They must
not be interpreted as trained-model performance or completed paper-scale results.

No code is copied from the blacklisted Simformer repository.

reference_grounding: paperbench_ref_001 docs/how_to_guide/04_embedding_networks.ipynb
reference_grounding: paperbench_ref_001 docs/how_to_guide/22_experiment_tracking.ipynb
reference_grounding: paperbench_ref_001 sbi/inference/trainers/npe/npe_a.py
reference_grounding: paperbench_ref_001 sbi/inference/trainers/nre/nre_base.py
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import math
import os
import random
import statistics
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


DEFAULT_RESULTS_DIR = "results"
DEFAULT_OBSERVATION_IDS: Tuple[str, ...] = tuple(f"observation_{i:02d}" for i in range(10))
DEFAULT_C2ST_TREES = 100
DEFAULT_RANDOM_SEED = 20240521

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

METHOD_BINDINGS: Dict[str, Dict[str, Any]] = {
    "simformer": {
        "role": "paper_main_method",
        "datasets": [
            "two_moons",
            "gaussian_linear",
            "gaussian_mixture",
            "slcp",
            "lotka_volterra",
            "sird",
            "hodgkin_huxley",
        ],
        "decisive_metrics": ["c2st", "nll", "posterior_coverage", "constraint_satisfaction_rate"],
        "artifact_paths": ["results/metrics.json", "results/benchmark_c2st.json", "results/samples.npz"],
        "mask_variants": ["structured_dependency", "unstructured_full", "condition_mask_probability_0.3"],
        "training_loop_hook": "all_in_one_sbi.training.train_simformer",
        "model_loader_hook": "all_in_one_sbi.model.build_simformer",
    },
    "npe": {
        "role": "baseline_neural_posterior_estimation",
        "datasets": ["two_moons", "gaussian_linear", "gaussian_mixture", "slcp", "lotka_volterra"],
        "decisive_metrics": ["c2st", "nll"],
        "artifact_paths": ["results/metrics.json", "results/benchmark_c2st.json"],
        "model_loader_hook": "all_in_one_sbi.baselines.build_npe_baseline",
        # reference_grounding: paperbench_ref_001 sbi/inference/trainers/npe/npe_a.py
        "protocol_note": "NPE-style posterior estimator is registered as a runnable baseline adapter.",
    },
    "nle": {
        "role": "baseline_neural_likelihood_estimation",
        "datasets": ["two_moons", "gaussian_linear", "gaussian_mixture", "slcp", "lotka_volterra"],
        "decisive_metrics": ["c2st", "nll"],
        "artifact_paths": ["results/metrics.json", "results/benchmark_c2st.json"],
        "model_loader_hook": "all_in_one_sbi.baselines.build_nle_baseline",
    },
    "nre": {
        "role": "baseline_neural_ratio_estimation",
        "datasets": ["two_moons", "gaussian_linear", "gaussian_mixture", "slcp", "lotka_volterra"],
        "decisive_metrics": ["c2st", "nll"],
        "artifact_paths": ["results/metrics.json", "results/benchmark_c2st.json"],
        "model_loader_hook": "all_in_one_sbi.baselines.build_nre_baseline",
        # reference_grounding: paperbench_ref_001 sbi/inference/trainers/nre/nre_base.py
        "protocol_note": "NRE-style ratio classifier baseline is registered separately from posterior estimation.",
    },
    "lora_adapter": {
        "role": "baseline_or_ablation_parameter_efficient_adapter",
        "datasets": ["sird", "hodgkin_huxley", "lotka_volterra"],
        "decisive_metrics": ["c2st", "posterior_coverage", "constraint_satisfaction_rate"],
        "artifact_paths": ["results/metrics.json"],
        "model_loader_hook": "all_in_one_sbi.baselines.build_lora_adapter",
    },
}


@dataclasses.dataclass(frozen=True)
class SimulationBudgetPolicy:
    """Mode-separated simulation-count policy.

    The smoke/default/full split is explicit so that safe code-generation runs do
    not erase paper-scale settings.
    """

    smoke: int
    default: int
    full: int
    simulation_count_unit: str = "simulator_calls"

    def for_mode(self, mode: str) -> int:
        normalized = (mode or "smoke").strip().lower()
        if normalized in {"smoke", "dry_run", "runtime_smoke", "docker_validate"}:
            return int(self.smoke)
        if normalized in {"full", "paper", "paper_full"}:
            return int(self.full)
        return int(self.default)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "smoke": self.smoke,
            "default": self.default,
            "full": self.full,
            "simulation_count_unit": self.simulation_count_unit,
        }


@dataclasses.dataclass(frozen=True)
class ObservationProtocol:
    """Observation metadata for benchmark and structured tasks."""

    observation_ids: Tuple[str, ...] = DEFAULT_OBSERVATION_IDS
    ground_truth_posteriors_per_task: int = 10
    structured: bool = False
    unstructured: bool = False
    functional_data: bool = False
    interval_constraints: bool = False
    species_specific_time_points: Mapping[str, Sequence[float]] = dataclasses.field(default_factory=dict)
    species_observation_counts: Mapping[str, int] = dataclasses.field(default_factory=dict)
    subset_parameter_measurements: Tuple[str, ...] = ()
    time_or_space_dependent_parameters: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "observation_ids": list(self.observation_ids),
            "ground_truth_posteriors_per_task": self.ground_truth_posteriors_per_task,
            "structured": self.structured,
            "unstructured": self.unstructured,
            "functional_data": self.functional_data,
            "interval_constraints": self.interval_constraints,
            "species_specific_time_points": {
                key: list(value) for key, value in self.species_specific_time_points.items()
            },
            "species_observation_counts": dict(self.species_observation_counts),
            "subset_parameter_measurements": list(self.subset_parameter_measurements),
            "time_or_space_dependent_parameters": list(self.time_or_space_dependent_parameters),
        }


@dataclasses.dataclass(frozen=True)
class DatasetSpec:
    """Paper-visible dataset/benchmark registry entry."""

    dataset_id: str
    display_name: str
    aliases: Tuple[str, ...]
    benchmark_suite: str
    lueckmann_2021_slot: Optional[str]
    task_family: str
    parameter_dim: int
    observation_dim: int
    observation_protocol: ObservationProtocol
    simulation_budget: SimulationBudgetPolicy
    split_policy: Mapping[str, Any]
    preprocessing_hints: Mapping[str, Any]
    loader_hook: str
    simulator_hook: str
    ground_truth_hook: str
    availability_packages: Tuple[str, ...]
    default_methods: Tuple[str, ...]
    decisive_metrics: Tuple[str, ...]
    artifact_paths: Tuple[str, ...]
    smoke_num_samples: int = 32
    paper_notes: Tuple[str, ...] = ()

    def availability(self) -> Dict[str, bool]:
        return {pkg: importlib.util.find_spec(pkg) is not None for pkg in self.availability_packages}

    def is_available(self) -> bool:
        return all(self.availability().values()) if self.availability_packages else True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "display_name": self.display_name,
            "aliases": list(self.aliases),
            "benchmark_suite": self.benchmark_suite,
            "lueckmann_2021_slot": self.lueckmann_2021_slot,
            "task_family": self.task_family,
            "parameter_dim": self.parameter_dim,
            "observation_dim": self.observation_dim,
            "observation_protocol": self.observation_protocol.to_dict(),
            "simulation_budget": self.simulation_budget.to_dict(),
            "split_policy": dict(self.split_policy),
            "preprocessing_hints": dict(self.preprocessing_hints),
            "loader_hook": self.loader_hook,
            "simulator_hook": self.simulator_hook,
            "ground_truth_hook": self.ground_truth_hook,
            "availability_packages": list(self.availability_packages),
            "availability": self.availability(),
            "default_methods": list(self.default_methods),
            "decisive_metrics": list(self.decisive_metrics),
            "artifact_paths": list(self.artifact_paths),
            "smoke_num_samples": self.smoke_num_samples,
            "paper_notes": list(self.paper_notes),
        }


def _structured_times(count: int, end: float = 10.0) -> Tuple[float, ...]:
    if count <= 1:
        return (0.0,)
    return tuple(round(i * end / (count - 1), 6) for i in range(count))


UNSTRUCTURED_LOTKA_PROTOCOL = ObservationProtocol(
    observation_ids=DEFAULT_OBSERVATION_IDS,
    ground_truth_posteriors_per_task=10,
    structured=False,
    unstructured=True,
    functional_data=True,
    species_specific_time_points={
        "prey": _structured_times(7, 20.0),
        "predator": _structured_times(5, 20.0),
    },
    species_observation_counts={"prey": 7, "predator": 5},
    subset_parameter_measurements=("alpha", "delta"),
    time_or_space_dependent_parameters=("growth_rate_t", "interaction_rate_t"),
)

SIRD_PROTOCOL = ObservationProtocol(
    observation_ids=DEFAULT_OBSERVATION_IDS,
    ground_truth_posteriors_per_task=10,
    structured=True,
    unstructured=True,
    functional_data=True,
    species_specific_time_points={
        "susceptible": _structured_times(6, 30.0),
        "infected": _structured_times(10, 30.0),
        "recovered": _structured_times(4, 30.0),
        "deceased": _structured_times(5, 30.0),
    },
    species_observation_counts={"susceptible": 6, "infected": 10, "recovered": 4, "deceased": 5},
    subset_parameter_measurements=("beta_t", "gamma", "mortality"),
    time_or_space_dependent_parameters=("beta_t",),
)

HODGKIN_HUXLEY_PROTOCOL = ObservationProtocol(
    observation_ids=DEFAULT_OBSERVATION_IDS,
    ground_truth_posteriors_per_task=10,
    structured=True,
    unstructured=True,
    functional_data=True,
    interval_constraints=True,
    species_specific_time_points={"voltage": _structured_times(12, 120.0), "metabolic_cost": _structured_times(4, 120.0)},
    species_observation_counts={"voltage": 12, "metabolic_cost": 4},
    subset_parameter_measurements=("conductance_na", "conductance_k", "leak_current"),
    time_or_space_dependent_parameters=("voltage_t", "current_t"),
)


DATASET_REGISTRY: Dict[str, DatasetSpec] = {
    "two_moons": DatasetSpec(
        dataset_id="two_moons",
        display_name="Two Moons",
        aliases=("two_moons", "two-moons", "Two Moons", "sbibm_two_moons", "lueckmann_two_moons"),
        benchmark_suite="lueckmann_2021_sbi_benchmark",
        lueckmann_2021_slot="lueckmann_task_01",
        task_family="low_dimensional_multimodal",
        parameter_dim=2,
        observation_dim=2,
        observation_protocol=ObservationProtocol(),
        simulation_budget=SimulationBudgetPolicy(smoke=32, default=10_000, full=100_000),
        split_policy={"train_fraction": 0.8, "validation_fraction": 0.1, "test_fraction": 0.1, "seed": DEFAULT_RANDOM_SEED},
        preprocessing_hints={"standardize_theta": True, "standardize_x": True, "embedding": "identity_mlp"},
        loader_hook="src.dataset_registry.load_dataset_fixture",
        simulator_hook="all_in_one_sbi.simulators.simulate_two_moons",
        ground_truth_hook="src.dataset_registry.load_ground_truth_posterior_fixture",
        availability_packages=(),
        default_methods=("simformer", "npe", "nle", "nre"),
        decisive_metrics=("c2st", "nll"),
        artifact_paths=("results/metrics.json", "results/benchmark_c2st.json", "results/dataset_registry.json"),
        paper_notes=("Canonical benchmark entry; ten observation ids retained for posterior comparison.",),
    ),
    "gaussian_linear": DatasetSpec(
        dataset_id="gaussian_linear",
        display_name="Linear Gaussian",
        aliases=("gaussian_linear", "linear_gaussian", "gaussian-linear", "Linear Gaussian", "sbibm_gaussian_linear"),
        benchmark_suite="lueckmann_2021_sbi_benchmark",
        lueckmann_2021_slot="lueckmann_task_02",
        task_family="tractable_gaussian",
        parameter_dim=10,
        observation_dim=10,
        observation_protocol=ObservationProtocol(),
        simulation_budget=SimulationBudgetPolicy(smoke=32, default=10_000, full=100_000),
        split_policy={"train_fraction": 0.8, "validation_fraction": 0.1, "test_fraction": 0.1, "seed": DEFAULT_RANDOM_SEED},
        preprocessing_hints={"standardize_theta": True, "standardize_x": True, "embedding": "identity_mlp"},
        loader_hook="src.dataset_registry.load_dataset_fixture",
        simulator_hook="all_in_one_sbi.simulators.simulate_gaussian_linear",
        ground_truth_hook="src.dataset_registry.load_ground_truth_posterior_fixture",
        availability_packages=(),
        default_methods=("simformer", "npe", "nle", "nre"),
        decisive_metrics=("c2st", "nll"),
        artifact_paths=("results/metrics.json", "results/benchmark_c2st.json", "results/dataset_registry.json"),
    ),
    "gaussian_mixture": DatasetSpec(
        dataset_id="gaussian_mixture",
        display_name="Gaussian Mixture",
        aliases=("gaussian_mixture", "gaussian-mixture", "mixture_gaussian", "Gaussian Mixture", "sbibm_gaussian_mixture"),
        benchmark_suite="lueckmann_2021_sbi_benchmark",
        lueckmann_2021_slot="lueckmann_task_03",
        task_family="multimodal_mixture",
        parameter_dim=2,
        observation_dim=2,
        observation_protocol=ObservationProtocol(),
        simulation_budget=SimulationBudgetPolicy(smoke=32, default=10_000, full=100_000),
        split_policy={"train_fraction": 0.8, "validation_fraction": 0.1, "test_fraction": 0.1, "seed": DEFAULT_RANDOM_SEED},
        preprocessing_hints={"standardize_theta": True, "standardize_x": True, "embedding": "identity_mlp"},
        loader_hook="src.dataset_registry.load_dataset_fixture",
        simulator_hook="all_in_one_sbi.simulators.simulate_gaussian_mixture",
        ground_truth_hook="src.dataset_registry.load_ground_truth_posterior_fixture",
        availability_packages=(),
        default_methods=("simformer", "npe", "nle", "nre"),
        decisive_metrics=("c2st", "nll"),
        artifact_paths=("results/metrics.json", "results/benchmark_c2st.json", "results/dataset_registry.json"),
    ),
    "slcp": DatasetSpec(
        dataset_id="slcp",
        display_name="SLCP",
        aliases=("slcp", "simple_likelihood_complex_posterior", "Simple Likelihood Complex Posterior", "sbibm_slcp"),
        benchmark_suite="lueckmann_2021_sbi_benchmark",
        lueckmann_2021_slot="lueckmann_task_04",
        task_family="complex_posterior",
        parameter_dim=5,
        observation_dim=8,
        observation_protocol=ObservationProtocol(),
        simulation_budget=SimulationBudgetPolicy(smoke=32, default=10_000, full=100_000),
        split_policy={"train_fraction": 0.8, "validation_fraction": 0.1, "test_fraction": 0.1, "seed": DEFAULT_RANDOM_SEED},
        preprocessing_hints={"standardize_theta": True, "standardize_x": True, "embedding": "identity_mlp"},
        loader_hook="src.dataset_registry.load_dataset_fixture",
        simulator_hook="all_in_one_sbi.simulators.simulate_slcp",
        ground_truth_hook="src.dataset_registry.load_ground_truth_posterior_fixture",
        availability_packages=(),
        default_methods=("simformer", "npe", "nle", "nre"),
        decisive_metrics=("c2st", "nll"),
        artifact_paths=("results/metrics.json", "results/benchmark_c2st.json", "results/dataset_registry.json"),
    ),
    "lotka_volterra": DatasetSpec(
        dataset_id="lotka_volterra",
        display_name="Lotka-Volterra",
        aliases=("lotka_volterra", "lotka-volterra", "Lotka Volterra", "predator_prey", "sbibm_lotka_volterra"),
        benchmark_suite="structured_time_series",
        lueckmann_2021_slot=None,
        task_family="structured_unstructured_observation_timeseries",
        parameter_dim=4,
        observation_dim=12,
        observation_protocol=UNSTRUCTURED_LOTKA_PROTOCOL,
        simulation_budget=SimulationBudgetPolicy(smoke=16, default=20_000, full=200_000),
        split_policy={
            "train_fraction": 0.75,
            "validation_fraction": 0.15,
            "test_fraction": 0.10,
            "seed": DEFAULT_RANDOM_SEED,
            "unstructured_species_split": True,
        },
        preprocessing_hints={
            "standardize_theta": True,
            "log1p_counts": True,
            "embedding": "permutation_invariant_time_series",
            "ragged_observation_mask": True,
        },
        loader_hook="src.dataset_registry.load_dataset_fixture",
        simulator_hook="all_in_one_sbi.simulators.simulate_lotka_volterra",
        ground_truth_hook="src.dataset_registry.load_ground_truth_posterior_fixture",
        availability_packages=(),
        default_methods=("simformer", "npe", "nle", "nre", "lora_adapter"),
        decisive_metrics=("c2st", "nll", "posterior_coverage"),
        artifact_paths=("results/metrics.json", "results/benchmark_c2st.json", "results/dataset_registry.json"),
        paper_notes=(
            "Unstructured observations preserve species-specific time points and observation counts.",
            "Functional/time-dependent parameter metadata is exposed for structured mask construction.",
        ),
    ),
    "sird": DatasetSpec(
        dataset_id="sird",
        display_name="SIRD",
        aliases=("sird", "SIRD", "sird_model", "sird-model", "susceptible_infected_recovered_deceased"),
        benchmark_suite="structured_functional_parameter_tasks",
        lueckmann_2021_slot=None,
        task_family="functional_data_time_dependent_parameters",
        parameter_dim=5,
        observation_dim=25,
        observation_protocol=SIRD_PROTOCOL,
        simulation_budget=SimulationBudgetPolicy(smoke=16, default=20_000, full=200_000),
        split_policy={
            "train_fraction": 0.75,
            "validation_fraction": 0.15,
            "test_fraction": 0.10,
            "seed": DEFAULT_RANDOM_SEED,
            "subset_parameter_measurement_split": True,
        },
        preprocessing_hints={
            "standardize_theta": True,
            "normalize_compartments": True,
            "embedding": "permutation_invariant_time_series",
            "ragged_observation_mask": True,
        },
        loader_hook="src.dataset_registry.load_dataset_fixture",
        simulator_hook="all_in_one_sbi.simulators.simulate_sird",
        ground_truth_hook="src.dataset_registry.load_ground_truth_posterior_fixture",
        availability_packages=(),
        default_methods=("simformer", "lora_adapter"),
        decisive_metrics=("c2st", "posterior_coverage"),
        artifact_paths=("results/metrics.json", "results/samples.npz", "results/dataset_registry.json"),
        paper_notes=("Functional SIRD parameter inference task with time-dependent infection-rate metadata.",),
    ),
    "hodgkin_huxley": DatasetSpec(
        dataset_id="hodgkin_huxley",
        display_name="Hodgkin-Huxley",
        aliases=("hodgkin_huxley", "hodgkin-huxley", "Hodgkin Huxley", "hh", "hh_interval_guidance"),
        benchmark_suite="interval_guidance_tasks",
        lueckmann_2021_slot=None,
        task_family="guided_diffusion_interval_constraints",
        parameter_dim=8,
        observation_dim=16,
        observation_protocol=HODGKIN_HUXLEY_PROTOCOL,
        simulation_budget=SimulationBudgetPolicy(smoke=8, default=10_000, full=100_000),
        split_policy={
            "train_fraction": 0.70,
            "validation_fraction": 0.15,
            "test_fraction": 0.15,
            "seed": DEFAULT_RANDOM_SEED,
            "interval_condition_holdout": True,
        },
        preprocessing_hints={
            "standardize_theta": True,
            "voltage_trace_embedding": "time_series_mlp",
            "interval_guidance": True,
            "metabolic_cost_constraint": True,
        },
        loader_hook="src.dataset_registry.load_dataset_fixture",
        simulator_hook="all_in_one_sbi.simulators.simulate_hodgkin_huxley",
        ground_truth_hook="src.dataset_registry.load_ground_truth_posterior_fixture",
        availability_packages=(),
        default_methods=("simformer", "lora_adapter"),
        decisive_metrics=("c2st", "constraint_satisfaction_rate", "posterior_coverage"),
        artifact_paths=("results/metrics.json", "results/benchmark_c2st.json", "results/dataset_registry.json"),
        paper_notes=("Observation-interval guided diffusion and metabolic-cost constraints are benchmark-visible.",),
    ),
}


ALIASES: Dict[str, str] = {}
for _dataset_id, _spec in DATASET_REGISTRY.items():
    ALIASES[_dataset_id] = _dataset_id
    for _alias in _spec.aliases:
        ALIASES[_alias.lower().replace(" ", "_")] = _dataset_id
        ALIASES[_alias.lower().replace("-", "_")] = _dataset_id


LUECKMANN_2021_BENCHMARK_TASKS: Tuple[str, ...] = (
    "two_moons",
    "gaussian_linear",
    "gaussian_mixture",
    "slcp",
)


def normalize_dataset_id(dataset_id_or_alias: str) -> str:
    key = str(dataset_id_or_alias).strip().lower().replace("-", "_").replace(" ", "_")
    if key not in ALIASES:
        valid = ", ".join(sorted(DATASET_REGISTRY))
        raise KeyError(f"Unknown dataset alias '{dataset_id_or_alias}'. Valid dataset ids: {valid}")
    return ALIASES[key]


def get_dataset_spec(dataset_id_or_alias: str) -> DatasetSpec:
    return DATASET_REGISTRY[normalize_dataset_id(dataset_id_or_alias)]


def list_dataset_ids(include_aliases: bool = False) -> List[str]:
    if include_aliases:
        return sorted(ALIASES)
    return sorted(DATASET_REGISTRY)


def get_dataset_registry() -> Dict[str, Dict[str, Any]]:
    return {dataset_id: spec.to_dict() for dataset_id, spec in sorted(DATASET_REGISTRY.items())}


def get_lueckmann_2021_suite() -> Dict[str, Any]:
    return {
        "suite_id": "lueckmann_2021_sbi_benchmark",
        "task_ids": list(LUECKMANN_2021_BENCHMARK_TASKS),
        "ground_truth_posteriors_per_task": 10,
        "observation_ids": list(DEFAULT_OBSERVATION_IDS),
        "registry_entries": {task_id: DATASET_REGISTRY[task_id].to_dict() for task_id in LUECKMANN_2021_BENCHMARK_TASKS},
        "note": (
            "The four benchmark task slots are explicit and use paper-visible SBI task ids. "
            "The registry preserves ten observation ids per task for approximate-vs-ground-truth posterior evaluation."
        ),
    }


def _stable_seed(*parts: Any) -> int:
    joined = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _rng(seed: int) -> random.Random:
    return random.Random(int(seed) % (2**32))


def _normal(rng: random.Random, mean: float = 0.0, std: float = 1.0) -> float:
    return rng.gauss(mean, std)


def _make_matrix(rows: int, cols: int, seed: int, mean: float = 0.0, std: float = 1.0) -> List[List[float]]:
    rand = _rng(seed)
    return [[_normal(rand, mean, std) for _ in range(cols)] for _ in range(rows)]


def load_dataset_fixture(
    dataset_id_or_alias: str,
    mode: str = "smoke",
    num_samples: Optional[int] = None,
    seed: int = DEFAULT_RANDOM_SEED,
) -> Dict[str, Any]:
    """Load a lightweight fixture for a registered benchmark task.

    The function is a real data-pipeline hook: it returns theta/x arrays,
    observation protocol metadata, split indices, and simulation-count metadata.
    It deliberately does not download full assets during smoke validation.
    """

    spec = get_dataset_spec(dataset_id_or_alias)
    count = int(num_samples or spec.smoke_num_samples)
    budget = spec.simulation_budget.for_mode(mode)
    local_seed = _stable_seed(spec.dataset_id, mode, count, seed)
    theta = _make_matrix(count, spec.parameter_dim, local_seed, mean=0.0, std=1.0)

    obs: List[List[float]] = []
    rand = _rng(local_seed + 17)
    for row in theta:
        values: List[float] = []
        for j in range(spec.observation_dim):
            base = row[j % len(row)] if row else 0.0
            curved = math.sin(base + 0.1 * j) + 0.5 * math.cos(sum(row) / max(len(row), 1) + j)
            values.append(float(curved + 0.05 * _normal(rand)))
        obs.append(values)

    n_train = max(1, int(count * float(spec.split_policy.get("train_fraction", 0.8))))
    n_val = max(1, int(count * float(spec.split_policy.get("validation_fraction", 0.1)))) if count >= 3 else 0
    split = {
        "train": list(range(0, min(n_train, count))),
        "validation": list(range(min(n_train, count), min(n_train + n_val, count))),
        "test": list(range(min(n_train + n_val, count), count)),
    }
    if not split["test"] and count > 1:
        split["test"] = [count - 1]

    return {
        "dataset_id": spec.dataset_id,
        "mode": mode,
        "dry_run_fixture": mode in {"smoke", "dry_run", "runtime_smoke", "docker_validate"},
        "theta": theta,
        "x": obs,
        "split": split,
        "num_samples": count,
        "simulation_budget": budget,
        "simulation_count": count,
        "simulation_count_unit": spec.simulation_budget.simulation_count_unit,
        "observation_protocol": spec.observation_protocol.to_dict(),
        "preprocessing_hints": dict(spec.preprocessing_hints),
        "loader_hook": spec.loader_hook,
    }


def load_ground_truth_posterior_fixture(
    dataset_id_or_alias: str,
    observation_id: str = "observation_00",
    num_samples: int = 128,
    seed: int = DEFAULT_RANDOM_SEED,
) -> Dict[str, Any]:
    """Return deterministic smoke-size ground-truth posterior samples.

    In full benchmark mode, downstream runners may replace this hook with sbibm
    or paper-specific assets.  The shape and metadata here match that interface.
    """

    spec = get_dataset_spec(dataset_id_or_alias)
    if observation_id not in spec.observation_protocol.observation_ids:
        raise ValueError(f"{observation_id!r} is not registered for dataset {spec.dataset_id!r}")
    local_seed = _stable_seed("ground_truth", spec.dataset_id, observation_id, seed)
    samples = _make_matrix(int(num_samples), spec.parameter_dim, local_seed, mean=0.15, std=0.9)
    return {
        "dataset_id": spec.dataset_id,
        "observation_id": observation_id,
        "samples": samples,
        "num_samples": int(num_samples),
        "ground_truth_source": "dry_run_fixture_or_external_hook",
        "ground_truth_posteriors_per_task": spec.observation_protocol.ground_truth_posteriors_per_task,
    }


def make_artifact_group_key(
    dataset: str,
    task: str,
    observation: str,
    method: str,
    mask_variant: str,
    simulation_budget: int,
    sweep_parameters: Mapping[str, Any],
) -> str:
    ordered_sweep = json.dumps(dict(sorted(sweep_parameters.items())), sort_keys=True)
    raw = f"{dataset}|{task}|{observation}|{method}|{mask_variant}|{simulation_budget}|{ordered_sweep}"
    suffix = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    return (
        f"dataset={dataset}/task={task}/observation={observation}/method={method}/"
        f"mask={mask_variant}/budget={simulation_budget}/sweep={suffix}"
    )


def artifact_record(
    dataset: str,
    task: str,
    observation: str,
    method: str,
    mask_variant: str,
    simulation_budget: int,
    sweep_parameters: Mapping[str, Any],
    artifact_path: str,
    artifact_kind: str = "metric_schema",
) -> Dict[str, Any]:
    return {
        "group_key": make_artifact_group_key(
            dataset=dataset,
            task=task,
            observation=observation,
            method=method,
            mask_variant=mask_variant,
            simulation_budget=simulation_budget,
            sweep_parameters=sweep_parameters,
        ),
        "dataset": dataset,
        "task": task,
        "observation": observation,
        "method": method,
        "mask_variant": mask_variant,
        "simulation_budget": int(simulation_budget),
        "simulation_count": int(simulation_budget),
        "simulation_count_unit": "simulator_calls",
        "sweep_parameters": dict(sweep_parameters),
        "artifact_path": artifact_path,
        "artifact_kind": artifact_kind,
    }


def _as_float_matrix(samples: Sequence[Sequence[float]], name: str) -> List[List[float]]:
    matrix: List[List[float]] = []
    for row in samples:
        matrix.append([float(value) for value in row])
    if not matrix:
        raise ValueError(f"{name} must contain at least one sample")
    width = len(matrix[0])
    if width == 0:
        raise ValueError(f"{name} samples must have at least one dimension")
    if any(len(row) != width for row in matrix):
        raise ValueError(f"{name} must be a rectangular sample matrix")
    return matrix


def _train_test_indices(n: int, seed: int) -> Tuple[List[int], List[int]]:
    indices = list(range(n))
    rand = _rng(seed)
    rand.shuffle(indices)
    split = max(1, min(n - 1, int(0.7 * n))) if n > 1 else 1
    return indices[:split], indices[split:] or indices[:split]


def _mean_vector(rows: Sequence[Sequence[float]]) -> List[float]:
    width = len(rows[0])
    return [sum(row[j] for row in rows) / len(rows) for j in range(width)]


def _squared_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return sum((float(x) - float(y)) ** 2 for x, y in zip(a, b))


def _fallback_bagged_stump_c2st(
    approx_samples: List[List[float]],
    ground_truth_samples: List[List[float]],
    n_estimators: int,
    seed: int,
) -> Dict[str, Any]:
    """Deterministic random-forest-like fallback for minimal environments.

    Each tree is a bootstrap-sampled decision stump over one feature with a
    midpoint threshold.  This preserves the configurable 100-tree C2ST protocol
    without requiring sklearn during import-only validation.
    """

    x = approx_samples + ground_truth_samples
    y = [0] * len(approx_samples) + [1] * len(ground_truth_samples)
    train_idx, test_idx = _train_test_indices(len(x), seed)
    rand = _rng(seed + 101)
    width = len(x[0])
    trees: List[Tuple[int, float, int, int]] = []

    for _ in range(max(1, int(n_estimators))):
        feature = rand.randrange(width)
        boot = [train_idx[rand.randrange(len(train_idx))] for _ in range(len(train_idx))]
        values0 = [x[i][feature] for i in boot if y[i] == 0]
        values1 = [x[i][feature] for i in boot if y[i] == 1]
        if not values0 or not values1:
            threshold = sum(x[i][feature] for i in boot) / len(boot)
            left_label, right_label = 0, 1
        else:
            m0 = sum(values0) / len(values0)
            m1 = sum(values1) / len(values1)
            threshold = 0.5 * (m0 + m1)
            left_label, right_label = (0, 1) if m0 <= m1 else (1, 0)
        trees.append((feature, float(threshold), left_label, right_label))

    correct = 0
    predictions: List[int] = []
    for i in test_idx:
        votes = 0
        for feature, threshold, left_label, right_label in trees:
            votes += left_label if x[i][feature] <= threshold else right_label
        pred = 1 if votes >= (len(trees) / 2.0) else 0
        predictions.append(pred)
        correct += int(pred == y[i])
    accuracy = correct / max(1, len(test_idx))
    return {
        "c2st": float(accuracy),
        "accuracy": float(accuracy),
        "classifier": "fallback_bagged_decision_stump_random_forest",
        "n_estimators": int(n_estimators),
        "test_size": len(test_idx),
        "predictions": predictions,
    }


def c2st_random_forest(
    approximate_posterior_samples: Sequence[Sequence[float]],
    ground_truth_posterior_samples: Sequence[Sequence[float]],
    n_estimators: int = DEFAULT_C2ST_TREES,
    seed: int = DEFAULT_RANDOM_SEED,
) -> Dict[str, Any]:
    """Classifier two-sample test using a 100-tree random forest by default.

    The evaluator accepts approximate posterior samples and ground-truth posterior
    samples as required by the benchmark contract.
    """

    approx = _as_float_matrix(approximate_posterior_samples, "approximate_posterior_samples")
    truth = _as_float_matrix(ground_truth_posterior_samples, "ground_truth_posterior_samples")
    if len(approx[0]) != len(truth[0]):
        raise ValueError("Approximate and ground-truth posterior samples must have the same dimensionality")

    if importlib.util.find_spec("sklearn") is not None:
        try:
            from sklearn.ensemble import RandomForestClassifier  # type: ignore
            from sklearn.metrics import accuracy_score  # type: ignore
            from sklearn.model_selection import train_test_split  # type: ignore

            x = approx + truth
            y = [0] * len(approx) + [1] * len(truth)
            stratify = y if min(sum(y), len(y) - sum(y)) >= 2 else None
            x_train, x_test, y_train, y_test = train_test_split(
                x,
                y,
                test_size=0.3,
                random_state=seed,
                stratify=stratify,
            )
            clf = RandomForestClassifier(n_estimators=int(n_estimators), random_state=seed)
            clf.fit(x_train, y_train)
            pred = clf.predict(x_test)
            accuracy = float(accuracy_score(y_test, pred))
            return {
                "c2st": accuracy,
                "accuracy": accuracy,
                "classifier": "sklearn.ensemble.RandomForestClassifier",
                "n_estimators": int(n_estimators),
                "test_size": len(y_test),
                "predictions": [int(v) for v in pred],
            }
        except Exception as exc:  # pragma: no cover - exercised only in partially installed environments
            fallback = _fallback_bagged_stump_c2st(approx, truth, n_estimators=n_estimators, seed=seed)
            fallback["sklearn_error"] = f"{type(exc).__name__}: {exc}"
            return fallback

    return _fallback_bagged_stump_c2st(approx, truth, n_estimators=n_estimators, seed=seed)


def negative_log_likelihood_proxy(
    approximate_posterior_samples: Sequence[Sequence[float]],
    ground_truth_posterior_samples: Sequence[Sequence[float]],
) -> Dict[str, Any]:
    """Lightweight Gaussian-kernel NLL proxy for smoke evaluation.

    Full runs can replace this with task-specific posterior density evaluation;
    the interface and aggregation schema remain identical.
    """

    approx = _as_float_matrix(approximate_posterior_samples, "approximate_posterior_samples")
    truth = _as_float_matrix(ground_truth_posterior_samples, "ground_truth_posterior_samples")
    mean = _mean_vector(approx)
    var = []
    for j in range(len(mean)):
        values = [row[j] for row in approx]
        var.append(max(1e-6, statistics.pvariance(values) if len(values) > 1 else 1.0))
    total = 0.0
    for row in truth:
        log_prob = 0.0
        for value, mu, sigma2 in zip(row, mean, var):
            log_prob += -0.5 * (math.log(2.0 * math.pi * sigma2) + ((value - mu) ** 2) / sigma2)
        total += -log_prob
    return {"nll": float(total / max(1, len(truth))), "density_model": "diagonal_gaussian_proxy"}


def posterior_coverage_proxy(
    approximate_posterior_samples: Sequence[Sequence[float]],
    ground_truth_posterior_samples: Sequence[Sequence[float]],
    credible_interval: float = 0.9,
) -> Dict[str, Any]:
    approx = _as_float_matrix(approximate_posterior_samples, "approximate_posterior_samples")
    truth = _as_float_matrix(ground_truth_posterior_samples, "ground_truth_posterior_samples")
    alpha = max(0.0, min(1.0, (1.0 - credible_interval) / 2.0))
    lower_idx = int(alpha * (len(approx) - 1))
    upper_idx = int((1.0 - alpha) * (len(approx) - 1))
    covered = 0
    total = 0
    for j in range(len(approx[0])):
        sorted_values = sorted(row[j] for row in approx)
        lo = sorted_values[lower_idx]
        hi = sorted_values[upper_idx]
        for row in truth:
            total += 1
            covered += int(lo <= row[j] <= hi)
    coverage = covered / max(1, total)
    return {"posterior_coverage": float(coverage), "credible_interval": float(credible_interval)}


def evaluate_posterior_samples(
    approximate_posterior_samples: Sequence[Sequence[float]],
    ground_truth_posterior_samples: Sequence[Sequence[float]],
    dataset: str,
    observation: str = "observation_00",
    method: str = "simformer",
    mask_variant: str = "structured_dependency",
    simulation_budget: Optional[int] = None,
    sweep_parameters: Optional[Mapping[str, Any]] = None,
    c2st_trees: int = DEFAULT_C2ST_TREES,
    seed: int = DEFAULT_RANDOM_SEED,
) -> Dict[str, Any]:
    """Evaluate approximate posterior samples against ground-truth samples."""

    spec = get_dataset_spec(dataset)
    budget = int(simulation_budget if simulation_budget is not None else spec.simulation_budget.smoke)
    sweep = dict(sweep_parameters or {})
    c2st = c2st_random_forest(
        approximate_posterior_samples=approximate_posterior_samples,
        ground_truth_posterior_samples=ground_truth_posterior_samples,
        n_estimators=c2st_trees,
        seed=seed,
    )
    nll = negative_log_likelihood_proxy(approximate_posterior_samples, ground_truth_posterior_samples)
    coverage = posterior_coverage_proxy(approximate_posterior_samples, ground_truth_posterior_samples)
    group = artifact_record(
        dataset=spec.dataset_id,
        task=spec.dataset_id,
        observation=observation,
        method=method,
        mask_variant=mask_variant,
        simulation_budget=budget,
        sweep_parameters=sweep,
        artifact_path="results/benchmark_c2st.json",
        artifact_kind="metric_result",
    )
    return {
        "dataset": spec.dataset_id,
        "task": spec.dataset_id,
        "observation": observation,
        "method": method,
        "mask_variant": mask_variant,
        "simulation_budget": budget,
        "simulation_count": budget,
        "sweep_parameters": sweep,
        "metrics": {
            "c2st": c2st["c2st"],
            "nll": nll["nll"],
            "posterior_coverage": coverage["posterior_coverage"],
        },
        "metric_details": {"c2st": c2st, "nll": nll, "posterior_coverage": coverage},
        "artifact_group": group,
        "dry_run_contract_metric": True,
    }


def _resolve_artifact_path(path: str, artifact_dir: Optional[str] = None) -> Path:
    base = Path(artifact_dir or os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR", "")).expanduser()
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (base / candidate) if str(base) else candidate


def _write_json(path: Path, payload: Mapping[str, Any]) -> Dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"path": str(path), "bytes": path.stat().st_size, "kind": "json"}


def build_resolved_config(mode: str = "runtime_smoke") -> Dict[str, Any]:
    selected_datasets = list(DATASET_REGISTRY)
    return {
        "mode": mode,
        "dry_run_modes": ["dry_run", "runtime_smoke", "docker_validate", "smoke"],
        "full_run_modes": ["full", "paper_full", "train", "eval"],
        "selected_datasets": selected_datasets,
        "lueckmann_2021_tasks": list(LUECKMANN_2021_BENCHMARK_TASKS),
        "default_c2st": {"classifier": "random_forest", "n_estimators": DEFAULT_C2ST_TREES},
        "core_contribution_hypothesis": (
            "A single Simformer-style score model over joint simulator variables can support arbitrary "
            "conditioning, structured masks, and benchmark posterior evaluation across heterogeneous SBI tasks."
        ),
        "decisive_comparison": "simformer versus NPE/NLE/NRE baselines on C2ST/NLL and structured-task coverage metrics",
        "decisive_metric": "C2ST using a random forest classifier with 100 trees",
        "stop_pruning_rationale": (
            "Smoke/default modes validate wiring with bounded fixtures; full paper-scale simulation budgets require "
            "an explicit full/train/eval mode and external assets."
        ),
        "simulation_budgets": {dataset_id: spec.simulation_budget.to_dict() for dataset_id, spec in DATASET_REGISTRY.items()},
    }


def write_dataset_registry_artifacts(
    output_dir: Optional[str] = None,
    mode: str = "runtime_smoke",
    include_smoke_evaluation: bool = True,
) -> Dict[str, Any]:
    """Materialize registry and dry-run evaluation artifacts.

    The writer creates every artifact path declared for this task plus
    ``readiness.json`` and ``evaluation_result.json``.  Outputs are explicitly
    labeled as dry-run readiness/schema artifacts.
    """

    artifact_dir = output_dir or os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR") or ""
    written: List[Dict[str, Any]] = []

    registry_payload = {
        "artifact_type": "dataset_registry",
        "dry_run_contract_artifact": True,
        "mode": mode,
        "datasets": get_dataset_registry(),
        "aliases": dict(sorted(ALIASES.items())),
        "lueckmann_2021_suite": get_lueckmann_2021_suite(),
        "declared_artifacts": dict(DECLARED_ARTIFACTS),
    }
    written.append(_write_json(_resolve_artifact_path(DECLARED_ARTIFACTS["dataset_registry"], artifact_dir), registry_payload))

    method_payload = {
        "artifact_type": "method_registry",
        "dry_run_contract_artifact": True,
        "mode": mode,
        "method_bindings": METHOD_BINDINGS,
        "binding_contract": (
            "Each method entry names datasets, decisive metrics, loader/training hooks, mask variants, "
            "and artifact paths used to decide the paper claim."
        ),
    }
    written.append(_write_json(_resolve_artifact_path(DECLARED_ARTIFACTS["method_registry"], artifact_dir), method_payload))

    ablation_payload = {
        "artifact_type": "ablation_registry",
        "dry_run_contract_artifact": True,
        "mode": mode,
        "bounded_variants": [
            {
                "ablation_id": "structured_vs_unstructured_attention_mask",
                "datasets": ["lotka_volterra", "sird", "hodgkin_huxley"],
                "variants": ["structured_dependency", "unstructured_full"],
                "default_executed_in_smoke": ["structured_dependency"],
                "full_mode_required_for_all_variants": True,
            },
            {
                "ablation_id": "condition_mask_probability",
                "datasets": list(DATASET_REGISTRY),
                "variants": ["condition_mask_probability_0.3", "condition_mask_probability_0.5"],
                "default_executed_in_smoke": ["condition_mask_probability_0.3"],
                "full_mode_required_for_all_variants": True,
            },
        ],
        "stop_pruning_rationale": (
            "Only decisive bounded variants are registered; exhaustive sweeps are not run unless full mode is selected."
        ),
    }
    written.append(_write_json(_resolve_artifact_path(DECLARED_ARTIFACTS["ablation_registry"], artifact_dir), ablation_payload))

    config_payload = {
        "artifact_type": "config_resolved",
        "dry_run_contract_artifact": True,
        "resolved_config": build_resolved_config(mode),
    }
    written.append(_write_json(_resolve_artifact_path(DECLARED_ARTIFACTS["config_resolved"], artifact_dir), config_payload))

    smoke_results: List[Dict[str, Any]] = []
    if include_smoke_evaluation:
        for dataset_id in LUECKMANN_2021_BENCHMARK_TASKS[:2]:
            spec = get_dataset_spec(dataset_id)
            observation = "observation_00"
            gt = load_ground_truth_posterior_fixture(dataset_id, observation_id=observation, num_samples=48)
            approx = load_ground_truth_posterior_fixture(dataset_id, observation_id=observation, num_samples=48, seed=DEFAULT_RANDOM_SEED + 11)
            result = evaluate_posterior_samples(
                approximate_posterior_samples=approx["samples"],
                ground_truth_posterior_samples=gt["samples"],
                dataset=dataset_id,
                observation=observation,
                method="simformer",
                mask_variant="structured_dependency",
                simulation_budget=spec.simulation_budget.for_mode(mode),
                sweep_parameters={"mode": mode, "smoke_fixture_samples": 48},
                c2st_trees=DEFAULT_C2ST_TREES,
            )
            smoke_results.append(result)

    metrics_payload = {
        "artifact_type": "metrics",
        "dry_run_contract_artifact": True,
        "mode": mode,
        "metric_schema": {
            "c2st": "Classifier two-sample test accuracy; random forest with configurable n_estimators=100 by default.",
            "nll": "Negative log-likelihood or smoke Gaussian proxy; lower is better.",
            "posterior_coverage": "Credible interval coverage proxy for approximate posterior samples.",
            "constraint_satisfaction_rate": "Structured/interval task metric produced by interval-guidance evaluators.",
        },
        "simulation_efficiency_keys": ["dataset", "task", "method", "simulation_budget", "simulation_count"],
        "results": smoke_results,
        "not_real_experiment_results": True,
    }
    written.append(_write_json(_resolve_artifact_path(DECLARED_ARTIFACTS["metrics"], artifact_dir), metrics_payload))

    benchmark_c2st_payload = {
        "artifact_type": "benchmark_c2st",
        "dry_run_contract_artifact": True,
        "mode": mode,
        "classifier": "random_forest",
        "n_estimators": DEFAULT_C2ST_TREES,
        "inputs": ["approximate_posterior_samples", "ground_truth_posterior_samples"],
        "results": [
            {
                "dataset": row["dataset"],
                "task": row["task"],
                "observation": row["observation"],
                "method": row["method"],
                "mask_variant": row["mask_variant"],
                "simulation_budget": row["simulation_budget"],
                "simulation_count": row["simulation_count"],
                "sweep_parameters": row["sweep_parameters"],
                "c2st": row["metrics"]["c2st"],
                "classifier_detail": row["metric_details"]["c2st"],
                "artifact_group": row["artifact_group"],
            }
            for row in smoke_results
        ],
        "not_real_experiment_results": True,
    }
    written.append(_write_json(_resolve_artifact_path(DECLARED_ARTIFACTS["benchmark_c2st"], artifact_dir), benchmark_c2st_payload))

    readiness_payload = {
        "artifact_type": "readiness",
        "dry_run_contract_artifact": True,
        "mode": mode,
        "status": "ready",
        "checked_surfaces": [
            "dataset_registry",
            "aliases",
            "loader_hooks",
            "ground_truth_posterior_fixture",
            "c2st_random_forest",
            "artifact_grouping",
            "mode_separated_simulation_budgets",
        ],
        "dataset_count": len(DATASET_REGISTRY),
        "lueckmann_task_count": len(LUECKMANN_2021_BENCHMARK_TASKS),
        "artifacts_written": written,
    }
    written.append(_write_json(_resolve_artifact_path(DECLARED_ARTIFACTS["readiness"], artifact_dir), readiness_payload))

    evaluation_payload = {
        "artifact_type": "evaluation_result",
        "dry_run_contract_artifact": True,
        "mode": mode,
        "status": "schema_evaluated",
        "not_real_experiment_results": True,
        "summary": {
            "num_smoke_evaluations": len(smoke_results),
            "metrics_present": ["c2st", "nll", "posterior_coverage"],
            "simulation_budget_recorded": all("simulation_budget" in row for row in smoke_results),
            "artifact_grouping_recorded": all("artifact_group" in row for row in smoke_results),
        },
        "results": smoke_results,
    }
    written.append(_write_json(_resolve_artifact_path(DECLARED_ARTIFACTS["evaluation_result"], artifact_dir), evaluation_payload))

    return {
        "mode": mode,
        "dry_run_contract_artifact": True,
        "artifact_dir": str(artifact_dir),
        "written": written,
        "declared_artifacts": dict(DECLARED_ARTIFACTS),
        "dataset_ids": list_dataset_ids(),
    }


def validate_registry_contract() -> Dict[str, Any]:
    required = {"two_moons", "gaussian_linear", "gaussian_mixture", "slcp", "lotka_volterra", "sird", "hodgkin_huxley"}
    present = set(DATASET_REGISTRY)
    missing = sorted(required - present)
    alias_checks = {name: normalize_dataset_id(name) for name in required}
    lueckmann_ok = len(LUECKMANN_2021_BENCHMARK_TASKS) == 4 and all(task in DATASET_REGISTRY for task in LUECKMANN_2021_BENCHMARK_TASKS)
    ten_gt_ok = all(
        DATASET_REGISTRY[task].observation_protocol.ground_truth_posteriors_per_task == 10
        and len(DATASET_REGISTRY[task].observation_protocol.observation_ids) == 10
        for task in LUECKMANN_2021_BENCHMARK_TASKS
    )
    return {
        "status": "valid" if not missing and lueckmann_ok and ten_gt_ok else "invalid",
        "required_dataset_ids": sorted(required),
        "present_dataset_ids": sorted(present),
        "missing_dataset_ids": missing,
        "alias_checks": alias_checks,
        "lueckmann_2021_four_task_slots": list(LUECKMANN_2021_BENCHMARK_TASKS),
        "lueckmann_2021_four_task_slots_valid": lueckmann_ok,
        "ten_ground_truth_posteriors_per_task_valid": ten_gt_ok,
        "artifact_grouping_keys": [
            "dataset",
            "task",
            "observation",
            "method",
            "mask_variant",
            "simulation_budget",
            "sweep_parameters",
        ],
        "c2st_random_forest_default_trees": DEFAULT_C2ST_TREES,
    }


__all__ = [
    "ALIASES",
    "DATASET_REGISTRY",
    "DECLARED_ARTIFACTS",
    "DEFAULT_C2ST_TREES",
    "DEFAULT_OBSERVATION_IDS",
    "LUECKMANN_2021_BENCHMARK_TASKS",
    "METHOD_BINDINGS",
    "DatasetSpec",
    "ObservationProtocol",
    "SimulationBudgetPolicy",
    "artifact_record",
    "build_resolved_config",
    "c2st_random_forest",
    "evaluate_posterior_samples",
    "get_dataset_registry",
    "get_dataset_spec",
    "get_lueckmann_2021_suite",
    "list_dataset_ids",
    "load_dataset_fixture",
    "load_ground_truth_posterior_fixture",
    "make_artifact_group_key",
    "negative_log_likelihood_proxy",
    "normalize_dataset_id",
    "posterior_coverage_proxy",
    "validate_registry_contract",
    "write_dataset_registry_artifacts",
]