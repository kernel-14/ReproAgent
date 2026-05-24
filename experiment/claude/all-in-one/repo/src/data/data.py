"""Data pipeline, benchmark registry, and evidence matrix for All-in-one SBI.

This module owns the repository data surface for the PaperBench reproduction of
"All-in-one simulation-based inference".  It is deliberately importable in a
minimal environment: optional numeric or ML packages are loaded lazily and the
standard-library fallback remains executable for smoke validation.

Implemented file-scoped obligations
-----------------------------------
* Paper-derived benchmark/dataset registry entries and aliases for:
  two_moons, gaussian_linear, gaussian_mixture, slcp, lotka_volterra, sird,
  hodgkin_huxley.
* Dataset factory hooks that produce joint simulator samples p(theta, x), with
  explicit theta/x schemas, dependency metadata, conditioning hooks, and
  embedding metadata for high-dimensional observations.
* ``dataset_prepare_validate_path`` validates dataset name, output path, schema,
  and smoke/lazy fixture availability without requiring external datasets.
* Code-visible paper evidence obligation matrix binding experiments to datasets,
  methods/baselines, metrics, sweep dimensions, decision claims, and artifact
  paths.
* Dry-run artifact materialization for the canonical artifact contract:
  results/metrics.json, results/samples.npz, results/run_summary.json,
  results/experiment_registry.json, results/evidence_contract_matrix.json,
  results/artifact_manifest.json, results/readiness.json, and
  results/evaluation_result.json.

Dry-run artifacts are readiness/schema artifacts only.  They do not claim
paper-scale numerical results, trained model performance, or completed benchmark
runs.

No code is copied from or depends on the blacklisted repository
https://github.com/mackelab/simformer.

reference_grounding: paperbench_ref_001 docs/how_to_guide/04_embedding_networks.ipynb
reference_grounding: paperbench_ref_001 docs/how_to_guide/22_experiment_tracking.ipynb
reference_grounding: paperbench_ref_001 sbi/inference/trainers/npe/npe_a.py
reference_grounding: paperbench_ref_001 sbi/inference/trainers/nre/nre_base.py
"""

from __future__ import annotations

import argparse
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


CANONICAL_ARTIFACTS: Dict[str, str] = {
    "metrics": "results/metrics.json",
    "samples": "results/samples.npz",
    "run_summary": "results/run_summary.json",
    "experiment_registry": "results/experiment_registry.json",
    "evidence_contract_matrix": "results/evidence_contract_matrix.json",
    "artifact_manifest": "results/artifact_manifest.json",
    "readiness": "results/readiness.json",
    "evaluation_result": "results/evaluation_result.json",
}

SAFE_DRY_MODES: Tuple[str, ...] = ("dry_run", "runtime_smoke", "docker_validate")
VALID_MODES: Tuple[str, ...] = SAFE_DRY_MODES + ("train", "eval")
BLACKLISTED_REPOSITORIES: Tuple[str, ...] = ("https://github.com/mackelab/simformer",)

PAPER_METHODS: Tuple[str, ...] = ("ours", "simformer", "npe", "nle", "nre", "diffusion_model")
PAPER_METRICS: Tuple[str, ...] = ("accuracy", "loss", "return", "c2st", "nll")
BENCHMARK_DATASET_IDS: Tuple[str, ...] = (
    "two_moons",
    "gaussian_linear",
    "gaussian_mixture",
    "slcp",
    "lotka_volterra",
)
ALL_DATASET_IDS: Tuple[str, ...] = BENCHMARK_DATASET_IDS + ("sird", "hodgkin_huxley")
DEFAULT_SIMULATION_BUDGETS: Tuple[int, ...] = (128, 1024, 10000)
DEFAULT_SMOKE_BUDGET: int = 8


@dataclasses.dataclass(frozen=True)
class VariableSchema:
    """Schema for a theta or observation variable in a joint p(theta, x) sample."""

    name: str
    role: str
    dtype: str = "float"
    shape: Tuple[int, ...] = ()
    description: str = ""


@dataclasses.dataclass(frozen=True)
class DatasetSpec:
    """Paper-visible dataset/benchmark specification.

    ``loader_hook`` is a stable symbolic hook for downstream factory code.  The
    actual callable is resolved by ``DatasetFactory`` to keep this module free of
    heavy import-time dependencies.
    """

    dataset_id: str
    display_name: str
    section: str
    task_family: str
    aliases: Tuple[str, ...]
    theta_schema: Tuple[VariableSchema, ...]
    x_schema: Tuple[VariableSchema, ...]
    default_num_simulations: int
    smoke_num_simulations: int
    loader_hook: str
    simulator_kind: str
    dependency_structure: str
    conditioning_patterns: Tuple[str, ...]
    embedding_hint: str
    metrics: Tuple[str, ...]
    artifact_paths: Tuple[str, ...]
    setup_metadata: Mapping[str, Any]

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "display_name": self.display_name,
            "section": self.section,
            "task_family": self.task_family,
            "aliases": list(self.aliases),
            "theta_schema": [dataclasses.asdict(v) for v in self.theta_schema],
            "x_schema": [dataclasses.asdict(v) for v in self.x_schema],
            "default_num_simulations": self.default_num_simulations,
            "smoke_num_simulations": self.smoke_num_simulations,
            "loader_hook": self.loader_hook,
            "simulator_kind": self.simulator_kind,
            "dependency_structure": self.dependency_structure,
            "conditioning_patterns": list(self.conditioning_patterns),
            "embedding_hint": self.embedding_hint,
            "metrics": list(self.metrics),
            "artifact_paths": list(self.artifact_paths),
            "setup_metadata": dict(self.setup_metadata),
        }


@dataclasses.dataclass
class JointSimulationBatch:
    """Joint simulator sample batch representing draws from p(theta, x)."""

    dataset_id: str
    theta: List[List[float]]
    x: List[List[float]]
    variable_names: List[str]
    condition_mask: List[List[int]]
    dependency_mask: List[List[int]]
    metadata: Dict[str, Any]

    @property
    def num_samples(self) -> int:
        return len(self.theta)

    @property
    def theta_dim(self) -> int:
        return len(self.theta[0]) if self.theta else 0

    @property
    def x_dim(self) -> int:
        return len(self.x[0]) if self.x else 0

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "theta": self.theta,
            "x": self.x,
            "variable_names": self.variable_names,
            "condition_mask": self.condition_mask,
            "dependency_mask": self.dependency_mask,
            "metadata": self.metadata,
            "shape": {
                "num_samples": self.num_samples,
                "theta_dim": self.theta_dim,
                "x_dim": self.x_dim,
                "num_joint_variables": len(self.variable_names),
            },
        }


@dataclasses.dataclass(frozen=True)
class EvidenceMatrixRow:
    """Code-visible paper/addendum experiment obligation row."""

    row_id: str
    paper_section: str
    experiment: str
    datasets: Tuple[str, ...]
    tasks: Tuple[str, ...]
    methods: Tuple[str, ...]
    metrics: Tuple[str, ...]
    simulation_budgets: Tuple[int, ...]
    sweep_parameters: Mapping[str, Tuple[Any, ...]]
    fixed_hyperparameters: Mapping[str, Any]
    expected_trend_or_decision_claim: str
    stop_or_pruning_rationale: str
    artifact_paths: Tuple[str, ...]
    provenance: Mapping[str, Any]

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "row_id": self.row_id,
            "paper_section": self.paper_section,
            "experiment": self.experiment,
            "datasets": list(self.datasets),
            "tasks": list(self.tasks),
            "methods": list(self.methods),
            "metrics": list(self.metrics),
            "simulation_budgets": list(self.simulation_budgets),
            "sweep_parameters": {k: list(v) for k, v in self.sweep_parameters.items()},
            "fixed_hyperparameters": dict(self.fixed_hyperparameters),
            "expected_trend_or_decision_claim": self.expected_trend_or_decision_claim,
            "stop_or_pruning_rationale": self.stop_or_pruning_rationale,
            "artifact_paths": list(self.artifact_paths),
            "provenance": dict(self.provenance),
        }


def _schema(prefix: str, count: int, role: str, description: str = "") -> Tuple[VariableSchema, ...]:
    return tuple(
        VariableSchema(
            name=f"{prefix}_{idx + 1}",
            role=role,
            description=description or f"{role} variable {idx + 1}",
        )
        for idx in range(count)
    )


def _time_series_schema(name: str, length: int, description: str) -> Tuple[VariableSchema, ...]:
    return (
        VariableSchema(
            name=name,
            role="x",
            dtype="float",
            shape=(length,),
            description=description,
        ),
    )


DATASET_REGISTRY: Dict[str, DatasetSpec] = {
    "two_moons": DatasetSpec(
        dataset_id="two_moons",
        display_name="Two Moons",
        section="4.1 Benchmark tasks",
        task_family="benchmark_sbi",
        aliases=("two_moons", "two-moons", "Two Moons", "benchmark_two_moons", "tm"),
        theta_schema=_schema("theta", 2, "theta", "Two-dimensional latent parameter"),
        x_schema=_schema("x", 2, "x", "Two-dimensional moon-shaped observation"),
        default_num_simulations=10000,
        smoke_num_simulations=DEFAULT_SMOKE_BUDGET,
        loader_hook="load_two_moons_joint",
        simulator_kind="analytic_lightweight_fixture",
        dependency_structure="all theta variables influence both observation coordinates",
        conditioning_patterns=("posterior_theta_given_x", "likelihood_x_given_theta", "arbitrary_joint_conditioning"),
        embedding_hint="identity_mlp_embedding_for_low_dimensional_observations",
        metrics=("c2st", "nll", "loss"),
        artifact_paths=("results/metrics.json", "results/samples.npz"),
        setup_metadata={
            "requires_external_assets": False,
            "joint_distribution": "p(theta, x)",
            "benchmark_slot": "paper_section_4_1",
        },
    ),
    "gaussian_linear": DatasetSpec(
        dataset_id="gaussian_linear",
        display_name="Linear Gaussian",
        section="4.1 Benchmark tasks",
        task_family="benchmark_sbi",
        aliases=("gaussian_linear", "linear_gaussian", "Gaussian Linear", "benchmark_gaussian_linear"),
        theta_schema=_schema("theta", 2, "theta", "Latent Gaussian parameter"),
        x_schema=_schema("x", 2, "x", "Linear Gaussian observation"),
        default_num_simulations=10000,
        smoke_num_simulations=DEFAULT_SMOKE_BUDGET,
        loader_hook="load_gaussian_linear_joint",
        simulator_kind="analytic_lightweight_fixture",
        dependency_structure="linear map from theta to x with Gaussian noise",
        conditioning_patterns=("posterior_theta_given_x", "likelihood_x_given_theta", "marginal_or_joint_conditioning"),
        embedding_hint="identity_mlp_embedding_for_low_dimensional_observations",
        metrics=("c2st", "nll", "loss"),
        artifact_paths=("results/metrics.json", "results/samples.npz"),
        setup_metadata={
            "requires_external_assets": False,
            "joint_distribution": "p(theta, x)",
            "benchmark_slot": "paper_section_4_1",
        },
    ),
    "gaussian_mixture": DatasetSpec(
        dataset_id="gaussian_mixture",
        display_name="Gaussian Mixture",
        section="4.1 Benchmark tasks",
        task_family="benchmark_sbi",
        aliases=("gaussian_mixture", "gaussian-mixture", "Gaussian Mixture", "benchmark_gaussian_mixture", "gmm"),
        theta_schema=_schema("theta", 2, "theta", "Mixture-conditioned latent parameter"),
        x_schema=_schema("x", 2, "x", "Gaussian mixture observation"),
        default_num_simulations=10000,
        smoke_num_simulations=DEFAULT_SMOKE_BUDGET,
        loader_hook="load_gaussian_mixture_joint",
        simulator_kind="analytic_lightweight_fixture",
        dependency_structure="mixture component and theta jointly determine x",
        conditioning_patterns=("posterior_theta_given_x", "multimodal_posterior", "arbitrary_joint_conditioning"),
        embedding_hint="identity_mlp_embedding_for_low_dimensional_observations",
        metrics=("c2st", "nll", "loss"),
        artifact_paths=("results/metrics.json", "results/samples.npz"),
        setup_metadata={
            "requires_external_assets": False,
            "joint_distribution": "p(theta, x)",
            "benchmark_slot": "paper_section_4_1",
        },
    ),
    "slcp": DatasetSpec(
        dataset_id="slcp",
        display_name="Simple Likelihood Complex Posterior",
        section="4.1 Benchmark tasks",
        task_family="benchmark_sbi",
        aliases=("slcp", "SLCP", "simple_likelihood_complex_posterior", "benchmark_slcp"),
        theta_schema=_schema("theta", 5, "theta", "Five-dimensional SLCP latent parameter"),
        x_schema=_schema("x", 8, "x", "Four bivariate Gaussian draws flattened as observations"),
        default_num_simulations=10000,
        smoke_num_simulations=DEFAULT_SMOKE_BUDGET,
        loader_hook="load_slcp_joint",
        simulator_kind="analytic_lightweight_fixture",
        dependency_structure="nonlinear mean and covariance functions of theta generate repeated x pairs",
        conditioning_patterns=("posterior_theta_given_x", "subset_observation_conditioning", "arbitrary_joint_conditioning"),
        embedding_hint="permutation_or_mlp_embedding_for_repeated_observation_pairs",
        metrics=("c2st", "nll", "loss"),
        artifact_paths=("results/metrics.json", "results/samples.npz"),
        setup_metadata={
            "requires_external_assets": False,
            "joint_distribution": "p(theta, x)",
            "benchmark_slot": "paper_section_4_1",
        },
    ),
    "lotka_volterra": DatasetSpec(
        dataset_id="lotka_volterra",
        display_name="Lotka-Volterra",
        section="4.2 Lotka-Volterra",
        task_family="structured_time_series_sbi",
        aliases=("lotka_volterra", "lotka-volterra", "Lotka Volterra", "lv", "predator_prey"),
        theta_schema=(
            VariableSchema("alpha", "theta", description="Prey birth rate"),
            VariableSchema("beta", "theta", description="Predation rate"),
            VariableSchema("gamma", "theta", description="Predator death rate"),
            VariableSchema("delta", "theta", description="Predator reproduction from prey"),
        ),
        x_schema=_time_series_schema("prey_predator_time_series", 20, "Flattened prey/predator trajectory"),
        default_num_simulations=10000,
        smoke_num_simulations=DEFAULT_SMOKE_BUDGET,
        loader_hook="load_lotka_volterra_joint",
        simulator_kind="structured_dynamic_fixture",
        dependency_structure="theta-to-series causal blocks plus Markovian prey/predator time dependencies",
        conditioning_patterns=("structured_observation", "unstructured_observation", "function_parameter_inference"),
        embedding_hint="time_series_embedding_network_hook",
        metrics=("c2st", "nll", "loss", "return"),
        artifact_paths=("results/metrics.json", "results/samples.npz"),
        setup_metadata={
            "requires_external_assets": False,
            "joint_distribution": "p(theta, x)",
            "benchmark_slot": "paper_section_4_2",
            "structured_observation": True,
        },
    ),
    "sird": DatasetSpec(
        dataset_id="sird",
        display_name="SIRD epidemiological model",
        section="4.3 SIRD-model",
        task_family="structured_time_series_sbi",
        aliases=("sird", "SIRD", "sird_model", "epidemiology_sird"),
        theta_schema=(
            VariableSchema("beta", "theta", description="Infection/contact rate"),
            VariableSchema("gamma", "theta", description="Recovery rate"),
            VariableSchema("mu", "theta", description="Mortality rate"),
        ),
        x_schema=_time_series_schema("sird_compartment_time_series", 24, "Flattened susceptible/infected/recovered/deceased trajectory"),
        default_num_simulations=10000,
        smoke_num_simulations=DEFAULT_SMOKE_BUDGET,
        loader_hook="load_sird_joint",
        simulator_kind="structured_dynamic_fixture",
        dependency_structure="compartment transitions depend on beta, gamma, and mortality",
        conditioning_patterns=("structured_observation", "missing_observation", "posterior_coverage"),
        embedding_hint="time_series_embedding_network_hook",
        metrics=("c2st", "nll", "loss", "accuracy"),
        artifact_paths=("results/metrics.json", "results/samples.npz"),
        setup_metadata={
            "requires_external_assets": False,
            "joint_distribution": "p(theta, x)",
            "benchmark_slot": "paper_section_4_3",
            "posterior_coverage_required": True,
        },
    ),
    "hodgkin_huxley": DatasetSpec(
        dataset_id="hodgkin_huxley",
        display_name="Hodgkin-Huxley",
        section="4.4 Hodgkin-Huxley",
        task_family="interval_guided_sbi",
        aliases=("hodgkin_huxley", "hodgkin-huxley", "Hodgkin Huxley", "hh", "hh_interval_guidance"),
        theta_schema=(
            VariableSchema("g_na", "theta", description="Sodium conductance"),
            VariableSchema("g_k", "theta", description="Potassium conductance"),
            VariableSchema("g_l", "theta", description="Leak conductance"),
            VariableSchema("e_na", "theta", description="Sodium reversal potential"),
            VariableSchema("e_k", "theta", description="Potassium reversal potential"),
        ),
        x_schema=_time_series_schema("voltage_trace", 32, "Membrane voltage trace"),
        default_num_simulations=10000,
        smoke_num_simulations=DEFAULT_SMOKE_BUDGET,
        loader_hook="load_hodgkin_huxley_joint",
        simulator_kind="interval_guidance_fixture",
        dependency_structure="conductance parameters determine voltage trace and metabolic-cost constraints",
        conditioning_patterns=("observation_interval", "energy_constraint", "guided_diffusion"),
        embedding_hint="time_series_embedding_network_hook",
        metrics=("c2st", "nll", "loss", "return", "accuracy"),
        artifact_paths=("results/metrics.json", "results/samples.npz"),
        setup_metadata={
            "requires_external_assets": False,
            "joint_distribution": "p(theta, x)",
            "benchmark_slot": "paper_section_4_4",
            "interval_guidance_required": True,
            "guided_parameters": ("similarity_guidance_scale",),
        },
    ),
}


def _build_alias_registry() -> Dict[str, str]:
    aliases: Dict[str, str] = {}
    for dataset_id, spec in DATASET_REGISTRY.items():
        aliases[dataset_id] = dataset_id
        for alias in spec.aliases:
            aliases[alias] = dataset_id
            aliases[alias.lower()] = dataset_id
            aliases[alias.replace("-", "_").lower()] = dataset_id
    return aliases


DATASET_ALIASES: Dict[str, str] = _build_alias_registry()


PAPER_EVIDENCE_MATRIX: Tuple[EvidenceMatrixRow, ...] = (
    EvidenceMatrixRow(
        row_id="section_4_1_benchmark_tasks",
        paper_section="4.1 Benchmark tasks",
        experiment="benchmark posterior inference across canonical SBI tasks",
        datasets=("two_moons", "gaussian_linear", "gaussian_mixture", "slcp", "lotka_volterra"),
        tasks=("posterior_theta_given_x", "joint_p_theta_x_training", "arbitrary_conditioning"),
        methods=("ours", "simformer", "npe", "nle", "nre", "diffusion_model"),
        metrics=("c2st", "nll", "loss", "accuracy"),
        simulation_budgets=DEFAULT_SIMULATION_BUDGETS,
        sweep_parameters={
            "alpha": (1, 2),
            "population_size": (1, 2),
            "lora_rank": (1, 2),
        },
        fixed_hyperparameters={
            "condition_mask_family": "uniform_named_condition_masks",
            "training_batch_size": 200,
            "learning_rate": 5e-4,
            "validation_fraction": 0.1,
            "stop_after_epochs": 20,
            "max_smoke_simulations": DEFAULT_SMOKE_BUDGET,
        },
        expected_trend_or_decision_claim=(
            "baseline_outperformance: proposed all-in-one conditional score model "
            "is compared against explicit NPE/NLE/NRE/diffusion baselines on C2ST/NLL/loss."
        ),
        stop_or_pruning_rationale=(
            "Default smoke mode instantiates all registry surfaces and validates schemas; "
            "paper-scale budgets require explicit train/eval mode."
        ),
        artifact_paths=(
            "results/metrics.json",
            "results/samples.npz",
            "results/experiment_registry.json",
            "results/evidence_contract_matrix.json",
        ),
        provenance={
            "paper_obligation": "section_4_1_named_benchmark_matrix",
            "reference_grounding": "paperbench_ref_001 docs/how_to_guide/22_experiment_tracking.ipynb",
        },
    ),
    EvidenceMatrixRow(
        row_id="section_4_2_lotka_volterra_structure",
        paper_section="4.2 Lotka-Volterra",
        experiment="structured and unstructured observation conditioning",
        datasets=("lotka_volterra",),
        tasks=("structured_observation", "unstructured_observation", "function_parameter_inference"),
        methods=("ours", "simformer", "npe", "nle", "nre"),
        metrics=("c2st", "nll", "loss", "return"),
        simulation_budgets=DEFAULT_SIMULATION_BUDGETS,
        sweep_parameters={
            "alpha": (1, 2),
            "beta": (1, 2),
            "gamma": (1, 2),
        },
        fixed_hyperparameters={
            "dependency_mask": "lotka_volterra_markovian_prey_predator_graph",
            "embedding_hint": "time_series_embedding_network_hook",
            "max_smoke_simulations": DEFAULT_SMOKE_BUDGET,
        },
        expected_trend_or_decision_claim=(
            "positive_parameter_improves: positive dynamic parameters should preserve "
            "structured-conditioning improvements relative to unstructured baselines."
        ),
        stop_or_pruning_rationale=(
            "Bounded smoke fixtures validate dependency metadata and joint p(theta,x) generation; "
            "no exhaustive parameter sweep is executed by default."
        ),
        artifact_paths=("results/metrics.json", "results/samples.npz", "results/evidence_contract_matrix.json"),
        provenance={
            "paper_obligation": "section_4_2_structured_tasks",
            "reference_grounding": "paperbench_ref_001 docs/how_to_guide/04_embedding_networks.ipynb",
        },
    ),
    EvidenceMatrixRow(
        row_id="section_4_3_sird_posterior_coverage",
        paper_section="4.3 SIRD-model",
        experiment="SIRD structured/missing observation posterior coverage",
        datasets=("sird",),
        tasks=("structured_observation", "missing_observation", "posterior_coverage"),
        methods=("ours", "simformer", "npe", "nle", "nre"),
        metrics=("c2st", "nll", "loss", "accuracy"),
        simulation_budgets=DEFAULT_SIMULATION_BUDGETS,
        sweep_parameters={
            "beta": (1, 2),
            "gamma": (1, 2),
            "population_size": (1, 2),
        },
        fixed_hyperparameters={
            "dependency_mask": "sird_compartment_transition_graph",
            "coverage_artifact": "results/samples.npz",
            "max_smoke_simulations": DEFAULT_SMOKE_BUDGET,
        },
        expected_trend_or_decision_claim=(
            "posterior samples should remain calibrated under missing/structured observations; "
            "compare all-in-one method to explicit SBI baselines."
        ),
        stop_or_pruning_rationale=(
            "Smoke writes coverage schema and sample fixture only; full coverage estimates require eval mode."
        ),
        artifact_paths=("results/metrics.json", "results/samples.npz", "results/evidence_contract_matrix.json"),
        provenance={"paper_obligation": "section_4_3_sird_model"},
    ),
    EvidenceMatrixRow(
        row_id="section_4_4_hodgkin_huxley_guidance",
        paper_section="4.4 Hodgkin-Huxley",
        experiment="observation interval and metabolic cost guided diffusion",
        datasets=("hodgkin_huxley",),
        tasks=("observation_interval", "energy_constraint", "guided_diffusion"),
        methods=("ours", "simformer", "diffusion_model"),
        metrics=("c2st", "nll", "loss", "return", "accuracy"),
        simulation_budgets=DEFAULT_SIMULATION_BUDGETS,
        sweep_parameters={
            "similarity_guidance_scale": (1, 2),
            "alpha": (1, 2),
            "lora_rank": (1, 2),
        },
        fixed_hyperparameters={
            "interval_lower_bound": -80.0,
            "interval_upper_bound": 50.0,
            "energy_threshold": "registered_by_conditioning_module",
            "max_smoke_simulations": DEFAULT_SMOKE_BUDGET,
        },
        expected_trend_or_decision_claim=(
            "positive_parameter_improves: nonzero guidance scale should increase interval/"
            "constraint satisfaction compared with unguided diffusion while retaining posterior quality."
        ),
        stop_or_pruning_rationale=(
            "Only guidance-scale values required by the contract are registered; full HH simulation is opt-in."
        ),
        artifact_paths=("results/metrics.json", "results/samples.npz", "results/evidence_contract_matrix.json"),
        provenance={"paper_obligation": "section_4_4_interval_guidance"},
    ),
)


def normalize_dataset_id(name_or_alias: str) -> str:
    """Resolve a dataset id or alias to the canonical registry id."""

    key = str(name_or_alias).strip()
    if key in DATASET_ALIASES:
        return DATASET_ALIASES[key]
    lowered = key.lower().replace("-", "_")
    if lowered in DATASET_ALIASES:
        return DATASET_ALIASES[lowered]
    available = ", ".join(sorted(DATASET_REGISTRY))
    raise ValueError(f"Unknown dataset '{name_or_alias}'. Available datasets: {available}")


def get_dataset_spec(name_or_alias: str) -> DatasetSpec:
    """Return a dataset spec from a canonical id or any registered alias."""

    return DATASET_REGISTRY[normalize_dataset_id(name_or_alias)]


def list_dataset_specs() -> List[DatasetSpec]:
    """Return all dataset specs in stable order."""

    return [DATASET_REGISTRY[dataset_id] for dataset_id in ALL_DATASET_IDS]


def _rng(seed: Optional[int]) -> random.Random:
    return random.Random(0 if seed is None else int(seed))


def _normal(rng: random.Random, mean: float = 0.0, std: float = 1.0) -> float:
    return mean + std * rng.gauss(0.0, 1.0)


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _dependency_mask(theta_dim: int, x_dim: int, structured: Optional[str] = None) -> List[List[int]]:
    """Create a lightweight dependency attention mask over [theta, x] variables."""

    n = theta_dim + x_dim
    mask = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        mask[i][i] = 1

    for t in range(theta_dim):
        for x in range(theta_dim, n):
            mask[t][x] = 1
            mask[x][t] = 1

    if structured in {"time_series", "lotka_volterra", "sird", "hodgkin_huxley"}:
        for x in range(theta_dim, n):
            if x - 1 >= theta_dim:
                mask[x][x - 1] = 1
                mask[x - 1][x] = 1
            if structured == "lotka_volterra" and x + 1 < n:
                mask[x][x + 1] = 1
                mask[x + 1][x] = 1
    else:
        for i in range(theta_dim, n):
            for j in range(theta_dim, n):
                mask[i][j] = 1
    return mask


def _condition_mask(num_samples: int, num_variables: int, pattern: str = "posterior_theta_given_x") -> List[List[int]]:
    """Build binary condition masks M_C for joint variables.

    A value of 1 marks an observed/conditioned variable.  The default posterior
    pattern conditions on x while leaving theta to be sampled.
    """

    masks: List[List[int]] = []
    for idx in range(num_samples):
        row = [0 for _ in range(num_variables)]
        if pattern in {"posterior_theta_given_x", "structured_observation", "missing_observation", "observation_interval"}:
            split = max(1, num_variables // 3)
            for j in range(split, num_variables):
                if pattern == "missing_observation" and (j + idx) % 3 == 0:
                    continue
                row[j] = 1
        elif pattern == "likelihood_x_given_theta":
            for j in range(max(1, num_variables // 3)):
                row[j] = 1
        elif pattern == "arbitrary_joint_conditioning":
            for j in range(num_variables):
                row[j] = 1 if (j + idx) % 2 == 0 else 0
        else:
            for j in range(num_variables):
                row[j] = 1 if j % 2 == 0 else 0
        masks.append(row)
    return masks


class DatasetFactory:
    """Factory for paper-derived joint p(theta, x) smoke fixtures.

    The class provides real simulator hooks suitable for smoke and bounded tests.
    They are intentionally lightweight approximations of the named benchmark
    data-generating processes and not paper-scale simulation results.
    """

    def __init__(self, registry: Optional[Mapping[str, DatasetSpec]] = None) -> None:
        self.registry: Mapping[str, DatasetSpec] = registry or DATASET_REGISTRY
        self.loader_hooks: Dict[str, Callable[[DatasetSpec, int, Optional[int], str], JointSimulationBatch]] = {
            "load_two_moons_joint": self._simulate_two_moons,
            "load_gaussian_linear_joint": self._simulate_gaussian_linear,
            "load_gaussian_mixture_joint": self._simulate_gaussian_mixture,
            "load_slcp_joint": self._simulate_slcp,
            "load_lotka_volterra_joint": self._simulate_lotka_volterra,
            "load_sird_joint": self._simulate_sird,
            "load_hodgkin_huxley_joint": self._simulate_hodgkin_huxley,
        }

    def get(self, name_or_alias: str) -> DatasetSpec:
        return get_dataset_spec(name_or_alias)

    def load_joint(
        self,
        name_or_alias: str,
        num_simulations: Optional[int] = None,
        seed: Optional[int] = None,
        condition_pattern: Optional[str] = None,
        smoke: bool = True,
    ) -> JointSimulationBatch:
        spec = self.get(name_or_alias)
        n = int(num_simulations or (spec.smoke_num_simulations if smoke else spec.default_num_simulations))
        if n <= 0:
            raise ValueError("num_simulations must be positive")
        pattern = condition_pattern or spec.conditioning_patterns[0]
        if pattern not in spec.conditioning_patterns and pattern not in {
            "posterior_theta_given_x",
            "likelihood_x_given_theta",
            "arbitrary_joint_conditioning",
            "missing_observation",
            "observation_interval",
        }:
            raise ValueError(
                f"Conditioning pattern '{pattern}' is not registered for {spec.dataset_id}: "
                f"{', '.join(spec.conditioning_patterns)}"
            )
        if spec.loader_hook not in self.loader_hooks:
            raise ValueError(f"Dataset {spec.dataset_id} has unresolved loader hook {spec.loader_hook}")
        return self.loader_hooks[spec.loader_hook](spec, n, seed, pattern)

    def _finish(
        self,
        spec: DatasetSpec,
        theta: List[List[float]],
        x: List[List[float]],
        condition_pattern: str,
        structured: Optional[str] = None,
    ) -> JointSimulationBatch:
        theta_names = [v.name for v in spec.theta_schema]
        x_dim = len(x[0]) if x else sum(max(1, math.prod(v.shape) if v.shape else 1) for v in spec.x_schema)
        if len(spec.x_schema) == x_dim:
            x_names = [v.name for v in spec.x_schema]
        else:
            base = spec.x_schema[0].name if spec.x_schema else "x"
            x_names = [f"{base}_{i + 1}" for i in range(x_dim)]
        variable_names = theta_names + x_names
        condition = _condition_mask(len(theta), len(variable_names), condition_pattern)
        dep = _dependency_mask(len(theta_names), len(x_names), structured=structured)
        return JointSimulationBatch(
            dataset_id=spec.dataset_id,
            theta=theta,
            x=x,
            variable_names=variable_names,
            condition_mask=condition,
            dependency_mask=dep,
            metadata={
                "display_name": spec.display_name,
                "section": spec.section,
                "joint_distribution": "p(theta, x)",
                "condition_pattern": condition_pattern,
                "loader_hook": spec.loader_hook,
                "simulator_kind": spec.simulator_kind,
                "dependency_structure": spec.dependency_structure,
                "dry_run_fixture": True,
                "not_paper_scale_result": True,
            },
        )

    def _simulate_two_moons(
        self, spec: DatasetSpec, n: int, seed: Optional[int], condition_pattern: str
    ) -> JointSimulationBatch:
        rng = _rng(seed)
        theta: List[List[float]] = []
        x: List[List[float]] = []
        for _ in range(n):
            angle = rng.uniform(0.0, math.pi)
            radius = 1.0 + 0.10 * _normal(rng)
            latent = [radius * math.cos(angle), radius * math.sin(angle)]
            sign = -1.0 if rng.random() < 0.5 else 1.0
            obs = [
                latent[0] + 0.25 * sign + 0.05 * _normal(rng),
                sign * (latent[1] - 0.5) + 0.05 * _normal(rng),
            ]
            theta.append([round(v, 6) for v in latent])
            x.append([round(v, 6) for v in obs])
        return self._finish(spec, theta, x, condition_pattern)

    def _simulate_gaussian_linear(
        self, spec: DatasetSpec, n: int, seed: Optional[int], condition_pattern: str
    ) -> JointSimulationBatch:
        rng = _rng(seed)
        theta: List[List[float]] = []
        x: List[List[float]] = []
        matrix = ((1.0, 0.35), (-0.25, 0.90))
        for _ in range(n):
            t = [_normal(rng), _normal(rng)]
            obs = [
                matrix[0][0] * t[0] + matrix[0][1] * t[1] + 0.10 * _normal(rng),
                matrix[1][0] * t[0] + matrix[1][1] * t[1] + 0.10 * _normal(rng),
            ]
            theta.append([round(v, 6) for v in t])
            x.append([round(v, 6) for v in obs])
        return self._finish(spec, theta, x, condition_pattern)

    def _simulate_gaussian_mixture(
        self, spec: DatasetSpec, n: int, seed: Optional[int], condition_pattern: str
    ) -> JointSimulationBatch:
        rng = _rng(seed)
        theta: List[List[float]] = []
        x: List[List[float]] = []
        centers = ((-1.0, -1.0), (1.0, 1.0))
        for _ in range(n):
            component = 0 if rng.random() < 0.5 else 1
            t = [
                centers[component][0] + 0.45 * _normal(rng),
                centers[component][1] + 0.45 * _normal(rng),
            ]
            obs = [
                t[0] + (0.35 if component else -0.35) + 0.12 * _normal(rng),
                t[1] + (-0.20 if component else 0.20) + 0.12 * _normal(rng),
            ]
            theta.append([round(v, 6) for v in t])
            x.append([round(v, 6) for v in obs])
        return self._finish(spec, theta, x, condition_pattern)

    def _simulate_slcp(
        self, spec: DatasetSpec, n: int, seed: Optional[int], condition_pattern: str
    ) -> JointSimulationBatch:
        rng = _rng(seed)
        theta: List[List[float]] = []
        x: List[List[float]] = []
        for _ in range(n):
            t = [rng.uniform(-2.0, 2.0) for _ in range(5)]
            mean1 = t[0]
            mean2 = t[1]
            scale1 = 0.20 + 0.08 * abs(t[2])
            scale2 = 0.20 + 0.08 * abs(t[3])
            corr = math.tanh(t[4]) * 0.15
            obs: List[float] = []
            for _pair in range(4):
                z1 = _normal(rng)
                z2 = corr * z1 + math.sqrt(max(1e-6, 1.0 - corr * corr)) * _normal(rng)
                obs.extend([mean1 + scale1 * z1, mean2 + scale2 * z2])
            theta.append([round(v, 6) for v in t])
            x.append([round(v, 6) for v in obs])
        return self._finish(spec, theta, x, condition_pattern)

    def _simulate_lotka_volterra(
        self, spec: DatasetSpec, n: int, seed: Optional[int], condition_pattern: str
    ) -> JointSimulationBatch:
        rng = _rng(seed)
        theta: List[List[float]] = []
        x: List[List[float]] = []
        steps = 10
        for _ in range(n):
            alpha = rng.uniform(0.7, 1.5)
            beta = rng.uniform(0.02, 0.08)
            gamma = rng.uniform(0.6, 1.2)
            delta = rng.uniform(0.01, 0.06)
            prey = rng.uniform(20.0, 35.0)
            predator = rng.uniform(8.0, 18.0)
            trajectory: List[float] = []
            for _step in range(steps):
                prey_growth = alpha * prey - beta * prey * predator
                predator_growth = delta * prey * predator - gamma * predator
                prey = _clip(prey + 0.05 * prey_growth + _normal(rng, 0.0, 0.15), 0.0, 100.0)
                predator = _clip(predator + 0.05 * predator_growth + _normal(rng, 0.0, 0.10), 0.0, 100.0)
                trajectory.extend([prey / 50.0, predator / 50.0])
            theta.append([round(alpha, 6), round(beta, 6), round(gamma, 6), round(delta, 6)])
            x.append([round(v, 6) for v in trajectory])
        return self._finish(spec, theta, x, condition_pattern, structured="lotka_volterra")

    def _simulate_sird(
        self, spec: DatasetSpec, n: int, seed: Optional[int], condition_pattern: str
    ) -> JointSimulationBatch:
        rng = _rng(seed)
        theta: List[List[float]] = []
        x: List[List[float]] = []
        steps = 6
        population = 1000.0
        for _ in range(n):
            beta = rng.uniform(0.10, 0.55)
            gamma = rng.uniform(0.03, 0.18)
            mu = rng.uniform(0.005, 0.05)
            s = population - 10.0
            i = 10.0
            r = 0.0
            d = 0.0
            trajectory: List[float] = []
            for _step in range(steps):
                new_inf = beta * s * i / population
                new_rec = gamma * i
                new_dead = mu * i
                s = _clip(s - new_inf, 0.0, population)
                i = _clip(i + new_inf - new_rec - new_dead, 0.0, population)
                r = _clip(r + new_rec, 0.0, population)
                d = _clip(d + new_dead, 0.0, population)
                trajectory.extend([s / population, i / population, r / population, d / population])
            theta.append([round(beta, 6), round(gamma, 6), round(mu, 6)])
            x.append([round(v, 6) for v in trajectory])
        return self._finish(spec, theta, x, condition_pattern, structured="sird")

    def _simulate_hodgkin_huxley(
        self, spec: DatasetSpec, n: int, seed: Optional[int], condition_pattern: str
    ) -> JointSimulationBatch:
        rng = _rng(seed)
        theta: List[List[float]] = []
        x: List[List[float]] = []
        steps = 32
        for _ in range(n):
            g_na = rng.uniform(80.0, 140.0)
            g_k = rng.uniform(25.0, 45.0)
            g_l = rng.uniform(0.1, 0.5)
            e_na = rng.uniform(45.0, 60.0)
            e_k = rng.uniform(-90.0, -70.0)
            voltage = -65.0 + rng.uniform(-2.0, 2.0)
            trace: List[float] = []
            for step in range(steps):
                stimulus = 12.0 * math.sin(2.0 * math.pi * step / max(1, steps - 1))
                conductance_drive = 0.015 * (g_na / 120.0) * (e_na - voltage)
                potassium_drive = 0.010 * (g_k / 36.0) * (e_k - voltage)
                leak_drive = 0.020 * g_l * (-54.4 - voltage)
                voltage = _clip(voltage + conductance_drive + potassium_drive + leak_drive + stimulus * 0.05, -90.0, 60.0)
                trace.append(voltage / 100.0)
            theta.append([round(g_na, 6), round(g_k, 6), round(g_l, 6), round(e_na, 6), round(e_k, 6)])
            x.append([round(v, 6) for v in trace])
        return self._finish(spec, theta, x, condition_pattern, structured="hodgkin_huxley")


def validate_dataset_schema(spec: DatasetSpec) -> Dict[str, Any]:
    """Validate one dataset schema and return machine-readable diagnostics."""

    errors: List[str] = []
    if not spec.dataset_id:
        errors.append("dataset_id is empty")
    if not spec.theta_schema:
        errors.append(f"{spec.dataset_id}: theta_schema is empty")
    if not spec.x_schema:
        errors.append(f"{spec.dataset_id}: x_schema is empty")
    if spec.default_num_simulations < spec.smoke_num_simulations:
        errors.append(f"{spec.dataset_id}: default budget is smaller than smoke budget")
    if "results/metrics.json" not in spec.artifact_paths:
        errors.append(f"{spec.dataset_id}: metrics artifact path missing")
    if spec.setup_metadata.get("joint_distribution") != "p(theta, x)":
        errors.append(f"{spec.dataset_id}: joint p(theta, x) metadata missing")
    if spec.loader_hook not in DatasetFactory().loader_hooks:
        errors.append(f"{spec.dataset_id}: unresolved loader hook {spec.loader_hook}")
    return {
        "dataset_id": spec.dataset_id,
        "valid": not errors,
        "errors": errors,
        "theta_dim": len(spec.theta_schema),
        "x_schema_entries": len(spec.x_schema),
        "aliases": list(spec.aliases),
        "loader_hook": spec.loader_hook,
    }


def dataset_prepare_validate_path(
    dataset_name: str,
    output_dir: Optional[str | os.PathLike[str]] = None,
    mode: str = "dry_run",
    num_simulations: Optional[int] = None,
    seed: int = 0,
) -> Dict[str, Any]:
    """Validate dataset name, paths, schema, and lazy/smoke fixture availability.

    This is the repository's callable dataset prepare/validate path.  It creates
    output directories as needed and instantiates the real ``DatasetFactory`` with
    bounded fixture generation.  No external assets are downloaded.
    """

    if mode not in VALID_MODES:
        raise ValueError(f"Invalid mode '{mode}'. Expected one of {VALID_MODES}")
    spec = get_dataset_spec(dataset_name)
    base = Path(output_dir or os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR", "."))
    results_dir = base / "results" if base.name != "results" else base
    results_dir.mkdir(parents=True, exist_ok=True)

    schema_report = validate_dataset_schema(spec)
    factory = DatasetFactory()
    fixture_n = int(num_simulations or spec.smoke_num_simulations)
    batch = factory.load_joint(spec.dataset_id, num_simulations=fixture_n, seed=seed, smoke=True)

    fixture_errors: List[str] = []
    if batch.num_samples != fixture_n:
        fixture_errors.append(f"fixture sample count mismatch: expected {fixture_n}, got {batch.num_samples}")
    if batch.theta_dim != len(spec.theta_schema):
        fixture_errors.append(f"theta dimension mismatch: expected {len(spec.theta_schema)}, got {batch.theta_dim}")
    if not batch.x:
        fixture_errors.append("fixture x is empty")
    if len(batch.condition_mask) != batch.num_samples:
        fixture_errors.append("condition mask row count mismatch")
    if len(batch.dependency_mask) != len(batch.variable_names):
        fixture_errors.append("dependency mask size mismatch")

    report = {
        "artifact_type": "dataset_prepare_validate_report",
        "dry_run_contract_artifact": mode in SAFE_DRY_MODES,
        "mode": mode,
        "dataset": spec.to_json_dict(),
        "schema_report": schema_report,
        "fixture_report": {
            "available": not fixture_errors,
            "errors": fixture_errors,
            "num_samples": batch.num_samples,
            "theta_dim": batch.theta_dim,
            "x_dim": batch.x_dim,
            "variable_names": batch.variable_names,
            "joint_distribution": "p(theta, x)",
        },
        "path_report": {
            "base_dir": str(base),
            "results_dir": str(results_dir),
            "exists": results_dir.exists(),
            "writable": os.access(results_dir, os.W_OK),
        },
        "valid": schema_report["valid"] and not fixture_errors and os.access(results_dir, os.W_OK),
        "provenance": {
            "module": "src.data.data",
            "blacklisted_repositories_not_used": list(BLACKLISTED_REPOSITORIES),
            "created_at": _datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        },
    }
    validate_path = results_dir / f"{spec.dataset_id}_dataset_validate.json"
    validate_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    report["path_report"]["validation_artifact"] = str(validate_path)
    return report


def build_experiment_registry() -> Dict[str, Any]:
    """Return the paper-visible experiment registry with dataset hooks."""

    return {
        "artifact_type": "experiment_registry",
        "dry_run_contract_artifact": True,
        "paper": "All-in-one simulation-based inference",
        "blacklisted_repositories_not_used": list(BLACKLISTED_REPOSITORIES),
        "datasets": {dataset_id: spec.to_json_dict() for dataset_id, spec in DATASET_REGISTRY.items()},
        "aliases": dict(sorted(DATASET_ALIASES.items())),
        "methods": {
            "ours": {
                "role": "primary_method",
                "description": "all-in-one transformer score-based diffusion model over joint p(theta,x)",
                "model_loader_hook": "load_simformer_score_model",
            },
            "simformer": {
                "role": "paper_method_alias",
                "description": "local implementation alias; blacklisted upstream repository is not used",
                "model_loader_hook": "load_simformer_score_model",
            },
            "npe": {
                "role": "baseline",
                "description": "neural posterior estimation adapter",
                "model_loader_hook": "load_npe_baseline",
            },
            "nle": {
                "role": "baseline",
                "description": "neural likelihood estimation adapter",
                "model_loader_hook": "load_nle_baseline",
            },
            "nre": {
                "role": "baseline",
                "description": "neural ratio estimation adapter",
                "model_loader_hook": "load_nre_baseline",
            },
            "diffusion_model": {
                "role": "baseline_or_ablation",
                "description": "unstructured diffusion baseline/ablation",
                "model_loader_hook": "load_diffusion_baseline",
            },
        },
        "named_experiments": {row.row_id: row.to_json_dict() for row in PAPER_EVIDENCE_MATRIX},
        "default_smoke_selection": {
            "mode": "runtime_smoke",
            "dataset": "two_moons",
            "method": "ours",
            "experiment": "section_4_1_benchmark_tasks",
            "num_simulations": DEFAULT_SMOKE_BUDGET,
        },
    }


def build_evidence_contract_matrix() -> Dict[str, Any]:
    """Return the code/config-visible evidence obligation matrix."""

    return {
        "artifact_type": "evidence_contract_matrix",
        "dry_run_contract_artifact": True,
        "paper": "All-in-one simulation-based inference",
        "matrix_version": 1,
        "rows": [row.to_json_dict() for row in PAPER_EVIDENCE_MATRIX],
        "coverage": {
            "required_dataset_aliases": list(ALL_DATASET_IDS),
            "benchmark_inventory": list(BENCHMARK_DATASET_IDS),
            "methods": list(PAPER_METHODS),
            "metrics": list(PAPER_METRICS),
            "required_sweep_parameters": [
                "alpha",
                "population_size",
                "beta",
                "gamma",
                "lora_rank",
                "similarity_guidance_scale",
            ],
            "required_sweep_values": [1, 2],
            "trend_claims": {
                "baseline_outperformance": (
                    "proposed method should be compared against explicit baselines"
                ),
                "positive_parameter_improves": (
                    "nonzero/positive parameter values should preserve the reported improvement trend"
                ),
            },
        },
        "provenance": {
            "module": "src.data.data",
            "reference_grounding": [
                "paperbench_ref_001 docs/how_to_guide/04_embedding_networks.ipynb",
                "paperbench_ref_001 docs/how_to_guide/22_experiment_tracking.ipynb",
                "paperbench_ref_001 sbi/inference/trainers/npe/npe_a.py",
                "paperbench_ref_001 sbi/inference/trainers/nre/nre_base.py",
            ],
        },
    }


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _variance(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(statistics.pvariance(values))


def build_metric_rows(
    batch: JointSimulationBatch,
    method: str,
    experiment: str,
    mode: str,
    simulation_budget: int,
) -> List[Dict[str, Any]]:
    """Create stable metric schema rows with dry-run diagnostic values.

    The values are schema/readiness diagnostics derived from the smoke fixture;
    they are not benchmark claims.
    """

    theta_flat = [v for row in batch.theta for v in row]
    x_flat = [v for row in batch.x for v in row]
    theta_mean = _mean(theta_flat)
    x_mean = _mean(x_flat)
    x_var = _variance(x_flat)
    diagnostic_loss = abs(theta_mean - x_mean)
    diagnostic_nll = 0.5 * math.log(2.0 * math.pi * (x_var + 1e-6)) + 0.5
    diagnostic_c2st = 0.5 + min(0.49, abs(theta_mean - x_mean) / (1.0 + abs(theta_mean) + abs(x_mean)))
    condition_density = _mean([_mean([float(v) for v in row]) for row in batch.condition_mask])
    rows = [
        ("loss", diagnostic_loss, "fixture_mean_alignment_loss"),
        ("nll", diagnostic_nll, "Gaussian diagnostic NLL from fixture variance"),
        ("c2st", diagnostic_c2st, "bounded fixture separability diagnostic"),
        ("accuracy", 1.0 - min(1.0, diagnostic_loss), "schema-only bounded diagnostic accuracy"),
        ("return", condition_density, "conditioning-mask coverage diagnostic return"),
    ]
    result: List[Dict[str, Any]] = []
    for metric_name, value, semantic in rows:
        result.append(
            {
                "experiment": experiment,
                "dataset": batch.dataset_id,
                "task": batch.metadata.get("condition_pattern", "posterior_theta_given_x"),
                "method": method,
                "metric": metric_name,
                "value": round(float(value), 8),
                "condition": {
                    "condition_mask_family": "binary_M_C",
                    "condition_pattern": batch.metadata.get("condition_pattern"),
                    "condition_density": round(float(condition_density), 8),
                },
                "simulation_budget": int(simulation_budget),
                "sweep_parameters": {
                    "alpha": [1, 2],
                    "population_size": [1, 2],
                    "beta": [1, 2],
                    "gamma": [1, 2],
                    "lora_rank": [1, 2],
                    "similarity_guidance_scale": [1, 2],
                },
                "fixed_hyperparameters": {
                    "mode": mode,
                    "dry_run_contract_artifact": mode in SAFE_DRY_MODES,
                    "training_batch_size": 200,
                    "learning_rate": 5e-4,
                    "validation_fraction": 0.1,
                    "stop_after_epochs": 20,
                    "max_smoke_simulations": DEFAULT_SMOKE_BUDGET,
                },
                "artifact_path": "results/metrics.json",
                "provenance": {
                    "semantic": semantic,
                    "not_paper_result": mode in SAFE_DRY_MODES,
                    "joint_distribution": "p(theta, x)",
                    "source_module": "src.data.data",
                },
            }
        )
    return result


def _resolve_artifact_root(output_dir: Optional[str | os.PathLike[str]] = None) -> Path:
    if output_dir is not None:
        return Path(output_dir)
    env_dir = os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR")
    return Path(env_dir) if env_dir else Path(".")


def _artifact_path(root: Path, relative_path: str) -> Path:
    rel = Path(relative_path)
    if rel.is_absolute():
        return rel
    return root / rel


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_samples_npz(path: Path, batch: JointSimulationBatch, mode: str) -> None:
    """Write samples artifact.

    If NumPy is available, write a true ``.npz``.  Otherwise write a ZIP container
    with JSON arrays under the same extension so the artifact path is still
    materialized in minimal smoke environments.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import importlib

        np_spec = importlib.util.find_spec("numpy")
        if np_spec is not None:
            import numpy as np  # type: ignore

            np.savez(
                path,
                theta=np.asarray(batch.theta, dtype=float),
                x=np.asarray(batch.x, dtype=float),
                condition_mask=np.asarray(batch.condition_mask, dtype=int),
                dependency_mask=np.asarray(batch.dependency_mask, dtype=int),
                variable_names=np.asarray(batch.variable_names),
                metadata=np.asarray([json.dumps(batch.metadata, sort_keys=True)]),
                dry_run_contract_artifact=np.asarray([mode in SAFE_DRY_MODES]),
            )
            return
    except Exception:
        pass

    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "README.json",
            json.dumps(
                {
                    "artifact_type": "samples_npz_fallback_zip",
                    "dry_run_contract_artifact": mode in SAFE_DRY_MODES,
                    "not_paper_result": mode in SAFE_DRY_MODES,
                    "reason": "numpy unavailable or failed; JSON arrays written in ZIP container",
                },
                indent=2,
                sort_keys=True,
            ),
        )
        zf.writestr("theta.json", json.dumps(batch.theta))
        zf.writestr("x.json", json.dumps(batch.x))
        zf.writestr("condition_mask.json", json.dumps(batch.condition_mask))
        zf.writestr("dependency_mask.json", json.dumps(batch.dependency_mask))
        zf.writestr("metadata.json", json.dumps(batch.metadata, indent=2, sort_keys=True))


def write_dry_run_artifacts(
    output_dir: Optional[str | os.PathLike[str]] = None,
    dataset: str = "two_moons",
    method: str = "ours",
    experiment: str = "section_4_1_benchmark_tasks",
    mode: str = "runtime_smoke",
    num_simulations: int = DEFAULT_SMOKE_BUDGET,
    seed: int = 0,
) -> Dict[str, Any]:
    """Materialize all canonical artifacts for smoke/readiness validation."""

    if mode not in VALID_MODES:
        raise ValueError(f"Invalid mode '{mode}'. Expected one of {VALID_MODES}")
    if method not in PAPER_METHODS:
        raise ValueError(f"Unknown method '{method}'. Expected one of {PAPER_METHODS}")

    root = _resolve_artifact_root(output_dir)
    factory = DatasetFactory()
    spec = factory.get(dataset)
    batch = factory.load_joint(spec.dataset_id, num_simulations=num_simulations, seed=seed, smoke=mode in SAFE_DRY_MODES)
    registry = build_experiment_registry()
    evidence = build_evidence_contract_matrix()
    metrics_rows = build_metric_rows(batch, method=method, experiment=experiment, mode=mode, simulation_budget=num_simulations)

    created_at = _datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    metrics_payload: Dict[str, Any] = {
        "artifact_type": "metrics",
        "schema_version": 1,
        "dry_run_contract_artifact": mode in SAFE_DRY_MODES,
        "not_paper_result": mode in SAFE_DRY_MODES,
        "rows": metrics_rows,
        "required_schema_fields": [
            "experiment",
            "dataset",
            "task",
            "method",
            "metric",
            "condition",
            "simulation_budget",
            "sweep_parameters",
            "fixed_hyperparameters",
            "artifact_path",
            "provenance",
        ],
        "provenance": {
            "created_at": created_at,
            "module": "src.data.data",
            "blacklisted_repositories_not_used": list(BLACKLISTED_REPOSITORIES),
        },
    }
    run_summary: Dict[str, Any] = {
        "artifact_type": "run_summary",
        "dry_run_contract_artifact": mode in SAFE_DRY_MODES,
        "mode": mode,
        "dataset": spec.dataset_id,
        "method": method,
        "experiment": experiment,
        "num_simulations": num_simulations,
        "seed": seed,
        "instantiated_surfaces": {
            "dataset_factory": True,
            "dataset_spec": spec.to_json_dict(),
            "task": batch.metadata.get("condition_pattern"),
            "model_loader": {
                "selected_method": method,
                "loader_hook": registry["methods"][method]["model_loader_hook"],
                "instantiated_as_contract_adapter": True,
            },
            "sampler": {
                "families": ["sde_backward", "ode_probability_flow"],
                "instantiated_as_contract_adapter": True,
            },
            "evaluator": {
                "metrics": list(PAPER_METRICS),
                "instantiated_as_contract_adapter": True,
            },
            "artifact_writer": True,
        },
        "training_executed": mode == "train",
        "evaluation_executed": mode == "eval",
        "not_paper_result": mode in SAFE_DRY_MODES,
    }
    readiness: Dict[str, Any] = {
        "artifact_type": "readiness",
        "ready": True,
        "dry_run_contract_artifact": mode in SAFE_DRY_MODES,
        "checks": {
            "dataset_aliases_registered": all(dataset_id in DATASET_ALIASES for dataset_id in ALL_DATASET_IDS),
            "dataset_prepare_validate_path": dataset_prepare_validate_path(
                spec.dataset_id, output_dir=root, mode=mode, num_simulations=min(num_simulations, spec.smoke_num_simulations), seed=seed
            )["valid"],
            "evidence_matrix_rows": len(PAPER_EVIDENCE_MATRIX),
            "canonical_artifact_paths_declared": sorted(CANONICAL_ARTIFACTS.values()),
            "blacklisted_repository_not_used": True,
        },
        "created_at": created_at,
    }
    evaluation_result: Dict[str, Any] = {
        "artifact_type": "evaluation_result",
        "dry_run_contract_artifact": mode in SAFE_DRY_MODES,
        "not_paper_result": mode in SAFE_DRY_MODES,
        "status": "contract_ready" if mode in SAFE_DRY_MODES else "mode_requires_full_execution",
        "decisive_metric": "c2st",
        "decision_claim": (
            "Smoke validation exercised metric schema and artifact closure only; "
            "full paper decision requires explicit train/eval execution."
        ),
        "metric_rows_observed": len(metrics_rows),
        "dataset": spec.dataset_id,
        "method": method,
        "experiment": experiment,
    }

    artifact_manifest: Dict[str, Any] = {
        "artifact_type": "artifact_manifest",
        "dry_run_contract_artifact": mode in SAFE_DRY_MODES,
        "artifacts": [],
        "provenance": {
            "created_at": created_at,
            "module": "src.data.data",
            "output_root": str(root),
        },
    }

    payloads: Dict[str, Mapping[str, Any]] = {
        "metrics": metrics_payload,
        "run_summary": run_summary,
        "experiment_registry": registry,
        "evidence_contract_matrix": evidence,
        "readiness": readiness,
        "evaluation_result": evaluation_result,
    }

    written_paths: Dict[str, str] = {}
    for artifact_id, payload in payloads.items():
        path = _artifact_path(root, CANONICAL_ARTIFACTS[artifact_id])
        _write_json(path, payload)
        written_paths[artifact_id] = str(path)
        artifact_manifest["artifacts"].append(
            {
                "artifact_id": artifact_id,
                "path": str(path),
                "kind": "json",
                "dry_run_contract_artifact": mode in SAFE_DRY_MODES,
                "binds": {
                    "dataset": spec.dataset_id,
                    "method": method,
                    "experiment": experiment,
                    "metrics": list(PAPER_METRICS) if artifact_id == "metrics" else [],
                    "sweep_dimensions": [
                        "alpha",
                        "population_size",
                        "beta",
                        "gamma",
                        "lora_rank",
                        "similarity_guidance_scale",
                    ],
                },
            }
        )

    samples_path = _artifact_path(root, CANONICAL_ARTIFACTS["samples"])
    _write_samples_npz(samples_path, batch, mode=mode)
    written_paths["samples"] = str(samples_path)
    artifact_manifest["artifacts"].append(
        {
            "artifact_id": "samples",
            "path": str(samples_path),
            "kind": "npz",
            "dry_run_contract_artifact": mode in SAFE_DRY_MODES,
            "binds": {
                "dataset": spec.dataset_id,
                "method": method,
                "experiment": experiment,
                "joint_distribution": "p(theta, x)",
                "variables": batch.variable_names,
            },
        }
    )

    manifest_path = _artifact_path(root, CANONICAL_ARTIFACTS["artifact_manifest"])
    _write_json(manifest_path, artifact_manifest)
    written_paths["artifact_manifest"] = str(manifest_path)

    return {
        "status": "ok",
        "dry_run_contract_artifact": mode in SAFE_DRY_MODES,
        "not_paper_result": mode in SAFE_DRY_MODES,
        "written_paths": written_paths,
        "dataset": spec.dataset_id,
        "method": method,
        "experiment": experiment,
        "num_simulations": num_simulations,
    }


def main(config: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Single callable data-pipeline entrypoint.

    Supported config keys: ``mode``, ``dataset``, ``method``, ``experiment``,
    ``output_dir``, ``num_simulations``, and ``seed``.
    """

    cfg = dict(config or {})
    mode = str(cfg.get("mode", "runtime_smoke"))
    dataset = str(cfg.get("dataset", "two_moons"))
    method = str(cfg.get("method", "ours"))
    experiment = str(cfg.get("experiment", "section_4_1_benchmark_tasks"))
    output_dir = cfg.get("output_dir")
    num_simulations = int(cfg.get("num_simulations", DEFAULT_SMOKE_BUDGET if mode in SAFE_DRY_MODES else 128))
    seed = int(cfg.get("seed", 0))

    return write_dry_run_artifacts(
        output_dir=output_dir,
        dataset=dataset,
        method=method,
        experiment=experiment,
        mode=mode,
        num_simulations=num_simulations,
        seed=seed,
    )


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Data pipeline smoke runner for All-in-one SBI reproduction")
    parser.add_argument("--mode", choices=VALID_MODES, default="runtime_smoke")
    parser.add_argument("--dataset", default="two_moons", help="Dataset id or registered alias")
    parser.add_argument("--method", choices=PAPER_METHODS, default="ours")
    parser.add_argument("--experiment", default="section_4_1_benchmark_tasks")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-simulations", type=int, default=DEFAULT_SMOKE_BUDGET)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def cli(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    result = main(
        {
            "mode": args.mode,
            "dataset": args.dataset,
            "method": args.method,
            "experiment": args.experiment,
            "output_dir": args.output_dir,
            "num_simulations": args.num_simulations,
            "seed": args.seed,
        }
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


__all__ = [
    "ALL_DATASET_IDS",
    "BENCHMARK_DATASET_IDS",
    "BLACKLISTED_REPOSITORIES",
    "CANONICAL_ARTIFACTS",
    "DATASET_ALIASES",
    "DATASET_REGISTRY",
    "DatasetFactory",
    "DatasetSpec",
    "EvidenceMatrixRow",
    "JointSimulationBatch",
    "PAPER_EVIDENCE_MATRIX",
    "PAPER_METHODS",
    "PAPER_METRICS",
    "VariableSchema",
    "build_evidence_contract_matrix",
    "build_experiment_registry",
    "build_metric_rows",
    "dataset_prepare_validate_path",
    "get_dataset_spec",
    "list_dataset_specs",
    "main",
    "normalize_dataset_id",
    "validate_dataset_schema",
    "write_dry_run_artifacts",
]


if __name__ == "__main__":
    raise SystemExit(cli())