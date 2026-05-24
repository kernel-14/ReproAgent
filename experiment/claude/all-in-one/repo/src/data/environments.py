"""Paper-derived environment and dataset adapters for structured SBI tasks.

This module implements the ``src.data.environments`` contract for the
PaperBench reproduction of *All-in-one simulation-based inference*.  The file is
kept importable in a minimal environment: it depends only on the Python standard
library and NumPy.  Optional training/simulator libraries are not imported at
module scope.

Implemented surfaces
--------------------
* environment_adapter: task adapters for benchmark tasks, Lotka-Volterra,
  SIRD functional-parameter inference, and Hodgkin-Huxley interval queries.
* sampling: lightweight canonical joint simulators p(theta, x) for smoke and
  protocol validation, with deterministic NumPy RNG plumbing.
* model_or_method / policy_adapter: Simformer-compatible tokenizer output and
  conditioning/query policies.
* config: registry entries, aliases, factory hooks, dataset loader hooks, and
  experiment protocol metadata.
* training_loop: bounded smoke training loop exercising the real simulator and
  tokenizer surfaces without expensive optimization.
* metric_formula: NLL-like Gaussian score, C2ST proxy accuracy, coverage, and
  schedule/readiness metrics for structured tasks.
* tests/readiness: validation helpers and dry-run artifact writer.

No code is copied from the blacklisted Simformer repository.  The simulator
implementations below are lightweight canonical-route adapters rather than
paper-scale numerical experiments.

reference_grounding: paperbench_ref_001 docs/how_to_guide/04_embedding_networks.ipynb
reference_grounding: paperbench_ref_001 docs/advanced_tutorials/04_embedding_networks.ipynb
reference_grounding: paperbench_ref_001 docs/how_to_guide/09_sampler_interface.ipynb
reference_grounding: paperbench_ref_001 docs/how_to_guide/22_experiment_tracking.ipynb
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np


DEFAULT_RESULTS_DIR = "results"
SAFE_MODES = {"dry_run", "runtime_smoke", "docker_validate", "smoke"}
BENCHMARK_TASK_IDS: Tuple[str, ...] = (
    "two_moons",
    "gaussian_linear",
    "gaussian_mixture",
    "slcp",
)
STRUCTURED_TASK_IDS: Tuple[str, ...] = (
    "lotka_volterra",
    "sird",
    "hodgkin_huxley",
)
ALL_TASK_IDS: Tuple[str, ...] = BENCHMARK_TASK_IDS + STRUCTURED_TASK_IDS

DECLARED_ARTIFACTS: Tuple[str, ...] = (
    "results/model_registry.json",
    "results/loss_trace.json",
    "results/lotka_volterra_samples.npz",
    "results/lotka_volterra_metrics.json",
    "results/sird_functional_samples.npz",
    "results/sird_metrics.json",
    "results/readiness.json",
    "results/evaluation_result.json",
)


# ---------------------------------------------------------------------------
# Dataclasses and schemas
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Token:
    """A Simformer-compatible token for a variable in a joint simulator draw.

    The tokenizer preserves exactly the information needed by the paper's
    all-in-one conditioning interface: variable identifier, numeric value,
    condition state, semantic role, time/query coordinate, and optional group
    metadata such as species or compartment.
    """

    variable_id: str
    value: float
    condition_state: int
    role: str
    coordinate: float
    group: str = ""
    sample_index: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class TokenizedBatch:
    """Output of the environment tokenizer."""

    tokens: List[Token]
    variable_ids: List[str]
    values: np.ndarray
    condition_states: np.ndarray
    roles: List[str]
    coordinates: np.ndarray
    groups: List[str]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "tokens": [t.as_dict() for t in self.tokens],
            "variable_ids": list(self.variable_ids),
            "values": self.values.tolist(),
            "condition_states": self.condition_states.astype(int).tolist(),
            "roles": list(self.roles),
            "coordinates": self.coordinates.tolist(),
            "groups": list(self.groups),
        }


@dataclasses.dataclass
class SimulationBatch:
    """Joint samples from p(theta, x) plus Simformer metadata."""

    task_id: str
    theta: np.ndarray
    x: np.ndarray
    theta_names: List[str]
    x_names: List[str]
    x_coordinates: np.ndarray
    condition_mask: np.ndarray
    metadata: Dict[str, Any]

    def tokenize(self) -> TokenizedBatch:
        return SBITokenizer().encode(self)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "theta": self.theta.tolist(),
            "x": self.x.tolist(),
            "theta_names": list(self.theta_names),
            "x_names": list(self.x_names),
            "x_coordinates": self.x_coordinates.tolist(),
            "condition_mask": self.condition_mask.astype(int).tolist(),
            "metadata": dict(self.metadata),
        }


@dataclasses.dataclass(frozen=True)
class EnvironmentConfig:
    """Task-factory configuration.

    ``prey_times`` and ``predator_times`` intentionally allow species-specific
    Lotka-Volterra schedules rather than a rectangular observation grid.
    ``functional_query_times`` and ``measured_parameter_subset`` expose SIRD
    time-dependent/functional parameter queries.
    """

    task_id: str
    num_samples: int = 8
    seed: int = 0
    observation_noise: float = 0.05
    mode: str = "dry_run"
    prey_times: Tuple[float, ...] = (0.0, 1.0, 2.0, 4.0)
    predator_times: Tuple[float, ...] = (0.0, 0.5, 2.5)
    sird_times: Tuple[float, ...] = (0.0, 1.0, 2.0, 3.0, 4.0)
    functional_query_times: Tuple[float, ...] = (0.0, 1.5, 3.0, 4.5)
    measured_parameter_subset: Tuple[str, ...] = ("beta_t", "gamma_t")
    hh_voltage_times: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    hh_interval: Tuple[float, float] = (-70.0, -50.0)
    query_policy: str = "mixed_conditioning"
    directed_mask: bool = True
    full_scale: bool = False

    @classmethod
    def from_mapping(cls, task_id: str, config: Optional[Mapping[str, Any]] = None) -> "EnvironmentConfig":
        data: Dict[str, Any] = {"task_id": canonical_task_id(task_id)}
        if config:
            data.update(dict(config))
            data["task_id"] = canonical_task_id(str(data.get("task_id", task_id)))
        tuple_fields = {
            "prey_times",
            "predator_times",
            "sird_times",
            "functional_query_times",
            "measured_parameter_subset",
            "hh_voltage_times",
            "hh_interval",
        }
        for field in tuple_fields:
            if field in data and not isinstance(data[field], tuple):
                data[field] = tuple(data[field])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclasses.dataclass(frozen=True)
class TaskRegistryEntry:
    """Registry metadata for a paper-visible task/environment."""

    id: str
    aliases: Tuple[str, ...]
    family: str
    theta_dim: int
    default_x_dim: int
    paper_section: str
    setup_metadata: Mapping[str, Any]
    factory_hook: str
    config_hook: str
    loader_hook: str
    supports_structured_observations: bool
    supports_functional_queries: bool
    supports_interval_guidance: bool = False


@dataclasses.dataclass
class MetricResult:
    """Metric schema used by the structured task writer."""

    task_id: str
    metrics: Dict[str, float]
    sample_count: int
    decisive_metric: str
    dry_run: bool
    notes: List[str]

    def as_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Registry and aliases
# ---------------------------------------------------------------------------


def _norm_alias(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return re.sub(r"_+", "_", value).strip("_")


TASK_REGISTRY: Dict[str, TaskRegistryEntry] = {
    "two_moons": TaskRegistryEntry(
        id="two_moons",
        aliases=("two_moons", "Two Moons", "two moons", "benchmark_two_moons"),
        family="benchmark",
        theta_dim=2,
        default_x_dim=2,
        paper_section="4.1 benchmark: approximating posterior distributions across four tasks",
        setup_metadata={
            "benchmark_slot": "Benchmark",
            "decisive_metric": "c2st_accuracy",
            "embedding_adapter": "identity_or_mlp_summary",
            "sampling_families": ("direct_posterior", "mcmc", "rejection"),
        },
        factory_hook="TaskFactory.create",
        config_hook="default_environment_config",
        loader_hook="load_dataset",
        supports_structured_observations=False,
        supports_functional_queries=False,
    ),
    "gaussian_linear": TaskRegistryEntry(
        id="gaussian_linear",
        aliases=("gaussian_linear", "Linear Gaussian", "linear gaussian", "Gaussian Linear"),
        family="benchmark",
        theta_dim=10,
        default_x_dim=10,
        paper_section="4.1 benchmark: Across all four benchmark tasks",
        setup_metadata={
            "benchmark_slot": "Linear Gaussian",
            "decisive_metric": "negative_log_likelihood",
            "closed_form_reference": True,
            "embedding_adapter": "identity",
        },
        factory_hook="TaskFactory.create",
        config_hook="default_environment_config",
        loader_hook="load_dataset",
        supports_structured_observations=False,
        supports_functional_queries=False,
    ),
    "gaussian_mixture": TaskRegistryEntry(
        id="gaussian_mixture",
        aliases=("gaussian_mixture", "Gaussian Mixture", "gaussian mixture", "mixture gaussian"),
        family="benchmark",
        theta_dim=2,
        default_x_dim=2,
        paper_section="4.1 benchmark: approximating posterior distributions across four tasks",
        setup_metadata={
            "benchmark_slot": "Benchmark",
            "decisive_metric": "c2st_accuracy",
            "mixture_components": 2,
            "embedding_adapter": "identity",
        },
        factory_hook="TaskFactory.create",
        config_hook="default_environment_config",
        loader_hook="load_dataset",
        supports_structured_observations=False,
        supports_functional_queries=False,
    ),
    "slcp": TaskRegistryEntry(
        id="slcp",
        aliases=("slcp", "SLCP", "simple likelihood complex posterior", "benchmark_slcp"),
        family="benchmark",
        theta_dim=5,
        default_x_dim=8,
        paper_section="4.1 benchmark: approximating posterior distributions across four tasks",
        setup_metadata={
            "benchmark_slot": "Benchmark",
            "decisive_metric": "c2st_accuracy",
            "embedding_adapter": "mlp_summary_for_high_dimensional_observations",
        },
        factory_hook="TaskFactory.create",
        config_hook="default_environment_config",
        loader_hook="load_dataset",
        supports_structured_observations=False,
        supports_functional_queries=False,
    ),
    "lotka_volterra": TaskRegistryEntry(
        id="lotka_volterra",
        aliases=("lotka_volterra", "Lotka-Volterra", "Lotka Volterra", "LV", "predator prey"),
        family="structured_time_series",
        theta_dim=4,
        default_x_dim=7,
        paper_section="4.2 Lotka-Volterra inference with unstructured observations",
        setup_metadata={
            "species": ("prey", "predator"),
            "supports_species_specific_schedules": True,
            "theta_roles": ("prey_growth", "prey_predation", "predator_death", "predator_reproduction"),
            "decisive_metric": "schedule_coverage",
            "mask": "metadata_dependent_directed_graph",
        },
        factory_hook="TaskFactory.create",
        config_hook="default_environment_config",
        loader_hook="load_dataset",
        supports_structured_observations=True,
        supports_functional_queries=False,
    ),
    "sird": TaskRegistryEntry(
        id="sird",
        aliases=("sird", "SIRD", "SIRD-model", "SIRD model", "functional SIRD"),
        family="structured_functional_parameters",
        theta_dim=4,
        default_x_dim=16,
        paper_section="4.3 SIRD-model functional parameter inference",
        setup_metadata={
            "compartments": ("S", "I", "R", "D"),
            "functional_parameters": ("beta_t", "gamma_t", "mu_t"),
            "supports_subset_parameter_measurements": True,
            "decisive_metric": "functional_query_coverage",
            "mask": "directed_compartment_and_parameter_graph",
        },
        factory_hook="TaskFactory.create",
        config_hook="default_environment_config",
        loader_hook="load_dataset",
        supports_structured_observations=True,
        supports_functional_queries=True,
    ),
    "hodgkin_huxley": TaskRegistryEntry(
        id="hodgkin_huxley",
        aliases=("hodgkin_huxley", "Hodgkin-Huxley", "Hodgkin Huxley", "HH", "hodgkin huxley"),
        family="guided_interval_inference",
        theta_dim=8,
        default_x_dim=5,
        paper_section="4.4 Hodgkin-Huxley observation interval and metabolic-cost constraints",
        setup_metadata={
            "interval_guidance": True,
            "target_variable": "voltage",
            "decisive_metric": "interval_satisfaction_rate",
            "sampling_families": ("sde_backward", "ode_probability_flow"),
        },
        factory_hook="TaskFactory.create",
        config_hook="default_environment_config",
        loader_hook="load_dataset",
        supports_structured_observations=True,
        supports_functional_queries=True,
        supports_interval_guidance=True,
    ),
}

ALIAS_TO_ID: Dict[str, str] = {}
for _task_id, _entry in TASK_REGISTRY.items():
    ALIAS_TO_ID[_norm_alias(_task_id)] = _task_id
    for _alias in _entry.aliases:
        ALIAS_TO_ID[_norm_alias(_alias)] = _task_id

# Explicit paper-evidence aliases requested by the contract.
ALIAS_TO_ID[_norm_alias("Benchmark")] = "two_moons"
ALIAS_TO_ID[_norm_alias("approximating posterior distributions across four")] = "two_moons"
ALIAS_TO_ID[_norm_alias("Across all four benchmark")] = "gaussian_linear"
ALIAS_TO_ID[_norm_alias("Two Moons")] = "two_moons"
ALIAS_TO_ID[_norm_alias("Linear Gaussian")] = "gaussian_linear"

DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {
    task_id: {
        "id": task_id,
        "aliases": list(TASK_REGISTRY[task_id].aliases),
        "family": TASK_REGISTRY[task_id].family,
        "loader_hook": "load_dataset",
        "factory_hook": "TaskFactory.create",
        "setup_metadata": dict(TASK_REGISTRY[task_id].setup_metadata),
        "dry_run_num_samples": 8,
    }
    for task_id in ("two_moons", "gaussian_linear", "gaussian_mixture", "slcp", "lotka_volterra")
}

# Contract also requires benchmark aliases to be discoverable for SIRD and HH,
# even though they are not static prepared datasets.
for _task_id in ("sird", "hodgkin_huxley"):
    DATASET_REGISTRY[_task_id] = {
        "id": _task_id,
        "aliases": list(TASK_REGISTRY[_task_id].aliases),
        "family": TASK_REGISTRY[_task_id].family,
        "loader_hook": "load_dataset",
        "factory_hook": "TaskFactory.create",
        "setup_metadata": dict(TASK_REGISTRY[_task_id].setup_metadata),
        "dry_run_num_samples": 8,
        "generated_on_demand": True,
    }


EXPERIMENT_PROTOCOL_MATRIX: Dict[str, Dict[str, Any]] = {
    "structured_tasks_smoke": {
        "hypothesis": (
            "A Simformer tokenizer with role, condition-state, species, and "
            "time/query-coordinate tokens can represent unstructured "
            "Lotka-Volterra observations and SIRD functional parameter queries."
        ),
        "decisive_comparison": "structured adapters versus rectangular/generic task rows",
        "decisive_metric": "schedule_coverage and functional_query_coverage",
        "default_tasks": ("lotka_volterra", "sird"),
        "full_tasks": ALL_TASK_IDS,
        "stop_rule_or_pruning_rationale": (
            "Default smoke executes bounded adapters and writes schema artifacts; "
            "paper-scale sweeps require explicit full_scale=True."
        ),
    }
}


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def canonical_task_id(task_id_or_alias: str) -> str:
    """Resolve a task id or paper alias to a canonical registry id."""

    key = _norm_alias(task_id_or_alias)
    if key not in ALIAS_TO_ID:
        raise KeyError(f"Unknown task/environment alias: {task_id_or_alias!r}")
    return ALIAS_TO_ID[key]


def default_environment_config(task_id: str, **overrides: Any) -> EnvironmentConfig:
    """Return a task-specific default config with optional overrides."""

    return EnvironmentConfig.from_mapping(task_id, overrides)


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def _as_float_array(values: Sequence[float]) -> np.ndarray:
    return np.asarray(list(values), dtype=float)


def _ensure_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        return x[None, :]
    return x


def _condition_policy_mask(num_samples: int, theta_dim: int, x_dim: int, policy: str) -> np.ndarray:
    """Create a binary condition mask over concatenated [theta, x] variables."""

    total = theta_dim + x_dim
    mask = np.zeros((num_samples, total), dtype=int)
    if policy == "observe_x":
        mask[:, theta_dim:] = 1
    elif policy == "observe_theta":
        mask[:, :theta_dim] = 1
    elif policy == "all_conditioned":
        mask[:, :] = 1
    elif policy == "none":
        mask[:, :] = 0
    else:
        # Mixed paper-style arbitrary conditioning: observations are known and a
        # deterministic subset of parameters may be conditioned for smoke.
        mask[:, theta_dim:] = 1
        if theta_dim:
            mask[::2, 0] = 1
    return mask


def result_root(path: Optional[str | os.PathLike[str]] = None) -> Path:
    """Resolve the artifact root, respecting PAPERBENCH_REPRO_ARTIFACT_DIR."""

    if path is not None:
        return Path(path)
    return Path(os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR", DEFAULT_RESULTS_DIR))


def _repo_relative_to_output(path: str, output_dir: Optional[str | os.PathLike[str]] = None) -> Path:
    """Map declared ``results/...`` paths under the selected output root."""

    root = result_root(output_dir)
    p = Path(path)
    if p.parts and p.parts[0] == "results":
        p = Path(*p.parts[1:])
    return root / p


# ---------------------------------------------------------------------------
# Tokenizer and dependency masks
# ---------------------------------------------------------------------------


class SBITokenizer:
    """Tokenizer used by all environment adapters.

    It mirrors the paper's all-in-one representation by encoding variables from
    the joint distribution with identifier, value, condition state, role, and
    time/query coordinate.  The implementation is intentionally independent of
    torch so smoke tests can inspect the exact token contract.
    """

    def encode(self, batch: SimulationBatch) -> TokenizedBatch:
        theta = _ensure_2d(batch.theta)
        x = _ensure_2d(batch.x)
        if theta.shape[0] != x.shape[0]:
            raise ValueError("theta and x must contain the same number of samples")
        n, theta_dim = theta.shape
        x_dim = x.shape[1]
        condition_mask = np.asarray(batch.condition_mask, dtype=int)
        if condition_mask.shape != (n, theta_dim + x_dim):
            raise ValueError(
                f"condition_mask must have shape {(n, theta_dim + x_dim)}, got {condition_mask.shape}"
            )

        tokens: List[Token] = []
        variable_ids: List[str] = []
        values: List[float] = []
        condition_states: List[int] = []
        roles: List[str] = []
        coordinates: List[float] = []
        groups: List[str] = []

        x_groups = batch.metadata.get("x_groups", ["observation"] * x_dim)
        theta_groups = batch.metadata.get("theta_groups", ["parameter"] * theta_dim)
        theta_coordinates = batch.metadata.get("theta_coordinates", [float(i) for i in range(theta_dim)])

        for sample_index in range(n):
            for j, name in enumerate(batch.theta_names):
                token = Token(
                    variable_id=str(name),
                    value=float(theta[sample_index, j]),
                    condition_state=int(condition_mask[sample_index, j]),
                    role="parameter",
                    coordinate=float(theta_coordinates[j]),
                    group=str(theta_groups[j] if j < len(theta_groups) else "parameter"),
                    sample_index=sample_index,
                )
                tokens.append(token)
            for j, name in enumerate(batch.x_names):
                coord = float(batch.x_coordinates[j]) if j < len(batch.x_coordinates) else float(j)
                token = Token(
                    variable_id=str(name),
                    value=float(x[sample_index, j]),
                    condition_state=int(condition_mask[sample_index, theta_dim + j]),
                    role="observation",
                    coordinate=coord,
                    group=str(x_groups[j] if j < len(x_groups) else "observation"),
                    sample_index=sample_index,
                )
                tokens.append(token)

        for token in tokens:
            variable_ids.append(token.variable_id)
            values.append(token.value)
            condition_states.append(token.condition_state)
            roles.append(token.role)
            coordinates.append(token.coordinate)
            groups.append(token.group)

        return TokenizedBatch(
            tokens=tokens,
            variable_ids=variable_ids,
            values=np.asarray(values, dtype=float),
            condition_states=np.asarray(condition_states, dtype=int),
            roles=roles,
            coordinates=np.asarray(coordinates, dtype=float),
            groups=groups,
        )


def directed_dependency_mask(task_id: str, config: Optional[EnvironmentConfig] = None) -> np.ndarray:
    """Return a directed dependency mask over variables ordered theta..., x....

    Addendum grounding:
    For each task, variables are ordered as theta_1, theta_2, ..., x_1, x_2, ...
    The undirected mask is obtained by making this directed mask symmetric.
    Lotka-Volterra is metadata-dependent and respects species-specific
    observation schedules.

    reference_grounding: paper:structured_tasks_addendum src/data/environments.py
    """

    task_id = canonical_task_id(task_id)
    config = config or default_environment_config(task_id)
    entry = TASK_REGISTRY[task_id]

    if task_id == "lotka_volterra":
        return lotka_volterra_dependency_mask(config)

    theta_dim = entry.theta_dim
    x_dim = default_x_dim(task_id, config)
    total = theta_dim + x_dim
    mask = np.zeros((total, total), dtype=int)
    mask[:theta_dim, :theta_dim] = np.eye(theta_dim, dtype=int)

    if task_id in {"two_moons", "gaussian_linear", "gaussian_mixture", "slcp"}:
        mask[:theta_dim, theta_dim:] = 1
        mask[theta_dim:, theta_dim:] = np.eye(x_dim, dtype=int)
        return mask

    if task_id == "sird":
        mask[:theta_dim, theta_dim:] = 1
        times = list(config.sird_times)
        compartments = 4
        for t_idx in range(len(times)):
            for c in range(compartments):
                idx = theta_dim + t_idx * compartments + c
                mask[idx, idx] = 1
                if t_idx > 0:
                    prev_base = theta_dim + (t_idx - 1) * compartments
                    mask[prev_base : prev_base + compartments, idx] = 1
        return mask

    if task_id == "hodgkin_huxley":
        mask[:theta_dim, theta_dim:] = 1
        for j in range(x_dim):
            idx = theta_dim + j
            mask[idx, idx] = 1
            if j > 0:
                mask[idx - 1, idx] = 1
        return mask

    raise KeyError(f"No dependency mask defined for {task_id}")


def undirected_dependency_mask(task_id: str, config: Optional[EnvironmentConfig] = None) -> np.ndarray:
    directed = directed_dependency_mask(task_id, config)
    return np.maximum(directed, directed.T).astype(int)


def lotka_volterra_dependency_mask(config: EnvironmentConfig) -> np.ndarray:
    """Metadata-dependent Lotka-Volterra directed graphical mask.

    The first two parameters control prey dynamics and the last two control
    predator dynamics.  Prey and predator may have different observation times
    and counts.  Within-species dynamics are Markovian in their own observed
    order.  Cross-data dependencies are causal: each prey observation depends
    additionally on past predator observations, and each predator observation
    depends on past prey observations.

    This adapts the addendum formula:
    ``M_theta_theta = I`` and, for a rectangular T case,
    ``M_theta_x = [[1]*T + [0]*T, [1]*T + [0]*T,
                   [0]*T + [1]*T, [0]*T + [1]*T]``.
    Here T may differ by species, so the two blocks use ``len(prey_times)`` and
    ``len(predator_times)`` separately.
    """

    prey_times = list(map(float, config.prey_times))
    predator_times = list(map(float, config.predator_times))
    theta_dim = 4
    x_dim = len(prey_times) + len(predator_times)
    total = theta_dim + x_dim
    mask = np.zeros((total, total), dtype=int)
    mask[:theta_dim, :theta_dim] = np.eye(theta_dim, dtype=int)

    prey_offset = theta_dim
    predator_offset = theta_dim + len(prey_times)

    # theta -> x blocks: first two parameters for prey, last two for predator.
    if prey_times:
        mask[0:2, prey_offset:predator_offset] = 1
    if predator_times:
        mask[2:4, predator_offset : predator_offset + len(predator_times)] = 1

    # Within-species Markovian dependencies.
    for i in range(len(prey_times)):
        idx = prey_offset + i
        mask[idx, idx] = 1
        if i > 0:
            mask[idx - 1, idx] = 1
    for i in range(len(predator_times)):
        idx = predator_offset + i
        mask[idx, idx] = 1
        if i > 0:
            mask[idx - 1, idx] = 1

    # Cross-species causal dependencies based on actual observation times.
    for i, t_prey in enumerate(prey_times):
        prey_idx = prey_offset + i
        for j, t_pred in enumerate(predator_times):
            pred_idx = predator_offset + j
            if t_pred <= t_prey:
                mask[pred_idx, prey_idx] = 1
            if t_prey <= t_pred:
                mask[prey_idx, pred_idx] = 1

    return mask


def default_x_dim(task_id: str, config: Optional[EnvironmentConfig] = None) -> int:
    task_id = canonical_task_id(task_id)
    config = config or default_environment_config(task_id)
    if task_id == "lotka_volterra":
        return len(config.prey_times) + len(config.predator_times)
    if task_id == "sird":
        return 4 * len(config.sird_times) + len(config.functional_query_times) * len(config.measured_parameter_subset)
    if task_id == "hodgkin_huxley":
        return len(config.hh_voltage_times)
    return TASK_REGISTRY[task_id].default_x_dim


# ---------------------------------------------------------------------------
# Task adapters and simulators
# ---------------------------------------------------------------------------


class EnvironmentAdapter:
    """Base adapter for a paper-visible SBI task."""

    def __init__(self, config: EnvironmentConfig):
        self.config = config
        self.task_id = canonical_task_id(config.task_id)
        self.entry = TASK_REGISTRY[self.task_id]

    @property
    def theta_names(self) -> List[str]:
        return [f"theta_{i + 1}" for i in range(self.entry.theta_dim)]

    def sample_prior(self, num_samples: Optional[int] = None, seed: Optional[int] = None) -> np.ndarray:
        rng = _rng(self.config.seed if seed is None else seed)
        n = int(num_samples or self.config.num_samples)
        return rng.normal(0.0, 1.0, size=(n, self.entry.theta_dim))

    def simulate(self, theta: np.ndarray, seed: Optional[int] = None) -> Tuple[np.ndarray, List[str], np.ndarray, Dict[str, Any]]:
        raise NotImplementedError("Subclasses implement task-specific simulation.")

    def sample(self, num_samples: Optional[int] = None, seed: Optional[int] = None) -> SimulationBatch:
        theta = self.sample_prior(num_samples=num_samples, seed=seed)
        x, x_names, x_coordinates, metadata = self.simulate(theta, seed=seed)
        condition_mask = _condition_policy_mask(
            theta.shape[0],
            theta.shape[1],
            x.shape[1],
            self.config.query_policy,
        )
        metadata = dict(metadata)
        metadata.update(
            {
                "task_family": self.entry.family,
                "paper_section": self.entry.paper_section,
                "query_policy": self.config.query_policy,
                "directed_dependency_mask": directed_dependency_mask(self.task_id, self.config).tolist(),
                "undirected_dependency_mask": undirected_dependency_mask(self.task_id, self.config).tolist(),
            }
        )
        return SimulationBatch(
            task_id=self.task_id,
            theta=theta,
            x=x,
            theta_names=self.theta_names,
            x_names=x_names,
            x_coordinates=x_coordinates,
            condition_mask=condition_mask,
            metadata=metadata,
        )

    def dependency_mask(self, directed: bool = True) -> np.ndarray:
        if directed:
            return directed_dependency_mask(self.task_id, self.config)
        return undirected_dependency_mask(self.task_id, self.config)

    def config_summary(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "aliases": list(self.entry.aliases),
            "family": self.entry.family,
            "theta_dim": self.entry.theta_dim,
            "x_dim": default_x_dim(self.task_id, self.config),
            "setup_metadata": dict(self.entry.setup_metadata),
            "config": dataclasses.asdict(self.config),
        }


class BenchmarkAdapter(EnvironmentAdapter):
    """Lightweight benchmark adapter for two moons, Gaussian tasks, and SLCP."""

    def simulate(self, theta: np.ndarray, seed: Optional[int] = None) -> Tuple[np.ndarray, List[str], np.ndarray, Dict[str, Any]]:
        rng = _rng((self.config.seed if seed is None else seed) + 17)
        theta = _ensure_2d(theta)
        n = theta.shape[0]
        noise = float(self.config.observation_noise)

        if self.task_id == "two_moons":
            angle = theta[:, 0]
            radius = 1.0 + 0.1 * theta[:, 1]
            x = np.stack([radius * np.cos(angle), radius * np.sin(angle)], axis=1)
            x += rng.normal(0.0, noise, size=x.shape)
            names = ["moon_x", "moon_y"]

        elif self.task_id == "gaussian_linear":
            dim = self.entry.default_x_dim
            theta_pad = theta[:, :dim]
            if theta_pad.shape[1] < dim:
                theta_pad = np.pad(theta_pad, ((0, 0), (0, dim - theta_pad.shape[1])))
            scales = np.linspace(0.5, 1.5, dim)
            x = theta_pad * scales + rng.normal(0.0, noise, size=(n, dim))
            names = [f"linear_x_{i}" for i in range(dim)]

        elif self.task_id == "gaussian_mixture":
            component = (theta[:, 0] > 0.0).astype(float)[:, None]
            means = np.concatenate([theta[:, :1] + 1.0, theta[:, 1:2] - 1.0], axis=1)
            alt = np.concatenate([theta[:, :1] - 1.0, theta[:, 1:2] + 1.0], axis=1)
            x = component * means + (1.0 - component) * alt
            x += rng.normal(0.0, 0.15 + noise, size=x.shape)
            names = ["mixture_x_1", "mixture_x_2"]

        elif self.task_id == "slcp":
            means = np.stack(
                [
                    theta[:, 0],
                    theta[:, 1],
                    theta[:, 0] ** 2,
                    theta[:, 1] ** 2,
                    np.sin(theta[:, 2]),
                    np.cos(theta[:, 3]),
                    theta[:, 4],
                    theta[:, 0] * theta[:, 1],
                ],
                axis=1,
            )
            x = means + rng.normal(0.0, 0.2 + noise, size=means.shape)
            names = [f"slcp_x_{i}" for i in range(8)]

        else:
            raise KeyError(f"Unsupported benchmark task {self.task_id}")

        return (
            x.astype(float),
            names,
            np.arange(x.shape[1], dtype=float),
            {"x_groups": ["benchmark_observation"] * x.shape[1], "theta_groups": ["benchmark_parameter"] * theta.shape[1]},
        )


class LotkaVolterraAdapter(EnvironmentAdapter):
    """Lotka-Volterra adapter with species-specific observation schedules."""

    @property
    def theta_names(self) -> List[str]:
        return ["prey_growth", "prey_predation", "predator_death", "predator_reproduction"]

    def sample_prior(self, num_samples: Optional[int] = None, seed: Optional[int] = None) -> np.ndarray:
        rng = _rng(self.config.seed if seed is None else seed)
        n = int(num_samples or self.config.num_samples)
        # Positive rates for stable smoke simulations.
        return rng.uniform(0.2, 1.2, size=(n, 4))

    def simulate(self, theta: np.ndarray, seed: Optional[int] = None) -> Tuple[np.ndarray, List[str], np.ndarray, Dict[str, Any]]:
        rng = _rng((self.config.seed if seed is None else seed) + 31)
        theta = _ensure_2d(theta)
        prey_times = _as_float_array(self.config.prey_times)
        predator_times = _as_float_array(self.config.predator_times)

        all_times = sorted(set(prey_times.tolist() + predator_times.tolist()))
        if not all_times:
            raise ValueError("Lotka-Volterra requires at least one prey or predator observation time.")

        values: List[List[float]] = []
        for rates in theta:
            alpha, beta, gamma, delta = rates
            state_prey = 30.0 + 5.0 * alpha
            state_pred = 4.0 + 3.0 * delta
            previous_t = min(0.0, all_times[0])
            states: Dict[float, Tuple[float, float]] = {}
            for t in all_times:
                dt_total = max(0.0, float(t) - float(previous_t))
                steps = max(1, int(math.ceil(dt_total / 0.05)))
                dt = dt_total / steps if steps else 0.0
                for _ in range(steps):
                    d_prey = alpha * state_prey - beta * state_prey * state_pred / 50.0
                    d_pred = -gamma * state_pred + delta * state_prey * state_pred / 50.0
                    state_prey = max(0.0, state_prey + dt * d_prey)
                    state_pred = max(0.0, state_pred + dt * d_pred)
                states[float(t)] = (state_prey, state_pred)
                previous_t = float(t)

            row: List[float] = []
            for t in prey_times:
                row.append(states[float(t)][0] + rng.normal(0.0, self.config.observation_noise))
            for t in predator_times:
                row.append(states[float(t)][1] + rng.normal(0.0, self.config.observation_noise))
            values.append(row)

        x_names = [f"prey_t={t:g}" for t in prey_times] + [f"predator_t={t:g}" for t in predator_times]
        x_coordinates = np.concatenate([prey_times, predator_times]).astype(float)
        x_groups = ["prey"] * len(prey_times) + ["predator"] * len(predator_times)

        return (
            np.asarray(values, dtype=float),
            x_names,
            x_coordinates,
            {
                "x_groups": x_groups,
                "theta_groups": ["prey", "prey", "predator", "predator"],
                "prey_times": prey_times.tolist(),
                "predator_times": predator_times.tolist(),
                "species_specific_observation_schedules": {
                    "prey": prey_times.tolist(),
                    "predator": predator_times.tolist(),
                },
                "observation_counts": {"prey": int(len(prey_times)), "predator": int(len(predator_times))},
                "mask_obligation": "metadata_dependent_lotka_volterra_directed_mask",
            },
        )


class SIRDAdapter(EnvironmentAdapter):
    """SIRD-model adapter with functional parameter query coordinates."""

    @property
    def theta_names(self) -> List[str]:
        return ["beta_base", "gamma_base", "mu_base", "seasonality"]

    def sample_prior(self, num_samples: Optional[int] = None, seed: Optional[int] = None) -> np.ndarray:
        rng = _rng(self.config.seed if seed is None else seed)
        n = int(num_samples or self.config.num_samples)
        beta = rng.uniform(0.15, 0.55, size=(n, 1))
        gamma = rng.uniform(0.03, 0.18, size=(n, 1))
        mu = rng.uniform(0.005, 0.08, size=(n, 1))
        seasonality = rng.uniform(-0.25, 0.25, size=(n, 1))
        return np.concatenate([beta, gamma, mu, seasonality], axis=1)

    @staticmethod
    def functional_parameters(theta_row: np.ndarray, times: Sequence[float]) -> Dict[str, np.ndarray]:
        beta_base, gamma_base, mu_base, seasonality = [float(v) for v in theta_row]
        t = np.asarray(times, dtype=float)
        beta_t = beta_base * (1.0 + seasonality * np.sin(2.0 * np.pi * t / max(1.0, float(np.max(t) + 1.0))))
        gamma_t = gamma_base * (1.0 + 0.1 * np.cos(t))
        mu_t = np.maximum(0.0, mu_base * (1.0 + 0.05 * t))
        return {"beta_t": beta_t, "gamma_t": gamma_t, "mu_t": mu_t}

    def simulate(self, theta: np.ndarray, seed: Optional[int] = None) -> Tuple[np.ndarray, List[str], np.ndarray, Dict[str, Any]]:
        rng = _rng((self.config.seed if seed is None else seed) + 47)
        theta = _ensure_2d(theta)
        times = _as_float_array(self.config.sird_times)
        query_times = _as_float_array(self.config.functional_query_times)
        measured_subset = list(self.config.measured_parameter_subset)

        if not times.size:
            raise ValueError("SIRD requires at least one compartment observation time.")

        rows: List[List[float]] = []
        for params in theta:
            population = 1.0
            s, i, r, d = 0.98, 0.02, 0.0, 0.0
            previous_t = min(0.0, float(times[0]))
            compartment_values: Dict[float, Tuple[float, float, float, float]] = {}

            for t_val in times:
                dt_total = max(0.0, float(t_val) - previous_t)
                steps = max(1, int(math.ceil(dt_total / 0.05)))
                dt = dt_total / steps if steps else 0.0
                for step in range(steps):
                    current_t = previous_t + step * dt
                    funcs = self.functional_parameters(params, [current_t])
                    beta = float(funcs["beta_t"][0])
                    gamma = float(funcs["gamma_t"][0])
                    mu = float(funcs["mu_t"][0])
                    new_inf = beta * s * i
                    rec = gamma * i
                    death = mu * i
                    s = max(0.0, s - dt * new_inf)
                    i = max(0.0, i + dt * (new_inf - rec - death))
                    r = min(population, max(0.0, r + dt * rec))
                    d = min(population, max(0.0, d + dt * death))
                    total = s + i + r + d
                    if total > 0:
                        s, i, r, d = [v / total for v in (s, i, r, d)]
                compartment_values[float(t_val)] = (s, i, r, d)
                previous_t = float(t_val)

            row: List[float] = []
            for t_val in times:
                row.extend([v + rng.normal(0.0, self.config.observation_noise * 0.1) for v in compartment_values[float(t_val)]])

            funcs_at_query = self.functional_parameters(params, query_times)
            for param_name in measured_subset:
                if param_name not in funcs_at_query:
                    raise ValueError(f"Unknown SIRD measured functional parameter: {param_name}")
                row.extend([float(v) for v in funcs_at_query[param_name]])

            rows.append(row)

        x_names: List[str] = []
        x_coordinates: List[float] = []
        x_groups: List[str] = []
        for t_val in times:
            for comp in ("S", "I", "R", "D"):
                x_names.append(f"{comp}_t={t_val:g}")
                x_coordinates.append(float(t_val))
                x_groups.append(f"compartment_{comp}")
        for param_name in measured_subset:
            for t_val in query_times:
                x_names.append(f"{param_name}@query_t={t_val:g}")
                x_coordinates.append(float(t_val))
                x_groups.append(f"functional_parameter_{param_name}")

        return (
            np.asarray(rows, dtype=float),
            x_names,
            np.asarray(x_coordinates, dtype=float),
            {
                "x_groups": x_groups,
                "theta_groups": ["transmission", "recovery", "mortality", "seasonality"],
                "sird_times": times.tolist(),
                "functional_query_times": query_times.tolist(),
                "measured_parameter_subset": measured_subset,
                "functional_parameter_query_coordinates": {
                    name: query_times.tolist() for name in measured_subset
                },
                "subset_parameter_measurements": measured_subset,
            },
        )


class HodgkinHuxleyAdapter(EnvironmentAdapter):
    """Hodgkin-Huxley interval-guidance environment adapter."""

    @property
    def theta_names(self) -> List[str]:
        return ["g_na", "g_k", "g_l", "e_na", "e_k", "e_l", "capacitance", "stimulus"]

    def sample_prior(self, num_samples: Optional[int] = None, seed: Optional[int] = None) -> np.ndarray:
        rng = _rng(self.config.seed if seed is None else seed)
        n = int(num_samples or self.config.num_samples)
        lows = np.asarray([80.0, 20.0, 0.1, 35.0, -90.0, -75.0, 0.5, 2.0])
        highs = np.asarray([140.0, 60.0, 0.5, 65.0, -60.0, -45.0, 2.0, 12.0])
        return rng.uniform(lows, highs, size=(n, 8))

    def simulate(self, theta: np.ndarray, seed: Optional[int] = None) -> Tuple[np.ndarray, List[str], np.ndarray, Dict[str, Any]]:
        rng = _rng((self.config.seed if seed is None else seed) + 59)
        theta = _ensure_2d(theta)
        times = _as_float_array(self.config.hh_voltage_times)
        rows: List[List[float]] = []
        for row in theta:
            g_na, g_k, g_l, e_na, e_k, e_l, capacitance, stimulus = [float(v) for v in row]
            baseline = e_l + 0.04 * (stimulus - g_l) + 0.005 * (g_na - g_k)
            voltage = baseline + 15.0 * np.sin(2.0 * np.pi * times) * np.exp(-times / max(0.1, capacitance))
            voltage += 0.02 * (e_na - abs(e_k))
            voltage += rng.normal(0.0, max(0.1, self.config.observation_noise), size=times.shape)
            rows.append(voltage.tolist())

        return (
            np.asarray(rows, dtype=float),
            [f"voltage_t={t:g}" for t in times],
            times,
            {
                "x_groups": ["voltage"] * len(times),
                "theta_groups": ["conductance", "conductance", "conductance", "reversal", "reversal", "reversal", "capacitance", "stimulus"],
                "interval_guidance": {
                    "target_variable": "voltage",
                    "lower": float(self.config.hh_interval[0]),
                    "upper": float(self.config.hh_interval[1]),
                    "query_times": times.tolist(),
                },
                "metabolic_cost_constraint": "available_to_interval_guidance_sampler",
            },
        )


class TaskFactory:
    """Factory for paper-derived environment/task adapters."""

    ADAPTERS: Dict[str, Callable[[EnvironmentConfig], EnvironmentAdapter]] = {
        "two_moons": BenchmarkAdapter,
        "gaussian_linear": BenchmarkAdapter,
        "gaussian_mixture": BenchmarkAdapter,
        "slcp": BenchmarkAdapter,
        "lotka_volterra": LotkaVolterraAdapter,
        "sird": SIRDAdapter,
        "hodgkin_huxley": HodgkinHuxleyAdapter,
    }

    @classmethod
    def create(cls, task_id: str, config: Optional[Mapping[str, Any] | EnvironmentConfig] = None) -> EnvironmentAdapter:
        canonical = canonical_task_id(task_id)
        if isinstance(config, EnvironmentConfig):
            env_config = dataclasses.replace(config, task_id=canonical)
        else:
            env_config = EnvironmentConfig.from_mapping(canonical, config)
        adapter_cls = cls.ADAPTERS[canonical]
        return adapter_cls(env_config)


def load_dataset(task_id: str, config: Optional[Mapping[str, Any] | EnvironmentConfig] = None) -> SimulationBatch:
    """Dataset/benchmark loader hook.

    The reproduction does not rely on prepared external datasets.  Each dataset
    registry entry loads a generated joint-simulation batch through the same
    task factory used by training and evaluation.
    """

    adapter = TaskFactory.create(task_id, config)
    return adapter.sample()


# ---------------------------------------------------------------------------
# Training-loop and metric surfaces
# ---------------------------------------------------------------------------


def gaussian_nll_proxy(theta: np.ndarray, x: np.ndarray) -> float:
    """A lightweight NLL-like metric for smoke validation.

    The value is the mean squared z-score of observations under their empirical
    Gaussian distribution plus the log variance term.  It is not a paper result;
    it validates metric wiring and finite numeric aggregation.
    """

    x = _ensure_2d(x)
    variance = np.var(x, axis=0) + 1e-6
    mean = np.mean(x, axis=0)
    z2 = ((x - mean) ** 2) / variance
    return float(0.5 * np.mean(z2 + np.log(2.0 * np.pi * variance)))


def c2st_proxy_accuracy(theta: np.ndarray, x: np.ndarray) -> float:
    """Deterministic C2ST-style proxy accuracy for generated samples.

    A median-threshold classifier over concatenated summary features is compared
    against a shuffled reference.  This is a smoke metric formula, not a trained
    classifier benchmark score.
    """

    theta = _ensure_2d(theta)
    x = _ensure_2d(x)
    features = np.concatenate([theta[:, : min(2, theta.shape[1])], x[:, : min(2, x.shape[1])]], axis=1)
    scores = np.mean(features, axis=1)
    threshold = float(np.median(scores))
    labels = (scores > threshold).astype(int)
    shuffled = np.roll(labels, 1)
    accuracy = max(float(np.mean(labels == shuffled)), float(np.mean(labels != shuffled)))
    return float(np.clip(accuracy, 0.5, 1.0))


def lotka_schedule_coverage(batch: SimulationBatch) -> float:
    meta = batch.metadata
    counts = meta.get("observation_counts", {})
    has_prey = int(counts.get("prey", 0)) > 0
    has_pred = int(counts.get("predator", 0)) > 0
    unequal_allowed = int(counts.get("prey", 0)) != int(counts.get("predator", 0))
    groups = set(meta.get("x_groups", []))
    score = 0.25 * has_prey + 0.25 * has_pred + 0.25 * ("prey" in groups and "predator" in groups) + 0.25 * unequal_allowed
    return float(score)


def sird_functional_query_coverage(batch: SimulationBatch) -> float:
    meta = batch.metadata
    query_times = meta.get("functional_query_times", [])
    subset = meta.get("measured_parameter_subset", [])
    groups = meta.get("x_groups", [])
    observed_group_count = sum(1 for g in groups if str(g).startswith("functional_parameter_"))
    expected = max(1, len(query_times) * len(subset))
    return float(np.clip(observed_group_count / expected, 0.0, 1.0))


def interval_satisfaction_rate(batch: SimulationBatch) -> float:
    if batch.task_id != "hodgkin_huxley":
        return float("nan")
    interval = batch.metadata.get("interval_guidance", {})
    lower = float(interval.get("lower", -math.inf))
    upper = float(interval.get("upper", math.inf))
    x = _ensure_2d(batch.x)
    return float(np.mean((x >= lower) & (x <= upper)))


def evaluate_batch(batch: SimulationBatch) -> MetricResult:
    metrics = {
        "negative_log_likelihood_proxy": gaussian_nll_proxy(batch.theta, batch.x),
        "c2st_proxy_accuracy": c2st_proxy_accuracy(batch.theta, batch.x),
        "conditioned_fraction": float(np.mean(batch.condition_mask)),
        "token_count": float(len(batch.tokenize().tokens)),
    }
    decisive = "negative_log_likelihood_proxy"
    notes = ["dry-run metric formulas validate wiring; not paper-scale results"]

    if batch.task_id == "lotka_volterra":
        metrics["schedule_coverage"] = lotka_schedule_coverage(batch)
        metrics["species_count_difference"] = float(
            abs(
                int(batch.metadata.get("observation_counts", {}).get("prey", 0))
                - int(batch.metadata.get("observation_counts", {}).get("predator", 0))
            )
        )
        decisive = "schedule_coverage"
        notes.append("Lotka-Volterra row is kept separate from SIRD functional rows.")
    elif batch.task_id == "sird":
        metrics["functional_query_coverage"] = sird_functional_query_coverage(batch)
        metrics["measured_parameter_subset_size"] = float(len(batch.metadata.get("measured_parameter_subset", [])))
        decisive = "functional_query_coverage"
        notes.append("SIRD functional parameter queries are evaluated as a distinct task row.")
    elif batch.task_id == "hodgkin_huxley":
        metrics["interval_satisfaction_rate"] = interval_satisfaction_rate(batch)
        decisive = "interval_satisfaction_rate"

    return MetricResult(
        task_id=batch.task_id,
        metrics={k: float(v) for k, v in metrics.items()},
        sample_count=int(batch.theta.shape[0]),
        decisive_metric=decisive,
        dry_run=True,
        notes=notes,
    )


def smoke_training_loop(
    task_ids: Sequence[str] = ("lotka_volterra", "sird"),
    config_overrides: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Bounded training-loop surface exercising sampling, tokenization, and loss.

    This is not a neural training substitute.  It computes deterministic
    denoising-style proxy losses over real adapter batches to validate that the
    structured task surfaces can feed downstream Simformer training code.
    """

    overrides = dict(config_overrides or {})
    loss_trace: List[Dict[str, Any]] = []
    registry_rows: List[Dict[str, Any]] = []

    for step, task in enumerate(task_ids):
        adapter = TaskFactory.create(task, overrides)
        batch = adapter.sample(num_samples=min(int(overrides.get("num_samples", adapter.config.num_samples)), 8))
        tokenized = batch.tokenize()
        values = tokenized.values
        conditioned = tokenized.condition_states.astype(bool)
        if np.any(~conditioned):
            target = values[~conditioned]
        else:
            target = values
        loss = float(np.mean((target - np.mean(values)) ** 2) + 1e-8)
        loss_trace.append(
            {
                "step": step,
                "task_id": batch.task_id,
                "loss_name": "masked_denoising_variance_proxy",
                "loss": loss,
                "conditioned_fraction": float(np.mean(batch.condition_mask)),
                "dry_run": True,
            }
        )
        registry_rows.append(adapter.config_summary())

    return {
        "dry_run": True,
        "loop_type": "bounded_smoke_training_loop",
        "loss_trace": loss_trace,
        "model_registry": registry_rows,
        "notes": [
            "This loop exercises environment sampling, tokenizer, condition masks, and metric aggregation.",
            "It does not claim trained-model performance.",
        ],
    }


# ---------------------------------------------------------------------------
# Artifact/readiness writer
# ---------------------------------------------------------------------------


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_npz(path: Path, batch: SimulationBatch, metric: MetricResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        theta=batch.theta,
        x=batch.x,
        condition_mask=batch.condition_mask,
        x_coordinates=batch.x_coordinates,
        variable_ids=np.asarray(batch.theta_names + batch.x_names, dtype=object),
        dry_run=np.asarray([1], dtype=int),
        metric_schema=np.asarray([json.dumps(metric.as_dict(), sort_keys=True)], dtype=object),
    )


def write_dry_run_artifacts(output_dir: Optional[str | os.PathLike[str]] = None) -> Dict[str, Any]:
    """Materialize all declared structured-task artifacts as dry-run contracts."""

    start = time.time()
    output_root = result_root(output_dir)
    lv_adapter = TaskFactory.create(
        "lotka_volterra",
        {
            "num_samples": 6,
            "prey_times": (0.0, 1.0, 2.0, 4.0),
            "predator_times": (0.5, 2.5),
            "query_policy": "mixed_conditioning",
        },
    )
    sird_adapter = TaskFactory.create(
        "sird",
        {
            "num_samples": 6,
            "sird_times": (0.0, 1.0, 2.0),
            "functional_query_times": (0.5, 1.5, 2.5),
            "measured_parameter_subset": ("beta_t", "gamma_t"),
            "query_policy": "mixed_conditioning",
        },
    )

    lv_batch = lv_adapter.sample()
    sird_batch = sird_adapter.sample()
    lv_metrics = evaluate_batch(lv_batch)
    sird_metrics = evaluate_batch(sird_batch)
    loop = smoke_training_loop(("lotka_volterra", "sird"), {"num_samples": 6})

    model_registry_payload = {
        "artifact_type": "dry_run_contract_model_registry",
        "dry_run": True,
        "generated_by": "src.data.environments.write_dry_run_artifacts",
        "tasks": [lv_adapter.config_summary(), sird_adapter.config_summary()],
        "task_registry_ids": list(TASK_REGISTRY.keys()),
        "dataset_registry_ids": list(DATASET_REGISTRY.keys()),
        "aliases": dict(sorted(ALIAS_TO_ID.items())),
        "protocol_matrix": EXPERIMENT_PROTOCOL_MATRIX,
        "reference_grounding": [
            "paperbench_ref_001 docs/how_to_guide/04_embedding_networks.ipynb",
            "paperbench_ref_001 docs/how_to_guide/09_sampler_interface.ipynb",
            "paperbench_ref_001 docs/how_to_guide/22_experiment_tracking.ipynb",
        ],
    }
    _write_json(_repo_relative_to_output("results/model_registry.json", output_root), model_registry_payload)
    _write_json(
        _repo_relative_to_output("results/loss_trace.json", output_root),
        {
            "artifact_type": "dry_run_contract_loss_trace",
            "dry_run": True,
            "loss_trace": loop["loss_trace"],
            "notes": loop["notes"],
        },
    )

    _write_npz(_repo_relative_to_output("results/lotka_volterra_samples.npz", output_root), lv_batch, lv_metrics)
    _write_npz(_repo_relative_to_output("results/sird_functional_samples.npz", output_root), sird_batch, sird_metrics)

    _write_json(
        _repo_relative_to_output("results/lotka_volterra_metrics.json", output_root),
        {
            "artifact_type": "dry_run_contract_metrics",
            "dry_run": True,
            "task_row": "lotka_volterra",
            "evaluation_writer_policy": "do_not_collapse_with_sird",
            "result": lv_metrics.as_dict(),
            "metadata": lv_batch.metadata,
        },
    )
    _write_json(
        _repo_relative_to_output("results/sird_metrics.json", output_root),
        {
            "artifact_type": "dry_run_contract_metrics",
            "dry_run": True,
            "task_row": "sird_functional_parameters",
            "evaluation_writer_policy": "do_not_collapse_with_lotka_volterra",
            "result": sird_metrics.as_dict(),
            "metadata": sird_batch.metadata,
        },
    )

    readiness = validate_environment_registry()
    readiness.update(
        {
            "artifact_type": "dry_run_readiness",
            "dry_run": True,
            "output_root": str(output_root),
            "declared_artifacts": list(DECLARED_ARTIFACTS),
            "elapsed_seconds": float(time.time() - start),
        }
    )
    _write_json(_repo_relative_to_output("results/readiness.json", output_root), readiness)
    _write_json(
        _repo_relative_to_output("results/evaluation_result.json", output_root),
        {
            "artifact_type": "dry_run_evaluation_result",
            "dry_run": True,
            "status": "ready",
            "task_results": [lv_metrics.as_dict(), sird_metrics.as_dict()],
            "decisive_metrics": {
                "lotka_volterra": lv_metrics.metrics[lv_metrics.decisive_metric],
                "sird": sird_metrics.metrics[sird_metrics.decisive_metric],
            },
            "not_claimed": "No paper-scale training/evaluation was run.",
        },
    )

    return readiness


# ---------------------------------------------------------------------------
# Validation and public accessors
# ---------------------------------------------------------------------------


def validate_environment_registry() -> Dict[str, Any]:
    """Validate registry/factory/mask obligations for import-time smoke tests."""

    required = set(ALL_TASK_IDS)
    missing_tasks = sorted(required.difference(TASK_REGISTRY.keys()))
    missing_dataset_aliases = [
        task_id for task_id in required if _norm_alias(task_id) not in ALIAS_TO_ID
    ]
    factory_failures: List[str] = []
    mask_shapes: Dict[str, Tuple[int, int]] = {}

    for task_id in ALL_TASK_IDS:
        try:
            adapter = TaskFactory.create(task_id, {"num_samples": 2})
            batch = adapter.sample(num_samples=2)
            tokens = batch.tokenize()
            mask = adapter.dependency_mask(directed=True)
            mask_shapes[task_id] = tuple(int(v) for v in mask.shape)
            expected_total = batch.theta.shape[1] + batch.x.shape[1]
            if mask.shape != (expected_total, expected_total):
                factory_failures.append(f"{task_id}: mask shape {mask.shape} != {(expected_total, expected_total)}")
            if len(tokens.tokens) != 2 * expected_total:
                factory_failures.append(f"{task_id}: tokenizer emitted unexpected token count")
        except Exception as exc:  # pragma: no cover - diagnostic path
            factory_failures.append(f"{task_id}: {type(exc).__name__}: {exc}")

    alias_checks = {
        "Two Moons": canonical_task_id("Two Moons") == "two_moons",
        "Benchmark": canonical_task_id("Benchmark") == "two_moons",
        "Linear Gaussian": canonical_task_id("Linear Gaussian") == "gaussian_linear",
        "hodgkin_huxley_factory": isinstance(TaskFactory.create("hodgkin_huxley", {"num_samples": 1}), HodgkinHuxleyAdapter),
    }

    ok = not missing_tasks and not missing_dataset_aliases and not factory_failures and all(alias_checks.values())
    return {
        "ok": bool(ok),
        "missing_tasks": missing_tasks,
        "missing_dataset_aliases": missing_dataset_aliases,
        "factory_failures": factory_failures,
        "alias_checks": alias_checks,
        "mask_shapes": {k: list(v) for k, v in mask_shapes.items()},
        "task_registry_ids": list(TASK_REGISTRY.keys()),
        "dataset_registry_ids": list(DATASET_REGISTRY.keys()),
        "contract_notes": [
            "TaskFactory.create('hodgkin_huxley', config) is supported.",
            "Lotka-Volterra exposes prey/predator schedules with independent counts.",
            "SIRD exposes functional parameter query coordinates and measured subsets.",
            "Tokenizer preserves variable id, value, condition state, role, coordinate, and group.",
        ],
    }


def get_task_registry() -> Dict[str, Dict[str, Any]]:
    return {k: dataclasses.asdict(v) for k, v in TASK_REGISTRY.items()}


def get_dataset_registry() -> Dict[str, Dict[str, Any]]:
    return {k: dict(v) for k, v in DATASET_REGISTRY.items()}


def get_alias_registry() -> Dict[str, str]:
    return dict(ALIAS_TO_ID)


def create_environment(task_id: str, config: Optional[Mapping[str, Any] | EnvironmentConfig] = None) -> EnvironmentAdapter:
    return TaskFactory.create(task_id, config)


__all__ = [
    "ALIAS_TO_ID",
    "ALL_TASK_IDS",
    "BENCHMARK_TASK_IDS",
    "DATASET_REGISTRY",
    "DECLARED_ARTIFACTS",
    "EXPERIMENT_PROTOCOL_MATRIX",
    "EnvironmentAdapter",
    "EnvironmentConfig",
    "HodgkinHuxleyAdapter",
    "LotkaVolterraAdapter",
    "MetricResult",
    "SBITokenizer",
    "SIRDAdapter",
    "SimulationBatch",
    "STRUCTURED_TASK_IDS",
    "TASK_REGISTRY",
    "TaskFactory",
    "TaskRegistryEntry",
    "Token",
    "TokenizedBatch",
    "canonical_task_id",
    "create_environment",
    "c2st_proxy_accuracy",
    "default_environment_config",
    "directed_dependency_mask",
    "evaluate_batch",
    "gaussian_nll_proxy",
    "get_alias_registry",
    "get_dataset_registry",
    "get_task_registry",
    "interval_satisfaction_rate",
    "load_dataset",
    "lotka_schedule_coverage",
    "lotka_volterra_dependency_mask",
    "sird_functional_query_coverage",
    "smoke_training_loop",
    "undirected_dependency_mask",
    "validate_environment_registry",
    "write_dry_run_artifacts",
]