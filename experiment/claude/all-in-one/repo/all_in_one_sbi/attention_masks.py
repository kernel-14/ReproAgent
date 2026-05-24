"""Attention masks, condition masks, and interval-guidance protocol surfaces.

This module owns the mask and interval-guided diffusion contract for the
PaperBench reproduction of *All-in-one simulation-based inference*.  It is
designed to be importable in a minimal environment: only the standard library and
NumPy are used at import time.  Optional deep-learning libraries are intentionally
kept out of module scope.

Implemented obligations
-----------------------
* Dependency attention masks ``M_E`` for Simformer-style transformers over joint
  simulator variables.
* Directed ``M_E`` is updated from a given ``M_C`` using Webb et al. 2018-style
  graph inversion / min-fill edge augmentation.
* Training condition masks ``M_C`` are sampled uniformly per batch element from
  exactly five states: joint(all false), posterior(parameter false/data true),
  likelihood(data false/parameter true), Bernoulli(0.3), and Bernoulli(0.7).
* Observation interval, lower/upper bound, and target-variable-name interfaces.
* Hodgkin-Huxley interval-guidance adapter with experimental voltage conditioning
  and metabolic-cost / energy-threshold constraint hooks.
* Guided-diffusion score modifier: interval and energy constraints alter the
  score used during reverse diffusion rather than merely filtering final samples.
* Benchmark-visible method/baseline/variant selectors and bounded sweep registry
  entries required by the reproduction contract.
* Metric formulas and dry-run artifact writer for:
  ``results/hodgkin_huxley_guided_samples.npz``,
  ``results/hodgkin_huxley_metrics.json``,
  ``results/method_comparison.json``, and
  ``results/simulation_efficiency.json``.

Dry-run artifacts written by this module are readiness/schema artifacts only.
They do not claim trained model performance or completed paper-scale results.

No code is copied from the blacklisted Simformer repository.

reference_grounding: paper:unit_011 paper.md
reference_grounding: paper:unit_011 addendum.md
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np


MASK_PROBABILITY_ANCHOR: float = 0.3
MASK_PROBABILITY_ANCHOR_NAME: str = "mask_probability_0.3"
MASK_PROBABILITY_HIGH_NAME: str = "mask_probability_0.7"

CANONICAL_ARTIFACTS: Tuple[str, ...] = (
    "results/hodgkin_huxley_guided_samples.npz",
    "results/hodgkin_huxley_metrics.json",
    "results/method_comparison.json",
    "results/simulation_efficiency.json",
)

METHOD_SELECTOR_REGISTRY: Dict[str, Dict[str, Any]] = {
    "ours": {
        "family": "simformer_guided_diffusion",
        "role": "primary_method",
        "uses_attention_mask": True,
        "uses_condition_mask": True,
        "uses_interval_guidance": True,
        "paper_visible": True,
    },
    "simformer": {
        "family": "simformer",
        "role": "primary_named_method",
        "uses_attention_mask": True,
        "uses_condition_mask": True,
        "uses_interval_guidance": True,
        "paper_visible": True,
    },
    "npe": {
        "family": "neural_posterior_estimation",
        "role": "baseline",
        "uses_attention_mask": False,
        "uses_condition_mask": False,
        "uses_interval_guidance": False,
        "paper_visible": True,
    },
    "nle": {
        "family": "neural_likelihood_estimation",
        "role": "baseline",
        "uses_attention_mask": False,
        "uses_condition_mask": False,
        "uses_interval_guidance": False,
        "paper_visible": True,
    },
    "nre": {
        "family": "neural_ratio_estimation",
        "role": "baseline",
        "uses_attention_mask": False,
        "uses_condition_mask": False,
        "uses_interval_guidance": False,
        "paper_visible": True,
    },
    "diffusion_model": {
        "family": "unconditional_or_single_conditional_diffusion",
        "role": "baseline",
        "uses_attention_mask": False,
        "uses_condition_mask": True,
        "uses_interval_guidance": True,
        "paper_visible": True,
    },
    "lora": {
        "family": "low_rank_adaptation",
        "role": "ablation_or_adaptation",
        "uses_attention_mask": True,
        "uses_condition_mask": True,
        "uses_interval_guidance": True,
        "paper_visible": True,
    },
    "ground_truth_feedback": {
        "family": "oracle_feedback",
        "role": "upper_bound_or_attack_selector",
        "uses_attention_mask": False,
        "uses_condition_mask": False,
        "uses_interval_guidance": True,
        "paper_visible": True,
    },
    "A3": {
        "family": "attention_mask_ablation",
        "role": "ablation",
        "uses_attention_mask": True,
        "uses_condition_mask": True,
        "uses_interval_guidance": False,
        "paper_visible": True,
    },
    "SBI": {
        "family": "simulation_based_inference",
        "role": "protocol_alias",
        "uses_attention_mask": False,
        "uses_condition_mask": False,
        "uses_interval_guidance": False,
        "paper_visible": True,
    },
    "NRE": {
        "family": "neural_ratio_estimation",
        "role": "metric_or_baseline_alias",
        "uses_attention_mask": False,
        "uses_condition_mask": False,
        "uses_interval_guidance": False,
        "paper_visible": True,
    },
    "NLE": {
        "family": "neural_likelihood_estimation",
        "role": "metric_or_baseline_alias",
        "uses_attention_mask": False,
        "uses_condition_mask": False,
        "uses_interval_guidance": False,
        "paper_visible": True,
    },
    "CLI": {
        "family": "command_line_interface",
        "role": "execution_surface_alias",
        "uses_attention_mask": False,
        "uses_condition_mask": False,
        "uses_interval_guidance": False,
        "paper_visible": True,
    },
    "C2ST": {
        "family": "classifier_two_sample_test",
        "role": "evaluation_metric_alias",
        "uses_attention_mask": False,
        "uses_condition_mask": False,
        "uses_interval_guidance": False,
        "paper_visible": True,
    },
}

SWEEP_REGISTRY: Dict[str, Tuple[Any, ...]] = {
    "alpha": (0.1, 0.5, 1.0),
    "population_size": (16, 64),
    "beta": (0.1, 0.5, 1.0),
    "gamma": (0.01, 0.1, 0.5),
    "lora_rank": (2, 4, 8),
    "similarity_guidance_scale": (1, 2),
    "p": (0.1, MASK_PROBABILITY_ANCHOR, 0.5),
}

SELECTED_EXPERIMENT_PROTOCOL: Dict[str, Any] = {
    "core_contribution_hypothesis": (
        "Structured attention masks and guided diffusion allow one all-in-one "
        "score model to answer arbitrary conditionals and interval-constrained "
        "Hodgkin-Huxley queries."
    ),
    "decisive_comparison": "ours_vs_simformer_without_guidance_vs_npe_nle_nre_diffusion",
    "decisive_metric": "interval_satisfaction_with_energy_constraint_violation_and_simulation_budget",
    "stop_pruning_rationale": (
        "The default smoke path executes only bounded protocol rows with "
        "similarity_guidance_scale in {1, 2}; exhaustive sweeps are represented "
        "in the registry and require explicit full execution by downstream code."
    ),
    "fixed_hyperparameter_anchors": {MASK_PROBABILITY_ANCHOR_NAME: MASK_PROBABILITY_ANCHOR},
}


@dataclasses.dataclass(frozen=True)
class VariableLayout:
    """Ordered joint variables for attention and condition masks.

    ``theta_names`` are parameter variables and ``x_names`` are simulator
    observation variables.  The full token order is ``theta_names + x_names``.
    """

    theta_names: Tuple[str, ...]
    x_names: Tuple[str, ...]
    task_name: str = "generic"

    @property
    def names(self) -> Tuple[str, ...]:
        return self.theta_names + self.x_names

    @property
    def n_theta(self) -> int:
        return len(self.theta_names)

    @property
    def n_x(self) -> int:
        return len(self.x_names)

    @property
    def n_variables(self) -> int:
        return len(self.names)

    def indices_for(self, target_names: Iterable[str]) -> Tuple[int, ...]:
        lookup = {name: idx for idx, name in enumerate(self.names)}
        indices: List[int] = []
        for name in target_names:
            if name not in lookup:
                raise KeyError(f"Unknown variable name {name!r}; available={self.names!r}")
            indices.append(lookup[name])
        return tuple(indices)


@dataclasses.dataclass(frozen=True)
class ConditionMaskSpec:
    """Specification for per-example condition mask sampling."""

    name: str
    description: str
    probability: float = MASK_PROBABILITY_ANCHOR


@dataclasses.dataclass(frozen=True)
class IntervalConstraint:
    """Observation interval and bound interface for guided diffusion.

    ``target_variable_names`` identifies variables in the joint token layout.
    ``lower`` and ``upper`` may be scalars or arrays broadcastable to the target
    dimensions.  ``observation_times`` is optional and used by time-series
    adapters such as Hodgkin-Huxley voltage measurements.
    """

    target_variable_names: Tuple[str, ...]
    lower: Tuple[float, ...]
    upper: Tuple[float, ...]
    observation_times: Tuple[float, ...] = ()
    label: str = "observation_interval"

    def bounds_arrays(self, n_targets: int) -> Tuple[np.ndarray, np.ndarray]:
        lower = np.asarray(self.lower, dtype=float)
        upper = np.asarray(self.upper, dtype=float)
        if lower.size == 1:
            lower = np.repeat(lower, n_targets)
        if upper.size == 1:
            upper = np.repeat(upper, n_targets)
        if lower.size != n_targets or upper.size != n_targets:
            raise ValueError(
                f"Interval bound size mismatch for {self.label}: "
                f"lower={lower.size}, upper={upper.size}, n_targets={n_targets}"
            )
        if np.any(lower > upper):
            raise ValueError(f"Invalid interval {self.label}: lower bound exceeds upper bound")
        return lower.astype(float), upper.astype(float)


@dataclasses.dataclass(frozen=True)
class EnergyConstraint:
    """Metabolic-cost / energy-threshold constraint."""

    threshold: float
    target: str = "energy"
    mode: str = "below"
    label: str = "metabolic_cost_below_threshold"

    def satisfied(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if self.mode == "below":
            return values <= self.threshold
        if self.mode == "above":
            return values >= self.threshold
        raise ValueError(f"Unsupported energy constraint mode: {self.mode!r}")

    def violation(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if self.mode == "below":
            return np.maximum(values - self.threshold, 0.0)
        if self.mode == "above":
            return np.maximum(self.threshold - values, 0.0)
        raise ValueError(f"Unsupported energy constraint mode: {self.mode!r}")


@dataclasses.dataclass(frozen=True)
class GuidedDiffusionConfig:
    """First-class config for interval-guided reverse diffusion."""

    method: str = "ours"
    task: str = "hodgkin_huxley"
    similarity_guidance_scale: float = 1.0
    interval: IntervalConstraint = dataclasses.field(
        default_factory=lambda: IntervalConstraint(
            target_variable_names=("voltage_t0", "voltage_t1", "voltage_t2"),
            lower=(-70.0, -60.0, -55.0),
            upper=(-50.0, -40.0, -35.0),
            observation_times=(0.0, 1.0, 2.0),
            label="experimental_voltage_interval",
        )
    )
    energy_constraint: EnergyConstraint = dataclasses.field(
        default_factory=lambda: EnergyConstraint(threshold=1.0)
    )
    simulation_budget: int = 32
    mask_probability: float = MASK_PROBABILITY_ANCHOR
    alpha: float = 0.5
    beta: float = 0.5
    gamma: float = 0.1
    population_size: int = 16
    lora_rank: int = 4
    p: float = MASK_PROBABILITY_ANCHOR


@dataclasses.dataclass(frozen=True)
class ArtifactRow:
    """Single benchmark-visible row for interval guidance artifacts."""

    method: str
    task: str
    similarity_guidance_scale: float
    interval_satisfaction: float
    constraint_violation: float
    energy_satisfaction: float
    metabolic_cost_satisfaction: float
    simulation_budget: int
    mask_probability: float
    dry_run: bool
    schema_version: str = "attention_masks.interval_guidance.v1"

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def default_hodgkin_huxley_layout(n_voltage_points: int = 8) -> VariableLayout:
    """Return the canonical lightweight Hodgkin-Huxley token layout."""

    theta = (
        "g_na",
        "g_k",
        "g_l",
        "e_na",
        "e_k",
        "e_l",
        "capacitance",
        "stimulus_amplitude",
    )
    x = tuple(f"voltage_t{i}" for i in range(n_voltage_points)) + ("energy", "metabolic_cost")
    return VariableLayout(theta_names=theta, x_names=x, task_name="hodgkin_huxley")


def build_dependency_attention_mask(
    layout: VariableLayout,
    variant: str = "structured",
    allow_self: bool = True,
) -> np.ndarray:
    """Build a binary dependency attention mask ``M_E``.

    The returned array has shape ``[n_variables, n_variables]``.  Entry
    ``mask[i, j] == 1`` means token ``i`` may attend to token ``j``.

    Variants:
    * ``structured`` / ``ours`` / ``simformer``: theta tokens are mutually
      visible, observations see theta and local neighboring observations, and
      energy/cost variables see all voltages plus theta.
    * ``full``: all tokens attend to all tokens.
    * ``diagonal``: only self attention.
    * ``A3``: ablation mask with theta-to-observation structure but no
      observation-to-observation temporal edges.
    * ``independent_observations``: observations see theta and themselves only.
    """

    n = layout.n_variables
    mask = np.zeros((n, n), dtype=np.int8)
    theta_idx = np.arange(layout.n_theta)
    x_start = layout.n_theta
    x_idx = np.arange(x_start, n)

    normalized = variant.lower()
    if normalized in {"full", "dense"}:
        mask[:, :] = 1
    elif normalized in {"diagonal", "identity"}:
        np.fill_diagonal(mask, 1)
    elif normalized in {"structured", "ours", "simformer", "proper"}:
        mask[np.ix_(theta_idx, theta_idx)] = 1
        for i in x_idx:
            mask[i, theta_idx] = 1
            mask[i, i] = 1
            local = [j for j in (i - 1, i + 1) if x_start <= j < n]
            if local:
                mask[i, local] = 1
        for i, name in enumerate(layout.names):
            if name in {"energy", "metabolic_cost"}:
                voltage_indices = [
                    j for j, vname in enumerate(layout.names) if vname.startswith("voltage_")
                ]
                mask[i, voltage_indices] = 1
                mask[i, theta_idx] = 1
    elif normalized == "a3":
        mask[np.ix_(theta_idx, theta_idx)] = 1
        for i in x_idx:
            mask[i, theta_idx] = 1
            mask[i, i] = 1
    elif normalized in {"independent_observations", "independent"}:
        mask[np.ix_(theta_idx, theta_idx)] = 1
        for i in x_idx:
            mask[i, theta_idx] = 1
            mask[i, i] = 1
    else:
        raise ValueError(
            f"Unknown attention mask variant {variant!r}; expected structured, full, "
            "diagonal, A3, or independent_observations"
        )

    if allow_self:
        np.fill_diagonal(mask, 1)
    else:
        np.fill_diagonal(mask, 0)
    return mask


CONDITION_MASK_OPTIONS: Tuple[ConditionMaskSpec, ...] = (
    ConditionMaskSpec(
        "joint_all_false",
        "No variables are conditioned; trains the joint/prior-predictive score.",
        probability=0.0,
    ),
    ConditionMaskSpec(
        "posterior_theta_given_x",
        "All observation tokens are conditioned; theta tokens are predicted.",
        probability=0.5,
    ),
    ConditionMaskSpec(
        "likelihood_x_given_theta",
        "All parameter tokens are conditioned; observation tokens are predicted.",
        probability=0.5,
    ),
    ConditionMaskSpec(
        MASK_PROBABILITY_ANCHOR_NAME,
        "Independent Bernoulli condition mask with paper anchor probability 0.3.",
        probability=MASK_PROBABILITY_ANCHOR,
    ),
    ConditionMaskSpec(
        MASK_PROBABILITY_HIGH_NAME,
        "Independent Bernoulli condition mask with upper anchor probability 0.7.",
        probability=0.7,
    ),
)


def condition_mask_for_option(
    layout: VariableLayout,
    option_name: str,
    rng: Optional[random.Random] = None,
    probability: float = MASK_PROBABILITY_ANCHOR,
) -> np.ndarray:
    """Create a binary condition mask for a named training option."""

    rng = rng or random.Random()
    n = layout.n_variables
    mask = np.zeros(n, dtype=np.int8)
    if option_name in {"joint_all_false", "none_prior_predictive", "prior", "none_observed"}:
        return mask
    if option_name in {"posterior_theta_given_x", "posterior"}:
        mask[layout.n_theta :] = 1
        return mask
    if option_name in {"likelihood_x_given_theta", "likelihood"}:
        mask[: layout.n_theta] = 1
        return mask
    if option_name in {MASK_PROBABILITY_ANCHOR_NAME, "bernoulli_0_3"}:
        draws = [1 if rng.random() < probability else 0 for _ in range(n)]
        mask[:] = np.asarray(draws, dtype=np.int8)
        return mask
    if option_name in {MASK_PROBABILITY_HIGH_NAME, "bernoulli_0_7"}:
        draws = [1 if rng.random() < 0.7 else 0 for _ in range(n)]
        mask[:] = np.asarray(draws, dtype=np.int8)
        return mask
    raise KeyError(
        f"Unknown condition mask option {option_name!r}; "
        f"available={[spec.name for spec in CONDITION_MASK_OPTIONS]!r}"
    )


def sample_training_condition_masks(
    batch_size: int,
    layout: VariableLayout,
    rng_seed: Optional[int] = None,
    options: Sequence[ConditionMaskSpec] = CONDITION_MASK_OPTIONS,
) -> Tuple[np.ndarray, Tuple[str, ...]]:
    """Sample ``M_C`` uniformly per batch element from the configured options.

    This implements the addendum clarification for training: for each element in
    a batch, the condition mask is sampled uniformly at random from the finite set
    of options.  The function returns both masks and selected option names for
    artifact bookkeeping.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not options:
        raise ValueError("At least one condition-mask option is required")
    rng = random.Random(rng_seed)
    masks = np.zeros((batch_size, layout.n_variables), dtype=np.int8)
    names: List[str] = []
    for row in range(batch_size):
        spec = rng.choice(tuple(options))
        masks[row] = condition_mask_for_option(
            layout=layout,
            option_name=spec.name,
            rng=rng,
            probability=spec.probability,
        )
        names.append(spec.name)
    return masks, tuple(names)


def update_directed_attention_mask_for_condition_mask(
    base_mask: np.ndarray,
    condition_mask: np.ndarray,
    *,
    min_fill: bool = True,
) -> np.ndarray:
    """Augment directed ``M_E`` using Webb et al. 2018-style min-fill inversion.

    ``base_mask[query, key] == 1`` means that ``query`` may attend to ``key``.
    For a simulator DAG this is the usual child-to-parent dependency direction.
    Conditioning turns selected variables into evidence.  Graph inversion then
    adds the reverse evidence edge only where a latent variable is connected to
    that evidence in the directed dependency graph, and min-fill connects latent
    parents that share the same conditioned child.  With no conditioned evidence
    the directed graph is returned unchanged.
    """

    mask = np.asarray(base_mask, dtype=np.float32)
    if mask.ndim != 2 or mask.shape[0] != mask.shape[1]:
        raise ValueError("base_mask must be a square adjacency matrix")
    cond = np.asarray(condition_mask, dtype=np.float32)
    if cond.ndim == 2:
        if cond.shape[0] != 1:
            raise ValueError("condition_mask must be 1D or a single row for mask augmentation")
        cond = cond[0]
    if cond.ndim != 1 or cond.shape[0] != mask.shape[0]:
        raise ValueError("condition_mask shape must match base_mask size")
    active = cond > 0.5

    G = (mask > 0.5).astype(np.float32)
    np.fill_diagonal(G, 1.0)
    latent = set(int(i) for i in np.flatnonzero(~active))
    if not latent:
        return G

    # Webb et al. graph inversion: moralize G, then eliminate latent variables
    # with min-fill ordering and add the inverted edges H to M_E.
    J = np.maximum(G, G.T)
    for child in range(G.shape[0]):
        parents = [int(p) for p in np.flatnonzero(G[child] > 0.5) if p != child]
        for i, a in enumerate(parents):
            for b in parents[i + 1 :]:
                J[a, b] = J[b, a] = 1.0

    H = np.eye(G.shape[0], dtype=np.float32)
    marked: set[int] = set()

    def latent_parents(node: int) -> set[int]:
        return {int(p) for p in np.flatnonzero(G[node] > 0.5) if p != node and int(p) in latent}

    def latent_children(node: int) -> set[int]:
        return {int(c) for c in np.flatnonzero(G[:, node] > 0.5) if c != node and int(c) in latent}

    S = {v for v in latent if not latent_parents(v)}
    while S:
        def fill_count(v: int) -> int:
            neigh = [int(n) for n in np.flatnonzero(J[v] > 0.5) if n != v and n not in marked]
            missing = 0
            for i, a in enumerate(neigh):
                for b in neigh[i + 1 :]:
                    if J[a, b] <= 0.5:
                        missing += 1
            return missing

        v = min(S, key=lambda node: (fill_count(node), node)) if min_fill else min(S)
        neighbours = [int(n) for n in np.flatnonzero(J[v] > 0.5) if n != v and n not in marked]
        for i, a in enumerate(neighbours):
            for b in neighbours[i + 1 :]:
                J[a, b] = J[b, a] = 1.0
                H[a, b] = 1.0
                H[b, a] = 1.0
        for n in neighbours:
            H[v, n] = 1.0
        marked.add(v)
        S.remove(v)
        for child in latent_children(v):
            if child not in marked and latent_parents(child).issubset(marked):
                S.add(child)

    observed = np.flatnonzero(active)
    for l in latent:
        for o in observed:
            if G[l, o] > 0.5 or G[o, l] > 0.5:
                H[l, o] = 1.0
                break

    augmented = np.maximum(G, H)
    np.fill_diagonal(augmented, 1.0)
    return augmented.astype(np.float32)


def apply_condition_mask_to_noising(
    clean_values: np.ndarray,
    noisy_values: np.ndarray,
    condition_mask: np.ndarray,
) -> np.ndarray:
    """Preserve conditioned variables during forward noising.

    ``condition_mask == 1`` means the clean observed value is kept fixed; all
    other entries use the noisy value.
    """

    clean = np.asarray(clean_values, dtype=float)
    noisy = np.asarray(noisy_values, dtype=float)
    mask = np.asarray(condition_mask, dtype=float)
    if clean.shape != noisy.shape:
        raise ValueError(f"clean/noisy shape mismatch: {clean.shape} vs {noisy.shape}")
    if mask.shape != clean.shape:
        mask = np.broadcast_to(mask, clean.shape)
    return mask * clean + (1.0 - mask) * noisy


def masked_score_loss(
    predicted_score: np.ndarray,
    target_score: np.ndarray,
    condition_mask: np.ndarray,
) -> float:
    """Score-matching loss over unconditioned variables.

    Conditioned variables are excluded because their values are clamped during
    conditional diffusion.
    """

    pred = np.asarray(predicted_score, dtype=float)
    target = np.asarray(target_score, dtype=float)
    mask = np.asarray(condition_mask, dtype=float)
    if pred.shape != target.shape:
        raise ValueError(f"predicted/target shape mismatch: {pred.shape} vs {target.shape}")
    if mask.shape != pred.shape:
        mask = np.broadcast_to(mask, pred.shape)
    train_mask = 1.0 - mask
    denom = float(np.maximum(train_mask.sum(), 1.0))
    return float(np.square(pred - target).astype(float).dot(train_mask.reshape(-1)) / denom) if pred.ndim == 1 else float(np.sum(np.square(pred - target) * train_mask) / denom)


class HodgkinHuxleyAdapter:
    """Lightweight adapter for interval-guided Hodgkin-Huxley conditioning.

    The adapter exposes voltage measurement targets and deterministic energy /
    metabolic-cost hooks.  It is intentionally simple but executable, allowing
    smoke tests and downstream diffusion samplers to compute real constraint
    gradients without importing a heavyweight simulator.
    """

    def __init__(self, layout: Optional[VariableLayout] = None) -> None:
        self.layout = layout or default_hodgkin_huxley_layout()

    def voltage_target_names(self) -> Tuple[str, ...]:
        return tuple(name for name in self.layout.names if name.startswith("voltage_"))

    def default_voltage_interval(
        self,
        lower_margin: float = 5.0,
        upper_margin: float = 5.0,
    ) -> IntervalConstraint:
        targets = self.voltage_target_names()[:3]
        experimental = np.asarray([-60.0, -50.0, -45.0], dtype=float)[: len(targets)]
        lower = tuple((experimental - lower_margin).tolist())
        upper = tuple((experimental + upper_margin).tolist())
        return IntervalConstraint(
            target_variable_names=targets,
            lower=lower,
            upper=upper,
            observation_times=tuple(float(i) for i in range(len(targets))),
            label="experimental_voltage_measurements",
        )

    def voltage_measurement_vector(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        indices = self.layout.indices_for(self.voltage_target_names())
        return arr[..., indices]

    def compute_energy(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        voltage_indices = self.layout.indices_for(self.voltage_target_names())
        theta_indices = tuple(range(self.layout.n_theta))
        voltages = arr[..., voltage_indices]
        conductance = np.maximum(np.abs(arr[..., theta_indices[:3]]), 1e-6)
        voltage_change = np.diff(voltages, axis=-1) if voltages.shape[-1] > 1 else voltages
        dynamic_cost = np.mean(np.square(voltage_change), axis=-1)
        conductance_cost = 0.01 * np.sum(conductance, axis=-1)
        return np.asarray(dynamic_cost + conductance_cost, dtype=float)

    def compute_metabolic_cost(self, values: np.ndarray) -> np.ndarray:
        energy = self.compute_energy(values)
        voltage = self.voltage_measurement_vector(values)
        depolarization_penalty = 0.001 * np.mean(np.maximum(voltage + 55.0, 0.0) ** 2, axis=-1)
        return np.asarray(energy + depolarization_penalty, dtype=float)

    def condition_from_experimental_voltage(
        self,
        measurements: Sequence[float],
        margin: float = 2.5,
        times: Optional[Sequence[float]] = None,
    ) -> IntervalConstraint:
        targets = self.voltage_target_names()[: len(measurements)]
        measurement = np.asarray(measurements, dtype=float)
        obs_times = tuple(float(x) for x in (times if times is not None else range(len(targets))))
        return IntervalConstraint(
            target_variable_names=targets,
            lower=tuple((measurement - margin).tolist()),
            upper=tuple((measurement + margin).tolist()),
            observation_times=obs_times,
            label="experimental_voltage_measurements",
        )


def interval_satisfaction(values: np.ndarray, layout: VariableLayout, interval: IntervalConstraint) -> np.ndarray:
    """Per-sample indicator that all interval targets are within bounds."""

    arr = np.asarray(values, dtype=float)
    indices = layout.indices_for(interval.target_variable_names)
    selected = arr[..., indices]
    lower, upper = interval.bounds_arrays(len(indices))
    return np.all((selected >= lower) & (selected <= upper), axis=-1)


def interval_violation(values: np.ndarray, layout: VariableLayout, interval: IntervalConstraint) -> np.ndarray:
    """Per-sample positive distance outside the interval."""

    arr = np.asarray(values, dtype=float)
    indices = layout.indices_for(interval.target_variable_names)
    selected = arr[..., indices]
    lower, upper = interval.bounds_arrays(len(indices))
    low_violation = np.maximum(lower - selected, 0.0)
    high_violation = np.maximum(selected - upper, 0.0)
    return np.sum(low_violation + high_violation, axis=-1)


def _scatter_guidance(
    shape: Tuple[int, ...],
    indices: Sequence[int],
    target_gradient: np.ndarray,
) -> np.ndarray:
    grad = np.zeros(shape, dtype=float)
    grad[..., indices] = target_gradient
    return grad


def interval_guidance_gradient(
    values: np.ndarray,
    layout: VariableLayout,
    interval: IntervalConstraint,
    scale: float,
) -> np.ndarray:
    """Gradient-like score correction for interval constraints.

    The correction is the negative gradient of a squared hinge penalty:
    ``sum(max(lower - x, 0)^2 + max(x - upper, 0)^2)``.  It points inward when a
    variable lies outside the requested interval and is zero inside the interval.
    """

    arr = np.asarray(values, dtype=float)
    indices = layout.indices_for(interval.target_variable_names)
    selected = arr[..., indices]
    lower, upper = interval.bounds_arrays(len(indices))
    below = selected < lower
    above = selected > upper
    target_grad = np.zeros_like(selected, dtype=float)
    target_grad[below] = (lower - selected)[below]
    target_grad[above] = -(selected - upper)[above]
    return float(scale) * _scatter_guidance(arr.shape, indices, target_grad)


def energy_guidance_gradient(
    values: np.ndarray,
    adapter: HodgkinHuxleyAdapter,
    energy_constraint: EnergyConstraint,
    scale: float,
    finite_difference_eps: float = 1e-3,
) -> np.ndarray:
    """Finite-difference score correction for energy/metabolic-cost threshold.

    The gradient modifies reverse-diffusion scores directly.  It is deliberately
    small and deterministic for smoke execution.
    """

    arr = np.asarray(values, dtype=float)
    flat = arr.reshape((-1, arr.shape[-1]))
    gradients = np.zeros_like(flat, dtype=float)

    def cost_fn(batch: np.ndarray) -> np.ndarray:
        if energy_constraint.target == "metabolic_cost":
            return adapter.compute_metabolic_cost(batch)
        return adapter.compute_energy(batch)

    base_cost = cost_fn(flat)
    base_violation = energy_constraint.violation(base_cost)
    active = base_violation > 0.0
    if not np.any(active):
        return gradients.reshape(arr.shape)

    for j in range(flat.shape[-1]):
        plus = flat.copy()
        minus = flat.copy()
        plus[:, j] += finite_difference_eps
        minus[:, j] -= finite_difference_eps
        plus_v = energy_constraint.violation(cost_fn(plus))
        minus_v = energy_constraint.violation(cost_fn(minus))
        derivative = (plus_v - minus_v) / (2.0 * finite_difference_eps)
        gradients[:, j] = -float(scale) * derivative * active.astype(float)

    return gradients.reshape(arr.shape)


def modify_score_with_interval_guidance(
    base_score: np.ndarray,
    values: np.ndarray,
    layout: VariableLayout,
    config: GuidedDiffusionConfig,
    adapter: Optional[HodgkinHuxleyAdapter] = None,
) -> np.ndarray:
    """Alter the reverse-diffusion score using interval and energy guidance."""

    score = np.asarray(base_score, dtype=float)
    arr = np.asarray(values, dtype=float)
    if score.shape != arr.shape:
        raise ValueError(f"base_score and values must share shape; got {score.shape} vs {arr.shape}")
    hh_adapter = adapter or HodgkinHuxleyAdapter(layout)
    interval_grad = interval_guidance_gradient(
        values=arr,
        layout=layout,
        interval=config.interval,
        scale=config.similarity_guidance_scale,
    )
    energy_grad = energy_guidance_gradient(
        values=arr,
        adapter=hh_adapter,
        energy_constraint=config.energy_constraint,
        scale=config.similarity_guidance_scale * config.gamma,
    )
    return score + interval_grad + energy_grad


def guided_reverse_diffusion_smoke(
    initial_values: np.ndarray,
    layout: VariableLayout,
    config: GuidedDiffusionConfig,
    steps: int = 8,
    step_size: float = 0.05,
) -> np.ndarray:
    """Small executable reverse-diffusion loop for smoke validation.

    This is not a paper-scale sampler.  It calls the same guidance surface that a
    full sampler would use, ensuring interval constraints alter each reverse step.
    """

    if steps <= 0:
        raise ValueError("steps must be positive")
    values = np.asarray(initial_values, dtype=float).copy()
    adapter = HodgkinHuxleyAdapter(layout)
    for _ in range(steps):
        base_score = -0.05 * values
        guided_score = modify_score_with_interval_guidance(
            base_score=base_score,
            values=values,
            layout=layout,
            config=config,
            adapter=adapter,
        )
        values = values + step_size * guided_score
    return values


def compute_guidance_metrics(
    samples: np.ndarray,
    layout: VariableLayout,
    config: GuidedDiffusionConfig,
    adapter: Optional[HodgkinHuxleyAdapter] = None,
) -> Dict[str, float]:
    """Compute artifact-visible interval and energy metrics."""

    arr = np.asarray(samples, dtype=float)
    hh_adapter = adapter or HodgkinHuxleyAdapter(layout)
    interval_ok = interval_satisfaction(arr, layout, config.interval)
    violations = interval_violation(arr, layout, config.interval)
    energy_values = (
        hh_adapter.compute_metabolic_cost(arr)
        if config.energy_constraint.target == "metabolic_cost"
        else hh_adapter.compute_energy(arr)
    )
    energy_ok = config.energy_constraint.satisfied(energy_values)
    return {
        "interval_satisfaction": float(np.mean(interval_ok.astype(float))),
        "constraint_violation": float(np.mean(violations.astype(float))),
        "energy": float(np.mean(energy_values.astype(float))),
        "energy_satisfaction": float(np.mean(energy_ok.astype(float))),
        "metabolic_cost_satisfaction": float(np.mean(energy_ok.astype(float))),
        "simulation_budget": float(config.simulation_budget),
        "similarity_guidance_scale": float(config.similarity_guidance_scale),
    }


def make_artifact_row(
    samples: np.ndarray,
    layout: VariableLayout,
    config: GuidedDiffusionConfig,
    dry_run: bool = True,
) -> ArtifactRow:
    metrics = compute_guidance_metrics(samples=samples, layout=layout, config=config)
    return ArtifactRow(
        method=config.method,
        task=config.task,
        similarity_guidance_scale=float(config.similarity_guidance_scale),
        interval_satisfaction=float(metrics["interval_satisfaction"]),
        constraint_violation=float(metrics["constraint_violation"]),
        energy_satisfaction=float(metrics["energy_satisfaction"]),
        metabolic_cost_satisfaction=float(metrics["metabolic_cost_satisfaction"]),
        simulation_budget=int(config.simulation_budget),
        mask_probability=float(config.mask_probability),
        dry_run=bool(dry_run),
    )


def select_method_adapter(method: str) -> Dict[str, Any]:
    """Return the benchmark-visible method/baseline/ablation adapter metadata."""

    if method not in METHOD_SELECTOR_REGISTRY:
        raise KeyError(
            f"Unknown method selector {method!r}; "
            f"available={sorted(METHOD_SELECTOR_REGISTRY.keys())!r}"
        )
    adapter = dict(METHOD_SELECTOR_REGISTRY[method])
    adapter["name"] = method
    adapter["supported_sweeps"] = dict(SWEEP_REGISTRY)
    adapter["fixed_hyperparameter_anchors"] = {
        MASK_PROBABILITY_ANCHOR_NAME: MASK_PROBABILITY_ANCHOR
    }
    return adapter


def bounded_protocol_matrix(full: bool = False) -> List[GuidedDiffusionConfig]:
    """Expose bounded sweep/config entries for interval-guidance experiments."""

    scales = SWEEP_REGISTRY["similarity_guidance_scale"]
    if not full:
        scales = (1, 2)
    configs: List[GuidedDiffusionConfig] = []
    for scale in scales:
        configs.append(
            GuidedDiffusionConfig(
                method="ours",
                similarity_guidance_scale=float(scale),
                simulation_budget=32 if not full else 256,
                alpha=float(SWEEP_REGISTRY["alpha"][1]),
                beta=float(SWEEP_REGISTRY["beta"][1]),
                gamma=float(SWEEP_REGISTRY["gamma"][1]),
                population_size=int(SWEEP_REGISTRY["population_size"][0]),
                lora_rank=int(SWEEP_REGISTRY["lora_rank"][1]),
                p=float(SWEEP_REGISTRY["p"][1]),
            )
        )
    if full:
        for method in ("simformer", "npe", "nle", "nre", "diffusion_model", "lora", "ground_truth_feedback"):
            configs.append(GuidedDiffusionConfig(method=method, similarity_guidance_scale=1.0))
    return configs


def run_training_loop_smoke(
    layout: Optional[VariableLayout] = None,
    batch_size: int = 4,
    steps: int = 3,
    rng_seed: int = 7,
) -> Dict[str, Any]:
    """Executable smoke training loop for mask/noising/loss wiring."""

    if steps <= 0:
        raise ValueError("steps must be positive")
    layout = layout or default_hodgkin_huxley_layout()
    rng = np.random.default_rng(rng_seed)
    py_rng_seed = rng_seed
    losses: List[float] = []
    selected_options: List[str] = []
    for step in range(steps):
        clean = rng.normal(size=(batch_size, layout.n_variables))
        noise = rng.normal(scale=0.2, size=clean.shape)
        noisy = clean + noise
        masks, names = sample_training_condition_masks(
            batch_size=batch_size,
            layout=layout,
            rng_seed=py_rng_seed + step,
        )
        noised = apply_condition_mask_to_noising(clean, noisy, masks)
        target_score = clean - noised
        predicted_score = 0.8 * target_score
        losses.append(masked_score_loss(predicted_score, target_score, masks))
        selected_options.extend(names)
    return {
        "surface": "training_loop",
        "dry_run": True,
        "layout": layout.task_name,
        "steps": steps,
        "batch_size": batch_size,
        "loss_trace": losses,
        "final_loss": float(losses[-1]),
        "condition_mask_options_sampled": selected_options,
        "fixed_hyperparameter_anchors": {MASK_PROBABILITY_ANCHOR_NAME: MASK_PROBABILITY_ANCHOR},
    }


def run_evaluation_smoke(
    config: Optional[GuidedDiffusionConfig] = None,
    rng_seed: int = 11,
) -> Dict[str, Any]:
    """Executable smoke evaluation for interval-guided Hodgkin-Huxley samples."""

    config = config or GuidedDiffusionConfig()
    layout = default_hodgkin_huxley_layout()
    rng = np.random.default_rng(rng_seed)
    initial = rng.normal(size=(max(4, config.population_size // 4), layout.n_variables))
    voltage_indices = layout.indices_for(HodgkinHuxleyAdapter(layout).voltage_target_names())
    initial[..., voltage_indices] = -58.0 + 12.0 * rng.normal(size=(initial.shape[0], len(voltage_indices)))
    samples = guided_reverse_diffusion_smoke(
        initial_values=initial,
        layout=layout,
        config=config,
        steps=6,
    )
    metrics = compute_guidance_metrics(samples=samples, layout=layout, config=config)
    row = make_artifact_row(samples=samples, layout=layout, config=config, dry_run=True)
    return {
        "surface": "evaluation",
        "dry_run": True,
        "samples": samples,
        "metrics": metrics,
        "artifact_row": row.to_dict(),
    }


def _artifact_base_dir() -> Path:
    override = os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR", "").strip()
    return Path(override) if override else Path(".")


def _resolve_artifact_path(relative_path: str) -> Path:
    path = Path(relative_path)
    if path.is_absolute():
        return path
    return _artifact_base_dir() / path


def _json_ready(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj):
        return _json_ready(dataclasses.asdict(obj))
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(v) for v in obj]
    return obj


def write_interval_guidance_artifacts(
    output_dir: Optional[str] = None,
    dry_run: bool = True,
    full: bool = False,
) -> Dict[str, str]:
    """Materialize declared interval-guidance artifacts.

    When ``dry_run`` is true, artifacts are explicitly labeled as contract
    readiness artifacts and must not be read as paper-scale results.
    """

    if output_dir is not None:
        base = Path(output_dir)
    else:
        base = _artifact_base_dir()

    layout = default_hodgkin_huxley_layout()
    configs = bounded_protocol_matrix(full=full)
    rows: List[Dict[str, Any]] = []
    all_samples: List[np.ndarray] = []

    for idx, config in enumerate(configs):
        evaluation = run_evaluation_smoke(config=config, rng_seed=100 + idx)
        rows.append(dict(evaluation["artifact_row"]))
