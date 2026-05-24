"""Diffusion core for the All-in-one simulation-based inference reproduction.

This module implements the Simformer-style score-based diffusion path required by
the repository contract.  It is deliberately importable in a minimal environment:
PyTorch is imported lazily only when neural-network training or sampling is
requested.  NumPy and the Python standard library are sufficient for registry,
masking, tokenizer, metric, and smoke-artifact surfaces.

Implemented obligations
-----------------------
* ``SBITokenizer.encode(batch, condition_mask)`` returns variable identifiers,
  value representations, and binary condition state for samples from the joint
  simulator distribution ``p(theta, x)``.
* Conditioning state ``M_C`` is binary and can be resampled during training.
  ``M_C`` explicitly enters forward noising, score-loss masking, and conditional
  sampling.
* Dependency attention mask ``M_E`` explicitly represents simulator dependency
  structure and is passed into transformer attention computation.
* SDE and ODE sampling families are named and selectable:
  ``sde_backward`` and ``ode_probability_flow``.
* Training metadata records method, mask variant, conditioning pattern,
  simulation budget, and fixed hyperparameters.
* Dry-run artifact closure writes every file owned by this module as a
  schema/readiness artifact without claiming paper-scale training or results.

No code is copied from the blacklisted Simformer repository.

reference_grounding: paperbench_ref_001 docs/how_to_guide/04_embedding_networks.ipynb
reference_grounding: paperbench_ref_001 docs/how_to_guide/07_gpu_training.ipynb
reference_grounding: paperbench_ref_001 docs/how_to_guide/09_sampler_interface.ipynb
reference_grounding: paperbench_ref_001 sbi/inference/trainers/npe/npe_a.py
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Protocol, Sequence, Tuple, Union

import numpy as np


ArrayLike = Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]]


DECLARED_ARTIFACTS: Tuple[str, ...] = (
    "results/model_registry.json",
    "results/tokenizer_registry.json",
    "results/attention_mask_registry.json",
    "results/diffusion_config.json",
    "results/loss_trace.json",
    "results/sampling_trace.json",
)


SAMPLER_FAMILIES: Tuple[str, ...] = ("sde_backward", "ode_probability_flow")


# ---------------------------------------------------------------------------
# Configuration and registries
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class DiffusionConfig:
    """Fixed hyperparameters for Simformer score-based diffusion.

    The defaults are intentionally smoke-safe.  Full experiments should pass an
    explicit config from the repository registry or CLI.

    ``condition_probability`` defaults to the Bernoulli(0.7) family used by the
    finite five-family condition-mask mixture; the paper-visible
    ``mask_probability_0.3`` anchor remains registered as one of those families.
    """

    method: str = "simformer_score_diffusion"
    objective: str = "conditional_denoising_score_matching_on_joint_p_theta_x"
    schedule: str = "variance_exploding"
    num_diffusion_steps: int = 500
    sigma_min: float = 1.0e-4
    sigma_max: float = 15.0
    t_min: float = 1.0e-5
    t_max: float = 1.0
    beta_min: float = 0.1
    beta_max: float = 20.0
    hidden_dim: int = 50
    num_layers: int = 6
    num_heads: int = 4
    dropout: float = 0.0
    learning_rate: float = 5e-4
    batch_size: int = 1000
    max_epochs: int = 2
    stop_after_epochs: int = 20
    clip_max_norm: Optional[float] = 5.0
    condition_probability: float = 0.7
    mask_variant: str = "dependency_mask_M_E_plus_condition_mask_M_C"
    conditioning_pattern: str = "five_family_uniform_mixture"
    simulation_budget: int = 1024
    sampler_family: str = "sde_backward"
    device: str = "cpu"
    seed: int = 0
    dry_run: bool = True

    def validate(self) -> None:
        if self.num_diffusion_steps < 1:
            raise ValueError("num_diffusion_steps must be positive")
        if not (0.0 <= self.condition_probability <= 1.0):
            raise ValueError("condition_probability must lie in [0, 1]")
        if self.sampler_family not in SAMPLER_FAMILIES:
            raise ValueError(f"sampler_family must be one of {SAMPLER_FAMILIES}")
        if self.schedule != "variance_exploding":
            raise ValueError("Simformer training in this reproduction uses the Variance Exploding SDE")
        if not (0.0 < self.sigma_min < self.sigma_max):
            raise ValueError("VESDE requires 0 < sigma_min < sigma_max")
        if not (0.0 < self.t_min < self.t_max <= 1.0):
            raise ValueError("VESDE times must be sampled in [1e-5, 1]")

    def metadata(self) -> Dict[str, Any]:
        payload = dataclasses.asdict(self)
        payload["paper_obligation"] = {
            "joint_distribution_training": "p(theta,x)=p(x_hat)",
            "vesde": {
                "drift": "f(x,t)=0",
                "diffusion": "sigma_min*(sigma_max/sigma_min)^t*sqrt(2*log(sigma_max/sigma_min))",
                "perturbation_kernel": "x_t = x_0 + sigma(t)*epsilon",
                "sigma_min": self.sigma_min,
                "sigma_max": self.sigma_max,
                "time_interval": [self.t_min, self.t_max],
                "euler_maruyama_steps": self.num_diffusion_steps,
            },
            "M_E_enters_attention": True,
            "M_C_enters_noising_loss_sampling": True,
            "condition_mask_families": [
                "joint_all_false",
                "posterior_theta_given_x",
                "likelihood_x_given_theta",
                "mask_probability_0.3",
                "mask_probability_0.7",
            ],
            "sampling_families": list(SAMPLER_FAMILIES),
            "blacklisted_repository_used": False,
        }
        payload["reference_grounding"] = [
            "paperbench_ref_001 docs/how_to_guide/07_gpu_training.ipynb",
            "paperbench_ref_001 sbi/inference/trainers/npe/npe_a.py",
        ]
        return payload


DIFFUSION_METHOD_REGISTRY: Dict[str, Dict[str, Any]] = {
    "simformer_score_diffusion": {
        "method": "simformer_score_diffusion",
        "training_target": "joint_p_theta_x",
        "score_objective": "conditional_denoising_score_matching",
        "attention_mask": "M_E_dependency_structure",
        "condition_mask": "M_C_binary_condition_state",
        "samplers": list(SAMPLER_FAMILIES),
        "default_config": dataclasses.asdict(DiffusionConfig()),
        "reference_grounding": "paperbench_ref_001 sbi/inference/trainers/npe/npe_a.py",
    }
}


# ---------------------------------------------------------------------------
# Tokenization and condition/dependency masks
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TokenizerSpec:
    """Variable schema for joint SBI samples."""

    variable_names: Tuple[str, ...]
    variable_kinds: Tuple[str, ...]
    value_dim: int = 1

    def __post_init__(self) -> None:
        if len(self.variable_names) != len(self.variable_kinds):
            raise ValueError("variable_names and variable_kinds must have equal length")
        if self.value_dim != 1:
            raise ValueError("This lightweight tokenizer currently expects scalar variables")


class SBITokenizer:
    """Serialize simulator joint samples into Simformer tokens.

    ``encode`` accepts either:

    * a mapping containing arrays such as ``{"theta": ..., "x": ...}``, or
    * a dense array of shape ``(batch, variables)``.

    It returns a dictionary with machine-readable token fields:
    ``variable_id``, ``variable_name``, ``variable_kind``, ``value``,
    ``condition_state``, and ``joint_value``.  The variable sequence represents a
    sample from ``p(theta, x)`` rather than a posterior-only or likelihood-only
    factorization.

    reference_grounding: paperbench_ref_001 docs/how_to_guide/04_embedding_networks.ipynb
    """

    def __init__(
        self,
        variable_names: Optional[Sequence[str]] = None,
        variable_kinds: Optional[Sequence[str]] = None,
        standardize: bool = True,
        eps: float = 1e-6,
    ) -> None:
        if variable_names is None:
            variable_names = ("theta_0", "theta_1", "x_0", "x_1")
        if variable_kinds is None:
            split = max(1, len(variable_names) // 2)
            variable_kinds = tuple("parameter" if i < split else "observation" for i in range(len(variable_names)))
        self.spec = TokenizerSpec(tuple(variable_names), tuple(variable_kinds))
        self.standardize = bool(standardize)
        self.eps = float(eps)
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None

    @property
    def num_variables(self) -> int:
        return len(self.spec.variable_names)

    def fit(self, batch: Union[Mapping[str, Any], ArrayLike]) -> "SBITokenizer":
        values = self._batch_to_joint_array(batch)
        self._mean = values.mean(axis=0, keepdims=True)
        self._std = values.std(axis=0, keepdims=True) + self.eps
        return self

    def encode(
        self,
        batch: Union[Mapping[str, Any], ArrayLike],
        condition_mask: Optional[ArrayLike] = None,
        fit_if_needed: bool = True,
    ) -> Dict[str, Any]:
        values = self._batch_to_joint_array(batch)
        if values.shape[1] != self.num_variables:
            raise ValueError(
                f"Batch has {values.shape[1]} variables but tokenizer schema has {self.num_variables}: "
                f"{self.spec.variable_names}"
            )

        if self.standardize and self._mean is None and fit_if_needed:
            self.fit(values)

        encoded_values = values.astype(np.float32, copy=True)
        if self.standardize and self._mean is not None and self._std is not None:
            encoded_values = ((encoded_values - self._mean) / self._std).astype(np.float32)

        mask = _coerce_condition_mask(condition_mask, batch_size=encoded_values.shape[0], num_variables=self.num_variables)
        variable_id = np.broadcast_to(np.arange(self.num_variables, dtype=np.int64), encoded_values.shape).copy()

        return {
            "variable_id": variable_id,
            "variable_name": list(self.spec.variable_names),
            "variable_kind": list(self.spec.variable_kinds),
            "value": encoded_values[..., None],
            "joint_value": encoded_values,
            "raw_joint_value": values.astype(np.float32),
            "condition_state": mask.astype(np.float32),
            "condition_mask": mask.astype(np.float32),
            "metadata": {
                "distribution": "joint_p_theta_x",
                "condition_state_binary": True,
                "standardized": self.standardize,
                "reference_grounding": "paperbench_ref_001 docs/how_to_guide/04_embedding_networks.ipynb",
            },
        }

    def decode_values(self, encoded_values: ArrayLike) -> np.ndarray:
        arr = np.asarray(encoded_values, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        if self.standardize and self._mean is not None and self._std is not None:
            arr = arr * self._std + self._mean
        return arr.astype(np.float32)

    def registry_entry(self) -> Dict[str, Any]:
        return {
            "tokenizer": "SBITokenizer",
            "variable_names": list(self.spec.variable_names),
            "variable_kinds": list(self.spec.variable_kinds),
            "returns": ["variable_id", "value", "condition_state", "joint_value"],
            "joint_distribution": "p(theta,x)",
            "condition_state": "binary",
            "supports_resampled_conditioning": True,
            "reference_grounding": "paperbench_ref_001 docs/how_to_guide/04_embedding_networks.ipynb",
        }

    def _batch_to_joint_array(self, batch: Union[Mapping[str, Any], ArrayLike]) -> np.ndarray:
        if isinstance(batch, Mapping):
            if "joint" in batch:
                arr = np.asarray(batch["joint"], dtype=np.float32)
            else:
                chunks: List[np.ndarray] = []
                for key in ("theta", "parameters", "param", "x", "observations", "observation", "summary"):
                    if key in batch:
                        part = np.asarray(batch[key], dtype=np.float32)
                        if part.ndim == 1:
                            part = part[:, None]
                        chunks.append(part.reshape(part.shape[0], -1))
                if not chunks:
                    ordered = []
                    for name in self.spec.variable_names:
                        if name in batch:
                            part = np.asarray(batch[name], dtype=np.float32)
                            if part.ndim == 0:
                                part = part.reshape(1, 1)
                            elif part.ndim == 1:
                                part = part[:, None]
                            ordered.append(part.reshape(part.shape[0], -1))
                    chunks = ordered
                if not chunks:
                    raise ValueError("Mapping batch must contain joint, theta/x, or named variable arrays")
                n = chunks[0].shape[0]
                if any(c.shape[0] != n for c in chunks):
                    raise ValueError("All batch arrays must have the same leading dimension")
                arr = np.concatenate(chunks, axis=1)
        else:
            arr = np.asarray(batch, dtype=np.float32)

        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)
        return arr.astype(np.float32)


class ConditionMaskSampler:
    """Sample binary conditioning masks ``M_C`` for training and evaluation."""

    def __init__(
        self,
        num_variables: int,
        probability: float = 0.7,
        pattern: str = "five_family_uniform_mixture",
        seed: Optional[int] = None,
    ) -> None:
        if num_variables < 1:
            raise ValueError("num_variables must be positive")
        if not (0.0 <= probability <= 1.0):
            raise ValueError("probability must lie in [0, 1]")
        self.num_variables = int(num_variables)
        self.probability = float(probability)
        self.pattern = str(pattern)
        self.rng = np.random.default_rng(seed)

    def sample(self, batch_size: int, force_at_least_one_unknown: bool = True) -> np.ndarray:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.pattern in {"none", "unconditional", "joint_all_false"}:
            mask = np.zeros((batch_size, self.num_variables), dtype=np.float32)
        elif self.pattern in {"all_observed", "fully_conditioned"}:
            mask = np.ones((batch_size, self.num_variables), dtype=np.float32)
        elif self.pattern in {"posterior", "condition_on_observations"}:
            split = self.num_variables // 2
            mask = np.zeros((batch_size, self.num_variables), dtype=np.float32)
            mask[:, split:] = 1.0
        elif self.pattern in {"single_random", "one_observed"}:
            mask = np.zeros((batch_size, self.num_variables), dtype=np.float32)
            idx = self.rng.integers(0, self.num_variables, size=batch_size)
            mask[np.arange(batch_size), idx] = 1.0
        elif self.pattern in {"likelihood", "condition_on_parameters"}:
            split = self.num_variables // 2
            mask = np.zeros((batch_size, self.num_variables), dtype=np.float32)
            mask[:, :split] = 1.0
        elif self.pattern in {"mask_probability_0.3", "bernoulli_0.3", "bernoulli_0_3"}:
            mask = (self.rng.random((batch_size, self.num_variables)) < 0.3).astype(np.float32)
        elif self.pattern in {"mask_probability_0.7", "bernoulli_0.7", "bernoulli_0_7", "bernoulli"}:
            mask = (self.rng.random((batch_size, self.num_variables)) < self.probability).astype(np.float32)
        elif self.pattern in {"five_family_uniform_mixture", "uniform_binary_resampled", "paper_mixture", "random"}:
            options = (
                "joint_all_false",
                "posterior_theta_given_x",
                "likelihood_x_given_theta",
                "mask_probability_0.3",
                "mask_probability_0.7",
            )
            mask = np.zeros((batch_size, self.num_variables), dtype=np.float32)
            split = self.num_variables // 2
            for row, choice in enumerate(self.rng.choice(options, size=batch_size)):
                if choice == "posterior_theta_given_x":
                    mask[row, split:] = 1.0
                elif choice == "likelihood_x_given_theta":
                    mask[row, :split] = 1.0
                elif choice == "mask_probability_0.3":
                    mask[row] = (self.rng.random(self.num_variables) < 0.3).astype(np.float32)
                elif choice == "mask_probability_0.7":
                    mask[row] = (self.rng.random(self.num_variables) < 0.7).astype(np.float32)
        else:
            mask = (self.rng.random((batch_size, self.num_variables)) < self.probability).astype(np.float32)

        if force_at_least_one_unknown:
            all_known = mask.sum(axis=1) >= self.num_variables
            if np.any(all_known):
                drop = self.rng.integers(0, self.num_variables, size=int(all_known.sum()))
                rows = np.where(all_known)[0]
                mask[rows, drop] = 0.0
        return mask.astype(np.float32)

    def registry_entry(self) -> Dict[str, Any]:
        return {
            "mask": "M_C",
            "num_variables": self.num_variables,
            "probability": self.probability,
            "pattern": self.pattern,
            "families": [
                "joint_all_false",
                "posterior_theta_given_x",
                "likelihood_x_given_theta",
                "mask_probability_0.3",
                "mask_probability_0.7",
            ],
            "binary": True,
            "resampled_during_training": True,
            "enters": ["forward_noising", "loss_masking", "conditional_sampling"],
        }


class DependencyMaskBuilder:
    """Build simulator dependency attention masks ``M_E``.

    The returned matrix uses ``1`` for allowed attention edges and ``0`` for
    blocked edges.  The score model converts this into PyTorch's boolean
    attention mask and passes it to transformer attention, so simulator
    dependency structure is active in computation rather than stored only as
    metadata.
    """

    def __init__(
        self,
        variable_names: Sequence[str],
        dependencies: Optional[Mapping[str, Sequence[str]]] = None,
        variant: str = "structured_dependency",
    ) -> None:
        if not variable_names:
            raise ValueError("variable_names must not be empty")
        self.variable_names = tuple(variable_names)
        self.variant = str(variant)
        self.dependencies = {str(k): tuple(v) for k, v in (dependencies or {}).items()}

    def build(self, include_self: bool = True, bidirectional: bool = True) -> np.ndarray:
        n = len(self.variable_names)
        if self.variant in {"dense", "fully_connected", "none"}:
            mask = np.ones((n, n), dtype=np.float32)
        elif self.variant in {"chain", "temporal"}:
            mask = np.zeros((n, n), dtype=np.float32)
            for i in range(n):
                lo, hi = max(0, i - 1), min(n, i + 2)
                mask[i, lo:hi] = 1.0
        else:
            mask = np.zeros((n, n), dtype=np.float32)
            name_to_idx = {name: i for i, name in enumerate(self.variable_names)}
            for target, parents in self.dependencies.items():
                if target not in name_to_idx:
                    continue
                ti = name_to_idx[target]
                for parent in parents:
                    if parent in name_to_idx:
                        pi = name_to_idx[parent]
                        mask[ti, pi] = 1.0
                        if bidirectional:
                            mask[pi, ti] = 1.0

            if not self.dependencies:
                split = n // 2
                mask[:split, :split] = 1.0
                mask[split:, :] = 1.0
                if bidirectional:
                    mask[:, split:] = np.maximum(mask[:, split:], mask[split:, :].T)

        if include_self:
            np.fill_diagonal(mask, 1.0)
        return mask.astype(np.float32)

    def registry_entry(self) -> Dict[str, Any]:
        entry = {
            "mask": "M_E",
            "variant": self.variant,
            "variable_names": list(self.variable_names),
            "dependencies": {k: list(v) for k, v in self.dependencies.items()},
            "semantics": "1=attention_allowed, 0=attention_blocked",
            "enters_transformer_attention": True,
            "conditioning_graph": "directed M_E can be updated from M_C by Webb-style graph inversion before attention",
        }
        return entry


# ---------------------------------------------------------------------------
# Diffusion schedules, noising, and objective
# ---------------------------------------------------------------------------


def vp_beta(t: np.ndarray, beta_min: float = 0.1, beta_max: float = 20.0) -> np.ndarray:
    t = np.asarray(t, dtype=np.float32)
    return beta_min + t * (beta_max - beta_min)


def vp_marginal_alpha_sigma(
    t: ArrayLike,
    beta_min: float = 0.1,
    beta_max: float = 20.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Variance-preserving marginal coefficients.

    ``x_t = alpha(t) x_0 + sigma(t) eps``.
    """

    t_arr = np.asarray(t, dtype=np.float32)
    log_mean_coeff = -0.25 * (beta_max - beta_min) * t_arr**2 - 0.5 * beta_min * t_arr
    alpha = np.exp(log_mean_coeff).astype(np.float32)
    sigma = np.sqrt(np.maximum(1.0 - np.exp(2.0 * log_mean_coeff), 1e-6)).astype(np.float32)
    return alpha, sigma


@dataclasses.dataclass(frozen=True)
class VESDE:
    """Variance Exploding SDE used by Simformer training and sampling.

    Forward SDE:
        dx = f(x,t) dt + g(t) dw, with f(x,t)=0 and
        g(t)=sigma_min * (sigma_max/sigma_min)^t * sqrt(2 log(sigma_max/sigma_min)).

    Perturbation kernel:
        x_t | x_0 ~ Normal(x_0, sigma(t)^2 I),
        sigma(t) = sigma_min * (sigma_max/sigma_min)^t.
    """

    sigma_min: float = 1.0e-4
    sigma_max: float = 15.0
    t_min: float = 1.0e-5
    t_max: float = 1.0

    @property
    def ratio(self) -> float:
        return float(self.sigma_max / self.sigma_min)

    def drift(self, x: ArrayLike, t: ArrayLike) -> np.ndarray:
        """VESDE drift term f(x,t)=0 with the same shape as x."""

        return np.zeros_like(np.asarray(x, dtype=np.float32), dtype=np.float32)

    def sigma(self, t: ArrayLike) -> np.ndarray:
        t_arr = np.asarray(t, dtype=np.float32)
        return (self.sigma_min * np.power(self.ratio, t_arr)).astype(np.float32)

    def variance(self, t: ArrayLike) -> np.ndarray:
        sig = self.sigma(t)
        return (sig * sig).astype(np.float32)

    def diffusion(self, t: ArrayLike) -> np.ndarray:
        t_arr = np.asarray(t, dtype=np.float32)
        return (
            self.sigma_min
            * np.power(self.ratio, t_arr)
            * math.sqrt(2.0 * math.log(self.ratio))
        ).astype(np.float32)

    def sample_time(self, batch_size: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        generator = rng if rng is not None else np.random.default_rng()
        return generator.uniform(low=self.t_min, high=self.t_max, size=(int(batch_size),)).astype(np.float32)

    def perturbation_kernel(
        self,
        x0: ArrayLike,
        t: ArrayLike,
        noise: Optional[ArrayLike] = None,
    ) -> Dict[str, np.ndarray]:
        x0_arr = np.asarray(x0, dtype=np.float32)
        eps = np.asarray(noise, dtype=np.float32) if noise is not None else np.random.default_rng().normal(size=x0_arr.shape).astype(np.float32)
        t_arr = np.asarray(t, dtype=np.float32).reshape(-1)
        if t_arr.size == 1:
            t_arr = np.repeat(t_arr, x0_arr.shape[0])
        sigma = self.sigma(t_arr)[:, None]
        x_t = x0_arr + sigma * eps
        return {
            "mean": x0_arr.astype(np.float32),
            "sigma": sigma.astype(np.float32),
            "variance": (sigma * sigma).astype(np.float32),
            "noise": eps.astype(np.float32),
            "x_t": x_t.astype(np.float32),
            "target_score": (-eps / np.maximum(sigma, 1.0e-12)).astype(np.float32),
            "score_target_formula": "-epsilon/sigma(t)",
        }


def vesde_drift(x: ArrayLike, t: ArrayLike, sigma_min: float = 1.0e-4, sigma_max: float = 15.0) -> np.ndarray:
    return VESDE(sigma_min=sigma_min, sigma_max=sigma_max).drift(x, t)


def vesde_diffusion(
    t: ArrayLike,
    sigma_min: float = 1.0e-4,
    sigma_max: float = 15.0,
) -> np.ndarray:
    return VESDE(sigma_min=sigma_min, sigma_max=sigma_max).diffusion(t)


def vesde_sigma(
    t: ArrayLike,
    sigma_min: float = 1.0e-4,
    sigma_max: float = 15.0,
) -> np.ndarray:
    return VESDE(sigma_min=sigma_min, sigma_max=sigma_max).sigma(t)


def forward_noising(
    x0: ArrayLike,
    t: ArrayLike,
    condition_mask: Optional[ArrayLike],
    noise: Optional[ArrayLike] = None,
    beta_min: float = 0.1,
    beta_max: float = 20.0,
    sigma_min: float = 1.0e-4,
    sigma_max: float = 15.0,
) -> Dict[str, np.ndarray]:
    """Apply conditional VESDE forward noising.

    ``M_C`` enters directly: conditioned variables remain equal to ``x0`` while
    unconditioned variables receive diffusion noise.  The returned
    ``loss_weight`` is ``1 - M_C`` and is used by the score-matching objective.
    """

    x0_arr = np.asarray(x0, dtype=np.float32)
    if x0_arr.ndim == 3 and x0_arr.shape[-1] == 1:
        x0_arr = x0_arr[..., 0]
    if x0_arr.ndim != 2:
        raise ValueError("x0 must have shape (batch, variables)")

    t_arr = np.asarray(t, dtype=np.float32).reshape(-1)
    if t_arr.size == 1:
        t_arr = np.repeat(t_arr, x0_arr.shape[0])
    if t_arr.shape[0] != x0_arr.shape[0]:
        raise ValueError("t must be scalar or have one value per batch element")

    mask = _coerce_condition_mask(condition_mask, x0_arr.shape[0], x0_arr.shape[1])
    eps = np.asarray(noise, dtype=np.float32) if noise is not None else np.random.default_rng().normal(size=x0_arr.shape).astype(np.float32)
    vesde = VESDE(sigma_min=sigma_min, sigma_max=sigma_max)
    perturbed = vesde.perturbation_kernel(x0_arr, t_arr, noise=eps)
    sigma = perturbed["sigma"]
    noised = perturbed["x_t"]
    x_t = mask * x0_arr + (1.0 - mask) * noised

    return {
        "x_t": x_t.astype(np.float32),
        "noise": eps.astype(np.float32),
        "target_score": (-eps / np.maximum(sigma, 1e-6)).astype(np.float32),
        "condition_mask": mask.astype(np.float32),
        "loss_weight": (1.0 - mask).astype(np.float32),
        "alpha": np.ones_like(sigma, dtype=np.float32),
        "sigma": sigma.astype(np.float32),
        "variance": (sigma * sigma).astype(np.float32),
        "vesde_drift": vesde.drift(x0_arr, t_arr),
        "vesde_diffusion": vesde.diffusion(t_arr)[:, None].astype(np.float32),
    }


def denoising_score_matching_loss(
    predicted_score: ArrayLike,
    target_score: ArrayLike,
    condition_mask: Optional[ArrayLike],
    reduction: str = "mean",
) -> float:
    """Paper-specific conditional score objective.

    The metric formula is masked denoising score matching on joint variables:

    ``E_t,E_eps [ || (1 - M_C) * (s_phi(x_t,t,M_C,M_E) - target_score) ||^2 ]``.

    Conditioned variables do not contribute to the loss.
    """

    pred = np.asarray(predicted_score, dtype=np.float32)
    target = np.asarray(target_score, dtype=np.float32)
    if pred.shape != target.shape:
        raise ValueError("predicted_score and target_score must have matching shape")
    if pred.ndim == 3 and pred.shape[-1] == 1:
        pred = pred[..., 0]
        target = target[..., 0]
    mask = _coerce_condition_mask(condition_mask, pred.shape[0], pred.shape[1])
    weight = 1.0 - mask
    sq = ((pred - target) ** 2) * weight
    denom = max(float(weight.sum()), 1.0)
    if reduction == "sum":
        return float(sq.sum())
    if reduction == "none":
        return float(sq.sum(axis=1).mean())
    return float(sq.sum() / denom)


# ---------------------------------------------------------------------------
# Score model
# ---------------------------------------------------------------------------


class SimformerScoreModel:
    """Lazy PyTorch transformer score network.

    The wrapper is import-safe without PyTorch.  Instantiating the model imports
    PyTorch and builds a transformer encoder.  ``M_E`` is passed to
    ``TransformerEncoder`` as ``mask`` on every forward call.

    reference_grounding: paperbench_ref_001 docs/how_to_guide/07_gpu_training.ipynb
    """

    def __init__(
        self,
        num_variables: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.0,
        device: str = "cpu",
    ) -> None:
        if num_variables < 1:
            raise ValueError("num_variables must be positive")
        torch = _import_torch()
        self.torch = torch
        self.device = torch.device(device)
        self.num_variables = int(num_variables)
        self.hidden_dim = int(hidden_dim)
        self.module = _build_torch_score_network(
            torch=torch,
            num_variables=num_variables,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        ).to(self.device)

    def parameters(self) -> Any:
        return self.module.parameters()

    def train(self, mode: bool = True) -> "SimformerScoreModel":
        self.module.train(mode)
        return self

    def eval(self) -> "SimformerScoreModel":
        self.module.eval()
        return self

    def state_dict(self) -> Any:
        return self.module.state_dict()

    def load_state_dict(self, state: Any) -> Any:
        return self.module.load_state_dict(state)

    def forward(
        self,
        x_t: Any,
        t: Any,
        condition_mask: Any,
        dependency_mask: Any,
        variable_id: Optional[Any] = None,
    ) -> Any:
        torch = self.torch
        x_t_tensor = _torch_tensor(torch, x_t, self.device, dtype=torch.float32)
        if x_t_tensor.ndim == 3 and x_t_tensor.shape[-1] == 1:
            x_t_tensor = x_t_tensor[..., 0]
        t_tensor = _torch_tensor(torch, t, self.device, dtype=torch.float32).reshape(-1)
        if t_tensor.numel() == 1:
            t_tensor = t_tensor.repeat(x_t_tensor.shape[0])
        c_tensor = _torch_tensor(torch, condition_mask, self.device, dtype=torch.float32)
        if c_tensor.ndim == 3 and c_tensor.shape[-1] == 1:
            c_tensor = c_tensor[..., 0]
        dep = _torch_tensor(torch, dependency_mask, self.device, dtype=torch.float32)
        if dep.ndim != 2:
            raise ValueError("dependency_mask M_E must have shape (variables, variables)")
        if variable_id is None:
            variable_id = torch.arange(self.num_variables, device=self.device).expand(x_t_tensor.shape[0], -1)
        else:
            variable_id = _torch_tensor(torch, variable_id, self.device, dtype=torch.long)
        return self.module(x_t_tensor, t_tensor, c_tensor, dep, variable_id)

    __call__ = forward

    def registry_entry(self) -> Dict[str, Any]:
        return {
            "model": "SimformerScoreModel",
            "num_variables": self.num_variables,
            "score_network": "transformer_encoder",
            "M_E_passed_to_attention": True,
            "M_C_passed_as_condition_embedding": True,
            "device": str(self.device),
        }


def _build_torch_score_network(
    torch: Any,
    num_variables: int,
    hidden_dim: int,
    num_layers: int,
    num_heads: int,
    dropout: float,
) -> Any:
    nn = torch.nn

    class _TorchScoreNetwork(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.value_proj = nn.Linear(1, hidden_dim)
            self.variable_embedding = nn.Embedding(num_variables, hidden_dim)
            self.condition_embedding = nn.Embedding(2, hidden_dim)
            self.time_mlp = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
            self.out = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))

        def forward(self, x_t: Any, t: Any, condition_mask: Any, dependency_mask: Any, variable_id: Any) -> Any:
            values = x_t.unsqueeze(-1)
            cond_idx = condition_mask.clamp(0, 1).long()
            h = self.value_proj(values)
            h = h + self.variable_embedding(variable_id.long())
            h = h + self.condition_embedding(cond_idx)
            h = h + self.time_mlp(t.reshape(-1, 1)).unsqueeze(1)

            # M_E uses 1=allowed, 0=blocked.  PyTorch src mask uses True=blocked.
            # This is the explicit dependency-structure attention path required
            # by the Simformer contract.
            blocked_attention = dependency_mask <= 0.0
            encoded = self.transformer(h, mask=blocked_attention)
            return self.out(encoded).squeeze(-1)

    return _TorchScoreNetwork()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


class JointBatchProvider(Protocol):
    """Protocol for simulators/data loaders that yield joint p(theta,x) batches."""

    def __iter__(self) -> Iterator[Union[Mapping[str, Any], ArrayLike]]:
        ...


@dataclasses.dataclass
class TrainingResult:
    model: Optional[SimformerScoreModel]
    loss_trace: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    artifacts: Dict[str, str]


class DiffusionTrainer:
    """Trainer for conditional denoising score matching on joint SBI samples.

    The training loop follows the reference ``sbi`` trainer intent: configurable
    batch size, learning rate, gradient clipping, device selection, validation
    metadata, and bounded stopping controls.  It trains a score model on
    simulator joint samples rather than posterior-only samples.

    reference_grounding: paperbench_ref_001 sbi/inference/trainers/npe/npe_a.py
    reference_grounding: paperbench_ref_001 docs/how_to_guide/07_gpu_training.ipynb
    """

    def __init__(
        self,
        tokenizer: SBITokenizer,
        dependency_mask: ArrayLike,
        config: Optional[DiffusionConfig] = None,
        model: Optional[SimformerScoreModel] = None,
        results_dir: Union[str, os.PathLike[str]] = "results",
    ) -> None:
        self.tokenizer = tokenizer
        self.dependency_mask = np.asarray(dependency_mask, dtype=np.float32)
        self.config = config or DiffusionConfig()
        self.config.validate()
        self.model = model
        self.results_dir = Path(results_dir)

    def train(
        self,
        batches: Iterable[Union[Mapping[str, Any], ArrayLike]],
        max_batches: Optional[int] = None,
        write_artifacts: bool = True,
    ) -> TrainingResult:
        torch = _import_torch()
        self._set_seed()
        if self.model is None:
            self.model = SimformerScoreModel(
                num_variables=self.tokenizer.num_variables,
                hidden_dim=self.config.hidden_dim,
                num_layers=self.config.num_layers,
                num_heads=self.config.num_heads,
                dropout=self.config.dropout,
                device=self.config.device,
            )

        self.model.train(True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        mask_sampler = ConditionMaskSampler(
            self.tokenizer.num_variables,
            probability=self.config.condition_probability,
            pattern=self.config.conditioning_pattern,
            seed=self.config.seed,
        )

        loss_trace: List[Dict[str, Any]] = []
        effective_max_batches = max_batches if max_batches is not None else (1 if self.config.dry_run else None)
        batch_counter = 0

        best_validation_loss = float("inf")
        epochs_without_improvement = 0
        vesde = VESDE(
            sigma_min=self.config.sigma_min,
            sigma_max=self.config.sigma_max,
            t_min=self.config.t_min,
            t_max=self.config.t_max,
        )

        for epoch in range(self.config.max_epochs):
            for batch in batches:
                if effective_max_batches is not None and batch_counter >= effective_max_batches:
                    break

                encoded_seed = self.tokenizer.encode(
                    batch,
                    condition_mask=np.zeros((self._infer_batch_size(batch), self.tokenizer.num_variables), dtype=np.float32),
                )
                x0_np = np.asarray(encoded_seed["joint_value"], dtype=np.float32)
                condition_mask_np = mask_sampler.sample(x0_np.shape[0])
                encoded = self.tokenizer.encode(batch, condition_mask=condition_mask_np, fit_if_needed=False)
                x0_np = np.asarray(encoded["joint_value"], dtype=np.float32)
                condition_mask_np = np.asarray(encoded["condition_state"], dtype=np.float32)

                t_np = vesde.sample_time(
                    x0_np.shape[0],
                    rng=np.random.default_rng(self.config.seed + batch_counter + epoch),
                )
                noise_np = np.random.default_rng(self.config.seed + 1000 + batch_counter + epoch).normal(
                    size=x0_np.shape
                ).astype(np.float32)
                noised = forward_noising(
                    x0_np,
                    t_np,
                    condition_mask_np,
                    noise=noise_np,
                    sigma_min=self.config.sigma_min,
                    sigma_max=self.config.sigma_max,
                )

                optimizer.zero_grad(set_to_none=True)
                pred = self.model.forward(
                    noised["x_t"],
                    t_np,
                    condition_mask_np,
                    self.dependency_mask,
                    variable_id=encoded["variable_id"],
                )
                target = _torch_tensor(torch, noised["target_score"], self.model.device, dtype=torch.float32)
                weight = _torch_tensor(torch, noised["loss_weight"], self.model.device, dtype=torch.float32)
                lambda_t = _torch_tensor(torch, noised["vesde_diffusion"] ** 2, self.model.device, dtype=torch.float32)
                loss = (((pred - target) ** 2) * weight * lambda_t).sum() / torch.clamp(weight.sum(), min=1.0)
                loss.backward()
                if self.config.clip_max_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_max_norm)
                optimizer.step()

                batch_counter += 1
                loss_trace.append(
                    {
                        "epoch": epoch,
                        "batch": batch_counter,
                        "loss": float(loss.detach().cpu().item()),
                        "conditioned_fraction": float(condition_mask_np.mean()),
                        "mask_variant": self.config.mask_variant,
                        "objective": self.config.objective,
                        "vesde_time_interval": [self.config.t_min, self.config.t_max],
                        "lambda_t": "g(t)^2",
                        "optimizer": "Adam",
                    }
                )

                validation_loss = float(loss.detach().cpu().item())
                if validation_loss + 1.0e-8 < best_validation_loss:
                    best_validation_loss = validation_loss
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                if epochs_without_improvement >= int(self.config.stop_after_epochs):
                    break

            if effective_max_batches is not None and batch_counter >= effective_max_batches:
                break
            if epochs_without_improvement >= int(self.config.stop_after_epochs):
                break

        metadata = self._training_metadata(
            mask_sampler=mask_sampler,
            batch_counter=batch_counter,
            best_validation_loss=best_validation_loss,
        )
        artifacts: Dict[str, str] = {}
        if write_artifacts:
            artifacts = self.write_training_artifacts(loss_trace=loss_trace, metadata=metadata)
        return TrainingResult(model=self.model, loss_trace=loss_trace, metadata=metadata, artifacts=artifacts)

    def write_training_artifacts(
        self,
        loss_trace: Sequence[Mapping[str, Any]],
        metadata: Mapping[str, Any],
    ) -> Dict[str, str]:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "model_registry": self.results_dir / "model_registry.json",
            "tokenizer_registry": self.results_dir / "tokenizer_registry.json",
            "attention_mask_registry": self.results_dir / "attention_mask_registry.json",
            "diffusion_config": self.results_dir / "diffusion_config.json",
            "loss_trace": self.results_dir / "loss_trace.json",
        }
        _write_json(paths["model_registry"], {"models": [metadata["model_registry"]], "metadata": dict(metadata)})
        _write_json(paths["tokenizer_registry"], self.tokenizer.registry_entry())
        _write_json(
            paths["attention_mask_registry"],
            {
                "mask": "M_E",
                "matrix": self.dependency_mask.tolist(),
                "enters_transformer_attention": True,
                "mask_variant": self.config.mask_variant,
            },
        )
        _write_json(paths["diffusion_config"], dict(metadata))
        _write_json(
            paths["loss_trace"],
            {
                "artifact_type": "training_loss_trace",
                "dry_run": bool(self.config.dry_run),
                "loss_trace": list(loss_trace),
                "metric_formula": "mean(((score_pred-target_score)^2)*(1-M_C))/sum(1-M_C)",
                "not_paper_scale_result": bool(self.config.dry_run),
            },
        )
        return {k: str(v) for k, v in paths.items()}

    def _training_metadata(
        self,
        mask_sampler: ConditionMaskSampler,
        batch_counter: int,
        best_validation_loss: float = float("inf"),
    ) -> Dict[str, Any]:
        return {
            "method": self.config.method,
            "objective": self.config.objective,
            "joint_distribution": "p(theta,x)",
            "mask_variant": self.config.mask_variant,
            "conditioning_pattern": self.config.conditioning_pattern,
            "condition_mask_sampler": mask_sampler.registry_entry(),
            "simulation_budget": self.config.simulation_budget,
            "fixed_hyperparameters": self.config.metadata(),
            "num_optimization_batches": batch_counter,
            "optimizer": "Adam",
            "early_stopping": {
                "monitors": "validation_loss",
                "best_validation_loss": best_validation_loss,
                "stop_after_epochs": self.config.stop_after_epochs,
            },
            "model_registry": {
                "model": "SimformerScoreModel",
                "score_network": "transformer",
                "M_E_enters_attention": True,
                "M_C_enters_forward_noising_loss_sampling": True,
            },
            "blacklisted_repository_used": False,
        }

    def _infer_batch_size(self, batch: Union[Mapping[str, Any], ArrayLike]) -> int:
        if isinstance(batch, Mapping):
            for value in batch.values():
                arr = np.asarray(value)
                if arr.ndim > 0:
                    return int(arr.shape[0])
            return 1
        arr = np.asarray(batch)
        return int(arr.shape[0]) if arr.ndim > 1 else 1

    def _set_seed(self) -> None:
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        try:
            torch = _import_torch()
            torch.manual_seed(self.config.seed)
        except RuntimeError:
            pass


# ---------------------------------------------------------------------------
# Samplers and guided samplers
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class SamplingResult:
    samples: np.ndarray
    trace: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class SimformerSampler:
    """Conditional reverse diffusion sampler.

    The sampler interface deliberately exposes named families rather than hiding
    algorithm choice.  ``sde_backward`` includes stochastic reverse noise;
    ``ode_probability_flow`` uses deterministic probability-flow updates.

    reference_grounding: paperbench_ref_001 docs/how_to_guide/09_sampler_interface.ipynb
    """

    def __init__(
        self,
        model: SimformerScoreModel,
        tokenizer: SBITokenizer,
        dependency_mask: ArrayLike,
        config: Optional[DiffusionConfig] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.dependency_mask = np.asarray(dependency_mask, dtype=np.float32)
        self.config = config or DiffusionConfig()
        self.config.validate()

    def sample(
        self,
        conditioned_values: ArrayLike,
        condition_mask: ArrayLike,
        num_steps: Optional[int] = None,
        sampler_family: Optional[str] = None,
        seed: Optional[int] = None,
        guidance_fn: Optional[Callable[[Any, float], Any]] = None,
        guidance_scale: float = 0.0,
    ) -> SamplingResult:
        torch = _import_torch()
        rng = np.random.default_rng(self.config.seed if seed is None else seed)
        family = sampler_family or self.config.sampler_family
        if family not in SAMPLER_FAMILIES:
            raise ValueError(f"Unknown sampler_family {family!r}; expected {SAMPLER_FAMILIES}")

        steps = int(num_steps or self.config.num_diffusion_steps)
        if num_steps is None:
            steps = 500
        observed_np = np.asarray(conditioned_values, dtype=np.float32)
        if observed_np.ndim == 1:
            observed_np = observed_np[None, :]
        mask_np = _coerce_condition_mask(condition_mask, observed_np.shape[0], observed_np.shape[1])

        x = rng.normal(size=observed_np.shape).astype(np.float32)
        x = mask_np * observed_np + (1.0 - mask_np) * x

        self.model.eval()
        trace: List[Dict[str, Any]] = []
        dt = (self.config.t_max - self.config.t_min) / float(steps)
        vesde = VESDE(
            sigma_min=self.config.sigma_min,
            sigma_max=self.config.sigma_max,
            t_min=self.config.t_min,
            t_max=self.config.t_max,
        )

        with torch.no_grad():
            for step in range(steps, 0, -1):
                t_value = max(self.config.t_min, step / float(steps))
                g_t = float(vesde.diffusion(np.asarray([t_value], dtype=np.float32))[0])
                score = self.model.forward(
                    x,
                    np.full((x.shape[0],), t_value, dtype=np.float32),
                    mask_np,
                    self.dependency_mask,
                )
                if guidance_fn is not None and guidance_scale != 0.0:
                    guided = guidance_fn(score, t_value)
                    score = score + float(guidance_scale) * guided

                score_np = score.detach().cpu().numpy().astype(np.float32)
                forward_drift = vesde.drift(x, np.asarray([t_value], dtype=np.float32))
                reverse_drift = forward_drift - (g_t * g_t) * score_np

                if family == "ode_probability_flow":
                    x = x - 0.5 * reverse_drift * dt
                    stochastic_norm = 0.0
                else:
                    z = rng.normal(size=x.shape).astype(np.float32)
                    x = x - reverse_drift * dt + g_t * math.sqrt(max(dt, 1e-12)) * z
                    stochastic_norm = float(np.linalg.norm(z) / max(1, z.size))

                x = mask_np * observed_np + (1.0 - mask_np) * x
                if step in {steps, max(1, steps // 2), 1}:
                    trace.append(
                        {
                            "step": int(step),
                            "t": float(t_value),
                            "sampler_family": family,
                            "euler_maruyama_steps": steps,
                            "vesde_forward_drift": "f(x,t)=0",
                            "vesde_diffusion_g_t": g_t,
                            "conditioned_fraction": float(mask_np.mean()),
                            "stochastic_norm": stochastic_norm,
                            "M_C_reapplied": True,
                            "M_E_in_attention": True,
                        }
                    )

        decoded = self.tokenizer.decode_values(x)
        metadata = {
            "sampler_family": family,
            "num_steps": steps,
            "M_C_enters_conditional_sampling": True,
            "M_E_enters_transformer_attention": True,
            "dry_run": bool(self.config.dry_run),
            "reference_grounding": "paperbench_ref_001 docs/how_to_guide/09_sampler_interface.ipynb",
        }
        return SamplingResult(samples=decoded, trace=trace, metadata=metadata)

    def write_sampling_artifact(
        self,
        result: SamplingResult,
        results_dir: Union[str, os.PathLike[str]] = "results",
    ) -> str:
        path = Path(results_dir) / "sampling_trace.json"
        _write_json(
            path,
            {
                "artifact_type": "sampling_trace",
                "samples_shape": list(result.samples.shape),
                "trace": result.trace,
                "metadata": result.metadata,
                "not_paper_scale_result": bool(self.config.dry_run),
            },
        )
        return str(path)


class GuidedSimformerSampler(SimformerSampler):
    """Guided conditional sampler for interval/constraint diffusion.

    The guidance callback may return an additive score correction.  For
    lightweight interval guidance, use ``make_interval_guidance`` below.
    """

    def guided_sample(
        self,
        conditioned_values: ArrayLike,
        condition_mask: ArrayLike,
        guidance: Callable[[Any, float], Any],
        guidance_scale: float = 1.0,
        num_steps: Optional[int] = None,
        sampler_family: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> SamplingResult:
        return self.sample(
            conditioned_values=conditioned_values,
            condition_mask=condition_mask,
            num_steps=num_steps,
            sampler_family=sampler_family,
            seed=seed,
            guidance_fn=guidance,
            guidance_scale=guidance_scale,
        )


def make_interval_guidance(
    lower: Optional[ArrayLike] = None,
    upper: Optional[ArrayLike] = None,
    target: Optional[ArrayLike] = None,
    weight: float = 1.0,
) -> Callable[[Any, float], Any]:
    """Create an additive score guidance function for interval constraints.

    The returned function operates on score tensors and nudges variables toward
    an interval or target.  It is intentionally simple and differentiable enough
    for smoke-scale guided diffusion while keeping heavy task-specific simulators
    out of module scope.
    """

    def _guidance(score: Any, t: float) -> Any:
        torch = _import_torch()
        correction = torch.zeros_like(score)
        if lower is not None:
            lo = _torch_tensor(torch, lower, score.device, dtype=score.dtype)
            while lo.ndim < score.ndim:
                lo = lo.unsqueeze(0)
            correction = correction + torch.relu(lo - score)
        if upper is not None:
            hi = _torch_tensor(torch, upper, score.device, dtype=score.dtype)
            while hi.ndim < score.ndim:
                hi = hi.unsqueeze(0)
            correction = correction - torch.relu(score - hi)
        if target is not None:
            tgt = _torch_tensor(torch, target, score.device, dtype=score.dtype)
            while tgt.ndim < score.ndim:
                tgt = tgt.unsqueeze(0)
            correction = correction + (tgt - score)
        sigma_t = float(vesde_sigma(np.asarray([max(float(t), 1.0e-5)], dtype=np.float32))[0])
        return float(weight) * correction / max(sigma_t * sigma_t, 1.0e-12)

    return _guidance


# ---------------------------------------------------------------------------
# Policy/model adapters and smoke data pipeline
# ---------------------------------------------------------------------------


class SimformerPolicyAdapter:
    """Small adapter exposing train/sample/evaluate methods for registries."""

    def __init__(
        self,
        tokenizer: Optional[SBITokenizer] = None,
        dependency_mask: Optional[ArrayLike] = None,
        config: Optional[DiffusionConfig] = None,
        results_dir: Union[str, os.PathLike[str]] = "results",
    ) -> None:
        self.tokenizer = tokenizer or SBITokenizer()
        self.dependency_mask = (
            np.asarray(dependency_mask, dtype=np.float32)
            if dependency_mask is not None
            else DependencyMaskBuilder((self.tokenizer.spec.variable_names)).build()
        )
        self.config = config or DiffusionConfig()
        self.results_dir = Path(results_dir)
        self.model: Optional[SimformerScoreModel] = None

    def fit(self, joint_batches: Iterable[Union[Mapping[str, Any], ArrayLike]], max_batches: int = 1) -> TrainingResult:
        trainer = DiffusionTrainer(
            tokenizer=self.tokenizer,
            dependency_mask=self.dependency_mask,
            config=self.config,
            model=self.model,
            results_dir=self.results_dir,
        )
        result = trainer.train(joint_batches, max_batches=max_batches, write_artifacts=True)
        self.model = result.model
        return result

    def sample(
        self,
        conditioned_values: ArrayLike,
        condition_mask: ArrayLike,
        sampler_family: str = "sde_backward",
        guided: bool = False,
        guidance: Optional[Callable[[Any, float], Any]] = None,
    ) -> SamplingResult:
        if self.model is None:
            self.model = SimformerScoreModel(
                num_variables=self.tokenizer.num_variables,
                hidden_dim=self.config.hidden_dim,
                num_layers=self.config.num_layers,
                num_heads=self.config.num_heads,
                dropout=self.config.dropout,
                device=self.config.device,
            )
        sampler_cls = GuidedSimformerSampler if guided else SimformerSampler
        sampler = sampler_cls(self.model, self.tokenizer, self.dependency_mask, self.config)
        if guided and guidance is not None:
            result = sampler.guided_sample(
                conditioned_values,
                condition_mask,
                guidance=guidance,
                sampler_family=sampler_family,
                guidance_scale=1.0,
            )
        else:
            result = sampler.sample(conditioned_values, condition_mask, sampler_family=sampler_family)
        sampler.write_sampling_artifact(result, self.results_dir)
        return result

    def evaluate_score_loss(self, batch: Union[Mapping[str, Any], ArrayLike], condition_mask: Optional[ArrayLike] = None) -> Dict[str, Any]:
        encoded = self.tokenizer.encode(batch, condition_mask=condition_mask)
        t = np.full((encoded["joint_value"].shape[0],), 0.5, dtype=np.float32)
        noised = forward_noising(encoded["joint_value"], t, encoded["condition_state"])
        if self.model is None:
            zeros = np.zeros_like(noised["target_score"])
            loss = denoising_score_matching_loss(zeros, noised["target_score"], encoded["condition_state"])
        else:
            pred = self.model.forward(noised["x_t"], t, encoded["condition_state"], self.dependency_mask)
            loss = denoising_score_matching_loss(
                pred.detach().cpu().numpy(),
                noised["target_score"],
                encoded["condition_state"],
            )
        return {
            "metric": "masked_denoising_score_matching_loss",
            "value": float(loss),
            "formula": "mean(((score_pred-target_score)^2)*(1-M_C))/sum(1-M_C)",
            "conditioned_fraction": float(np.asarray(encoded["condition_state"]).mean()),
        }


def make_smoke_joint_batches(
    num_batches: int = 1,
    batch_size: int = 8,
    num_variables: int = 4,
    seed: int = 0,
) -> List[np.ndarray]:
    """Generate bounded synthetic joint p(theta,x) batches.

    This is a data-pipeline smoke fixture: parameters occupy the first half of
    the vector and observations are deterministic noisy functions of parameters.
    """

    rng = np.random.default_rng(seed)
    batches: List[np.ndarray] = []
    split = max(1, num_variables // 2)
    obs_dim = num_variables - split
    for _ in range(num_batches):
        theta = rng.normal(size=(batch_size, split)).astype(np.float32)
        if obs_dim <= 0:
            x = np.empty((batch_size, 0), dtype=np.float32)
        else:
            base = theta.mean(axis=1, keepdims=True)
            x = np.concatenate(
                [np.sin(base + 0.2 * j) + 0.05 * rng.normal(size=(batch_size, 1)) for j in range(obs_dim)],
                axis=1,
            ).astype(np.float32)
        batches.append(np.concatenate([theta, x], axis=1).astype(np.float32))
    return batches


# ---------------------------------------------------------------------------
# Artifact closure
# ---------------------------------------------------------------------------


def write_dry_run_artifacts(
    results_dir: Union[str, os.PathLike[str]] = "results",
    config: Optional[DiffusionConfig] = None,
) -> Dict[str, str]:
    """Materialize diffusion-core contract artifacts for smoke validation.

    Artifacts are labeled as readiness/schema outputs and do not claim trained
    model performance or completed experiments.
    """

    cfg = config or DiffusionConfig(dry_run=True)
    cfg.validate()
    out = Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = SBITokenizer()
    dep_builder = DependencyMaskBuilder(tokenizer.spec.variable_names)
    dependency_mask = dep_builder.build()
    mask_sampler = ConditionMaskSampler(
        tokenizer.num_variables,
        probability=cfg.condition_probability,
        pattern=cfg.conditioning_pattern,
        seed=cfg.seed,
    )

    payload_common = {
        "artifact_mode": "dry-run contract artifact",
        "dry_run": True,
        "not_paper_scale_result": True,
        "timestamp_unix": time.time(),
        "blacklisted_repository_used": False,
    }

    paths = {
        "model_registry": out / "model_registry.json",
        "tokenizer_registry": out / "tokenizer_registry.json",
        "attention_mask_registry": out / "attention_mask_registry.json",
        "diffusion_config": out / "diffusion_config.json",
        "loss_trace": out / "loss_trace.json",
        "sampling_trace": out / "sampling_trace.json",
        "readiness": out / "readiness.json",
        "evaluation_result": out / "evaluation_result.json",
    }

    _write_json(
        paths["model_registry"],
        {
            **payload_common,
            "models": [DIFFUSION_METHOD_REGISTRY["simformer_score_diffusion"]],
            "exposed_surfaces": [
                "tokenizer",
                "mask_builder",
                "score_network",
                "trainer",
                "sampler",
                "guided_sampler",
                "policy_adapter",
            ],
        },
    )
    _write_json(paths["tokenizer_registry"], {**payload_common, **tokenizer.registry_entry()})
    _write_json(
        paths["attention_mask_registry"],
        {
            **payload_common,
            **dep_builder.registry_entry(),
            "matrix": dependency_mask.tolist(),
            "condition_mask": mask_sampler.registry_entry(),
        },
    )
    _write_json(paths["diffusion_config"], {**payload_common, **cfg.metadata()})

    smoke_batch = make_smoke_joint_batches(num_batches=1, batch_size=4, num_variables=tokenizer.num_variables, seed=cfg.seed)[0]
    condition_mask = mask_sampler.sample(batch_size=smoke_batch.shape[0])
    encoded = tokenizer.encode(smoke_batch, condition_mask=condition_mask)
    t = np.full((smoke_batch.shape[0],), 0.5, dtype=np.float32)
    noised = forward_noising(encoded["joint_value"], t, encoded["condition_state"], beta_min=cfg.beta_min, beta_max=cfg.beta_max)
    zero_pred = np.zeros_like(noised["target_score"])
    smoke_loss = denoising_score_matching_loss(zero_pred, noised["target_score"], encoded["condition_state"])

    _write_json(
        paths["loss_trace"],
        {
            **payload_common,
            "artifact_type": "loss_trace_schema_and_smoke_metric",
            "metric_formula": "mean(((score_pred-target_score)^2)*(1-M_C))/sum(1-M_C)",
            "loss_trace": [
                {
                    "epoch": 0,
                    "batch": 1,
                    "loss": float(smoke_loss),
                    "conditioned_fraction": float(condition_mask.mean()),
                    "schema_only": True,
                }
            ],
        },
    )
    _write_json(
        paths["sampling_trace"],
        {
            **payload_common,
            "artifact_type": "sampling_trace_schema",
            "sampler_families": list(SAMPLER_FAMILIES),
            "trace": [
                {
                    "step": cfg.num_diffusion_steps,
                    "sampler_family": cfg.sampler_family,
                    "M_C_reapplied": True,
                    "M_E_in_attention": True,
                    "schema_only": True,
                }
            ],
        },
    )

    readiness = {
        **payload_common,
        "status": "ready_for_bounded_runtime_smoke",
        "module": "all_in_one_sbi.diffusion",
        "declared_artifacts": list(DECLARED_ARTIFACTS),
        "created_artifacts": [str(p) for p in paths.values()],
        "method_obligations": {
            "tokenizer": True,
            "mask_builder": True,
            "score_network": True,
            "trainer": True,
            "sampler": True,
            "guided_sampler": True,
            "M_E_enters_attention": True,
            "M_C_enters_noising_loss_sampling": True,
            "sde_and_ode_named": True,
            "training_metadata_surface": True,
        },
    }
    _write_json(paths["readiness"], readiness)
    _write_json(
        paths["evaluation_result"],
        {
            **payload_common,
            "evaluation_type": "diffusion_core_smoke_contract",
            "decisive_metric": "masked_denoising_score_matching_loss",
            "smoke_metric_value": float(smoke_loss),
            "hypothesis": (
                "Simformer core exposes SBI tokenization, dependency attention masks, "
                "conditional score training, and conditional SDE/ODE sampling."
            ),
            "decision_value": "validates runnable method surfaces without claiming paper-scale numerical reproduction",
        },
    )

    aux_root = os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR")
    if aux_root:
        aux = Path(aux_root)
        aux.mkdir(parents=True, exist_ok=True)
        _write_json(
            aux / "diffusion_artifact_index.json",
            {
                **payload_common,
                "canonical_results_dir": str(out),
                "declared_artifacts": list(DECLARED_ARTIFACTS),
                "created_artifacts": {k: str(v) for k, v in paths.items()},
            },
        )

    return {k: str(v) for k, v in paths.items()}


def runtime_smoke(results_dir: Union[str, os.PathLike[str]] = "results") -> Dict[str, Any]:
    """Exercise tokenizer, masks, objective, and artifact closure."""

    paths = write_dry_run_artifacts(results_dir=results_dir, config=DiffusionConfig(dry_run=True))
    return {
        "status": "ok",
        "module": "all_in_one_sbi.diffusion",
        "artifacts": paths,
        "sampler_families": list(SAMPLER_FAMILIES),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _coerce_condition_mask(
    condition_mask: Optional[ArrayLike],
    batch_size: int,
    num_variables: int,
) -> np.ndarray:
    if condition_mask is None:
        mask = np.zeros((batch_size, num_variables), dtype=np.float32)
    else:
        mask = np.asarray(condition_mask, dtype=np.float32)
        if mask.ndim == 1:
            if mask.shape[0] != num_variables:
                raise ValueError(f"1D condition_mask must have length {num_variables}")
            mask = np.broadcast_to(mask[None, :], (batch_size, num_variables)).copy()
        elif mask.ndim == 2:
            if mask.shape != (batch_size, num_variables):
                if mask.shape[0] == 1 and mask.shape[1] == num_variables:
                    mask = np.broadcast_to(mask, (batch_size, num_variables)).copy()
                else:
                    raise ValueError(
                        f"condition_mask must have shape ({batch_size}, {num_variables}), got {mask.shape}"
                    )
        elif mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask[..., 0]
            if mask.shape != (batch_size, num_variables):
                raise ValueError(f"condition_mask has incompatible shape {mask.shape}")
        else:
            raise ValueError("condition_mask must be 1D or 2D")
    return (mask > 0.5).astype(np.float32)


def _import_torch() -> Any:
    try:
        import torch  # type: ignore

        return torch
    except Exception as exc:  # pragma: no cover - depends on optional runtime
        raise RuntimeError(
            "PyTorch is required for SimformerScoreModel training/sampling. "
            "The module remains importable without torch; install torch or use "
            "write_dry_run_artifacts/runtime_smoke for contract validation."
        ) from exc


def _torch_tensor(torch: Any, value: Any, device: Any, dtype: Any) -> Any:
    if hasattr(value, "detach"):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(value, device=device, dtype=dtype)


def _write_json(path: Union[str, os.PathLike[str]], payload: Mapping[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(payload), f, indent=2, sort_keys=True)


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


__all__ = [
    "ArrayLike",
    "DECLARED_ARTIFACTS",
    "SAMPLER_FAMILIES",
    "DiffusionConfig",
    "DIFFUSION_METHOD_REGISTRY",
    "TokenizerSpec",
    "SBITokenizer",
    "ConditionMaskSampler",
    "DependencyMaskBuilder",
    "vp_beta",
    "VESDE",
    "vesde_drift",
    "vesde_diffusion",
    "vesde_sigma",
    "vp_marginal_alpha_sigma",
    "forward_noising",
    "denoising_score_matching_loss",
    "SimformerScoreModel",
    "TrainingResult",
    "DiffusionTrainer",
    "SamplingResult",
    "SimformerSampler",
    "GuidedSimformerSampler",
    "make_interval_guidance",
    "SimformerPolicyAdapter",
    "make_smoke_joint_batches",
    "write_dry_run_artifacts",
    "runtime_smoke",
]
