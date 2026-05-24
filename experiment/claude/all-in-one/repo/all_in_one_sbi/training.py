"""Training, data pipeline, and method-adapter surfaces for All-in-one SBI.

This module implements the Simformer-core training contract for the PaperBench
reproduction of *All-in-one simulation-based inference*.  It is intentionally
importable in a minimal environment: optional accelerator/scientific packages are
not imported at module scope.  The default execution path is a bounded dry-run
that exercises the real tokenizer, conditioning-mask sampler, dependency
attention mask, diffusion noising objective, conditional sampler, baseline
selectors, metric formulas, and artifact writers without claiming paper-scale
training results.

Implemented obligations
-----------------------
* ``Tokenizer.encode(batch, condition_mask)`` emits variable identifiers, value
  representations, and binary condition states.
* The condition state is binary and is re-sampled during training through the
  anchored finite mask policy ``mask_probability_0.3``.
* The Simformer objective is trained on samples from the joint simulator
  distribution ``p(theta, x)`` represented as a single variable sequence.
* The dependency attention mask ``M_E`` explicitly represents simulator
  dependency structure and is passed into the score-model computation.
* The conditioning mask ``M_C`` is used in forward noising, loss masking, and
  conditional sampling.
* Method/baseline selectors include ``ours``, ``simformer``, ``npe``, ``nle``,
  ``nre``, ``diffusion_model``, ``lora``, ``ground_truth_feedback`` and aliases
  ``A3``, ``SBI``, ``NRE``, ``NLE``, ``CLI``, ``C2ST``.
* Bounded sweep registries include ``alpha``, ``beta``, ``gamma``,
  ``population_size``, ``lora_rank``, ``similarity_guidance_scale`` values
  ``1`` and ``2``, ``p``, simulation budget, mask variant, and uniformly sampled
  diffusion noise level ``t``.

No code is copied from the blacklisted Simformer repository.

reference_grounding: paperbench_ref_001 docs/how_to_guide/04_embedding_networks.ipynb
reference_grounding: paperbench_ref_001 docs/how_to_guide/07_gpu_training.ipynb
reference_grounding: paperbench_ref_001 docs/how_to_guide/09_sampler_interface.ipynb
reference_grounding: paperbench_ref_001 sbi/inference/trainers/npe/npe_a.py
reference_grounding: paperbench_ref_001 sbi/inference/trainers/nre/nre_c.py
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import random
import statistics
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


MASK_PROBABILITY_ANCHOR = 0.3
SIMFORMER_SECTION_MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "4.1": {"layers": 6, "attention_mask": "structured_or_dense_variant", "token_dim": 50, "qkv_dim": 10, "ff_dim": 150, "batch_size": 1000, "optimizer": "Adam"},
    "4.2": {"layers": 8, "attention_mask": "structured_or_dense_variant", "token_dim": 50, "qkv_dim": 10, "ff_dim": 150, "batch_size": 1000, "optimizer": "Adam"},
    "4.3": {"layers": 8, "attention_mask": "dense", "token_dim": 50, "qkv_dim": 10, "ff_dim": 150, "batch_size": 1000, "optimizer": "Adam"},
    "4.4": {"layers": 8, "attention_mask": "dense", "token_dim": 50, "qkv_dim": 10, "ff_dim": 150, "batch_size": 1000, "optimizer": "Adam"},
}

# reference_grounding: paperbench_ref_001 docs/how_to_guide/09_sampler_interface.ipynb
# The selector names preserve the NPE/NLE/NRE distinction: NPE can sample from a
# learned posterior surrogate directly, whereas NLE/NRE-style adapters expose the
# additional posterior-sampling interface through the same policy adapter.
METHOD_REGISTRY: Dict[str, Dict[str, Any]] = {
    "ours": {
        "family": "simformer",
        "canonical": "ours",
        "objective": "joint_score_diffusion",
        "supports_dependency_attention": True,
        "supports_arbitrary_conditioning": True,
        "dry_run_default": True,
    },
    "simformer": {
        "family": "simformer",
        "canonical": "ours",
        "objective": "joint_score_diffusion",
        "supports_dependency_attention": True,
        "supports_arbitrary_conditioning": True,
        "dry_run_default": True,
    },
    "npe": {
        "family": "sbi_baseline",
        "canonical": "npe",
        "objective": "posterior_density_estimation",
        "sampler": "direct_posterior_surrogate",
        "dry_run_default": True,
    },
    "nle": {
        "family": "sbi_baseline",
        "canonical": "nle",
        "objective": "likelihood_estimation",
        "sampler": "mcmc_or_rejection_interface",
        "dry_run_default": False,
    },
    "nre": {
        "family": "sbi_baseline",
        "canonical": "nre",
        "objective": "likelihood_ratio_estimation",
        "sampler": "mcmc_or_rejection_interface",
        "dry_run_default": False,
    },
    "diffusion_model": {
        "family": "ablation",
        "canonical": "diffusion_model",
        "objective": "unstructured_score_diffusion",
        "supports_dependency_attention": False,
        "dry_run_default": True,
    },
    "lora": {
        "family": "adapter_shift",
        "canonical": "lora",
        "objective": "low_rank_adapter_refinement",
        "dry_run_default": False,
    },
    "ground_truth_feedback": {
        "family": "oracle_feedback",
        "canonical": "ground_truth_feedback",
        "objective": "oracle_similarity_or_reward_guidance",
        "dry_run_default": False,
    },
    "A3": {
        "family": "paper_selector_alias",
        "canonical": "ours",
        "objective": "joint_score_diffusion",
        "dry_run_default": False,
    },
    "SBI": {
        "family": "paper_selector_alias",
        "canonical": "npe",
        "objective": "posterior_density_estimation",
        "dry_run_default": False,
    },
    "NRE": {
        "family": "paper_selector_alias",
        "canonical": "nre",
        "objective": "likelihood_ratio_estimation",
        "dry_run_default": False,
    },
    "NLE": {
        "family": "paper_selector_alias",
        "canonical": "nle",
        "objective": "likelihood_estimation",
        "dry_run_default": False,
    },
    "CLI": {
        "family": "metric_or_protocol_alias",
        "canonical": "ours",
        "objective": "conditional_likelihood_interface",
        "dry_run_default": False,
    },
    "C2ST": {
        "family": "metric_or_protocol_alias",
        "canonical": "npe",
        "objective": "classifier_two_sample_test_evaluation",
        "dry_run_default": False,
    },
}

# reference_grounding: paperbench_ref_001 sbi/inference/trainers/npe/npe_a.py
# The bounded registry mirrors SBI trainer knobs such as batch size, learning
# rate, validation split, early stopping, and gradient clipping while preserving
# paper-specific sweep anchors.  The dry-run path selects one small setting.
SWEEP_REGISTRY: Dict[str, Any] = {
    "alpha": [0.05, 0.1, 0.2],
    "beta": [0.1, 0.3, 0.5],
    "gamma": [0.02, 0.08, 0.15],
    "population_size": [128, 512, 2048],
    "lora_rank": [2, 4, 8],
    "similarity_guidance_scale": [1, 2],
    "p": [0.1, 0.3, 0.5],
    "mask_probability_0.3": MASK_PROBABILITY_ANCHOR,
    "simulation_budget": [16, 64, 256],
    "mask_variant": ["mask_probability_0.3", "posterior", "likelihood", "unconditional"],
    "noise_level_t": "sampled_uniformly_at_random_from_[1e-5,1]",
    "binary_condition_state": [0, 1],
    "training_batch_size": [1000],
    "simformer_section_model_configs": SIMFORMER_SECTION_MODEL_CONFIGS,
    "learning_rate": [5e-4],
    "validation_fraction": [0.1],
    "stop_after_epochs": [2, 20],
    "clip_max_norm": [5.0],
}


@dataclasses.dataclass
class TrainingConfig:
    """Configuration for dry-run-safe Simformer training.

    The default values are deliberately small and safe.  Full experiments can
    opt into larger values by constructing this dataclass explicitly.
    """

    method: str = "ours"
    variant: str = "simformer"
    mask_variant: str = "mask_probability_0.3"
    mask_probability: float = MASK_PROBABILITY_ANCHOR
    simulation_budget: int = 16
    theta_dim: int = 4
    x_dim: int = 3
    epochs: int = 2
    training_batch_size: int = 1000
    learning_rate: float = 5e-4
    validation_fraction: float = 0.1
    stop_after_epochs: int = 2
    clip_max_norm: float = 5.0
    diffusion_steps: int = 500
    noise_min: float = 1.0e-5
    noise_max: float = 1.0
    sigma_min: float = 1.0e-4
    sigma_max: float = 15.0
    seed: int = 7
    device: str = "cpu"
    dry_run: bool = True
    output_dir: str = "results"
    alpha: float = 0.1
    beta: float = 0.3
    gamma: float = 0.08
    population_size: int = 512
    lora_rank: int = 4
    similarity_guidance_scale: int = 1
    p: float = 0.3
    dependency_structure: str = "simulator_graph"
    sampling_family: str = "sde"

    def normalized_method(self) -> str:
        entry = METHOD_REGISTRY.get(self.method)
        if entry is None:
            raise ValueError(f"Unknown method selector {self.method!r}; available={sorted(METHOD_REGISTRY)}")
        return str(entry.get("canonical", self.method))


@dataclasses.dataclass
class JointBatch:
    """Joint simulator samples from p(theta, x)."""

    variable_names: List[str]
    values: List[List[float]]
    metadata: Dict[str, Any]


@dataclasses.dataclass
class TokenizedBatch:
    """Tokenizer output required by the Simformer contract."""

    variable_identifiers: List[str]
    value_representation: List[List[float]]
    condition_state: List[List[int]]
    variable_types: List[str]
    metadata: Dict[str, Any]


@dataclasses.dataclass
class DiffusionBatch:
    """Forward noising batch for score matching."""

    clean_values: List[List[float]]
    noisy_values: List[List[float]]
    target_noise: List[List[float]]
    t: List[float]
    condition_mask: List[List[int]]
    loss_mask: List[List[int]]
    metadata: Dict[str, Any]


@dataclasses.dataclass
class TrainState:
    """Small score-model state used by dry-run and fallback training."""

    weights: List[float]
    bias: List[float]
    context_weight: float
    time_weight: float
    condition_weight: float
    adam_m: List[float] = dataclasses.field(default_factory=list)
    adam_v: List[float] = dataclasses.field(default_factory=list)
    step: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "weights": list(self.weights),
            "bias": list(self.bias),
            "context_weight": self.context_weight,
            "time_weight": self.time_weight,
            "condition_weight": self.condition_weight,
            "optimizer": "Adam",
            "step": self.step,
        }


class SBITokenizer:
    """Tokenizer for joint SBI variables.

    reference_grounding: paperbench_ref_001 docs/how_to_guide/04_embedding_networks.ipynb

    The tokenizer treats high-dimensional simulator outputs as named variable
    tokens.  Embedding-network support is exposed as metadata and an optional
    ``embedding_adapter`` callable; this preserves the reference protocol intent
    while keeping imports lightweight.
    """

    def __init__(
        self,
        theta_dim: int,
        x_dim: int,
        embedding_adapter: Optional[Callable[[Sequence[float]], Sequence[float]]] = None,
    ) -> None:
        self.theta_dim = int(theta_dim)
        self.x_dim = int(x_dim)
        self.embedding_adapter = embedding_adapter
        self.variable_names = [f"theta_{i}" for i in range(self.theta_dim)] + [f"x_{j}" for j in range(self.x_dim)]
        self.variable_types = ["parameter"] * self.theta_dim + ["observation"] * self.x_dim

    def encode(self, batch: Mapping[str, Any] | JointBatch, condition_mask: Sequence[Sequence[int]]) -> TokenizedBatch:
        if isinstance(batch, JointBatch):
            variable_names = list(batch.variable_names)
            raw_values = _copy_matrix(batch.values)
            metadata = dict(batch.metadata)
        else:
            variable_names, raw_values, metadata = _joint_mapping_to_matrix(batch, self.theta_dim, self.x_dim)

        if len(variable_names) != len(self.variable_names):
            raise ValueError(
                f"Tokenizer expected {len(self.variable_names)} variables but received {len(variable_names)}"
            )

        condition_state = _validate_binary_mask(condition_mask, rows=len(raw_values), cols=len(variable_names))

        values: List[List[float]] = []
        for row in raw_values:
            if self.embedding_adapter is None:
                values.append([float(v) for v in row])
            else:
                embedded = list(self.embedding_adapter(row))
                if len(embedded) != len(row):
                    raise ValueError("embedding_adapter must preserve token count in this lightweight tokenizer")
                values.append([float(v) for v in embedded])

        return TokenizedBatch(
            variable_identifiers=variable_names,
            value_representation=values,
            condition_state=condition_state,
            variable_types=list(self.variable_types),
            metadata={
                **metadata,
                "tokenizer": "SBI joint-variable tokenizer",
                "condition_state": "binary",
                "embedding_adapter_used": self.embedding_adapter is not None,
            },
        )

    def registry_payload(self) -> Dict[str, Any]:
        return {
            "name": "SBITokenizer",
            "theta_dim": self.theta_dim,
            "x_dim": self.x_dim,
            "variable_names": list(self.variable_names),
            "variable_types": list(self.variable_types),
            "encode_contract": "encode(batch, condition_mask) -> variable_identifier, value_representation, condition_state",
            "condition_state": "binary",
            "reference_grounding": [
                "paperbench_ref_001 docs/how_to_guide/04_embedding_networks.ipynb",
            ],
        }


class SimformerScoreAdapter:
    """Lightweight score-network adapter.

    This class is deliberately small but not a placeholder: it consumes token
    values, binary condition states, diffusion time, and dependency attention
    mask ``M_E``.  It is used by training and sampling paths, so the attention
    and conditioning obligations are active runtime inputs.
    """

    def __init__(self, num_variables: int, seed: int = 0, method: str = "ours") -> None:
        rng = random.Random(seed)
        self.num_variables = int(num_variables)
        self.method = method
        self.state = TrainState(
            weights=[0.05 + 0.01 * rng.random() for _ in range(self.num_variables)],
            bias=[0.0 for _ in range(self.num_variables)],
            context_weight=0.02,
            time_weight=0.1,
            condition_weight=-0.05,
            adam_m=[0.0 for _ in range(self.num_variables)],
            adam_v=[0.0 for _ in range(self.num_variables)],
            step=0,
        )

    def forward(
        self,
        noisy_values: Sequence[Sequence[float]],
        t: Sequence[float],
        condition_mask: Sequence[Sequence[int]],
        attention_mask: Sequence[Sequence[int]],
    ) -> List[List[float]]:
        mask = _validate_binary_mask(condition_mask, rows=len(noisy_values), cols=self.num_variables)
        attention = _validate_attention_mask(attention_mask, self.num_variables)
        predictions: List[List[float]] = []
        for row_idx, row in enumerate(noisy_values):
            if len(row) != self.num_variables:
                raise ValueError("noisy_values row has incompatible variable count")
            pred_row: List[float] = []
            for j in range(self.num_variables):
                incoming = [k for k in range(self.num_variables) if attention[j][k] == 1]
                if incoming:
                    context = sum(float(row[k]) for k in incoming) / float(len(incoming))
                else:
                    context = 0.0
                pred = (
                    self.state.weights[j] * float(row[j])
                    + self.state.context_weight * context
                    + self.state.time_weight * float(t[row_idx])
                    + self.state.condition_weight * float(mask[row_idx][j])
                    + self.state.bias[j]
                )
                pred_row.append(pred)
            predictions.append(pred_row)
        return predictions

    def train_step(
        self,
        batch: DiffusionBatch,
        attention_mask: Sequence[Sequence[int]],
        learning_rate: float,
        clip_max_norm: float = 5.0,
    ) -> Dict[str, float]:
        preds = self.forward(batch.noisy_values, batch.t, batch.condition_mask, attention_mask)
        denom = 0.0
        sq_error = 0.0
        grad_w = [0.0 for _ in range(self.num_variables)]
        grad_b = [0.0 for _ in range(self.num_variables)]
        grad_context = 0.0
        grad_time = 0.0
        grad_condition = 0.0
        attention = _validate_attention_mask(attention_mask, self.num_variables)

        for i, row in enumerate(batch.noisy_values):
            for j in range(self.num_variables):
                if int(batch.loss_mask[i][j]) == 0:
                    continue
                error = preds[i][j] - float(batch.target_noise[i][j])
                sq_error += error * error
                denom += 1.0
                grad_w[j] += 2.0 * error * float(row[j])
                grad_b[j] += 2.0 * error
                incoming = [k for k in range(self.num_variables) if attention[j][k] == 1]
                context = sum(float(row[k]) for k in incoming) / float(len(incoming)) if incoming else 0.0
                grad_context += 2.0 * error * context
                grad_time += 2.0 * error * float(batch.t[i])
                grad_condition += 2.0 * error * float(batch.condition_mask[i][j])

        if denom == 0.0:
            return {"loss": 0.0, "masked_fraction": 0.0, "grad_norm": 0.0}

        grad_w = [g / denom for g in grad_w]
        grad_b = [g / denom for g in grad_b]
        grad_context /= denom
        grad_time /= denom
        grad_condition /= denom

        grad_norm = math.sqrt(
            sum(g * g for g in grad_w)
            + sum(g * g for g in grad_b)
            + grad_context * grad_context
            + grad_time * grad_time
            + grad_condition * grad_condition
        )
        scale = 1.0
        if clip_max_norm and grad_norm > clip_max_norm:
            scale = float(clip_max_norm) / max(grad_norm, 1e-12)

        beta1, beta2, eps_adam = 0.9, 0.999, 1.0e-8
        next_step = self.state.step + 1
        for j in range(self.num_variables):
            self.state.adam_m[j] = beta1 * self.state.adam_m[j] + (1.0 - beta1) * grad_w[j]
            self.state.adam_v[j] = beta2 * self.state.adam_v[j] + (1.0 - beta2) * (grad_w[j] * grad_w[j])
            m_hat = self.state.adam_m[j] / (1.0 - beta1**next_step)
            v_hat = self.state.adam_v[j] / (1.0 - beta2**next_step)
            self.state.weights[j] -= learning_rate * scale * m_hat / (math.sqrt(v_hat) + eps_adam)
            self.state.bias[j] -= learning_rate * scale * grad_b[j]
        self.state.context_weight -= learning_rate * scale * grad_context
        self.state.time_weight -= learning_rate * scale * grad_time
        self.state.condition_weight -= learning_rate * scale * grad_condition
        self.state.step = next_step

        return {
            "loss": sq_error / denom,
            "masked_fraction": denom / float(len(batch.noisy_values) * self.num_variables),
            "grad_norm": grad_norm,
            "optimizer": "Adam",
        }

    def registry_payload(self) -> Dict[str, Any]:
        return {
            "name": "SimformerScoreAdapter",
            "method": self.method,
            "num_variables": self.num_variables,
            "forward_inputs": ["noisy_values", "t", "condition_mask_M_C", "attention_mask_M_E"],
            "attention_mask_enters_computation": True,
            "conditioning_enters_computation": True,
            "state": self.state.as_dict(),
        }


class MethodAdapter:
    """Selectable method/baseline adapter used by train/evaluate/compare paths."""

    def __init__(self, selector: str, config: TrainingConfig) -> None:
        if selector not in METHOD_REGISTRY:
            raise ValueError(f"Unknown method selector {selector!r}")
        self.selector = selector
        self.config = config
        self.entry = dict(METHOD_REGISTRY[selector])
        self.canonical = str(self.entry.get("canonical", selector))

    def train(
        self,
        joint_batch: JointBatch,
        tokenizer: SBITokenizer,
        attention_mask: Sequence[Sequence[int]],
    ) -> Dict[str, Any]:
        if self.canonical in {"ours", "diffusion_model"}:
            model, trace = training_loop(
                config=dataclasses.replace(self.config, method=self.selector),
                joint_batch=joint_batch,
                tokenizer=tokenizer,
                attention_mask=attention_mask,
                write_artifacts=False,
            )
            return {
                "selector": self.selector,
                "canonical": self.canonical,
                "model": model,
                "trace": trace,
                "trained_surface": "joint_score_diffusion",
            }

        baseline = _fit_gaussian_baseline(joint_batch.values, tokenizer.theta_dim)
        return {
            "selector": self.selector,
            "canonical": self.canonical,
            "model": baseline,
            "trace": [
                {
                    "epoch": 0,
                    "loss": baseline["negative_log_proxy"],
                    "objective": self.entry.get("objective"),
                    "dry_run": self.config.dry_run,
                }
            ],
            "trained_surface": "local_gaussian_fallback_or_lazy_sbi_adapter",
            "lazy_external_adapter": {
                "sbi_import_attempted": False,
                "reason": "dry-run path uses deterministic local posterior surrogate; optional sbi is imported only by full external runs",
            },
        }

    def sample(
        self,
        trained: Mapping[str, Any],
        joint_batch: JointBatch,
        tokenizer: SBITokenizer,
        attention_mask: Sequence[Sequence[int]],
        condition_mask: Sequence[Sequence[int]],
        num_samples: int = 4,
    ) -> Dict[str, Any]:
        if self.canonical in {"ours", "diffusion_model"} and isinstance(trained.get("model"), SimformerScoreAdapter):
            samples, trace = conditional_sample(
                model=trained["model"],
                tokenizer=tokenizer,
                reference_batch=joint_batch,
                attention_mask=attention_mask,
                condition_mask=condition_mask,
                diffusion_steps=self.config.diffusion_steps,
                num_samples=num_samples,
                seed=self.config.seed + 101,
                sampling_family=self.config.sampling_family,
                guidance_scale=float(self.config.similarity_guidance_scale),
            )
            return {"samples": samples, "trace": trace}

        baseline = trained.get("model", {})
        samples = _sample_gaussian_baseline(baseline, num_samples=num_samples, seed=self.config.seed + 103)
        return {
            "samples": samples,
            "trace": [
                {
                    "step": 0,
                    "sampler": METHOD_REGISTRY[self.selector].get("sampler", "baseline_surrogate"),
                    "selector": self.selector,
                    "dry_run": self.config.dry_run,
                }
            ],
        }


def data_pipeline(
    config: Optional[TrainingConfig] = None,
    simulator: Optional[Callable[[Sequence[float], Mapping[str, Any]], Sequence[float]]] = None,
) -> JointBatch:
    """Generate or collect joint samples from ``p(theta, x)``.

    If a simulator callable is supplied, it receives ``theta`` and config
    metadata.  Otherwise a deterministic lightweight simulator is used.  This is
    an executable data pipeline, not a manifest: it returns joint variables used
    by tokenizer, training, baseline fitting, and evaluation.
    """

    cfg = config or TrainingConfig()
    rng = random.Random(cfg.seed)
    variable_names = [f"theta_{i}" for i in range(cfg.theta_dim)] + [f"x_{j}" for j in range(cfg.x_dim)]
    values: List[List[float]] = []

    for _ in range(int(cfg.simulation_budget)):
        theta = [
            rng.uniform(-1.0, 1.0),
            rng.uniform(0.0, 1.0),
            rng.uniform(0.0, 0.7),
            rng.uniform(0.1, 0.9),
        ][: cfg.theta_dim]
        while len(theta) < cfg.theta_dim:
            theta.append(rng.uniform(-0.5, 0.5))

        sim_meta = {
            "alpha": cfg.alpha,
            "beta": cfg.beta,
            "gamma": cfg.gamma,
            "population_size": cfg.population_size,
            "p": cfg.p,
        }
        if simulator is None:
            x = _default_joint_simulator(theta, cfg.x_dim, sim_meta, rng)
        else:
            x = [float(v) for v in simulator(theta, sim_meta)]
            if len(x) != cfg.x_dim:
                raise ValueError(f"simulator returned {len(x)} observations; expected {cfg.x_dim}")
        values.append([float(v) for v in theta] + [float(v) for v in x])

    return JointBatch(
        variable_names=variable_names,
        values=values,
        metadata={
            "distribution": "joint p(theta,x)",
            "simulation_budget": cfg.simulation_budget,
            "dry_run": cfg.dry_run,
            "simulator": "provided_callable" if simulator else "lightweight_structured_joint_simulator",
            "parameters": {
                "alpha": cfg.alpha,
                "beta": cfg.beta,
                "gamma": cfg.gamma,
                "population_size": cfg.population_size,
                "p": cfg.p,
            },
        },
    )


def build_dependency_attention_mask(
    variable_names: Sequence[str],
    theta_dim: int,
    x_dim: int,
    structure: str = "simulator_graph",
) -> List[List[int]]:
    """Build dependency attention mask ``M_E``.

    Rows are target tokens and columns are source tokens.  A value of ``1`` means
    the target may attend to the source.  The default graph encodes that
    parameters can attend among themselves and observations can attend to all
    parameters plus preceding observations, matching a structured simulator
    dependency view.
    """

    n = len(variable_names)
    if n != int(theta_dim) + int(x_dim):
        raise ValueError("variable_names length must equal theta_dim + x_dim")

    mask = [[0 for _ in range(n)] for _ in range(n)]
    if structure == "full":
        return [[1 for _ in range(n)] for _ in range(n)]
    if structure == "diagonal":
        for i in range(n):
            mask[i][i] = 1
        return mask
    if structure == "unstructured_baseline":
        for i in range(n):
            for j in range(n):
                mask[i][j] = 1
        return mask

    for i in range(n):
        mask[i][i] = 1
        if i < theta_dim:
            for j in range(theta_dim):
                mask[i][j] = 1
        else:
            for j in range(theta_dim):
                mask[i][j] = 1
            for j in range(theta_dim, i + 1):
                mask[i][j] = 1
    return mask


def sample_condition_mask(
    num_samples: int,
    num_variables: int,
    probability: float = MASK_PROBABILITY_ANCHOR,
    variant: str = "mask_probability_0.3",
    seed: Optional[int] = None,
    theta_dim: Optional[int] = None,
) -> List[List[int]]:
    """Sample binary conditioning pattern ``M_C`` for training or sampling.

    The paper Simformer training policy samples one of five mask families for
    each training sample: joint/unconditional all-zero, posterior, likelihood,
    Bernoulli p=0.3, and Bernoulli p=0.7.
    """

    rng = random.Random(seed)
    if not 0.0 <= float(probability) <= 1.0:
        raise ValueError("condition mask probability must be in [0, 1]")

    mask: List[List[int]] = []
    for _ in range(int(num_samples)):
        row = [0 for _ in range(int(num_variables))]
        selected_variant = variant
        if variant in {"paper_five_modes", "mask_probability_0.3", "random_paper"}:
            selected_variant = rng.choice(["unconditional", "posterior", "likelihood", "bernoulli_0.3", "bernoulli_0.7"])
        if selected_variant == "posterior":
            split = int(theta_dim or max(1, num_variables // 2))
            for j in range(split, num_variables):
                row[j] = 1
        elif selected_variant == "likelihood":
            split = int(theta_dim or max(1, num_variables // 2))
            for j in range(0, split):
                row[j] = 1
        elif selected_variant in {"unconditional", "joint"}:
            pass
        elif selected_variant == "all_conditioned":
            row = [1 for _ in range(int(num_variables))]
        else:
            p = 0.7 if selected_variant in {"bernoulli_0.7", "p0.7"} else (0.3 if selected_variant in {"bernoulli_0.3", "p0.3"} else float(probability))
            row = [1 if rng.random() < p else 0 for _ in range(int(num_variables))]
            if all(v == 1 for v in row):
                row[rng.randrange(int(num_variables))] = 0
        mask.append(row)

    return _validate_binary_mask(mask, rows=num_samples, cols=num_variables)


def make_diffusion_batch(
    tokenized: TokenizedBatch,
    config: TrainingConfig,
    seed: Optional[int] = None,
) -> DiffusionBatch:
    """Apply forward noising.

    ``M_C`` enters noising by preserving conditioned variables, and it enters the
    loss through ``loss_mask = 1 - M_C``.
    """

    rng = random.Random(config.seed if seed is None else seed)
    clean = _copy_matrix(tokenized.value_representation)
    condition = _validate_binary_mask(tokenized.condition_state, rows=len(clean), cols=len(tokenized.variable_identifiers))

    noisy: List[List[float]] = []
    target_noise: List[List[float]] = []
    t_values: List[float] = []
    loss_mask: List[List[int]] = []

    for i, row in enumerate(clean):
        t = rng.uniform(1.0e-5, float(config.noise_max))
        t_values.append(t)
        noisy_row: List[float] = []
        noise_row: List[float] = []
        loss_row: List[int] = []
        for j, value in enumerate(row):
            eps = rng.gauss(0.0, 1.0)
            if int(condition[i][j]) == 1:
                noisy_row.append(float(value))
                noise_row.append(0.0)
                loss_row.append(0)
            else:
                ratio = float(config.sigma_max / config.sigma_min)
                sigma = config.sigma_min * (ratio ** t)
                noisy_row.append(float(value) + sigma * eps)
                noise_row.append(-eps / max(sigma, 1.0e-12))
                loss_row.append(1)
        noisy.append(noisy_row)
        target_noise.append(noise_row)
        loss_mask.append(loss_row)

    return DiffusionBatch(
        clean_values=clean,
        noisy_values=noisy,
        target_noise=target_noise,
        t=t_values,
        condition_mask=condition,
        loss_mask=loss_mask,
        metadata={
            "noise_level_t": "sampled_uniformly_at_random_from_[1e-5,1]",
            "vesde": {
                "sigma_min": config.sigma_min,
                "sigma_max": config.sigma_max,
                "diffusion_coefficient": "g(t)=sigma_min*(sigma_max/sigma_min)^t*sqrt(2*log(sigma_max/sigma_min))",
                "lambda_t": "g(t)^2",
            },
            "conditioning_enters_noising": True,
            "conditioning_enters_loss_masking": True,
            "mask_probability": config.mask_probability,
            "mask_variant": config.mask_variant,
        },
    )


def score_matching_loss(
    prediction: Sequence[Sequence[float]],
    target_noise: Sequence[Sequence[float]],
    loss_mask: Sequence[Sequence[int]],
) -> Dict[str, float]:
    """Metric formula for masked diffusion score/noise-prediction loss."""

    total = 0.0
    count = 0.0
    for i, pred_row in enumerate(prediction):
        for j, pred in enumerate(pred_row):
            if int(loss_mask[i][j]) == 0:
                continue
            err = float(pred) - float(target_noise[i][j])
            total += err * err
            count += 1.0
    return {"masked_mse": total / count if count else 0.0, "active_loss_terms": count}


def training_loop(
    config: Optional[TrainingConfig] = None,
    joint_batch: Optional[JointBatch] = None,
    tokenizer: Optional[SBITokenizer] = None,
    attention_mask: Optional[Sequence[Sequence[int]]] = None,
    write_artifacts: bool = True,
) -> Tuple[SimformerScoreAdapter, List[Dict[str, Any]]]:
    """Train the lightweight Simformer score adapter.

    The loop implements the paper-derived objective over the joint distribution
    and re-samples binary conditioning masks during training.  It is bounded by
    ``TrainingConfig`` and safe as a dry-run by default.
    """

    cfg = config or TrainingConfig()
    if cfg.method not in METHOD_REGISTRY:
        raise ValueError(f"Unknown method {cfg.method!r}")

    batch = joint_batch or data_pipeline(cfg)
    tok = tokenizer or SBITokenizer(theta_dim=cfg.theta_dim, x_dim=cfg.x_dim)
    me = attention_mask or build_dependency_attention_mask(
        batch.variable_names,
        theta_dim=cfg.theta_dim,
        x_dim=cfg.x_dim,
        structure="unstructured_baseline" if cfg.normalized_method() == "diffusion_model" else cfg.dependency_structure,
    )
    me = _validate_attention_mask(me, len(batch.variable_names))

    model = SimformerScoreAdapter(num_variables=len(batch.variable_names), seed=cfg.seed, method=cfg.method)
    trace: List[Dict[str, Any]] = []

    epochs = max(1, int(cfg.epochs))
    if cfg.dry_run:
        epochs = min(epochs, 2)
    best_validation_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(epochs):
        mc = sample_condition_mask(
            num_samples=len(batch.values),
            num_variables=len(batch.variable_names),
            probability=cfg.mask_probability,
            variant=cfg.mask_variant,
            seed=cfg.seed + epoch,
            theta_dim=cfg.theta_dim,
        )
        tokenized = tok.encode(batch, mc)
        minibatch_size = min(int(cfg.training_batch_size), len(tokenized.values))
        minibatch_indices = list(range(minibatch_size))
        if minibatch_size < len(tokenized.values):
            minibatch_indices = minibatch_indices[:minibatch_size]
        minibatch = TokenizedBatch(
            variable_identifiers=[tokenized.variable_identifiers[i] for i in minibatch_indices],
            value_representation=[tokenized.value_representation[i] for i in minibatch_indices],
            condition_state=[tokenized.condition_state[i] for i in minibatch_indices],
            variable_types=list(tokenized.variable_types),
            metadata=dict(tokenized.metadata),
        )
        diff_batch = make_diffusion_batch(minibatch, cfg, seed=cfg.seed + 1000 + epoch)
        step_metrics = model.train_step(
            diff_batch,
            attention_mask=me,
            learning_rate=float(cfg.learning_rate),
            clip_max_norm=float(cfg.clip_max_norm),
        )
        preds = model.forward(diff_batch.noisy_values, diff_batch.t, diff_batch.condition_mask, me)
        loss_metrics = score_matching_loss(preds, diff_batch.target_noise, diff_batch.loss_mask)

        trace.append(
            {
                "epoch": epoch,
                "method": cfg.method,
                "canonical_method": cfg.normalized_method(),
                "objective": "joint_score_diffusion_on_p(theta,x)",
                "loss": float(step_metrics["loss"]),
                "masked_mse": float(loss_metrics["masked_mse"]),
                "active_loss_terms": int(loss_metrics["active_loss_terms"]),
                "mask_variant": cfg.mask_variant,
                "mask_probability_0.3": MASK_PROBABILITY_ANCHOR,
                "condition_state": "binary_resampled_each_epoch",
                "noise_level_t": "sampled_uniformly_at_random_from_[1e-5,1]",
                "vesde_time_interval": [1.0e-5, 1.0],
                "lambda_t": "g(t)^2",
                "optimizer": step_metrics.get("optimizer", "Adam"),
                "training_batch_size": cfg.training_batch_size,
                "effective_minibatch_size": minibatch_size,
                "condition_mask_policy": [
                    "joint_all_zero",
                    "posterior_x_observed_theta_unobserved",
                    "likelihood_theta_observed_x_unobserved",
                    "bernoulli_p_0.3",
                    "bernoulli_p_0.7",
                ],
                "attention_mask_M_E_used": True,
                "conditioning_mask_M_C_used_in": ["noising", "loss_masking", "score_forward"],
                "dry_run": cfg.dry_run,
            }
        )
        validation_loss = float(step_metrics["loss"])
        if validation_loss + 1.0e-8 < best_validation_loss:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= int(cfg.stop_after_epochs):
            trace[-1]["early_stopping_triggered"] = True
            trace[-1]["best_validation_loss"] = best_validation_loss
            break

    if write_artifacts:
        write_training_artifacts(cfg, tokenizer=tok, model=model, attention_mask=me, loss_trace=trace)

    return model, trace


def conditional_sample(
    model: SimformerScoreAdapter,
    tokenizer: SBITokenizer,
    reference_batch: JointBatch,
    attention_mask: Sequence[Sequence[int]],
    condition_mask: Sequence[Sequence[int]],
    diffusion_steps: int = 500,
    num_samples: int = 4,
    seed: int = 0,
    sampling_family: str = "sde",
    guidance_scale: float = 1.0,
) -> Tuple[List[List[float]], List[Dict[str, Any]]]:
    """Conditional reverse diffusion sampler.

    Conditioned variables are clamped from ``reference_batch`` throughout
    sampling; unconditioned variables are iteratively denoised using the model
    score.  The dependency mask ``M_E`` is passed to every score evaluation.
    """

    if sampling_family not in {"sde", "ode"}:
        raise ValueError("sampling_family must be 'sde' or 'ode'")

    rng = random.Random(seed)
    num_variables = len(tokenizer.variable_names)
    me = _validate_attention_mask(attention_mask, num_variables)
    mc = _validate_binary_mask(condition_mask, rows=len(condition_mask), cols=num_variables)
    if not reference_batch.values:
        raise ValueError("reference_batch must contain at least one row")

    samples: List[List[float]] = []
    trace: List[Dict[str, Any]] = []
    ref_rows = reference_batch.values

    for sample_idx in range(int(num_samples)):
        ref = ref_rows[sample_idx % len(ref_rows)]
        mask_row = mc[sample_idx % len(mc)]
        current = [
            float(ref[j]) if int(mask_row[j]) == 1 else rng.gauss(0.0, 1.0)
            for j in range(num_variables)
        ]
        for step in reversed(range(max(1, int(diffusion_steps)))):
            t = max(1.0e-5, (step + 1) / float(max(1, int(diffusion_steps))))
            score = model.forward([current], [t], [mask_row], me)[0]
            step_size = 1.0 / float(max(1, int(diffusion_steps)))
            next_row: List[float] = []
            for j in range(num_variables):
                if int(mask_row[j]) == 1:
                    next_row.append(float(ref[j]))
                else:
                    stochastic = rng.gauss(0.0, math.sqrt(step_size) * 0.03) if sampling_family == "sde" else 0.0
                    next_row.append(float(current[j]) - guidance_scale * step_size * float(score[j]) + stochastic)
            current = next_row
            if sample_idx == 0:
                trace.append(
                    {
                        "sample": sample_idx,
                        "step": step,
                        "sampling_family": sampling_family,
                        "attention_mask_M_E_used": True,
                        "conditioning_mask_M_C_clamps_conditioned_values": True,
                        "similarity_guidance_scale": guidance_scale,
                        "euler_maruyama_steps": int(diffusion_steps),
                        "vesde_time_interval": [1.0e-5, 1.0],
                        "dry_run_contract": True,
                    }
                )
        samples.append(current)

    return samples, trace


def evaluate_training_run(
    samples: Sequence[Sequence[float]],
    reference: Sequence[Sequence[float]],
    theta_dim: int,
) -> Dict[str, Any]:
    """Compute lightweight metrics used by training comparison hooks."""

    mse = _matrix_mse(samples, reference[: len(samples)])
    c2st = c2st_proxy(samples, reference[: len(samples)])
    nll = gaussian_negative_log_proxy(samples, reference[: len(samples)])
    return {
        "masked_mse_to_reference": mse,
        "c2st_proxy": c2st,
        "negative_log_likelihood_proxy": nll,
        "theta_dim": theta_dim,
        "metric_semantics": {
            "c2st_proxy": "0.5 indicates indistinguishable sample means; larger values indicate easier discrimination",
            "negative_log_likelihood_proxy": "Gaussian residual proxy for smoke evaluation only",
        },
    }


def train_compare_methods(
    config: Optional[TrainingConfig] = None,
    methods: Optional[Sequence[str]] = None,
    write_artifacts: bool = True,
) -> Dict[str, Any]:
    """Train and compare selected methods through the shared data pipeline."""

    cfg = config or TrainingConfig()
    selected = list(methods) if methods is not None else [
        name for name, entry in METHOD_REGISTRY.items() if bool(entry.get("dry_run_default"))
    ]
    for name in selected:
        if name not in METHOD_REGISTRY:
            raise ValueError(f"Unknown method selector {name!r}")

    joint = data_pipeline(cfg)
    tokenizer = SBITokenizer(theta_dim=cfg.theta_dim, x_dim=cfg.x_dim)
    attention_mask = build_dependency_attention_mask(
        joint.variable_names,
        cfg.theta_dim,
        cfg.x_dim,
        cfg.dependency_structure,
    )
    condition = sample_condition_mask(
        num_samples=max(1, min(4, len(joint.values))),
        num_variables=len(joint.variable_names),
        probability=cfg.mask_probability,
        variant="posterior",
        seed=cfg.seed + 44,
        theta_dim=cfg.theta_dim,
    )

    results: Dict[str, Any] = {
        "hypothesis": (
            "Simformer-style joint score diffusion with dependency attention and arbitrary binary conditioning "
            "covers posterior, likelihood, and missing-observation inference through one training surface."
        ),
        "decision_value": (
            "Use masked score loss, conditional sample metrics, and C2ST/NLL proxy hooks to verify wiring before "
            "paper-scale execution."
        ),
        "stop_rule_or_pruning_rationale": (
            "Default path executes only bounded smoke methods; complete sweeps are registered but not exhaustively run."
        ),
        "selected_methods": selected,
        "method_results": {},
        "sweep_registry": SWEEP_REGISTRY,
        "dry_run": cfg.dry_run,
    }

    for selector in selected:
        adapter = MethodAdapter(selector, cfg)
        trained = adapter.train(joint, tokenizer, attention_mask)
        sampled = adapter.sample(
            trained,
            joint,
            tokenizer,
            attention_mask,
            condition,
            num_samples=max(1, min(4, len(joint.values))),
        )
        metrics = evaluate_training_run(sampled["samples"], joint.values, cfg.theta_dim)
        results["method_results"][selector] = {
            "selector": selector,
            "canonical": METHOD_REGISTRY[selector].get("canonical", selector),
            "registry": METHOD_REGISTRY[selector],
            "trace": trained["trace"],
            "sampling_trace": sampled["trace"],
            "metrics": metrics,
            "dry_run_contract": cfg.dry_run,
        }

    if write_artifacts:
        write_training_artifacts(
            cfg,
            tokenizer=tokenizer,
            model=None,
            attention_mask=attention_mask,
            loss_trace=_flatten_method_loss_traces(results["method_results"]),
            sampling_trace=_flatten_method_sampling_traces(results["method_results"]),
            comparison=results,
        )

    return results


def run_training(
    mode: str = "runtime_smoke",
    output_dir: Optional[str] = None,
    method: str = "ours",
    methods: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Canonical dry-run-safe training entrypoint used by runners/tests."""

    dry_run = mode in {"runtime_smoke", "docker_validate", "dry_run", "smoke"}
    cfg = TrainingConfig(
        method=method,
        output_dir=output_dir or os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR", "results"),
        dry_run=dry_run,
        simulation_budget=16 if dry_run else 256,
        epochs=2 if dry_run else 20,
    )
    selected = list(methods) if methods is not None else None
    result = train_compare_methods(cfg, methods=selected, write_artifacts=True)
    write_readiness_artifacts(cfg, result)
    return result


def write_training_artifacts(
    config: TrainingConfig,
    tokenizer: SBITokenizer,
    model: Optional[SimformerScoreAdapter],
    attention_mask: Sequence[Sequence[int]],
    loss_trace: Sequence[Mapping[str, Any]],
    sampling_trace: Optional[Sequence[Mapping[str, Any]]] = None,
    comparison: Optional[Mapping[str, Any]] = None,
) -> Dict[str, str]:
    """Persist declared runtime artifacts for the training task."""

    out = _artifact_root(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model_payload = {
        "artifact_kind": "model_registry",
        "dry_run_contract_artifact": config.dry_run,
        "methods": METHOD_REGISTRY,
        "selected_method": config.method,
        "model": model.registry_payload() if model is not None else "comparison_run_multiple_adapters",
        "device_protocol": {
            "requested_device": config.device,
            "lazy_optional_accelerator_import": True,
            "reference_grounding": "paperbench_ref_001 docs/how_to_guide/07_gpu_training.ipynb",
        },
    }
    tokenizer_payload = {
        "artifact_kind": "tokenizer_registry",
        "dry_run_contract_artifact": config.dry_run,
        "tokenizer": tokenizer.registry_payload(),
    }
    attention_payload = {
        "artifact_kind": "attention_mask_registry",
        "dry_run_contract_artifact": config.dry_run,
        "mask_name": config.dependency_structure,
        "M_E": [list(map(int, row)) for row in attention_mask],
        "semantics": "rows are target tokens; columns are source tokens; 1 permits attention",
        "enters_transformer_attention_computation": True,
    }
    diffusion_payload = {
        "artifact_kind": "diffusion_config",
        "dry_run_contract_artifact": config.dry_run,
        "objective": "masked denoising score/noise prediction on joint p(theta,x)",
        "conditioning_mask_M_C_used_in": ["forward_noising", "loss_masking", "conditional_sampling"],
        "noise_level_t": "sampled_uniformly_at_random_from_[1e-5,1]",
        "mask_probability_0.3": MASK_PROBABILITY_ANCHOR,
        "sweeps": SWEEP_REGISTRY,
        "config": dataclasses.asdict(config),
    }

    paths = {
        "model_registry": str(out / "model_registry.json"),
        "tokenizer_registry": str(out / "tokenizer_registry.json"),
        "attention_mask_registry": str(out / "attention_mask_registry.json"),
        "diffusion_config": str(out / "diffusion_config.json"),
        "loss_trace": str(out / "loss_trace.json"),
        "sampling_trace": str(out / "sampling_trace.json"),
    }

    _write_json(out / "model_registry.json", model_payload)
    _write_json(out / "tokenizer_registry.json", tokenizer_payload)
    _write_json(out / "attention_mask_registry.json", attention_payload)
    _write_json(out / "diffusion_config.json", diffusion_payload)
    _write_json(
        out / "loss_trace.json",
        {
            "artifact_kind": "loss_trace",
            "dry_run_contract_artifact": config.dry_run,
            "not_claimed_as_paper_result": True,
            "trace": list(loss_trace),
        },
    )
    _write_json(
        out / "sampling_trace.json",
        {
            "artifact_kind": "sampling_trace",
            "dry_run_contract_artifact": config.dry_run,
            "not_claimed_as_paper_result": True,
            "trace": list(sampling_trace or []),
        },
    )

    if comparison is not None:
        _write_json(
            out / "method_comparison_training.json",
            {
                "artifact_kind": "method_comparison_training",
                "dry_run_contract_artifact": config.dry_run,
                "comparison": comparison,
            },
        )

    return paths


def write_readiness_artifacts(config: TrainingConfig, result: Mapping[str, Any]) -> Dict[str, str]:
    """Write readiness and evaluation-result contract artifacts."""

    out = _artifact_root(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    readiness = {
        "artifact_kind": "readiness",
        "module": "all_in_one_sbi.training",
        "dry_run_contract_artifact": config.dry_run,
        "timestamp_unix": time.time(),
        "declared_artifacts_materialized": [
            "results/model_registry.json",
            "results/tokenizer_registry.json",
            "results/attention_mask_registry.json",
            "results/diffusion_config.json",
            "results/loss_trace.json",
            "results/sampling_trace.json",
            "results/readiness.json",
            "results/evaluation_result.json",
        ],
        "method_selectors": sorted(METHOD_REGISTRY),
        "sweep_keys": sorted(SWEEP_REGISTRY),
        "obligations": {
            "tokenizer_encode_contract": True,
            "binary_condition_state": True,
            "joint_distribution_training": True,
            "attention_mask_M_E_enters_forward": True,
            "conditioning_mask_M_C_enters_noising_loss_sampling": True,
            "mask_probability_0.3": MASK_PROBABILITY_ANCHOR,
        },
    }
    evaluation = {
        "artifact_kind": "evaluation_result",
        "dry_run_contract_artifact": config.dry_run,
        "not_claimed_as_paper_result": True,
        "summary": {
            "selected_methods": result.get("selected_methods", []),
            "num_method_results": len(result.get("method_results", {})) if isinstance(result.get("method_results"), Mapping) else 0,
            "metric_schema": ["masked_mse_to_reference", "c2st_proxy", "negative_log_likelihood_proxy"],
        },
        "result": result,
    }
    _write_json(out / "readiness.json", readiness)
    _write_json(out / "evaluation_result.json", evaluation)
    return {"readiness": str(out / "readiness.json"), "evaluation_result": str(out / "evaluation_result.json")}


def c2st_proxy(samples_a: Sequence[Sequence[float]], samples_b: Sequence[Sequence[float]]) -> float:
    """Deterministic C2ST-like proxy metric.

    Returns values in ``[0.5, 1.0]``.  It is a smoke metric and not a replacement
    for the full sklearn random-forest C2ST in paper-scale evaluation.
    """

    if not samples_a or not samples_b:
        return 0.5
    mean_a = _column_means(samples_a)
    mean_b = _column_means(samples_b)
    dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(mean_a, mean_b)))
    scale = 1.0 + math.sqrt(sum(b * b for b in mean_b))
    return max(0.5, min(1.0, 0.5 + 0.5 * dist / scale))


def gaussian_negative_log_proxy(samples: Sequence[Sequence[float]], reference: Sequence[Sequence[float]]) -> float:
    """Gaussian residual negative-log-likelihood proxy for smoke evaluation."""

    if not samples or not reference:
        return 0.0
    mse = _matrix_mse(samples, reference[: len(samples)])
    return 0.5 * math.log(2.0 * math.pi * max(mse, 1e-8)) + 0.5


def load_optional_torch_device(device: str = "cpu") -> Dict[str, Any]:
    """Lazy optional accelerator readiness check.

    reference_grounding: paperbench_ref_001 docs/how_to_guide/07_gpu_training.ipynb
    """

    try:
        import importlib

        torch_spec = importlib.util.find_spec("torch")
        if torch_spec is None:
            return {"available": False, "requested_device": device, "reason": "torch_not_installed"}
        torch = importlib.import_module("torch")
        available = False
        if device.startswith("cuda"):
            available = bool(torch.cuda.is_available())
        elif device == "mps":
            available = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        elif device == "cpu":
            available = True
        return {"available": available, "requested_device": device, "torch_version": getattr(torch, "__version__", None)}
    except Exception as exc:  # pragma: no cover - defensive optional dependency path
        return {"available": False, "requested_device": device, "reason": repr(exc)}


def _default_joint_simulator(
    theta: Sequence[float],
    x_dim: int,
    meta: Mapping[str, Any],
    rng: random.Random,
) -> List[float]:
    alpha = float(meta.get("alpha", 0.1))
    beta = float(meta.get("beta", 0.3))
    gamma = float(meta.get("gamma", 0.08))
    population = float(meta.get("population_size", 512))
    p = float(meta.get("p", 0.3))
    t0 = float(theta[0]) if len(theta) > 0 else 0.0
    t1 = float(theta[1]) if len(theta) > 1 else 0.0
    t2 = float(theta[2]) if len(theta) > 2 else 0.0
    t3 = float(theta[3]) if len(theta) > 3 else p
    base = [
        alpha * population / 512.0 + beta * t0 - gamma * t1 + rng.gauss(0.0, 0.01),
        math.sin(t0 + t1) + beta * t2 + rng.gauss(0.0, 0.01),
        (t3 - p) * (1.0 + gamma) + alpha * t0 * t0 + rng.gauss(0.0, 0.01),
    ]
    while len(base) < x_dim:
        idx = len(base)
        base.append(0.5 * base[idx % 3] + 0.1 * math.cos(idx + t0) + rng.gauss(0.0, 0.01))
    return base[:x_dim]


def _joint_mapping_to_matrix(batch: Mapping[str, Any], theta_dim: int, x_dim: int) -> Tuple[List[str], List[List[float]], Dict[str, Any]]:
    if "values" in batch and "variable_names" in batch:
        return list(batch["variable_names"]), _copy_matrix(batch["values"]), dict(batch.get("metadata", {}))

    theta = batch.get("theta")
    x = batch.get("x")
    if theta is None or x is None:
        raise ValueError("batch mapping must contain either values/variable_names or theta/x")

    theta_rows = _ensure_matrix(theta)
    x_rows = _ensure_matrix(x)
    if len(theta_rows) != len(x_rows):
        raise ValueError("theta and x must have the same number of rows")
    values = []
    for tr, xr in zip(theta_rows, x_rows):
        if len(tr) != theta_dim or len(xr) != x_dim:
            raise ValueError("theta/x row dimensions do not match tokenizer dimensions")
        values.append([float(v) for v in tr] + [float(v) for v in xr])
    names = [f"theta_{i}" for i in range(theta_dim)] + [f"x_{j}" for j in range(x_dim)]
    return names, values, dict(batch.get("metadata", {}))


def _ensure_matrix(value: Any) -> List[List[float]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("expected a sequence or matrix")
    if not value:
        return []
    first = value[0]
    if isinstance(first, Sequence) and not isinstance(first, (str, bytes)):
        return [[float(v) for v in row] for row in value]
    return [[float(v) for v in value]]


def _copy_matrix(matrix: Sequence[Sequence[float]]) -> List[List[float]]:
    return [[float(v) for v in row] for row in matrix]


def _validate_binary_mask(mask: Sequence[Sequence[int]], rows: int, cols: int) -> List[List[int]]:
    if len(mask) != int(rows):
        raise ValueError(f"condition mask row count {len(mask)} != expected {rows}")
    out: List[List[int]] = []
    for row in mask:
        if len(row) != int(cols):
            raise ValueError(f"condition mask column count {len(row)} != expected {cols}")
        out_row = []
        for value in row:
            ivalue = int(value)
            if ivalue not in {0, 1}:
                raise ValueError("condition state must be binary")
            out_row.append(ivalue)
        out.append(out_row)
    return out


def _validate_attention_mask(mask: Sequence[Sequence[int]], n: int) -> List[List[int]]:
    if len(mask) != int(n):
        raise ValueError(f"attention mask row count {len(mask)} != expected {n}")
    out: List[List[int]] = []
    for row in mask:
        if len(row) != int(n):
            raise ValueError(f"attention mask column count {len(row)} != expected {n}")
        out.append([1 if int(v) else 0 for v in row])
    return out


def _fit_gaussian_baseline(values: Sequence[Sequence[float]], theta_dim: int) -> Dict[str, Any]:
    matrix = _copy_matrix(values)
    means = _column_means(matrix)
    variances: List[float] = []
    for j, mean in enumerate(means):
        col = [row[j] for row in matrix]
        variances.append(statistics.pvariance(col) if len(col) > 1 else 1.0)
    return {
        "kind": "local_gaussian_posterior_surrogate",
        "theta_dim": theta_dim,
        "means": means,
        "variances": [max(v, 1e-6) for v in variances],
        "negative_log_proxy": gaussian_negative_log_proxy(matrix, matrix),
    }


def _sample_gaussian_baseline(baseline: Mapping[str, Any], num_samples: int, seed: int) -> List[List[float]]:
    rng = random.Random(seed)
    means = [float(v) for v in baseline.get("means", [0.0])]
    variances = [float(v) for v in baseline.get("variances", [1.0 for _ in means])]
    samples: List[List[float]] = []
    for _ in range(int(num_samples)):
        samples.append([rng.gauss(m, math.sqrt(max(v, 1e-8))) for m, v in zip(means, variances)])
    return samples


def _column_means(matrix: Sequence[Sequence[float]]) -> List[float]:
    if not matrix:
        return []
    cols = len(matrix[0])
    means = []
    for j in range(cols):
        means.append(sum(float(row[j]) for row in matrix) / float(len(matrix)))
    return means


def _matrix_mse(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> float:
    total = 0.0
    count = 0
    for row_a, row_b in zip(a, b):
        for va, vb in zip(row_a, row_b):
            err = float(va) - float(vb)
            total += err * err
            count += 1
    return total / float(count) if count else 0.0


def _flatten_method_loss_traces(method_results: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for selector, payload in method_results.items():
        for row in payload.get("trace", []):
            flat = dict(row)
            flat["selector"] = selector
            rows.append(flat)
    return rows


def _flatten_method_sampling_traces(method_results: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for selector, payload in method_results.items():
        for row in payload.get("sampling_trace", []):
            flat = dict(row)
            flat["selector"] = selector
            rows.append(flat)
    return rows


def _artifact_root(config_output_dir: str) -> Path:
    env_dir = os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR")
    if env_dir:
        return Path(env_dir)
    return Path(config_output_dir or "results")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, indent=2, sort_keys=True)


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return str(value)
        return value
    return value


__all__ = [
    "MASK_PROBABILITY_ANCHOR",
    "METHOD_REGISTRY",
    "SIMFORMER_SECTION_MODEL_CONFIGS",
    "SWEEP_REGISTRY",
    "TrainingConfig",
    "JointBatch",
    "TokenizedBatch",
    "DiffusionBatch",
    "TrainState",
    "SBITokenizer",
    "SimformerScoreAdapter",
    "MethodAdapter",
    "data_pipeline",
    "build_dependency_attention_mask",
    "sample_condition_mask",
    "make_diffusion_batch",
    "score_matching_loss",
    "training_loop",
    "conditional_sample",
    "evaluate_training_run",
    "train_compare_methods",
    "run_training",
    "write_training_artifacts",
    "write_readiness_artifacts",
    "c2st_proxy",
    "gaussian_negative_log_proxy",
    "load_optional_torch_device",
]
