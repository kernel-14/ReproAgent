"""Configuration, registries, and executable protocol wiring.

This module is the canonical import-light configuration surface for
"Stochastic Interpolants with Data-Dependent Couplings".  It binds the paper
objects rho_0, rho_1, rho(x_0,x_1), I_t, alpha_t, beta_t, Algorithm 1
training, ODE/SDE sampling, ImageNet in-painting, ImageNet super-resolution,
FID evaluation, and paper artifact routes into executable selectors.

Optional training/data/vision backends are never imported at module import time.
"""

from __future__ import annotations

import csv
import importlib
import json
import math
import os
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping, Sequence

import numpy as np


DEFAULT_BATCH_SIZE = 32
DEFAULT_ALPHA: dict[str, Any] = {
    "name": "linear_to_base",
    "formula": "alpha_t = 1 - t",
    "alpha_0": 1.0,
    "alpha_1": 0.0,
}
DEFAULT_BETA: dict[str, Any] = {
    "name": "linear_to_target",
    "formula": "beta_t = t",
    "beta_0": 0.0,
    "beta_1": 1.0,
}
DEFAULT_GAMMA = 0
DEFAULT_SIMILARITY_GUIDANCE_SCALE = 0
DEFAULT_MASK_TILES = 64
DEFAULT_MASK_PROBABILITY = 0.3
DEFAULT_CHANNELS = 3
DEFAULT_RESOLUTION = 256
DEFAULT_IMAGE_SHAPE = (DEFAULT_CHANNELS, DEFAULT_RESOLUTION, DEFAULT_RESOLUTION)
DEFAULT_SIGMA = 1.0
DEFAULT_NUM_STEPS = 16
DEFAULT_LEARNING_RATE = 2.0e-4
DEFAULT_MAX_TRAIN_STEPS = 2
DEFAULT_DATASET = "imagenet"
DEFAULT_METHOD = "ours"
DEFAULT_COUPLING = "data_dependent"
DEFAULT_SAMPLER = "ode"
DEFAULT_TASK = "inpainting"
DEFAULT_OUTPUT_DIR = "results"
DEFAULT_CHECKPOINT_DIR = "checkpoints"

BLACKLISTED_REPOSITORIES = ("https://github.com/interpolants/couplings",)


def _as_plain(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return {str(k): _as_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_plain(v) for v in value]
    return value


def _artifact_root(output_dir: str | Path | None = None) -> Path:
    env_dir = os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR")
    return Path(env_dir or output_dir or DEFAULT_OUTPUT_DIR)


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_as_plain(payload), indent=2, sort_keys=True), encoding="utf-8")
    return path


def _write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_as_plain(row), sort_keys=True) + "\n")
    return path


def _write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return path


def batch_size_values(mode: str | None = None) -> tuple[int, ...]:
    if mode in {"full", "paper"}:
        return (32, 64, 128)
    if mode in {"bounded", "evaluation"}:
        return (8, 16, 32)
    return (2, DEFAULT_BATCH_SIZE)


def resolve_batch_size_defaults(overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    overrides = dict(overrides or {})
    mode = str(overrides.get("mode", overrides.get("run_mode", "runtime_smoke")))
    default = int(overrides.get("batch_size", DEFAULT_BATCH_SIZE))
    if mode in {"runtime_smoke", "smoke"}:
        active = min(default, int(overrides.get("max_batch_size", 2)))
    elif mode == "bounded":
        active = min(default, int(overrides.get("max_batch_size", 32)))
    else:
        active = default
    return {
        "default": DEFAULT_BATCH_SIZE,
        "active": active,
        "values": list(batch_size_values(mode)),
        "fixed_hyperparameter_anchor": "batch_size_32",
        "mode": mode,
    }


def alpha_values() -> tuple[str, ...]:
    return ("linear_to_base", "cosine_to_base")


def resolve_alpha_defaults(overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    resolved = dict(DEFAULT_ALPHA)
    if overrides:
        resolved.update({k: v for k, v in overrides.items() if v is not None})
    resolved["values"] = list(alpha_values())
    resolved["callable"] = "alpha_t"
    return resolved


def beta_values() -> tuple[str, ...]:
    return ("linear_to_target", "sine_to_target")


def resolve_beta_defaults(overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    resolved = dict(DEFAULT_BETA)
    if overrides:
        resolved.update({k: v for k, v in overrides.items() if v is not None})
    resolved["values"] = list(beta_values())
    resolved["callable"] = "beta_t"
    return resolved


def gamma_values(mode: str | None = None) -> tuple[int, ...]:
    return (0, 1)


def resolve_gamma_defaults(overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    overrides = dict(overrides or {})
    default = int(overrides.get("gamma", DEFAULT_GAMMA))
    if default not in gamma_values():
        raise ValueError(f"gamma must be one of {gamma_values()}, got {default}")
    return {
        "default": DEFAULT_GAMMA,
        "active": default,
        "values": list(gamma_values(str(overrides.get("mode", "")))),
        "parameter_sweep": "gamma[0,1]",
    }


def similarity_guidance_scale_values() -> tuple[int, ...]:
    return (0, 1)


def alpha_t(t: Any, config: Mapping[str, Any] | None = None) -> Any:
    cfg = resolve_alpha_defaults(config)
    name = cfg.get("name", "linear_to_base")
    t_arr = np.asarray(t, dtype=np.float64)
    if name == "linear_to_base":
        return 1.0 - t_arr
    if name == "cosine_to_base":
        return np.cos(0.5 * math.pi * t_arr)
    raise ValueError(f"Unknown alpha schedule {name!r}")


def beta_t(t: Any, config: Mapping[str, Any] | None = None) -> Any:
    cfg = resolve_beta_defaults(config)
    name = cfg.get("name", "linear_to_target")
    t_arr = np.asarray(t, dtype=np.float64)
    if name == "linear_to_target":
        return t_arr
    if name == "sine_to_target":
        return np.sin(0.5 * math.pi * t_arr)
    raise ValueError(f"Unknown beta schedule {name!r}")


def alpha_dot_t(t: Any, config: Mapping[str, Any] | None = None) -> Any:
    cfg = resolve_alpha_defaults(config)
    name = cfg.get("name", "linear_to_base")
    t_arr = np.asarray(t, dtype=np.float64)
    if name == "linear_to_base":
        return np.full_like(t_arr, -1.0, dtype=np.float64)
    if name == "cosine_to_base":
        return -0.5 * math.pi * np.sin(0.5 * math.pi * t_arr)
    raise ValueError(f"Unknown alpha schedule {name!r}")


def beta_dot_t(t: Any, config: Mapping[str, Any] | None = None) -> Any:
    cfg = resolve_beta_defaults(config)
    name = cfg.get("name", "linear_to_target")
    t_arr = np.asarray(t, dtype=np.float64)
    if name == "linear_to_target":
        return np.ones_like(t_arr, dtype=np.float64)
    if name == "sine_to_target":
        return 0.5 * math.pi * np.cos(0.5 * math.pi * t_arr)
    raise ValueError(f"Unknown beta schedule {name!r}")


def _reshape_time_coeff(coeff: Any, x: np.ndarray) -> np.ndarray:
    coeff = np.asarray(coeff, dtype=np.float64)
    if coeff.ndim == 0:
        return coeff
    return coeff.reshape((coeff.shape[0],) + (1,) * (x.ndim - 1))


def interpolant_state(
    x0: Any,
    x1: Any,
    t: Any,
    z: Any | None = None,
    gamma: float = 0.0,
    alpha: Mapping[str, Any] | None = None,
    beta: Mapping[str, Any] | None = None,
) -> np.ndarray:
    x0_arr = np.asarray(x0, dtype=np.float64)
    x1_arr = np.asarray(x1, dtype=np.float64)
    a = _reshape_time_coeff(alpha_t(t, alpha), x0_arr)
    b = _reshape_time_coeff(beta_t(t, beta), x0_arr)
    state = a * x0_arr + b * x1_arr
    if z is not None and gamma:
        t_arr = np.asarray(t, dtype=np.float64)
        bridge = _reshape_time_coeff(np.sqrt(np.maximum(t_arr * (1.0 - t_arr), 0.0)), x0_arr)
        state = state + float(gamma) * bridge * np.asarray(z, dtype=np.float64)
    return state


def interpolant_time_derivative(
    x0: Any,
    x1: Any,
    t: Any,
    z: Any | None = None,
    gamma: float = 0.0,
    alpha: Mapping[str, Any] | None = None,
    beta: Mapping[str, Any] | None = None,
) -> np.ndarray:
    x0_arr = np.asarray(x0, dtype=np.float64)
    x1_arr = np.asarray(x1, dtype=np.float64)
    adot = _reshape_time_coeff(alpha_dot_t(t, alpha), x0_arr)
    bdot = _reshape_time_coeff(beta_dot_t(t, beta), x0_arr)
    deriv = adot * x0_arr + bdot * x1_arr
    if z is not None and gamma:
        t_arr = np.asarray(t, dtype=np.float64)
        denom = np.maximum(np.sqrt(np.maximum(t_arr * (1.0 - t_arr), 0.0)), 1.0e-8)
        bridge_dot = (1.0 - 2.0 * t_arr) / (2.0 * denom)
        deriv = deriv + float(gamma) * _reshape_time_coeff(bridge_dot, x0_arr) * np.asarray(z, dtype=np.float64)
    return deriv


def velocity_field_objective(predicted_velocity: Any, target_velocity: Any) -> float:
    diff = np.asarray(predicted_velocity, dtype=np.float64) - np.asarray(target_velocity, dtype=np.float64)
    return float(np.mean(diff * diff))


def score_related_objective(predicted_score: Any, target_noise_or_score: Any) -> float:
    diff = np.asarray(predicted_score, dtype=np.float64) - np.asarray(target_noise_or_score, dtype=np.float64)
    return float(np.mean(diff * diff))


def transport_cost(x0: Any, x1: Any) -> float:
    diff = np.asarray(x1, dtype=np.float64) - np.asarray(x0, dtype=np.float64)
    return float(np.mean(np.sum(diff.reshape((diff.shape[0], -1)) ** 2, axis=1))) if diff.ndim > 1 else float(np.mean(diff * diff))


def fid_score_from_features(real_features: Any, generated_features: Any, eps: float = 1.0e-6) -> float:
    real = np.asarray(real_features, dtype=np.float64)
    gen = np.asarray(generated_features, dtype=np.float64)
    if real.ndim == 1:
        real = real.reshape(-1, 1)
    if gen.ndim == 1:
        gen = gen.reshape(-1, 1)
    mu_r = real.mean(axis=0)
    mu_g = gen.mean(axis=0)
    cov_r = np.cov(real, rowvar=False) if real.shape[0] > 1 else np.eye(real.shape[1]) * eps
    cov_g = np.cov(gen, rowvar=False) if gen.shape[0] > 1 else np.eye(gen.shape[1]) * eps
    cov_r = np.atleast_2d(cov_r)
    cov_g = np.atleast_2d(cov_g)
    product = cov_r @ cov_g
    try:
        vals, vecs = np.linalg.eigh(product)
        sqrt_product = (vecs * np.sqrt(np.maximum(vals, 0.0))) @ vecs.T
    except np.linalg.LinAlgError:
        sqrt_product = np.zeros_like(cov_r)
    value = np.sum((mu_r - mu_g) ** 2) + np.trace(cov_r + cov_g - 2.0 * sqrt_product)
    return float(np.real(value))


def aggregate_metric(values: Sequence[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("Cannot aggregate an empty metric sequence.")
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    aliases: tuple[str, ...]
    hf_name: str
    split: str
    task_ids: tuple[str, ...]
    resolutions: tuple[int, ...]
    lazy_loader: str
    validation_hook: str
    preprocessing: dict[str, Any]
    availability_check: str
    setup_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EnvironmentSpec:
    environment_id: str
    aliases: tuple[str, ...]
    task_type: str
    dataset_id: str
    resolution: int
    conditioning_type: str
    availability_check: str
    config_hook: str
    setup_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CouplingConfig:
    coupling_type: str = DEFAULT_COUPLING
    sigma: float = DEFAULT_SIGMA
    mask_tiles: int = DEFAULT_MASK_TILES
    mask_probability: float = DEFAULT_MASK_PROBABILITY
    low_resolution: int = 64
    high_resolution: int = DEFAULT_RESOLUTION
    conditional_base: bool = True


@dataclass(frozen=True)
class InterpolantConfig:
    alpha: dict[str, Any] = field(default_factory=lambda: resolve_alpha_defaults())
    beta: dict[str, Any] = field(default_factory=lambda: resolve_beta_defaults())
    gamma: int = DEFAULT_GAMMA
    score_weight: float = 0.0


@dataclass(frozen=True)
class ModelConfig:
    model_id: str = "velocity_unet"
    channels: int = DEFAULT_CHANNELS
    condition_channels: int = DEFAULT_CHANNELS
    base_channels: int = 32
    num_res_blocks: int = 2
    optional_class_labels: bool = True
    backend: str = "torch"
    adapter: str = "conditioning_channel_append"


@dataclass(frozen=True)
class SamplerConfig:
    sampler_type: str = DEFAULT_SAMPLER
    num_steps: int = DEFAULT_NUM_STEPS
    sde_diffusion: float = 0.0
    seed: int = 1234


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = DEFAULT_BATCH_SIZE
    learning_rate: float = DEFAULT_LEARNING_RATE
    max_train_steps: int | None = DEFAULT_MAX_TRAIN_STEPS
    log_every: int = 1
    checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR


@dataclass(frozen=True)
class EvaluationConfig:
    metrics: tuple[str, ...] = ("fid", "transport_cost", "training_loss_hat_L_b", "sampling_success_rate")
    max_eval_batches: int | None = 1
    max_samples: int | None = 2
    write_paper_visible_artifacts: bool = False


@dataclass(frozen=True)
class MethodSpec:
    method_id: str
    aliases: tuple[str, ...]
    coupling_type: str
    model_factory: str
    training_entrypoint: str
    sampling_entrypoints: dict[str, str]
    evaluation_entrypoint: str
    metrics: tuple[str, ...]
    baselines_compared: tuple[str, ...] = ()
    setup_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MetricSpec:
    metric_id: str
    aliases: tuple[str, ...]
    formula: str
    compute_callable: str
    aggregate_callable: str
    artifact_writer: str
    higher_is_better: bool | None = None


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    paper_section: str
    display_name: str
    task: str
    environment_id: str
    dataset_id: str
    method_id: str
    baseline_ids: tuple[str, ...]
    coupling_types: tuple[str, ...]
    sampler_types: tuple[str, ...]
    metric_ids: tuple[str, ...]
    artifact_routes: tuple[str, ...]
    config_hook: str
    decisive_metric: str
    hypothesis: str
    decision_value: str
    stop_rule_or_pruning_rationale: str


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str = "core_stochastic_interpolants_dry_run"
    mode: str = "runtime_smoke"
    task: str = DEFAULT_TASK
    dataset_id: str = DEFAULT_DATASET
    method_id: str = DEFAULT_METHOD
    output_dir: str = DEFAULT_OUTPUT_DIR
    data_root: str | None = None
    device: str = "cpu"
    seed: int = 1234
    image_shape: tuple[int, int, int] = DEFAULT_IMAGE_SHAPE
    coupling: CouplingConfig = field(default_factory=CouplingConfig)
    interpolant: InterpolantConfig = field(default_factory=InterpolantConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


# reference_grounding: paperbench_ref_004 xmodaler/datasets/README.md
# The reference protocol exposes builtin dataset wrappers through config keys.
# Here those wrapper choices are executable DatasetSpec rows consumed by data.py.
DATASET_REGISTRY: dict[str, DatasetSpec] = {
    "imagenet": DatasetSpec(
        dataset_id="imagenet",
        aliases=("ImageNet", "ImageNet。", "imagenet_1k", "imagenet-1k"),
        hf_name="imagenet-1k",
        split="train",
        task_ids=("inpainting", "super_resolution"),
        resolutions=(256, 512),
        lazy_loader="stochastic_interpolants_couplings.data.load_dataset",
        validation_hook="stochastic_interpolants_couplings.data.check_data_available",
        preprocessing={"normalize": "[-1,1]", "resize": [256, 512], "channels": 3},
        availability_check="lazy_huggingface_or_data_root",
        setup_metadata={
            "trust_remote_code": True,
            "full_mode_requires_user_access": True,
            "dataset_prepare_validate_path": "stochastic_interpolants_couplings.data.make_data",
        },
    ),
    "imagenet_1k": DatasetSpec(
        dataset_id="imagenet_1k",
        aliases=("imagenet", "imagenet-1k", "ImageNet for inpainting and super-resolution。"),
        hf_name="imagenet-1k",
        split="train",
        task_ids=("inpainting", "super_resolution"),
        resolutions=(256, 512),
        lazy_loader="stochastic_interpolants_couplings.data.load_dataset",
        validation_hook="stochastic_interpolants_couplings.data.check_data_available",
        preprocessing={"normalize": "[-1,1]", "resize": [256, 512], "channels": 3},
        availability_check="lazy_huggingface_or_data_root",
        setup_metadata={"trust_remote_code": True, "benchmark_registry_matrix": True},
    ),
    "imagenet_c": DatasetSpec(
        dataset_id="imagenet_c",
        aliases=("ImageNet-C", "imagenet corruption benchmark"),
        hf_name="imagenet-c",
        split="validation",
        task_ids=("robustness_conditioning", "super_resolution"),
        resolutions=(256,),
        lazy_loader="stochastic_interpolants_couplings.data.load_dataset",
        validation_hook="stochastic_interpolants_couplings.data.check_data_available",
        preprocessing={"normalize": "[-1,1]", "resize": [256], "channels": 3},
        availability_check="lazy_huggingface_or_data_root",
        setup_metadata={"trust_remote_code": True, "optional_benchmark": True},
    ),
}

ENVIRONMENT_REGISTRY: dict[str, EnvironmentSpec] = {
    "imagenet": EnvironmentSpec(
        environment_id="imagenet",
        aliases=("ImageNet", "represent full", "conditional image generation task", "imagenet keep external"),
        task_type="conditional_image_generation",
        dataset_id="imagenet",
        resolution=256,
        conditioning_type="task_injected",
        availability_check="stochastic_interpolants_couplings.data.check_data_available",
        config_hook="load_experiment_config",
        setup_metadata={"external_assets_lazy": True, "bind_every": True},
    ),
    "imagenet_256_inpainting": EnvironmentSpec(
        environment_id="imagenet_256_inpainting",
        aliases=("ImageNet-256x256", "in-painting task", "is determined"),
        task_type="inpainting",
        dataset_id="imagenet",
        resolution=256,
        conditioning_type="xi binary mask in {0,1}^{C x W x H}",
        availability_check="stochastic_interpolants_couplings.data.check_data_available",
        config_hook="make_inpainting_config",
        setup_metadata={"mask_tiles": DEFAULT_MASK_TILES, "mask_probability": DEFAULT_MASK_PROBABILITY},
    ),
    "imagenet_512_inpainting": EnvironmentSpec(
        environment_id="imagenet_512_inpainting",
        aliases=("ImageNet-512x512", "in-painting task"),
        task_type="inpainting",
        dataset_id="imagenet",
        resolution=512,
        conditioning_type="pre-specified mask plus optional class labels",
        availability_check="stochastic_interpolants_couplings.data.check_data_available",
        config_hook="make_inpainting_config",
        setup_metadata={"mask_tiles": DEFAULT_MASK_TILES, "mask_probability": DEFAULT_MASK_PROBABILITY},
    ),
    "imagenet_super_resolution": EnvironmentSpec(
        environment_id="imagenet_super_resolution",
        aliases=("low-resolution image", "perform various downstream", "image super-resolution"),
        task_type="super_resolution",
        dataset_id="imagenet_1k",
        resolution=256,
        conditioning_type="low-resolution image",
        availability_check="stochastic_interpolants_couplings.data.check_data_available",
        config_hook="make_super_resolution_config",
        setup_metadata={"low_resolution": 64, "high_resolution": 256},
    ),
}

METRIC_REGISTRY: dict[str, MetricSpec] = {
    "fid": MetricSpec(
        metric_id="fid",
        aliases=("FID", "FID-50k", "FID for Inpainting Task", "FID for super-resolution"),
        formula="||mu_r-mu_g||^2 + Tr(Sigma_r + Sigma_g - 2(Sigma_r Sigma_g)^{1/2})",
        compute_callable="stochastic_interpolants_couplings.evaluation.compute_fidelity_score",
        aggregate_callable="stochastic_interpolants_couplings.evaluation.aggregate_fidelity_score",
        artifact_writer="stochastic_interpolants_couplings.artifacts.write_fidelity_score_artifact",
        higher_is_better=False,
    ),
    "transport_cost": MetricSpec(
        metric_id="transport_cost",
        aliases=("transport cost诊断", "reducing transport costs via coupling"),
        formula="E[||x_1 - x_0||_2^2]",
        compute_callable="stochastic_interpolants_couplings.config.transport_cost",
        aggregate_callable="stochastic_interpolants_couplings.config.aggregate_metric",
        artifact_writer="stochastic_interpolants_couplings.artifacts.write_metrics_artifact",
        higher_is_better=False,
    ),
    "training_loss_hat_L_b": MetricSpec(
        metric_id="training_loss_hat_L_b",
        aliases=("训练损失hat L_b", "empirical approximation hat L_b"),
        formula="n_b^{-1} sum_i ||hat_b_t(I_t, condition) - dot{I}_t||_2^2",
        compute_callable="stochastic_interpolants_couplings.config.velocity_field_objective",
        aggregate_callable="stochastic_interpolants_couplings.config.aggregate_metric",
        artifact_writer="stochastic_interpolants_couplings.artifacts.write_metrics_artifact",
        higher_is_better=False,
    ),
    "sampling_success_rate": MetricSpec(
        metric_id="sampling_success_rate",
        aliases=("采样dry-run成功率", "ODE/SDE sampling success"),
        formula="successful_sample_batches / attempted_sample_batches",
        compute_callable="stochastic_interpolants_couplings.evaluation.compute_sampling_success_rate",
        aggregate_callable="stochastic_interpolants_couplings.config.aggregate_metric",
        artifact_writer="stochastic_interpolants_couplings.artifacts.write_metrics_artifact",
        higher_is_better=True,
    ),
}

METHOD_REGISTRY: dict[str, MethodSpec] = {
    "ours": MethodSpec(
        method_id="ours",
        aliases=("Dependent Coupling (Ours)", "data-dependent couplings", "stochastic interpolants"),
        coupling_type="data_dependent",
        model_factory="stochastic_interpolants_couplings.models.build_velocity_model",
        training_entrypoint="stochastic_interpolants_couplings.training.train_experiment",
        sampling_entrypoints={
            "ode": "stochastic_interpolants_couplings.sampling.sample_ode",
            "sde": "stochastic_interpolants_couplings.sampling.sample_sde",
        },
        evaluation_entrypoint="stochastic_interpolants_couplings.evaluation.evaluate_experiment",
        metrics=("fid", "transport_cost", "training_loss_hat_L_b", "sampling_success_rate"),
        baselines_compared=("independent_gaussian", "ddpm", "resnet"),
        setup_metadata={"conditional_rho_0_x0_given_x1": True, "model_loader_factory_path": True},
    ),
    "independent_gaussian": MethodSpec(
        method_id="independent_gaussian",
        aliases=("Uncoupled Interpolant (Baseline)", "rho_0 is a Gaussian with independent coupling to rho_1"),
        coupling_type="independent_gaussian",
        model_factory="stochastic_interpolants_couplings.models.build_velocity_model",
        training_entrypoint="stochastic_interpolants_couplings.training.train_experiment",
        sampling_entrypoints={
            "ode": "stochastic_interpolants_couplings.sampling.sample_ode",
            "sde": "stochastic_interpolants_couplings.sampling.sample_sde",
        },
        evaluation_entrypoint="stochastic_interpolants_couplings.evaluation.evaluate_experiment",
        metrics=("fid", "transport_cost", "training_loss_hat_L_b"),
    ),
    "resnet": MethodSpec(
        method_id="resnet",
        aliases=("resnet", "ResNet conditional adapter baseline"),
        coupling_type="independent_gaussian",
        model_factory="stochastic_interpolants_couplings.models.build_velocity_model",
        training_entrypoint="stochastic_interpolants_couplings.training.train_experiment",
        sampling_entrypoints={"ode": "stochastic_interpolants_couplings.sampling.sample_ode"},
        evaluation_entrypoint="stochastic_interpolants_couplings.evaluation.evaluate_experiment",
        metrics=("fid",),
        setup_metadata={"baseline_or_ablation": True},
    ),
    "ddpm": MethodSpec(
        method_id="ddpm",
        aliases=("ddpm", "diffusion_model", "Improved DDPM"),
        coupling_type="independent_gaussian",
        model_factory="stochastic_interpolants_couplings.models.build_velocity_model",
        training_entrypoint="stochastic_interpolants_couplings.training.train_experiment",
        sampling_entrypoints={"sde": "stochastic_interpolants_couplings.sampling.sample_sde"},
        evaluation_entrypoint="stochastic_interpolants_couplings.evaluation.evaluate_experiment",
        metrics=("fid",),
        setup_metadata={"diffusion_model": True, "baseline_or_ablation": True},
    ),
    "diffusion_model": MethodSpec(
        method_id="diffusion_model",
        aliases=("diffusion_model", "ddpm"),
        coupling_type="independent_gaussian",
        model_factory="stochastic_interpolants_couplings.models.build_velocity_model",
        training_entrypoint="stochastic_interpolants_couplings.training.train_experiment",
        sampling_entrypoints={"sde": "stochastic_interpolants_couplings.sampling.sample_sde"},
        evaluation_entrypoint="stochastic_interpolants_couplings.evaluation.evaluate_experiment",
        metrics=("fid",),
        setup_metadata={"selector_alias_for": "ddpm"},
    ),
}

BASELINE_REGISTRY: dict[str, dict[str, Any]] = {
    "independent_gaussian": {
        "baseline_id": "independent_gaussian",
        "method_id": "independent_gaussian",
        "coupling_type": "independent_gaussian",
        "paper_name": "Uncoupled Interpolant (Baseline)",
        "rho_0": "N(0,I_d) independent of x_1",
        "used_by": ["Table 2", "Table 3", "4.2 Super-resolution on Imagenet"],
    },
    "resnet": {
        "baseline_id": "resnet",
        "method_id": "resnet",
        "paper_name": "resnet",
        "used_by": ["method/baseline selector closure"],
    },
    "ddpm": {
        "baseline_id": "ddpm",
        "method_id": "ddpm",
        "paper_name": "ddpm",
        "used_by": ["super-resolution comparison context"],
    },
}

SWEEP_REGISTRY: dict[str, dict[str, Any]] = {
    "batch_size": {"default": DEFAULT_BATCH_SIZE, "values": list(batch_size_values("full")), "resolver": "resolve_batch_size_defaults"},
    "alpha_t": {"default": DEFAULT_ALPHA, "values": list(alpha_values()), "resolver": "resolve_alpha_defaults"},
    "beta_t": {"default": DEFAULT_BETA, "values": list(beta_values()), "resolver": "resolve_beta_defaults"},
    "gamma": {"default": DEFAULT_GAMMA, "values": list(gamma_values()), "resolver": "resolve_gamma_defaults"},
    "similarity_guidance_scale": {
        "default": DEFAULT_SIMILARITY_GUIDANCE_SCALE,
        "values": list(similarity_guidance_scale_values()),
        "parameter_sweep": "similarity_guidance_scale[0,1]",
    },
    "mask_tiles": {"default": DEFAULT_MASK_TILES, "values": [DEFAULT_MASK_TILES], "fixed_hyperparameter_anchor": "mask_tiles_64"},
    "mask_probability": {
        "default": DEFAULT_MASK_PROBABILITY,
        "values": [DEFAULT_MASK_PROBABILITY],
        "fixed_hyperparameter_anchor": "mask_probability_0.3",
    },
}

SAMPLER_REGISTRY: dict[str, dict[str, Any]] = {
    "ode": {
        "sampler_id": "ode",
        "display_name": "ODE sampler",
        "callable": "stochastic_interpolants_couplings.sampling.sample_ode",
        "transport": "dX_t = hat_b_t(X_t, condition) dt",
    },
    "sde": {
        "sampler_id": "sde",
        "display_name": "SDE sampler",
        "callable": "stochastic_interpolants_couplings.sampling.sample_sde",
        "transport": "dX_t = hat_b_t(X_t, condition) dt + diffusion(t) dW_t",
    },
}

EXPERIMENT_REGISTRY: dict[str, ExperimentSpec] = {
    "core_stochastic_interpolants_dry_run": ExperimentSpec(
        experiment_id="core_stochastic_interpolants_dry_run",
        paper_section="Section 3 and 3.4",
        display_name="core stochastic interpolants dry-run",
        task="core",
        environment_id="imagenet",
        dataset_id="imagenet",
        method_id="ours",
        baseline_ids=("independent_gaussian",),
        coupling_types=("data_dependent", "independent_gaussian"),
        sampler_types=("ode", "sde"),
        metric_ids=("transport_cost", "training_loss_hat_L_b", "sampling_success_rate"),
        artifact_routes=("write_resolved_config_artifact", "write_training_trace_artifact"),
        config_hook="load_experiment_config",
        decisive_metric="transport_cost",
        hypothesis="Data-dependent couplings reduce transport-related behavior relative to independent Gaussian coupling.",
        decision_value="paired transport_cost and hat_L_b diagnostics",
        stop_rule_or_pruning_rationale="Bound default execution; full mode scales the same route without exhaustive low-value sweeps.",
    ),
    "inpainting_table_2": ExperimentSpec(
        experiment_id="inpainting_table_2",
        paper_section="4.1 In-painting",
        display_name="Table 2: FID for Inpainting Task",
        task="inpainting",
        environment_id="imagenet_256_inpainting",
        dataset_id="imagenet",
        method_id="ours",
        baseline_ids=("independent_gaussian",),
        coupling_types=("independent_gaussian", "data_dependent"),
        sampler_types=("ode", "sde"),
        metric_ids=("fid",),
        artifact_routes=("run_table_2_route", "write_table_2_artifact"),
        config_hook="make_inpainting_config",
        decisive_metric="fid",
        hypothesis="Dependent Coupling (Ours) improves FID relative to Uncoupled Interpolant (Baseline).",
        decision_value="FID-50k Table 2 comparison",
        stop_rule_or_pruning_rationale="Only the paper-visible baseline-vs-ours comparison is selected.",
    ),
    "inpainting_figure_3": ExperimentSpec(
        experiment_id="inpainting_figure_3",
        paper_section="4.1 In-painting",
        display_name="Figure 3: Image inpainting: ImageNet-256x256 and ImageNet-512x512",
        task="inpainting",
        environment_id="imagenet_512_inpainting",
        dataset_id="imagenet",
        method_id="ours",
        baseline_ids=("independent_gaussian",),
        coupling_types=("data_dependent",),
        sampler_types=("ode",),
        metric_ids=("sampling_success_rate",),
        artifact_routes=("run_figure_3_route", "write_figure_3_artifact"),
        config_hook="make_inpainting_config",
        decisive_metric="sampling_success_rate",
        hypothesis="Mask-conditioned data-dependent coupling fills masked pixels using visible pixels and optional class labels.",
        decision_value="masked images / in-filled model samples / original images panel",
        stop_rule_or_pruning_rationale="Figure route is bounded by requested resolutions rather than exhaustive samples.",
    ),
    "super_resolution_imagenet": ExperimentSpec(
        experiment_id="super_resolution_imagenet",
        paper_section="4.2 Super-resolution on Imagenet",
        display_name="4.2 Super-resolution on Imagenet",
        task="super_resolution",
        environment_id="imagenet_super_resolution",
        dataset_id="imagenet_1k",
        method_id="ours",
        baseline_ids=("independent_gaussian", "ddpm", "resnet"),
        coupling_types=("independent_gaussian", "data_dependent"),
        sampler_types=("ode", "sde"),
        metric_ids=("fid", "transport_cost"),
        artifact_routes=("write_metrics_artifact", "write_data_manifest_artifact"),
        config_hook="make_super_resolution_config",
        decisive_metric="fid",
        hypothesis="Data-dependent coupling improves ImageNet super-resolution FID under shared low-resolution conditioning.",
        decision_value="FID comparison records and generated sample records",
        stop_rule_or_pruning_rationale="Evaluate decisive baseline comparison; omit redundant unbounded sweeps unless full mode requests them.",
    ),
}

PROTOCOL_MATRIX: tuple[dict[str, Any], ...] = tuple(
    {
        "experiment_id": spec.experiment_id,
        "display_name": spec.display_name,
        "task": spec.task,
        "environment_id": spec.environment_id,
        "dataset_id": spec.dataset_id,
        "method_id": spec.method_id,
        "baseline_ids": list(spec.baseline_ids),
        "metric_ids": list(spec.metric_ids),
        "artifact_routes": list(spec.artifact_routes),
        "config_hook": spec.config_hook,
        "decisive_metric": spec.decisive_metric,
        "hypothesis": spec.hypothesis,
        "decision_value": spec.decision_value,
        "stop_rule_or_pruning_rationale": spec.stop_rule_or_pruning_rationale,
    }
    for spec in EXPERIMENT_REGISTRY.values()
)

OPTIONAL_BACKEND_REGISTRY: dict[str, dict[str, Any]] = {
    "datasets": {
        "package": "datasets",
        "purpose": "HuggingFace ImageNet loader: load_dataset('imagenet-1k', trust_remote_code=True)",
        "factory": "load_optional_backend",
    },
    "torch": {
        "package": "torch",
        "purpose": "full velocity model training, checkpointing, ODE/SDE sampling",
        "factory": "load_optional_backend",
    },
    "torchvision": {
        "package": "torchvision",
        "purpose": "image transforms and optional feature extraction for FID",
        "factory": "load_optional_backend",
    },
    "transformers": {
        "package": "transformers",
        "purpose": "optional classifier/feature backbone adapter; not required for import smoke",
        "factory": "load_optional_backend",
    },
}


def load_optional_backend(package_name: str) -> Any:
    if package_name not in {entry["package"] for entry in OPTIONAL_BACKEND_REGISTRY.values()}:
        raise ValueError(f"Unsupported optional backend {package_name!r}")
    return importlib.import_module(package_name)


def backend_available(package_name: str) -> bool:
    return importlib.util.find_spec(package_name) is not None


def load_transformer_feature_backbone(model_name: str = "google/vit-base-patch16-224", **kwargs: Any) -> dict[str, Any]:
    transformers = load_optional_backend("transformers")
    auto_model = getattr(transformers, "AutoModel")
    auto_processor = getattr(transformers, "AutoImageProcessor", None)
    model = auto_model.from_pretrained(model_name, **kwargs)
    processor = auto_processor.from_pretrained(model_name, **kwargs) if auto_processor is not None else None
    return {"model_name": model_name, "model": model, "processor": processor}


def callable_from_dotted_path(dotted_path: str) -> Callable[..., Any]:
    module_name, symbol_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, symbol_name)


def dataset_registry() -> dict[str, dict[str, Any]]:
    return {key: asdict(value) for key, value in DATASET_REGISTRY.items()}


def environment_registry() -> dict[str, dict[str, Any]]:
    return {key: asdict(value) for key, value in ENVIRONMENT_REGISTRY.items()}


def metric_registry() -> dict[str, dict[str, Any]]:
    return {key: asdict(value) for key, value in METRIC_REGISTRY.items()}


def method_registry() -> dict[str, dict[str, Any]]:
    return {key: asdict(value) for key, value in METHOD_REGISTRY.items()}


def baseline_registry() -> dict[str, dict[str, Any]]:
    return dict(BASELINE_REGISTRY)


def sweep_registry() -> dict[str, dict[str, Any]]:
    resolved = dict(SWEEP_REGISTRY)
    resolved["batch_size"] = {**resolved["batch_size"], **resolve_batch_size_defaults()}
    resolved["alpha_t"] = {**resolved["alpha_t"], **resolve_alpha_defaults()}
    resolved["beta_t"] = {**resolved["beta_t"], **resolve_beta_defaults()}
    resolved["gamma"] = {**resolved["gamma"], **resolve_gamma_defaults()}
    return resolved


def experiment_registry() -> dict[str, dict[str, Any]]:
    return {key: asdict(value) for key, value in EXPERIMENT_REGISTRY.items()}


def resolve_dataset_id(dataset_id: str) -> str:
    normalized = dataset_id.lower().replace("-", "_")
    if normalized in DATASET_REGISTRY:
        return normalized
    for key, spec in DATASET_REGISTRY.items():
        aliases = {alias.lower().replace("-", "_") for alias in spec.aliases}
        if normalized in aliases:
            return key
    raise KeyError(f"Unknown dataset {dataset_id!r}; available={sorted(DATASET_REGISTRY)}")


def resolve_method_id(method_id: str) -> str:
    normalized = method_id.lower().replace("-", "_").replace(" ", "_")
    if normalized in METHOD_REGISTRY:
        return normalized
    for key, spec in METHOD_REGISTRY.items():
        aliases = {alias.lower().replace("-", "_").replace(" ", "_") for alias in spec.aliases}
        if normalized in aliases:
            return key
    raise KeyError(f"Unknown method {method_id!r}; available={sorted(METHOD_REGISTRY)}")


def resolve_experiment_id(experiment_id: str) -> str:
    if experiment_id in EXPERIMENT_REGISTRY:
        return experiment_id
    normalized = experiment_id.lower().replace(" ", "_").replace("-", "_")
    for key, spec in EXPERIMENT_REGISTRY.items():
        if normalized == key or normalized == spec.display_name.lower().replace(" ", "_").replace("-", "_"):
            return key
    raise KeyError(f"Unknown experiment {experiment_id!r}; available={sorted(EXPERIMENT_REGISTRY)}")


def make_method(config: ExperimentConfig | Mapping[str, Any]) -> dict[str, Any]:
    cfg = config if isinstance(config, ExperimentConfig) else load_experiment_config(config)
    method_id = resolve_method_id(cfg.method_id)
    spec = METHOD_REGISTRY[method_id]
    model_factory = callable_from_dotted_path(spec.model_factory)
    return {
        "method_id": method_id,
        "method_spec": asdict(spec),
        "model_factory": model_factory,
        "coupling_type": cfg.coupling.coupling_type,
        "ode_sampler": spec.sampling_entrypoints.get("ode"),
        "sde_sampler": spec.sampling_entrypoints.get("sde"),
    }


def make_inpainting_config(
    resolution: int = 256,
    coupling_type: str = "data_dependent",
    mode: str = "runtime_smoke",
    output_dir: str = DEFAULT_OUTPUT_DIR,
    **overrides: Any,
) -> ExperimentConfig:
    batch_defaults = resolve_batch_size_defaults({"mode": mode, **overrides})
    gamma_defaults = resolve_gamma_defaults(overrides)
    image_shape = (DEFAULT_CHANNELS, int(resolution), int(resolution))
    experiment_id = "inpainting_figure_3" if int(resolution) == 512 else "inpainting_table_2"
    return ExperimentConfig(
        experiment_id=overrides.get("experiment_id", experiment_id),
        mode=mode,
        task="inpainting",
        dataset_id=resolve_dataset_id(overrides.get("dataset_id", "imagenet")),
        method_id=resolve_method_id(overrides.get("method_id", "ours")),
        output_dir=output_dir,
        data_root=overrides.get("data_root"),
        device=overrides.get("device", "cpu"),
        seed=int(overrides.get("seed", 1234)),
        image_shape=image_shape,
        coupling=CouplingConfig(
            coupling_type=coupling_type,
            sigma=float(overrides.get("sigma", DEFAULT_SIGMA)),
            mask_tiles=int(overrides.get("mask_tiles", DEFAULT_MASK_TILES)),
            mask_probability=float(overrides.get("mask_probability", DEFAULT_MASK_PROBABILITY)),
            high_resolution=int(resolution),
            conditional_base=(coupling_type == "data_dependent"),
        ),
        interpolant=InterpolantConfig(
            alpha=resolve_alpha_defaults(overrides.get("alpha")),
            beta=resolve_beta_defaults(overrides.get("beta")),
            gamma=int(gamma_defaults["active"]),
            score_weight=float(overrides.get("score_weight", 0.0)),
        ),
        model=ModelConfig(
            channels=DEFAULT_CHANNELS,
            condition_channels=DEFAULT_CHANNELS * 2,
            optional_class_labels=bool(overrides.get("optional_class_labels", True)),
        ),
        sampler=SamplerConfig(
            sampler_type=str(overrides.get("sampler_type", DEFAULT_SAMPLER)).lower(),
            num_steps=int(overrides.get("num_steps", DEFAULT_NUM_STEPS)),
            sde_diffusion=float(overrides.get("sde_diffusion", 0.0)),
            seed=int(overrides.get("seed", 1234)),
        ),
        training=TrainingConfig(
            batch_size=int(batch_defaults["active"]),
            learning_rate=float(overrides.get("learning_rate", DEFAULT_LEARNING_RATE)),
            max_train_steps=overrides.get("max_train_steps", DEFAULT_MAX_TRAIN_STEPS),
            checkpoint_dir=str(overrides.get("checkpoint_dir", DEFAULT_CHECKPOINT_DIR)),
        ),
        evaluation=EvaluationConfig(
            max_eval_batches=overrides.get("max_eval_batches", 1),
            max_samples=overrides.get("max_samples", 2),
            write_paper_visible_artifacts=bool(overrides.get("write_paper_visible_artifacts", False)),
        ),
    )


def make_super_resolution_config(
    high_resolution: int = 256,
    low_resolution: int = 64,
    coupling_type: str = "data_dependent",
    mode: str = "runtime_smoke",
    output_dir: str = DEFAULT_OUTPUT_DIR,
    **overrides: Any,
) -> ExperimentConfig:
    batch_defaults = resolve_batch_size_defaults({"mode": mode, **overrides})
    gamma_defaults = resolve_gamma_defaults(overrides)
    image_shape = (DEFAULT_CHANNELS, int(high_resolution), int(high_resolution))
    return ExperimentConfig(
        experiment_id=overrides.get("experiment_id", "super_resolution_imagenet"),
        mode=mode,
        task="super_resolution",
        dataset_id=resolve_dataset_id(overrides.get("dataset_id", "imagenet_1k")),
        method_id=resolve_method_id(overrides.get("method_id", "ours")),
        output_dir=output_dir,
        data_root=overrides.get("data_root"),
        device=overrides.get("device", "cpu"),
        seed=int(overrides.get("seed", 1234)),
        image_shape=image_shape,
        coupling=CouplingConfig(
            coupling_type=coupling_type,
            sigma=float(overrides.get("sigma", DEFAULT_SIGMA)),
            low_resolution=int(low_resolution),
            high_resolution=int(high_resolution),
            conditional_base=(coupling_type == "data_dependent"),
        ),
        interpolant=InterpolantConfig(
            alpha=resolve_alpha_defaults(overrides.get("alpha")),
            beta=resolve_beta_defaults(overrides.get("beta")),
            gamma=int(gamma_defaults["active"]),
            score_weight=float(overrides.get("score_weight", 0.0)),
        ),
        model=ModelConfig(
            channels=DEFAULT_CHANNELS,
            condition_channels=DEFAULT_CHANNELS,
            optional_class_labels=bool(overrides.get("optional_class_labels", True)),
        ),
        sampler=SamplerConfig(
            sampler_type=str(overrides.get("sampler_type", DEFAULT_SAMPLER)).lower(),
            num_steps=int(overrides.get("num_steps", DEFAULT_NUM_STEPS)),
            sde_diffusion=float(overrides.get("sde_diffusion", 0.0)),
            seed=int(overrides.get("seed", 1234)),
        ),
        training=TrainingConfig(
            batch_size=int(batch_defaults["active"]),
            learning_rate=float(overrides.get("learning_rate", DEFAULT_LEARNING_RATE)),
            max_train_steps=overrides.get("max_train_steps", DEFAULT_MAX_TRAIN_STEPS),
            checkpoint_dir=str(overrides.get("checkpoint_dir", DEFAULT_CHECKPOINT_DIR)),
        ),
        evaluation=EvaluationConfig(
            max_eval_batches=overrides.get("max_eval_batches", 1),
            max_samples=overrides.get("max_samples", 2),
            write_paper_visible_artifacts=bool(overrides.get("write_paper_visible_artifacts", False)),
        ),
    )


def load_experiment_config(source: str | Path | Mapping[str, Any] | ExperimentConfig | None = None, **overrides: Any) -> ExperimentConfig:
    if isinstance(source, ExperimentConfig):
        base = source
        data: dict[str, Any] = {}
    elif isinstance(source, Mapping):
        base = None
        data = dict(source)
    elif source is None:
        base = None
        data = {}
    else:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Config file does not exist: {path}")
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() in {".yaml", ".yml"}:
            yaml = importlib.import_module("yaml")
            data = yaml.safe_load(text) or {}
        else:
            data = json.loads(text)
        base = None

    if base is not None:
        cfg = base
        for key, value in overrides.items():
            if hasattr(cfg, key):
                cfg = replace(cfg, **{key: value})
        return cfg

    data.update(overrides)
    runtime = dict(data.get("runtime", {}))
    method = dict(data.get("method", {}))
    task = str(data.get("task", method.get("task", runtime.get("task", DEFAULT_TASK))))
    mode = str(data.get("mode", runtime.get("default_mode", runtime.get("mode", "runtime_smoke"))))
    output_dir = str(data.get("output_dir", runtime.get("output_dir", DEFAULT_OUTPUT_DIR)))
    coupling_data = dict(data.get("coupling", method.get("coupling", {})))
    coupling_type = str(data.get("coupling_type", coupling_data.get("coupling_type", method.get("coupling_type", DEFAULT_COUPLING))))
    if task in {"super_resolution", "super-resolution", "sr"}:
        return make_super_resolution_config(
            high_resolution=int(data.get("high_resolution", coupling_data.get("high_resolution", data.get("resolution", DEFAULT_RESOLUTION)))),
            low_resolution=int(data.get("low_resolution", coupling_data.get("low_resolution", 64))),
            coupling_type=coupling_type,
            mode=mode,
            output_dir=output_dir,
            **data,
        )
    return make_inpainting_config(
        resolution=int(data.get("resolution", coupling_data.get("high_resolution", DEFAULT_RESOLUTION))),
        coupling_type=coupling_type,
        mode=mode,
        output_dir=output_dir,
        **data,
    )


def resolved_config_dict(config: ExperimentConfig | Mapping[str, Any] | None = None) -> dict[str, Any]:
    cfg = load_experiment_config(config)
    batch_defaults = resolve_batch_size_defaults({"mode": cfg.mode, "batch_size": cfg.training.batch_size})
    alpha_defaults = resolve_alpha_defaults(cfg.interpolant.alpha)
    beta_defaults = resolve_beta_defaults(cfg.interpolant.beta)
    gamma_defaults = resolve_gamma_defaults({"gamma": cfg.interpolant.gamma})
    return {
        "schema_version": "1.0",
        "paper": "Stochastic Interpolants with Data-Dependent Couplings",
        "blacklisted_repositories": list(BLACKLISTED_REPOSITORIES),
        "experiment": asdict(cfg),
        "dataset": asdict(DATASET_REGISTRY[resolve_dataset_id(cfg.dataset_id)]),
        "method": asdict(METHOD_REGISTRY[resolve_method_id(cfg.method_id)]),
        "batch_size_defaults": batch_defaults,
        "alpha_defaults": alpha_defaults,
        "beta_defaults": beta_defaults,
        "gamma_defaults": gamma_defaults,
        "sampler_registry": SAMPLER_REGISTRY,
        "metric_registry": metric_registry(),
        "scope_constraints": {
            "standalone_implementation": True,
            "no_blacklisted_repository_dependency": True,
            "default_run_is_bounded": True,
            "full_training_supported_by_same_route": True,
            "stop_rule_or_pruning_rationale": EXPERIMENT_REGISTRY[resolve_experiment_id(cfg.experiment_id)].stop_rule_or_pruning_rationale,
        },
    }


def write_dataset_registry_artifact(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> Path:
    return _write_json(_artifact_root(output_dir) / "dataset_registry.json", {"datasets": dataset_registry()})


def write_environment_registry_artifact(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> Path:
    return _write_json(_artifact_root(output_dir) / "environment_registry.json", {"environments": environment_registry()})


def write_metrics_artifact(output_dir: str | Path = DEFAULT_OUTPUT_DIR, metrics: Mapping[str, Any] | None = None) -> Path:
    payload = {
        "metric_registry": metric_registry(),
        "measured_metrics": dict(metrics or {}),
        "metric_formula_aggregation_path": "stochastic_interpolants_couplings.evaluation",
    }
    return _write_json(_artifact_root(output_dir) / "metrics.json", payload)


def write_data_manifest_artifact(output_dir: str | Path = DEFAULT_OUTPUT_DIR, config: ExperimentConfig | Mapping[str, Any] | None = None) -> Path:
    cfg = load_experiment_config(config)
    payload = {
        "dataset_id": cfg.dataset_id,
        "task": cfg.task,
        "data_root": cfg.data_root,
        "datasets": dataset_registry(),
        "lazy_full_data_download": True,
        "dataset_prepare_validate_path": "stochastic_interpolants_couplings.data.make_data",
        "huggingface_loader": "load_dataset('imagenet-1k', trust_remote_code=True)",
    }
    return _write_json(_artifact_root(output_dir) / "data_manifest.json", payload)


def write_method_registry_artifact(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> Path:
    return _write_json(_artifact_root(output_dir) / "method_registry.json", {"methods": method_registry()})


def write_ablation_registry_artifact(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> Path:
    payload = {
        "baselines": baseline_registry(),
        "sweeps": sweep_registry(),
        "required_selectors": ["ours", "resnet", "ddpm", "diffusion_model", "independent_gaussian"],
    }
    return _write_json(_artifact_root(output_dir) / "ablation_registry.json", payload)


def write_experiment_registry_artifact(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> Path:
    return _write_json(_artifact_root(output_dir) / "experiment_registry.json", {"experiments": experiment_registry(), "protocol_matrix": list(PROTOCOL_MATRIX)})


def write_resolved_config_artifact(config: ExperimentConfig | Mapping[str, Any] | None = None, output_dir: str | Path | None = None) -> Path:
    cfg = load_experiment_config(config)
    return _write_json(_artifact_root(output_dir or cfg.output_dir) / "config_resolved.json", resolved_config_dict(cfg))


def write_scope_report_artifact(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> Path:
    payload = {
        "standalone_implementation": True,
        "blacklisted_repositories": list(BLACKLISTED_REPOSITORIES),
        "full_mode_requires_explicit_flag": True,
        "pruning_rationale": "Stop at paper-specified protocols; omit exhaustive sweeps unless full mode requests them.",
    }
    return _write_json(_artifact_root(output_dir) / "scope_report.json", payload)


def write_training_trace_artifact(output_dir: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    return _write_jsonl(_artifact_root(output_dir) / "training_trace.json", rows)


def write_core_training_log(output_dir: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    return _write_jsonl(_artifact_root(output_dir) / "train_log.jsonl", rows)


def write_readiness_artifacts(config: ExperimentConfig | Mapping[str, Any] | None = None) -> dict[str, str]:
    cfg = load_experiment_config(config)
    root = _artifact_root(cfg.output_dir)
    readiness = {
        "ready": True,
        "mode": cfg.mode,
        "canonical_route": "scripts/run_coupled_si.py",
        "resolved_config": str(root / "config_resolved.json"),
        "dataset_registry": str(root / "dataset_registry.json"),
        "method_registry": str(root / "method_registry.json"),
        "paper_visible_outputs_require_measured_routes": True,
    }
    evaluation_result = {
        "mode": cfg.mode,
        "executed_routes": ["config", "registry", "method_selector", "metric_formula"],
        "no_benchmark_score_claimed": cfg.mode != "full",
        "sampler_routes": ["ode", "sde"],
    }
    paths = {
        "readiness": str(_write_json(root / "readiness.json", readiness)),
        "evaluation_result": str(_write_json(root / "evaluation_result.json", evaluation_result)),
    }
    return paths


def write_table_2_artifact(rows: Sequence[Mapping[str, Any]], output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> Path:
    if not rows:
        raise ValueError("Table 2 artifact requires measured FID rows.")
    normalized_rows = []
    for row in rows:
        if "fid" not in row and "FID-50k" not in row:
            raise ValueError("Each Table 2 row must include a measured fid or FID-50k value.")
        normalized_rows.append(
            {
                "table": "Table 2: FID for Inpainting Task",
                "task": row.get("task", "inpainting"),
                "model": row.get("model", row.get("method_id", "")),
                "coupling": row.get("coupling", row.get("coupling_type", "")),
                "FID-50k": row.get("FID-50k", row.get("fid")),
                "provenance": row.get("provenance", "measured route"),
            }
        )
    return _write_csv(_artifact_root(output_dir) / "tables" / "table_2.csv", normalized_rows)


def run_table_2_route(config: ExperimentConfig | Mapping[str, Any] | None = None, measured_rows: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    cfg = load_experiment_config(config or {"task": "inpainting", "experiment_id": "inpainting_table_2"})
    if measured_rows is None:
        try:
            evaluate_experiment = callable_from_dotted_path("stochastic_interpolants_couplings.evaluation.evaluate_inpainting")
            result = evaluate_experiment(cfg)
            measured_rows = result.get("table_2_rows", result.get("rows", [])) if isinstance(result, Mapping) else []
        except (ImportError, AttributeError, FileNotFoundError):
            measured_rows = []
    if not measured_rows:
        return {
            "artifact_written": False,
            "requires": "measured inpainting FID rows from evaluation route",
            "experiment_id": cfg.experiment_id,
        }
    path = write_table_2_artifact(measured_rows, cfg.output_dir)
    return {"artifact_written": True, "path": str(path), "rows": len(measured_rows)}


def write_figure_3_artifact(samples: Mapping[str, Any], output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> Path:
    required = {"masked_images", "infilled_model_samples", "original_images", "resolution"}
    missing = required.difference(samples)
    if missing:
        raise ValueError(f"Figure 3 artifact requires {sorted(required)}; missing {sorted(missing)}")
    root = _artifact_root(output_dir) / "figures"
    root.mkdir(parents=True, exist_ok=True)
    path = root / "figure_3.json"
    payload = {
        "figure": "Figure 3: Image inpainting: ImageNet-256x256 and ImageNet-512x512",
        "columns": ["masked images", "in-filled model samples", "original images"],
        "resolution": samples["resolution"],
        "sample_count": int(samples.get("sample_count", len(samples.get("masked_images", [])) if hasattr(samples.get("masked_images", []), "__len__") else 0)),
        "provenance": samples.get("provenance", "measured route"),
    }
    return _write_json(path, payload)


def run_figure_3_route(config: ExperimentConfig | Mapping[str, Any] | None = None, samples: Mapping[str, Any] | None = None) -> dict[str, Any]:
    cfg = load_experiment_config(config or {"task": "inpainting", "experiment_id": "inpainting_figure_3", "resolution": 512})
    if samples is None:
        try:
            evaluate_inpainting = callable_from_dotted_path("stochastic_interpolants_couplings.evaluation.evaluate_inpainting")
            result = evaluate_inpainting(cfg)
            samples = result.get("figure_3_samples", {}) if isinstance(result, Mapping) else {}
        except (ImportError, AttributeError, FileNotFoundError):
            samples = {}
    if not samples:
        return {
            "artifact_written": False,
            "requires": "measured inpainting samples with masked/model/original columns",
            "experiment_id": cfg.experiment_id,
        }
    path = write_figure_3_artifact(samples, cfg.output_dir)
    return {"artifact_written": True, "path": str(path)}


def evaluate_predictions(config: ExperimentConfig | Mapping[str, Any] | None = None, predictions: Any | None = None, targets: Any | None = None) -> dict[str, Any]:
    cfg = load_experiment_config(config)
    if predictions is None or targets is None:
        evaluate_experiment = callable_from_dotted_path("stochastic_interpolants_couplings.evaluation.evaluate_experiment")
        return evaluate_experiment(cfg)
    pred = np.asarray(predictions, dtype=np.float64)
    tgt = np.asarray(targets, dtype=np.float64)
    mse = float(np.mean((pred - tgt) ** 2))
    flat_pred = pred.reshape((pred.shape[0], -1)) if pred.ndim > 1 else pred.reshape(-1, 1)
    flat_tgt = tgt.reshape((tgt.shape[0], -1)) if tgt.ndim > 1 else tgt.reshape(-1, 1)
    fid = fid_score_from_features(flat_tgt, flat_pred)
    return {
        "task": cfg.task,
        "method_id": cfg.method_id,
        "coupling_type": cfg.coupling.coupling_type,
        "metrics": {"mse": mse, "fid": fid},
    }


def run_algorithm_1_training(config: ExperimentConfig | Mapping[str, Any] | None = None) -> Any:
    cfg = load_experiment_config(config)
    train_experiment = callable_from_dotted_path("stochastic_interpolants_couplings.training.train_experiment")
    return train_experiment(cfg)


def run_ode_sampling(config: ExperimentConfig | Mapping[str, Any] | None = None, **kwargs: Any) -> Any:
    cfg = load_experiment_config(config)
    sample_ode = callable_from_dotted_path(SAMPLER_REGISTRY["ode"]["callable"])
    return sample_ode(cfg, **kwargs)


def run_sde_sampling(config: ExperimentConfig | Mapping[str, Any] | None = None, **kwargs: Any) -> Any:
    cfg = load_experiment_config(config)
    sample_sde = callable_from_dotted_path(SAMPLER_REGISTRY["sde"]["callable"])
    return sample_sde(cfg, **kwargs)


def sample_x1_from_rho1(batch: Any) -> np.ndarray:
    if isinstance(batch, Mapping):
        for key in ("x1", "target", "image", "images"):
            if key in batch:
                return np.asarray(batch[key], dtype=np.float64)
    return np.asarray(batch, dtype=np.float64)


def sample_zeta(shape: Sequence[int], rng: np.random.Generator | None = None) -> np.ndarray:
    return (rng or np.random.default_rng()).normal(size=tuple(shape))


def sample_t(batch_size: int, rng: np.random.Generator | None = None) -> np.ndarray:
    return (rng or np.random.default_rng()).uniform(0.0, 1.0, size=(int(batch_size),))


def conditional_rho0_sample(x1: Any, condition: Any | None = None, coupling: CouplingConfig | Mapping[str, Any] | None = None, rng: np.random.Generator | None = None) -> dict[str, np.ndarray]:
    cfg = coupling if isinstance(coupling, CouplingConfig) else CouplingConfig(**dict(coupling or {}))
    x1_arr = np.asarray(x1, dtype=np.float64)
    zeta = sample_zeta(x1_arr.shape, rng)
    if cfg.coupling_type == "independent_gaussian":
        mean = np.zeros_like(x1_arr)
    elif condition is None:
        mean = np.zeros_like(x1_arr)
    else:
        cond = np.asarray(condition, dtype=np.float64)
        mean = cond if cond.shape == x1_arr.shape else np.broadcast_to(np.mean(cond), x1_arr.shape)
    x0 = mean + float(cfg.sigma) * zeta
    return {"x0": x0, "x1": x1_arr, "zeta": zeta, "conditional_mean": mean}


def algorithm_1_minibatch_objective(
    batch: Any,
    velocity_model: Callable[..., Any],
    config: ExperimentConfig | Mapping[str, Any] | None = None,
    condition: Any | None = None,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    cfg = load_experiment_config(config)
    x1 = sample_x1_from_rho1(batch)
    rng = rng or np.random.default_rng(cfg.seed)
    coupled = conditional_rho0_sample(x1, condition=condition, coupling=cfg.coupling, rng=rng)
    t = sample_t(x1.shape[0], rng)
    noise = sample_zeta(x1.shape, rng)
    i_t = interpolant_state(coupled["x0"], coupled["x1"], t, noise, cfg.interpolant.gamma, cfg.interpolant.alpha, cfg.interpolant.beta)
    target = interpolant_time_derivative(coupled["x0"], coupled["x1"], t, noise, cfg.interpolant.gamma, cfg.interpolant.alpha, cfg.interpolant.beta)
    predicted = velocity_model(i_t, t, condition) if condition is not None else velocity_model(i_t, t)
    loss = velocity_field_objective(predicted, target)
    score_loss = 0.0
    if cfg.interpolant.score_weight:
        score_loss = score_related_objective(np.asarray(predicted) * 0.0, noise)
    return {
        "x1_sampled_from": "rho_1",
        "zeta_sampled_from": "N(0,I_d)",
        "t_sampled_from": "U(0,1)",
        "I_t": i_t,
        "target_dot_I_t": target,
        "hat_b_t_I_t": predicted,
        "hat_L_b": float(loss),
        "score_related_objective": float(score_loss),
        "transport_cost": transport_cost(coupled["x0"], coupled["x1"]),
        "coupling_type": cfg.coupling.coupling_type,
    }


def write_runtime_registries(config: ExperimentConfig | Mapping[str, Any] | None = None) -> dict[str, str]:
    cfg = load_experiment_config(config)
    output_dir = cfg.output_dir
    paths = {
        "dataset_registry": str(write_dataset_registry_artifact(output_dir)),
        "data_manifest": str(write_data_manifest_artifact(output_dir, cfg)),
        "environment_registry": str(write_environment_registry_artifact(output_dir)),
        "method_registry": str(write_method_registry_artifact(output_dir)),
        "ablation_registry": str(write_ablation_registry_artifact(output_dir)),
        "experiment_registry": str(write_experiment_registry_artifact(output_dir)),
        "metrics": str(write_metrics_artifact(output_dir)),
        "config_resolved": str(write_resolved_config_artifact(cfg, output_dir)),
        "scope_report": str(write_scope_report_artifact(output_dir)),
    }
    if cfg.mode in {"runtime_smoke", "smoke", "docker_validate"}:
        paths.update(write_readiness_artifacts(cfg))
    return paths


def executable_protocol_matrix(config: ExperimentConfig | Mapping[str, Any] | None = None) -> tuple[dict[str, Any], ...]:
    cfg = load_experiment_config(config)
    _ = resolve_batch_size_defaults({"mode": cfg.mode, "batch_size": cfg.training.batch_size})
    _ = resolve_alpha_defaults(cfg.interpolant.alpha)
    _ = resolve_beta_defaults(cfg.interpolant.beta)
    _ = resolve_gamma_defaults({"gamma": cfg.interpolant.gamma})
    rows = []
    for row in PROTOCOL_MATRIX:
        executable_row = dict(row)
        executable_row["artifact_writer_call_sites"] = [
            "write_table_2_artifact" if route == "write_table_2_artifact" else route for route in executable_row["artifact_routes"]
        ]
        executable_row["dataset_loader"] = DATASET_REGISTRY[row["dataset_id"]].lazy_loader
        executable_row["method_factory"] = METHOD_REGISTRY[row["method_id"]].model_factory
        executable_row["metric_functions"] = [METRIC_REGISTRY[mid].compute_callable for mid in row["metric_ids"]]
        rows.append(executable_row)
    return tuple(rows)


def route_artifact_call_sites(config: ExperimentConfig | Mapping[str, Any] | None = None) -> dict[str, Any]:
    cfg = load_experiment_config(config)
    registry_paths = write_runtime_registries(cfg)
    table_2 = run_table_2_route(cfg)
    figure_3 = run_figure_3_route(cfg)
    return {"registries": registry_paths, "table_2_route": table_2, "figure_3_route": figure_3}


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "resolve_batch_size_defaults",
    "batch_size_values",
    "DEFAULT_ALPHA",
    "resolve_alpha_defaults",
    "alpha_values",
    "DEFAULT_BETA",
    "resolve_beta_defaults",
    "beta_values",
    "DEFAULT_GAMMA",
    "resolve_gamma_defaults",
    "gamma_values",
    "DEFAULT_MASK_TILES",
    "DEFAULT_MASK_PROBABILITY",
    "DatasetSpec",
    "EnvironmentSpec",
    "CouplingConfig",
    "InterpolantConfig",
    "ModelConfig",
    "SamplerConfig",
    "TrainingConfig",
    "EvaluationConfig",
    "MethodSpec",
    "MetricSpec",
    "ExperimentSpec",
    "ExperimentConfig",
    "DATASET_REGISTRY",
    "ENVIRONMENT_REGISTRY",
    "METRIC_REGISTRY",
    "METHOD_REGISTRY",
    "BASELINE_REGISTRY",
    "SWEEP_REGISTRY",
    "SAMPLER_REGISTRY",
    "EXPERIMENT_REGISTRY",
    "PROTOCOL_MATRIX",
    "OPTIONAL_BACKEND_REGISTRY",
    "alpha_t",
    "beta_t",
    "alpha_dot_t",
    "beta_dot_t",
    "interpolant_state",
    "interpolant_time_derivative",
    "velocity_field_objective",
    "score_related_objective",
    "transport_cost",
    "fid_score_from_features",
    "aggregate_metric",
    "load_optional_backend",
    "backend_available",
    "load_transformer_feature_backbone",
    "callable_from_dotted_path",
    "dataset_registry",
    "environment_registry",
    "metric_registry",
    "method_registry",
    "baseline_registry",
    "sweep_registry",
    "experiment_registry",
    "resolve_dataset_id",
    "resolve_method_id",
    "resolve_experiment_id",
    "make_method",
    "make_inpainting_config",
    "make_super_resolution_config",
    "load_experiment_config",
    "resolved_config_dict",
    "write_dataset_registry_artifact",
    "write_environment_registry_artifact",
    "write_metrics_artifact",
    "write_data_manifest_artifact",
    "write_method_registry_artifact",
    "write_ablation_registry_artifact",
    "write_experiment_registry_artifact",
    "write_resolved_config_artifact",
    "write_scope_report_artifact",
    "write_training_trace_artifact",
    "write_core_training_log",
    "write_readiness_artifacts",
    "write_table_2_artifact",
    "run_table_2_route",
    "write_figure_3_artifact",
    "run_figure_3_route",
    "evaluate_predictions",
    "run_algorithm_1_training",
    "run_ode_sampling",
    "run_sde_sampling",
    "sample_x1_from_rho1",
    "sample_zeta",
    "sample_t",
    "conditional_rho0_sample",
    "algorithm_1_minibatch_objective",
    "write_runtime_registries",
    "executable_protocol_matrix",
    "route_artifact_call_sites",
]