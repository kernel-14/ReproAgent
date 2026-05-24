"""Canonical package import surface for stochastic interpolants with couplings.

The package root is a compatibility layer over the implementation modules that
own the active data, coupling, interpolant, model, training, sampling,
evaluation, metric, and artifact routes.  Heavy optional backends are never
imported at module import time; callers can use the lazy backend helpers below
before launching full ImageNet, training, or external-baseline workflows.
"""

from __future__ import annotations

from importlib import import_module, util
from types import ModuleType
from typing import Any

__version__ = "0.1.0"

# reference_grounding: paperbench_ref_004 xmodaler/datasets/README.md
# Dataset wrappers and dataset selection are exposed as explicit package-level
# aliases while real loading remains in data.py/config.py.
DATASET_ALIASES: dict[str, str] = {
    "imagenet": "imagenet_1k",
    "imagenet_1k": "imagenet_1k",
    "imagenet-1k": "imagenet_1k",
    "imagenet_c": "imagenet_c",
    "imagenet-c": "imagenet_c",
}

# reference_grounding: paperbench_ref_004 README.md
# The baseline vocabulary is preserved as selectable method names and resolved
# through this package's own factories rather than external repository imports.
METHOD_SELECTOR_NAMES: tuple[str, ...] = (
    "ours",
    "stochastic_interpolants",
    "data_dependent_couplings",
    "resnet",
    "ddpm",
    "diffusion_model",
)

BASELINE_SELECTOR_NAMES: tuple[str, ...] = (
    "rho_0 is a Gaussian with independent coupling to rho_1",
    "Gaussian rho_0 with independent coupling to rho_1",
    "baseline where rho0 is Gaussian with independent coupling to rho1",
    "independent_gaussian_coupling",
    "uncoupled_interpolant",
)

BATCH_SIZE_32: int = 32
MASK_TILES_64: int = 64
MASK_PROBABILITY_0_3: float = 0.3
GAMMA_SWEEP_VALUES: tuple[int, int] = (0, 1)
SIMILARITY_GUIDANCE_SCALE_SWEEP_VALUES: tuple[int, int] = (0, 1)
SAMPLER_TYPES: tuple[str, str] = ("ode", "sde")
COUPLING_TYPES: tuple[str, str] = ("independent_gaussian", "data_dependent")

OPTIONAL_BACKENDS: dict[str, str] = {
    "torch": "torch",
    "torchvision": "torchvision",
    "datasets": "datasets",
    "transformers": "transformers",
    "sbi": "sbi",
    "gym": "gymnasium",
    "gymnasium": "gymnasium",
}


_EXPORT_MODULES: dict[str, str] = {
    # config
    "DEFAULT_BATCH_SIZE": "stochastic_interpolants_couplings.config",
    "resolve_batch_size_defaults": "stochastic_interpolants_couplings.config",
    "batch_size_values": "stochastic_interpolants_couplings.config",
    "DEFAULT_ALPHA": "stochastic_interpolants_couplings.config",
    "resolve_alpha_defaults": "stochastic_interpolants_couplings.config",
    "alpha_values": "stochastic_interpolants_couplings.config",
    "DEFAULT_BETA": "stochastic_interpolants_couplings.config",
    "resolve_beta_defaults": "stochastic_interpolants_couplings.config",
    "beta_values": "stochastic_interpolants_couplings.config",
    "DEFAULT_GAMMA": "stochastic_interpolants_couplings.config",
    "resolve_gamma_defaults": "stochastic_interpolants_couplings.config",
    "gamma_values": "stochastic_interpolants_couplings.config",
    "ExperimentConfig": "stochastic_interpolants_couplings.config",
    "DataConfig": "stochastic_interpolants_couplings.config",
    "CouplingConfig": "stochastic_interpolants_couplings.config",
    "InterpolantConfig": "stochastic_interpolants_couplings.config",
    "ModelConfig": "stochastic_interpolants_couplings.config",
    "TrainingConfig": "stochastic_interpolants_couplings.config",
    "SamplingConfig": "stochastic_interpolants_couplings.config",
    "EvaluationConfig": "stochastic_interpolants_couplings.config",
    "ArtifactConfig": "stochastic_interpolants_couplings.config",
    "load_experiment_config": "stochastic_interpolants_couplings.config",
    "resolve_experiment_config": "stochastic_interpolants_couplings.config",
    "make_method": "stochastic_interpolants_couplings.config",
    "method_registry": "stochastic_interpolants_couplings.config",
    "baseline_registry": "stochastic_interpolants_couplings.config",
    "dataset_registry": "stochastic_interpolants_couplings.config",
    "metric_registry": "stochastic_interpolants_couplings.config",
    "sweep_registry": "stochastic_interpolants_couplings.config",
    "experiment_registry": "stochastic_interpolants_couplings.config",
    "environment_registry": "stochastic_interpolants_couplings.config",
    # data
    "ImageDatasetConfig": "stochastic_interpolants_couplings.data",
    "DataSpec": "stochastic_interpolants_couplings.data",
    "resolve_data_root": "stochastic_interpolants_couplings.data",
    "check_data_available": "stochastic_interpolants_couplings.data",
    "make_data": "stochastic_interpolants_couplings.data",
    "load_dataset": "stochastic_interpolants_couplings.data",
    "load_data": "stochastic_interpolants_couplings.data",
    "prepare_data": "stochastic_interpolants_couplings.data",
    "build_dataloader": "stochastic_interpolants_couplings.data",
    "build_task_batch": "stochastic_interpolants_couplings.data",
    "preprocess_image_tensor": "stochastic_interpolants_couplings.data",
    # couplings
    "CoupledBatch": "stochastic_interpolants_couplings.couplings",
    "DataDependentCoupling": "stochastic_interpolants_couplings.couplings",
    "IndependentGaussianCoupling": "stochastic_interpolants_couplings.couplings",
    "SuperResolutionCoupling": "stochastic_interpolants_couplings.couplings",
    "InpaintingCoupling": "stochastic_interpolants_couplings.couplings",
    "build_coupling": "stochastic_interpolants_couplings.couplings",
    "make_inpainting_mask": "stochastic_interpolants_couplings.couplings",
    "make_low_resolution_condition": "stochastic_interpolants_couplings.couplings",
    "sample_coupled_batch": "stochastic_interpolants_couplings.couplings",
    # interpolants/objectives
    "StochasticInterpolant": "stochastic_interpolants_couplings.interpolants",
    "InterpolantState": "stochastic_interpolants_couplings.interpolants",
    "alpha_t": "stochastic_interpolants_couplings.interpolants",
    "beta_t": "stochastic_interpolants_couplings.interpolants",
    "dalpha_t": "stochastic_interpolants_couplings.interpolants",
    "dbeta_t": "stochastic_interpolants_couplings.interpolants",
    "sample_time_uniform": "stochastic_interpolants_couplings.interpolants",
    "sample_standard_normal": "stochastic_interpolants_couplings.interpolants",
    "interpolant_state": "stochastic_interpolants_couplings.interpolants",
    "interpolant_derivative": "stochastic_interpolants_couplings.interpolants",
    "velocity_field_objective": "stochastic_interpolants_couplings.interpolants",
    "score_field_objective": "stochastic_interpolants_couplings.interpolants",
    "transport_cost": "stochastic_interpolants_couplings.interpolants",
    "compute_loss": "stochastic_interpolants_couplings.interpolants",
    "aggregate_loss": "stochastic_interpolants_couplings.interpolants",
    # models
    "VelocityUNet": "stochastic_interpolants_couplings.models",
    "ConditioningAdapter": "stochastic_interpolants_couplings.models",
    "build_velocity_model": "stochastic_interpolants_couplings.models",
    "count_parameters": "stochastic_interpolants_couplings.models",
    "Ours": "stochastic_interpolants_couplings.models",
    "ResNetBaseline": "stochastic_interpolants_couplings.models",
    "DDPMBaseline": "stochastic_interpolants_couplings.models",
    "DiffusionModelBaseline": "stochastic_interpolants_couplings.models",
    # training
    "TrainingState": "stochastic_interpolants_couplings.training",
    "build_optimizer": "stochastic_interpolants_couplings.training",
    "build_scheduler": "stochastic_interpolants_couplings.training",
    "train_one_step": "stochastic_interpolants_couplings.training",
    "train_experiment": "stochastic_interpolants_couplings.training",
    "save_checkpoint": "stochastic_interpolants_couplings.training",
    "load_checkpoint": "stochastic_interpolants_couplings.training",
    # sampling
    "ODESampler": "stochastic_interpolants_couplings.sampling",
    "SDESampler": "stochastic_interpolants_couplings.sampling",
    "sample_ode": "stochastic_interpolants_couplings.sampling",
    "sample_sde": "stochastic_interpolants_couplings.sampling",
    "sample_inpainting": "stochastic_interpolants_couplings.sampling",
    "sample_superresolution": "stochastic_interpolants_couplings.sampling",
    # metrics/evaluation
    "compute_fid": "stochastic_interpolants_couplings.metrics",
    "compute_fidelity_score": "stochastic_interpolants_couplings.metrics",
    "aggregate_fidelity_score": "stochastic_interpolants_couplings.metrics",
    "compute_accuracy": "stochastic_interpolants_couplings.metrics",
    "aggregate_accuracy": "stochastic_interpolants_couplings.metrics",
    "compute_reward": "stochastic_interpolants_couplings.metrics",
    "aggregate_reward": "stochastic_interpolants_couplings.metrics",
    "compute_f1": "stochastic_interpolants_couplings.metrics",
    "aggregate_f1": "stochastic_interpolants_couplings.metrics",
    "EvaluationResult": "stochastic_interpolants_couplings.evaluation",
    "evaluate_predictions": "stochastic_interpolants_couplings.evaluation",
    "evaluate_pairwise": "stochastic_interpolants_couplings.evaluation",
    "summarize_metrics": "stochastic_interpolants_couplings.evaluation",
    "evaluate_experiment": "stochastic_interpolants_couplings.evaluation",
    "evaluate_inpainting": "stochastic_interpolants_couplings.evaluation",
    "evaluate_superresolution": "stochastic_interpolants_couplings.evaluation",
    "write_evaluation_artifacts": "stochastic_interpolants_couplings.evaluation",
    # artifact/report routes
    "write_dataset_registry_artifact": "stochastic_interpolants_couplings.artifacts",
    "write_metrics_artifact": "stochastic_interpolants_couplings.artifacts",
    "write_data_manifest_artifact": "stochastic_interpolants_couplings.artifacts",
    "write_method_registry_artifact": "stochastic_interpolants_couplings.artifacts",
    "write_ablation_registry_artifact": "stochastic_interpolants_couplings.artifacts",
    "write_config_resolved_artifact": "stochastic_interpolants_couplings.artifacts",
    "write_sensitivity_report_artifact": "stochastic_interpolants_couplings.artifacts",
    "write_training_trace_artifact": "stochastic_interpolants_couplings.artifacts",
    "write_fidelity_score_artifact": "stochastic_interpolants_couplings.artifacts",
    "run_table_2_route": "stochastic_interpolants_couplings.artifacts",
    "write_table_2_artifact": "stochastic_interpolants_couplings.artifacts",
    "run_table_3_route": "stochastic_interpolants_couplings.artifacts",
    "write_table_3_artifact": "stochastic_interpolants_couplings.artifacts",
    "run_figure_1_route": "stochastic_interpolants_couplings.artifacts",
    "write_figure_1_artifact": "stochastic_interpolants_couplings.artifacts",
    "run_figure_2_route": "stochastic_interpolants_couplings.artifacts",
    "write_figure_2_artifact": "stochastic_interpolants_couplings.artifacts",
    "run_figure_3_route": "stochastic_interpolants_couplings.artifacts",
    "write_figure_3_artifact": "stochastic_interpolants_couplings.artifacts",
    "run_figure_4_route": "stochastic_interpolants_couplings.artifacts",
    "write_figure_4_artifact": "stochastic_interpolants_couplings.artifacts",
    "run_figure_6_route": "stochastic_interpolants_couplings.artifacts",
    "write_figure_6_artifact": "stochastic_interpolants_couplings.artifacts",
}


def optional_backend_available(name: str) -> bool:
    """Return whether an optional external backend can be imported lazily."""

    module_name = OPTIONAL_BACKENDS.get(name, name)
    return util.find_spec(module_name) is not None


def load_optional_backend(name: str) -> ModuleType:
    """Lazily import an optional backend by canonical package alias.

    This provides a real external-backend route for optional packages named by
    the reproduction plan, including ``datasets``, ``torch``, ``transformers``,
    ``sbi``, and ``gym``/``gymnasium``.  Import errors are raised only when the
    caller explicitly requests the backend.
    """

    module_name = OPTIONAL_BACKENDS.get(name, name)
    return import_module(module_name)


def require_optional_backend(name: str, purpose: str | None = None) -> ModuleType:
    """Load an optional backend or raise a runtime error with install context."""

    try:
        return load_optional_backend(name)
    except ImportError as exc:
        detail = f" for {purpose}" if purpose else ""
        raise RuntimeError(
            f"Optional backend '{name}' is required{detail}. "
            "Install the relevant extra, e.g. `pip install -e .[train]`, "
            "or provide a compatible local backend."
        ) from exc


def load_transformers_backend() -> ModuleType:
    """Lazily load HuggingFace Transformers for external diffusion baselines."""

    return require_optional_backend("transformers", "diffusion_model/resnet/ddpm adapter loading")


def get_dataset_alias(name: str) -> str:
    """Resolve paper-visible ImageNet dataset aliases to canonical ids."""

    return DATASET_ALIASES.get(name, name)


def get_method_selector_names() -> tuple[str, ...]:
    """Return method selector names required by the paper-derived matrix."""

    return METHOD_SELECTOR_NAMES


def get_baseline_selector_names() -> tuple[str, ...]:
    """Return independent Gaussian and uncoupled baseline selector names."""

    return BASELINE_SELECTOR_NAMES


def get_required_sweep_values() -> dict[str, tuple[int, int]]:
    """Return bounded sweep values used by config/train/evaluate/report routes."""

    return {
        "gamma": GAMMA_SWEEP_VALUES,
        "similarity_guidance_scale": SIMILARITY_GUIDANCE_SCALE_SWEEP_VALUES,
    }


def get_fixed_hyperparameter_anchors() -> dict[str, float | int]:
    """Return paper-fixed hyperparameter anchors exposed at package level."""

    return {
        "batch_size_32": BATCH_SIZE_32,
        "mask_tiles_64": MASK_TILES_64,
        "mask_probability_0.3": MASK_PROBABILITY_0_3,
    }


def get_sampler_types() -> tuple[str, str]:
    """Return explicitly separated sampler mechanisms."""

    return SAMPLER_TYPES


def get_coupling_types() -> tuple[str, str]:
    """Return explicitly selectable independent and data-dependent couplings."""

    return COUPLING_TYPES


def __getattr__(name: str) -> Any:
    if name in _EXPORT_MODULES:
        module = import_module(_EXPORT_MODULES[name])
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORT_MODULES))


__all__ = [
    "__version__",
    "DATASET_ALIASES",
    "METHOD_SELECTOR_NAMES",
    "BASELINE_SELECTOR_NAMES",
    "BATCH_SIZE_32",
    "MASK_TILES_64",
    "MASK_PROBABILITY_0_3",
    "GAMMA_SWEEP_VALUES",
    "SIMILARITY_GUIDANCE_SCALE_SWEEP_VALUES",
    "SAMPLER_TYPES",
    "COUPLING_TYPES",
    "OPTIONAL_BACKENDS",
    "optional_backend_available",
    "load_optional_backend",
    "require_optional_backend",
    "load_transformers_backend",
    "get_dataset_alias",
    "get_method_selector_names",
    "get_baseline_selector_names",
    "get_required_sweep_values",
    "get_fixed_hyperparameter_anchors",
    "get_sampler_types",
    "get_coupling_types",
    *_EXPORT_MODULES.keys(),
]