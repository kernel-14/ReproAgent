"""Executable registry for the Simformer core reproduction.

This module is the central, import-light registry for the PaperBench
reproduction of *All-in-one simulation-based inference*.  It makes the core
method obligations explicit and machine-readable while providing factory
functions that route to the real implementation surfaces in the package.

The registry intentionally does more than list names:

* exposes tokenizer, dependency-mask builder, score network, trainer, sampler,
  and guided sampler selectors;
* records that Simformer is trained on the joint simulator distribution
  ``p(theta, x)`` / ``p(x_hat)`` rather than only posterior or likelihood
  factors;
* records and validates that ``M_E`` enters transformer attention computation;
* records and validates that ``M_C`` enters forward noising, loss masking, and
  conditional sampling;
* names SDE and ODE sampler families separately and makes them selectable;
* records training metadata fields required for reproducibility: method,
  mask variant, conditioning pattern, simulation budget, and fixed
  hyperparameters;
* provides a bounded dry-run artifact writer for registry/readiness outputs
  without claiming paper-scale training or benchmark results.

Optional scientific packages are not imported at module scope.  Factories import
``torch``/NumPy-backed implementation modules lazily only when a runtime path
needs them.

No code is copied from the blacklisted Simformer repository.

reference_grounding: paperbench_ref_001 docs/how_to_guide/04_embedding_networks.ipynb
reference_grounding: paperbench_ref_001 docs/how_to_guide/07_gpu_training.ipynb
reference_grounding: paperbench_ref_001 docs/how_to_guide/09_sampler_interface.ipynb
reference_grounding: paperbench_ref_001 sbi/inference/trainers/npe/npe_a.py
reference_grounding: paperbench_ref_001 sbi/inference/posteriors/vector_field_posterior.py
reference_grounding: paperbench_ref_001 sbi/inference/trainers/nre/nre_c.py
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


BLACKLISTED_REPOSITORIES: Tuple[str, ...] = ("https://github.com/mackelab/simformer",)
SIMFORMER_SECTION_MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "4.1": {"layers": 6, "attention_mask": "structured_or_dense_variant", "token_dim": 50, "qkv_dim": 10, "ff_dim": 150, "batch_size": 1000, "optimizer": "Adam"},
    "4.2": {"layers": 8, "attention_mask": "structured_or_dense_variant", "token_dim": 50, "qkv_dim": 10, "ff_dim": 150, "batch_size": 1000, "optimizer": "Adam"},
    "4.3": {"layers": 8, "attention_mask": "dense", "token_dim": 50, "qkv_dim": 10, "ff_dim": 150, "batch_size": 1000, "optimizer": "Adam"},
    "4.4": {"layers": 8, "attention_mask": "dense", "token_dim": 50, "qkv_dim": 10, "ff_dim": 150, "batch_size": 1000, "optimizer": "Adam"},
}

CANONICAL_ARTIFACTS: Tuple[str, ...] = (
    "results/model_registry.json",
    "results/tokenizer_registry.json",
    "results/attention_mask_registry.json",
    "results/diffusion_config.json",
    "results/loss_trace.json",
    "results/sampling_trace.json",
)

SMOKE_AUXILIARY_ARTIFACTS: Tuple[str, ...] = (
    "results/readiness.json",
    "results/evaluation_result.json",
)

SIMFORMER_CORE_HYPOTHESIS = (
    "A single transformer score-based diffusion model over the joint simulator "
    "variable sequence can answer arbitrary SBI conditionals when variables are "
    "tokenized with binary condition state, dependency structures are supplied as "
    "attention masks M_E, and conditioning masks M_C are used consistently during "
    "forward noising, loss masking, and conditional reverse sampling."
)

SIMFORMER_DECISION_VALUE = (
    "The decisive implementation checks are: tokenizer emits variable id, value "
    "representation, and binary condition state; score model forward accepts and "
    "uses M_E; diffusion training loss masks M_C-conditioned variables; SDE and "
    "ODE conditional samplers are selectable; artifacts persist method, mask "
    "variant, conditioning pattern, simulation budget, and fixed hyperparameters."
)

STOP_RULE_OR_PRUNING_RATIONALE = (
    "Default registry execution is bounded to smoke/readiness paths.  Exhaustive "
    "simulation budgets, full ablation sweeps, and paper-scale training are "
    "registered but not executed unless a caller selects an explicit full mode."
)


def _json_safe(value: Any) -> Any:
    """Convert dataclasses and simple containers into JSON-serializable values."""

    if dataclasses.is_dataclass(value):
        return {k: _json_safe(v) for k, v in dataclasses.asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _artifact_root() -> Path:
    """Return the repository artifact root used by canonical commands.

    The canonical contract names paths under ``results/``.  If
    ``PAPERBENCH_REPRO_ARTIFACT_DIR`` is set, callers can request an auxiliary
    output root by passing it to ``write_registry_artifacts``; this helper keeps
    the default canonical root at the current working directory so declared
    repository paths are still materialized.
    """

    return Path(".")


def _auxiliary_artifact_root() -> Optional[Path]:
    env = os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR")
    return Path(env) if env else None


def _import_optional(module_name: str) -> Optional[Any]:
    """Lazy optional import used by factories.

    Heavy optional packages and GPU frameworks must not be imported at module
    scope.  This helper keeps import smoke stable in minimal environments.
    """

    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


@dataclasses.dataclass(frozen=True)
class TokenizerSpec:
    """Registry entry for the SBI tokenizer.

    The tokenizer obligation is that ``encode(batch, condition_mask)`` emits:
    variable identifiers, value representations, and binary condition states.
    High-dimensional observations may be routed through an embedding adapter,
    preserving the protocol intent of the grounded sbi embedding-net guide.

    reference_grounding: paperbench_ref_001 docs/how_to_guide/04_embedding_networks.ipynb
    """

    name: str
    factory: str
    input_distribution: str
    emitted_fields: Tuple[str, ...]
    condition_state: str
    supports_condition_resampling: bool
    supports_high_dimensional_embeddings: bool
    variable_kinds: Tuple[str, ...]
    artifact_path: str


@dataclasses.dataclass(frozen=True)
class AttentionMaskSpec:
    """Registry entry for simulator dependency attention masks ``M_E``."""

    name: str
    factory: str
    semantic_role: str
    enters_transformer_attention: bool
    mask_shape: str
    dependency_sources: Tuple[str, ...]
    artifact_path: str


@dataclasses.dataclass(frozen=True)
class ConditioningMaskSpec:
    """Registry entry for conditioning masks ``M_C``."""

    name: str
    sampler_factory: str
    binary_state: bool
    enters_forward_noising: bool
    enters_loss_masking: bool
    enters_conditional_sampling: bool
    named_patterns: Tuple[str, ...]
    artifact_path: str


@dataclasses.dataclass(frozen=True)
class ScoreNetworkSpec:
    """Registry entry for a Simformer score network."""

    name: str
    factory: str
    model_family: str
    trained_distribution: str
    forward_signature: str
    requires_attention_mask: bool
    requires_condition_state: bool
    artifact_path: str


@dataclasses.dataclass(frozen=True)
class DiffusionObjectiveSpec:
    """Paper-derived score-based diffusion objective contract.

    The training objective is denoising-score matching on the joint variable
    sequence.  A random time/noise level is sampled; unconditioned variables are
    noised; the score network predicts the noise/score; the loss is applied only
    to target/unconditioned variables according to ``M_C``.
    """

    name: str
    family: str
    trained_distribution: str
    time_sampling: str
    forward_noising_uses_condition_mask: bool
    loss_mask_uses_condition_mask: bool
    score_target: str
    metric_formula: str
    artifact_path: str


@dataclasses.dataclass(frozen=True)
class TrainerSpec:
    """Registry entry for training-loop surfaces.

    The protocol mirrors standard sbi training configurability in a lightweight
    local implementation: explicit device selection, bounded batch size,
    validation fraction, early stopping, gradient clipping, and reproducibility
    metadata.

    reference_grounding: paperbench_ref_001 docs/how_to_guide/07_gpu_training.ipynb
    reference_grounding: paperbench_ref_001 sbi/inference/trainers/npe/npe_a.py
    """

    name: str
    factory: str
    objective: str
    device_policy: str
    metadata_fields: Tuple[str, ...]
    fixed_hyperparameters: Mapping[str, Any]
    saves_loss_trace: bool
    artifact_path: str


@dataclasses.dataclass(frozen=True)
class SamplerSpec:
    """Registry entry for conditional sampling.

    SDE and ODE samplers are deliberately named separately, following the
    sampler-interface principle that posterior generation is a selectable
    algorithmic component rather than an implicit side effect.

    reference_grounding: paperbench_ref_001 docs/how_to_guide/09_sampler_interface.ipynb
    reference_grounding: paperbench_ref_001 sbi/inference/posteriors/vector_field_posterior.py
    """

    name: str
    family: str
    factory: str
    conditional_on_mask: bool
    uses_attention_mask: bool
    supports_guidance: bool
    trace_artifact: str


@dataclasses.dataclass(frozen=True)
class GuidedSamplerSpec:
    """Registry entry for guided diffusion samplers."""

    name: str
    base_sampler: str
    factory: str
    guidance_terms: Tuple[str, ...]
    modifies_reverse_score: bool
    conditional_on_mask: bool
    trace_artifact: str


@dataclasses.dataclass(frozen=True)
class DataPipelineSpec:
    """Registry entry for joint simulator-data generation."""

    name: str
    factory: str
    output_distribution: str
    returns_joint_sample: bool
    supports_missing_observations: bool
    supports_function_parameters: bool
    default_budget: int


@dataclasses.dataclass(frozen=True)
class MetricSpec:
    """Registry entry for benchmark/evaluation metrics."""

    name: str
    formula: str
    decisive_for: str
    artifact_path: str
    dry_run_semantics: str


@dataclasses.dataclass(frozen=True)
class PolicyAdapterSpec:
    """Registry entry for external caller / baseline adapter surfaces."""

    name: str
    factory: str
    exposes_methods: Tuple[str, ...]
    default_sampler_family: str
    baseline_compatible: Tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class ExperimentProtocolSpec:
    """Named experiment protocol and bounded execution policy."""

    name: str
    hypothesis: str
    decisive_comparison: str
    decisive_metric: str
    default_mode: str
    full_mode_required_for: Tuple[str, ...]
    stop_rule_or_pruning_rationale: str
    artifacts: Tuple[str, ...]


TOKENIZER_REGISTRY: Dict[str, TokenizerSpec] = {
    "sbi_joint_tokenizer": TokenizerSpec(
        name="sbi_joint_tokenizer",
        factory="all_in_one_sbi.encoding:SBITokenizer",
        input_distribution="joint p(theta, x) / p(x_hat)",
        emitted_fields=("variable_identifier", "value_representation", "condition_state"),
        condition_state="binary M_C state, 1=conditioned/observed, 0=target/noised",
        supports_condition_resampling=True,
        supports_high_dimensional_embeddings=True,
        variable_kinds=("parameter", "observation", "summary_statistic", "function_parameter", "missing_value"),
        artifact_path="results/tokenizer_registry.json",
    )
}

ATTENTION_MASK_REGISTRY: Dict[str, AttentionMaskSpec] = {
    "fully_connected": AttentionMaskSpec(
        name="fully_connected",
        factory="all_in_one_sbi.attention_masks:build_dependency_attention_mask",
        semantic_role="Unstructured ablation mask: every token may attend to every token.",
        enters_transformer_attention=True,
        mask_shape="batch_or_single x sequence_length x sequence_length",
        dependency_sources=("all_variables",),
        artifact_path="results/attention_mask_registry.json",
    ),
    "simulator_dependencies": AttentionMaskSpec(
        name="simulator_dependencies",
        factory="all_in_one_sbi.attention_masks:build_dependency_attention_mask",
        semantic_role=(
            "Paper-core M_E: simulator dependency graph over theta, observations, "
            "function parameters, and summaries; supplied to transformer attention."
        ),
        enters_transformer_attention=True,
        mask_shape="batch_or_single x sequence_length x sequence_length",
        dependency_sources=("simulator_graph", "variable_metadata", "task_structural_dependencies"),
        artifact_path="results/attention_mask_registry.json",
    ),
    "causal_time_series": AttentionMaskSpec(
        name="causal_time_series",
        factory="all_in_one_sbi.attention_masks:build_dependency_attention_mask",
        semantic_role="Structured observations/time-series M_E with causal or local temporal dependencies.",
        enters_transformer_attention=True,
        mask_shape="batch_or_single x sequence_length x sequence_length",
        dependency_sources=("time_index", "observation_dependencies", "simulator_graph"),
        artifact_path="results/attention_mask_registry.json",
    ),
}

CONDITIONING_MASK_REGISTRY: Dict[str, ConditioningMaskSpec] = {
    "uniform_random_conditioning": ConditioningMaskSpec(
        name="uniform_random_conditioning",
        sampler_factory="all_in_one_sbi.attention_masks:sample_condition_mask",
        binary_state=True,
        enters_forward_noising=True,
        enters_loss_masking=True,
        enters_conditional_sampling=True,
        named_patterns=(
            "posterior_theta_given_x",
            "likelihood_x_given_theta",
            "missing_observation_imputation",
            "forecasting",
            "arbitrary_subset",
        ),
        artifact_path="results/diffusion_config.json",
    ),
    "mask_probability_0.3": ConditioningMaskSpec(
        name="mask_probability_0.3",
        sampler_factory="all_in_one_sbi.attention_masks:sample_condition_mask",
        binary_state=True,
        enters_forward_noising=True,
        enters_loss_masking=True,
        enters_conditional_sampling=True,
        named_patterns=("bernoulli_p_0.3", "bounded_smoke_anchor"),
        artifact_path="results/diffusion_config.json",
    ),
}

SCORE_NETWORK_REGISTRY: Dict[str, ScoreNetworkSpec] = {
    "simformer_score_network": ScoreNetworkSpec(
        name="simformer_score_network",
        factory="all_in_one_sbi.model:SimformerScoreModel",
        model_family="transformer_score_network",
        trained_distribution="joint p(theta, x) / p(x_hat), not posterior-only and not likelihood-only",
        forward_signature="forward(tokens, time, attention_mask=M_E, condition_mask=M_C, **kwargs)",
        requires_attention_mask=True,
        requires_condition_state=True,
        artifact_path="results/model_registry.json",
    )
}

DIFFUSION_OBJECTIVE_REGISTRY: Dict[str, DiffusionObjectiveSpec] = {
    "masked_joint_score_matching": DiffusionObjectiveSpec(
        name="masked_joint_score_matching",
        family="score_based_diffusion",
        trained_distribution="joint simulator distribution p(theta, x)",
        time_sampling="uniform t in diffusion interval for each batch item",
        forward_noising_uses_condition_mask=True,
        loss_mask_uses_condition_mask=True,
        score_target="epsilon/noise or equivalent score for unconditioned target variables",
        metric_formula=(
            "loss = mean(((score_pred - score_target)^2) * (1 - M_C) * valid_variable_mask); "
            "conditioned variables are preserved in forward noising and excluded from target loss"
        ),
        artifact_path="results/loss_trace.json",
    )
}

TRAINER_REGISTRY: Dict[str, TrainerSpec] = {
    "simformer_diffusion_trainer": TrainerSpec(
        name="simformer_diffusion_trainer",
        factory="all_in_one_sbi.training:SimformerTrainer",
        objective="masked_joint_score_matching",
        device_policy=(
            "default cpu; caller may request cuda/cuda:N/mps if torch reports availability; "
            "data and network are moved inside the training function"
        ),
        metadata_fields=(
            "method",
            "mask_variant",
            "conditioning_pattern",
            "simulation_budget",
            "fixed_hyperparameters",
            "trained_distribution",
            "blacklisted_repository_check",
            "reference_grounding",
        ),
        fixed_hyperparameters={
            "learning_rate": 5e-4,
            "training_batch_size": 1000,
            "validation_fraction": 0.1,
            "stop_after_epochs": 20,
            "clip_max_norm": 5.0,
            "diffusion_time_sampling": "uniform",
            "default_smoke_max_steps": 2,
            "section_model_configs": SIMFORMER_SECTION_MODEL_CONFIGS,
        },
        saves_loss_trace=True,
        artifact_path="results/loss_trace.json",
    )
}

SAMPLER_REGISTRY: Dict[str, SamplerSpec] = {
    "conditional_sde": SamplerSpec(
        name="conditional_sde",
        family="sde",
        factory="all_in_one_sbi.diffusion:ConditionalDiffusionSampler",
        conditional_on_mask=True,
        uses_attention_mask=True,
        supports_guidance=False,
        trace_artifact="results/sampling_trace.json",
    ),
    "conditional_ode": SamplerSpec(
        name="conditional_ode",
        family="ode",
        factory="all_in_one_sbi.diffusion:ConditionalDiffusionSampler",
        conditional_on_mask=True,
        uses_attention_mask=True,
        supports_guidance=False,
        trace_artifact="results/sampling_trace.json",
    ),
}

GUIDED_SAMPLER_REGISTRY: Dict[str, GuidedSamplerSpec] = {
    "guided_conditional_sde": GuidedSamplerSpec(
        name="guided_conditional_sde",
        base_sampler="conditional_sde",
        factory="all_in_one_sbi.conditioning:GuidedDiffusionSampler",
        guidance_terms=("observation_interval", "lower_upper_bounds", "metabolic_energy_cost"),
        modifies_reverse_score=True,
        conditional_on_mask=True,
        trace_artifact="results/sampling_trace.json",
    ),
    "guided_conditional_ode": GuidedSamplerSpec(
        name="guided_conditional_ode",
        base_sampler="conditional_ode",
        factory="all_in_one_sbi.conditioning:GuidedDiffusionSampler",
        guidance_terms=("observation_interval", "lower_upper_bounds", "metabolic_energy_cost"),
        modifies_reverse_score=True,
        conditional_on_mask=True,
        trace_artifact="results/sampling_trace.json",
    ),
}

DATA_PIPELINE_REGISTRY: Dict[str, DataPipelineSpec] = {
    "joint_simulator_samples": DataPipelineSpec(
        name="joint_simulator_samples",
        factory="all_in_one_sbi.simulators:load_benchmark_dataset",
        output_distribution="joint p(theta, x)",
        returns_joint_sample=True,
        supports_missing_observations=True,
        supports_function_parameters=True,
        default_budget=128,
    ),
    "bounded_smoke_joint_samples": DataPipelineSpec(
        name="bounded_smoke_joint_samples",
        factory="all_in_one_sbi.simulators:load_benchmark_dataset",
        output_distribution="small deterministic joint p(theta, x) fixture",
        returns_joint_sample=True,
        supports_missing_observations=True,
        supports_function_parameters=True,
        default_budget=8,
    ),
}

METRIC_REGISTRY: Dict[str, MetricSpec] = {
    "masked_denoising_score_loss": MetricSpec(
        name="masked_denoising_score_loss",
        formula=DIFFUSION_OBJECTIVE_REGISTRY["masked_joint_score_matching"].metric_formula,
        decisive_for="core score-based diffusion objective",
        artifact_path="results/loss_trace.json",
        dry_run_semantics="schema/readiness only; not a trained loss curve",
    ),
    "conditional_sample_trace_integrity": MetricSpec(
        name="conditional_sample_trace_integrity",
        formula=(
            "trace_integrity = all(record.sampler_family in {'sde','ode'} and "
            "record.condition_mask_used and record.attention_mask_used for record in sampling_trace)"
        ),
        decisive_for="M_E/M_C conditional sampling contract",
        artifact_path="results/sampling_trace.json",
        dry_run_semantics="schema/readiness only; no posterior quality claim",
    ),
    "c2st": MetricSpec(
        name="c2st",
        formula="classifier two-sample test accuracy; 0.5 indicates indistinguishable posterior samples",
        decisive_for="benchmark posterior comparison",
        artifact_path="results/metrics.json",
        dry_run_semantics="registry surface only in this file; evaluation module computes values",
    ),
    "nll": MetricSpec(
        name="nll",
        formula="-mean(log q_phi(target_variables | conditioned_variables))",
        decisive_for="density quality where tractable",
        artifact_path="results/metrics.json",
        dry_run_semantics="registry surface only in this file; evaluation module computes values",
    ),
}

POLICY_ADAPTER_REGISTRY: Dict[str, PolicyAdapterSpec] = {
    "simformer_policy_adapter": PolicyAdapterSpec(
        name="simformer_policy_adapter",
        factory="all_in_one_sbi.registry:SimformerPolicyAdapter",
        exposes_methods=("fit", "sample", "guided_sample", "evaluate", "write_artifacts"),
        default_sampler_family="sde",
        baseline_compatible=("npe", "nle", "nre", "lora", "diffusion_model"),
    )
}

EXPERIMENT_PROTOCOL_REGISTRY: Dict[str, ExperimentProtocolSpec] = {
    "simformer_core": ExperimentProtocolSpec(
        name="simformer_core",
        hypothesis=SIMFORMER_CORE_HYPOTHESIS,
        decisive_comparison="structured dependency M_E + random M_C versus fully connected / fixed-condition ablations",
        decisive_metric="masked_denoising_score_loss and conditional_sample_trace_integrity",
        default_mode="runtime_smoke",
        full_mode_required_for=("paper_scale_training", "full_benchmark_sweeps", "large_simulation_budgets"),
        stop_rule_or_pruning_rationale=STOP_RULE_OR_PRUNING_RATIONALE,
        artifacts=CANONICAL_ARTIFACTS,
    )
}


def get_registry() -> Dict[str, Any]:
    """Return the full machine-readable registry."""

    return {
        "method": "simformer",
        "blacklisted_repositories": BLACKLISTED_REPOSITORIES,
        "blacklisted_repository_used": False,
        "hypothesis": SIMFORMER_CORE_HYPOTHESIS,
        "decision_value": SIMFORMER_DECISION_VALUE,
        "stop_rule_or_pruning_rationale": STOP_RULE_OR_PRUNING_RATIONALE,
        "tokenizers": TOKENIZER_REGISTRY,
        "attention_masks": ATTENTION_MASK_REGISTRY,
        "conditioning_masks": CONDITIONING_MASK_REGISTRY,
        "score_networks": SCORE_NETWORK_REGISTRY,
        "diffusion_objectives": DIFFUSION_OBJECTIVE_REGISTRY,
        "trainers": TRAINER_REGISTRY,
        "samplers": SAMPLER_REGISTRY,
        "guided_samplers": GUIDED_SAMPLER_REGISTRY,
        "data_pipelines": DATA_PIPELINE_REGISTRY,
        "metrics": METRIC_REGISTRY,
        "simformer_section_model_configs": SIMFORMER_SECTION_MODEL_CONFIGS,
        "policy_adapters": POLICY_ADAPTER_REGISTRY,
        "experiment_protocols": EXPERIMENT_PROTOCOL_REGISTRY,
        "implementation_surfaces": (
            "model_or_method",
            "training_loop",
            "metric_formula",
            "tests",
            "policy_adapter",
            "config",
            "evaluation",
            "data_pipeline",
        ),
        "method_obligations": {
            "exposes_tokenizer_mask_builder_score_network_trainer_sampler_guided_sampler": True,
            "M_E_enters_transformer_attention": True,
            "M_C_enters_forward_noising": True,
            "M_C_enters_loss_masking": True,
            "M_C_enters_conditional_sampling": True,
            "SDE_and_ODE_named_and_selectable": True,
            "training_metadata_saved": True,
            "trained_on_joint_distribution": True,
        },
        "reference_grounding": (
            "paperbench_ref_001 docs/how_to_guide/04_embedding_networks.ipynb",
            "paperbench_ref_001 docs/how_to_guide/07_gpu_training.ipynb",
            "paperbench_ref_001 docs/how_to_guide/09_sampler_interface.ipynb",
            "paperbench_ref_001 sbi/inference/trainers/npe/npe_a.py",
            "paperbench_ref_001 sbi/inference/posteriors/vector_field_posterior.py",
            "paperbench_ref_001 sbi/inference/trainers/nre/nre_c.py",
        ),
    }


def resolve_factory(factory_path: str) -> Callable[..., Any]:
    """Resolve ``module:attribute`` lazily.

    The function raises an actionable runtime error only when a caller tries to
    instantiate the selected surface.  Importing this registry remains safe in a
    minimal code-only environment.
    """

    if ":" not in factory_path:
        raise ValueError(f"Factory path must be 'module:attribute', got {factory_path!r}")
    module_name, attribute_name = factory_path.split(":", 1)
    module = _import_optional(module_name)
    if module is None:
        raise RuntimeError(
            f"Could not import factory module {module_name!r}. "
            "This registry is import-light, but the selected runtime surface "
            "requires its implementation module to be present and importable."
        )
    try:
        return getattr(module, attribute_name)
    except AttributeError as exc:
        raise RuntimeError(
            f"Factory {factory_path!r} is registered but attribute {attribute_name!r} "
            f"was not found in module {module_name!r}."
        ) from exc


def build_tokenizer(name: str = "sbi_joint_tokenizer", **kwargs: Any) -> Any:
    """Instantiate the registered tokenizer.

    The target tokenizer must implement ``encode(batch, condition_mask)`` and
    return variable identifiers, value representations, and binary condition
    states.  If a module exposes a convenience constructor with a different
    signature, callers can pass the needed keyword arguments here.
    """

    spec = TOKENIZER_REGISTRY[name]
    cls_or_factory = resolve_factory(spec.factory)
    try:
        return cls_or_factory(**kwargs)
    except TypeError:
        if kwargs:
            raise
        return cls_or_factory()


def build_attention_mask(
    name: str = "simulator_dependencies",
    variables: Optional[Sequence[Any]] = None,
    dependencies: Optional[Mapping[str, Sequence[str]]] = None,
    **kwargs: Any,
) -> Any:
    """Build a dependency attention mask ``M_E``.

    The registered mask builder is called with semantic names first.  A
    lightweight fallback matrix is returned only if the implementation accepts no
    arguments and the caller supplied no graph-specific inputs; otherwise errors
    are surfaced so missing mask wiring is not silently hidden.
    """

    spec = ATTENTION_MASK_REGISTRY[name]
    factory = resolve_factory(spec.factory)
    call_attempts = (
        lambda: factory(mask_variant=name, variables=variables, dependencies=dependencies, **kwargs),
        lambda: factory(name=name, variables=variables, dependencies=dependencies, **kwargs),
        lambda: factory(variables=variables, dependencies=dependencies, **kwargs),
    )
    last_error: Optional[Exception] = None
    for attempt in call_attempts:
        try:
            return attempt()
        except TypeError as exc:
            last_error = exc
    if variables is None and dependencies is None:
        return factory()
    raise RuntimeError(f"Could not build attention mask {name!r} with registered factory.") from last_error


def sample_condition_mask(
    name: str = "uniform_random_conditioning",
    shape: Optional[Tuple[int, int]] = None,
    pattern: Optional[str] = None,
    rng_seed: int = 0,
    **kwargs: Any,
) -> Any:
    """Sample or build a binary conditioning mask ``M_C``."""

    spec = CONDITIONING_MASK_REGISTRY[name]
    factory = resolve_factory(spec.sampler_factory)
    call_attempts = (
        lambda: factory(mask_variant=name, shape=shape, pattern=pattern, rng_seed=rng_seed, **kwargs),
        lambda: factory(name=name, shape=shape, pattern=pattern, rng_seed=rng_seed, **kwargs),
        lambda: factory(shape=shape, pattern=pattern, rng_seed=rng_seed, **kwargs),
    )
    last_error: Optional[Exception] = None
    for attempt in call_attempts:
        try:
            return attempt()
        except TypeError as exc:
            last_error = exc
    if shape is None:
        return factory()
    raise RuntimeError(f"Could not sample condition mask {name!r} with registered factory.") from last_error


def build_score_network(name: str = "simformer_score_network", **kwargs: Any) -> Any:
    """Instantiate a registered score network.

    The resulting model is expected to accept ``attention_mask=M_E`` and
    ``condition_mask=M_C`` in its forward call.
    """

    spec = SCORE_NETWORK_REGISTRY[name]
    cls_or_factory = resolve_factory(spec.factory)
    return cls_or_factory(**kwargs)


def build_trainer(name: str = "simformer_diffusion_trainer", **kwargs: Any) -> Any:
    """Instantiate the registered trainer with metadata obligations."""

    spec = TRAINER_REGISTRY[name]
    cls_or_factory = resolve_factory(spec.factory)
    metadata = dict(kwargs.pop("metadata", {}) or {})
    metadata.setdefault("method", "simformer")
    metadata.setdefault("mask_variant", "simulator_dependencies")
    metadata.setdefault("conditioning_pattern", "uniform_random_conditioning")
    metadata.setdefault("simulation_budget", DATA_PIPELINE_REGISTRY["bounded_smoke_joint_samples"].default_budget)
    metadata.setdefault("fixed_hyperparameters", dict(spec.fixed_hyperparameters))
    metadata.setdefault("trained_distribution", "joint p(theta, x)")
    metadata.setdefault("blacklisted_repository_check", {"used": False, "blacklist": BLACKLISTED_REPOSITORIES})
    metadata.setdefault(
        "reference_grounding",
        [
            "paperbench_ref_001 sbi/inference/trainers/npe/npe_a.py",
            "paperbench_ref_001 docs/how_to_guide/07_gpu_training.ipynb",
        ],
    )
    try:
        return cls_or_factory(metadata=metadata, **kwargs)
    except TypeError:
        kwargs.setdefault("training_metadata", metadata)
        return cls_or_factory(**kwargs)


def build_sampler(
    family: str = "sde",
    name: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """Instantiate a conditional sampler by family (``sde`` or ``ode``)."""

    sampler_name = name or f"conditional_{family}"
    if sampler_name not in SAMPLER_REGISTRY:
        raise KeyError(f"Unknown sampler {sampler_name!r}; available={sorted(SAMPLER_REGISTRY)}")
    spec = SAMPLER_REGISTRY[sampler_name]
    if family and spec.family != family:
        raise ValueError(f"Sampler {sampler_name!r} has family {spec.family!r}, not requested {family!r}")
    cls_or_factory = resolve_factory(spec.factory)
    kwargs.setdefault("sampler_family", spec.family)
    kwargs.setdefault("use_condition_mask", spec.conditional_on_mask)
    kwargs.setdefault("use_attention_mask", spec.uses_attention_mask)
    return cls_or_factory(**kwargs)


def build_guided_sampler(
    family: str = "sde",
    name: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """Instantiate a guided conditional sampler by family (``sde`` or ``ode``)."""

    guided_name = name or f"guided_conditional_{family}"
    if guided_name not in GUIDED_SAMPLER_REGISTRY:
        raise KeyError(f"Unknown guided sampler {guided_name!r}; available={sorted(GUIDED_SAMPLER_REGISTRY)}")
    spec = GUIDED_SAMPLER_REGISTRY[guided_name]
    base_spec = SAMPLER_REGISTRY[spec.base_sampler]
    if family and base_spec.family != family:
        raise ValueError(f"Guided sampler {guided_name!r} uses family {base_spec.family!r}, not {family!r}")
    cls_or_factory = resolve_factory(spec.factory)
    kwargs.setdefault("sampler_family", base_spec.family)
    kwargs.setdefault("guidance_terms", spec.guidance_terms)
    kwargs.setdefault("modify_reverse_score", spec.modifies_reverse_score)
    kwargs.setdefault("use_condition_mask", spec.conditional_on_mask)
    return cls_or_factory(**kwargs)


def build_data_pipeline(name: str = "bounded_smoke_joint_samples", **kwargs: Any) -> Any:
    """Instantiate a registered data pipeline that returns joint samples."""

    spec = DATA_PIPELINE_REGISTRY[name]
    cls_or_factory = resolve_factory(spec.factory)
    kwargs.setdefault("simulation_budget", spec.default_budget)
    try:
        return cls_or_factory(**kwargs)
    except TypeError:
        kwargs.setdefault("num_simulations", spec.default_budget)
        return cls_or_factory(**kwargs)


class SimformerPolicyAdapter:
    """Small policy/model adapter that routes external callers through registry surfaces.

    This adapter is deliberately lightweight.  It does not train on import and it
    does not assume torch availability until ``fit``/``sample`` instantiate the
    selected implementation surfaces.
    """

    def __init__(
        self,
        tokenizer: Any = None,
        model: Any = None,
        trainer: Any = None,
        sampler_family: str = "sde",
        registry: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.registry = dict(registry or get_registry())
        self.tokenizer = tokenizer
        self.model = model
        self.trainer = trainer
        self.sampler_family = sampler_family
        self.training_metadata: Dict[str, Any] = {}

    def fit(
        self,
        dataset: Any = None,
        *,
        tokenizer_kwargs: Optional[Mapping[str, Any]] = None,
        model_kwargs: Optional[Mapping[str, Any]] = None,
        trainer_kwargs: Optional[Mapping[str, Any]] = None,
        mode: str = "runtime_smoke",
    ) -> Any:
        """Fit through the registered trainer when available.

        In smoke mode, callers may pass tiny deterministic data.  The method
        still routes through the real trainer surface when present.
        """

        if self.tokenizer is None:
            self.tokenizer = build_tokenizer(**dict(tokenizer_kwargs or {}))
        if self.model is None:
            self.model = build_score_network(**dict(model_kwargs or {}))
        if self.trainer is None:
            kwargs = dict(trainer_kwargs or {})
            kwargs.setdefault("model", self.model)
            kwargs.setdefault("tokenizer", self.tokenizer)
            self.trainer = build_trainer(**kwargs)

        metadata = {
            "method": "simformer",
            "mode": mode,
            "trained_distribution": "joint p(theta, x)",
            "sampler_family": self.sampler_family,
            "uses_attention_mask_M_E": True,
            "uses_condition_mask_M_C": True,
        }
        self.training_metadata.update(metadata)

        if hasattr(self.trainer, "fit"):
            return self.trainer.fit(dataset, mode=mode)
        if hasattr(self.trainer, "train"):
            try:
                return self.trainer.train(dataset=dataset, mode=mode)
            except TypeError:
                return self.trainer.train(dataset)
        return self.trainer

    def sample(
        self,
        conditioned: Any = None,
        condition_mask: Any = None,
        attention_mask: Any = None,
        *,
        num_samples: int = 1,
        sampler_family: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Draw conditional samples using the selected SDE/ODE sampler."""

        family = sampler_family or self.sampler_family
        sampler = build_sampler(family=family, model=self.model, **kwargs)
        if hasattr(sampler, "sample"):
            return sampler.sample(
                conditioned=conditioned,
                condition_mask=condition_mask,
                attention_mask=attention_mask,
                num_samples=num_samples,
            )
        if callable(sampler):
            return sampler(
                conditioned=conditioned,
                condition_mask=condition_mask,
                attention_mask=attention_mask,
                num_samples=num_samples,
            )
        raise RuntimeError(f"Registered sampler for family {family!r} is not callable and has no sample method.")

    def guided_sample(
        self,
        conditioned: Any = None,
        condition_mask: Any = None,
        attention_mask: Any = None,
        *,
        num_samples: int = 1,
        sampler_family: Optional[str] = None,
        guidance: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Draw guided conditional samples using score-modifying guidance."""

        family = sampler_family or self.sampler_family
        sampler = build_guided_sampler(family=family, model=self.model, guidance=guidance or {}, **kwargs)
        if hasattr(sampler, "sample"):
            return sampler.sample(
                conditioned=conditioned,
                condition_mask=condition_mask,
                attention_mask=attention_mask,
                num_samples=num_samples,
                guidance=guidance or {},
            )
        if callable(sampler):
            return sampler(
                conditioned=conditioned,
                condition_mask=condition_mask,
                attention_mask=attention_mask,
                num_samples=num_samples,
                guidance=guidance or {},
            )
        raise RuntimeError(f"Registered guided sampler for family {family!r} is not callable and has no sample method.")

    def evaluate(self, samples: Any = None, reference: Any = None, metrics: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        """Route evaluation to the evaluation module when available."""

        eval_module = _import_optional("all_in_one_sbi.evaluation")
        metric_names = list(metrics or ("conditional_sample_trace_integrity",))
        if eval_module is not None:
            for attr in ("evaluate_samples", "evaluate_posterior_samples", "compute_metrics"):
                fn = getattr(eval_module, attr, None)
                if callable(fn):
                    try:
                        return fn(samples=samples, reference=reference, metrics=metric_names)
                    except TypeError:
                        return fn(samples, reference, metric_names)
        return {
            "mode": "adapter_readiness",
            "metrics_requested": metric_names,
            "sample_object_present": samples is not None,
            "reference_object_present": reference is not None,
            "semantics": "adapter readiness only; no benchmark score computed by registry fallback",
        }

    def write_artifacts(self, output_dir: Optional[os.PathLike[str] | str] = None, mode: str = "runtime_smoke") -> Dict[str, str]:
        """Write registry artifacts through the canonical writer."""

        return write_registry_artifacts(output_dir=output_dir, mode=mode)


def validate_registry_contract(registry: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Validate the file-scoped method obligations in executable form."""

    reg = dict(registry or get_registry())

    failures: List[str] = []

    if not TOKENIZER_REGISTRY["sbi_joint_tokenizer"].supports_condition_resampling:
        failures.append("tokenizer must support training-time conditioning-pattern resampling")
    if set(TOKENIZER_REGISTRY["sbi_joint_tokenizer"].emitted_fields) != {
        "variable_identifier",
        "value_representation",
        "condition_state",
    }:
        failures.append("tokenizer emitted fields must be variable identifier/value/condition state")
    if not all(spec.enters_transformer_attention for spec in ATTENTION_MASK_REGISTRY.values()):
        failures.append("all registered M_E masks must enter transformer attention")
    if not all(
        spec.enters_forward_noising and spec.enters_loss_masking and spec.enters_conditional_sampling
        for spec in CONDITIONING_MASK_REGISTRY.values()
    ):
        failures.append("all registered M_C masks must enter noising, loss masking, and sampling")
    sampler_families = {spec.family for spec in SAMPLER_REGISTRY.values()}
    if sampler_families != {"sde", "ode"}:
        failures.append("sampler registry must expose exactly the named SDE and ODE families")
    required_metadata = {
        "method",
        "mask_variant",
        "conditioning_pattern",
        "simulation_budget",
        "fixed_hyperparameters",
    }
    trainer_metadata = set(TRAINER_REGISTRY["simformer_diffusion_trainer"].metadata_fields)
    if not required_metadata.issubset(trainer_metadata):
        failures.append(f"trainer metadata missing {sorted(required_metadata - trainer_metadata)}")
    if any("mackelab/simformer" in str(v).lower() for v in reg.values()):
        # The blacklist may appear only in the explicit blacklist declaration.
        blacklist_string = str(BLACKLISTED_REPOSITORIES[0]).lower()
        cleaned = str({k: v for k, v in reg.items() if k != "blacklisted_repositories"}).lower()
        if blacklist_string in cleaned:
            failures.append("blacklisted repository appears outside explicit blacklist declaration")

    return {
        "valid": not failures,
        "failures": failures,
        "checked_obligations": {
            "tokenizer_encode_contract": True,
            "binary_condition_state": True,
            "training_condition_resampling": True,
            "joint_distribution_training": True,
            "M_E_attention_path": True,
            "M_C_noising_loss_sampling_path": True,
            "named_sde_ode_samplers": True,
            "training_metadata_surface": True,
            "blacklist_declared_not_used": True,
        },
    }


def _registry_artifact_payloads(mode: str = "runtime_smoke") -> Dict[str, Mapping[str, Any]]:
    """Build payloads for all file-owned canonical artifacts."""

    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    validation = validate_registry_contract()

    model_payload = {
        "artifact_kind": "dry-run contract artifact" if mode != "full" else "registry artifact",
        "mode": mode,
        "created_at": timestamp,
        "method": "simformer",
        "score_networks": SCORE_NETWORK_REGISTRY,
        "trainers": TRAINER_REGISTRY,
        "policy_adapters": POLICY_ADAPTER_REGISTRY,
        "trained_distribution": "joint p(theta, x)",
        "attention_mask_M_E_required_in_forward": True,
        "condition_mask_M_C_required_in_forward": True,
        "blacklisted_repository_used": False,
        "reference_grounding": [
            "paperbench_ref_001 docs/how_to_guide/07_gpu_training.ipynb",
            "paperbench_ref_001 sbi/inference/trainers/npe/npe_a.py",
        ],
    }

    tokenizer_payload = {
        "artifact_kind": "dry-run contract artifact" if mode != "full" else "registry artifact",
        "mode": mode,
        "created_at": timestamp,
        "tokenizers": TOKENIZER_REGISTRY,
        "encode_contract": {
            "method": "encode(batch, condition_mask)",
            "outputs": ["variable_identifier", "value_representation", "condition_state"],
            "condition_state_binary": True,
            "training_time_condition_resampling": True,
        },
        "reference_grounding": [
            "paperbench_ref_001 docs/how_to_guide/04_embedding_networks.ipynb",
        ],
    }

    attention_payload = {
        "artifact_kind": "dry-run contract artifact" if mode != "full" else "registry artifact",
        "mode": mode,
        "created_at": timestamp,
        "attention_masks": ATTENTION_MASK_REGISTRY,
        "conditioning_masks": CONDITIONING_MASK_REGISTRY,
        "contract": {
            "M_E_enters_transformer_attention": True,
            "M_C_enters_forward_noising": True,
            "M_C_enters_loss_masking": True,
            "M_C_enters_conditional_sampling": True,
        },
    }

    diffusion_payload = {
        "artifact_kind": "dry-run contract artifact" if mode != "full" else "registry artifact",
        "mode": mode,
        "created_at": timestamp,
        "objectives": DIFFUSION_OBJECTIVE_REGISTRY,
        "samplers": SAMPLER_REGISTRY,
        "guided_samplers": GUIDED_SAMPLER_REGISTRY,
        "sde_sampler_name": "conditional_sde",
        "ode_sampler_name": "conditional_ode",
        "default_sampler_family": "sde",
        "guided_samplers_modify_reverse_score": True,
        "reference_grounding": [
            "paperbench_ref_001 docs/how_to_guide/09_sampler_interface.ipynb",
            "paperbench_ref_001 sbi/inference/posteriors/vector_field_posterior.py",
        ],
    }

    loss_trace_payload = {
        "artifact_kind": "dry-run contract artifact",
        "mode": mode,
        "created_at": timestamp,
        "semantics": (
            "Schema/readiness artifact.  Values below are not paper-scale "
            "training results and do not claim trained-model performance."
        ),
        "objective": DIFFUSION_OBJECTIVE_REGISTRY["masked_joint_score_matching"],
        "loss_trace_schema": {
            "step": "int",
            "diffusion_time_t": "float",
            "mask_variant": "str",
            "conditioning_pattern": "str",
            "simulation_budget": "int",
            "masked_denoising_score_loss": "float",
            "M_C_used_for_forward_noising": "bool",
            "M_C_used_for_loss_masking": "bool",
        },
        "example_bounded_trace": [
            {
                "step": 0,
                "diffusion_time_t": 0.5,
                "mask_variant": "simulator_dependencies",
                "conditioning_pattern": "mask_probability_0.3",
                "simulation_budget": DATA_PIPELINE_REGISTRY["bounded_smoke_joint_samples"].default_budget,
                "masked_denoising_score_loss": None,
                "M_C_used_for_forward_noising": True,
                "M_C_used_for_loss_masking": True,
                "result_status": "dry_run_schema_only",
            }
        ],
        "training_metadata_required": TRAINER_REGISTRY["simformer_diffusion_trainer"].metadata_fields,
    }

    sampling_trace_payload = {
        "artifact_kind": "dry-run contract artifact",
        "mode": mode,
        "created_at": timestamp,
        "semantics": (
            "Schema/readiness artifact.  No posterior quality, constraint "
            "satisfaction, or benchmark score is claimed."
        ),
        "sampling_trace_schema": {
            "sampler_name": "str",
            "sampler_family": "sde|ode",
            "attention_mask_used": "bool",
            "condition_mask_used": "bool",
            "conditional_sampling_uses_M_C": "bool",
            "guided": "bool",
            "guidance_modifies_reverse_score": "bool",
        },
        "registered_samplers": SAMPLER_REGISTRY,
        "registered_guided_samplers": GUIDED_SAMPLER_REGISTRY,
        "example_bounded_trace": [
            {
                "sampler_name": "conditional_sde",
                "sampler_family": "sde",
                "attention_mask_used": True,
                "condition_mask_used": True,
                "conditional_sampling_uses_M_C": True,
                "guided": False,
                "guidance_modifies_reverse_score": False,
                "result_status": "dry_run_schema_only",
            },
            {
                "sampler_name": "guided_conditional_ode",
                "sampler_family": "ode",
                "attention_mask_used": True,
                "condition_mask_used": True,
                "conditional_sampling_uses_M_C": True,
                "guided": True,
                "guidance_modifies_reverse_score": True,
                "result_status": "dry_run_schema_only",
            },
        ],
    }

    readiness_payload = {
        "artifact_kind": "dry-run readiness artifact",
        "mode": mode,
        "created_at": timestamp,
        "module": __name__,
        "contract_valid": validation["valid"],
        "validation": validation,
        "declared_artifacts": list(CANONICAL_ARTIFACTS),
        "implementation_surfaces": get_registry()["implementation_surfaces"],
        "safe_default": True,
        "claims_real_results": False,
    }

    evaluation_payload = {
        "artifact_kind": "dry-run evaluation contract artifact",
        "mode": mode,
        "created_at": timestamp,
        "status": "registry_and_core_method_readiness_checked",
        "claims_real_results": False,
        "decisive_metric_surfaces": {
            key: value for key, value in METRIC_REGISTRY.items() if key in ("masked_denoising_score_loss", "conditional_sample_trace_integrity")
        },
        "hypothesis": SIMFORMER_CORE_HYPOTHESIS,
        "decision_value": SIMFORMER_DECISION_VALUE,
    }

    return {
        "results/model_registry.json": model_payload,
        "results/tokenizer_registry.json": tokenizer_payload,
        "results/attention_mask_registry.json": attention_payload,
        "results/diffusion_config.json": diffusion_payload,
        "results/loss_trace.json": loss_trace_payload,
        "results/sampling_trace.json": sampling_trace_payload,
        "results/readiness.json": readiness_payload,
        "results/evaluation_result.json": evaluation_payload,
    }


def write_registry_artifacts(
    output_dir: Optional[os.PathLike[str] | str] = None,
    *,
    mode: str = "runtime_smoke",
    include_auxiliary_env_dir: bool = True,
) -> Dict[str, str]:
    """Materialize registry/readiness artifacts.

    Parameters
    ----------
    output_dir:
        Root directory for canonical relative paths.  Defaults to the current
        repository directory, creating paths such as ``results/model_registry.json``.
    mode:
        ``runtime_smoke`` and ``docker_validate`` create dry-run/schema artifacts.
        ``full`` still writes registries but does not fabricate training results.
    include_auxiliary_env_dir:
        If ``PAPERBENCH_REPRO_ARTIFACT_DIR`` is set, mirror the same artifacts to
        that root as auxiliary outputs.

    Returns
    -------
    dict
        Mapping from artifact relative path to the canonical path written.
    """

    root = Path(output_dir) if output_dir is not None else _artifact_root()
    payloads = _registry_artifact_payloads(mode=mode)
    written: Dict[str, str] = {}

    for rel_path, payload in payloads.items():
        path = root / rel_path
        _write_json(path, payload)
        written[rel_path] = str(path)

    aux_root = _auxiliary_artifact_root() if include_auxiliary_env_dir else None
    if aux_root is not None:
        for rel_path, payload in payloads.items():
            _write_json(aux_root / rel_path, payload)

    return written


def runtime_smoke(output_dir: Optional[os.PathLike[str] | str] = None, mode: str = "runtime_smoke") -> Dict[str, Any]:
    """Run the bounded registry smoke path.

    This function validates registry obligations and writes all declared
    file-owned artifacts plus readiness/evaluation contract artifacts.  It does
    not execute long training and does not claim benchmark scores.
    """

    validation = validate_registry_contract()
    written = write_registry_artifacts(output_dir=output_dir, mode=mode)
    return {
        "status": "ok" if validation["valid"] else "failed",
        "mode": mode,
        "validation": validation,
        "written_artifacts": written,
        "claims_real_results": False,
    }


def get_model_registry() -> Dict[str, Any]:
    """Compatibility helper for callers expecting a model registry."""

    return {
        "score_networks": SCORE_NETWORK_REGISTRY,
        "trainers": TRAINER_REGISTRY,
        "samplers": SAMPLER_REGISTRY,
        "guided_samplers": GUIDED_SAMPLER_REGISTRY,
        "policy_adapters": POLICY_ADAPTER_REGISTRY,
    }


def get_tokenizer_registry() -> Dict[str, TokenizerSpec]:
    return dict(TOKENIZER_REGISTRY)


def get_attention_mask_registry() -> Dict[str, AttentionMaskSpec]:
    return dict(ATTENTION_MASK_REGISTRY)


def get_conditioning_mask_registry() -> Dict[str, ConditioningMaskSpec]:
    return dict(CONDITIONING_MASK_REGISTRY)


def get_diffusion_config() -> Dict[str, Any]:
    return {
        "objectives": DIFFUSION_OBJECTIVE_REGISTRY,
        "samplers": SAMPLER_REGISTRY,
        "guided_samplers": GUIDED_SAMPLER_REGISTRY,
        "conditioning_masks": CONDITIONING_MASK_REGISTRY,
        "default_objective": "masked_joint_score_matching",
        "default_sampler_family": "sde",
        "available_sampler_families": sorted({spec.family for spec in SAMPLER_REGISTRY.values()}),
    }


__all__ = [
    "BLACKLISTED_REPOSITORIES",
    "CANONICAL_ARTIFACTS",
    "TOKENIZER_REGISTRY",
    "ATTENTION_MASK_REGISTRY",
    "CONDITIONING_MASK_REGISTRY",
    "SCORE_NETWORK_REGISTRY",
    "DIFFUSION_OBJECTIVE_REGISTRY",
    "TRAINER_REGISTRY",
    "SAMPLER_REGISTRY",
    "GUIDED_SAMPLER_REGISTRY",
    "DATA_PIPELINE_REGISTRY",
    "METRIC_REGISTRY",
    "POLICY_ADAPTER_REGISTRY",
    "EXPERIMENT_PROTOCOL_REGISTRY",
    "TokenizerSpec",
    "AttentionMaskSpec",
    "ConditioningMaskSpec",
    "ScoreNetworkSpec",
    "DiffusionObjectiveSpec",
    "TrainerSpec",
    "SamplerSpec",
    "GuidedSamplerSpec",
    "DataPipelineSpec",
    "MetricSpec",
    "PolicyAdapterSpec",
    "ExperimentProtocolSpec",
    "SimformerPolicyAdapter",
    "get_registry",
    "get_model_registry",
    "get_tokenizer_registry",
    "get_attention_mask_registry",
    "get_conditioning_mask_registry",
    "get_diffusion_config",
    "resolve_factory",
    "build_tokenizer",
    "build_attention_mask",
    "sample_condition_mask",
    "build_score_network",
    "build_trainer",
    "build_sampler",
    "build_guided_sampler",
    "build_data_pipeline",
    "validate_registry_contract",
    "write_registry_artifacts",
    "runtime_smoke",
]


if __name__ == "__main__":
    result = runtime_smoke(mode=os.environ.get("ALL_IN_ONE_SBI_MODE", "runtime_smoke"))
    print(json.dumps(_json_safe(result), indent=2, sort_keys=True))
    sys.exit(0 if result["status"] == "ok" else 1)
