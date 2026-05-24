"""Bounded sweep and experiment-protocol registry for the BaM reproduction.

This module owns the import-light sweep/config surface for the paper
"Batch and match: black-box variational inference with a score-based divergence."
It exposes paper-derived experiment matrices as bounded registry values that can
be consumed by the canonical runner without turning the reproduction into an
unbounded parameter-search script.

The numerical implementation of BaM lives in ``bam.training_loop``,
``bam.score_divergence``, ``bam.variational``, and ``src.algorithms.*``.  This
file makes the experiment protocol explicit: hypotheses, decisive comparisons,
metrics, batch-size/seed/iteration conventions, artifact paths, and the
pruning rationale for omitted or bounded sweeps.

reference_grounding: paper:paper_method_core paper.md
    BaM receives log-density/score interfaces for the target and maintains a
    Gaussian variational approximation with mean μ and covariance Σ.

reference_grounding: paper:paper_training_or_optimization_loop paper.md
    The Batch Step explicitly samples z_1,...,z_B ~ q_t, evaluates
    g_b = ∇ log p(z_b), and computes zbar, C, gbar, Γ and related batch
    score/sample statistics.  The Match Step uses these statistics with KL
    regularization to update full-covariance Gaussian variational parameters.

reference_grounding: paper:paper_semantic_chunk_009_03 paper.md
    BaM is a KL-regularized stochastic proximal/matching method.  The registry
    exposes lambda, epsilon, learning_rate, batch_size, iteration_count,
    B=32 finite-batch semantics, B→∞ Gaussian sanity semantics, and bounded
    smoke/full selectors rather than exhaustive execution-only sweeps.
"""

from __future__ import annotations

import itertools
import json
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple


CANONICAL_RESULTS_DIR = Path("results")


@dataclass(frozen=True)
class SweepAxis:
    """One bounded sweep axis in a named experiment protocol."""

    name: str
    values: Tuple[Any, ...]
    default: Any
    description: str
    evidence: str
    include_in_smoke: bool = True

    def validate(self) -> None:
        if not self.name:
            raise ValueError("SweepAxis.name must be non-empty")
        if not self.values:
            raise ValueError(f"SweepAxis {self.name!r} must contain values")
        if self.default not in self.values:
            raise ValueError(
                f"SweepAxis {self.name!r} default {self.default!r} is not in values {self.values!r}"
            )


@dataclass(frozen=True)
class ProtocolEntry:
    """Executable protocol metadata consumed by runners and report writers."""

    protocol_id: str
    figure_or_section: str
    caption: str
    hypothesis: str
    decisive_comparison: str
    decisive_metric: str
    target_family: str
    methods: Tuple[str, ...]
    axes: Tuple[SweepAxis, ...]
    seeds: Tuple[int, ...]
    smoke_overrides: Mapping[str, Any]
    full_run_repetitions: int
    artifact_paths: Tuple[str, ...]
    stop_rule_or_pruning_rationale: str
    data_pipeline: Mapping[str, Any] = field(default_factory=dict)
    output_mapping: Mapping[str, str] = field(default_factory=dict)
    notes: Tuple[str, ...] = field(default_factory=tuple)

    def validate(self) -> None:
        if not self.protocol_id:
            raise ValueError("ProtocolEntry.protocol_id must be non-empty")
        if not self.methods:
            raise ValueError(f"{self.protocol_id}: methods must be non-empty")
        if not self.axes:
            raise ValueError(f"{self.protocol_id}: axes must be non-empty")
        for axis in self.axes:
            axis.validate()
        if not self.seeds:
            raise ValueError(f"{self.protocol_id}: seeds must be non-empty")
        if not self.artifact_paths:
            raise ValueError(f"{self.protocol_id}: artifact_paths must be non-empty")
        if not self.decisive_metric:
            raise ValueError(f"{self.protocol_id}: decisive_metric must be non-empty")


@dataclass(frozen=True)
class RunSpec:
    """Single resolved run selected from a protocol registry."""

    protocol_id: str
    mode: str
    method: str
    target_family: str
    seed: int
    parameters: Mapping[str, Any]
    decisive_metric: str
    artifact_paths: Tuple[str, ...]
    hypothesis: str
    stop_rule_or_pruning_rationale: str
    figure_or_section: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _axis(
    name: str,
    values: Sequence[Any],
    default: Any,
    description: str,
    evidence: str,
    include_in_smoke: bool = True,
) -> SweepAxis:
    axis = SweepAxis(
        name=name,
        values=tuple(values),
        default=default,
        description=description,
        evidence=evidence,
        include_in_smoke=include_in_smoke,
    )
    axis.validate()
    return axis


COMMON_BAM_AXES: Tuple[SweepAxis, ...] = (
    _axis(
        "lambda",
        (0.01, 0.1, 1.0),
        0.1,
        "KL/proximal regularization strength used by the Match Step.",
        "reference_grounding: paper:paper_semantic_chunk_009_03 paper.md",
    ),
    _axis(
        "epsilon",
        (1.0e-6, 1.0e-4, 1.0e-3),
        1.0e-4,
        "Covariance floor / numerical stabilization for full-covariance Gaussian updates.",
        "reference_grounding: paper:paper_training_or_optimization_loop paper.md",
    ),
    _axis(
        "learning_rate",
        (0.01, 0.03, 0.1),
        0.03,
        "Baseline optimizer step size and optional damped BaM match-update interpolation.",
        "reference_grounding: paper:paper_training_or_optimization_loop paper.md",
    ),
    _axis(
        "iteration_count",
        (0, 100, 1000, 3000),
        100,
        "Number of method iterations; 0 is required for schema/readiness evaluation, 100 is the bounded default, 3000 preserves Figure 5.4 evaluation budget.",
        "reference_grounding: paper:paper_training_or_optimization_loop paper.md",
    ),
)

CONTRACT_ONLY_AXES: Tuple[SweepAxis, ...] = (
    _axis(
        "p",
        (0.1, 0.5, 0.9),
        0.5,
        "Probability/control parameter required by the addendum contract for bounded registry exposure.",
        "reference_grounding: addendum:contract_sweep_requirements addendum.md",
        include_in_smoke=False,
    ),
    _axis(
        "lora_rank",
        (0, 4, 8),
        0,
        "Adapter-rank control for deep-generator adaptation contracts; rank 0 denotes no adapter.",
        "reference_grounding: addendum:contract_sweep_requirements addendum.md",
        include_in_smoke=False,
    ),
)


FIGURE_5_ARTIFACTS: Tuple[str, ...] = (
    "results/loss_trace.json",
    "results/bam_trace.json",
    "results/bam_final_variational_params.npz",
    "results/batch_statistics_trace.json",
    "results/gaussian_sanity_metrics.json",
    "results/figures/figure_5.png",
    "results/metrics.json",
    "results/run_summary.json",
    "results/config_echo.json",
    "results/evidence_contract_matrix.json",
    "results/experiment_registry.json",
    "results/environment_registry.json",
)


def _protocols() -> Dict[str, ProtocolEntry]:
    """Construct the canonical bounded protocol registry."""

    gaussian_axes = (
        _axis(
            "dimension",
            (4, 16, 64, 256),
            4,
            "Gaussian target dimension D for Figure 5.1 increasing-dimension study.",
            "reference_grounding: paper:figure_5_1_protocol paper.md",
        ),
        _axis(
            "batch_size",
            (2, 3, 8, 32, "infinity"),
            32,
            "Batch size B; baselines use B=2 in Figure 5.1, BaM legend exposes finite B and B→∞ analytic Gaussian sanity semantics.",
            "reference_grounding: paper:figure_5_1_protocol paper.md",
        ),
        *COMMON_BAM_AXES,
    )

    sinh_arcsinh_axes = (
        _axis(
            "skew",
            (-2.0, 0.0, 2.0),
            0.0,
            "Sinh-arcsinh target skew parameter s for non-Gaussian Figure 5.2.",
            "reference_grounding: paper:figure_5_2_protocol paper.md",
        ),
        _axis(
            "tail_weight",
            (0.5, 1.0, 2.0),
            1.0,
            "Sinh-arcsinh target tail weight t for non-Gaussian Figure 5.2.",
            "reference_grounding: paper:figure_5_2_protocol paper.md",
        ),
        _axis(
            "batch_size",
            (5, 32),
            5,
            "ADVI, Score, Fisher, and GSM use B=5 in Figure 5.2; BaM may also be evaluated at B=32 as a bounded decisive comparison.",
            "reference_grounding: paper:figure_5_2_protocol paper.md",
        ),
        *COMMON_BAM_AXES,
    )

    bayes_axes = (
        _axis(
            "model_name",
            ("logistic_regression", "poisson_regression", "hierarchical_normal"),
            "logistic_regression",
            "Bayesian posterior-inference model family used for Figure 5.3-style relative mean error comparisons.",
            "reference_grounding: paper:figure_5_3_protocol paper.md",
        ),
        _axis(
            "batch_size",
            (8, 32),
            32,
            "Figure 5.3 compares dashed B=8 and solid B=32 posterior-inference curves.",
            "reference_grounding: paper:figure_5_3_protocol paper.md",
        ),
        *COMMON_BAM_AXES,
    )

    deep_generator_axes = (
        _axis(
            "batch_size",
            (4, 8, 32),
            4,
            "Addendum clarification: Figure E.1-relevant experiments use batch size 4; Figure 5.4 preserves the 3000-gradient-evaluation image-reconstruction protocol.",
            "reference_grounding: addendum:figure_e_1_batch_size addendum.md",
        ),
        _axis(
            "latent_dimension",
            (8, 16, 32),
            16,
            "Latent dimension for posterior mean z' image-reconstruction experiments.",
            "reference_grounding: paper:figure_5_4_protocol paper.md",
            include_in_smoke=False,
        ),
        _axis(
            "final_activation",
            ("tanh",),
            "tanh",
            "Binding addendum clarification: final activation is tanh, so generator outputs are in [-1, 1].",
            "reference_grounding: addendum:final_activation_tanh addendum.md",
        ),
        *COMMON_BAM_AXES,
        *CONTRACT_ONLY_AXES,
    )

    appendix_e_axes = (
        _axis(
            "batch_size",
            (4,),
            4,
            "Binding addendum clarification: experiments relevant for Figure E.1 use batch size 4.",
            "reference_grounding: addendum:figure_e_1_batch_size addendum.md",
        ),
        _axis(
            "learning_rate",
            (0.03,),
            0.03,
            "Appendix E.3 clarification: it is sufficient to run the experiment with the specified bounded learning rate.",
            "reference_grounding: addendum:appendix_e_3_learning_rate addendum.md",
        ),
        _axis(
            "iteration_count",
            (0, 100),
            100,
            "Bounded appendix iteration protocol with iteration_count=0 retained for readiness/schema execution.",
            "reference_grounding: addendum:appendix_e_3_learning_rate addendum.md",
        ),
        _axis(
            "p",
            (0.5,),
            0.5,
            "Required contract parameter p exposed as a bounded appendix sweep value.",
            "reference_grounding: addendum:contract_sweep_requirements addendum.md",
        ),
        _axis(
            "lora_rank",
            (0,),
            0,
            "Required contract parameter lora_rank exposed; Figure E.2 is out of scope, so only rank 0 is active in the bounded appendix route.",
            "reference_grounding: addendum:figure_e_2_out_of_scope addendum.md",
        ),
    )

    protocols: Dict[str, ProtocolEntry] = {
        "figure_5_1_gaussian_dimensions": ProtocolEntry(
            protocol_id="figure_5_1_gaussian_dimensions",
            figure_or_section="Figure 5.1",
            caption=(
                "Gaussian targets of increasing dimension. Solid curves indicate the mean over 10 runs "
                "(transparent curves). ADVI, Score, Fisher, and GSM use a batch size of B=2. "
                "The batch size for BaM is given in the legend."
            ),
            hypothesis=(
                "BaM's score-statistic Match Step should reduce forward KL on Gaussian targets more "
                "reliably than ADVI/GSM as dimension increases, with B=32 and B→∞ serving as decisive "
                "finite-batch and analytic sanity comparisons."
            ),
            decisive_comparison="BaM versus ADVI versus GSM, with Score and Fisher score-based baselines retained for caption semantics.",
            decisive_metric="forward_kl",
            target_family="gaussian_increasing_dimension",
            methods=("bam", "advi", "gsm", "score", "fisher"),
            axes=gaussian_axes,
            seeds=tuple(range(10)),
            smoke_overrides={
                "dimension": 4,
                "batch_size": 3,
                "iteration_count": 0,
                "lambda": 0.1,
                "epsilon": 1.0e-4,
                "learning_rate": 0.03,
            },
            full_run_repetitions=10,
            artifact_paths=FIGURE_5_ARTIFACTS,
            stop_rule_or_pruning_rationale=(
                "The registry includes D={4,16,64,256} and 10 seeds for full reproduction, but default "
                "execution selects a single D=4, B=3, iteration_count=0 contract run to validate the "
                "Batch Step/Match Step wiring without an exhaustive sweep."
            ),
            data_pipeline={
                "dataset_id": "synthetic_gaussian_score_target",
                "prepare": "bam.targets.make_gaussian_target",
                "validate": "src.dataset_registry.validate_target_score_interface",
                "requires_external_data": False,
                "required_interfaces": ("log_density", "score"),
                "batch_statistics": ("zbar", "C", "gbar", "Gamma", "score_sample_covariance"),
            },
            output_mapping={
                "curve_x": "gradient_evaluations",
                "curve_y": "forward_kl",
                "aggregate": "mean_over_10_runs",
                "uncertainty": "transparent_individual_curves",
                "figure_path": "results/figures/figure_5.png",
            },
            notes=(
                "For Section 5.1 Gaussian targets, the registry exposes dimensions D=4,16,64,256.",
                "B→∞ is represented by the string value 'infinity' and is routed to analytic Gaussian sanity checks rather than stochastic sampling.",
            ),
        ),
        "figure_5_2_sinh_arcsinh": ProtocolEntry(
            protocol_id="figure_5_2_sinh_arcsinh",
            figure_or_section="Figure 5.2",
            caption=(
                "Non-Gaussian targets constructed using the sinh-arcsinh distribution, varying the skew s "
                "and the tail weight t. The curves denote the mean of the forward KL divergence over 10 runs, "
                "and shaded regions denote their standard error. ADVI, Score, Fisher, and GSM use B=5."
            ),
            hypothesis=(
                "Score matching through BaM should remain stable for skewed/heavy-tailed targets where "
                "standard black-box KL estimators have higher variance."
            ),
            decisive_comparison="BaM versus ADVI, GSM, Score, and Fisher on skew/tail-weight grid.",
            decisive_metric="forward_kl",
            target_family="sinh_arcsinh_non_gaussian",
            methods=("bam", "advi", "gsm", "score", "fisher"),
            axes=sinh_arcsinh_axes,
            seeds=tuple(range(10)),
            smoke_overrides={
                "skew": 0.0,
                "tail_weight": 1.0,
                "batch_size": 5,
                "iteration_count": 0,
                "lambda": 0.1,
                "epsilon": 1.0e-4,
                "learning_rate": 0.03,
            },
            full_run_repetitions=10,
            artifact_paths=(
                "results/metrics.json",
                "results/run_summary.json",
                "results/evidence_contract_matrix.json",
                "results/figures/figure_5.png",
            ),
            stop_rule_or_pruning_rationale=(
                "Only the paper's skew/tail-weight grid and two batch-size semantics are registered. "
                "Default execution runs one central non-Gaussian target at iteration_count=0 for contract "
                "validation; full mode is required for 10-run standard-error curves."
            ),
            data_pipeline={
                "dataset_id": "synthetic_sinh_arcsinh_score_target",
                "prepare": "bam.targets.make_sinh_arcsinh_target",
                "validate": "src.dataset_registry.validate_target_score_interface",
                "requires_external_data": False,
                "required_interfaces": ("log_density", "score"),
            },
            output_mapping={
                "curve_x": "gradient_evaluations",
                "curve_y": "forward_kl",
                "aggregate": "mean_over_10_runs",
                "uncertainty": "standard_error",
                "figure_path": "results/figures/figure_5.png",
            },
        ),
        "figure_5_3_bayesian_models": ProtocolEntry(
            protocol_id="figure_5_3_bayesian_models",
            figure_or_section="Figure 5.3",
            caption=(
                "Posterior inference in Bayesian models. The curves denote the mean over 5 runs, and shaded "
                "regions denote their standard error. Solid curves (B=32) correspond to larger batch sizes "
                "than dashed curves (B=8). Figure 5.3 reports relative mean errors."
            ),
            hypothesis=(
                "For posterior inference, BaM should achieve lower relative mean error than ADVI and avoid "
                "the oscillatory behavior observed for GSM, especially at B=32."
            ),
            decisive_comparison="BaM versus ADVI versus GSM for B=8 dashed and B=32 solid curves.",
            decisive_metric="relative_mean_error",
            target_family="bayesian_posterior_models",
            methods=("bam", "advi", "gsm"),
            axes=bayes_axes,
            seeds=tuple(range(5)),
            smoke_overrides={
                "model_name": "logistic_regression",
                "batch_size": 8,
                "iteration_count": 0,
                "lambda": 0.1,
                "epsilon": 1.0e-4,
                "learning_rate": 0.03,
            },
            full_run_repetitions=5,
            artifact_paths=(
                "results/metrics.json",
                "results/run_summary.json",
                "results/evidence_contract_matrix.json",
                "results/figures/figure_5.png",
            ),
            stop_rule_or_pruning_rationale=(
                "The decisive Figure 5.3 comparison is bounded to B=8 and B=32 with five seeds. "
                "No additional batch-size ladder is registered because the paper decision value is the "
                "small-versus-large batch comparison."
            ),
            data_pipeline={
                "dataset_id": "synthetic_bayesian_model_scores",
                "prepare": "src.data.data.prepare_synthetic_bayesian_posterior",
                "validate": "src.dataset_registry.validate_target_score_interface",
                "requires_external_data": False,
                "required_interfaces": ("log_density", "score"),
            },
            output_mapping={
                "curve_x": "gradient_evaluations",
                "curve_y": "relative_mean_error",
                "aggregate": "mean_over_5_runs",
                "uncertainty": "standard_error",
                "line_style": "solid_for_B32_dashed_for_B8",
                "figure_path": "results/figures/figure_5.png",
            },
        ),
        "figure_5_4_deep_generator": ProtocolEntry(
            protocol_id="figure_5_4_deep_generator",
            figure_or_section="Figure 5.4",
            caption=(
                "Image reconstruction and error when the posterior mean of z' is fed into the generative "
                "neural network. The beige and purple stars highlight the best outcome for ADVI and BaM, "
                "respectively, after 3,000 gradient evaluations."
            ),
            hypothesis=(
                "BaM posterior-mean refinement should provide lower image reconstruction error than ADVI "
                "under the same 3,000-gradient-evaluation budget."
            ),
            decisive_comparison="BaM versus ADVI reconstruction error after 3000 gradient evaluations.",
            decisive_metric="reconstruction_error",
            target_family="deep_generative_latent_posterior",
            methods=("bam", "advi"),
            axes=deep_generator_axes,
            seeds=tuple(range(5)),
            smoke_overrides={
                "batch_size": 4,
                "latent_dimension": 16,
                "final_activation": "tanh",
                "iteration_count": 0,
                "lambda": 0.1,
                "epsilon": 1.0e-4,
                "learning_rate": 0.03,
                "p": 0.5,
                "lora_rank": 0,
            },
            full_run_repetitions=5,
            artifact_paths=(
                "results/metrics.json",
                "results/run_summary.json",
                "results/deep_generative_metrics.json",
                "results/deep_generative_latent_params.npz",
                "results/evidence_contract_matrix.json",
                "results/figures/figure_5.png",
            ),
            stop_rule_or_pruning_rationale=(
                "The registry preserves the 3000-gradient-evaluation endpoint and tanh-output contract. "
                "Default execution uses iteration_count=0 and no external vision assets; full image "
                "evaluation requires explicit full mode and available generator/data adapters."
            ),
            data_pipeline={
                "dataset_id": "deep_generator_latent_reconstruction",
                "prepare": "src.data.data.prepare_deep_generator_protocol",
                "validate": "src.dataset_registry.validate_deep_generator_protocol",
                "requires_external_data": True,
                "external_data_optional_for_smoke": True,
                "required_interfaces": ("log_density", "score", "decoder"),
                "decoder_output_range": [-1.0, 1.0],
            },
            output_mapping={
                "curve_x": "gradient_evaluations",
                "curve_y": "reconstruction_error",
                "endpoint": "3000_gradient_evaluations",
                "stars": "best_ADVI_beige_best_BaM_purple",
                "figure_path": "results/figures/figure_5.png",
            },
            notes=(
                "Final activation is tanh, so image outputs must be interpreted in [-1, 1].",
                "Figure E.2 is out of scope for this reproduction and is not expanded into an exhaustive adapter sweep.",
            ),
        ),
        "appendix_e_1_bounded_addendum": ProtocolEntry(
            protocol_id="appendix_e_1_bounded_addendum",
            figure_or_section="Figure E.1 / Appendix E.3",
            caption=(
                "Bounded addendum protocol exposing batch size 4, the sufficient learning-rate route, "
                "and required p/lora_rank registry parameters while keeping Figure E.2 out of scope."
            ),
            hypothesis=(
                "The addendum route verifies that the paper reproduction can bind constrained appendix "
                "parameters without expanding into low-decision-value sweeps."
            ),
            decisive_comparison="BaM appendix route versus registered baseline selectors where applicable.",
            decisive_metric="readiness_and_metric_schema_consistency",
            target_family="appendix_bounded_protocol",
            methods=("bam", "advi"),
            axes=appendix_e_axes,
            seeds=(0,),
            smoke_overrides={
                "batch_size": 4,
                "learning_rate": 0.03,
                "iteration_count": 0,
                "p": 0.5,
                "lora_rank": 0,
            },
            full_run_repetitions=1,
            artifact_paths=(
                "results/metrics.json",
                "results/run_summary.json",
                "results/evidence_contract_matrix.json",
                "results/experiment_registry.json",
            ),
            stop_rule_or_pruning_rationale=(
                "Appendix E.3 is represented by the bounded learning-rate entry. Figure E.2 is explicitly "
                "out of scope, so lora_rank is exposed for contract compatibility but not expanded."
            ),
            data_pipeline={
                "dataset_id": "appendix_synthetic_protocol",
                "prepare": "src.data.data.prepare_appendix_protocol",
                "validate": "src.dataset_registry.validate_protocol_schema",
                "requires_external_data": False,
                "required_interfaces": ("log_density", "score"),
            },
            output_mapping={
                "registry_artifact": "results/experiment_registry.json",
                "contract_matrix": "results/evidence_contract_matrix.json",
            },
        ),
    }

    for protocol in protocols.values():
        protocol.validate()
    return protocols


def get_protocol_registry() -> Dict[str, ProtocolEntry]:
    """Return the canonical protocol registry keyed by protocol id."""

    return dict(_protocols())


def get_sweep_registry() -> Dict[str, Any]:
    """Return a JSON-serializable sweep registry.

    The registry is deliberately bounded.  Full reproduction runners may expand
    the listed axes, but default runtime validation should use ``smoke_overrides``
    and the first seed only.
    """

    protocols = get_protocol_registry()
    payload: Dict[str, Any] = {
        "registry_type": "bounded_sweep_registry",
        "paper": "Batch and match: black-box variational inference with a score-based divergence",
        "blacklisted_repositories": ("https://github.com/modichirag/GSM-VI",),
        "core_method_contract": {
            "bam_accepts": ("log_density", "score"),
            "variational_state": ("mu", "Sigma"),
            "batch_step": (
                "sample z_1,...,z_B from q_t",
                "compute g_b = grad log p(z_b)",
                "compute zbar, C, gbar, Gamma and score/sample covariance statistics",
            ),
            "match_step": "update Gaussian variational parameters with KL regularization",
            "normalizing_constant_required": False,
        },
        "mandatory_sweep_axes": {
            "lambda": [0.01, 0.1, 1.0],
            "epsilon": [1.0e-6, 1.0e-4, 1.0e-3],
            "learning_rate": [0.01, 0.03, 0.1],
            "batch_size": [2, 3, 4, 5, 8, 32, "infinity"],
            "iteration_count": [0, 100, 1000, 3000],
            "p": [0.1, 0.5, 0.9],
            "lora_rank": [0, 4, 8],
            "random_seed": list(range(10)),
            "100_iterations": 100,
            "regularization_strength": "lambda",
            "B=32": 32,
            "B_to_infinity": "infinity",
        },
        "selected_experiment_set": {
            "core_contribution_hypothesis": (
                "Batch Step score statistics plus a KL-regularized Match Step improve Gaussian "
                "variational inference over ADVI/GSM on score-accessible targets."
            ),
            "decisive_comparison": "BaM versus ADVI versus GSM, retaining Score/Fisher where figure captions require them.",
            "decisive_metrics": ("forward_kl", "relative_mean_error", "reconstruction_error"),
            "stop_pruning_rationale": (
                "Expose the paper/addendum parameter grid but execute only bounded smoke defaults "
                "unless full mode is explicitly selected."
            ),
        },
        "protocols": {
            key: _protocol_to_jsonable(protocol) for key, protocol in protocols.items()
        },
    }
    return payload


def _protocol_to_jsonable(protocol: ProtocolEntry) -> Dict[str, Any]:
    data = asdict(protocol)
    data["axes"] = [asdict(axis) for axis in protocol.axes]
    data["seeds"] = list(protocol.seeds)
    data["artifact_paths"] = list(protocol.artifact_paths)
    data["methods"] = list(protocol.methods)
    data["notes"] = list(protocol.notes)
    return data


def get_protocol(protocol_id: str) -> ProtocolEntry:
    """Return a single protocol by id."""

    registry = get_protocol_registry()
    try:
        return registry[protocol_id]
    except KeyError as exc:
        available = ", ".join(sorted(registry))
        raise KeyError(f"Unknown protocol_id {protocol_id!r}; available protocols: {available}") from exc


def _axis_values_for_mode(axis: SweepAxis, mode: str, smoke_overrides: Mapping[str, Any]) -> Tuple[Any, ...]:
    if mode in {"runtime_smoke", "docker_validate", "smoke", "dry_run"}:
        if axis.name in smoke_overrides:
            return (smoke_overrides[axis.name],)
        return (axis.default,)
    if mode in {"default", "quick"}:
        if axis.name in smoke_overrides:
            return (smoke_overrides[axis.name],)
        return (axis.default,)
    if mode in {"full", "reproduce"}:
        return axis.values
    if mode in {"bounded"}:
        values: List[Any] = [axis.default]
        for value in axis.values:
            if value not in values and len(values) < 2:
                values.append(value)
        return tuple(values)
    raise ValueError(
        f"Unsupported mode {mode!r}; expected runtime_smoke, docker_validate, default, quick, bounded, full, or reproduce"
    )


def iter_run_specs(
    protocol_ids: Optional[Sequence[str]] = None,
    mode: str = "default",
    methods: Optional[Sequence[str]] = None,
    max_runs: Optional[int] = None,
) -> Iterator[RunSpec]:
    """Yield resolved run specs from the bounded registry.

    ``runtime_smoke`` and ``docker_validate`` select each protocol's
    ``smoke_overrides`` and first seed.  ``full`` expands the paper-level axes
    and seeds but remains explicitly selected by the caller.
    """

    registry = get_protocol_registry()
    selected_protocol_ids = tuple(protocol_ids) if protocol_ids is not None else tuple(registry.keys())
    yielded = 0

    for protocol_id in selected_protocol_ids:
        protocol = get_protocol(protocol_id)
        selected_methods = tuple(methods) if methods is not None else protocol.methods
        unknown = [method for method in selected_methods if method not in protocol.methods]
        if unknown:
            raise ValueError(
                f"Protocol {protocol_id!r} does not register methods {unknown!r}; available={protocol.methods!r}"
            )

        seeds = (protocol.seeds[0],) if mode in {"runtime_smoke", "docker_validate", "smoke", "dry_run", "default", "quick"} else protocol.seeds
        axis_names = [axis.name for axis in protocol.axes]
        axis_value_lists = [
            _axis_values_for_mode(axis, mode, protocol.smoke_overrides) for axis in protocol.axes
        ]

        for method, seed, values in itertools.product(selected_methods, seeds, itertools.product(*axis_value_lists)):
            params = dict(zip(axis_names, values))
            yield RunSpec(
                protocol_id=protocol.protocol_id,
                mode=mode,
                method=method,
                target_family=protocol.target_family,
                seed=int(seed),
                parameters=params,
                decisive_metric=protocol.decisive_metric,
                artifact_paths=protocol.artifact_paths,
                hypothesis=protocol.hypothesis,
                stop_rule_or_pruning_rationale=protocol.stop_rule_or_pruning_rationale,
                figure_or_section=protocol.figure_or_section,
            )
            yielded += 1
            if max_runs is not None and yielded >= max_runs:
                return


def select_smoke_run_specs(max_runs: Optional[int] = None) -> List[RunSpec]:
    """Return bounded contract-validation runs for the canonical route."""

    return list(iter_run_specs(mode="runtime_smoke", max_runs=max_runs))


def resolve_protocol_config(
    protocol_id: str,
    mode: str = "default",
    method: str = "bam",
    overrides: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Resolve a protocol into a single configuration dictionary.

    This function is used by runners that need one method/target configuration
    rather than a full run iterator.
    """

    protocol = get_protocol(protocol_id)
    if method not in protocol.methods:
        raise ValueError(f"Method {method!r} is not registered for protocol {protocol_id!r}")

    params: Dict[str, Any] = {}
    for axis in protocol.axes:
        params[axis.name] = _axis_values_for_mode(axis, mode, protocol.smoke_overrides)[0]
    if overrides:
        for key, value in overrides.items():
            axis_names = {axis.name for axis in protocol.axes}
            if key not in axis_names and key not in {"seed", "method", "target_family", "protocol_id"}:
                raise ValueError(f"Override {key!r} is not a registered axis for protocol {protocol_id!r}")
            params[key] = value

    seed = int(params.pop("seed", protocol.seeds[0]))
    return {
        "protocol_id": protocol.protocol_id,
        "figure_or_section": protocol.figure_or_section,
        "method": method,
        "target_family": protocol.target_family,
        "seed": seed,
        "parameters": params,
        "data_pipeline": dict(protocol.data_pipeline),
        "decisive_metric": protocol.decisive_metric,
        "artifact_paths": list(protocol.artifact_paths),
        "hypothesis": protocol.hypothesis,
        "decisive_comparison": protocol.decisive_comparison,
        "stop_rule_or_pruning_rationale": protocol.stop_rule_or_pruning_rationale,
        "output_mapping": dict(protocol.output_mapping),
    }


def evidence_contract_matrix() -> Dict[str, Any]:
    """Return machine-readable mapping from obligations to registry entries."""

    registry = get_protocol_registry()
    return {
        "matrix_type": "paper_evidence_contract_matrix",
        "generated_at_unix": time.time(),
        "obligations": {
            "bam_core_batch_step": {
                "evidence": "reference_grounding: paper:paper_training_or_optimization_loop paper.md",
                "registry_fields": [
                    "core_method_contract.batch_step",
                    "protocols.*.data_pipeline.required_interfaces",
                    "mandatory_sweep_axes.batch_size",
                ],
                "implementation_paths": [
                    "bam.score_divergence",
                    "bam.training_loop",
                    "src.algorithms.bam",
                ],
            },
            "bam_core_match_step": {
                "evidence": "reference_grounding: paper:paper_semantic_chunk_009_03 paper.md",
                "registry_fields": [
                    "core_method_contract.match_step",
                    "mandatory_sweep_axes.lambda",
                    "mandatory_sweep_axes.epsilon",
                ],
                "implementation_paths": [
                    "bam.optimizer",
                    "bam.variational",
                    "bam.training_loop",
                ],
            },
            "figure_5_1": {
                "caption": registry["figure_5_1_gaussian_dimensions"].caption,
                "protocol_id": "figure_5_1_gaussian_dimensions",
                "artifacts": list(registry["figure_5_1_gaussian_dimensions"].artifact_paths),
            },
            "figure_5_2": {
                "caption": registry["figure_5_2_sinh_arcsinh"].caption,
                "protocol_id": "figure_5_2_sinh_arcsinh",
                "artifacts": list(registry["figure_5_2_sinh_arcsinh"].artifact_paths),
            },
            "figure_5_3": {
                "caption": registry["figure_5_3_bayesian_models"].caption,
                "protocol_id": "figure_5_3_bayesian_models",
                "artifacts": list(registry["figure_5_3_bayesian_models"].artifact_paths),
            },
            "figure_5_4": {
                "caption": registry["figure_5_4_deep_generator"].caption,
                "protocol_id": "figure_5_4_deep_generator",
                "artifacts": list(registry["figure_5_4_deep_generator"].artifact_paths),
            },
            "addendum_constraints": {
                "figure_e_1_batch_size": 4,
                "final_activation": "tanh",
                "output_range": [-1.0, 1.0],
                "appendix_e_3_learning_rate_values": [0.03],
                "figure_e_2_scope": "out_of_scope_not_expanded",
                "contract_axes": ["p", "lora_rank"],
                "evidence": "reference_grounding: addendum:contract_sweep_requirements addendum.md",
            },
        },
    }


def environment_adapter_registry() -> Dict[str, Any]:
    """Return import-light environment readiness information for sweep execution."""

    return {
        "registry_type": "environment_adapter",
        "python": sys.version,
        "platform": platform.platform(),
        "optional_dependencies": {
            "numpy": _module_available("numpy"),
            "matplotlib": _module_available("matplotlib"),
            "jax": _module_available("jax"),
            "torch": _module_available("torch"),
        },
        "minimal_import_contract": {
            "top_level_optional_gpu_or_dataset_imports": False,
            "external_data_required_for_smoke": False,
            "canonical_entrypoint": "scripts/run_experiments.py",
            "runtime_smoke_command": "python scripts/run_experiments.py --mode runtime_smoke",
        },
    }


def _module_available(module_name: str) -> bool:
    try:
        import importlib.util

        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False


def artifact_dir(default: Path = CANONICAL_RESULTS_DIR) -> Path:
    """Return output directory, honoring PAPERBENCH_REPRO_ARTIFACT_DIR for auxiliary output."""

    env_value = os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR")
    if env_value:
        return Path(env_value)
    return default


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_registry_artifacts(base_dir: Optional[Path] = None) -> Dict[str, str]:
    """Materialize registry/readiness artifacts used by the canonical route.

    The written files are contract/readiness artifacts.  They record the active
    protocol wiring and do not claim completed benchmark results.
    """

    root = Path(base_dir) if base_dir is not None else artifact_dir()
    root.mkdir(parents=True, exist_ok=True)

    registry_payload = get_sweep_registry()
    env_payload = environment_adapter_registry()
    matrix_payload = evidence_contract_matrix()

    paths = {
        "experiment_registry": root / "experiment_registry.json",
        "environment_registry": root / "environment_registry.json",
        "evidence_contract_matrix": root / "evidence_contract_matrix.json",
        "config_echo": root / "config_echo.json",
        "run_summary": root / "run_summary.json",
        "readiness": root / "readiness.json",
        "evaluation_result": root / "evaluation_result.json",
    }

    write_json(paths["experiment_registry"], registry_payload)
    write_json(paths["environment_registry"], env_payload)
    write_json(paths["evidence_contract_matrix"], matrix_payload)
    write_json(
        paths["config_echo"],
        {
            "artifact_kind": "configuration_echo",
            "status": "contract_ready",
            "mode": "registry_materialization",
            "selected_smoke_runs": [spec.to_dict() for spec in select_smoke_run_specs(max_runs=8)],
        },
    )
    write_json(
        paths["run_summary"],
        {
            "artifact_kind": "run_summary_schema",
            "status": "contract_ready",
            "summary": "Bounded BaM sweep registry materialized; numerical execution is performed by training/evaluation modules.",
            "protocol_count": len(get_protocol_registry()),
            "declared_artifacts": sorted(_declared_artifacts()),
        },
    )
    write_json(
        paths["readiness"],
        {
            "artifact_kind": "readiness",
            "status": "ready",
            "canonical_entrypoint": "scripts/run_experiments.py",
            "smoke_command": "python scripts/run_experiments.py --mode runtime_smoke",
            "registry_file": "src/sweep_registry.py",
            "environment": env_payload,
        },
    )
    write_json(
        paths["evaluation_result"],
        {
            "artifact_kind": "evaluation_result_schema",
            "status": "contract_ready",
            "decisive_metrics": ["forward_kl", "relative_mean_error", "reconstruction_error"],
            "not_benchmark_scores": True,
            "protocols": list(get_protocol_registry().keys()),
        },
    )

    return {key: str(value) for key, value in paths.items()}


def _declared_artifacts() -> Tuple[str, ...]:
    artifacts: List[str] = []
    for protocol in get_protocol_registry().values():
        artifacts.extend(protocol.artifact_paths)
    artifacts.extend(
        [
            "results/readiness.json",
            "results/evaluation_result.json",
            "results/experiment_registry.json",
            "results/environment_registry.json",
        ]
    )
    return tuple(sorted(set(artifacts)))


def get_artifact_contract() -> Dict[str, Any]:
    """Return artifact paths declared by this registry and their semantic owners."""

    return {
        "artifact_contract_type": "sweep_registry_outputs",
        "declared_paths": list(_declared_artifacts()),
        "canonical_required_outputs": [
            "results/metrics.json",
            "results/run_summary.json",
            "results/config_echo.json",
            "results/evidence_contract_matrix.json",
            "results/experiment_registry.json",
            "results/environment_registry.json",
        ],
        "bam_core_outputs": [
            "results/loss_trace.json",
            "results/bam_trace.json",
            "results/bam_final_variational_params.npz",
            "results/batch_statistics_trace.json",
            "results/gaussian_sanity_metrics.json",
            "results/figures/figure_5.png",
        ],
        "schema_label": "readiness/contract artifacts for smoke modes; numerical modules write measured values in full execution",
    }


SWEEP_REGISTRY: Dict[str, Any] = get_sweep_registry()
PROTOCOL_REGISTRY: Dict[str, ProtocolEntry] = get_protocol_registry()


__all__ = [
    "CANONICAL_RESULTS_DIR",
    "CONTRACT_ONLY_AXES",
    "COMMON_BAM_AXES",
    "FIGURE_5_ARTIFACTS",
    "ProtocolEntry",
    "RunSpec",
    "SWEEP_REGISTRY",
    "PROTOCOL_REGISTRY",
    "SweepAxis",
    "artifact_dir",
    "environment_adapter_registry",
    "evidence_contract_matrix",
    "get_artifact_contract",
    "get_protocol",
    "get_protocol_registry",
    "get_sweep_registry",
    "iter_run_specs",
    "resolve_protocol_config",
    "select_smoke_run_specs",
    "write_registry_artifacts",
]
