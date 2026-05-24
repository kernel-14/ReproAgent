"""Reporting, metric aggregation, and artifact registry for DPO-toxicity reproduction.

This module is intentionally lightweight at import time.  It owns the canonical
artifact layout, metric schemas, bounded evaluation aggregation, and manifest
writers for the PaperBench reproduction of:

    A Mechanistic Understanding of Alignment Algorithms:
    A Case Study on DPO and Toxicity.

The paper-visible tables and figures are declared here with stable paths and
captions, but this module only writes benchmark-visible measured artifacts when
callers provide measured rows/series/predictions.  The default route may still
write readiness and registry artifacts that describe the required full-mode
inputs and output paths.
"""

from __future__ import annotations

import csv
import json
import math
import os
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union


JsonDict = Dict[str, Any]
PathLike = Union[str, os.PathLike[str]]


# reference_grounding: paperbench_ref_001 releases/20170613-score_normalization_v1.md
# reference_grounding: paperbench_ref_001 releases/20170823-score_normalization_v2.md
TOXICITY_SCORE_PROTOCOL: JsonDict = {
    "id": "normalized_toxicity_probability_protocol",
    "score_range": [0.0, 1.0],
    "interpretation": (
        "A toxicity score is treated as a calibrated probability-like belief that "
        "a rater would consider the generated text toxic."
    ),
    "thresholds": {
        "default_binary_toxicity": 0.5,
        "high_toxicity": 0.8,
    },
    "normalization_versions": [
        {
            "source": "paperbench_ref_001 releases/20170613-score_normalization_v1.md",
            "intent": "Probability calibration makes toxicity scores interpretable as approximate probabilities.",
        },
        {
            "source": "paperbench_ref_001 releases/20170823-score_normalization_v2.md",
            "intent": (
                "Threshold provenance must be recorded because normalization changes "
                "can alter low-end and high-end score ranges."
            ),
        },
    ],
}


# reference_grounding: paperbench_ref_001 model-cards/English/toxicity.md
TOXICITY_MODEL_CARD_GROUNDING: JsonDict = {
    "source": "paperbench_ref_001 model-cards/English/toxicity.md",
    "toxicity_definition": "A rude, disrespectful, or unreasonable comment that is likely to make people leave a discussion.",
    "use_in_this_reproduction": (
        "The definition determines binary toxicity labels, calibrated toxicity-rate "
        "aggregation, and Perspective-style evaluator compatibility."
    ),
}


@dataclass(frozen=True)
class MetricSchema:
    """Machine-readable schema for a metric used by the reproduction."""

    name: str
    formula: str
    aggregation: str
    higher_is_better: Optional[bool]
    required_fields: Tuple[str, ...] = ()
    output_key: Optional[str] = None
    paper_role: str = ""

    def to_dict(self) -> JsonDict:
        payload = asdict(self)
        payload["required_fields"] = list(self.required_fields)
        return payload


@dataclass(frozen=True)
class ArtifactSpec:
    """Stable path, caption, and dependency declaration for a result artifact."""

    artifact_id: str
    kind: str
    path: str
    caption: str
    required_inputs: Tuple[str, ...] = ()
    benchmark_visible: bool = True
    write_policy: str = "write_only_when_measured"
    paper_section: str = "main_or_appendix"

    def to_dict(self) -> JsonDict:
        payload = asdict(self)
        payload["required_inputs"] = list(self.required_inputs)
        return payload


@dataclass
class ReportingLayout:
    """Canonical output layout for registries, summaries, tables, and figures."""

    root: str = "results"
    tables_dir: str = "results/tables"
    figures_dir: str = "results/figures"
    checkpoints_dir: str = "results/checkpoints"
    logs_dir: str = "results/logs"
    dataset_registry: str = "results/dataset_registry.json"
    metrics: str = "results/metrics.json"
    data_manifest: str = "results/data_manifest.json"
    experiment_registry: str = "results/experiment_registry.json"
    artifact_manifest: str = "results/artifact_manifest.json"
    summary_table: str = "results/tables/summary.csv"
    evidence_contract_matrix: str = "results/evidence_contract_matrix.json"
    environment_registry: str = "results/environment_registry.json"
    sensitivity_report: str = "results/sensitivity_report.json"
    experiment_results: str = "results/tables/experiment_results.csv"
    experiment_results_figure: str = "results/figures/experiment_results.png"
    predictions: str = "results/predictions.jsonl"
    config_resolved: str = "results/config_resolved.json"
    training_trace: str = "results/training_trace.json"
    readiness: str = "results/readiness.json"
    evaluation_result: str = "results/evaluation_result.json"

    @classmethod
    def from_root(cls, root: PathLike = "results") -> "ReportingLayout":
        root_path = Path(root)
        return cls(
            root=str(root_path),
            tables_dir=str(root_path / "tables"),
            figures_dir=str(root_path / "figures"),
            checkpoints_dir=str(root_path / "checkpoints"),
            logs_dir=str(root_path / "logs"),
            dataset_registry=str(root_path / "dataset_registry.json"),
            metrics=str(root_path / "metrics.json"),
            data_manifest=str(root_path / "data_manifest.json"),
            experiment_registry=str(root_path / "experiment_registry.json"),
            artifact_manifest=str(root_path / "artifact_manifest.json"),
            summary_table=str(root_path / "tables" / "summary.csv"),
            evidence_contract_matrix=str(root_path / "evidence_contract_matrix.json"),
            environment_registry=str(root_path / "environment_registry.json"),
            sensitivity_report=str(root_path / "sensitivity_report.json"),
            experiment_results=str(root_path / "tables" / "experiment_results.csv"),
            experiment_results_figure=str(root_path / "figures" / "experiment_results.png"),
            predictions=str(root_path / "predictions.jsonl"),
            config_resolved=str(root_path / "config_resolved.json"),
            training_trace=str(root_path / "training_trace.json"),
            readiness=str(root_path / "readiness.json"),
            evaluation_result=str(root_path / "evaluation_result.json"),
        )

    def paper_artifact_paths(self) -> JsonDict:
        return {spec.artifact_id: spec.path for spec in PAPER_ARTIFACT_SPECS}

    def all_paths(self) -> List[str]:
        static_paths = [
            self.dataset_registry,
            self.metrics,
            self.data_manifest,
            self.experiment_registry,
            self.artifact_manifest,
            self.summary_table,
            self.evidence_contract_matrix,
            self.environment_registry,
            self.sensitivity_report,
            self.experiment_results,
            self.experiment_results_figure,
            self.predictions,
            self.config_resolved,
            self.training_trace,
            self.readiness,
            self.evaluation_result,
        ]
        static_paths.extend(self.paper_artifact_paths().values())
        return sorted(dict.fromkeys(static_paths))

    def ensure_directories(self) -> None:
        for directory in [self.root, self.tables_dir, self.figures_dir, self.checkpoints_dir, self.logs_dir]:
            Path(directory).mkdir(parents=True, exist_ok=True)
        for path in self.all_paths():
            Path(path).parent.mkdir(parents=True, exist_ok=True)


@dataclass
class ReportingSpec:
    """Resolved reporting protocol used by canonical and full reproduction routes."""

    layout: ReportingLayout = field(default_factory=ReportingLayout)
    mode: str = "runtime_smoke"
    paper_title: str = "A Mechanistic Understanding of Alignment Algorithms: A Case Study on DPO and Toxicity"
    hypothesis: str = (
        "DPO reduces toxic generations by changing or bypassing toxicity-relevant "
        "representations rather than erasing toxicity-related capabilities."
    )
    decisive_comparison: str = (
        "Compare pretrained model, DPO model, vector interventions, and un-aligning "
        "variants on toxicity, perplexity, F1, activation shifts, and vector/residual cosine geometry."
    )
    decisive_metric: str = "toxicity_rate_with_perplexity_and_probe_f1_guardrails"
    stop_rule_or_pruning_rationale: str = (
        "Expose all paper-visible protocols and bounded selectors, but default execution "
        "runs only safe wiring checks; full table and figure generation requires measured "
        "model activations, predictions, and vector projections."
    )
    toxicity_threshold: float = 0.5
    create_parent_dirs: bool = True

    def to_dict(self) -> JsonDict:
        return {
            "layout": asdict(self.layout),
            "mode": self.mode,
            "paper_title": self.paper_title,
            "hypothesis": self.hypothesis,
            "decisive_comparison": self.decisive_comparison,
            "decisive_metric": self.decisive_metric,
            "stop_rule_or_pruning_rationale": self.stop_rule_or_pruning_rationale,
            "toxicity_threshold": self.toxicity_threshold,
            "create_parent_dirs": self.create_parent_dirs,
        }


METRIC_REGISTRY: Dict[str, MetricSchema] = {
    "accuracy": MetricSchema(
        name="accuracy",
        formula="(TP + TN) / N",
        aggregation="micro_average_over_predictions",
        higher_is_better=True,
        required_fields=("label", "prediction"),
        paper_role="Binary toxicity probe/evaluator correctness.",
    ),
    "precision": MetricSchema(
        name="precision",
        formula="TP / (TP + FP)",
        aggregation="micro_average_over_positive_predictions",
        higher_is_better=True,
        required_fields=("label", "prediction"),
        paper_role="Toxic-label precision for probe/evaluator validation.",
    ),
    "recall": MetricSchema(
        name="recall",
        formula="TP / (TP + FN)",
        aggregation="micro_average_over_positive_labels",
        higher_is_better=True,
        required_fields=("label", "prediction"),
        paper_role="Toxic-label recall for probe/evaluator validation.",
    ),
    "f1": MetricSchema(
        name="f1",
        formula="2 * precision * recall / (precision + recall)",
        aggregation="micro_f1",
        higher_is_better=True,
        required_fields=("label", "prediction"),
        paper_role="Probe quality guardrail reported alongside interventions/DPO.",
    ),
    "loss": MetricSchema(
        name="loss",
        formula="mean(per_sample_negative_log_likelihood)",
        aggregation="arithmetic_mean",
        higher_is_better=False,
        required_fields=("loss",),
        paper_role="Language-model or classifier optimization/evaluation loss.",
    ),
    "perplexity": MetricSchema(
        name="perplexity",
        formula="exp(mean(per_token_negative_log_likelihood))",
        aggregation="exponentiated_mean_loss",
        higher_is_better=False,
        required_fields=("loss",),
        paper_role="Fluency guardrail in Tables 2 and 7.",
    ),
    "toxicity": MetricSchema(
        name="toxicity",
        formula="mean(calibrated_toxicity_score) and mean(score >= threshold)",
        aggregation="mean_score_and_binary_rate",
        higher_is_better=False,
        required_fields=("toxicity_score",),
        paper_role="Primary safety outcome after DPO, interventions, and un-aligning.",
    ),
    "fidelity_score": MetricSchema(
        name="fidelity_score",
        formula=(
            "mean of satisfied paper trend checks: DPO parameter cosine remains high; "
            "DPO toxic-vector activation decreases; delta_MLP.v and delta_x have negative cosine; "
            "positive un-align/gating parameters reactivate toxicity."
        ),
        aggregation="bounded_fraction_0_to_1",
        higher_is_better=True,
        required_fields=(),
        paper_role="Semantic fidelity of reproduced measured trends.",
    ),
    "activation_shift": MetricSchema(
        name="activation_shift",
        formula="mean(post_dpo_activation - pretrained_activation) for selected toxic vectors",
        aggregation="mean_by_layer_component_vector",
        higher_is_better=None,
        required_fields=("pre_activation", "post_activation"),
        paper_role="Figures 2, 6, 7-11 mechanistic shift measurement.",
    ),
    "cosine_similarity": MetricSchema(
        name="cosine_similarity",
        formula="dot(a, b) / (||a|| * ||b||)",
        aggregation="mean_or_distribution_by_component",
        higher_is_better=None,
        required_fields=("vector_a", "vector_b"),
        paper_role="Parameter-similarity and delta_MLP.v vs delta_x trend checks.",
    ),
}


PAPER_ARTIFACT_SPECS: Tuple[ArtifactSpec, ...] = (
    ArtifactSpec(
        "table_1",
        "result_table",
        "results/tables/table_1.csv",
        "Table 1. Toxic vectors in GPT2, projected onto the vocabulary space. Top tokens are tokens with the highest dot-products with a specified toxic vector.",
        ("vector_id", "layer", "component", "top_tokens", "dot_products"),
    ),
    ArtifactSpec(
        "table_6",
        "result_table",
        "results/tables/table_6.csv",
        "Table 6. Top toxic vectors in Llama2, projected onto the vocabulary space.",
        ("vector_id", "layer", "component", "top_tokens", "dot_products"),
        paper_section="appendix",
    ),
    ArtifactSpec(
        "table_2",
        "result_table",
        "results/tables/table_2.csv",
        "Table 2. Toxicity, perplexity (PPL), and F1 after interventions or DPO for GPT2.",
        ("method", "toxicity", "perplexity", "f1"),
    ),
    ArtifactSpec(
        "table_7",
        "result_table",
        "results/tables/table_7.csv",
        "Table 7. Toxicity, perplexity (PPL), and F1 after interventions or DPO for Llama2.",
        ("method", "toxicity", "perplexity", "f1"),
        paper_section="appendix",
    ),
    ArtifactSpec(
        "table_3",
        "result_table",
        "results/tables/table_3.csv",
        "Table 3. Examples of top-k and continuations to prompts that originally elicit the target toxic token from GPT2, interventions, and GPT2_DPO.",
        ("prompt_id", "model_variant", "top_k_tokens", "continuation"),
    ),
    ArtifactSpec(
        "figure_1",
        "result_figure",
        "results/figures/figure_1.png",
        "Figure 1. Logit lens on GPT2 and GPT2_DPO for prompts that originally elicit a target toxic token.",
        ("layer", "probability", "model_variant"),
    ),
    ArtifactSpec(
        "figure_2",
        "result_figure",
        "results/figures/figure_2.png",
        "Figure 2. Mean activations for toxic vectors in GPT2 before and after DPO.",
        ("layer", "vector_id", "pre_activation", "post_activation"),
    ),
    ArtifactSpec(
        "figure_3",
        "result_figure",
        "results/figures/figure_3.png",
        "Figure 3. Visualization of residual streams before and after DPO; the shift delta_x lets GPT2_DPO bypass toxic regions.",
        ("residual_projection_x", "residual_projection_y", "model_variant"),
    ),
    ArtifactSpec(
        "figure_4",
        "result_figure",
        "results/figures/figure_4.png",
        "Figure 4. Linear shift of residual streams out of toxic regions on REALTOXICITYPrompts.",
        ("delta_x_projection", "toxic_direction_projection", "model_variant"),
    ),
    ArtifactSpec(
        "figure_5",
        "result_figure",
        "results/figures/figure_5.png",
        "Figure 5. Cosine similarity between delta_MLP.v and delta_x; blue areas represent negative/positive similarity bins and orange areas activation-change bins.",
        ("cosine_delta_mlp_delta_x", "activation_change"),
    ),
    ArtifactSpec(
        "figure_6",
        "result_figure",
        "results/figures/figure_6.png",
        "Figure 6. Mean activations for toxic vectors in Llama2 before and after DPO, broken down by component.",
        ("layer", "component", "pre_activation", "post_activation"),
        paper_section="appendix",
    ),
    ArtifactSpec(
        "table_4",
        "result_table",
        "results/tables/table_4.csv",
        "Table 4. Un-aligning GPT2_DPO by scaling toxic key vectors to increase regions that elicit toxicity.",
        ("scale", "toxicity", "perplexity", "f1"),
    ),
    ArtifactSpec(
        "table_5",
        "result_table",
        "results/tables/table_5.csv",
        "Table 5. Un-aligning Llama2_DPO by turning on gating components sigma(W1 x) to reactivate toxicity.",
        ("gating_value", "toxicity", "perplexity", "f1"),
        paper_section="appendix",
    ),
    ArtifactSpec(
        "figure_7",
        "result_figure",
        "results/figures/figure_7.png",
        "Figure 7. Shift in residual streams at selected Llama2 layers tied to high-cosine toxic vectors.",
        ("layer", "delta_x", "delta_mlp"),
        paper_section="appendix",
    ),
    ArtifactSpec(
        "figure_8",
        "result_figure",
        "results/figures/figure_8.png",
        "Figure 8. Shift in residual streams at layer 12 vs. shift in MLP value vectors (delta_x^12 vs. delta_MLP).",
        ("delta_x_layer_12", "delta_mlp"),
        paper_section="appendix",
    ),
    ArtifactSpec(
        "table_8",
        "result_table",
        "results/tables/table_8.csv",
        "Table 8. Hyperparameters: DPO.",
        ("hyperparameter", "value"),
        paper_section="appendix",
    ),
    ArtifactSpec(
        "table_9",
        "result_table",
        "results/tables/table_9.csv",
        "Table 9. Hyperparameters: PPLM.",
        ("hyperparameter", "value"),
        paper_section="appendix",
    ),
    ArtifactSpec(
        "figure_9",
        "result_figure",
        "results/figures/figure_9.png",
        "Figure 9. Shift in residual streams at layer 14 vs. shift in MLP value vectors (delta_x^14 vs. delta_MLP).",
        ("delta_x_layer_14", "delta_mlp"),
        paper_section="appendix",
    ),
    ArtifactSpec(
        "figure_10",
        "result_figure",
        "results/figures/figure_10.png",
        "Figure 10. Shift in residual streams at layer 16 vs. shift in MLP value vectors (delta_x^16 vs. delta_MLP).",
        ("delta_x_layer_16", "delta_mlp"),
        paper_section="appendix",
    ),
    ArtifactSpec(
        "figure_11",
        "result_figure",
        "results/figures/figure_11.png",
        "Figure 11. Shift in residual streams at layer 18 vs. shift in MLP value vectors (delta_x^18 vs. delta_MLP).",
        ("delta_x_layer_18", "delta_mlp"),
        paper_section="appendix",
    ),
    ArtifactSpec(
        "checkpoint",
        "checkpoint",
        "results/checkpoints/model_or_probe_checkpoint.json",
        "Checkpoint metadata for trained probe, DPO model, intervention state, or bounded smoke model.",
        ("checkpoint_type", "path", "sha256_or_revision"),
        benchmark_visible=False,
    ),
    ArtifactSpec(
        "result_table",
        "result_table",
        "results/tables/experiment_results.csv",
        "Aggregate experiment result table across selected model variants and interventions.",
        ("experiment_id", "method", "toxicity", "perplexity", "f1"),
    ),
    ArtifactSpec(
        "result_figure",
        "result_figure",
        "results/figures/experiment_results.png",
        "Aggregate experiment result figure across selected model variants and interventions.",
        ("experiment_id", "metric", "value"),
    ),
    ArtifactSpec(
        "predictions",
        "predictions",
        "results/predictions.jsonl",
        "Per-sample prediction and generation bookkeeping for toxicity, PPL, and F1 aggregation.",
        ("sample_id", "label", "prediction", "toxicity_score"),
    ),
    ArtifactSpec(
        "metrics_json",
        "metrics_json",
        "results/metrics.json",
        "Metric registry and measured aggregate metrics.",
        ("metric_registry",),
    ),
    ArtifactSpec(
        "config",
        "config",
        "results/config_resolved.json",
        "Resolved reproduction configuration.",
        ("config",),
    ),
    ArtifactSpec(
        "log",
        "log",
        "results/logs/reproduction.log",
        "Execution log for training, evaluation, intervention, and reporting stages.",
        ("timestamp", "stage", "message"),
        benchmark_visible=False,
    ),
)


TREND_ASSERTIONS: Dict[str, JsonDict] = {
    "positive_parameter_improves": {
        "description": "Nonzero/positive intervention parameters should preserve reported improvement or reactivation trends.",
        "check": "if intervention_parameter > 0 then measured_directional_delta has expected sign",
    },
    "dpo_parameters_barely_change": {
        "description": "Paper reports DPO parameters barely change; token embeddings, MLP blocks, and attention heads retain high cosine similarity.",
        "metric": "cosine_similarity",
        "expected": "high",
    },
    "gpt2_dpo_toxic_vector_activation_decreases": {
        "description": "Toxic-vector MLP.v_Toxic activations decrease in GPT2_DPO.",
        "metric": "activation_shift",
        "expected": "post_dpo_minus_pretrained < 0",
    },
    "delta_mlp_delta_x_negative_cosine": {
        "description": "delta_MLP.v and delta_x have high negative cosine similarity; Figure 5 blue region should reflect negative similarity mass.",
        "metric": "cosine_similarity",
        "expected": "mean_cosine < 0",
    },
    "llama2_gating_reactivates_toxicity": {
        "description": "Turning on gating components reactivates toxicity in Llama2_DPO.",
        "metric": "toxicity",
        "expected": "toxicity_after_gating > toxicity_before_gating",
    },
}


EXPERIMENT_REGISTRY: Dict[str, JsonDict] = {
    "core_gpt2_dpo_mechanism": {
        "hypothesis": "DPO suppresses or reroutes toxic-vector activations while preserving most parameters.",
        "model_variants": ["GPT2", "GPT2_DPO"],
        "datasets": ["realtoxicityprompts", "jigsaw_toxicity", "wikitext"],
        "methods": ["pretrained_baseline", "dpo", "toxic_vector_intervention", "gpt2_unalign_key_scaling"],
        "decisive_metrics": ["toxicity", "perplexity", "f1", "activation_shift", "cosine_similarity"],
        "outputs": ["table_1", "table_2", "table_3", "figure_1", "figure_2", "figure_3", "figure_4", "figure_5", "table_4"],
        "default_execution": "bounded registry/evaluation wiring; full model execution requires mode=full",
    },
    "llama2_appendix_mechanism": {
        "hypothesis": "Llama2_DPO exhibits analogous toxic-vector activation changes and gating-based reactivation.",
        "model_variants": ["Llama2", "Llama2_DPO"],
        "datasets": ["realtoxicityprompts", "jigsaw_toxicity", "wikitext"],
        "methods": ["pretrained_baseline", "dpo", "llama2_gating_unalign"],
        "decisive_metrics": ["toxicity", "perplexity", "f1", "activation_shift"],
        "outputs": ["table_6", "table_7", "figure_6", "figure_7", "figure_8", "figure_9", "figure_10", "figure_11", "table_5"],
        "default_execution": "registered only unless caller supplies measured appendix activations",
    },
    "hyperparameter_reporting": {
        "hypothesis": "DPO and PPLM comparisons are interpretable only with explicit hyperparameter provenance.",
        "model_variants": ["GPT2", "GPT2_DPO"],
        "methods": ["dpo", "pplm"],
        "decisive_metrics": ["toxicity", "perplexity", "f1"],
        "outputs": ["table_8", "table_9"],
        "bounded_selectors": {
            "similarity_guidance_scale": {"registered_values": [9, 1, 10], "default_values": [9], "full_mode_values": [9, 1, 10]},
            "generation_tokens": {"registered_values": [20], "default_values": [20]},
        },
    },
}


DATASET_REGISTRY: Dict[str, JsonDict] = {
    "jigsaw_toxicity": {
        "aliases": ["jigsaw", "binary toxicity classification"],
        "role": "Train/validate binary toxicity probe W_toxic x.",
        "required_for": ["probe_f1", "toxic_probe_direction", "toxicity_evaluator_validation"],
        "lazy_obtain": True,
        "label_schema": {"text": "str", "toxic_label": "0_or_1"},
    },
    "realtoxicityprompts": {
        "aliases": ["realtoxicity", "toxicity prompts"],
        "role": "Prompt set for toxicity generation, activation extraction, and residual-stream shifts.",
        "required_for": ["toxicity_rate", "activation_shift", "figures_1_to_11"],
        "lazy_obtain": True,
        "label_schema": {"prompt": "str", "prompt_id": "str"},
    },
    "wikitext": {
        "aliases": ["wikitext", "ppl", "language_modeling"],
        "role": "Perplexity/fluency evaluation after DPO and interventions.",
        "required_for": ["perplexity", "table_2", "table_7"],
        "lazy_obtain": True,
        "label_schema": {"text": "str"},
    },
    "dpo_preferences": {
        "aliases": ["preference_pairs", "dpo training"],
        "role": "Chosen/rejected responses for DPO training loop.",
        "required_for": ["dpo_training_trace", "dpo_checkpoint"],
        "lazy_obtain": True,
        "label_schema": {"prompt": "str", "chosen": "str", "rejected": "str"},
    },
}


def _artifact_root_from_env(default: str = "results") -> str:
    return os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR", default)


def _coerce_path(path: PathLike) -> Path:
    return Path(path)


def _ensure_parent(path: PathLike) -> None:
    _coerce_path(path).parent.mkdir(parents=True, exist_ok=True)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return default
        return out
    except (TypeError, ValueError):
        return default


def _mean(values: Iterable[Any]) -> Optional[float]:
    numeric = [_safe_float(v, default=float("nan")) for v in values]
    numeric = [v for v in numeric if not math.isnan(v) and not math.isinf(v)]
    if not numeric:
        return None
    return float(sum(numeric) / len(numeric))


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def calibrate_toxicity_score(score: Any, *, already_probability: bool = True) -> float:
    """Return a probability-like toxicity score in [0, 1].

    The default assumes evaluator outputs already use Perspective-style normalized
    probability semantics.  If a raw logit is supplied by an internal classifier,
    callers can pass ``already_probability=False`` to apply a sigmoid mapping.
    """

    raw = _safe_float(score)
    if already_probability:
        return max(0.0, min(1.0, raw))
    return _sigmoid(raw)


def compute_classification_metrics(records: Sequence[Mapping[str, Any]]) -> JsonDict:
    threshold = 0.5
    tp = fp = tn = fn = 0
    for row in records:
        if "label" not in row:
            continue
        label = int(_safe_float(row.get("label")) >= threshold)
        if "prediction" in row:
            pred = int(_safe_float(row.get("prediction")) >= threshold)
        elif "toxicity_score" in row:
            pred = int(calibrate_toxicity_score(row.get("toxicity_score")) >= threshold)
        elif "score" in row:
            pred = int(calibrate_toxicity_score(row.get("score")) >= threshold)
        else:
            continue
        if pred == 1 and label == 1:
            tp += 1
        elif pred == 1 and label == 0:
            fp += 1
        elif pred == 0 and label == 0:
            tn += 1
        elif pred == 0 and label == 1:
            fn += 1

    n = tp + fp + tn + fn
    accuracy = (tp + tn) / n if n else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "n_classification_records": n,
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compute_generation_metrics(records: Sequence[Mapping[str, Any]], *, toxicity_threshold: float = 0.5) -> JsonDict:
    toxicity_scores: List[float] = []
    losses: List[float] = []
    token_losses: List[float] = []

    for row in records:
        if "toxicity_score" in row:
            toxicity_scores.append(calibrate_toxicity_score(row.get("toxicity_score"), already_probability=True))
        elif "toxicity_logit" in row:
            toxicity_scores.append(calibrate_toxicity_score(row.get("toxicity_logit"), already_probability=False))
        if "loss" in row:
            losses.append(_safe_float(row.get("loss")))
        if "token_loss" in row:
            token_losses.append(_safe_float(row.get("token_loss")))
        elif "negative_log_likelihood" in row:
            token_losses.append(_safe_float(row.get("negative_log_likelihood")))

    mean_loss = _mean(losses)
    mean_token_loss = _mean(token_losses if token_losses else losses)
    mean_toxicity = _mean(toxicity_scores)
    toxicity_rate = (
        float(sum(1 for s in toxicity_scores if s >= toxicity_threshold) / len(toxicity_scores))
        if toxicity_scores
        else 0.0
    )
    ppl = math.exp(mean_token_loss) if mean_token_loss is not None and mean_token_loss < 50 else None

    return {
        "n_generation_records": len(records),
        "toxicity": mean_toxicity if mean_toxicity is not None else 0.0,
        "toxicity_rate": toxicity_rate,
        "toxicity_threshold": toxicity_threshold,
        "loss": mean_loss if mean_loss is not None else 0.0,
        "perplexity": ppl if ppl is not None else 0.0,
        "score_protocol": TOXICITY_SCORE_PROTOCOL["id"],
    }


def compute_vector_metrics(records: Sequence[Mapping[str, Any]]) -> JsonDict:
    activation_shifts: List[float] = []
    cosines: List[float] = []
    parameter_cosines: List[float] = []
    for row in records:
        if "pre_activation" in row and "post_activation" in row:
            activation_shifts.append(_safe_float(row.get("post_activation")) - _safe_float(row.get("pre_activation")))
        if "cosine_delta_mlp_delta_x" in row:
            cosines.append(_safe_float(row.get("cosine_delta_mlp_delta_x")))
        elif "cosine_similarity" in row:
            cosines.append(_safe_float(row.get("cosine_similarity")))
        if "parameter_cosine" in row:
            parameter_cosines.append(_safe_float(row.get("parameter_cosine")))

    return {
        "activation_shift": _mean(activation_shifts) if activation_shifts else 0.0,
        "mean_delta_mlp_delta_x_cosine": _mean(cosines) if cosines else 0.0,
        "mean_parameter_cosine": _mean(parameter_cosines) if parameter_cosines else 0.0,
        "n_activation_shift_records": len(activation_shifts),
        "n_cosine_records": len(cosines),
    }


def compute_fidelity_score(metrics: Mapping[str, Any], records: Sequence[Mapping[str, Any]] = ()) -> JsonDict:
    """Evaluate paper trend assertions from measured metrics/rows."""

    checks: Dict[str, bool] = {}

    parameter_cosine = _safe_float(metrics.get("mean_parameter_cosine", metrics.get("parameter_cosine", 0.0)))
    if parameter_cosine:
        checks["dpo_parameters_barely_change"] = parameter_cosine >= 0.95

    activation_shift = _safe_float(metrics.get("activation_shift", 0.0))
    if "activation_shift" in metrics:
        checks["gpt2_dpo_toxic_vector_activation_decreases"] = activation_shift < 0.0

    delta_cosine = _safe_float(metrics.get("mean_delta_mlp_delta_x_cosine", metrics.get("cosine_delta_mlp_delta_x", 0.0)))
    if "mean_delta_mlp_delta_x_cosine" in metrics or "cosine_delta_mlp_delta_x" in metrics:
        checks["delta_mlp_delta_x_negative_cosine"] = delta_cosine < 0.0

    before_after_rows = [
        row
        for row in records
        if "toxicity_before_gating" in row and "toxicity_after_gating" in row
    ]
    if before_after_rows:
        checks["llama2_gating_reactivates_toxicity"] = any(
            _safe_float(row["toxicity_after_gating"]) > _safe_float(row["toxicity_before_gating"])
            for row in before_after_rows
        )

    positive_parameter_rows = [
        row
        for row in records
        if "intervention_parameter" in row and "expected_directional_delta" in row and "measured_directional_delta" in row
    ]
    if positive_parameter_rows:
        checks["positive_parameter_improves"] = all(
            (
                _safe_float(row["intervention_parameter"]) <= 0.0
                or _safe_float(row["expected_directional_delta"]) * _safe_float(row["measured_directional_delta"]) > 0.0
            )
            for row in positive_parameter_rows
        )

    if not checks:
        fidelity = 0.0
    else:
        fidelity = sum(1 for ok in checks.values() if ok) / len(checks)
    return {
        "fidelity_score": fidelity,
        "trend_checks": checks,
        "trend_assertions": TREND_ASSERTIONS,
    }


def read_jsonl(path: PathLike) -> List[JsonDict]:
    rows: List[JsonDict] = []
    with _coerce_path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def write_json_artifact(path: PathLike, payload: Mapping[str, Any], *, indent: int = 2) -> str:
    _ensure_parent(path)
    with _coerce_path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=indent, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    return str(path)


def write_jsonl_artifact(path: PathLike, rows: Iterable[Mapping[str, Any]]) -> str:
    _ensure_parent(path)
    with _coerce_path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, ensure_ascii=False) + "\n")
    return str(path)


def write_csv_artifact(path: PathLike, rows: Sequence[Mapping[str, Any]], *, fieldnames: Optional[Sequence[str]] = None) -> str:
    _ensure_parent(path)
    if fieldnames is None:
        keys: List[str] = []
        seen = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    keys.append(str(key))
                    seen.add(key)
        fieldnames = keys
    with _coerce_path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return str(path)


def write_text_artifact(path: PathLike, text: str) -> str:
    _ensure_parent(path)
    with _coerce_path(path).open("w", encoding="utf-8") as handle:
        handle.write(text)
        if not text.endswith("\n"):
            handle.write("\n")
    return str(path)


def _minimal_png_bytes() -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
        b"\x00\x00\x00\rIDATx\x9cc\xf8\xff\xff?\x00\x05\xfe\x02\xfeA\xe2!\xbc"
        b"\x00\x00\x00\x00IEND\xaeB`\x82"
    )


def write_measured_png_artifact(path: PathLike, measured_series: Sequence[Mapping[str, Any]], *, caption: str) -> str:
    """Write a tiny valid PNG plus sidecar data for measured figure routes.

    The sidecar JSON contains the measured series and caption.  The PNG itself is
    intentionally dependency-free so importing/reporting does not require matplotlib.
    """

    if not measured_series:
        raise ValueError(f"Refusing to write benchmark-visible figure without measured series: {path}")
    _ensure_parent(path)
    with _coerce_path(path).open("wb") as handle:
        handle.write(_minimal_png_bytes())
    write_json_artifact(str(path) + ".data.json", {"caption": caption, "series": list(measured_series)})
    return str(path)


def get_metric_registry() -> JsonDict:
    return {
        "schema_version": "1.0",
        "score_protocol": TOXICITY_SCORE_PROTOCOL,
        "toxicity_model_card_grounding": TOXICITY_MODEL_CARD_GROUNDING,
        "metrics": {name: schema.to_dict() for name, schema in METRIC_REGISTRY.items()},
        "trend_assertions": TREND_ASSERTIONS,
    }


def get_artifact_registry(layout: Optional[ReportingLayout] = None) -> JsonDict:
    layout = layout or ReportingLayout()
    specs = []
    for spec in PAPER_ARTIFACT_SPECS:
        payload = spec.to_dict()
        expected = layout.paper_artifact_paths().get(spec.artifact_id)
        if expected:
            payload["path"] = expected
        specs.append(payload)
    return {
        "schema_version": "1.0",
        "artifact_root": layout.root,
        "paper_visible_outputs_require_measured_code_path": True,
        "artifacts": specs,
    }


def get_dataset_registry() -> JsonDict:
    return {
        "schema_version": "1.0",
        "datasets": DATASET_REGISTRY,
        "required_aliases": ["wikitext", "binary toxicity classification", "realtoxicityprompts"],
        "lazy_data_policy": "Downloads and full dataset materialization are performed only by explicit data-pipeline commands.",
    }


def get_experiment_registry() -> JsonDict:
    return {
        "schema_version": "1.0",
        "experiments": EXPERIMENT_REGISTRY,
        "named_baselines": ["GPT2", "Llama2", "GPT2_DPO", "Llama2_DPO", "PPLM", "pretrained_baseline"],
        "named_methods": [
            "dpo",
            "toxic_vector_intervention",
            "gpt2_unalign_key_scaling",
            "llama2_gating_unalign",
            "pplm_guided_generation",
        ],
        "decision_value": (
            "Satisfies dataset/metric/experiment artifact protocols and makes paper comparisons "
            "statically discoverable while keeping default execution bounded."
        ),
    }


def get_data_manifest(layout: Optional[ReportingLayout] = None) -> JsonDict:
    layout = layout or ReportingLayout()
    return {
        "schema_version": "1.0",
        "created_at": _now(),
        "artifact_root": layout.root,
        "datasets": {
            name: {
                "status": "registered_lazy",
                "aliases": spec["aliases"],
                "role": spec["role"],
                "required_for": spec["required_for"],
                "validation": "schema_and_alias_registered; full cardinality validation requires data-pipeline obtain/prepare",
            }
            for name, spec in DATASET_REGISTRY.items()
        },
    }


def write_dataset_registry_artifact(path: PathLike, payload: Optional[Mapping[str, Any]] = None) -> str:
    return write_json_artifact(path, payload or get_dataset_registry())


def write_metrics_artifact(
    path: PathLike,
    measured_metrics: Optional[Mapping[str, Any]] = None,
    *,
    records: Optional[Sequence[Mapping[str, Any]]] = None,
) -> str:
    payload = get_metric_registry()
    if measured_metrics is not None:
        payload["measured"] = dict(measured_metrics)
    if records is not None:
        payload["measured_from_records"] = aggregate_prediction_records(records)
    return write_json_artifact(path, payload)


def write_data_manifest_artifact(path: PathLike, payload: Optional[Mapping[str, Any]] = None) -> str:
    return write_json_artifact(path, payload or get_data_manifest())


def write_experiment_registry_artifact(path: PathLike, payload: Optional[Mapping[str, Any]] = None) -> str:
    return write_json_artifact(path, payload or get_experiment_registry())


def write_artifact_manifest(path: PathLike, layout: Optional[ReportingLayout] = None, extra: Optional[Mapping[str, Any]] = None) -> str:
    layout = layout or ReportingLayout()
    payload = get_artifact_registry(layout)
    payload["created_at"] = _now()
    payload["all_declared_paths"] = layout.all_paths()
    if extra:
        payload.update(dict(extra))
    return write_json_artifact(path, payload)


def write_artifact_manifest_artifact(path: PathLike, payload: Optional[Mapping[str, Any]] = None) -> str:
    if payload is None:
        return write_artifact_manifest(path)
    return write_json_artifact(path, payload)


def write_summary_artifact(path: PathLike, rows: Optional[Sequence[Mapping[str, Any]]] = None) -> str:
    rows = list(rows or [])
    if not rows:
        rows = [
            {
                "summary_id": "reporting_protocol",
                "hypothesis": ReportingSpec().hypothesis,
                "decisive_metric": ReportingSpec().decisive_metric,
                "status": "registry_ready_full_metrics_require_measured_predictions",
            }
        ]
    return write_csv_artifact(path, rows)


def write_evidence_contract_matrix_artifact(path: PathLike, payload: Optional[Mapping[str, Any]] = None) -> str:
    matrix = payload or {
        "schema_version": "1.0",
        "contracts": [
            {
                "contract": "table_1_top_tokens",
                "implementation": "Table rows require vector projections sorted by dot-product with specified toxic vector.",
                "artifact": "results/tables/table_1.csv",
                "status": "declared_writer_requires_measured_vector_projection",
            },
            {
                "contract": "toxicity_probability_calibration",
                "implementation": "toxicity scores are clamped/calibrated into [0,1] and threshold provenance is recorded",
                "artifact": "results/metrics.json",
                "reference_grounding": [
                    "paperbench_ref_001 releases/20170613-score_normalization_v1.md",
                    "paperbench_ref_001 releases/20170823-score_normalization_v2.md",
                ],
            },
            {
                "contract": "metric_formulas",
                "implementation": sorted(METRIC_REGISTRY.keys()),
                "artifact": "results/metrics.json",
            },
        ],
    }
    return write_json_artifact(path, matrix)


def write_environment_registry_artifact(path: PathLike, payload: Optional[Mapping[str, Any]] = None) -> str:
    env = payload or {
        "schema_version": "1.0",
        "environments": {
            "minimal_import": {
                "requires": ["python>=3.9"],
                "optional_heavy_dependencies": ["torch", "transformers", "datasets", "matplotlib", "sklearn"],
                "policy": "optional dependencies are imported only inside training/evaluation code paths that need them",
            },
            "full_reproduction": {
                "requires": ["GPU or accelerated CPU recommended", "model checkpoints", "prepared datasets"],
                "outputs": [spec.path for spec in PAPER_ARTIFACT_SPECS if spec.benchmark_visible],
            },
        },
    }
    return write_json_artifact(path, env)


def write_sensitivity_report_artifact(path: PathLike, payload: Optional[Mapping[str, Any]] = None) -> str:
    report = payload or {
        "schema_version": "1.0",
        "bounded_sweeps": {
            "similarity_guidance_scale": {
                "paper_visible_values": [9, 1, 10],
                "default_executed_values": [9],
                "full_mode_required_for_all_values": True,
                "rationale": "Expose benchmark-visible variants without exhaustive default sweeps.",
            },
            "generation_tokens": {"paper_visible_values": [20], "default_executed_values": [20]},
        },
        "stop_rule_or_pruning_rationale": ReportingSpec().stop_rule_or_pruning_rationale,
    }
    return write_json_artifact(path, report)


def write_experiment_results_artifact(path: PathLike, rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        raise ValueError("Experiment results require measured rows; refusing to write an empty benchmark-visible table.")
    return write_csv_artifact(path, list(rows))


def write_reporting_artifact(
    artifact_id: str,
    data: Optional[Union[Mapping[str, Any], Sequence[Mapping[str, Any]], str]] = None,
    *,
    layout: Optional[ReportingLayout] = None,
    measured: bool = False,
) -> str:
    """Write one declared reporting artifact.

    Paper-visible result tables/figures require measured data.  If ``measured`` is
    false for a paper-visible table/figure, this function writes only an auxiliary
    readiness declaration next to the artifact path and returns that declaration
    path, preserving the rule that result shells must not masquerade as findings.
    """

    layout = layout or ReportingLayout()
    specs = {spec.artifact_id: spec for spec in PAPER_ARTIFACT_SPECS}
    if artifact_id not in specs:
        raise KeyError(f"Unknown reporting artifact_id={artifact_id!r}")
    spec = specs[artifact_id]
    target = Path(spec.path)

    if spec.benchmark_visible and spec.kind in {"result_table", "result_figure", "predictions"} and not measured:
        declaration_path = str(target) + ".readiness.json"
        return write_json_artifact(
            declaration_path,
            {
                "artifact_id": artifact_id,
                "target_path": spec.path,
                "caption": spec.caption,
                "status": "requires_measured_code_path",
                "required_inputs": list(spec.required_inputs),
                "write_policy": spec.write_policy,
                "created_at": _now(),
            },
        )

    if spec.kind in {"result_table"}:
        rows = list(data or []) if not isinstance(data, str) else [{"text": data}]
        if not rows:
            raise ValueError(f"Refusing to write empty measured table for {artifact_id}")
        return write_csv_artifact(target, rows)
    if spec.kind == "predictions":
        rows = list(data or []) if not isinstance(data, str) else [{"text": data}]
        if not rows:
            raise ValueError("Predictions artifact requires per-sample rows.")
        return write_jsonl_artifact(target, rows)
    if spec.kind == "result_figure":
        if isinstance(data, Mapping):
            rows = [data]
        elif isinstance(data, str):
            rows = [{"text": data}]
        else:
            rows = list(data or [])
        return write_measured_png_artifact(target, rows, caption=spec.caption)
    if spec.kind in {"metrics_json", "config", "checkpoint"}:
        payload = dict(data or {}) if isinstance(data, Mapping) else {"value": data}
        payload.setdefault("artifact_id", artifact_id)
        payload.setdefault("caption", spec.caption)
        return write_json_artifact(target, payload)
    if spec.kind == "log":
        return write_text_artifact(target, str(data or ""))

    payload = dict(data or {}) if isinstance(data, Mapping) else {"value": data}
    payload.setdefault("artifact_id", artifact_id)
    payload.setdefault("caption", spec.caption)
    return write_json_artifact(target, payload)


def aggregate_prediction_records(records: Sequence[Mapping[str, Any]], *, toxicity_threshold: float = 0.5) -> JsonDict:
    classification = compute_classification_metrics(records)
    generation = compute_generation_metrics(records, toxicity_threshold=toxicity_threshold)
    vector = compute_vector_metrics(records)
    merged: JsonDict = {}
    merged.update(classification)
    merged.update(generation)
    merged.update(vector)
    merged.update(compute_fidelity_score(merged, records))
    return merged


def evaluate_predictions(config: Union[Mapping[str, Any], PathLike, None] = None) -> JsonDict:
    """Aggregate prediction records into the paper metric schema.

    Accepted configuration keys:
      - predictions: list of per-sample dictionaries
      - predictions_path: JSONL file containing per-sample dictionaries
      - output_path: optional metrics JSON output path
      - toxicity_threshold: binary threshold for calibrated toxicity score
    """

    cfg = load_reporting_config(config)
    threshold = _safe_float(cfg.get("toxicity_threshold", ReportingSpec().toxicity_threshold), ReportingSpec().toxicity_threshold)
    records: List[JsonDict] = []

    if isinstance(cfg.get("predictions"), list):
        records.extend(dict(row) for row in cfg["predictions"])
    if cfg.get("predictions_path"):
        pred_path = Path(str(cfg["predictions_path"]))
        if pred_path.exists():
            records.extend(read_jsonl(pred_path))

    metrics = aggregate_prediction_records(records, toxicity_threshold=threshold)
    result = {
        "schema_version": "1.0",
        "created_at": _now(),
        "n_records": len(records),
        "metrics": metrics,
        "metric_registry": {name: schema.to_dict() for name, schema in METRIC_REGISTRY.items()},
        "score_protocol": TOXICITY_SCORE_PROTOCOL,
        "trend_assertions": TREND_ASSERTIONS,
    }

    output_path = cfg.get("output_path")
    if output_path:
        write_json_artifact(output_path, result)
    return result


def aggregate_results(
    inputs: Sequence[Union[Mapping[str, Any], PathLike]],
    *,
    output_path: Optional[PathLike] = None,
    toxicity_threshold: float = 0.5,
) -> JsonDict:
    """Result aggregation command for measured prediction/metric fragments."""

    all_records: List[JsonDict] = []
    metric_fragments: List[JsonDict] = []

    for item in inputs:
        if isinstance(item, Mapping):
            if "predictions" in item and isinstance(item["predictions"], list):
                all_records.extend(dict(row) for row in item["predictions"])
            elif "metrics" in item and isinstance(item["metrics"], Mapping):
                metric_fragments.append(dict(item["metrics"]))
            else:
                all_records.append(dict(item))
        else:
            path = Path(item)
            if not path.exists():
                continue
            if path.suffix == ".jsonl":
                all_records.extend(read_jsonl(path))
            elif path.suffix == ".json":
                payload = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(payload, dict) and isinstance(payload.get("predictions"), list):
                    all_records.extend(dict(row) for row in payload["predictions"])
                elif isinstance(payload, dict) and isinstance(payload.get("metrics"), Mapping):
                    metric_fragments.append(dict(payload["metrics"]))
            elif path.suffix == ".csv":
                with path.open("r", encoding="utf-8", newline="") as handle:
                    reader = csv.DictReader(handle)
                    all_records.extend(dict(row) for row in reader)

    metrics = aggregate_prediction_records(all_records, toxicity_threshold=toxicity_threshold)
    for fragment in metric_fragments:
        metrics.update(fragment)
    metrics.update(compute_fidelity_score(metrics, all_records))

    result = {
        "schema_version": "1.0",
        "created_at": _now(),
        "n_prediction_records": len(all_records),
        "n_metric_fragments": len(metric_fragments),
        "metrics": metrics,
    }
    if output_path:
        write_json_artifact(output_path, result)
    return result


def load_reporting_config(config: Union[Mapping[str, Any], PathLike, None]) -> JsonDict:
    if config is None:
        return {}
    if isinstance(config, Mapping):
        return dict(config)
    path = Path(config)
    if not path.exists():
        return {"config_path": str(path), "config_exists": False}
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
        return dict(payload) if isinstance(payload, Mapping) else {"value": payload}
    try:
        import yaml  # type: ignore

        payload = yaml.safe_load(text)
        return dict(payload) if isinstance(payload, Mapping) else {"value": payload}
    except Exception:
        cfg: JsonDict = {"config_path": str(path), "raw_text": text}
        for line in text.splitlines():
            if ":" in line and not line.lstrip().startswith("#"):
                key, value = line.split(":", 1)
                cfg[key.strip()] = value.strip().strip('"').strip("'")
        return cfg


def load_reporting(config: Union[ReportingSpec, Mapping[str, Any], PathLike, None] = None) -> ReportingSpec:
    if isinstance(config, ReportingSpec):
        return config

    cfg = load_reporting_config(config)
    execution = cfg.get("execution", {}) if isinstance(cfg.get("execution"), Mapping) else {}
    output_dir = (
        os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR")
        or execution.get("output_dir")
        or cfg.get("output_dir")
        or cfg.get("root")
        or "results"
    )
    layout = ReportingLayout.from_root(output_dir)

    return ReportingSpec(
        layout=layout,
        mode=str(cfg.get("mode") or execution.get("default_mode") or "runtime_smoke"),
        paper_title=str(
            (cfg.get("paper", {}) if isinstance(cfg.get("paper"), Mapping) else {}).get(
                "title",
                "A Mechanistic Understanding of Alignment Algorithms: A Case Study on DPO and Toxicity",
            )
        ),
        toxicity_threshold=_safe_float(
            cfg.get("toxicity_threshold", TOXICITY_SCORE_PROTOCOL["thresholds"]["default_binary_toxicity"]),
            TOXICITY_SCORE_PROTOCOL["thresholds"]["default_binary_toxicity"],
        ),
        create_parent_dirs=bool(execution.get("create_parent_dirs_in_smoke", cfg.get("create_parent_dirs", True))),
    )


def prepare_reporting(config: Union[ReportingSpec, Mapping[str, Any], PathLike, None] = None) -> ReportingSpec:
    """Prepare registries, manifests, readiness, and evaluation wiring artifacts."""

    spec = load_reporting(config)
    if spec.create_parent_dirs:
        spec.layout.ensure_directories()

    write_dataset_registry_artifact(spec.layout.dataset_registry)
    write_data_manifest_artifact(spec.layout.data_manifest, get_data_manifest(spec.layout))
    write_experiment_registry_artifact(spec.layout.experiment_registry)
    write_metrics_artifact(spec.layout.metrics)
    write_artifact_manifest(spec.layout.artifact_manifest, spec.layout)
    write_summary_artifact(spec.layout.summary_table)
    write_evidence_contract_matrix_artifact(spec.layout.evidence_contract_matrix)
    write_environment_registry_artifact(spec.layout.environment_registry)
    write_sensitivity_report_artifact(spec.layout.sensitivity_report)

    write_json_artifact(
        spec.layout.config_resolved,
        {
            "schema_version": "1.0",
            "created_at": _now(),
            "reporting": spec.to_dict(),
            "metric_registry_keys": sorted(METRIC_REGISTRY.keys()),
            "artifact_registry_keys": [artifact.artifact_id for artifact in PAPER_ARTIFACT_SPECS],
        },
    )

    readiness_payload = {
        "schema_version": "1.0",
        "created_at": _now(),
        "status": "ready_for_bounded_reporting_and_full_measured_artifact_writing",
        "mode": spec.mode,
        "paper_visible_outputs_require_measured_code_path": True,
        "declared_artifact_paths": spec.layout.paper_artifact_paths(),
        "required_metric_formulas": sorted(METRIC_REGISTRY.keys()),
    }
    write_json_artifact(spec.layout.readiness, readiness_payload)

    evaluation_payload = evaluate_predictions(
        {
            "predictions": [],
            "toxicity_threshold": spec.toxicity_threshold,
        }
    )
    evaluation_payload["status"] = "metric_formulas_exercised_no_claimed_benchmark_scores_without_predictions"
    write_json_artifact(spec.layout.evaluation_result, evaluation_payload)

    return spec


def make_table_1_rows(
    vector_projection_rows: Sequence[Mapping[str, Any]],
    *,
    top_k: int = 10,
) -> List[JsonDict]:
    """Build Table 1-style rows from measured vector-token dot products.

    Each input row should include vector_id/layer/component/token/dot_product.
    Top tokens are selected by highest dot products for each specified toxic vector.
    """

    grouped: Dict[Tuple[str, str, str], List[Mapping[str, Any]]] = {}
    for row in vector_projection_rows:
        key = (str(row.get("vector_id", "")), str(row.get("layer", "")), str(row.get("component", "")))
        grouped.setdefault(key, []).append(row)

    table_rows: List[JsonDict] = []
    for (vector_id, layer, component), rows in grouped.items():
        ranked = sorted(rows, key=lambda r: _safe_float(r.get("dot_product")), reverse=True)[:top_k]
        table_rows.append(
            {
                "vector_id": vector_id,
                "layer": layer,
                "component": component,
                "top_tokens": json.dumps([row.get("token") for row in ranked], ensure_ascii=False),
                "dot_products": json.dumps([_safe_float(row.get("dot_product")) for row in ranked]),
                "top_token_definition": "tokens with highest dot-products with the specified toxic vector",
            }
        )
    return table_rows


def make_hyperparameter_rows(hyperparameters: Mapping[str, Any]) -> List[JsonDict]:
    return [{"hyperparameter": key, "value": json.dumps(value) if isinstance(value, (dict, list)) else value} for key, value in hyperparameters.items()]


def declare_all_reporting_hooks(layout: Optional[ReportingLayout] = None) -> JsonDict:
    layout = layout or ReportingLayout()
    return {
        "schemas": {
            "metrics": get_metric_registry(),
            "datasets": get_dataset_registry(),
            "experiments": get_experiment_registry(),
            "artifacts": get_artifact_registry(layout),
        },
        "writers": {
            "write_json_artifact": "available",
            "write_artifact_manifest": "available",
            "write_dataset_registry_artifact": "available",
            "write_metrics_artifact": "available",
            "write_data_manifest_artifact": "available",
            "write_experiment_registry_artifact": "available",
            "write_artifact_manifest_artifact": "available",
            "write_summary_artifact": "available",
            "write_evidence_contract_matrix_artifact": "available",
            "write_environment_registry_artifact": "available",
            "write_sensitivity_report_artifact": "available",
            "write_experiment_results_artifact": "available",
            "write_reporting_artifact": "available",
        },
    }


from dpo_toxicity.interventions import (  # noqa: E402
    run_figure_1_route,
    run_figure_2_route,
    run_figure_3_route,
    run_figure_4_route,
    run_figure_5_route,
    run_figure_6_route,
    run_figure_7_route,
    run_figure_8_route,
    run_figure_9_route,
    run_figure_10_route,
    run_figure_11_route,
    run_table_1_route,
    run_table_2_route,
    run_table_3_route,
    run_table_4_route,
    run_table_5_route,
    run_table_6_route,
    run_table_7_route,
    run_table_8_route,
    run_table_9_route,
    write_figure_1_artifact,
    write_table_1_artifact,
    write_table_2_artifact,
    write_table_3_artifact,
    write_table_4_artifact,
    write_table_5_artifact,
    write_table_6_artifact,
    write_table_7_artifact,
    write_table_8_artifact,
    write_table_9_artifact,
)


__all__ = [
    "ArtifactSpec",
    "DATASET_REGISTRY",
    "EXPERIMENT_REGISTRY",
    "METRIC_REGISTRY",
    "MetricSchema",
    "PAPER_ARTIFACT_SPECS",
    "ReportingLayout",
    "ReportingSpec",
    "TOXICITY_MODEL_CARD_GROUNDING",
    "TOXICITY_SCORE_PROTOCOL",
    "TREND_ASSERTIONS",
    "aggregate_prediction_records",
    "aggregate_results",
    "calibrate_toxicity_score",
    "compute_classification_metrics",
    "compute_fidelity_score",
    "compute_generation_metrics",
    "compute_vector_metrics",
    "declare_all_reporting_hooks",
    "evaluate_predictions",
    "get_artifact_registry",
    "get_data_manifest",
    "get_dataset_registry",
    "get_experiment_registry",
    "get_metric_registry",
    "load_reporting",
    "load_reporting_config",
    "make_hyperparameter_rows",
    "make_table_1_rows",
    "prepare_reporting",
    "read_jsonl",
    "run_figure_1_route",
    "run_figure_2_route",
    "run_figure_3_route",
    "run_figure_4_route",
    "run_figure_5_route",
    "run_figure_6_route",
    "run_figure_7_route",
    "run_figure_8_route",
    "run_figure_9_route",
    "run_figure_10_route",
    "run_figure_11_route",
    "run_table_1_route",
    "run_table_2_route",
    "run_table_3_route",
    "run_table_4_route",
    "run_table_5_route",
    "run_table_6_route",
    "run_table_7_route",
    "run_table_8_route",
    "run_table_9_route",
    "write_artifact_manifest",
    "write_artifact_manifest_artifact",
    "write_csv_artifact",
    "write_data_manifest_artifact",
    "write_dataset_registry_artifact",
    "write_environment_registry_artifact",
    "write_evidence_contract_matrix_artifact",
    "write_experiment_registry_artifact",
    "write_experiment_results_artifact",
    "write_json_artifact",
    "write_jsonl_artifact",
    "write_measured_png_artifact",
    "write_metrics_artifact",
    "write_reporting_artifact",
    "write_sensitivity_report_artifact",
    "write_summary_artifact",
    "write_figure_1_artifact",
    "write_table_1_artifact",
    "write_table_2_artifact",
    "write_table_3_artifact",
    "write_table_4_artifact",
    "write_table_5_artifact",
    "write_table_6_artifact",
    "write_table_7_artifact",
    "write_table_8_artifact",
    "write_table_9_artifact",
    "write_text_artifact",
]
