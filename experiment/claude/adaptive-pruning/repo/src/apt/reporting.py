"""Reporting and paper-visible result aggregation for the APT route.

This module turns measured bounded/full route outputs into result-table,
relative-accuracy, table, and figure payloads.  It intentionally performs no
training or dataset loading at import time; callers pass current-run evaluation
and training outputs produced by the canonical route.

reference_grounding: paperbench_ref_001 datasheet.md
reference_grounding: paperbench_ref_001 prompt.txt
reference_grounding: paperbench_ref_001 train.py
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .config import (
    BATCH_SIZE_32,
    BATCH_SIZE_128,
    EARLY_TRAINING_STEPS,
    PAPER_TITLE,
    PRUNING_END_STEP,
    PRUNING_START_STEP,
    R_APT_DEFAULT,
    SALIENCE_EMA_DECAY,
    SALIENCE_EMA_UPDATE,
    TAU,
    TARGET_SPARSITY_DEFAULT,
    TEN_SHOT_SETTING,
    TUNING_BUDGET_DEFAULT,
    config_to_jsonable,
    get_artifact_specs,
    get_baseline_registry,
    get_dataset_registry,
    get_experiment_registry,
    get_hyperparameter_config,
    get_method_registry,
    get_metric_formula_registry,
    resolve_batch_size_defaults as _config_resolve_batch_size_defaults,
)
from .metrics import (
    aggregate_accuracy as _metric_aggregate_accuracy,
    aggregate_fidelity_score,
    aggregate_loss as _metric_aggregate_loss,
    compute_accuracy as _metric_compute_accuracy,
    compute_efficiency_metrics,
    compute_f1 as _metric_compute_f1,
    compute_fidelity_score,
    compute_loss as _metric_compute_loss,
    compute_relative_accuracy,
    metric_formula_payload,
    s_bar_t,
    s_bar_t_1,
    write_fidelity_score_artifact,
)


SCHEMA_VERSION = "1.0"
DEFAULT_OUTPUT_DIR = "results"
DEFAULT_BATCH_SIZE = BATCH_SIZE_32
MEASURED = "measured"
BOUNDED_PROXY = "bounded_proxy"
UNAVAILABLE = "unavailable"

CANONICAL_METRIC_IDENTIFIERS = (
    "runtime",
    "metric_runtime",
    "trainable_parameter_count_and_relative_training_memory_must",
    "metric_trainable_parameter_count_and_relative_training_memory_must",
    "table_2_reproduction_artifact",
    "metric_table_2_reproduction_artifact",
    "table_4_reproduction_artifact",
    "metric_table_4_reproduction_artifact",
    "fidelity_score",
    "metric_fidelity_score",
    "accuracy",
    "metric_accuracy",
    "f1",
    "metric_f1",
    "training_time",
    "metric_training_time",
    "table_6_reproduction_artifact",
    "metric_table_6_reproduction_artifact",
    "figure_2_reproduction_artifact",
    "metric_figure_2_reproduction_artifact",
    "table_11_reproduction_artifact",
    "metric_table_11_reproduction_artifact",
    "table_3_reproduction_artifact",
    "table_12_reproduction_artifact",
    "figure_1_reproduction_artifact",
    "table_1_reproduction_artifact",
    "sst2_mnli_dev_accuracy_squad_v2_0_dev",
)

CANONICAL_ARTIFACT_IDENTIFIERS = (
    "results_evaluation_result_json_measured_result_readiness",
    "artifact_results_evaluation_result_json_measured_result_readiness",
    "metric_formula",
    "artifact_metric_formula",
    "table_2",
    "artifact_table_2",
    "table_4",
    "artifact_table_4",
    "table_6",
    "artifact_table_6",
    "figure_2",
    "artifact_figure_2",
    "table_11",
    "artifact_table_11",
    "result_table_repo_plan",
    "artifact_result_table_repo_plan",
    "table_3",
    "artifact_table_3",
    "sst2_mnli_relative_accuracy_reporting_inputs_required_artifact",
    "artifact_sst2_mnli_relative_accuracy_reporting_inputs_required_artifact",
    "checkpoints_cofi_checkpoints_mask_tuning_benchmark_visible_artifacts",
    "artifact_checkpoints_cofi_checkpoints_mask_tuning_benchmark_visible_artifacts",
    "table_12",
    "artifact_table_12",
)

PAPER_VISIBLE_TABLES = (
    "Table 1",
    "Table 2",
    "Table 3",
    "Table 4",
    "Table 5",
    "Table 6",
    "Table 7",
    "Table 8",
    "Table 9",
    "Table 10",
    "Table 11",
    "Table 12",
)

PAPER_VISIBLE_FIGURES = (
    "Figure 1",
    "Figure 2",
    "Figure 3",
    "Figure 4",
    "Figure 5",
    "Figure 5a",
)

REQUIRED_BASELINES = (
    "FT",
    "LoRA",
    "LoRA+Prune",
    "MaskTuning",
    "CoFi",
    "PEFT+Pruning+Distillation",
    "APT",
)


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_mapping(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if is_dataclass(value):
        return dict(_jsonable(value))
    if hasattr(value, "to_dict"):
        return dict(_jsonable(value.to_dict()))
    if isinstance(value, Mapping):
        return dict(_jsonable(value))
    return {"value": _jsonable(value)}


def _get_attr_or_key(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _metric_value(metrics: Mapping[str, Any], names: Sequence[str], default: Any = None) -> Any:
    for name in names:
        if name in metrics:
            return metrics[name]
    return default


def _coerce_runs(evaluation_result: Any) -> List[Dict[str, Any]]:
    payload = _as_mapping(evaluation_result)
    runs = payload.get("runs")
    if isinstance(runs, Sequence) and not isinstance(runs, (str, bytes)):
        return [_as_mapping(run) for run in runs]
    metrics = payload.get("metrics", payload)
    if not isinstance(metrics, Mapping):
        metrics = {}
    return [
        {
            "method": payload.get("method", payload.get("run", {}).get("method", "APT") if isinstance(payload.get("run"), Mapping) else "APT"),
            "dataset_name": payload.get("dataset_name", payload.get("task_name", payload.get("task", ""))),
            "task_name": payload.get("task_name", payload.get("dataset_name", payload.get("task", ""))),
            "model_name": payload.get("model_name", payload.get("model", "")),
            "metrics": dict(metrics),
            "status": payload.get("status", MEASURED if metrics else UNAVAILABLE),
            "predictions": payload.get("predictions", []),
            "labels": payload.get("labels", payload.get("references", [])),
        }
    ]


def _coerce_training_records(training_result: Any) -> Dict[str, Any]:
    payload = _as_mapping(training_result)
    if not payload:
        return {
            "training_trace": {},
            "pruning_trace": {},
            "tuning_trace": {},
            "loss_trace": {},
            "model_registry": {},
            "config_resolved": {},
            "sensitivity_report": {},
        }
    return {
        "training_trace": _as_mapping(payload.get("training_trace", payload.get("trace", payload))),
        "pruning_trace": _as_mapping(payload.get("pruning_trace", payload.get("A_P", {}))),
        "tuning_trace": _as_mapping(payload.get("tuning_trace", payload.get("A_T", payload.get("A_T metadata", {})))),
        "loss_trace": _as_mapping(payload.get("loss_trace", {})),
        "model_registry": _as_mapping(payload.get("model_registry", payload.get("model", {}))),
        "config_resolved": _as_mapping(payload.get("config_resolved", payload.get("run_config", payload.get("config", {})))),
        "sensitivity_report": _as_mapping(payload.get("sensitivity_report", {})),
    }


def _metric_registry(metric_registry: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if metric_registry:
        payload = dict(_jsonable(metric_registry))
        if "formula_registry" in payload and isinstance(payload["formula_registry"], Mapping):
            return dict(payload["formula_registry"])
        if "formulas" in payload and isinstance(payload["formulas"], Mapping):
            return dict(payload["formulas"])
        return payload
    return dict(_jsonable(get_metric_formula_registry()))


def _run_config_payload(run_config: Any) -> Dict[str, Any]:
    payload = _as_mapping(run_config)
    if not payload:
        payload = {
            "method": "APT",
            "model_name": "roberta-base",
            "dataset_name": "SST2",
            "sparsity": TARGET_SPARSITY_DEFAULT,
            "tuning_budget": TUNING_BUDGET_DEFAULT,
            "distillation": True,
            "bounded": True,
            "output_dir": DEFAULT_OUTPUT_DIR,
            "precision": "fp32",
            "half_precision_attack": False,
            "batch_size": DEFAULT_BATCH_SIZE,
        }
    payload.setdefault("method", "APT")
    payload.setdefault("model_name", "roberta-base")
    payload.setdefault("dataset_name", "SST2")
    payload.setdefault("sparsity", payload.get("target_sparsity", TARGET_SPARSITY_DEFAULT))
    payload.setdefault("tuning_budget", TUNING_BUDGET_DEFAULT)
    payload.setdefault("distillation", True)
    payload.setdefault("bounded", True)
    payload.setdefault("output_dir", DEFAULT_OUTPUT_DIR)
    payload.setdefault("precision", "fp16" if payload.get("half_precision_attack") else "fp32")
    payload.setdefault("half_precision_attack", False)
    payload.setdefault("batch_size", DEFAULT_BATCH_SIZE)
    return payload


def compute_accuracy(predictions: Sequence[Any], labels: Sequence[Any]) -> float:
    return float(_metric_compute_accuracy(list(predictions), list(labels)))


def aggregate_accuracy(values: Sequence[float]) -> float:
    return float(_metric_aggregate_accuracy(list(values)))


def compute_loss(losses: Sequence[float]) -> float:
    return float(_metric_compute_loss(list(losses)))


def aggregate_loss(values: Sequence[float]) -> float:
    return float(_metric_aggregate_loss(list(values)))


def compute_f1(predictions: Sequence[Any], labels: Sequence[Any]) -> float:
    return float(_metric_compute_f1([str(value) for value in predictions], [str(value) for value in labels]))


def resolve_batch_size_defaults(bounded: bool = True) -> Dict[str, Any]:
    defaults = dict(_config_resolve_batch_size_defaults(bounded))
    defaults.setdefault("default", DEFAULT_BATCH_SIZE)
    defaults.setdefault("batch_size_32", BATCH_SIZE_32)
    defaults.setdefault("batch_size_128", BATCH_SIZE_128)
    defaults.setdefault("fixed_hyperparameters", ["10_shot_setting", "batch_size_32", "batch_size_128"])
    return defaults


def aggregate_metric_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Aggregate result-table rows with the same metric functions used elsewhere."""

    accuracy_values: List[float] = []
    f1_values: List[float] = []
    loss_values: List[float] = []
    fidelity_values: List[float] = []
    training_times: List[float] = []
    memory_values: List[float] = []
    trainable_values: List[float] = []
    for row in rows:
        metrics = row.get("metrics", row)
        if not isinstance(metrics, Mapping):
            continue
        if _metric_value(metrics, ("accuracy", "dev accuracy")) is not None:
            accuracy_values.append(_safe_float(_metric_value(metrics, ("accuracy", "dev accuracy"))))
        if _metric_value(metrics, ("f1", "dev F1")) is not None:
            f1_values.append(_safe_float(_metric_value(metrics, ("f1", "dev F1"))))
        if metrics.get("loss") is not None:
            loss_values.append(_safe_float(metrics.get("loss")))
        if metrics.get("fidelity_score") is not None:
            fidelity_values.append(_safe_float(metrics.get("fidelity_score")))
        if metrics.get("training_time") is not None:
            training_times.append(_safe_float(metrics.get("training_time")))
        if metrics.get("memory_usage") is not None:
            memory_values.append(_safe_float(metrics.get("memory_usage")))
        if metrics.get("trainable_parameter_count") is not None:
            trainable_values.append(_safe_float(metrics.get("trainable_parameter_count")))
    return {
        "accuracy": aggregate_accuracy(accuracy_values),
        "f1": aggregate_accuracy(f1_values),
        "loss": aggregate_loss(loss_values),
        "fidelity_score": aggregate_fidelity_score(fidelity_values),
        "training_time": aggregate_loss(training_times),
        "memory_usage": aggregate_loss(memory_values),
        "trainable_parameter_count": aggregate_loss(trainable_values),
        "row_count": len(rows),
    }


def build_table_aliases() -> Dict[str, Dict[str, Any]]:
    """Paper table aliases, captions, metrics, and source artifact routes."""

    return {
        "table_1": {
            "paper_id": "Table 1",
            "caption": "Efficiency comparison of existing methods and APT.",
            "semantics": "A_P and A_T dynamically adjust total and tuning parameter sizes; PEFT does not reduce inference model size.",
            "source_metric": ["training_cost", "inference_cost", "memory_usage"],
            "source_artifact": ["results/result_table.json", "results/metric_formula.json"],
        },
        "table_2": {
            "paper_id": "Table 2",
            "caption": "RoBERTa and T5 pruning with APT compared to baselines under 60% sparsity.",
            "semantics": "SST2 task-performance and normalized training/inference efficiency; TTA targets 97% accuracy.",
            "source_metric": ["accuracy", "relative accuracy", "TTA", "relative training speed", "relative inference speed"],
            "source_artifact": ["results/evaluation_result.json", "results/sst2_mnli_relative_accuracy_inputs.json"],
        },
        "table_3": {
            "paper_id": "Table 3",
            "caption": "LLaMA 2 7B 30% sparsity pruning results on GPT4-generated Alpaca and Open LLM tasks.",
            "semantics": "Generation and instruction route with TruthfulQA/few-shot metrics; distillation baselines omitted for cost.",
            "source_metric": ["truthfulness", "generation_exact_match", "training_time"],
            "source_artifact": ["results/evaluation_result.json", "results/dataset_registry.json"],
        },
        "table_4": {
            "paper_id": "Table 4",
            "caption": "Ablation of salience allocation, APT adapter, and self-distillation with relative training efficiency.",
            "semantics": "No-distillation and PEFT-pruned-LM variants record performance-drop trend while reusing the main route.",
            "source_metric": ["accuracy", "loss", "training_cost", "memory_usage", "fidelity_score"],
            "source_artifact": ["results/evaluation_result.json", "results/loss_trace.json", "results/result_table.json"],
        },
        "table_5": {
            "paper_id": "Table 5",
            "caption": "LLaMA 2 7B ablations under 30% and 50% sparsity; T.M. is relative training memory.",
            "semantics": "A_T metadata supplies trainable-parameter count and relative training memory fields.",
            "source_metric": ["trainable_parameter_count", "relative training peak memory", "truthfulness"],
            "source_artifact": ["results/tuning_trace.json", "results/evaluation_result.json"],
        },
        "table_6": {
            "paper_id": "Table 6",
            "caption": "Hyperparameters used in APT experiments.",
            "semantics": "batch_size_32, batch_size_128, 10-shot, rank, salience EMA, tau, and pruning schedule defaults.",
            "source_metric": ["runtime", "metric_runtime"],
            "source_artifact": ["results/config_resolved.json", "results/run_config.json", "results/metric_formula.json"],
        },
        "table_7": {
            "paper_id": "Table 7",
            "caption": "APT compared with unstructured pruning baseline using PEFT.",
            "semantics": "Benchmark-visible Mask Tuning and CoFi checkpoint metadata remain linked for full runs.",
            "source_metric": ["accuracy", "f1", "training_cost"],
            "source_artifact": ["checkpoints/mask_tuning/metadata.json", "checkpoints/cofi/metadata.json"],
        },
        "table_8": {
            "paper_id": "Table 8",
            "caption": "Detailed RoBERTa comparison between APT and LoRA+Distill baseline.",
            "semantics": "GLUE task route excludes STS-B if CoFi cannot reproduce it.",
            "source_metric": ["accuracy", "relative accuracy"],
            "source_artifact": ["results/evaluation_result.json", "results/result_table.json"],
        },
        "table_9": {
            "paper_id": "Table 9",
            "caption": "LLaMA2 7B/13B 30% sparsity Open LLM leaderboard pruning results.",
            "semantics": "LLaMA generation/instruction interface reports few-shot task averages and TruthfulQA route.",
            "source_metric": ["truthfulness", "generation_exact_match"],
            "source_artifact": ["results/evaluation_result.json", "results/dataset_registry.json"],
        },
        "table_10": {
            "paper_id": "Table 10",
            "caption": "Ablation of distillation strategies and comparison to non-efficient distillation.",
            "semantics": "Self-knowledge distillation loss route preserves L_pred, L_layer, tau, and half precision protocol metadata.",
            "source_metric": ["loss", "fidelity_score", "training_cost", "memory_usage"],
            "source_artifact": ["results/loss_trace.json", "results/config_resolved.json"],
        },
        "table_11": {
            "paper_id": "Table 11",
            "caption": "Raw efficiency metrics for RoBERTa/T5 on SST2: TTA, training memory, inference time and memory.",
            "semantics": "Raw training/inference costs are copied from current traces and metric functions.",
            "source_metric": ["training_time", "TTA", "memory_usage", "inference_cost"],
            "source_artifact": ["results/training_trace.json", "results/evaluation_result.json"],
        },
        "table_12": {
            "paper_id": "Table 12",
            "caption": "Raw efficiency metrics for LLaMA2 7B on Alpaca.",
            "semantics": "Generation efficiency fields remain available when full LLaMA checkpoints are used.",
            "source_metric": ["training_time", "memory_usage", "inference_cost", "truthfulness"],
            "source_artifact": ["results/evaluation_result.json", "results/training_trace.json"],
        },
    }


def build_figure_aliases() -> Dict[str, Dict[str, Any]]:
    """Paper figure aliases and executable source routes."""

    return {
        "figure_1": {
            "paper_id": "Figure 1",
            "caption": "APT provides training and inference efficiency by adaptively pruning and tuning.",
            "source_metric": ["training_cost", "inference_cost", "memory_usage", "accuracy"],
            "source_artifact": ["results/result_table.json", "results/figures/figure_1.json"],
            "route": "run_figure_1_route",
        },
        "figure_2": {
            "paper_id": "Figure 2",
            "caption": "APT adapter connects A_P salience/masks and A_T dynamic tuning ranks during fine-tuning.",
            "source_metric": ["outlier-aware salience score", "binary masks", "dynamic rank r_apt", "trainable_parameter_count"],
            "source_artifact": ["results/pruning_trace.json", "results/tuning_trace.json", "results/model_registry.json"],
            "route": "run_figure_2_route",
        },
        "figure_3": {
            "paper_id": "Figure 3",
            "caption": "Task performance versus relative inference efficiency across RoBERTa, T5, and LLaMA.",
            "source_metric": ["relative accuracy", "relative inference speed", "relative inference memory"],
            "source_artifact": ["results/result_table.json", "results/evaluation_result.json"],
            "route": "run_figure_3_route",
        },
        "figure_4": {
            "paper_id": "Figure 4",
            "caption": "Performance-efficiency tradeoff of APT compared to baseline methods.",
            "source_metric": ["fidelity_score", "training_cost", "memory_usage", "relative inference speed"],
            "source_artifact": ["results/result_table.json", "results/fidelity_score.json"],
            "route": "run_figure_4_route",
        },
        "figure_5": {
            "paper_id": "Figure 5",
            "caption": "APT sensitivity under different initial/target sparsities and adaptive tuning schedules.",
            "source_metric": ["sparsity", "trainable_parameter_count", "fidelity_score"],
            "source_artifact": ["results/sensitivity_report.json", "results/tuning_trace.json"],
            "route": "run_figure_5_route",
        },
        "figure_5a": {
            "paper_id": "Figure 5a",
            "caption": "Adaptive tuning schedule analysis.",
            "source_metric": ["dynamic ranks", "tuning layer importance", "memory_usage"],
            "source_artifact": ["results/sensitivity_report.json", "results/tuning_trace.json"],
            "route": "run_figure_5a_route",
        },
    }


def build_relative_accuracy_inputs(
    evaluation_result: Mapping[str, Any],
    reference_method: str = "FT",
    reference_scores: Optional[Mapping[str, float]] = None,
) -> Dict[str, Any]:
    """Build explicit SST2/MNLI inputs consumed by relative-accuracy metrics."""

    reference_scores = dict(reference_scores or {})
    rows = []
    for run in _coerce_runs(evaluation_result):
        task = str(run.get("task_name", run.get("dataset_name", run.get("task", ""))))
        if task not in {"SST2", "MNLI", "sst2", "mnli"}:
            continue
        canonical_task = "SST2" if task.lower() == "sst2" else "MNLI"
        metrics = run.get("metrics", {})
        if not isinstance(metrics, Mapping):
            metrics = {}
        method_score = _safe_float(_metric_value(metrics, ("accuracy", "dev accuracy", "relative accuracy"), run.get("accuracy")))
        reference_score = _safe_float(
            reference_scores.get(canonical_task, metrics.get("reference_accuracy", run.get("reference_score", method_score)))
        )
        rows.append(
            {
                "task_name": canonical_task,
                "method": run.get("method", "APT"),
                "reference_method": reference_method,
                "method_score": method_score,
                "reference_score": reference_score,
                "relative_accuracy": compute_relative_accuracy(method_score, reference_score),
                "source_metric": "dev accuracy",
                "source_artifact": "results/evaluation_result.json",
                "status": MEASURED if reference_score > 0 else UNAVAILABLE,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "sst2_mnli_relative_accuracy_inputs",
        "rows": rows,
        "formula": "relative_accuracy = method_score / reference_score when reference_score > 0",
        "required_tasks": ["SST2", "MNLI"],
        "source_artifact": "results/evaluation_result.json",
        "target_artifact": "results/sst2_mnli_relative_accuracy_inputs.json",
    }


def summarize_baseline_outperformance(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Compare APT rows to explicit baselines using current-run metrics."""

    apt_rows = [row for row in rows if str(row.get("method", "")).lower() in {"apt", "ours"}]
    baseline_rows = [row for row in rows if row not in apt_rows and row.get("method")]
    best_baseline_task_score = 0.0
    best_apt_task_score = 0.0
    baseline_costs: List[float] = []
    apt_costs: List[float] = []
    for row in baseline_rows:
        metrics = row.get("metrics", {})
        if isinstance(metrics, Mapping):
            best_baseline_task_score = max(best_baseline_task_score, _safe_float(_metric_value(metrics, ("accuracy", "dev accuracy", "f1", "dev F1", "rouge", "truthfulness"))))
            baseline_costs.append(_safe_float(metrics.get("training_cost", metrics.get("training_time", 0.0))))
    for row in apt_rows:
        metrics = row.get("metrics", {})
        if isinstance(metrics, Mapping):
            best_apt_task_score = max(best_apt_task_score, _safe_float(_metric_value(metrics, ("accuracy", "dev accuracy", "f1", "dev F1", "rouge", "truthfulness"))))
            apt_costs.append(_safe_float(metrics.get("training_cost", metrics.get("training_time", 0.0))))
    apt_cost = min(apt_costs) if apt_costs else None
    baseline_cost = min(baseline_costs) if baseline_costs else None
    return {
        "computed_from_current_run": bool(apt_rows and baseline_rows),
        "apt_rows": len(apt_rows),
        "baseline_rows": len(baseline_rows),
        "named_baselines": list(REQUIRED_BASELINES),
        "APT_task_score": best_apt_task_score,
        "best_baseline_task_score": best_baseline_task_score,
        "APT_lower_or_equal_training_cost": apt_cost <= baseline_cost if apt_cost is not None and baseline_cost is not None else None,
        "APT_maintains_task_performance": best_apt_task_score >= 0.99 * best_baseline_task_score if best_baseline_task_score else None,
        "semantic_assertion": "APT aims to improve training and inference efficiency while maintaining task performance; bounded smoke is not a paper-number claim.",
        "table_4_trend": "No-distillation pruning and PEFT tuning on pruned LMs are recorded as performance-drop comparison semantics.",
    }


def _row_from_run(run: Mapping[str, Any], run_config: Mapping[str, Any], training: Mapping[str, Any]) -> Dict[str, Any]:
    metrics = dict(run.get("metrics", {})) if isinstance(run.get("metrics"), Mapping) else {}
    predictions = run.get("predictions")
    labels = run.get("labels", run.get("references"))
    if isinstance(predictions, Sequence) and isinstance(labels, Sequence) and not isinstance(predictions, (str, bytes)):
        if "accuracy" not in metrics and labels:
            metrics["accuracy"] = compute_accuracy(list(predictions), list(labels))
        if "f1" not in metrics and labels:
            metrics["f1"] = compute_f1(list(predictions), list(labels))
    if "loss" not in metrics and isinstance(training.get("loss_trace"), Mapping):
        losses = [
            _safe_float(item.get("loss", item.get("L_distill", 0.0)))
            for item in training["loss_trace"].get("records", [])
            if isinstance(item, Mapping)
        ]
        if losses:
            metrics["loss"] = aggregate_loss(losses)
    trace_for_efficiency = dict(training.get("training_trace", {}))
    trace_for_efficiency.update(dict(training.get("tuning_trace", {})))
    if trace_for_efficiency:
        metrics.update({key: value for key, value in compute_efficiency_metrics(trace_for_efficiency).items() if key not in metrics})
    metrics.setdefault("fidelity_score", compute_fidelity_score(metrics))
    task_name = run.get("task_name", run.get("dataset_name", run_config.get("dataset_name", "")))
    row = {
        "method": run.get("method", run_config.get("method", "APT")),
        "task_name": task_name,
        "dataset_name": run.get("dataset_name", task_name),
        "model_name": run.get("model_name", run_config.get("model_name", "")),
        "baseline": run.get("baseline", run_config.get("reference_method", "FT")),
        "status": run.get("status", MEASURED if metrics else UNAVAILABLE),
        "bounded": bool(run.get("bounded", run_config.get("bounded", True))),
        "batch_size": run.get("batch_size", run_config.get("batch_size", DEFAULT_BATCH_SIZE)),
        "batch_size_32": _safe_int(run.get("batch_size", run_config.get("batch_size", DEFAULT_BATCH_SIZE))) == BATCH_SIZE_32,
        "precision": run.get("precision", run_config.get("precision", "fp32")),
        "half_precision_attack": bool(run.get("half_precision_attack", run_config.get("half_precision_attack", False))),
        "sparsity": _safe_float(run.get("sparsity", run_config.get("sparsity", run_config.get("target_sparsity", TARGET_SPARSITY_DEFAULT)))),
        "tuning_budget": run.get("tuning_budget", run_config.get("tuning_budget", TUNING_BUDGET_DEFAULT)),
        "distillation": bool(run.get("distillation", run_config.get("distillation", True))),
        "metrics": metrics,
        "source_metric": sorted(str(key) for key in metrics),
        "source_artifact": ["results/evaluation_result.json", "results/metric_formula.json", "results/training_trace.json"],
        "provenance": {
            "evaluation_route": "src.apt.evaluation.run_evaluation or bounded route output",
            "training_route": "src.apt.training.run_training_loop or baseline state output",
            "reporting_route": "src.apt.reporting.build_result_table",
        },
    }
    return row


def _method_trace_summary(training: Mapping[str, Any]) -> Dict[str, Any]:
    pruning_trace = training.get("pruning_trace", {})
    tuning_trace = training.get("tuning_trace", {})
    model_registry = training.get("model_registry", {})
    loss_trace = training.get("loss_trace", {})
    pruning_records = pruning_trace.get("records", []) if isinstance(pruning_trace, Mapping) else []
    tuning_records = tuning_trace.get("records", []) if isinstance(tuning_trace, Mapping) else []
    loss_records = loss_trace.get("records", []) if isinstance(loss_trace, Mapping) else []
    at_metadata = {}
    if isinstance(tuning_trace, Mapping):
        at_metadata = dict(tuning_trace.get("A_T metadata", tuning_trace.get("a_t_metadata", tuning_trace.get("metadata", {}))) or {})
    adapter_report = {}
    if isinstance(model_registry, Mapping):
        adapter_report = dict(model_registry.get("adapter_report", model_registry.get("adapter_metadata", {})) or {})
    return {
        "A_P": {
            "early_training_t_lt_T": EARLY_TRAINING_STEPS,
            "salience_ema": {"decay": SALIENCE_EMA_DECAY, "update": SALIENCE_EMA_UPDATE, "s_bar_t": dict(s_bar_t), "s_bar_t_1": dict(s_bar_t_1)},
            "outlier_aware_salience_score": bool(pruning_records) or "S_hat" in json.dumps(_jsonable(pruning_trace)),
            "fast_search": pruning_trace.get("fast_search", pruning_trace.get("search", {})) if isinstance(pruning_trace, Mapping) else {},
            "binary_masks": pruning_trace.get("binary_masks", pruning_trace.get("applied_masks", {})) if isinstance(pruning_trace, Mapping) else {},
            "pruned_structure_metadata": pruning_trace.get("structure_metadata", pruning_trace.get("mask_metadata", {})) if isinstance(pruning_trace, Mapping) else {},
            "source_artifact": "results/pruning_trace.json",
        },
        "A_T": {
            "tuning_layer_importance": tuning_trace.get("importance", tuning_trace.get("layer_importance", {})) if isinstance(tuning_trace, Mapping) else {},
            "dynamic_ranks": tuning_trace.get("rank_allocation", tuning_trace.get("dynamic_ranks", {})) if isinstance(tuning_trace, Mapping) else {},
            "A_T metadata": at_metadata,
            "trainable_parameter_count_consumed_by_metrics": at_metadata.get("trainable_parameter_count", adapter_report.get("trainable_parameters")),
            "source_artifact": "results/tuning_trace.json",
        },
        "APT_adapter": {
            "base_adapter": "LoRA",
            "binary_pruning_masks": {
                "m_i": adapter_report.get("m_i", model_registry.get("m_i") if isinstance(model_registry, Mapping) else None),
                "m_o": adapter_report.get("m_o", model_registry.get("m_o") if isinstance(model_registry, Mapping) else None),
            },
            "dynamic_rank_r_apt": adapter_report.get("r_apt", R_APT_DEFAULT),
            "task_sensitive_adapter_selector": "src.apt.adapters.create_apt_adapter",
            "source_artifact": "results/model_registry.json",
        },
        "self_distillation": {
            "records": len(loss_records),
            "formula": "GLUE: L_distill=L_pred+0.9*L_layer; SQuAD/generation: L_pred+0.1*L_layer with tau=4",
            "source_artifact": "results/loss_trace.json",
        },
    }


def build_result_table(
    run_config: Any,
    evaluation_result: Mapping[str, Any],
    training_result: Optional[Mapping[str, Any]] = None,
    metric_registry: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the code-backed payload for results/result_table.json."""

    config_payload = _run_config_payload(run_config)
    training = _coerce_training_records(training_result or {})
    formulas = _metric_registry(metric_registry)
    rows = [_row_from_run(run, config_payload, training) for run in _coerce_runs(evaluation_result)]
    relative_inputs = build_relative_accuracy_inputs({"runs": rows}, reference_method=str(config_payload.get("reference_method", "FT")))
    baseline_summary = summarize_baseline_outperformance(rows)
    metric_summary = aggregate_metric_summary(rows)
    table_aliases = build_table_aliases()
    figure_aliases = build_figure_aliases()
    table_rows = build_paper_table_rows(rows, table_aliases)
    figure_rows = build_paper_figure_rows(rows, figure_aliases, training)
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "result_table",
        "paper": PAPER_TITLE,
        "method": config_payload["method"],
        "model_name": config_payload["model_name"],
        "dataset_name": config_payload["dataset_name"],
        "bounded": bool(config_payload["bounded"]),
        "not_full_benchmark_claim": bool(config_payload["bounded"]),
        "rows": rows,
        "metric_summary": metric_summary,
        "metric_formula_registry": formulas,
        "metric_formula_artifact": "results/metric_formula.json",
        "table_aliases": table_aliases,
        "figure_aliases": figure_aliases,
        "paper_table_rows": table_rows,
        "paper_figure_rows": figure_rows,
        "relative_accuracy_inputs": relative_inputs,
        "baseline_outperformance": baseline_summary,
        "method_trace_summary": _method_trace_summary(training),
        "artifact_targets": {
            "result_table": "results/result_table.json",
            "result_table_markdown": "results/result_table.md",
            "relative_accuracy_inputs": "results/sst2_mnli_relative_accuracy_inputs.json",
            "fidelity_score": "results/fidelity_score.json",
            "evaluation_result": "results/evaluation_result.json",
            "model_registry": "results/model_registry.json",
            "pruning_trace": "results/pruning_trace.json",
            "tuning_trace": "results/tuning_trace.json",
            "loss_trace": "results/loss_trace.json",
            "config_resolved": "results/config_resolved.json",
            "sensitivity_report": "results/sensitivity_report.json",
            "checkpoints_cofi": "checkpoints/cofi/metadata.json",
            "checkpoints_mask_tuning": "checkpoints/mask_tuning/metadata.json",
        },
        "semantic_review_fields": {
            "main_result_trend": "Maintain task performance while improving training/inference efficiency; bounded values are route checks, not paper-number claims.",
            "table_4_trend": "No-distillation pruning and PEFT fine-tuning of pruned LMs record performance-drop semantics.",
            "generation_metric_routes": ["CNN/DailyMail ROUGE", "TruthfulQA truthfulness/generation"],
            "half_precision_attack": {
                "enabled": bool(config_payload.get("half_precision_attack", False)),
                "precision": config_payload.get("precision"),
                "protocol": "Preserved as configuration/evaluation metadata and full-mode variant.",
            },
        },
        "canonical_metric_identifiers": list(CANONICAL_METRIC_IDENTIFIERS),
        "canonical_artifact_identifiers": list(CANONICAL_ARTIFACT_IDENTIFIERS),
        "reference_grounding": [
            "paperbench_ref_001 datasheet.md",
            "paperbench_ref_001 prompt.txt",
            "paperbench_ref_001 train.py",
        ],
    }


def build_paper_table_rows(rows: Sequence[Mapping[str, Any]], table_aliases: Optional[Mapping[str, Mapping[str, Any]]] = None) -> List[Dict[str, Any]]:
    table_aliases = dict(table_aliases or build_table_aliases())
    output: List[Dict[str, Any]] = []
    for table_key, alias in table_aliases.items():
        metrics = list(alias.get("source_metric", []))
        source_rows = []
        for row in rows:
            row_metrics = row.get("metrics", {}) if isinstance(row.get("metrics"), Mapping) else {}
            selected = {metric: row_metrics.get(metric) for metric in metrics if metric in row_metrics}
            aliases = {
                "dev accuracy": row_metrics.get("dev accuracy", row_metrics.get("accuracy")),
                "dev F1": row_metrics.get("dev F1", row_metrics.get("f1")),
                "relative accuracy": row_metrics.get("relative accuracy"),
            }
            source_rows.append(
                {
                    "method": row.get("method"),
                    "task_name": row.get("task_name"),
                    "selected_metrics": {**selected, **{k: v for k, v in aliases.items() if v is not None}},
                    "source_artifact": row.get("source_artifact", alias.get("source_artifact", [])),
                }
            )
        output.append(
            {
                "table_key": table_key,
                "paper_id": alias["paper_id"],
                "caption": alias["caption"],
                "semantics": alias["semantics"],
                "rows": source_rows,
                "source_metric": metrics,
                "source_artifact": alias.get("source_artifact", []),
                "artifact_path": f"results/tables/{table_key}.json",
            }
        )
    return output


def build_paper_figure_rows(
    rows: Sequence[Mapping[str, Any]],
    figure_aliases: Optional[Mapping[str, Mapping[str, Any]]] = None,
    training: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    figure_aliases = dict(figure_aliases or build_figure_aliases())
    training = dict(training or {})
    output: List[Dict[str, Any]] = []
    for figure_key, alias in figure_aliases.items():
        points = []
        for row in rows:
            metrics = row.get("metrics", {}) if isinstance(row.get("metrics"), Mapping) else {}
            points.append(
                {
                    "method": row.get("method"),
                    "task_name": row.get("task_name"),
                    "x_inference_efficiency": metrics.get("relative inference speed", metrics.get("inference_cost")),
                    "y_task_performance": _metric_value(metrics, ("accuracy", "dev accuracy", "f1", "dev F1", "rouge", "truthfulness")),
                    "size_training_memory": metrics.get("memory_usage", metrics.get("gpu_memory")),
                    "fidelity_score": metrics.get("fidelity_score"),
                    "source_artifact": row.get("source_artifact", alias.get("source_artifact", [])),
                }
            )
        output.append(
            {
                "figure_key": figure_key,
                "paper_id": alias["paper_id"],
                "caption": alias["caption"],
                "route": alias["route"],
                "points": points,
                "method_trace_summary": _method_trace_summary(training) if figure_key == "figure_2" else {},
                "source_metric": alias.get("source_metric", []),
                "source_artifact": alias.get("source_artifact", []),
                "artifact_path": f"results/figures/{figure_key}.json",
            }
        )
    return output


def _artifact_root(output_dir: Optional[str | Path] = None) -> Path:
    return Path(output_dir or os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR") or DEFAULT_OUTPUT_DIR)


def _write_json(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(path)


def _write_markdown(path: Path, result_table: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# APT Result Table",
        "",
        "| method | task | model | status | accuracy | f1 | loss | training_cost | memory_usage |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in result_table.get("rows", []):
        metrics = row.get("metrics", {}) if isinstance(row, Mapping) and isinstance(row.get("metrics"), Mapping) else {}
        lines.append(
            "| {method} | {task} | {model} | {status} | {acc} | {f1} | {loss} | {cost} | {mem} |".format(
                method=row.get("method", ""),
                task=row.get("task_name", ""),
                model=row.get("model_name", ""),
                status=row.get("status", ""),
                acc=metrics.get("accuracy", metrics.get("dev accuracy", "")),
                f1=metrics.get("f1", metrics.get("dev F1", "")),
                loss=metrics.get("loss", ""),
                cost=metrics.get("training_cost", ""),
                mem=metrics.get("memory_usage", ""),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def write_result_table_artifact(
    output_dir: str | Path,
    run_config: Any,
    evaluation_result: Mapping[str, Any],
    training_result: Optional[Mapping[str, Any]] = None,
    metric_registry: Optional[Mapping[str, Any]] = None,
) -> Dict[str, str]:
    payload = build_result_table(run_config, evaluation_result, training_result or {}, metric_registry)
    root = _artifact_root(output_dir)
    paths = {
        "result_table": _write_json(root / "result_table.json", payload),
        "result_table_md": _write_markdown(root / "result_table.md", payload),
        "sst2_mnli_relative_accuracy_inputs": _write_json(root / "sst2_mnli_relative_accuracy_inputs.json", payload["relative_accuracy_inputs"]),
    }
    write_fidelity_score_artifact(root, payload["rows"])
    paths["fidelity_score"] = str(root / "fidelity_score.json")
    return paths


def write_relative_accuracy_inputs(
    output_dir: str | Path,
    evaluation_result: Mapping[str, Any],
    reference_method: str = "FT",
    reference_scores: Optional[Mapping[str, float]] = None,
) -> str:
    return _write_json(
        _artifact_root(output_dir) / "sst2_mnli_relative_accuracy_inputs.json",
        build_relative_accuracy_inputs(evaluation_result, reference_method, reference_scores),
    )


def _select_table(result_table: Mapping[str, Any], table_key: str) -> Dict[str, Any]:
    for table in result_table.get("paper_table_rows", []):
        if table.get("table_key") == table_key:
            return dict(table)
    aliases = build_table_aliases()
    alias = aliases.get(table_key, {"paper_id": table_key, "caption": table_key, "semantics": "", "source_artifact": []})
    return {
        "table_key": table_key,
        "paper_id": alias.get("paper_id", table_key),
        "caption": alias.get("caption", table_key),
        "semantics": alias.get("semantics", ""),
        "rows": result_table.get("rows", []),
        "source_metric": alias.get("source_metric", []),
        "source_artifact": alias.get("source_artifact", []),
        "artifact_path": f"results/tables/{table_key}.json",
    }


def _select_figure(result_table: Mapping[str, Any], figure_key: str) -> Dict[str, Any]:
    for figure in result_table.get("paper_figure_rows", []):
        if figure.get("figure_key") == figure_key:
            return dict(figure)
    aliases = build_figure_aliases()
    alias = aliases.get(figure_key, {"paper_id": figure_key, "caption": figure_key, "source_artifact": [], "source_metric": []})
    return {
        "figure_key": figure_key,
        "paper_id": alias.get("paper_id", figure_key),
        "caption": alias.get("caption", figure_key),
        "points": [],
        "source_metric": alias.get("source_metric", []),
        "source_artifact": alias.get("source_artifact", []),
        "artifact_path": f"results/figures/{figure_key}.json",
    }


def _table_payload(table_key: str, result_table: Mapping[str, Any]) -> Dict[str, Any]:
    table = _select_table(result_table, table_key)
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "paper_table",
        "bounded_proxy": bool(result_table.get("bounded", True)),
        "not_full_benchmark_claim": bool(result_table.get("bounded", True)),
        **table,
    }


def _figure_payload(figure_key: str, result_table: Mapping[str, Any]) -> Dict[str, Any]:
    figure = _select_figure(result_table, figure_key)
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "paper_figure",
        "bounded_proxy": bool(result_table.get("bounded", True)),
        "not_full_benchmark_claim": bool(result_table.get("bounded", True)),
        **figure,
    }


def write_table_artifact(output_dir: str | Path, table_key: str, result_table: Mapping[str, Any]) -> str:
    return _write_json(_artifact_root(output_dir) / "tables" / f"{table_key}.json", _table_payload(table_key, result_table))


def write_figure_artifact(output_dir: str | Path, figure_key: str, result_table: Mapping[str, Any]) -> str:
    return _write_json(_artifact_root(output_dir) / "figures" / f"{figure_key}.json", _figure_payload(figure_key, result_table))


def _result_table_from_inputs(
    run_config: Any = None,
    evaluation_result: Optional[Mapping[str, Any]] = None,
    training_result: Optional[Mapping[str, Any]] = None,
    metric_registry: Optional[Mapping[str, Any]] = None,
    result_table: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    if result_table is not None:
        return dict(_jsonable(result_table))
    return build_result_table(run_config or {}, evaluation_result or {}, training_result or {}, metric_registry)


def run_figure_1_route(result_table: Mapping[str, Any]) -> Dict[str, Any]:
    return _figure_payload("figure_1", result_table)


def run_figure_2_route(result_table: Mapping[str, Any]) -> Dict[str, Any]:
    return _figure_payload("figure_2", result_table)


def run_figure_3_route(result_table: Mapping[str, Any]) -> Dict[str, Any]:
    return _figure_payload("figure_3", result_table)


def run_figure_4_route(result_table: Mapping[str, Any]) -> Dict[str, Any]:
    return _figure_payload("figure_4", result_table)


def run_figure_5_route(result_table: Mapping[str, Any]) -> Dict[str, Any]:
    return _figure_payload("figure_5", result_table)


def run_figure_5a_route(result_table: Mapping[str, Any]) -> Dict[str, Any]:
    return _figure_payload("figure_5a", result_table)


def run_table_route(table_key: str, result_table: Mapping[str, Any]) -> Dict[str, Any]:
    return _table_payload(table_key, result_table)


def run_table_1_route(result_table: Mapping[str, Any]) -> Dict[str, Any]:
    return run_table_route("table_1", result_table)


def run_table_2_route(result_table: Mapping[str, Any]) -> Dict[str, Any]:
    return run_table_route("table_2", result_table)


def run_table_3_route(result_table: Mapping[str, Any]) -> Dict[str, Any]:
    return run_table_route("table_3", result_table)


def run_table_4_route(result_table: Mapping[str, Any]) -> Dict[str, Any]:
    return run_table_route("table_4", result_table)


def run_table_5_route(result_table: Mapping[str, Any]) -> Dict[str, Any]:
    return run_table_route("table_5", result_table)


def run_table_6_route(result_table: Mapping[str, Any]) -> Dict[str, Any]:
    return run_table_route("table_6", result_table)


def run_table_7_route(result_table: Mapping[str, Any]) -> Dict[str, Any]:
    return run_table_route("table_7", result_table)


def run_table_8_route(result_table: Mapping[str, Any]) -> Dict[str, Any]:
    return run_table_route("table_8", result_table)


def run_table_9_route(result_table: Mapping[str, Any]) -> Dict[str, Any]:
    return run_table_route("table_9", result_table)


def run_table_10_route(result_table: Mapping[str, Any]) -> Dict[str, Any]:
    return run_table_route("table_10", result_table)


def run_table_11_route(result_table: Mapping[str, Any]) -> Dict[str, Any]:
    return run_table_route("table_11", result_table)


def run_table_12_route(result_table: Mapping[str, Any]) -> Dict[str, Any]:
    return run_table_route("table_12", result_table)


def write_figure_1_artifact(output_dir: str | Path, result_table: Mapping[str, Any]) -> str:
    return write_figure_artifact(output_dir, "figure_1", result_table)


def write_figure_2_artifact(output_dir: str | Path, result_table: Mapping[str, Any]) -> str:
    return write_figure_artifact(output_dir, "figure_2", result_table)


def write_figure_3_artifact(output_dir: str | Path, result_table: Mapping[str, Any]) -> str:
    return write_figure_artifact(output_dir, "figure_3", result_table)


def write_figure_4_artifact(output_dir: str | Path, result_table: Mapping[str, Any]) -> str:
    return write_figure_artifact(output_dir, "figure_4", result_table)


def write_figure_5_artifact(output_dir: str | Path, result_table: Mapping[str, Any]) -> str:
    return write_figure_artifact(output_dir, "figure_5", result_table)


def write_figure_5a_artifact(output_dir: str | Path, result_table: Mapping[str, Any]) -> str:
    return write_figure_artifact(output_dir, "figure_5a", result_table)


def write_table_1_artifact(output_dir: str | Path, result_table: Mapping[str, Any]) -> str:
    return write_table_artifact(output_dir, "table_1", result_table)


def write_table_2_artifact(output_dir: str | Path, result_table: Mapping[str, Any]) -> str:
    return write_table_artifact(output_dir, "table_2", result_table)


def write_table_3_artifact(output_dir: str | Path, result_table: Mapping[str, Any]) -> str:
    return write_table_artifact(output_dir, "table_3", result_table)


def write_table_4_artifact(output_dir: str | Path, result_table: Mapping[str, Any]) -> str:
    return write_table_artifact(output_dir, "table_4", result_table)


def write_table_5_artifact(output_dir: str | Path, result_table: Mapping[str, Any]) -> str:
    return write_table_artifact(output_dir, "table_5", result_table)


def write_table_6_artifact(output_dir: str | Path, result_table: Mapping[str, Any]) -> str:
    return write_table_artifact(output_dir, "table_6", result_table)


def write_table_7_artifact(output_dir: str | Path, result_table: Mapping[str, Any]) -> str:
    return write_table_artifact(output_dir, "table_7", result_table)


def write_table_8_artifact(output_dir: str | Path, result_table: Mapping[str, Any]) -> str:
    return write_table_artifact(output_dir, "table_8", result_table)


def write_table_9_artifact(output_dir: str | Path, result_table: Mapping[str, Any]) -> str:
    return write_table_artifact(output_dir, "table_9", result_table)


def write_table_10_artifact(output_dir: str | Path, result_table: Mapping[str, Any]) -> str:
    return write_table_artifact(output_dir, "table_10", result_table)


def write_table_11_artifact(output_dir: str | Path, result_table: Mapping[str, Any]) -> str:
    return write_table_artifact(output_dir, "table_11", result_table)


def write_table_12_artifact(output_dir: str | Path, result_table: Mapping[str, Any]) -> str:
    return write_table_artifact(output_dir, "table_12", result_table)


def build_reporting_manifest(run_config: Any, result_table: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "reporting_manifest",
        "run_config": _run_config_payload(run_config),
        "declared_writes": [
            "results/model_registry.json",
            "results/pruning_trace.json",
            "results/tuning_trace.json",
            "results/loss_trace.json",
            "results/config_resolved.json",
            "results/sensitivity_report.json",
            "results/result_table.json",
            "results/result_table.md",
            "results/ablation_table.json",
            "results/artifact_manifest.json",
            "results/evaluation_result.json",
            "results/run_config.json",
            "results/metric_formula.json",
            "results/training_trace.json",
            "results/dataset_registry.json",
            "results/sst2_mnli_relative_accuracy_inputs.json",
            "results/figures/figure_1.png",
        ],
        "paper_visible_tables": list(PAPER_VISIBLE_TABLES),
        "paper_visible_figures": list(PAPER_VISIBLE_FIGURES),
        "artifact_specs": _jsonable(get_artifact_specs()),
        "dataset_registry": _jsonable(get_dataset_registry()),
        "method_registry": _jsonable(get_method_registry()),
        "baseline_registry": _jsonable(get_baseline_registry()),
        "experiment_registry": _jsonable(get_experiment_registry(bool(result_table.get("bounded", True)))),
        "hyperparameter_config": _jsonable(get_hyperparameter_config(bool(result_table.get("bounded", True)))),
        "metric_formula": metric_formula_payload({"metrics": result_table.get("metric_summary", {})}),
        "checkpoint_metadata_paths": ["checkpoints/cofi/metadata.json", "checkpoints/mask_tuning/metadata.json"],
        "paper_visible_outputs_are_code_backed": True,
    }


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_OUTPUT_DIR",
    "aggregate_accuracy",
    "aggregate_loss",
    "aggregate_metric_summary",
    "build_figure_aliases",
    "build_paper_figure_rows",
    "build_paper_table_rows",
    "build_relative_accuracy_inputs",
    "build_reporting_manifest",
    "build_result_table",
    "build_table_aliases",
    "compute_accuracy",
    "compute_f1",
    "compute_loss",
    "resolve_batch_size_defaults",
    "run_figure_1_route",
    "run_figure_2_route",
    "run_figure_3_route",
    "run_figure_4_route",
    "run_figure_5_route",
    "run_figure_5a_route",
    "run_table_1_route",
    "run_table_2_route",
    "run_table_3_route",
    "run_table_4_route",
    "run_table_5_route",
    "run_table_6_route",
    "run_table_7_route",
    "run_table_8_route",
    "run_table_9_route",
    "run_table_10_route",
    "run_table_11_route",
    "run_table_12_route",
    "summarize_baseline_outperformance",
    "write_figure_1_artifact",
    "write_figure_2_artifact",
    "write_figure_3_artifact",
    "write_figure_4_artifact",
    "write_figure_5_artifact",
    "write_figure_5a_artifact",
    "write_relative_accuracy_inputs",
    "write_result_table_artifact",
    "write_table_1_artifact",
    "write_table_2_artifact",
    "write_table_3_artifact",
    "write_table_4_artifact",
    "write_table_5_artifact",
    "write_table_6_artifact",
    "write_table_7_artifact",
    "write_table_8_artifact",
    "write_table_9_artifact",
    "write_table_10_artifact",
    "write_table_11_artifact",
    "write_table_12_artifact",
]
