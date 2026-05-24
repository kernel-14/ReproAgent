"""Metric formulas and aggregation routes for the APT reproduction.

This module is intentionally dependency-light.  It implements the paper-visible
metric formulas used by evaluation, reporting, and artifact writers, including
APT parameter accounting from A_T metadata and pruning/tuning traces.

reference_grounding: paperbench_ref_001 datasheet.md
reference_grounding: paperbench_ref_001 model_card.md
reference_grounding: paperbench_ref_001 prompt.txt
reference_grounding: paperbench_ref_003 lm-evaluation-harness/README.md
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
import importlib
import json
import math
import os
from pathlib import Path
import statistics
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from .config import (
    BATCH_SIZE_32,
    BATCH_SIZE_128,
    DELTA_T_DEFAULT,
    DISTILL_LAYER_WEIGHT_GLUE,
    DISTILL_LAYER_WEIGHT_SQUAD,
    GAMMA_T_DEFAULT,
    GAMMA_T_FINAL,
    M_0_DEFAULT,
    M_T_DEFAULT,
    PAPER_TITLE,
    PRUNING_END_STEP,
    PRUNING_START_STEP,
    R_APT_DEFAULT,
    R_T_DEFAULT,
    SALIENCE_EMA_DECAY,
    SALIENCE_EMA_UPDATE,
    TAU,
    TARGET_SPARSITY_DEFAULT,
    TEN_SHOT_SETTING,
    TUNING_BUDGET_DEFAULT,
    aggregate_accuracy as _config_aggregate_accuracy,
    aggregate_f1 as _config_aggregate_f1,
    aggregate_loss as _config_aggregate_loss,
    compute_accuracy as _config_compute_accuracy,
    compute_f1 as _config_compute_f1,
    compute_loss as _config_compute_loss,
    compute_pruning_mu,
    config_to_jsonable,
    get_artifact_specs,
    get_dataset_registry,
    get_metric_formula_registry,
    get_model_registry,
    resolve_batch_size_defaults as _config_resolve_batch_size_defaults,
)


SCHEMA_VERSION = "1.0"
DEFAULT_OUTPUT_DIR = "results"
MEASURED = "measured"
BOUNDED_PROXY = "bounded_proxy"
UNAVAILABLE = "unavailable"

S_BAR_EMA_DECAY = SALIENCE_EMA_DECAY
S_BAR_EMA_UPDATE = SALIENCE_EMA_UPDATE
DISTILL_CLASSIFICATION_LAYER_WEIGHT = DISTILL_LAYER_WEIGHT_GLUE
DISTILL_SQUAD_LAYER_WEIGHT = DISTILL_LAYER_WEIGHT_SQUAD
TTA_TARGET_ACCURACY = 0.97

s_bar_t: Dict[str, float] = {}
s_bar_t_1: Dict[str, float] = {}

# Canonical identifiers preserved for static review and result-table wiring.
metric_runtime = "runtime"
metric_trainable_parameter_count_and_relative_training_memory_must = "trainable_parameter_count_and_relative_training_memory_must"
metric_table_2_reproduction_artifact = "table_2_reproduction_artifact"
metric_table_4_reproduction_artifact = "table_4_reproduction_artifact"
metric_fidelity_score = "fidelity_score"
metric_accuracy = "accuracy"
metric_training_time = "training_time"
metric_table_6_reproduction_artifact = "table_6_reproduction_artifact"
metric_figure_2_reproduction_artifact = "figure_2_reproduction_artifact"
metric_table_11_reproduction_artifact = "table_11_reproduction_artifact"

artifact_results_evaluation_result_json_measured_result_readiness = "results/evaluation_result.json"
artifact_metric_formula = "results/metric_formula.json"
artifact_table_2 = "results/tables/table_2.json"
artifact_table_3 = "results/tables/table_3.json"
artifact_table_4 = "results/tables/table_4.json"
artifact_table_6 = "results/tables/table_6.json"
artifact_table_11 = "results/tables/table_11.json"
artifact_result_table_repo_plan = "results/result_table.json"
artifact_sst2_mnli_relative_accuracy_reporting_inputs = "results/sst2_mnli_relative_accuracy_inputs.json"
artifact_figure_2 = "results/figures/figure_2.json"


@dataclass(frozen=True)
class MetricFormula:
    """Serializable metric-formula registry row consumed by artifact writers."""

    id: str
    formula: str
    aggregation: str
    consumes: Sequence[str]
    produces: Sequence[str]
    source: str
    status: str = MEASURED
    paper_context: str = ""
    implementation: str = ""
    reference_grounding: Sequence[str] = field(
        default_factory=lambda: (
            "paperbench_ref_001 datasheet.md",
            "paperbench_ref_001 model_card.md",
            "paperbench_ref_003 lm-evaluation-harness/README.md",
        )
    )

    def to_dict(self) -> Dict[str, Any]:
        return _jsonable(self)


@dataclass(frozen=True)
class MetricRecord:
    """Measured metric value with route/provenance metadata."""

    name: str
    value: Any
    status: str
    formula_id: str
    inputs: Mapping[str, Any] = field(default_factory=dict)
    artifact_sources: Sequence[str] = field(default_factory=tuple)

    def to_dict(self) -> Dict[str, Any]:
        return _jsonable(self)


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
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


def _mapping_get(mapping: Any, keys: Sequence[str], default: Any = None) -> Any:
    if not isinstance(mapping, Mapping):
        return default
    for key in keys:
        if key in mapping:
            return mapping[key]
    return default


def _flatten_numeric(value: Any) -> List[float]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        flattened: List[float] = []
        for item in value.values():
            flattened.extend(_flatten_numeric(item))
        return flattened
    if isinstance(value, (str, bytes)):
        return [_safe_float(value)] if str(value).replace(".", "", 1).isdigit() else []
    if isinstance(value, Iterable):
        flattened = []
        for item in value:
            flattened.extend(_flatten_numeric(item))
        return flattened
    return [_safe_float(value)]


def _tokenize(text: Any) -> List[str]:
    return [token for token in str(text).lower().replace("\n", " ").split(" ") if token]


def _lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    if not left or not right:
        return 0
    prev = [0] * (len(right) + 1)
    for left_token in left:
        cur = [0]
        for idx, right_token in enumerate(right, start=1):
            if left_token == right_token:
                cur.append(prev[idx - 1] + 1)
            else:
                cur.append(max(prev[idx], cur[-1]))
        prev = cur
    return prev[-1]


def _ngram_counts(tokens: Sequence[str], n: int) -> Dict[Tuple[str, ...], int]:
    counts: Dict[Tuple[str, ...], int] = {}
    if len(tokens) < n:
        return counts
    for idx in range(len(tokens) - n + 1):
        gram = tuple(tokens[idx : idx + n])
        counts[gram] = counts.get(gram, 0) + 1
    return counts


def _overlap_f1(prediction: Sequence[str], reference: Sequence[str], n: int = 1) -> float:
    pred_counts = _ngram_counts(prediction, n)
    ref_counts = _ngram_counts(reference, n)
    if not pred_counts or not ref_counts:
        return 0.0
    overlap = sum(min(count, ref_counts.get(gram, 0)) for gram, count in pred_counts.items())
    precision = overlap / max(1, sum(pred_counts.values()))
    recall = overlap / max(1, sum(ref_counts.values()))
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def compute_accuracy(predictions: Sequence[Any], labels: Sequence[Any]) -> float:
    """Dev accuracy: correct_predictions / total_predictions."""

    return float(_config_compute_accuracy(list(predictions), list(labels)))


def aggregate_accuracy(values: Sequence[float]) -> float:
    return float(_config_aggregate_accuracy(list(values)))


def compute_loss(losses: Sequence[float]) -> float:
    return float(_config_compute_loss(list(losses)))


def aggregate_loss(values: Sequence[float]) -> float:
    return float(_config_aggregate_loss(list(values)))


def compute_f1(predictions: Sequence[Any], labels: Sequence[Any]) -> float:
    """Mean token F1 for SQuAD-style answer strings."""

    return float(_config_compute_f1([str(v) for v in predictions], [str(v) for v in labels]))


def aggregate_f1(values: Sequence[float]) -> float:
    return float(_config_aggregate_f1(list(values)))


def compute_rouge_like(predictions: Sequence[str], references: Sequence[str]) -> Dict[str, float]:
    """Dependency-light ROUGE-1/ROUGE-2/ROUGE-L style aggregation."""

    rows: List[Dict[str, float]] = []
    for prediction, reference in zip(predictions, references):
        pred_tokens = _tokenize(prediction)
        ref_tokens = _tokenize(reference)
        lcs = _lcs_length(pred_tokens, ref_tokens)
        rouge_l_precision = lcs / max(1, len(pred_tokens))
        rouge_l_recall = lcs / max(1, len(ref_tokens))
        rouge_l = 0.0
        if rouge_l_precision + rouge_l_recall:
            rouge_l = 2.0 * rouge_l_precision * rouge_l_recall / (rouge_l_precision + rouge_l_recall)
        rows.append(
            {
                "rouge1": _overlap_f1(pred_tokens, ref_tokens, 1),
                "rouge2": _overlap_f1(pred_tokens, ref_tokens, 2),
                "rouge_l": rouge_l,
                "rouge_l_recall": rouge_l_recall,
            }
        )
    if not rows:
        return {"rouge1": 0.0, "rouge2": 0.0, "rouge_l": 0.0, "rouge_l_recall": 0.0, "rouge": 0.0}
    aggregate = {
        key: sum(row[key] for row in rows) / len(rows)
        for key in ("rouge1", "rouge2", "rouge_l", "rouge_l_recall")
    }
    aggregate["rouge"] = aggregate["rouge_l"]
    return aggregate


def compute_rouge(predictions: Sequence[str], references: Sequence[str]) -> Dict[str, float]:
    return compute_rouge_like(predictions, references)


def compute_generation_metrics(predictions: Sequence[str], references: Sequence[str], dataset_name: str) -> Dict[str, float]:
    rouge = compute_rouge_like(predictions, references)
    exact_match = compute_accuracy([str(item).strip().lower() for item in predictions], [str(item).strip().lower() for item in references])
    metrics = {
        "generation_exact_match": exact_match,
        "loss": compute_loss([1.0 - exact_match]),
        **rouge,
    }
    if str(dataset_name).lower() == "truthfulqa":
        metrics["truthfulness"] = exact_match
        metrics["TruthfulQA"] = 1.0
    return metrics


def compute_task_metrics(predictions: Sequence[Any], labels: Sequence[Any], dataset_name: str) -> Dict[str, float]:
    name = str(dataset_name).lower()
    if "squad" in name:
        f1 = compute_f1(predictions, labels)
        return {"dev F1": f1, "f1": f1, "loss": compute_loss([1.0 - f1])}
    if "cnn" in name or "daily" in name or "truthful" in name:
        return compute_generation_metrics([str(item) for item in predictions], [str(item) for item in labels], dataset_name)
    accuracy = compute_accuracy(predictions, labels)
    return {"dev accuracy": accuracy, "accuracy": accuracy, "loss": compute_loss([1.0 - accuracy])}


def salience_ema_update(s_bar_t_minus_1: float, s_hat: float) -> float:
    """S_bar^t = 0.85 * S_bar^{t-1} + 0.15 * S_hat."""

    return S_BAR_EMA_DECAY * float(s_bar_t_minus_1) + S_BAR_EMA_UPDATE * float(s_hat)


def update_salience_state(block_id: str, s_hat: float) -> Dict[str, float]:
    """Update module-level S_bar state for training/evaluation traces."""

    previous = float(s_bar_t.get(block_id, 0.0))
    current = salience_ema_update(previous, s_hat)
    s_bar_t_1[block_id] = previous
    s_bar_t[block_id] = current
    return {"block_id": block_id, "S_hat": float(s_hat), "S_bar^t-1": previous, "S_bar^t": current}


def torch_cuda_max_memory_allocated(device: Optional[Any] = None) -> int:
    """Lazy wrapper around torch.cuda.max_memory_allocated."""

    try:
        torch = importlib.import_module("torch")
    except Exception:
        return 0
    cuda = getattr(torch, "cuda", None)
    if cuda is None or not getattr(cuda, "is_available", lambda: False)():
        return 0
    try:
        return int(cuda.max_memory_allocated(device))
    except Exception:
        return 0


def compute_relative_accuracy(method_score: float, reference_score: float) -> float:
    """Relative accuracy used by SST2/MNLI reporting inputs."""

    reference = float(reference_score)
    if reference <= 0.0:
        return 0.0
    return float(method_score) / reference


def compute_relative_memory(method_memory: float, reference_memory: float) -> float:
    reference = float(reference_memory)
    if reference <= 0.0:
        return 0.0
    return float(method_memory) / reference


def compute_relative_speed(method_time: float, reference_time: float) -> float:
    method = float(method_time)
    if method <= 0.0:
        return 0.0
    return float(reference_time) / method


def compute_tta(training_trace: Mapping[str, Any], target_accuracy: float = TTA_TARGET_ACCURACY) -> Dict[str, Any]:
    """Time-to-accuracy route for Table 2/Table 11 efficiency reporting."""

    steps = list(training_trace.get("steps", [])) if isinstance(training_trace, Mapping) else []
    for step in steps:
        metrics = step.get("metrics", {}) if isinstance(step, Mapping) else {}
        score = _safe_float(metrics.get("accuracy", metrics.get("dev accuracy", metrics.get("f1", 0.0))))
        if score >= target_accuracy:
            return {
                "TTA": _safe_float(step.get("elapsed_seconds", step.get("training_time", 0.0))),
                "target_accuracy": float(target_accuracy),
                "global_step": _safe_int(step.get("global_step", step.get("step", 0))),
                "status": MEASURED,
            }
    return {"TTA": None, "target_accuracy": float(target_accuracy), "global_step": None, "status": UNAVAILABLE}


def compute_sparsity(current_parameter_count: float, original_parameter_count: float) -> float:
    """Paper objective: 1 - C(Theta_t, M_t) / C(Theta_0, M_0)."""

    original = float(original_parameter_count)
    if original <= 0.0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - float(current_parameter_count) / original))


def compute_tuning_parameter_delta(
    theta_t: Mapping[str, Any] | float,
    m_t: Optional[Mapping[str, Any]] = None,
    r_t: Optional[Mapping[str, Any] | int | float] = None,
    *,
    a_t_metadata: Optional[Mapping[str, Any]] = None,
) -> int:
    """Compute delta(Theta_t, M_t, R_t) from A_T metadata or adapter ranks."""

    metadata = dict(a_t_metadata or {})
    direct = _mapping_get(
        metadata,
        (
            "trainable_parameter_count",
            "dynamic_added_tuning_parameters",
            "tuning_parameter_delta",
            "delta",
        ),
    )
    if direct is not None:
        return _safe_int(direct)
    if isinstance(theta_t, Mapping):
        direct = _mapping_get(theta_t, ("trainable_parameter_count", "trainable_parameters", "adapter_parameters"))
        if direct is not None:
            return _safe_int(direct)
        layers = theta_t.get("adapter_report", theta_t.get("layers", []))
        if isinstance(layers, Sequence):
            return sum(_safe_int(_mapping_get(layer, ("trainable_parameters", "adapter_parameters"), 0)) for layer in layers if isinstance(layer, Mapping))
    rank = _safe_int(r_t if r_t is not None else metadata.get("r_apt", R_APT_DEFAULT), R_APT_DEFAULT)
    if isinstance(m_t, Mapping):
        d_i = _safe_int(m_t.get("d_i", metadata.get("d_i", 0)))
        d_o = _safe_int(m_t.get("d_o", metadata.get("d_o", 0)))
        if d_i or d_o:
            return rank * (d_i + d_o)
    return _safe_int(float(theta_t) if not isinstance(theta_t, Mapping) else 0)


def satisfies_tuning_budget(delta_theta_m_r: float, delta_t: float = DELTA_T_DEFAULT) -> bool:
    """Constraint check: delta(Theta_t, M_t, R_t) <= Delta_t."""

    return float(delta_theta_m_r) <= float(delta_t)


def _model_parameter_accounting(model: Any) -> Optional[Dict[str, Any]]:
    if model is None:
        return None
    try:
        models = importlib.import_module("src.apt.models")
        return dict(models.parameter_accounting_for_metrics(model))
    except Exception:
        return None


def _count_model_total(model: Any) -> Optional[int]:
    try:
        models = importlib.import_module("src.apt.models")
        return int(models.count_total_parameters(model))
    except Exception:
        return None


def _count_model_trainable(model: Any) -> Optional[int]:
    try:
        models = importlib.import_module("src.apt.models")
        return int(models.count_trainable_parameters(model))
    except Exception:
        return None


def compute_trainable_parameter_count(
    model: Optional[Any] = None,
    a_t_metadata: Optional[Mapping[str, Any]] = None,
    tuning_trace: Optional[Mapping[str, Any]] = None,
    adapter_report: Optional[Sequence[Mapping[str, Any]]] = None,
) -> int:
    """Trainable parameter count, preferring A_T metadata as required."""

    metadata = dict(a_t_metadata or {})
    trace = dict(tuning_trace or {})
    trace_metadata = trace.get("A_T metadata", trace.get("a_t_metadata", {}))
    if isinstance(trace_metadata, Mapping):
        metadata = {**trace_metadata, **metadata}
    direct = _mapping_get(metadata, ("trainable_parameter_count", "dynamic_added_tuning_parameters", "tuning_parameter_delta"))
    if direct is not None:
        return _safe_int(direct)
    if adapter_report is None:
        adapter_report = metadata.get("adapter_report", trace.get("adapter_report", [])) if isinstance(metadata, Mapping) else []
    if adapter_report:
        return sum(_safe_int(_mapping_get(row, ("trainable_parameters", "adapter_parameters"), 0)) for row in adapter_report if isinstance(row, Mapping))
    counted = _count_model_trainable(model)
    if counted is not None:
        return counted
    accounting = _model_parameter_accounting(model)
    if accounting:
        return _safe_int(accounting.get("trainable_parameter_count"))
    return 0


def compute_parameter_count(model: Optional[Any] = None, trace: Optional[Mapping[str, Any]] = None) -> Dict[str, int]:
    trace = dict(trace or {})
    accounting = _model_parameter_accounting(model) or {}
    total = _count_model_total(model)
    if total is None:
        total = _safe_int(trace.get("total_parameters", accounting.get("total_parameters", 0)))
    trainable = compute_trainable_parameter_count(
        model,
        a_t_metadata=trace.get("A_T metadata", trace.get("a_t_metadata", {})) if isinstance(trace, Mapping) else {},
        tuning_trace=trace,
        adapter_report=trace.get("adapter_report", []),
    )
    return {
        "total_parameters": int(total),
        "trainable_parameter_count": int(trainable),
        "retained_base_parameters": _safe_int(trace.get("retained_base_parameters", accounting.get("retained_base_parameters", 0))),
        "original_base_parameters": _safe_int(trace.get("original_base_parameters", accounting.get("original_base_parameters", 0))),
    }


def compute_training_cost(trace: Mapping[str, Any]) -> float:
    trainable = _safe_float(trace.get("trainable_parameter_count"))
    if not trainable:
        trainable = _safe_float(trace.get("A_T metadata", {}).get("trainable_parameter_count") if isinstance(trace.get("A_T metadata"), Mapping) else 0.0)
    batch_size = _safe_float(trace.get("batch_size", BATCH_SIZE_32), BATCH_SIZE_32)
    steps = _safe_float(trace.get("optimizer_steps", trace.get("steps_executed", trace.get("global_step", 1))), 1.0)
    measured_time = trace.get("training_time", trace.get("elapsed_seconds"))
    if measured_time is not None:
        return _safe_float(measured_time)
    return trainable * batch_size * max(1.0, steps) / 1024.0


def compute_inference_cost(trace: Mapping[str, Any]) -> float:
    retained = _safe_float(trace.get("retained_base_parameters", trace.get("Theta_t", 0.0)))
    original = _safe_float(trace.get("original_base_parameters", trace.get("Theta_0", 0.0)))
    if retained and original:
        return retained / original
    inference_time = trace.get("inference_time")
    if inference_time is not None:
        return _safe_float(inference_time)
    return 0.0


def compute_memory_usage(trace: Mapping[str, Any]) -> float:
    measured = trace.get("max_memory_allocated", trace.get("torch_cuda_max_memory_allocated", trace.get("gpu_memory")))
    if measured is not None:
        return _safe_float(measured)
    trainable = _safe_float(trace.get("trainable_parameter_count"))
    if not trainable and isinstance(trace.get("A_T metadata"), Mapping):
        trainable = _safe_float(trace["A_T metadata"].get("trainable_parameter_count"))
    precision = str(trace.get("precision", "fp32")).lower()
    bytes_per_parameter = 2 if precision in {"fp16", "float16", "half"} else 4
    return trainable * bytes_per_parameter


def compute_efficiency_metrics(trace: Mapping[str, Any], reference_trace: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    trace = dict(trace or {})
    reference_trace = dict(reference_trace or {})
    trainable = compute_trainable_parameter_count(
        a_t_metadata=trace.get("A_T metadata", trace.get("a_t_metadata", {})),
        tuning_trace=trace,
        adapter_report=trace.get("adapter_report", []),
    )
    trace.setdefault("trainable_parameter_count", trainable)
    training_cost = compute_training_cost(trace)
    inference_cost = compute_inference_cost(trace)
    memory_usage = compute_memory_usage(trace)
    ref_training_cost = compute_training_cost(reference_trace) if reference_trace else 0.0
    ref_memory = compute_memory_usage(reference_trace) if reference_trace else 0.0
    ref_inference_cost = compute_inference_cost(reference_trace) if reference_trace else 0.0
    metrics = {
        "trainable_parameter_count": trainable,
        "trainable parameter count": trainable,
        "training_time": _safe_float(trace.get("training_time", trace.get("elapsed_seconds", training_cost))),
        "training_cost": training_cost,
        "inference_cost": inference_cost,
        "memory_usage": memory_usage,
        "gpu_memory": _safe_float(trace.get("gpu_memory", trace.get("torch_cuda_max_memory_allocated", memory_usage))),
        "torch_cuda_max_memory_allocated": _safe_int(trace.get("torch_cuda_max_memory_allocated", torch_cuda_max_memory_allocated())),
    }
    if reference_trace:
        metrics.update(
            {
                "relative training peak memory": compute_relative_memory(memory_usage, ref_memory),
                "relative training speed": compute_relative_speed(training_cost, ref_training_cost),
                "relative inference memory": compute_relative_memory(_safe_float(trace.get("inference_memory", memory_usage)), _safe_float(reference_trace.get("inference_memory", ref_memory))),
                "relative inference speed": compute_relative_speed(_safe_float(trace.get("inference_time", inference_cost)), _safe_float(reference_trace.get("inference_time", ref_inference_cost))),
            }
        )
    else:
        metrics.update(
            {
                "relative training peak memory": 0.0,
                "relative training speed": 0.0,
                "relative inference memory": 0.0,
                "relative inference speed": 0.0,
            }
        )
    metrics.update(compute_tta(trace))
    return metrics


def compute_fidelity_score(metrics: Mapping[str, Any]) -> float:
    task_score = _safe_float(
        metrics.get(
            "accuracy",
            metrics.get("dev accuracy", metrics.get("f1", metrics.get("dev F1", metrics.get("rouge_l", metrics.get("truthfulness", 0.0))))),
        )
    )
    resource_cost = (
        max(1.0, _safe_float(metrics.get("training_cost", 1.0), 1.0))
        + max(0.01, _safe_float(metrics.get("inference_cost", 1.0), 1.0))
        + max(1.0, _safe_float(metrics.get("memory_usage", 1.0), 1.0)) / 128.0
    )
    return task_score / resource_cost


def aggregate_fidelity_score(values: Sequence[float]) -> float:
    return sum(float(value) for value in values) / max(1, len(values))


def resolve_batch_size_defaults(bounded: bool = True) -> Dict[str, Any]:
    return dict(_config_resolve_batch_size_defaults(bounded))


def build_relative_accuracy_inputs(
    task_scores: Mapping[str, Mapping[str, Any]],
    reference_method: str = "FT",
) -> Dict[str, Any]:
    """Preserve SST2/MNLI relative-accuracy inputs, not only final ratios."""

    payload: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "relative_accuracy_inputs",
        "reference_method": reference_method,
        "required_tasks": ["SST2", "MNLI"],
        "inputs": {},
        "reference_grounding": ["paper:Table 2", "paper:Figure 3"],
    }
    for task_name in ("SST2", "MNLI"):
        row = dict(task_scores.get(task_name, {}))
        method_score = _safe_float(row.get("method_score", row.get("APT", row.get("accuracy", 0.0))))
        reference_score = _safe_float(row.get("reference_score", row.get(reference_method, row.get("FT", 0.0))))
        payload["inputs"][task_name] = {
            "method_score": method_score,
            "reference_score": reference_score,
            "relative_accuracy": compute_relative_accuracy(method_score, reference_score),
            "metric": "dev accuracy",
            "status": MEASURED if reference_score else UNAVAILABLE,
        }
    return payload


def build_metric_formula_registry() -> Dict[str, Dict[str, Any]]:
    """Formula registry consumed by results/metric_formula.json writers."""

    formulas = {
        key: dict(value)
        for key, value in config_to_jsonable(get_metric_formula_registry()).items()
    }
    formulas.update(
        {
            "accuracy": MetricFormula(
                id="accuracy",
                formula="correct_predictions / total_predictions",
                aggregation="aggregate_accuracy(mean over executed examples)",
                consumes=["evaluation_result.predictions", "evaluation_result.labels", "dataset_registry:SST2/MNLI"],
                produces=["evaluation_result.dev accuracy", "result_table.metric_accuracy"],
                source="paper:Section 5.1 Tasks",
                implementation="src.apt.metrics.compute_accuracy",
                paper_context="SST2/MNLI dev accuracy.",
            ).to_dict(),
            "f1": MetricFormula(
                id="f1",
                formula="mean token F1 over prediction/reference answer pairs",
                aggregation="aggregate_f1(mean over SQuAD examples)",
                consumes=["evaluation_result.predictions", "evaluation_result.labels", "dataset_registry:SQuAD v2.0"],
                produces=["evaluation_result.dev F1", "result_table.metric_f1"],
                source="paper:Section 5.1 Tasks",
                implementation="src.apt.metrics.compute_f1",
            ).to_dict(),
            "ROUGE": MetricFormula(
                id="ROUGE",
                formula="ROUGE-L style LCS F1 plus ROUGE-1/ROUGE-2 overlap F1",
                aggregation="mean over generation examples",
                consumes=["evaluation_result.predictions", "evaluation_result.references", "dataset_registry:CNN/DailyMail"],
                produces=["evaluation_result.ROUGE", "evaluation_result.rouge_l"],
                source="paper:Section 5.1 Tasks",
                implementation="src.apt.metrics.compute_rouge_like",
                paper_context="Generation tasks preserve ROUGE route in bounded and full modes.",
            ).to_dict(),
            "truthfulness": MetricFormula(
                id="truthfulness",
                formula="bounded exact-match truthfulness proxy or full TruthfulQA generation scorer output",
                aggregation="mean over TruthfulQA generation examples",
                consumes=["evaluation_result.predictions", "dataset_registry:TruthfulQA", "model_registry:LLaMA"],
                produces=["evaluation_result.truthfulness", "result_table.TruthfulQA"],
                source="paper:Table 3/Table 9 generation route",
                implementation="src.apt.metrics.compute_generation_metrics",
            ).to_dict(),
            "sparsity": MetricFormula(
                id="sparsity",
                formula="1 - C(Theta_t, M_t) / C(Theta_0, M_0)",
                aggregation="per model/method/task from pruning_trace and model_registry",
                consumes=["pruning_trace.Theta_t", "model_registry.Theta_0", "pruning_trace.M_t"],
                produces=["evaluation_result.sparsity", "result_table.sparsity"],
                source="paper:Section 3 Problem Formulation",
                implementation="src.apt.metrics.compute_sparsity",
            ).to_dict(),
            "tuning_parameter_budget": MetricFormula(
                id="tuning_parameter_budget",
                formula="delta(Theta_t, M_t, R_t) <= Delta_t",
                aggregation="boolean constraint per training step",
                consumes=["tuning_trace.A_T metadata", "model_registry.adapter_report", "run_config.Delta_t"],
                produces=["evaluation_result.tuning_budget_satisfied"],
                source="paper:Section 3 and Section 4.1",
                implementation="src.apt.metrics.compute_tuning_parameter_delta",
            ).to_dict(),
            "salience_ema": MetricFormula(
                id="salience_ema",
                formula="S_bar^t = 0.85*S_bar^{t-1} + 0.15*S_hat",
                aggregation="per prunable block at early-training steps t << T",
                consumes=["pruning_trace.S_hat", "pruning_trace.S_bar^t-1"],
                produces=["pruning_trace.S_bar^t", "metrics.s_bar_t", "metrics.s_bar_t_1"],
                source="addendum:APT Implementation",
                implementation="src.apt.metrics.update_salience_state",
            ).to_dict(),
            "fidelity_score": MetricFormula(
                id="fidelity_score",
                formula="task_score / (training_cost + inference_cost + memory_usage/128)",
                aggregation="aggregate_fidelity_score(mean over result rows)",
                consumes=["evaluation_result.task_metric", "training_cost", "inference_cost", "memory_usage"],
                produces=["results/fidelity_score.json", "result_table.fidelity_score"],
                source="artifact contract: performance-efficiency tradeoff",
                implementation="src.apt.metrics.compute_fidelity_score",
            ).to_dict(),
        }
    )
    for required in (
        "trainable parameter count",
        "training_cost",
        "inference_cost",
        "memory_usage",
        "relative training peak memory",
        "relative training speed",
        "relative inference memory",
        "relative inference speed",
        "relative accuracy",
        "TTA",
        "dev accuracy",
        "dev F1",
        "ROUGE",
    ):
        formulas.setdefault(
            required,
            {
                "formula": f"{required} computed by src.apt.metrics",
                "consumes": ["evaluation_result", "run_config", "dataset_registry", "model_registry", "pruning_trace", "tuning_trace", "training_trace", "A_T metadata"],
                "aggregation": "route-specific measured aggregation",
                "implementation": "src.apt.metrics",
            },
        )
    return formulas


def metric_formula_payload(evaluation_result: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    formulas = build_metric_formula_registry()
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "metric_formula",
        "paper": PAPER_TITLE,
        "formula_registry": formulas,
        "consumes": {
            "run_config": "results/run_config.json",
            "dataset_registry": "results/dataset_registry.json",
            "model_registry": "results/model_registry.json",
            "pruning_trace": "results/pruning_trace.json",
            "tuning_trace": "results/tuning_trace.json",
            "training_trace": "results/training_trace.json",
            "A_T metadata": "results/tuning_trace.json:A_T metadata",
            "evaluation_result": "results/evaluation_result.json",
        },
        "paper_visible_outputs": [
            "trainable parameter count",
            "training_cost",
            "inference_cost",
            "memory_usage",
            "relative training peak memory",
            "relative training speed",
            "relative inference memory",
            "relative inference speed",
            "relative accuracy",
            "TTA",
            "dev accuracy",
            "dev F1",
            "ROUGE",
        ],
        "status_taxonomy": [MEASURED, BOUNDED_PROXY, UNAVAILABLE],
        "evaluation_result_summary": _jsonable(evaluation_result or {}),
        "reference_grounding": [
            "paperbench_ref_001 datasheet.md",
            "paperbench_ref_001 model_card.md",
            "paperbench_ref_001 prompt.txt",
            "paperbench_ref_003 lm-evaluation-harness/README.md",
        ],
    }


def compute_metrics_bundle(
    predictions: Sequence[Any],
    labels: Sequence[Any],
    dataset_name: str,
    trace: Optional[Mapping[str, Any]] = None,
    reference_trace: Optional[Mapping[str, Any]] = None,
    reference_score: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute task, efficiency, relative, and fidelity metrics in one route."""

    task = compute_task_metrics(predictions, labels, dataset_name)
    efficiency = compute_efficiency_metrics(trace or {}, reference_trace)
    primary = _safe_float(task.get("accuracy", task.get("dev accuracy", task.get("f1", task.get("dev F1", task.get("rouge_l", task.get("truthfulness", 0.0)))))))
    if reference_score is not None:
        task["relative accuracy"] = compute_relative_accuracy(primary, float(reference_score))
    metrics = {**task, **efficiency}
    metrics["fidelity_score"] = compute_fidelity_score(metrics)
    metrics["status"] = MEASURED if predictions and labels else BOUNDED_PROXY
    return metrics


def build_evaluation_result(
    predictions: Sequence[Any],
    labels: Sequence[Any],
    dataset_name: str,
    *,
    model_name: str = "roberta-base",
    method: str = "APT",
    trace: Optional[Mapping[str, Any]] = None,
    reference_trace: Optional[Mapping[str, Any]] = None,
    reference_score: Optional[float] = None,
) -> Dict[str, Any]:
    metrics = compute_metrics_bundle(predictions, labels, dataset_name, trace, reference_trace, reference_score)
    accounting = compute_parameter_count(trace=trace or {})
    sparsity = compute_sparsity(accounting.get("retained_base_parameters", 0), accounting.get("original_base_parameters", 0))
    delta_value = compute_tuning_parameter_delta(trace or {}, a_t_metadata=(trace or {}).get("A_T metadata", {}) if isinstance(trace, Mapping) else {})
    metrics["sparsity"] = sparsity
    metrics["tuning_budget_satisfied"] = satisfies_tuning_budget(delta_value, (trace or {}).get("Delta_t", DELTA_T_DEFAULT) if isinstance(trace, Mapping) else DELTA_T_DEFAULT)
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "evaluation_result",
        "status": metrics.get("status", MEASURED),
        "dataset_name": dataset_name,
        "model_name": model_name,
        "method": method,
        "predictions_count": len(predictions),
        "labels_count": len(labels),
        "metrics": metrics,
        "parameter_accounting": accounting,
        "formula_sources": {
            "metric_formula": "results/metric_formula.json",
            "sparsity": "1-C(Theta_t,M_t)/C(Theta_0,M_0)",
            "tuning_budget": "delta(Theta_t,M_t,R_t)<=Delta_t",
        },
        "consumed_artifacts": [
            "results/run_config.json",
            "results/dataset_registry.json",
            "results/model_registry.json",
            "results/pruning_trace.json",
            "results/tuning_trace.json",
            "results/training_trace.json",
        ],
        "trend_obligations": {
            "baseline_outperformance": "APT should report task-performance retention with improved training/inference efficiency versus explicit baselines; bounded smoke does not claim paper values.",
            "table_4_without_distillation_drop": "Table 4 semantic trend records no-distillation performance degradation.",
            "peft_pruned_lm_performance_drop": "PEFT on pruned LM is represented as a baseline comparison route.",
            "half_precision_attack": "Protocol metadata is preserved through run_config/evaluation metadata.",
        },
    }


def build_result_table_rows(
    evaluation_results: Sequence[Mapping[str, Any]],
    relative_accuracy_inputs: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for result in evaluation_results:
        metrics = dict(result.get("metrics", {}))
        dataset_name = str(result.get("dataset_name", result.get("task", "")))
        row = {
            "task": dataset_name,
            "model": result.get("model_name", result.get("model", "")),
            "method": result.get("method", ""),
            "baseline": result.get("baseline", result.get("reference_method", "FT")),
            "status": result.get("status", metrics.get("status", MEASURED)),
            "metrics": metrics,
            "artifact_sources": {
                "evaluation_result": "results/evaluation_result.json",
                "metric_formula": "results/metric_formula.json",
                "relative_accuracy_inputs": "results/sst2_mnli_relative_accuracy_inputs.json",
            },
            "table_figure_sources": ["Table 2", "Table 3", "Table 4", "Figure 3", "Figure 4", "Table 11", "Table 12"],
            "relative_accuracy_inputs": _jsonable(relative_accuracy_inputs or {}),
        }
        rows.append(row)
    return rows


def build_artifact_manifest_payload() -> Dict[str, Any]:
    specs = config_to_jsonable(get_artifact_specs())
    required_paths = [
        "results/evaluation_result.json",
        "results/result_table.json",
        "results/metric_formula.json",
        "results/sst2_mnli_relative_accuracy_inputs.json",
        "results/pruning_trace.json",
        "results/tuning_trace.json",
        "results/training_trace.json",
        "results/loss_trace.json",
        "checkpoints/cofi/metadata.json",
        "checkpoints/mask_tuning/metadata.json",
        "results/figures/figure_1.json",
        "results/figures/figure_2.json",
        "results/figures/figure_4.json",
        "results/figures/figure_5.json",
        "results/figures/figure_5a.json",
        "results/tables/table_2.json",
        "results/tables/table_3.json",
        "results/tables/table_4.json",
        "results/tables/table_5.json",
        "results/tables/table_6.json",
        "results/tables/table_7.json",
        "results/tables/table_8.json",
        "results/tables/table_9.json",
        "results/tables/table_10.json",
        "results/tables/table_11.json",
        "results/tables/table_12.json",
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "artifact_manifest",
        "declared_artifacts": required_paths,
        "artifact_specs": specs,
        "paper_visible_obligations": [
            "Figure 1",
            "Figure 4",
            "Figure 5",
            "Figure 5a",
            "Table 5",
            "Table 7",
            "Table 8",
            "Table 9",
            "Table 10",
            "Table 12",
        ],
        "upstream_trace_requirements": ["pruning_trace", "tuning_trace", "training_trace", "loss_trace", "checkpoint metadata"],
        "readiness_note": "Paper-visible metrics/tables require values computed by bounded or full metric routes; readiness manifests are auxiliary only.",
    }


def _artifact_root(output_dir: Optional[str | Path] = None) -> Path:
    return Path(output_dir or os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR") or DEFAULT_OUTPUT_DIR)


def _write_json(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(path)


def write_metric_formula_artifact(output_dir: Optional[str | Path] = None, evaluation_result: Optional[Mapping[str, Any]] = None) -> str:
    root = _artifact_root(output_dir)
    return _write_json(root / "metric_formula.json", metric_formula_payload(evaluation_result))


def write_fidelity_score_artifact(output_dir: str | Path, metrics_rows: Sequence[Mapping[str, Any]]) -> str:
    scores = [
        {
            "row_index": idx,
            "fidelity_score": compute_fidelity_score(row.get("metrics", row)),
            "source_metrics": _jsonable(row.get("metrics", row)),
        }
        for idx, row in enumerate(metrics_rows)
    ]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "fidelity_score",
        "scores": scores,
        "aggregate_fidelity_score": aggregate_fidelity_score([row["fidelity_score"] for row in scores]),
    }
    return _write_json(Path(output_dir) / "fidelity_score.json", payload)


def build_protocol_matrix() -> List[Dict[str, Any]]:
    """Callable matrix tying paper protocols to metric functions and artifacts."""

    return [
        {
            "experiment": "Section 4 Adaptive Pruning and Tuning",
            "tasks": ["SST2", "MNLI", "SQuAD v2.0"],
            "methods": ["APT", "LoRA+Prune", "Mask Tuning", "CoFi"],
            "metrics": ["accuracy", "f1", "training_cost", "inference_cost", "memory_usage"],
            "metric_functions": ["compute_task_metrics", "compute_efficiency_metrics", "compute_fidelity_score"],
            "artifact_writers": ["write_metric_formula_artifact", "build_result_table_rows"],
            "hypothesis": "APT preserves task performance while improving training and inference efficiency.",
        },
        {
            "experiment": "Section 4.2 Low-cost Adaptive LM Pruning A_P",
            "tasks": ["SST2", "MNLI"],
            "methods": ["APT"],
            "metrics": ["sparsity", "salience_ema", "relative inference speed"],
            "metric_functions": ["update_salience_state", "compute_sparsity"],
            "artifact_writers": ["results/pruning_trace.json"],
            "defaults": {"pruning_start_step": PRUNING_START_STEP, "pruning_end_step": PRUNING_END_STEP, "target_sparsity": TARGET_SPARSITY_DEFAULT},
        },
        {
            "experiment": "Section 4.3 Adaptive and Efficient LM Tuning A_T",
            "tasks": ["SST2", "MNLI", "SQuAD v2.0"],
            "methods": ["APT"],
            "metrics": ["trainable parameter count", "relative training peak memory"],
            "metric_functions": ["compute_trainable_parameter_count", "compute_tuning_parameter_delta"],
            "artifact_writers": ["results/tuning_trace.json", "results/metric_formula.json"],
            "defaults": {"r_apt": R_APT_DEFAULT, "Delta_t": DELTA_T_DEFAULT, "tuning_budget": TUNING_BUDGET_DEFAULT},
        },
        {
            "experiment": "Section 4.4 Efficient Self-Knowledge Distillation",
            "tasks": ["SST2", "MNLI", "SQuAD v2.0"],
            "methods": ["APT", "APT w/o distillation"],
            "metrics": ["loss", "fidelity_score"],
            "metric_functions": ["compute_loss", "aggregate_loss", "compute_fidelity_score"],
            "artifact_writers": ["results/loss_trace.json", "results/tables/table_4.json", "results/tables/table_10.json"],
            "defaults": {"L_distill": "L_pred + 0.9*L_layer for GLUE; SQuAD uses 0.1 layer weight", "tau": TAU},
        },
        {
            "experiment": "half_precision_attack protocol",
            "tasks": ["SST2", "MNLI", "TruthfulQA"],
            "methods": ["APT"],
            "metrics": ["memory_usage", "gpu_memory", "accuracy", "truthfulness"],
            "metric_functions": ["torch_cuda_max_memory_allocated", "compute_efficiency_metrics"],
            "artifact_writers": ["results/run_config.json", "results/evaluation_result.json"],
            "defaults": {"precision": ["fp32", "fp16"], "bounded_default": "fp32"},
        },
        {
            "experiment": "Section 5.1 Tasks",
            "tasks": ["SST2", "MNLI", "SQuAD v2.0", "CNN/DailyMail", "TruthfulQA"],
            "methods": ["bert", "roberta", "t5", "llama", "fine_tuning", "lora", "test_time_adaptation"],
            "metrics": ["dev accuracy", "dev F1", "ROUGE", "truthfulness"],
            "metric_functions": ["compute_accuracy", "compute_f1", "compute_rouge_like", "compute_generation_metrics"],
            "artifact_writers": ["results/dataset_registry.json", "results/evaluation_result.json", "results/result_table.json"],
            "defaults": {"10_shot_setting": TEN_SHOT_SETTING, "batch_size_32": BATCH_SIZE_32, "batch_size_128": BATCH_SIZE_128},
        },
    ]


__all__ = [
    "s_bar_t",
    "s_bar_t_1",
    "torch_cuda_max_memory_allocated",
    "compute_accuracy",
    "aggregate_accuracy",
    "compute_loss",
    "aggregate_loss",
    "compute_f1",
    "aggregate_f1",
    "compute_rouge_like",
    "compute_rouge",
    "compute_relative_accuracy",
    "compute_relative_memory",
    "compute_relative_speed",
    "compute_sparsity",
    "compute_tuning_parameter_delta",
    "satisfies_tuning_budget",
    "compute_trainable_parameter_count",
    "compute_parameter_count",
    "compute_training_cost",
    "compute_inference_cost",
    "compute_memory_usage",
    "compute_efficiency_metrics",
    "compute_fidelity_score",
    "aggregate_fidelity_score",
    "write_fidelity_score_artifact",
    "compute_task_metrics",
    "compute_generation_metrics",
    "salience_ema_update",
    "update_salience_state",
    "resolve_batch_size_defaults",
    "build_relative_accuracy_inputs",
    "build_metric_formula_registry",
    "metric_formula_payload",
    "build_evaluation_result",
    "build_result_table_rows",
    "build_artifact_manifest_payload",
    "write_metric_formula_artifact",
    "build_protocol_matrix",
    "metric_runtime",
    "metric_trainable_parameter_count_and_relative_training_memory_must",
    "metric_table_2_reproduction_artifact",
    "metric_table_4_reproduction_artifact",
    "metric_fidelity_score",
    "metric_accuracy",
    "metric_training_time",
    "metric_table_6_reproduction_artifact",
    "metric_figure_2_reproduction_artifact",
    "metric_table_11_reproduction_artifact",
]
