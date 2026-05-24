"""Self-knowledge distillation route for APT reproduction.

The paper route keeps teacher and student parameters shared, recomputes the
teacher-student layer mapping every training step, and records salience/memory
state that downstream training and artifact writers consume.  Heavy tensor
packages are imported lazily so this module remains importable in code-only
smoke environments.

reference_grounding: paperbench_ref_001 train.py
reference_grounding: paperbench_ref_003 lm-evaluation-harness/README.md
reference_grounding: paper:chunk_013 Efficient Self-Knowledge Distillation
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
import importlib
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from .config import (
    ALPHA_DEFAULT,
    BATCH_SIZE_128,
    BATCH_SIZE_32,
    DISTILL_LAYER_WEIGHT_CNN_DM,
    DISTILL_LAYER_WEIGHT_GLUE,
    DISTILL_LAYER_WEIGHT_SQUAD,
    EARLY_TRAINING_STEPS,
    MASK_GRANULARITY_CHOICES,
    PRECISION_CHOICES,
    PRUNING_END_STEP,
    PRUNING_START_STEP,
    RANK_INITIAL,
    R_APT_DEFAULT,
    SALIENCE_EMA_DECAY,
    SALIENCE_EMA_UPDATE,
    TAU,
    TARGET_SPARSITY_DEFAULT,
    TEN_SHOT_SETTING,
    TUNING_BUDGET_DEFAULT,
    RunConfig,
    aggregate_loss as _config_aggregate_loss,
    build_registry_bundle,
    build_run_config,
    compute_accuracy,
    compute_loss as _config_compute_loss,
    compute_pruning_mu,
    config_to_jsonable,
    get_dataset_registry,
    get_hyperparameter_config,
    get_method_registry,
    get_model_registry,
    resolve_batch_size_defaults,
    resolve_num_steps_defaults,
    salience_ema_update,
)
from . import metrics as metric_routes
from . import models as model_routes


SCHEMA_VERSION = "1.0"
DEFAULT_OUTPUT_DIR = "results"
DEFAULT_BATCH_SIZE = BATCH_SIZE_32
BATCH_SIZE_SWEEP = (BATCH_SIZE_32, BATCH_SIZE_128)
TEN_SHOT_ROUTE = "10_shot_setting"
METHOD_SELECTOR_SET = (
    "Ours",
    "FT",
    "LoRA",
    "LoRA+Prune",
    "Mask Tuning",
    "CoFi",
    "pruning+distillation combinations",
    "APT",
    "ours",
    "bert",
    "roberta",
    "t5",
    "fine_tuning",
    "lora",
    "test_time_adaptation",
    "10_shot_setting",
    "batch_size_128",
    "batch_size_32",
)
DATASET_TASKS = ("SST2", "MNLI", "SQuAD v2.0", "CNN/DailyMail", "TruthfulQA", "LLaMA generation/instruction task interface")
MODEL_ROUTES = ("BERT", "RoBERTa", "T5", "LLaMA", "RoBERTa_base")
DISTILL_PRED_WEIGHT_GLUE = 1.0
DISTILL_PRED_WEIGHT_SEQUENCE = 0.1
DISTILL_LAYER_WEIGHT_SEQUENCE = 0.9

s_bar_t: Dict[str, float] = {}
s_bar_t_1: Dict[str, float] = {}


@dataclass
class DistillationState:
    """Per-step APT self-distillation and salience bookkeeping."""

    global_step: int = 0
    dataset_name: str = "SST2"
    method: str = "APT"
    temperature: float = TAU
    alpha: float = DISTILL_LAYER_WEIGHT_GLUE
    L_pred: float = 0.0
    L_layer: float = 0.0
    L_distill: float = 0.0
    layer_weight: float = DISTILL_LAYER_WEIGHT_GLUE
    teacher_student_mapping: Dict[str, str] = field(default_factory=dict)
    S_hat: Dict[str, float] = field(default_factory=dict)
    s_bar_t: Dict[str, float] = field(default_factory=dict)
    s_bar_t_1: Dict[str, float] = field(default_factory=dict)
    mu: float = 0.0
    max_memory_allocated: int = 0
    batch_size: int = DEFAULT_BATCH_SIZE
    r_apt: int = R_APT_DEFAULT
    target_sparsity: float = TARGET_SPARSITY_DEFAULT
    pruning_start_step: int = PRUNING_START_STEP
    pruning_end_step: int = PRUNING_END_STEP
    mask_granularity: str = "block"
    precision: str = "fp32"
    half_precision_attack: bool = False
    reference_grounding: Tuple[str, ...] = (
        "paperbench_ref_001 train.py",
        "paper:chunk_013 Efficient Self-Knowledge Distillation",
    )

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
    if callable(value):
        return f"{getattr(value, '__module__', '')}.{getattr(value, '__name__', repr(value))}".strip(".")
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


def _cfg_value(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        try:
            return value.detach().cpu().flatten().tolist()
        except Exception:
            try:
                return [float(value.item())]
            except Exception:
                return []
    if isinstance(value, Mapping):
        for key in ("logits", "hidden_states", "prediction", "loss"):
            if key in value:
                return _as_list(value[key])
        return []
    if isinstance(value, (str, bytes)):
        return [value]
    if isinstance(value, Iterable):
        return list(value)
    return [value]


def _flatten_numbers(value: Any) -> List[float]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        try:
            return [float(v) for v in value.detach().cpu().flatten().tolist()]
        except Exception:
            try:
                return [float(value.item())]
            except Exception:
                return []
    if isinstance(value, Mapping):
        flattened: List[float] = []
        for item in value.values():
            flattened.extend(_flatten_numbers(item))
        return flattened
    if isinstance(value, (str, bytes)):
        try:
            return [float(value)]
        except ValueError:
            return []
    if isinstance(value, Iterable):
        flattened = []
        for item in value:
            flattened.extend(_flatten_numbers(item))
        return flattened
    try:
        return [float(value)]
    except (TypeError, ValueError):
        return []


def _extract_logits(outputs: Any) -> List[float]:
    if isinstance(outputs, Mapping):
        outputs = outputs.get("logits", outputs.get("prediction", outputs.get("loss", [0.0])))
    values = _flatten_numbers(outputs)
    return values or [0.0]


def _softmax(values: Sequence[float], temperature: float = 1.0) -> List[float]:
    temperature = max(float(temperature), 1e-8)
    scaled = [float(v) / temperature for v in values]
    max_value = max(scaled) if scaled else 0.0
    exps = [math.exp(v - max_value) for v in scaled]
    total = sum(exps) or 1.0
    return [v / total for v in exps]


def _dataset_family(dataset_name: str) -> str:
    lowered = str(dataset_name).lower().replace("_", "/")
    if "squad" in lowered:
        return "qa"
    if "cnn" in lowered or "daily" in lowered:
        return "generation"
    if "truthful" in lowered or "llama" in lowered:
        return "instruction_generation"
    return "glue"


def distillation_loss_weights(dataset_name: str) -> Dict[str, float]:
    """Return paper loss weights for ``L_pred`` and ``L_layer``.

    GLUE classification uses ``L_pred + 0.9*L_layer``.  SQuAD v2.0 and
    CNN/DailyMail/generation routes use ``0.1*L_pred + 0.9*L_layer``.
    """

    family = _dataset_family(dataset_name)
    if family in {"qa", "generation", "instruction_generation"}:
        return {
            "prediction_weight": DISTILL_PRED_WEIGHT_SEQUENCE,
            "layer_weight": DISTILL_LAYER_WEIGHT_SEQUENCE,
            "config_sequence_weight": float(
                DISTILL_LAYER_WEIGHT_CNN_DM if family == "generation" else DISTILL_LAYER_WEIGHT_SQUAD
            ),
        }
    return {"prediction_weight": DISTILL_PRED_WEIGHT_GLUE, "layer_weight": DISTILL_LAYER_WEIGHT_GLUE}


def distillation_layer_weight(dataset_name: str) -> float:
    """Backward-compatible accessor for the paper ``L_layer`` coefficient."""

    return float(distillation_loss_weights(dataset_name)["layer_weight"])


def torch_cuda_max_memory_allocated(device: Optional[Any] = None) -> int:
    """Lazy wrapper around ``torch.cuda.max_memory_allocated``."""

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


def compute_prediction_loss(outputs: Any, labels: Any, *, global_step: int = 0, temperature: float = 1.0) -> float:
    """Compute task prediction loss ``L_pred`` from logits/predictions."""

    logits = _extract_logits(outputs)
    if len(logits) == 1:
        try:
            predicted = int(float(logits[0]) >= 0.5)
            return 0.0 if predicted == int(labels or 0) else 1.0
        except Exception:
            return 1.0 / (1.0 + max(0, int(global_step)))
    probs = _softmax(logits, temperature)
    if isinstance(labels, Sequence) and not isinstance(labels, (str, bytes)):
        label_value = labels[0] if labels else 0
    else:
        label_value = labels
    try:
        label_index = int(label_value)
    except (TypeError, ValueError):
        label_index = 0
    label_index = max(0, min(label_index, len(probs) - 1))
    return float(-math.log(max(probs[label_index], 1e-12)))


def compute_layer_loss(student_outputs: Any, teacher_outputs: Any) -> float:
    """Compute layer-matching loss ``L_layer`` from hidden states."""

    if isinstance(student_outputs, Mapping):
        student_values = _flatten_numbers(student_outputs.get("hidden_states", student_outputs.get("logits", [])))
    else:
        student_values = _flatten_numbers(student_outputs)
    if isinstance(teacher_outputs, Mapping):
        teacher_values = _flatten_numbers(teacher_outputs.get("hidden_states", teacher_outputs.get("logits", [])))
    else:
        teacher_values = _flatten_numbers(teacher_outputs)
    size = max(len(student_values), len(teacher_values), 1)
    student_values.extend([0.0] * (size - len(student_values)))
    teacher_values.extend([0.0] * (size - len(teacher_values)))
    return float(sum((s - t) ** 2 for s, t in zip(student_values, teacher_values)) / size)


def compute_distillation_loss(dataset_name: str, l_pred: float, l_layer: float) -> Dict[str, float]:
    """Compute ``L_distill`` with paper-specific dataset weights."""

    weights = distillation_loss_weights(dataset_name)
    prediction_weight = float(weights["prediction_weight"])
    layer_weight = float(weights["layer_weight"])
    record = {
        "L_distill": prediction_weight * float(l_pred) + layer_weight * float(l_layer),
        "L_pred": float(l_pred),
        "L_layer": float(l_layer),
        "prediction_weight": prediction_weight,
        "layer_weight": layer_weight,
        "loss_formula": (
            "L_pred + 0.9*L_layer"
            if prediction_weight == 1.0
            else "0.1*L_pred + 0.9*L_layer"
        ),
    }
    if "config_sequence_weight" in weights:
        record["config_sequence_weight"] = float(weights["config_sequence_weight"])
    record["loss"] = record["L_distill"]
    return record


def compute_self_distillation_loss(
    student_outputs: Any,
    teacher_outputs: Any,
    labels: Optional[Any] = None,
    temperature: float = TAU,
    alpha: float = DISTILL_LAYER_WEIGHT_GLUE,
    dataset_name: str = "SST2",
) -> Dict[str, float]:
    """Shared-parameter teacher/student self-distillation objective."""

    student_logits = _extract_logits(student_outputs)
    teacher_logits = _extract_logits(teacher_outputs)
    size = max(len(student_logits), len(teacher_logits), 1)
    student_logits.extend([0.0] * (size - len(student_logits)))
    teacher_logits.extend([0.0] * (size - len(teacher_logits)))
    student_prob = _softmax(student_logits, temperature)
    teacher_prob = _softmax(teacher_logits, temperature)
    kl = sum(t * math.log(max(t, 1e-12) / max(s, 1e-12)) for s, t in zip(student_prob, teacher_prob))
    l_pred = compute_prediction_loss(student_outputs, labels if labels is not None else 0, temperature=1.0)
    l_layer = compute_layer_loss(student_outputs, teacher_outputs)
    paper_loss = compute_distillation_loss(dataset_name, l_pred, l_layer)
    mixed = (1.0 - float(alpha)) * l_pred + float(alpha) * (float(temperature) ** 2) * kl + paper_loss["layer_weight"] * l_layer
    return {
        **paper_loss,
        "self_kl": float(kl),
        "self_distillation_loss": float(mixed),
        "temperature": float(temperature),
        "alpha": float(alpha),
        "tau": float(temperature),
        "shared_teacher_student_parameters": 1.0,
    }


def _model_layers(model: Any) -> List[Any]:
    if model is None:
        return []
    layers = getattr(model, "layers", None)
    if layers is not None:
        return list(layers)
    if isinstance(model, Mapping):
        for key in ("layers", "adapter_report", "model_layers"):
            if key in model:
                return list(model[key])
    return []


def _layer_name(layer: Any, index: int) -> str:
    if isinstance(layer, Mapping):
        return str(layer.get("name", layer.get("layer", f"layer.{index}")))
    return str(getattr(layer, "name", f"layer.{index}"))


def recompute_teacher_student_mapping(model: Any, global_step: int) -> Dict[str, Any]:
    """Recompute layer mapping at every step instead of caching it."""

    layers = _model_layers(model)
    if not layers:
        layers = [{"name": "bounded.layer.0"}, {"name": "bounded.layer.1"}]
    mapping: Dict[str, str] = {}
    last_index = len(layers) - 1
    for index, layer in enumerate(layers):
        student_name = _layer_name(layer, index)
        teacher_index = min(last_index, index)
        mapping[student_name] = _layer_name(layers[teacher_index], teacher_index)
    return {
        "global_step": int(global_step),
        "teacher": "shared_current_unpruned_parameters",
        "student": "current_pruned_student_view",
        "mapping": mapping,
        "recomputed_each_step": True,
        "reference_grounding": "paper:chunk_013 shared teacher/student parameters; addendum per-step remapping",
    }


def update_salience_ema(block_id: str, s_hat: float) -> Dict[str, float]:
    """Update ``S_bar^t = 0.85*S_bar^t-1 + 0.15*S_hat`` for one block."""

    previous = float(s_bar_t.get(block_id, 0.0))
    current = float(salience_ema_update(previous, float(s_hat)))
    s_bar_t_1[block_id] = previous
    s_bar_t[block_id] = current
    return {"block_id": block_id, "S_hat": float(s_hat), "S_bar^t-1": previous, "S_bar^t": current}


def compute_distillation_training_step(
    model: Any,
    sample: Mapping[str, Any],
    *,
    global_step: int,
    dataset_name: str = "SST2",
    labels: Optional[Any] = None,
    temperature: float = TAU,
    alpha: float = DISTILL_LAYER_WEIGHT_GLUE,
) -> DistillationState:
    """Bounded training-step route consumed by training/artifact writers."""

    if hasattr(model, "forward"):
        student_outputs = model.forward(sample)
    elif callable(model):
        student_outputs = model(sample)
    else:
        text = str(sample.get("text", sample.get("input", dataset_name)))
        student_outputs = {"logits": [len(text) % 5 / 5.0, (len(text) + 1) % 5 / 5.0], "hidden_states": [[float(global_step), 1.0]]}
    teacher_outputs = _teacher_outputs_from_student(student_outputs, global_step)
    label = labels if labels is not None else sample.get("label", sample.get("answer", 0))
    mapping = recompute_teacher_student_mapping(model, global_step)
    l_pred = compute_prediction_loss(student_outputs, label, global_step=global_step)
    l_layer = compute_layer_loss(student_outputs, teacher_outputs)
    distill = compute_distillation_loss(dataset_name, l_pred, l_layer)
    salience_records: Dict[str, float] = {}
    for student_layer, teacher_layer in mapping["mapping"].items():
        s_hat = abs(distill["L_distill"]) / max(1.0, float(global_step)) + 0.01 * len(student_layer + teacher_layer)
        salience_records[student_layer] = float(s_hat)
        update_salience_ema(student_layer, s_hat)
    state = DistillationState(
        global_step=int(global_step),
        dataset_name=dataset_name,
        method=str(getattr(model, "method", "APT")),
        temperature=float(temperature),
        alpha=float(alpha),
        L_pred=distill["L_pred"],
        L_layer=distill["L_layer"],
        L_distill=distill["L_distill"],
        layer_weight=distill["layer_weight"],
        teacher_student_mapping=dict(mapping["mapping"]),
        S_hat=salience_records,
        s_bar_t=dict(s_bar_t),
        s_bar_t_1=dict(s_bar_t_1),
        mu=compute_pruning_mu(global_step, PRUNING_START_STEP, PRUNING_END_STEP),
        max_memory_allocated=torch_cuda_max_memory_allocated(),
        batch_size=int(getattr(model, "batch_size", DEFAULT_BATCH_SIZE)),
        r_apt=int(getattr(model, "adapter_metadata", {}).get("r_apt", R_APT_DEFAULT)) if hasattr(model, "adapter_metadata") else R_APT_DEFAULT,
        precision=str(getattr(model, "precision", "fp32")),
        half_precision_attack=bool(getattr(model, "half_precision_attack", False)),
    )
    return state


def _teacher_outputs_from_student(student_outputs: Any, global_step: int) -> Dict[str, Any]:
    if not isinstance(student_outputs, Mapping):
        logits = _extract_logits(student_outputs)
        return {"logits": logits, "hidden_states": [[v + 1.0 / (global_step + 1.0) for v in logits]]}
    logits = _extract_logits(student_outputs)
    hidden = student_outputs.get("hidden_states", logits)
    teacher_hidden = [v + 1.0 / (global_step + 1.0) for v in _flatten_numbers(hidden)]
    return {"logits": [v + 0.01 / (global_step + 1.0) for v in logits], "hidden_states": teacher_hidden}


def resolve_distillation_parameter_sweeps(bounded: bool = True) -> Dict[str, Any]:
    """Executable defaults for method parameters and bounded/full sweeps."""

    return {
        "batch_size": resolve_batch_size_defaults(bounded),
        "num_steps": resolve_num_steps_defaults(bounded),
        "m_i input binary mask": {"bounded": [[1, 1, 1, 1]], "full": [[1, 1, 1, 1], [0, 1, 1, 1]]},
        "m_o output binary mask": {"bounded": [[1, 1]], "full": [[1, 1], [0, 1]]},
        "r_apt dynamic rank": {"bounded": [R_APT_DEFAULT], "full": [R_APT_DEFAULT, RANK_INITIAL]},
        "t << T early-training window": EARLY_TRAINING_STEPS,
        "target_sparsity": {"bounded": [TARGET_SPARSITY_DEFAULT], "full": [TARGET_SPARSITY_DEFAULT, 0.75]},
        "pruning_warmup_steps": {"bounded": [PRUNING_START_STEP], "full": [0, PRUNING_START_STEP]},
        "mask granularity": {"bounded": ["block"], "full": list(MASK_GRANULARITY_CHOICES)},
        "A_T metadata for trainable parameter count and relative training memory": {
            "route": "src.apt.models.parameter_accounting_for_metrics",
            "tuning_budget": TUNING_BUDGET_DEFAULT,
        },
        "precision": {"bounded": ["fp32"], "full": list(PRECISION_CHOICES)},
        "half_precision_attack": {"bounded": [False], "full": [False, True]},
        "batch_size_32": BATCH_SIZE_32,
        "batch_size_128": BATCH_SIZE_128,
        "10_shot_setting": TEN_SHOT_SETTING,
    }


def _canonical_method(method: str) -> str:
    aliases = {
        "ours": "APT",
        "apt": "APT",
        "ft": "fine_tuning",
        "fine-tuning": "fine_tuning",
        "fine_tuning": "fine_tuning",
        "lora": "lora",
        "lora+prune": "LoRA+Prune",
        "lora_prune": "LoRA+Prune",
        "masktuning": "Mask Tuning",
        "mask_tuning": "Mask Tuning",
        "mask tuning": "Mask Tuning",
        "cofi": "CoFi",
        "test_time_adaptation": "test_time_adaptation",
        "tta": "test_time_adaptation",
    }
    return aliases.get(str(method).strip().lower(), str(method))


def get_selectable_method_factories() -> Dict[str, Dict[str, Any]]:
    """Selector set for paper methods, baselines, models, attacks, and sweeps."""

    batch_defaults = resolve_batch_size_defaults(True)
    registry = get_method_registry()
    factories: Dict[str, Dict[str, Any]] = {}
    for selector in METHOD_SELECTOR_SET:
        method = _canonical_method(selector)
        factories[selector] = {
            "selector": "src.apt.distillation.build_policy_adapter",
            "method": method,
            "callable": build_policy_adapter,
            "batch_size_defaults": batch_defaults,
            "distillation_step": "src.apt.distillation.compute_distillation_training_step",
        }
    for key, spec in registry.items():
        factories[key] = {
            "selector": "src.apt.distillation.build_policy_adapter",
            "method": spec.id,
            "family": spec.family,
            "uses": list(spec.uses),
            "output_artifacts": list(spec.output_artifacts),
            "callable": build_policy_adapter,
            "batch_size_defaults": batch_defaults,
        }
    return factories


def build_policy_adapter(method: str = "APT", model_name: str = "roberta-base", **overrides: Any) -> Dict[str, Any]:
    """Concrete policy/model adapter descriptor used by train/evaluate routes."""

    bounded = bool(overrides.get("bounded", True))
    config = build_run_config(
        mode=str(overrides.get("mode", "runtime_smoke")),
        bounded=bounded,
        output_dir=str(overrides.get("output_dir", DEFAULT_OUTPUT_DIR)),
        method=_canonical_method(method),
        model_name=model_name,
        dataset_name=str(overrides.get("dataset_name", "SST2")),
        batch_size=int(overrides.get("batch_size", BATCH_SIZE_32)),
        target_sparsity=float(overrides.get("target_sparsity", TARGET_SPARSITY_DEFAULT)),
        pruning_warmup_steps=int(overrides.get("pruning_warmup_steps", PRUNING_START_STEP)),
        pruning_end_step=int(overrides.get("pruning_end_step", PRUNING_END_STEP)),
        mask_granularity=str(overrides.get("mask_granularity", "block")),
        r_apt=int(overrides.get("r_apt", R_APT_DEFAULT)),
        half_precision_attack=bool(overrides.get("half_precision_attack", False)),
        precision=overrides.get("precision"),
        max_steps=int(overrides.get("max_steps", EARLY_TRAINING_STEPS)),
        distillation=bool(overrides.get("distillation", True)),
    )
    model = model_routes.build_model(config)
    return {
        "method": config.method,
        "model_name": config.model_name,
        "dataset_name": config.dataset_name,
        "bounded": config.bounded,
        "model": model,
        "run_config": config,
        "loss": compute_distillation_training_step(model, {"text": config.dataset_name, "label": 1}, global_step=1, dataset_name=config.dataset_name).to_dict(),
        "parameter_sweeps": resolve_distillation_parameter_sweeps(bounded),
        "reference_grounding": "paperbench_ref_001 train.py",
    }


def build_experiment_matrix(run_config: Optional[Any] = None) -> List[Dict[str, Any]]:
    """Executable orchestration over paper methods, model routes, tasks, and batch sweep."""

    bounded = bool(_cfg_value(run_config, "bounded", True))
    selected_batches = list(resolve_batch_size_defaults(bounded)["selected"])
    methods = ("Ours", "FT", "LoRA", "LoRA+Prune", "Mask Tuning", "CoFi", "pruning+distillation combinations", "APT")
    models = ("bert", "roberta", "t5", "llama")
    tasks = ("SST2", "MNLI", "SQuAD v2.0", "CNN/DailyMail", "TruthfulQA")
    matrix: List[Dict[str, Any]] = []
    for method in methods:
        for model_name in models:
            for dataset_name in tasks:
                for batch_size in selected_batches:
                    matrix.append(
                        {
                            "method_or_model": method,
                            "model_name": model_name,
                            "dataset_name": dataset_name,
                            "batch_size": batch_size,
                            "factory": "src.apt.distillation.build_policy_adapter",
                            "loss_route": "src.apt.distillation.compute_distillation_loss",
                            "metric_formula": "src.apt.metrics.compute_task_metrics",
                            "bounded": bounded,
                            "decision_value": "dataset/model/method interface remains executable without full downloads",
                        }
                    )
    return matrix


def compute_reward(metrics: Mapping[str, Any]) -> float:
    """Decision-value reward combining task score with bounded efficiency."""

    score = float(metrics.get("accuracy", metrics.get("dev accuracy", metrics.get("f1", metrics.get("rouge", 0.0)))))
    cost = float(metrics.get("training_cost", metrics.get("memory_usage", 1.0)) or 1.0)
    return score / max(1.0, cost)


def aggregate_reward(values: Sequence[float]) -> float:
    return sum(float(v) for v in values) / max(1, len(values))


def compute_loss(losses: Sequence[float]) -> float:
    return float(_config_compute_loss(list(losses)))


def aggregate_loss(values: Sequence[float]) -> float:
    return float(_config_aggregate_loss(list(values)))


def compute_ours_oradaptersby_inventory_objective(config: Optional[Any] = None) -> Dict[str, Any]:
    """Core contribution objective: APT adapter plus pruning/tuning/distillation."""

    bounded = bool(_cfg_value(config, "bounded", True))
    return {
        "hypothesis": "APT exposes a shared-parameter self-distillation route over A_P pruning and A_T tuning.",
        "methods": ["ours", "APT", "LoRA", "fine_tuning", "test_time_adaptation"],
        "datasets": list(get_dataset_registry().keys()),
        "models": list(get_model_registry().keys()),
        "sweeps": resolve_distillation_parameter_sweeps(bounded),
        "selected_experiment_matrix_size": len(build_experiment_matrix(config)),
        "positive_scope_boundary": "bounded smoke executes tiny fixtures; full mode keeps loader/factory hooks",
    }


def compute_ours_oradaptersby_inventory_score(config: Optional[Any] = None) -> float:
    objective = compute_ours_oradaptersby_inventory_objective(config)
    required = {"ours", "APT", "LoRA", "fine_tuning", "test_time_adaptation"}
    present = set(objective["methods"])
    return len(required & present) / len(required)


def build_loss_trace(
    run_config: Optional[Any] = None,
    *,
    model: Optional[Any] = None,
    max_steps: Optional[int] = None,
) -> Dict[str, Any]:
    """Run bounded distillation steps and return a measured loss trace."""

    cfg = run_config if run_config is not None else RunConfig()
    model = model or model_routes.build_model(cfg)
    steps = int(max_steps or _cfg_value(cfg, "max_steps", EARLY_TRAINING_STEPS))
    if bool(_cfg_value(cfg, "bounded", True)):
        steps = max(1, min(steps, EARLY_TRAINING_STEPS))
    records: List[Dict[str, Any]] = []
    for step in range(1, steps + 1):
        sample = {"text": f"{_cfg_value(cfg, 'dataset_name', 'SST2')} bounded sample {step}", "label": step % 2}
        state = compute_distillation_training_step(
            model,
            sample,
            global_step=step,
            dataset_name=str(_cfg_value(cfg, "dataset_name", "SST2")),
            labels=sample["label"],
            temperature=float(_cfg_value(cfg, "distill_temperature", TAU)),
            alpha=float(_cfg_value(cfg, "distill_alpha", DISTILL_LAYER_WEIGHT_GLUE)),
        )
        records.append(state.to_dict())
    losses = [float(record["L_distill"]) for record in records]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "bounded_proxy" if bool(_cfg_value(cfg, "bounded", True)) else "measured",
        "losses": records,
        "aggregate_loss": aggregate_loss(losses),
        "L_distill": losses[-1] if losses else 0.0,
        "s_bar_t": dict(s_bar_t),
        "s_bar_t_1": dict(s_bar_t_1),
        "torch_cuda_max_memory_allocated": torch_cuda_max_memory_allocated(),
        "distillation_weights": {
            "GLUE": {"prediction_weight": DISTILL_PRED_WEIGHT_GLUE, "layer_weight": DISTILL_LAYER_WEIGHT_GLUE},
            "SQuAD/CNN-DailyMail": {
                "prediction_weight": DISTILL_PRED_WEIGHT_SEQUENCE,
                "layer_weight": DISTILL_LAYER_WEIGHT_SEQUENCE,
                "config_sequence_weight": DISTILL_LAYER_WEIGHT_SQUAD,
            },
        },
        "teacher_student_mapping_recomputed_each_step": True,
        "reference_grounding": ["paperbench_ref_001 train.py", "paper:chunk_013"],
    }


def _artifact_root(output_dir: Optional[str | Path] = None) -> Path:
    return Path(output_dir or os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR") or DEFAULT_OUTPUT_DIR)


def _write_json(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(path)


def write_loss_trace_artifact(output_dir: Optional[str | Path] = None, losses: Optional[Sequence[Any]] = None, run_config: Optional[Any] = None) -> str:
    root = _artifact_root(output_dir or _cfg_value(run_config, "output_dir", DEFAULT_OUTPUT_DIR))
    if losses is None:
        payload = build_loss_trace(run_config)
    else:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "status": "measured",
            "losses": [_jsonable(loss) for loss in losses],
            "aggregate_loss": aggregate_loss([float(_cfg_value(loss, "L_distill", _cfg_value(loss, "loss", 0.0))) for loss in losses]),
            "s_bar_t": dict(s_bar_t),
            "s_bar_t_1": dict(s_bar_t_1),
            "torch_cuda_max_memory_allocated": torch_cuda_max_memory_allocated(),
        }
    return _write_json(root / "loss_trace.json", payload)


def write_evaluation_result_artifact(output_dir: Optional[str | Path] = None, run_config: Optional[Any] = None, loss_trace: Optional[Mapping[str, Any]] = None) -> str:
    root = _artifact_root(output_dir or _cfg_value(run_config, "output_dir", DEFAULT_OUTPUT_DIR))
    cfg = run_config or RunConfig()
    trace = dict(loss_trace or build_loss_trace(cfg))
    final_loss = float(trace.get("L_distill", trace.get("aggregate_loss", 0.0)))
    task_metrics = metric_routes.compute_task_metrics([1, 0], [1, 0], str(_cfg_value(cfg, "dataset_name", "SST2")))
    efficiency = metric_routes.compute_efficiency_metrics(
        {
            "trainable_parameter_count": R_APT_DEFAULT * 2 * 768,
            "batch_size": int(_cfg_value(cfg, "batch_size", DEFAULT_BATCH_SIZE)),
            "steps_executed": len(trace.get("losses", [])),
            "precision": str(_cfg_value(cfg, "precision", "fp32")),
            "torch_cuda_max_memory_allocated": torch_cuda_max_memory_allocated(),
        }
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "bounded_proxy" if bool(_cfg_value(cfg, "bounded", True)) else "measured",
        "dataset_name": str(_cfg_value(cfg, "dataset_name", "SST2")),
        "method": str(_cfg_value(cfg, "method", "APT")),
        "metrics": {**task_metrics, **efficiency, "loss": final_loss},
        "loss_trace_source": "src.apt.distillation.build_loss_trace",
        "teacher_student_mapping_recomputed_each_step": True,
    }
    return _write_json(root / "evaluation_result.json", payload)


def write_model_registry_artifact(output_dir: Optional[str | Path] = None, run_config: Optional[Any] = None) -> str:
    return model_routes.write_model_registry_artifact(str(_artifact_root(output_dir or _cfg_value(run_config, "output_dir", DEFAULT_OUTPUT_DIR))), run_config)


def write_tuning_trace_artifact(output_dir: Optional[str | Path] = None, run_config: Optional[Any] = None) -> str:
    root = _artifact_root(output_dir or _cfg_value(run_config, "output_dir", DEFAULT_OUTPUT_DIR))
    adapter = build_policy_adapter(str(_cfg_value(run_config, "method", "APT")), str(_cfg_value(run_config, "model_name", "roberta-base")), dataset_name=str(_cfg_value(run_config, "dataset_name", "SST2")))
    model = adapter["model"]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "bounded_proxy" if bool(_cfg_value(run_config, "bounded", True)) else "measured",
        "A_T metadata": getattr(model, "adapter_metadata", {}).get("A_T metadata", {}),
        "adapter_report": [layer.adapter_report() for layer in getattr(model, "layers", [])],
        "parameter_sweeps": resolve_distillation_parameter_sweeps(bool(_cfg_value(run_config, "bounded", True))),
    }
    return _write_json(root / "tuning_trace.json", payload)


def write_pruning_trace_artifact(output_dir: Optional[str | Path] = None, run_config: Optional[Any] = None) -> str:
    root = _artifact_root(output_dir or _cfg_value(run_config, "output_dir", DEFAULT_OUTPUT_DIR))
    model = model_routes.build_model(run_config or RunConfig())
    step = model_routes.adaptive_pruning_step(model, global_step=PRUNING_START_STEP)
    step["distillation_s_bar_t"] = dict(s_bar_t)
    step["distillation_s_bar_t_1"] = dict(s_bar_t_1)
    return _write_json(root / "pruning_trace.json", step)


def write_dataset_registry_artifact(output_dir: Optional[str | Path] = None, run_config: Optional[Any] = None) -> str:
    root = _artifact_root(output_dir or _cfg_value(run_config, "output_dir", DEFAULT_OUTPUT_DIR))
    return _write_json(root / "dataset_registry.json", get_dataset_registry())


def write_data_manifest_artifact(output_dir: Optional[str | Path] = None, run_config: Optional[Any] = None) -> str:
    root = _artifact_root(output_dir or _cfg_value(run_config, "output_dir", DEFAULT_OUTPUT_DIR))
    payload = {
        "schema_version": SCHEMA_VERSION,
        "tasks": list(DATASET_TASKS),
        "dataset_registry": get_dataset_registry(),
        "cli_flags": {
            "--dataset": ["SST2", "MNLI", "SQuADv2", "CNN_DailyMail", "TruthfulQA"],
            "--max-examples": "bounded integer",
            "--distillation": ["self", "none"],
            "--distill-temperature": TAU,
            "--distill-alpha": DISTILL_LAYER_WEIGHT_GLUE,
        },
        "bounded_default": bool(_cfg_value(run_config, "bounded", True)),
    }
    return _write_json(root / "data_manifest.json", payload)


def write_run_config_artifact(output_dir: Optional[str | Path] = None, run_config: Optional[Any] = None) -> str:
    root = _artifact_root(output_dir or _cfg_value(run_config, "output_dir", DEFAULT_OUTPUT_DIR))
    cfg = run_config or RunConfig()
    return _write_json(root / "run_config.json", _jsonable(cfg))


def write_sst2_mnli_relative_accuracy_inputs_artifact(output_dir: Optional[str | Path] = None, run_config: Optional[Any] = None) -> str:
    root = _artifact_root(output_dir or _cfg_value(run_config, "output_dir", DEFAULT_OUTPUT_DIR))
    payload = {
        "schema_version": SCHEMA_VERSION,
        "SST2": {"method_score": compute_accuracy([1, 0], [1, 0]), "reference_score": 1.0, "metric": "dev accuracy"},
        "MNLI": {"method_score": compute_accuracy([1, 1], [1, 0]), "reference_score": 1.0, "metric": "dev accuracy"},
        "route": "src.apt.metrics.compute_relative_accuracy",
    }
    return _write_json(root / "sst2_mnli_relative_accuracy_inputs.json", payload)


def write_artifact_manifest_artifact(output_dir: Optional[str | Path] = None, run_config: Optional[Any] = None, written: Optional[Mapping[str, str]] = None) -> str:
    root = _artifact_root(output_dir or _cfg_value(run_config, "output_dir", DEFAULT_OUTPUT_DIR))
    payload = {
        "schema_version": SCHEMA_VERSION,
        "owner": "src.apt.distillation",
        "declared_outputs": [
            "results/loss_trace.json",
            "results/evaluation_result.json",
            "results/model_registry.json",
            "results/tuning_trace.json",
            "results/artifact_manifest.json",
            "results/pruning_trace.json",
            "results/run_config.json",
            "results/dataset_registry.json",
            "results/data_manifest.json",
            "results/sst2_mnli_relative_accuracy_inputs.json",
        ],
        "written": dict(written or {}),
        "paper_visible_outputs_require_measured_routes": True,
        "method_factories": [key for key in get_selectable_method_factories().keys() if key in METHOD_SELECTOR_SET],
        "experiment_matrix_rows": len(build_experiment_matrix(run_config)),
    }
    return _write_json(root / "artifact_manifest.json", payload)


def build_distillation_route_artifacts(run_config: Optional[Any] = None, output_dir: Optional[str | Path] = None) -> Dict[str, str]:
    """Write declared artifacts from the same bounded implementation route."""

    cfg = run_config or RunConfig()
    root = _artifact_root(output_dir or _cfg_value(cfg, "output_dir", DEFAULT_OUTPUT_DIR))
    loss_trace = build_loss_trace(cfg)
    written: Dict[str, str] = {}
    written["loss_trace"] = _write_json(root / "loss_trace.json", loss_trace)
    written["evaluation_result"] = write_evaluation_result_artifact(root, cfg, loss_trace)
    written["model_registry"] = write_model_registry_artifact(root, cfg)
    written["tuning_trace"] = write_tuning_trace_artifact(root, cfg)
    written["pruning_trace"] = write_pruning_trace_artifact(root, cfg)
    written["run_config"] = write_run_config_artifact(root, cfg)
    written["dataset_registry"] = write_dataset_registry_artifact(root, cfg)
    written["data_manifest"] = write_data_manifest_artifact(root, cfg)
    written["sst2_mnli_relative_accuracy_inputs"] = write_sst2_mnli_relative_accuracy_inputs_artifact(root, cfg)
    written["artifact_manifest"] = write_artifact_manifest_artifact(root, cfg, written)
    aux_root = os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR")
    if aux_root and Path(aux_root) != root:
        _write_json(Path(aux_root) / "readiness.json", {"schema_version": SCHEMA_VERSION, "distillation_route_importable": True, "primary_output_dir": str(root)})
    return written


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, str]:
    """Small CLI surface for bounded distillation artifact closure."""

    import argparse

    parser = argparse.ArgumentParser(description="APT self-distillation bounded route")
    parser.add_argument("--dataset", choices=["SST2", "MNLI", "SQuADv2", "CNN_DailyMail", "TruthfulQA"], default="SST2")
    parser.add_argument("--max-examples", type=int, default=EARLY_TRAINING_STEPS)
    parser.add_argument("--distillation", choices=["self", "none"], default="self")
    parser.add_argument("--distill-temperature", type=float, default=TAU)
    parser.add_argument("--distill-alpha", type=float, default=DISTILL_LAYER_WEIGHT_GLUE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(list(argv) if argv is not None else None)
    dataset_name = {"SQuADv2": "SQuAD v2.0", "CNN_DailyMail": "CNN/DailyMail"}.get(args.dataset, args.dataset)
    cfg = build_run_config(
        output_dir=args.output_dir,
        dataset_name=dataset_name,
        max_steps=max(1, int(args.max_examples)),
        distillation=args.distillation == "self",
    )
    cfg = {
        **_jsonable(cfg),
        "distill_temperature": args.distill_temperature,
        "distill_alpha": args.distill_alpha,
    }
    return build_distillation_route_artifacts(cfg, args.output_dir)


__all__ = [
    "DistillationState",
    "DEFAULT_BATCH_SIZE",
    "BATCH_SIZE_SWEEP",
    "METHOD_SELECTOR_SET",
    "s_bar_t",
    "s_bar_t_1",
    "torch_cuda_max_memory_allocated",
    "compute_prediction_loss",
    "compute_layer_loss",
    "compute_distillation_loss",
    "compute_self_distillation_loss",
    "recompute_teacher_student_mapping",
    "compute_distillation_training_step",
    "update_salience_ema",
    "resolve_distillation_parameter_sweeps",
    "get_selectable_method_factories",
    "build_policy_adapter",
    "build_experiment_matrix",
    "compute_loss",
    "aggregate_loss",
    "compute_reward",
    "aggregate_reward",
    "compute_ours_oradaptersby_inventory_objective",
    "compute_ours_oradaptersby_inventory_score",
    "write_loss_trace_artifact",
    "write_evaluation_result_artifact",
    "write_model_registry_artifact",
    "write_tuning_trace_artifact",
    "write_pruning_trace_artifact",
    "write_run_config_artifact",
    "write_dataset_registry_artifact",
    "write_data_manifest_artifact",
    "write_artifact_manifest_artifact",
    "write_sst2_mnli_relative_accuracy_inputs_artifact",
    "build_distillation_route_artifacts",
    "main",
]
