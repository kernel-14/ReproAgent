"""Adaptive tuning route for APT.

This module owns the paper-visible A_T mechanism: tuning layer importance,
dynamic rank allocation ``R_t``/``r_apt`` under the ``Delta_t`` budget, and
metric-consumable A_T metadata for trainable parameters and memory formulas.
It is deliberately dependency-light; full tensor/model backends are reached
through lazy factories in sibling modules.

reference_grounding: paperbench_ref_001 train.py
reference_grounding: paperbench_ref_001 datasheet.md
reference_grounding: paperbench_ref_001 prompt.txt
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from .config import (
    ALPHA_DEFAULT,
    BATCH_SIZE_128,
    BATCH_SIZE_32,
    DELTA_T_DEFAULT,
    EARLY_TRAINING_STEPS,
    MASK_GRANULARITY_CHOICES,
    PRECISION_CHOICES,
    PRUNING_END_STEP,
    PRUNING_START_STEP,
    RANK_INITIAL,
    R_APT_DEFAULT,
    R_T_DEFAULT,
    SALIENCE_EMA_DECAY,
    SALIENCE_EMA_UPDATE,
    TARGET_SPARSITY_DEFAULT,
    TEN_SHOT_SETTING,
    TUNING_BUDGET_DEFAULT,
    aggregate_loss as _config_aggregate_loss,
    compute_loss as _config_compute_loss,
    compute_pruning_mu,
    resolve_batch_size_defaults as _config_resolve_batch_size_defaults,
    resolve_num_steps_defaults,
)
from . import adapters as adapter_routes
from . import metrics as metric_routes
from . import models as model_routes


SCHEMA_VERSION = "1.0"
DEFAULT_OUTPUT_DIR = "results"
DEFAULT_METHOD = "APT"
DEFAULT_MODEL_NAME = "roberta-base"
DEFAULT_DATASET_NAME = "SST2"
DEFAULT_BATCH_SIZE = BATCH_SIZE_32
DEFAULT_R_APT = R_APT_DEFAULT
DEFAULT_R_T = R_T_DEFAULT
DEFAULT_DELTA_T = DELTA_T_DEFAULT
DEFAULT_TUNING_BUDGET = TUNING_BUDGET_DEFAULT
DEFAULT_TARGET_SPARSITY = TARGET_SPARSITY_DEFAULT
EARLY_TRAINING_WINDOW_T_LT_T = EARLY_TRAINING_STEPS
S_BAR_EMA_DECAY = SALIENCE_EMA_DECAY
S_BAR_EMA_UPDATE = SALIENCE_EMA_UPDATE

BATCH_SIZE_SWEEP = (BATCH_SIZE_32, BATCH_SIZE_128)
batch_size_values = BATCH_SIZE_SWEEP
PRECISION_SWEEP = PRECISION_CHOICES
HALF_PRECISION_ATTACK_SWEEP = (False, True)
MASK_GRANULARITY_SWEEP = MASK_GRANULARITY_CHOICES

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

PAPER_VISIBLE_ARTIFACTS = (
    "results/model_registry.json",
    "results/pruning_trace.json",
    "results/tuning_trace.json",
    "results/loss_trace.json",
    "results/config_resolved.json",
    "results/sensitivity_report.json",
    "results/result_table.json",
    "results/result_table.md",
)

APT_NLU_JOINT_EXPERIMENT = "apt_nlu_joint_prune_tune"
APT_GENERATION_EXPERIMENT = "apt_generation_instruction_coverage"
BASELINE_EFFICIENCY_CONTRACT_EXPERIMENT = "baseline_relative_efficiency_artifact_contract"
globals()["APT在NLU任务上的联合剪枝与调参复现实验"] = APT_NLU_JOINT_EXPERIMENT
globals()["APT在生成与指令接口上的任务覆盖实验"] = APT_GENERATION_EXPERIMENT
globals()["基线比较、相对效率指标与可见工件契约实验"] = BASELINE_EFFICIENCY_CONTRACT_EXPERIMENT
globals()["配置、注册表与运行入口模块"] = "src.apt.config.RunConfig -> src.apt.training.run_training"
globals()["构建数据集注册表"] = "src.apt.config.get_dataset_registry"
globals()["构建方法与基线注册表"] = "src.apt.tuning.get_method_variant_factories"
globals()["APT adapter与LoRA基础适配器模块"] = "src.apt.adapters.create_apt_adapter"

# Live S_bar state read by training.py, models.py, metrics.py, and adapters.py.
s_bar_t: Dict[str, float] = {}
s_bar_t_1: Dict[str, float] = {}


@dataclass
class TuningState:
    """Runtime state for the A_T adaptive tuning algorithm."""

    method: str = DEFAULT_METHOD
    model_name: str = DEFAULT_MODEL_NAME
    dataset_name: str = DEFAULT_DATASET_NAME
    global_step: int = 0
    base_rank: int = DEFAULT_R_APT
    r_apt: int = DEFAULT_R_APT
    R_t: int = DEFAULT_R_T
    Delta_t: int = DEFAULT_DELTA_T
    tuning_budget: int = DEFAULT_TUNING_BUDGET
    target_sparsity: float = DEFAULT_TARGET_SPARSITY
    pruning_start_step: int = PRUNING_START_STEP
    pruning_end_step: int = PRUNING_END_STEP
    mask_granularity: str = "block"
    precision: str = "fp32"
    half_precision_attack: bool = False
    batch_size: int = DEFAULT_BATCH_SIZE
    m_i: List[int] = field(default_factory=lambda: [1, 1, 1, 1])
    m_o: List[int] = field(default_factory=lambda: [1, 1])
    salience_ema: Dict[str, float] = field(default_factory=dict)
    salience_ema_previous: Dict[str, float] = field(default_factory=dict)
    tuning_layer_importance: Dict[str, float] = field(default_factory=dict)
    rank_allocation: Dict[str, int] = field(default_factory=dict)
    rank_history: List[Dict[str, Any]] = field(default_factory=list)
    adapter_metadata: Dict[str, Any] = field(default_factory=dict)
    pruning_metadata: Dict[str, Any] = field(default_factory=dict)
    trace_records: List[Dict[str, Any]] = field(default_factory=list)
    reference_grounding: str = "paper:chunk_012 Adaptive and Efficient LM Tuning A_T"

    def __post_init__(self) -> None:
        self.base_rank = max(1, int(self.base_rank))
        self.r_apt = max(1, int(self.r_apt))
        self.R_t = max(1, int(self.R_t))
        self.Delta_t = max(1, int(self.Delta_t))
        self.tuning_budget = max(1, int(self.tuning_budget))
        self.batch_size = max(1, int(self.batch_size))
        if self.mask_granularity not in MASK_GRANULARITY_CHOICES:
            raise ValueError(f"mask_granularity must be one of {MASK_GRANULARITY_CHOICES}")
        if self.precision not in PRECISION_CHOICES:
            raise ValueError(f"precision must be one of {PRECISION_CHOICES}")
        self.m_i = _binary_mask(self.m_i)
        self.m_o = _binary_mask(self.m_o)

    def to_dict(self) -> Dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class MethodVariant:
    """Selectable method/baseline row backed by executable package routes."""

    id: str
    aliases: Sequence[str]
    family: str
    adapter_factory: str
    tuning_route: str
    pruning_route: str
    bounded_defaults: Mapping[str, Any] = field(default_factory=dict)
    full_mode_requirements: Sequence[str] = field(default_factory=tuple)
    reference_grounding: str = "paperbench_ref_001 train.py"

    def to_dict(self) -> Dict[str, Any]:
        return _jsonable(asdict(self))


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


def _cfg_value(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


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


def _binary_mask(values: Optional[Sequence[Any]], width: Optional[int] = None) -> List[int]:
    if values is None:
        values = []
    mask = [1 if int(v) else 0 for v in values]
    if width is not None:
        width = max(1, int(width))
        if len(mask) < width:
            mask.extend([1] * (width - len(mask)))
        mask = mask[:width]
    if not mask:
        mask = [1] * max(1, int(width or 1))
    return mask


def _model_layers(model: Any) -> List[Any]:
    layers = getattr(model, "layers", None)
    if layers is None and isinstance(model, Mapping):
        layers = model.get("layers")
    if layers is None:
        return []
    return list(layers)


def _layer_name(layer: Any, index: int) -> str:
    if isinstance(layer, Mapping):
        return str(layer.get("name", f"layer.{index}"))
    return str(getattr(layer, "name", f"layer.{index}"))


def _layer_rank(layer: Any, default: int = DEFAULT_R_APT) -> int:
    if isinstance(layer, Mapping):
        return max(1, _safe_int(layer.get("r_apt", layer.get("rank", default)), default))
    return max(1, _safe_int(getattr(layer, "r_apt", getattr(layer, "rank", default)), default))


def _set_layer_rank(layer: Any, rank: int) -> None:
    rank = max(1, int(rank))
    updater = getattr(layer, "update_rank", None)
    if callable(updater):
        updater(rank)
        return
    if isinstance(layer, MutableMapping):
        layer["r_apt"] = rank
        layer["R_t"] = rank
        return
    if hasattr(layer, "r_apt"):
        setattr(layer, "r_apt", rank)


def _mask_density_from_layer(layer: Any) -> float:
    if isinstance(layer, Mapping):
        masks = [layer.get("input_mask", layer.get("m_i", [])), layer.get("output_mask", layer.get("m_o", []))]
    else:
        masks = [getattr(layer, "input_mask", getattr(layer, "m_i", [])), getattr(layer, "output_mask", getattr(layer, "m_o", []))]
    values: List[int] = []
    for mask in masks:
        values.extend(_binary_mask(list(mask) if mask is not None else []))
    if not values:
        return 1.0
    return sum(values) / max(1, len(values))


def _loss_scale(loss_record: Optional[Mapping[str, Any]]) -> float:
    record = dict(loss_record or {})
    loss = _safe_float(record.get("L_distill", record.get("loss", record.get("L_pred", 0.0))), 0.0)
    layer = _safe_float(record.get("L_layer", 0.0), 0.0)
    return 1.0 + max(0.0, loss) + 0.1 * max(0.0, layer)


def _sparsity_progress(global_step: int, target_sparsity: float, start: int, end: int) -> float:
    return float(target_sparsity) * compute_pruning_mu(global_step, start, end)


def sync_salience_state(block_id: str, s_hat: float, *, global_step: Optional[int] = None) -> Dict[str, float]:
    """Update local and sibling-module ``S_bar`` EMA state.

    The formula follows the addendum route:
    ``S_bar^t = 0.85 * S_bar^(t-1) + 0.15 * S_hat``.
    """

    previous = float(s_bar_t.get(str(block_id), 0.0))
    try:
        record = adapter_routes.compute_salience_ema(str(block_id), float(s_hat), global_step=global_step)
        current = float(record.get("S_bar^t", SALIENCE_EMA_DECAY * previous + SALIENCE_EMA_UPDATE * float(s_hat)))
        previous = float(record.get("S_bar^t-1", previous))
    except Exception:
        current = SALIENCE_EMA_DECAY * previous + SALIENCE_EMA_UPDATE * float(s_hat)
        record = {
            "block_id": str(block_id),
            "global_step": None if global_step is None else int(global_step),
            "S_hat": float(s_hat),
            "S_bar^t-1": previous,
            "S_bar^t": current,
        }
    s_bar_t_1[str(block_id)] = previous
    s_bar_t[str(block_id)] = current
    for module in (adapter_routes, metric_routes, model_routes):
        if hasattr(module, "s_bar_t"):
            module.s_bar_t[str(block_id)] = current
        if hasattr(module, "s_bar_t_1"):
            module.s_bar_t_1[str(block_id)] = previous
    record.update({"s_bar_t": current, "s_bar_t_1": previous, "ema_decay": SALIENCE_EMA_DECAY, "ema_update": SALIENCE_EMA_UPDATE})
    return _jsonable(record)


def compute_tuning_layer_importance(
    model: Any,
    salience_ema: Mapping[str, float],
    loss_record: Optional[Mapping[str, Any]] = None,
) -> Dict[str, float]:
    """Compute task-sensitive APT adapter importance for A_T.

    Importance combines A_P salience state, current distillation/task loss, mask
    density after pruning, and existing layer rank.  Higher scores receive more
    dynamic tuning rank from ``allocate_dynamic_rank``.
    """

    loss_scale = _loss_scale(loss_record)
    layers = _model_layers(model)
    layer_lookup = {_layer_name(layer, idx): layer for idx, layer in enumerate(layers)}
    keys = list(salience_ema.keys()) or list(layer_lookup.keys()) or ["layer.0"]
    importance: Dict[str, float] = {}
    for idx, layer_id in enumerate(keys):
        layer = layer_lookup.get(str(layer_id), layers[idx] if idx < len(layers) else None)
        salience = max(0.0, _safe_float(salience_ema.get(layer_id, s_bar_t.get(str(layer_id), 0.0))))
        if not salience and str(layer_id) in s_bar_t:
            salience = float(s_bar_t[str(layer_id)])
        mask_density = _mask_density_from_layer(layer) if layer is not None else 1.0
        current_rank = _layer_rank(layer, DEFAULT_R_APT) if layer is not None else DEFAULT_R_APT
        rank_pressure = 1.0 / math.sqrt(max(1.0, float(current_rank)))
        score = salience * loss_scale * (0.5 + 0.5 * mask_density) * rank_pressure
        importance[str(layer_id)] = float(score)
    total = sum(importance.values())
    if total <= 0.0:
        uniform = 1.0 / max(1, len(importance))
        importance = {name: uniform for name in importance}
    else:
        importance = {name: value / total for name, value in importance.items()}
    return importance


def allocate_dynamic_rank(
    importance: Mapping[str, float],
    *,
    base_rank: int = DEFAULT_R_APT,
    tuning_budget: int = DEFAULT_TUNING_BUDGET,
    delta_t: int = DEFAULT_DELTA_T,
    max_rank: Optional[int] = None,
) -> Dict[str, int]:
    """Allocate ``r_apt``/``R_t`` under the paper's ``Delta_t`` constraint."""

    if not importance:
        return {}
    base_rank = max(1, int(base_rank))
    delta_t = max(1, int(delta_t))
    max_rank = max_rank if max_rank is not None else base_rank + RANK_INITIAL
    max_rank = max(base_rank, int(max_rank))
    names = [str(name) for name in importance.keys()]
    scores = {str(k): max(0.0, _safe_float(v)) for k, v in importance.items()}
    total_score = sum(scores.values()) or float(len(scores))
    budget = max(len(names), int(tuning_budget))

    allocation: Dict[str, int] = {}
    for name in names:
        share = scores[name] / total_score if total_score else 1.0 / len(names)
        dynamic_slots = int(round(budget * share / delta_t))
        allocation[name] = max(1, min(max_rank, base_rank + dynamic_slots))

    # Correct rounding drift while keeping at least one rank per selected layer.
    max_total_rank = max(len(names), budget + len(names) * base_rank)
    while sum(allocation.values()) > max_total_rank:
        candidate = max(allocation, key=lambda key: allocation[key])
        if allocation[candidate] <= 1:
            break
        allocation[candidate] -= 1
    return allocation


def tuning_parameter_delta(rank_allocation: Mapping[str, int], model: Optional[Any] = None) -> int:
    """Delta_t-style count of dynamic APT tuning parameters."""

    layers = _model_layers(model)
    layer_lookup = {_layer_name(layer, idx): layer for idx, layer in enumerate(layers)}
    total = 0
    for layer_id, rank in rank_allocation.items():
        layer = layer_lookup.get(str(layer_id))
        if layer is None:
            total += max(1, int(rank)) * 2
            continue
        d_i = _safe_int(getattr(layer, "d_i", None) if not isinstance(layer, Mapping) else layer.get("d_i"), 1)
        d_o = _safe_int(getattr(layer, "d_o", None) if not isinstance(layer, Mapping) else layer.get("d_o"), 1)
        total += max(1, int(rank)) * max(1, d_i + d_o)
    return int(total)


def count_trainable_parameters(model: Any) -> int:
    """A_T metadata dependency wrapper."""

    try:
        return int(model_routes.count_trainable_parameters(model))
    except Exception:
        try:
            return int(adapter_routes.count_trainable_parameters(model))
        except Exception:
            return 0


def compute_loss(losses: Sequence[float]) -> float:
    if hasattr(metric_routes, "compute_loss"):
        return float(metric_routes.compute_loss(losses))
    return float(_config_compute_loss(losses))


def aggregate_loss(values: Sequence[float]) -> float:
    if hasattr(metric_routes, "aggregate_loss"):
        return float(metric_routes.aggregate_loss(values))
    return float(_config_aggregate_loss(values))


def compute_reward(metrics: Mapping[str, Any]) -> float:
    if hasattr(metric_routes, "compute_reward"):
        return float(metric_routes.compute_reward(metrics))
    if hasattr(adapter_routes, "compute_reward"):
        return float(adapter_routes.compute_reward(metrics))
    return _safe_float(metrics.get("accuracy", 0.0)) - _safe_float(metrics.get("loss", 0.0))


def aggregate_reward(values: Sequence[float]) -> float:
    values = [_safe_float(v) for v in values]
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def compute_ours_oradaptersby_inventory_objective(metrics: Mapping[str, Any]) -> float:
    """Decision objective: task reward minus A_T training/memory costs."""

    reward = compute_reward(metrics)
    training_cost = _safe_float(metrics.get("training_cost", 0.0))
    memory_usage = _safe_float(metrics.get("memory_usage", metrics.get("gpu_memory", 0.0)))
    return float(reward - 0.01 * training_cost - 0.001 * memory_usage)


def compute_ours_oradaptersby_inventory_score(metrics: Mapping[str, Any]) -> Dict[str, float]:
    objective = compute_ours_oradaptersby_inventory_objective(metrics)
    return {
        "ours_oradaptersby_inventory_objective": objective,
        "reward": compute_reward(metrics),
        "training_cost": _safe_float(metrics.get("training_cost", 0.0)),
        "memory_usage": _safe_float(metrics.get("memory_usage", metrics.get("gpu_memory", 0.0))),
    }


def build_tuning_state(config: Optional[Any] = None, model: Optional[Any] = None) -> TuningState:
    """Create A_T state from RunConfig-like objects or dicts."""

    return TuningState(
        method=str(_cfg_value(config, "method", getattr(model, "method", DEFAULT_METHOD))),
        model_name=str(_cfg_value(config, "model_name", getattr(model, "model_name", DEFAULT_MODEL_NAME))),
        dataset_name=str(_cfg_value(config, "dataset_name", getattr(model, "dataset_name", DEFAULT_DATASET_NAME))),
        base_rank=int(_cfg_value(config, "r_apt", DEFAULT_R_APT)),
        r_apt=int(_cfg_value(config, "r_apt", DEFAULT_R_APT)),
        R_t=int(_cfg_value(config, "R_t", DEFAULT_R_T)),
        Delta_t=int(_cfg_value(config, "Delta_t", DEFAULT_DELTA_T)),
        tuning_budget=int(_cfg_value(config, "tuning_budget", DEFAULT_TUNING_BUDGET)),
        target_sparsity=float(_cfg_value(config, "target_sparsity", _cfg_value(config, "sparsity", DEFAULT_TARGET_SPARSITY))),
        pruning_start_step=int(_cfg_value(config, "pruning_warmup_steps", PRUNING_START_STEP)),
        pruning_end_step=int(_cfg_value(config, "pruning_end_step", PRUNING_END_STEP)),
        mask_granularity=str(_cfg_value(config, "mask_granularity", "block")),
        precision=str(_cfg_value(config, "precision", getattr(model, "precision", "fp32"))),
        half_precision_attack=bool(_cfg_value(config, "half_precision_attack", getattr(model, "half_precision_attack", False))),
        batch_size=int(_cfg_value(config, "batch_size", getattr(model, "batch_size", DEFAULT_BATCH_SIZE))),
    )


def update_tuning_state(
    model: Any,
    adapter_state: Any,
    rank_allocation: Mapping[str, int],
    *,
    global_step: int,
) -> Dict[str, Any]:
    """Apply dynamic ranks and return metric/artifact-consumable A_T metadata."""

    ranks = {str(k): max(1, int(v)) for k, v in dict(rank_allocation or {}).items()}
    for idx, layer in enumerate(_model_layers(model)):
        layer_id = _layer_name(layer, idx)
        _set_layer_rank(layer, ranks.get(layer_id, getattr(adapter_state, "r_apt", DEFAULT_R_APT)))

    if ranks:
        mean_rank = max(1, int(round(sum(ranks.values()) / len(ranks))))
    else:
        mean_rank = max(1, _safe_int(getattr(adapter_state, "r_apt", DEFAULT_R_APT), DEFAULT_R_APT))
    if hasattr(adapter_state, "r_apt"):
        setattr(adapter_state, "r_apt", mean_rank)
    if hasattr(adapter_state, "R_t"):
        setattr(adapter_state, "R_t", max(ranks.values()) if ranks else mean_rank)
    if hasattr(adapter_state, "global_step"):
        setattr(adapter_state, "global_step", int(global_step))

    trainable = count_trainable_parameters(model)
    delta_params = tuning_parameter_delta(ranks, model)
    if trainable <= 0:
        trainable = delta_params
    precision = str(getattr(adapter_state, "precision", getattr(model, "precision", "fp32")))
    bytes_per_parameter = 2 if precision.lower() in {"fp16", "float16", "half"} else 4
    batch_size = max(1, _safe_int(getattr(adapter_state, "batch_size", getattr(model, "batch_size", DEFAULT_BATCH_SIZE)), DEFAULT_BATCH_SIZE))
    target_sparsity = _safe_float(getattr(adapter_state, "target_sparsity", getattr(model, "target_sparsity", DEFAULT_TARGET_SPARSITY)), DEFAULT_TARGET_SPARSITY)
    pruning_start = _safe_int(getattr(adapter_state, "pruning_start_step", PRUNING_START_STEP), PRUNING_START_STEP)
    pruning_end = _safe_int(getattr(adapter_state, "pruning_end_step", PRUNING_END_STEP), PRUNING_END_STEP)
    gamma_t = _sparsity_progress(global_step, target_sparsity, pruning_start, pruning_end)

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "global_step": int(global_step),
        "algorithm": "A_T adaptive efficient LM tuning",
        "tuning_layer_importance": dict(getattr(adapter_state, "tuning_layer_importance", {})),
        "dynamic_ranks": ranks,
        "r_apt": mean_rank,
        "R_t": max(ranks.values()) if ranks else mean_rank,
        "Delta_t": _safe_int(getattr(adapter_state, "Delta_t", DEFAULT_DELTA_T), DEFAULT_DELTA_T),
        "delta": delta_params,
        "dynamic_added_tuning_parameters": delta_params,
        "trainable_parameter_count": int(trainable),
        "relative_training_memory_source": "A_T metadata for trainable parameter count and memory_usage formulas",
        "memory_usage": int(trainable * bytes_per_parameter),
        "training_cost_proxy": float(trainable * batch_size / 1024.0),
        "batch_size": batch_size,
        "batch_size_sweep": list(BATCH_SIZE_SWEEP),
        "batch_size_32": BATCH_SIZE_32,
        "batch_size_128": BATCH_SIZE_128,
        "precision": precision,
        "half_precision_attack": bool(getattr(adapter_state, "half_precision_attack", getattr(model, "half_precision_attack", False))),
        "gamma_t": gamma_t,
        "gamma_T": target_sparsity,
        "early_training_t_lt_T": int(global_step) <= EARLY_TRAINING_WINDOW_T_LT_T,
        "adapter_selector": "task-sensitive APT adapter selector",
        "m_i": _jsonable(getattr(adapter_state, "m_i", [])),
        "m_o": _jsonable(getattr(adapter_state, "m_o", [])),
        "s_bar_t": dict(s_bar_t),
        "S_bar^t": dict(s_bar_t),
        "S_bar^t-1": dict(s_bar_t_1),
        "reference_grounding": "paper:chunk_012 Adaptive and Efficient LM Tuning A_T",
    }

    if hasattr(adapter_state, "rank_allocation"):
        setattr(adapter_state, "rank_allocation", dict(ranks))
    if hasattr(adapter_state, "adapter_metadata") and isinstance(adapter_state.adapter_metadata, MutableMapping):
        adapter_state.adapter_metadata["A_T metadata"] = metadata
        adapter_state.adapter_metadata["r_apt"] = mean_rank
        adapter_state.adapter_metadata["R_t"] = metadata["R_t"]
        adapter_state.adapter_metadata["Delta_t"] = metadata["Delta_t"]
    if hasattr(adapter_state, "rank_history") and isinstance(adapter_state.rank_history, list):
        adapter_state.rank_history.append(metadata)
    if hasattr(adapter_state, "trace_records") and isinstance(adapter_state.trace_records, list):
        adapter_state.trace_records.append({"global_step": int(global_step), "A_T metadata": metadata, "dynamic_ranks": ranks})
    if hasattr(model, "adapter_metadata") and isinstance(model.adapter_metadata, MutableMapping):
        model.adapter_metadata["A_T metadata"] = metadata
        model.adapter_metadata["r_apt"] = mean_rank
        model.adapter_metadata["R_t"] = metadata["R_t"]
    return _jsonable(metadata)


def adaptive_tuning_step(
    model: Any,
    adapter_state: Any,
    salience_scores: Mapping[str, float],
    loss_record: Optional[Mapping[str, Any]] = None,
    *,
    global_step: int,
) -> Dict[str, Any]:
    """Single bounded A_T step used by training, ablation, and smoke routes."""

    salience_records: Dict[str, Dict[str, float]] = {}
    for layer_id, score in salience_scores.items():
        salience_records[str(layer_id)] = sync_salience_state(str(layer_id), float(score), global_step=global_step)
    current_salience = {name: record["S_bar^t"] for name, record in salience_records.items()}
    if hasattr(adapter_state, "salience_ema"):
        adapter_state.salience_ema = dict(current_salience)
    if hasattr(adapter_state, "salience_ema_previous"):
        adapter_state.salience_ema_previous = {name: record["S_bar^t-1"] for name, record in salience_records.items()}

    importance = compute_tuning_layer_importance(model, current_salience, loss_record)
    if hasattr(adapter_state, "tuning_layer_importance"):
        adapter_state.tuning_layer_importance = dict(importance)
    ranks = allocate_dynamic_rank(
        importance,
        base_rank=_safe_int(getattr(adapter_state, "r_apt", DEFAULT_R_APT), DEFAULT_R_APT),
        tuning_budget=_safe_int(getattr(adapter_state, "tuning_budget", DEFAULT_TUNING_BUDGET), DEFAULT_TUNING_BUDGET),
        delta_t=_safe_int(getattr(adapter_state, "Delta_t", DEFAULT_DELTA_T), DEFAULT_DELTA_T),
    )
    metadata = update_tuning_state(model, adapter_state, ranks, global_step=global_step)
    return {
        "global_step": int(global_step),
        "salience_records": salience_records,
        "tuning_layer_importance": importance,
        "dynamic_ranks": ranks,
        "A_T metadata": metadata,
    }


def create_tuning_experiment_matrix(config: Optional[Any] = None) -> List[Dict[str, Any]]:
    """Executable bounded selector matrix over methods and batch-size sweeps."""

    bounded = bool(_cfg_value(config, "bounded", True))
    batch_defaults = resolve_batch_size_defaults(bounded)
    step_defaults = resolve_num_steps_defaults(bounded)
    rows: List[Dict[str, Any]] = []
    for method in METHOD_SELECTOR_SET:
        canonical = str(method).lower().replace(" ", "_")
        if canonical in {"batch_size_32", "batch_size_128", "10_shot_setting"}:
            continue
        for batch_size in BATCH_SIZE_SWEEP:
            rows.append(
                {
                    "method": method,
                    "model_name": _cfg_value(config, "model_name", DEFAULT_MODEL_NAME),
                    "dataset_name": _cfg_value(config, "dataset_name", DEFAULT_DATASET_NAME),
                    "batch_size": batch_size,
                    "batch_size_defaults": batch_defaults,
                    "num_steps_defaults": step_defaults,
                    "target_sparsity": _cfg_value(config, "target_sparsity", DEFAULT_TARGET_SPARSITY),
                    "pruning_warmup_steps": _cfg_value(config, "pruning_warmup_steps", PRUNING_START_STEP),
                    "mask_granularity": _cfg_value(config, "mask_granularity", "block"),
                    "precision": _cfg_value(config, "precision", "fp32"),
                    "half_precision_attack": _cfg_value(config, "half_precision_attack", False),
                    "bounded": bounded,
                    "route": "src.apt.tuning.adaptive_tuning_step",
                }
            )
    return rows


def resolve_batch_size_defaults(bounded: bool = True) -> Dict[str, Any]:
    return _jsonable(_config_resolve_batch_size_defaults(bounded))


def get_tuning_hyperparameter_defaults() -> Dict[str, Any]:
    return {
        "m_i": "input binary mask",
        "m_o": "output binary mask",
        "r_apt": DEFAULT_R_APT,
        "R_t": DEFAULT_R_T,
        "Delta_t": DEFAULT_DELTA_T,
        "t << T early-training window": EARLY_TRAINING_WINDOW_T_LT_T,
        "target_sparsity": DEFAULT_TARGET_SPARSITY,
        "pruning_warmup_steps": PRUNING_START_STEP,
        "pruning_end_step": PRUNING_END_STEP,
        "mask_granularity": list(MASK_GRANULARITY_CHOICES),
        "precision": list(PRECISION_CHOICES),
        "half_precision_attack": list(HALF_PRECISION_ATTACK_SWEEP),
        "10_shot_setting": TEN_SHOT_SETTING,
        "batch_size_32": BATCH_SIZE_32,
        "batch_size_128": BATCH_SIZE_128,
        "alpha": ALPHA_DEFAULT,
        "salience_ema_decay": SALIENCE_EMA_DECAY,
        "salience_ema_update": SALIENCE_EMA_UPDATE,
        "reference_grounding": "paper:addendum and paper:appendix C Adaptive Pruning and Tuning Details",
    }


def _variant_defaults(method: str) -> Dict[str, Any]:
    method_lower = method.lower()
    return {
        "method": method,
        "r_apt": DEFAULT_R_APT if method_lower not in {"ft", "fine_tuning", "bert", "roberta", "t5"} else 0,
        "tuning_budget": DEFAULT_TUNING_BUDGET,
        "target_sparsity": DEFAULT_TARGET_SPARSITY if method_lower in {"apt", "ours", "lora+prune", "mask tuning", "cofi"} else 0.0,
        "batch_size": BATCH_SIZE_32,
        "precision": "fp32",
        "half_precision_attack": False,
    }


def get_method_variant_factories() -> Dict[str, MethodVariant]:
    """Method/baseline selector set required by the paper evidence contract."""

    variants = [
        MethodVariant("ours", ("Ours", "APT"), "APT", "src.apt.adapters.create_apt_adapter", "src.apt.tuning.adaptive_tuning_step", "src.apt.pruning.AdaptivePruner"),
        MethodVariant("APT", ("APT。", "ours"), "APT", "src.apt.adapters.create_apt_adapter", "src.apt.tuning.adaptive_tuning_step", "src.apt.pruning.AdaptivePruner"),
        MethodVariant("fine_tuning", ("FT", "fine_tuning"), "baseline", "src.apt.models.attach_adapter_metadata:none", "src.apt.training.run_training", "disabled"),
        MethodVariant("lora", ("LoRA", "lora"), "PEFT", "src.apt.adapters.build_lora_adapter", "src.apt.training.run_training", "disabled"),
        MethodVariant("lora_prune", ("LoRA+Prune", "lora_prune"), "PEFT+pruning", "src.apt.adapters.build_lora_adapter", "src.apt.tuning.adaptive_tuning_step", "src.apt.pruning.AdaptivePruner"),
        MethodVariant("mask_tuning", ("Mask Tuning", "mask_tuning"), "baseline", "src.apt.adapters.create_apt_adapter", "src.apt.tuning.adaptive_tuning_step", "src.apt.pruning.AdaptivePruner"),
        MethodVariant("cofi", ("CoFi", "cofi"), "baseline", "src.apt.adapters.create_apt_adapter", "src.apt.tuning.adaptive_tuning_step", "src.apt.pruning.AdaptivePruner"),
        MethodVariant("pruning_distillation", ("pruning+distillation combinations",), "APT_ablation", "src.apt.adapters.create_apt_adapter", "src.apt.tuning.adaptive_tuning_step", "src.apt.pruning.AdaptivePruner"),
        MethodVariant("bert", ("bert", "bert-base"), "model_route", "src.apt.models.build_model", "src.apt.training.run_training", "src.apt.pruning.AdaptivePruner", ("transformers",)),
        MethodVariant("roberta", ("roberta", "roberta-base"), "model_route", "src.apt.models.build_model", "src.apt.training.run_training", "src.apt.pruning.AdaptivePruner", ("transformers",)),
        MethodVariant("t5", ("t5", "t5-small"), "model_route", "src.apt.models.build_model", "src.apt.training.run_training", "src.apt.pruning.AdaptivePruner", ("transformers",)),
        MethodVariant("test_time_adaptation", ("test_time_adaptation", "TTA"), "adaptation", "src.apt.baselines.run_test_time_adaptation", "src.apt.tuning.adaptive_tuning_step", "disabled"),
    ]
    return {
        variant.id: MethodVariant(
            id=variant.id,
            aliases=variant.aliases,
            family=variant.family,
            adapter_factory=variant.adapter_factory,
            tuning_route=variant.tuning_route,
            pruning_route=variant.pruning_route,
            bounded_defaults=_variant_defaults(variant.id),
            full_mode_requirements=variant.full_mode_requirements,
            reference_grounding=variant.reference_grounding,
        )
        for variant in variants
    }


def select_method_variant(name: str) -> MethodVariant:
    name_lower = str(name).lower()
    for variant in get_method_variant_factories().values():
        aliases = {variant.id.lower(), *(str(alias).lower() for alias in variant.aliases)}
        if name_lower in aliases:
            return variant
    raise KeyError(f"unknown APT tuning method or baseline: {name}")


def make_tuning_trace(state: TuningState, records: Optional[Sequence[Mapping[str, Any]]] = None) -> Dict[str, Any]:
    records = list(records if records is not None else state.trace_records)
    metadata = records[-1].get("A_T metadata", {}) if records else state.adapter_metadata.get("A_T metadata", {})
    return {
        "schema_version": SCHEMA_VERSION,
        "algorithm": "A_T adaptive efficient tuning",
        "records": _jsonable(records),
        "A_T metadata": _jsonable(metadata),
        "importance": _jsonable(state.tuning_layer_importance),
        "dynamic_ranks": _jsonable(state.rank_allocation),
        "r_apt": state.r_apt,
        "R_t": state.R_t,
        "Delta_t": state.Delta_t,
        "s_bar_t": dict(s_bar_t),
        "s_bar_t_1": dict(s_bar_t_1),
        "reference_grounding": "paper:chunk_012 Adaptive and Efficient LM Tuning A_T",
    }


def _artifact_root(output_dir: Optional[str | Path] = None) -> Path:
    env_root = os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR")
    return Path(output_dir or env_root or DEFAULT_OUTPUT_DIR)


def _write_json(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(path)


def write_tuning_trace_artifact(output_dir: Optional[str | Path], payload: Mapping[str, Any]) -> str:
    """Write measured/bounded A_T trace from an executed tuning route."""

    if not payload.get("records"):
        raise ValueError("tuning_trace requires executed A_T records; readiness shells are not written here")
    return _write_json(_artifact_root(output_dir) / "tuning_trace.json", payload)


def write_sensitivity_report_artifact(output_dir: Optional[str | Path], trace: Mapping[str, Any]) -> str:
    """Write code-backed rank/batch-size sensitivity summary from trace records."""

    records = [row for row in trace.get("records", []) if isinstance(row, Mapping)]
    if not records:
        raise ValueError("sensitivity_report requires tuning records")
    ranks: List[float] = []
    for row in records:
        ranks.extend(_safe_float(v) for v in dict(row.get("dynamic_ranks", {})).values())
    payload = {
        "schema_version": SCHEMA_VERSION,
        "source_trace": "results/tuning_trace.json",
        "batch_size_sweep": list(BATCH_SIZE_SWEEP),
        "rank_count": len(ranks),
        "rank_mean": aggregate_loss(ranks) if ranks else 0.0,
        "rank_min": min(ranks) if ranks else 0.0,
        "rank_max": max(ranks) if ranks else 0.0,
        "half_precision_attack": trace.get("A_T metadata", {}).get("half_precision_attack", False) if isinstance(trace.get("A_T metadata"), Mapping) else False,
        "reference_grounding": "paper:appendix C Adaptive Pruning and Tuning Details",
    }
    return _write_json(_artifact_root(output_dir) / "sensitivity_report.json", payload)


def write_model_registry_artifact(output_dir: Optional[str] = None, run_config: Optional[Any] = None) -> str:
    return model_routes.write_model_registry_artifact(output_dir=output_dir, run_config=run_config)


def write_pruning_trace_artifact(output_dir: Optional[str | Path], payload: Mapping[str, Any]) -> str:
    root = _artifact_root(output_dir)
    if not payload.get("records"):
        raise ValueError("pruning_trace requires executed A_P records; readiness shells are not written here")
    return _write_json(root / "pruning_trace.json", payload)


def run_bounded_tuning_probe(config: Optional[Any] = None) -> Dict[str, Any]:
    """Small measured route for import/smoke callers; it does not claim benchmark results."""

    model = model_routes.build_model(config)
    state = build_tuning_state(config, model)
    losses = [0.3, 0.2]
    loss_record = {"loss": compute_loss(losses), "L_distill": aggregate_loss(losses), "L_layer": 0.05}
    salience = {getattr(layer, "name", f"layer.{idx}"): float(idx + 1) for idx, layer in enumerate(_model_layers(model)[:2])}
    if not salience:
        salience = {"layer.0": 1.0}
    record = adaptive_tuning_step(model, state, salience, loss_record, global_step=1)
    trace = make_tuning_trace(state, [record])
    score = compute_ours_oradaptersby_inventory_score(
        {
            "accuracy": 1.0,
            "loss": loss_record["L_distill"],
            "training_cost": trace["A_T metadata"].get("training_cost_proxy", 0.0),
            "memory_usage": trace["A_T metadata"].get("memory_usage", 0.0),
        }
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "bounded_proxy_measured",
        "tuning_trace": trace,
        "objective": score,
        "method_variants": {key: value.to_dict() for key, value in get_method_variant_factories().items()},
        "hyperparameters": get_tuning_hyperparameter_defaults(),
        "experiment_matrix": create_tuning_experiment_matrix(config)[:4],
        "reference_grounding": "paperbench_ref_001 train.py",
    }


def main(config: Optional[Any] = None) -> Dict[str, Any]:
    """Callable ``main(config) -> dict`` entry for package and review smoke."""

    return run_bounded_tuning_probe(config)


__all__ = [
    "TuningState",
    "s_bar_t",
    "s_bar_t_1",
    "compute_tuning_layer_importance",
    "allocate_dynamic_rank",
    "update_tuning_state",
    "adaptive_tuning_step",
    "build_tuning_state",
    "sync_salience_state",
    "tuning_parameter_delta",
    "count_trainable_parameters",
    "resolve_batch_size_defaults",
    "resolve_num_steps_defaults",
    "compute_loss",
    "aggregate_loss",
    "compute_reward",
    "aggregate_reward",
    "compute_ours_oradaptersby_inventory_objective",
    "compute_ours_oradaptersby_inventory_score",
    "get_tuning_hyperparameter_defaults",
    "get_method_variant_factories",
    "select_method_variant",
    "create_tuning_experiment_matrix",
    "make_tuning_trace",
    "write_tuning_trace_artifact",
    "write_sensitivity_report_artifact",
    "write_model_registry_artifact",
    "write_pruning_trace_artifact",
    "run_bounded_tuning_probe",
    "main",
]
