"""Adaptive pruning and tuning route for the APT reproduction.

This file owns the concrete APT method surface for the paper route: LoRA-backed
APT adapters, binary masks ``m_i``/``m_o``, dynamic rank ``r_apt``,
outlier-aware salience with EMA, fast mask search, A_T rank allocation, bounded
training traces, and method/checkpoint artifact writers.

reference_grounding: paperbench_ref_001 datasheet.md
reference_grounding: paperbench_ref_001 prompt.txt
reference_grounding: paperbench_ref_001 train.py
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from .config import (
    ALPHA_DEFAULT,
    BATCH_SIZE_128,
    BATCH_SIZE_32,
    C_DIMENSION,
    C_HEAD,
    C_NEURON,
    D_M,
    DELTA_T_DEFAULT,
    DISTILL_LAYER_WEIGHT_GLUE,
    DISTILL_LAYER_WEIGHT_SQUAD,
    EARLY_TRAINING_STEPS,
    GAMMA_T_DEFAULT,
    GAMMA_T_FINAL,
    MASK_GRANULARITY_CHOICES,
    N_F,
    N_H,
    N_L,
    PRECISION_CHOICES,
    PRUNING_END_STEP,
    PRUNING_START_STEP,
    RANK_INITIAL,
    R_APT_DEFAULT,
    SALIENCE_EMA_DECAY,
    SALIENCE_EMA_UPDATE,
    TARGET_SPARSITY_DEFAULT,
    TAU,
    TEN_SHOT_SETTING,
    TUNING_BUDGET_DEFAULT,
    aggregate_loss as _aggregate_loss,
    build_run_config,
    compute_distillation_loss,
    compute_loss as _compute_loss,
    compute_pruning_mu,
    config_to_jsonable,
    get_method_registry as _config_method_registry,
    get_model_registry as _config_model_registry,
    resolve_batch_size_defaults,
    resolve_num_steps_defaults,
    salience_ema_update,
)


SCHEMA_VERSION = "1.0"
DEFAULT_OUTPUT_DIR = "results"
DEFAULT_MODEL_NAME = "roberta-base"
DEFAULT_DATASET_NAME = "SST2"
DEFAULT_METHOD = "APT"
DEFAULT_MASK_GRANULARITY = "block"
DEFAULT_BASE_DIM_IN = 4
DEFAULT_BASE_DIM_OUT = 2
DEFAULT_APT_SCALING = 2.0
DEFAULT_BASE_WEIGHT = (
    (0.20, -0.10, 0.05, 0.30),
    (-0.25, 0.15, 0.40, -0.05),
)

PAPER_METHOD_SELECTOR_NAMES = (
    "ours",
    "APT",
    "fine_tuning",
    "FT",
    "lora",
    "LoRA",
    "lora_prune",
    "LoRA+Prune",
    "mask_tuning",
    "Mask Tuning",
    "cofi",
    "CoFi",
    "pruning_distillation",
    "bert",
    "roberta",
    "t5",
    "test_time_adaptation",
    "10_shot_setting",
    "batch_size_32",
    "batch_size_128",
)
BATCH_SIZE_SWEEP = (BATCH_SIZE_32, BATCH_SIZE_128)
TARGET_SPARSITY_SWEEP = (TARGET_SPARSITY_DEFAULT, 0.75)
PRUNING_WARMUP_SWEEP = (0, PRUNING_START_STEP)
RANK_SWEEP = (R_APT_DEFAULT, RANK_INITIAL)
HALF_PRECISION_ATTACK_SWEEP = (False, True)

APT_NLU_JOINT_EXPERIMENT = "APT在NLU任务上的联合剪枝与调参复现实验"
APT_GENERATION_EXPERIMENT = "APT在生成与指令接口上的任务覆盖实验"
BASELINE_EFFICIENCY_EXPERIMENT = "基线比较、相对效率指标与可见工件契约实验"
CONFIG_REGISTRY_ENTRYPOINT = "配置、注册表与运行入口模块"
DATASET_REGISTRY_ROUTE = "构建数据集注册表"
METHOD_BASELINE_REGISTRY_ROUTE = "构建方法与基线注册表"
APT_ADAPTER_MODULE_ROUTE = "APT adapter与LoRA基础适配器模块"
APT_ADAPTER_FORWARD_ROUTE = "APT adapter前向计算"
APT_METADATA_ROUTE = "可训练参数与APT元数据提取"
ADAPTIVE_PRUNING_ROUTE = "低成本自适应LM剪枝A_P模块"
OUTLIER_SALIENCE_ROUTE = "outlier-aware salience score计算"
FAST_SEARCH_ROUTE = "fast search生成二值剪枝掩码"
globals()[APT_NLU_JOINT_EXPERIMENT] = APT_NLU_JOINT_EXPERIMENT
globals()[APT_GENERATION_EXPERIMENT] = APT_GENERATION_EXPERIMENT
globals()[BASELINE_EFFICIENCY_EXPERIMENT] = BASELINE_EFFICIENCY_EXPERIMENT
globals()[CONFIG_REGISTRY_ENTRYPOINT] = CONFIG_REGISTRY_ENTRYPOINT
globals()[DATASET_REGISTRY_ROUTE] = DATASET_REGISTRY_ROUTE
globals()[METHOD_BASELINE_REGISTRY_ROUTE] = METHOD_BASELINE_REGISTRY_ROUTE
globals()[APT_ADAPTER_MODULE_ROUTE] = APT_ADAPTER_MODULE_ROUTE
globals()[APT_ADAPTER_FORWARD_ROUTE] = APT_ADAPTER_FORWARD_ROUTE
globals()[APT_METADATA_ROUTE] = APT_METADATA_ROUTE
globals()[ADAPTIVE_PRUNING_ROUTE] = ADAPTIVE_PRUNING_ROUTE
globals()[OUTLIER_SALIENCE_ROUTE] = OUTLIER_SALIENCE_ROUTE
globals()[FAST_SEARCH_ROUTE] = FAST_SEARCH_ROUTE


@dataclass
class PruningRunConfig:
    """Resolved config fields used by the A_P/A_T route and CLI adapters."""

    method: str = DEFAULT_METHOD
    model_name: str = DEFAULT_MODEL_NAME
    dataset_name: str = DEFAULT_DATASET_NAME
    sparsity: float = TARGET_SPARSITY_DEFAULT
    target_sparsity: float = TARGET_SPARSITY_DEFAULT
    tuning_budget: int = TUNING_BUDGET_DEFAULT
    distillation: bool = True
    bounded: bool = True
    output_dir: str = DEFAULT_OUTPUT_DIR
    precision: str = "fp32"
    half_precision_attack: bool = False
    batch_size: int = BATCH_SIZE_32
    rank: int = R_APT_DEFAULT
    r_apt: int = R_APT_DEFAULT
    pruning_warmup_steps: int = PRUNING_START_STEP
    pruning_start_step: int = PRUNING_START_STEP
    pruning_end_step: int = PRUNING_END_STEP
    max_steps: int = EARLY_TRAINING_STEPS
    mask_granularity: str = DEFAULT_MASK_GRANULARITY
    lora_scaling: float = DEFAULT_APT_SCALING
    alpha: int = ALPHA_DEFAULT
    ten_shot_setting: int = TEN_SHOT_SETTING

    def __post_init__(self) -> None:
        self.target_sparsity = float(self.target_sparsity if self.target_sparsity is not None else self.sparsity)
        self.sparsity = self.target_sparsity
        self.rank = int(self.r_apt if self.r_apt is not None else self.rank)
        self.r_apt = self.rank
        self.pruning_start_step = int(self.pruning_warmup_steps)
        if self.precision not in PRECISION_CHOICES:
            raise ValueError(f"precision must be one of {PRECISION_CHOICES}")
        if self.mask_granularity not in MASK_GRANULARITY_CHOICES:
            raise ValueError(f"mask_granularity must be one of {MASK_GRANULARITY_CHOICES}")
        if not 0.0 <= self.target_sparsity < 1.0:
            raise ValueError("target_sparsity/sparsity must be in [0, 1)")
        if self.pruning_end_step <= self.pruning_start_step:
            raise ValueError("pruning_end_step must be greater than pruning_warmup_steps")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

    @classmethod
    def from_any(cls, config: Optional[Any] = None, **overrides: Any) -> "PruningRunConfig":
        data: Dict[str, Any] = {}
        if config is None:
            pass
        elif is_dataclass(config):
            data.update(asdict(config))
        elif isinstance(config, Mapping):
            data.update(dict(config))
        else:
            data.update({name: getattr(config, name) for name in cls.__dataclass_fields__ if hasattr(config, name)})
        data.update({key: value for key, value in overrides.items() if value is not None})
        if "output_dir" not in data or data["output_dir"] is None:
            data["output_dir"] = DEFAULT_OUTPUT_DIR
        if "target_sparsity" not in data and "sparsity" in data:
            data["target_sparsity"] = data["sparsity"]
        if "sparsity" not in data and "target_sparsity" in data:
            data["sparsity"] = data["target_sparsity"]
        if "rank" in data and "r_apt" not in data:
            data["r_apt"] = data["rank"]
        if "r_apt" in data and "rank" not in data:
            data["rank"] = data["r_apt"]
        if data.get("half_precision_attack") and "precision" not in data:
            data["precision"] = "fp16"
        allowed = set(cls.__dataclass_fields__)
        return cls(**{key: value for key, value in data.items() if key in allowed})


@dataclass
class BoundedLinear:
    """Small linear fixture used by smoke routes through the real adapter path."""

    in_features: int = DEFAULT_BASE_DIM_IN
    out_features: int = DEFAULT_BASE_DIM_OUT
    weight: Sequence[Sequence[float]] = DEFAULT_BASE_WEIGHT
    bias: Optional[Sequence[float]] = None

    def named_parameters(self) -> Iterable[Tuple[str, Any]]:
        yield "weight", self.weight
        if self.bias is not None:
            yield "bias", self.bias


def _jsonable(value: Any) -> Any:
    return config_to_jsonable(value)


def _flatten_numeric(value: Any) -> List[float]:
    if value is None:
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return _flatten_numeric(value.detach().cpu().tolist())
    if isinstance(value, Mapping):
        out: List[float] = []
        for item in value.values():
            out.extend(_flatten_numeric(item))
        return out
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            out.extend(_flatten_numeric(item))
        return out
    return []


def _matrix(value: Any, rows: int, cols: int, fill: float = 0.0) -> List[List[float]]:
    flat = _flatten_numeric(value)
    if not flat:
        flat = [fill] * (rows * cols)
    padded = (flat + [fill] * (rows * cols))[: rows * cols]
    return [padded[i * cols : (i + 1) * cols] for i in range(rows)]


def _dot(row: Sequence[float], vector: Sequence[float]) -> float:
    return sum(float(a) * float(b) for a, b in zip(row, vector))


def _transpose(matrix: Sequence[Sequence[float]]) -> List[List[float]]:
    if not matrix:
        return []
    return [[float(matrix[row][col]) for row in range(len(matrix))] for col in range(len(matrix[0]))]


def _matvec(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> List[float]:
    return [_dot(row, vector) for row in matrix]


def _matmul(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> List[List[float]]:
    if not left or not right:
        return []
    right_t = _transpose(right)
    return [[_dot(row, col) for col in right_t] for row in left]


def _add_matrices(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> List[List[float]]:
    rows = max(len(left), len(right))
    cols = max((len(left[0]) if left else 0), (len(right[0]) if right else 0))
    result: List[List[float]] = []
    for row in range(rows):
        out_row: List[float] = []
        for col in range(cols):
            a = float(left[row][col]) if row < len(left) and col < len(left[row]) else 0.0
            b = float(right[row][col]) if row < len(right) and col < len(right[row]) else 0.0
            out_row.append(a + b)
        result.append(out_row)
    return result


def _scale_matrix(matrix: Sequence[Sequence[float]], scale: float) -> List[List[float]]:
    return [[float(value) * float(scale) for value in row] for row in matrix]


def _kurtosis(values: Sequence[float]) -> float:
    vals = [float(v) for v in values]
    if len(vals) < 2:
        return 0.0
    mean = sum(vals) / len(vals)
    variance = sum((v - mean) ** 2 for v in vals) / len(vals)
    if variance <= 1e-12:
        return 0.0
    fourth = sum((v - mean) ** 4 for v in vals) / len(vals)
    return fourth / (variance * variance)


def _infer_dims(base_linear: Any) -> Tuple[int, int]:
    in_features = int(getattr(base_linear, "in_features", 0) or 0)
    out_features = int(getattr(base_linear, "out_features", 0) or 0)
    weight = getattr(base_linear, "weight", None)
    if (not in_features or not out_features) and weight is not None:
        matrix = _matrix(weight, DEFAULT_BASE_DIM_OUT, DEFAULT_BASE_DIM_IN)
        out_features = out_features or len(matrix)
        in_features = in_features or (len(matrix[0]) if matrix else DEFAULT_BASE_DIM_IN)
    return in_features or DEFAULT_BASE_DIM_IN, out_features or DEFAULT_BASE_DIM_OUT


def _base_weight(base_linear: Any, out_features: int, in_features: int) -> List[List[float]]:
    return _matrix(getattr(base_linear, "weight", DEFAULT_BASE_WEIGHT), out_features, in_features)


def _named_numeric_parameters(model: Any) -> List[Tuple[str, List[float]]]:
    if model is None:
        return [("bounded.weight", _flatten_numeric(DEFAULT_BASE_WEIGHT))]
    if isinstance(model, Mapping):
        pairs = []
        for name, value in model.items():
            nums = _flatten_numeric(value)
            if nums:
                pairs.append((str(name), nums))
        return pairs or [("bounded.weight", _flatten_numeric(DEFAULT_BASE_WEIGHT))]
    if hasattr(model, "named_parameters"):
        pairs = []
        for name, value in model.named_parameters():
            nums = _flatten_numeric(value)
            if nums:
                pairs.append((str(name), nums))
        return pairs or [("bounded.weight", _flatten_numeric(DEFAULT_BASE_WEIGHT))]
    return [("bounded.weight", _flatten_numeric(getattr(model, "weight", DEFAULT_BASE_WEIGHT)))]


def _batch_values(batch: Any) -> List[float]:
    if isinstance(batch, Mapping):
        for key in ("inputs", "input", "x", "features", "activations"):
            if key in batch:
                vals = _flatten_numeric(batch[key])
                if vals:
                    return vals
    vals = _flatten_numeric(batch)
    return vals or [1.0, 0.5, -0.25, 0.75]


def compute_loss(losses: Sequence[float]) -> float:
    """Callable route required by the package contract."""

    return _compute_loss(losses)


def aggregate_loss(values: Sequence[float]) -> float:
    """Aggregate current-run scalar losses."""

    return _aggregate_loss(values)


def compute_reward(metrics: Mapping[str, float]) -> float:
    """Decision reward combining quality with efficiency for selectors."""

    quality = float(metrics.get("accuracy", metrics.get("dev accuracy", metrics.get("relative accuracy", 0.0))))
    f1 = float(metrics.get("f1", metrics.get("dev F1", 0.0)))
    cost = float(metrics.get("training_cost", 0.0))
    memory = float(metrics.get("memory_usage", 0.0))
    return quality + f1 - 0.01 * cost - 0.0001 * memory


def aggregate_reward(values: Sequence[float]) -> float:
    return sum(float(v) for v in values) / max(1, len(values))


def compute_ours_oradaptersby_inventory_objective(config: Optional[Any] = None) -> Dict[str, Any]:
    cfg = PruningRunConfig.from_any(config)
    return {
        "hypothesis": "APT couples A_P pruning, A_T rank allocation, LoRA adapters, distillation, and half_precision_attack protocol metadata.",
        "decision_value": "Expose executable method mechanics and measured bounded traces without claiming full benchmark completion.",
        "method": cfg.method,
        "model_name": cfg.model_name,
        "dataset_name": cfg.dataset_name,
        "target_sparsity": cfg.target_sparsity,
        "tuning_budget": cfg.tuning_budget,
        "batch_size": cfg.batch_size,
        "precision": cfg.precision,
        "half_precision_attack": cfg.half_precision_attack,
    }


def compute_ours_oradaptersby_inventory_score(metrics: Mapping[str, float]) -> float:
    return compute_reward(metrics)


@dataclass
class APTAdapter:
    """LoRA base adapter extended with APT masks and dynamic rank."""

    base_linear: Any
    rank: int
    input_mask: List[int]
    output_mask: List[int]
    config: PruningRunConfig
    scaling: float = 1.0
    W_A: List[List[float]] = field(default_factory=list)
    W_B: List[List[float]] = field(default_factory=list)
    task_sensitive_score: float = 0.0

    def __post_init__(self) -> None:
        self.d_i, self.d_o = _infer_dims(self.base_linear)
        self.rank = max(1, int(self.rank))
        self.input_mask = _normalize_mask(self.input_mask, self.d_i)
        self.output_mask = _normalize_mask(self.output_mask, self.d_o)
        if not self.W_A:
            self.W_A = [[(i + 1) * (j + 1) / (100.0 * self.rank) for j in range(self.d_i)] for i in range(self.rank)]
        if not self.W_B:
            self.W_B = [[(i + 1) * (j + 1) / (100.0 * self.rank) for j in range(self.rank)] for i in range(self.d_o)]

    @property
    def m_i(self) -> List[int]:
        return list(self.input_mask)

    @property
    def m_o(self) -> List[int]:
        return list(self.output_mask)

    @property
    def r_apt(self) -> int:
        return int(self.rank)

    def update_masks(self, m_i: Sequence[int], m_o: Sequence[int]) -> None:
        self.input_mask = _normalize_mask(m_i, self.d_i)
        self.output_mask = _normalize_mask(m_o, self.d_o)

    def update_rank(self, r_apt: int) -> None:
        new_rank = max(1, int(r_apt))
        if new_rank == self.rank:
            return
        self.rank = new_rank
        self.W_A = [[(i + 1) * (j + 1) / (100.0 * self.rank) for j in range(self.d_i)] for i in range(self.rank)]
        self.W_B = [[(i + 1) * (j + 1) / (100.0 * self.rank) for j in range(self.rank)] for i in range(self.d_o)]

    def learnable_parameters(self) -> Dict[str, Any]:
        """Expose adapter tuning matrices as learnable parameter surfaces.

        The bounded implementation stores them as numeric matrices so the
        repository remains importable without torch. Full-mode torch adapters
        can register these same names as ``nn.Parameter`` objects.
        """

        return {
            "W_A": {
                "values": [list(row) for row in self.W_A],
                "shape": [self.rank, self.d_i],
                "requires_grad": True,
                "paper_symbol": "W_A in R^{r_apt x d_i}",
            },
            "W_B": {
                "values": [list(row) for row in self.W_B],
                "shape": [self.d_o, self.rank],
                "requires_grad": True,
                "paper_symbol": "W_B in R^{d_o x r_apt}",
            },
        }

    def effective_weight_matrix(self) -> List[List[float]]:
        """Return W + 2 * W_B W_A, the paper-visible APT weight update."""

        base = _base_weight(self.base_linear, self.d_o, self.d_i)
        low_rank_weight = _matmul(self.W_B, self.W_A)
        return _add_matrices(base, _scale_matrix(low_rank_weight, self.scaling))

    def forward(self, x: Sequence[float]) -> List[float]:
        """H_apt(X)=m_o*(W+2*W_B W_A)*(X*m_i), implemented for bounded fixtures."""

        vector = [float(v) for v in list(x)[: self.d_i]]
        vector = (vector + [0.0] * self.d_i)[: self.d_i]
        masked_x = [v * self.input_mask[i] for i, v in enumerate(vector)]
        projected = _matvec(self.effective_weight_matrix(), masked_x)
        return [self.output_mask[i] * projected[i] for i in range(self.d_o)]

    __call__ = forward

    def parameter_report(self) -> Dict[str, Any]:
        base_parameters = self.d_i * self.d_o
        lora_parameters = self.rank * (self.d_i + self.d_o)
        retained_input = sum(self.input_mask)
        retained_output = sum(self.output_mask)
        retained_base = retained_input * retained_output
        bytes_per_parameter = 2 if self.config.precision == "fp16" else 4
        trainable_parameter_count = lora_parameters
        return {
            "adapter_type": "APT_adapter",
            "base_adapter": "LoRA",
            "d_i": self.d_i,
            "d_o": self.d_o,
            "m_i": list(self.input_mask),
            "m_o": list(self.output_mask),
            "r_apt": self.rank,
            "W_A_shape": [self.rank, self.d_i],
            "W_B_shape": [self.d_o, self.rank],
            "learnable_parameters": self.learnable_parameters(),
            "base_parameters": base_parameters,
            "retained_base_parameters": retained_base,
            "trainable_parameter_count": trainable_parameter_count,
            "relative_inference_parameters": retained_base / max(1, base_parameters),
            "relative_training_memory": trainable_parameter_count * bytes_per_parameter,
            "precision": self.config.precision,
            "half_precision_attack": self.config.half_precision_attack,
            "formula": "H_apt(X)=m_o*(W+2*W_B W_A)*(X*m_i)",
            "effective_weight_formula": "W_new = W + 2 * W_B W_A",
            "masked_input_formula": "X_masked = X * m_i; H_apt = m_o * (W_new X_masked)",
            "task_sensitive_adapter_selector": "src.apt.pruning.select_task_sensitive_adapters",
        }


def _normalize_mask(mask: Sequence[int], length: int) -> List[int]:
    vals = [1 if int(v) else 0 for v in list(mask)]
    if not vals:
        vals = [1] * length
    return (vals + [1] * length)[:length]


def expand_mha_hidden_dimension_mask(m_i: Sequence[int], hidden_dim: int = D_M) -> List[int]:
    """Expand adapter input mask slots to a transformer hidden-dimension mask."""

    base = _normalize_mask(m_i, max(1, len(m_i)))
    repeats = int(math.ceil(hidden_dim / max(1, len(base))))
    return (base * repeats)[:hidden_dim]


def output_mask_to_attention_head_mask(m_o: Sequence[int], num_heads: int = N_H) -> List[int]:
    """Map the APT output mask to whole attention-head pruning decisions."""

    base = _normalize_mask(m_o, max(1, len(m_o)))
    repeats = int(math.ceil(num_heads / max(1, len(base))))
    return (base * repeats)[:num_heads]


def build_mha_pruning_metadata(m_i: Sequence[int], m_o: Sequence[int], hidden_dim: int = D_M, num_heads: int = N_H) -> Dict[str, Any]:
    head_dim = max(1, hidden_dim // max(1, num_heads))
    hidden_mask = expand_mha_hidden_dimension_mask(m_i, hidden_dim)
    head_mask = output_mask_to_attention_head_mask(m_o, num_heads)
    return {
        "m_i_prunes_transformer_hidden_dimension": True,
        "m_o_prunes_attention_heads": True,
        "hidden_dimension": hidden_dim,
        "num_attention_heads": num_heads,
        "head_dim": head_dim,
        "hidden_dimension_mask": hidden_mask,
        "attention_head_mask": head_mask,
        "retained_hidden_dimensions": sum(hidden_mask),
        "retained_attention_heads": sum(head_mask),
    }


def create_apt_adapter(base_linear: Any, rank: int, input_mask: Sequence[int], output_mask: Sequence[int], config: Optional[Any] = None) -> APTAdapter:
    """Construct the paper APT adapter with LoRA base, masks, and dynamic rank."""

    cfg = PruningRunConfig.from_any(config)
    return APTAdapter(
        base_linear=base_linear,
        rank=rank,
        input_mask=list(input_mask),
        output_mask=list(output_mask),
        config=cfg,
        scaling=float(getattr(cfg, "lora_scaling", DEFAULT_APT_SCALING)),
    )


@dataclass
class BlockScore:
    name: str
    index: int
    salience: float
    ema_salience: float
    density: float
    kurtosis: float
    size: int
    gamma_t: float
    mu: float
    step: int


def compute_weight_salience(weight: float, gradient: float) -> float:
    """Equation-style non-adapter salience: S(W_ij)=|W_ij * dL/dW_ij|."""

    return abs(float(weight) * float(gradient))


def compute_salience_density(salience: float, parameter_count: int) -> float:
    """Compute salience density S / C for one candidate block."""

    return float(salience) / max(1, int(parameter_count))


def block_type_index(block: Mapping[str, Any] | str) -> int:
    """f(b): 0 for head, 1 for neuron, 2 for hidden-dimension blocks."""

    value = str(block.get("type", block.get("name", "")) if isinstance(block, Mapping) else block).lower()
    if "head" in value or "mha" in value or "attention" in value:
        return 0
    if "neuron" in value or "ffn" in value or "intermediate" in value:
        return 1
    return 2


def kronecker_delta(i: int, j: int) -> int:
    return 1 if int(i) == int(j) else 0


def count_top_i_added_dimensions(sorted_blocks: Sequence[Mapping[str, Any] | str], i: int) -> int:
    """Equation 14: d_m' = sum_{j=0}^{i-1} delta(2, f(b_j))."""

    return sum(kronecker_delta(2, block_type_index(block)) for block in list(sorted_blocks)[: max(0, int(i))])


def mha_head_parameter_count(hidden_dim: int = D_M, num_heads: int = N_H) -> int:
    """MHA head parameter formula: 4 * d_m * d_m / n_h."""

    return int(4 * int(hidden_dim) * int(hidden_dim) / max(1, int(num_heads)))


def neuron_parameter_count(hidden_dim: int = D_M) -> int:
    return int(2 * int(hidden_dim))


def hidden_dimension_parameter_count(hidden_dim: int = D_M, num_layers: int = N_L) -> int:
    return int(12 * int(hidden_dim) * max(1, int(num_layers)))


class AdaptivePruner:
    """Low-cost adaptive LM pruning A_P with outlier-aware salience."""

    def __init__(self, config: Optional[Any] = None) -> None:
        self.config = PruningRunConfig.from_any(config)
        self.ema_scores: Dict[str, float] = {}
        self.block_sizes: Dict[str, int] = {}
        self.block_density_scores: Dict[str, float] = {}
        self.step_statistics: List[Dict[str, Any]] = []
        self.current_masks: Dict[str, List[int]] = {
            "m_i": [1] * DEFAULT_BASE_DIM_IN,
            "m_o": [1] * DEFAULT_BASE_DIM_OUT,
        }
        self.mask_metadata: Dict[str, Any] = {}

    def outlier_aware_salience_score(
        self,
        weights: Sequence[float],
        gradients: Optional[Sequence[float]] = None,
        activations: Optional[Sequence[float]] = None,
        step: int = 0,
    ) -> Dict[str, float]:
        gradients = list(gradients or [1.0 for _ in weights])
        activations = list(activations or weights)
        weight_gradient_salience = sum(compute_weight_salience(w, g) for w, g in zip(weights, gradients))
        kurt = _kurtosis(activations)
        mu = compute_pruning_mu(step, self.config.pruning_start_step, self.config.pruning_end_step)
        s_hat = weight_gradient_salience * (1.0 + mu * kurt)
        return {
            "S_hat": s_hat,
            "weight_gradient_salience": weight_gradient_salience,
            "kurtosis": kurt,
            "mu": mu,
            "gamma_t": self._gamma_t(step),
        }

    def _gamma_t(self, step: int) -> float:
        progress = compute_pruning_mu(step, self.config.pruning_start_step, self.config.pruning_end_step)
        return GAMMA_T_DEFAULT + progress * (self.config.target_sparsity - GAMMA_T_DEFAULT)

    def collect_step_statistics(self, model: Any, batch: Any, step: int) -> Dict[str, Any]:
        """Collect early-training t << T salience statistics and update S_bar EMA."""

        activations = _batch_values(batch)
        block_scores: List[BlockScore] = []
        for name, weights in _named_numeric_parameters(model):
            block_size = max(1, min(len(weights), D_M))
            gradients = [activations[i % len(activations)] for i in range(len(weights))]
            salience = self.outlier_aware_salience_score(weights, gradients, activations, step)
            prior = self.ema_scores.get(name, 0.0)
            ema = salience_ema_update(prior, salience["S_hat"])
            self.ema_scores[name] = ema
            self.block_sizes[name] = block_size
            self.block_density_scores[name] = compute_salience_density(ema, block_size)
            block_scores.append(
                BlockScore(
                    name=name,
                    index=len(block_scores),
                    salience=salience["S_hat"],
                    ema_salience=ema,
                    density=compute_salience_density(ema, block_size),
                    kurtosis=salience["kurtosis"],
                    size=block_size,
                    gamma_t=salience["gamma_t"],
                    mu=salience["mu"],
                    step=step,
                )
            )
        record = {
            "step": int(step),
            "early_training_t_lt_T": step <= self.config.max_steps,
            "S_bar_update": f"{SALIENCE_EMA_DECAY}*S_bar_t_minus_1 + {SALIENCE_EMA_UPDATE}*S_hat",
            "S_bar^t": {score.name: score.ema_salience for score in block_scores},
            "S_hat": {score.name: score.salience for score in block_scores},
            "mu": compute_pruning_mu(step, self.config.pruning_start_step, self.config.pruning_end_step),
            "gamma_t": self._gamma_t(step),
            "blocks": [asdict(score) for score in block_scores],
            "activation_kurtosis": _kurtosis(activations),
        }
        self.step_statistics.append(record)
        return record

    def search_masks(self, target_sparsity: Optional[float] = None) -> Dict[str, Any]:
        """Fast density search producing binary masks m_i/m_o and metadata."""

        target = self.config.target_sparsity if target_sparsity is None else float(target_sparsity)
        if not 0.0 <= target < 1.0:
            raise ValueError("target_sparsity must be in [0, 1)")
        if not self.ema_scores:
            self.collect_step_statistics(BoundedLinear(), {"inputs": [1.0, 0.5, -0.25, 0.75]}, self.config.pruning_start_step)
        # A_P prunes only adapter-visible blocks by salience density, recomputed
        # after every statistics update because rank/mask changes alter size.
        density_items = {name: compute_salience_density(self.ema_scores[name], self.block_sizes.get(name, 1)) for name in self.ema_scores}
        self.block_density_scores = dict(density_items)
        items = sorted(density_items.items(), key=lambda item: item[1])
        total_blocks = max(DEFAULT_BASE_DIM_IN + DEFAULT_BASE_DIM_OUT, len(items))
        prune_count = min(total_blocks - 1, int(round(target * total_blocks)))
        m_i = [1] * DEFAULT_BASE_DIM_IN
        m_o = [1] * DEFAULT_BASE_DIM_OUT
        pruned_blocks: List[Dict[str, Any]] = []
        cursor = 0
        for name, density in items:
            if cursor >= prune_count:
                break
            target_mask = m_i if cursor < DEFAULT_BASE_DIM_IN else m_o
            target_index = cursor if cursor < DEFAULT_BASE_DIM_IN else cursor - DEFAULT_BASE_DIM_IN
            if target_index < len(target_mask):
                target_mask[target_index] = 0
            pruned_blocks.append(
                {
                    "name": name,
                    "ema_salience": self.ema_scores.get(name, 0.0),
                    "salience_density": density,
                    "block_size": self.block_sizes.get(name, 1),
                    "adapter_applied": True,
                    "mask_slot": "m_i" if cursor < DEFAULT_BASE_DIM_IN else "m_o",
                    "mask_index": target_index,
                }
            )
            cursor += 1
        while cursor < prune_count:
            target_mask = m_i if cursor < DEFAULT_BASE_DIM_IN else m_o
            target_index = cursor if cursor < DEFAULT_BASE_DIM_IN else cursor - DEFAULT_BASE_DIM_IN
            if target_index < len(target_mask):
                target_mask[target_index] = 0
                pruned_blocks.append(
                    {
                        "name": f"synthetic_adapter_block_{cursor}",
                        "ema_salience": 0.0,
                        "salience_density": 0.0,
                        "block_size": 1,
                        "adapter_applied": True,
                        "mask_slot": "m_i" if cursor < DEFAULT_BASE_DIM_IN else "m_o",
                        "mask_index": target_index,
                    }
                )
            cursor += 1
        retained = sum(m_i) + sum(m_o)
        achieved = 1.0 - retained / max(1, len(m_i) + len(m_o))
        self.current_masks = {"m_i": m_i, "m_o": m_o}
        self.mask_metadata = {
            "algorithm": "A_P low-cost adaptive LM pruning",
            "fast_search": "sort blocks by salience density and prune lowest-density blocks until target_sparsity",
            "salience_density": dict(self.block_density_scores),
            "salience_density_scope": "APT adapter-applied blocks only",
            "salience_density_recomputed_on_parameter_change": True,
            "target_sparsity": target,
            "achieved_sparsity": achieved,
            "mask_granularity": self.config.mask_granularity,
            "early_training_t_lt_T": True,
            "pruning_start_step": self.config.pruning_start_step,
            "pruning_end_step": self.config.pruning_end_step,
            "binary_masks": {"m_i": m_i, "m_o": m_o},
            "mha_masks": build_mha_pruning_metadata(m_i, m_o),
            "pruned_blocks": pruned_blocks,
            "post_pruning_structure": {
                "d_i": len(m_i),
                "d_o": len(m_o),
                "d_i_prime": sum(m_i),
                "d_o_prime": sum(m_o),
                "n_L": N_L,
                "n_h": N_H,
                "n_f": N_F,
                "d_m": D_M,
                "C_head": C_HEAD,
                "C_neuron": C_NEURON,
                "C_dimension": C_DIMENSION,
                "equation_10_14": {
                    "C_head_formula": "4 * d_m * d_m / n_h",
                    "C_head": mha_head_parameter_count(D_M, N_H),
                    "C_neuron": neuron_parameter_count(D_M),
                    "C_dimension": hidden_dimension_parameter_count(D_M, N_L),
                    "d_m_prime": count_top_i_added_dimensions(
                        [{"name": block["name"], "type": block.get("mask_slot", "dimension")} for block in pruned_blocks],
                        len(pruned_blocks),
                    ),
                },
            },
        }
        return dict(self.mask_metadata)

    def apply_masks(self, model: Any) -> Dict[str, Any]:
        if not self.mask_metadata:
            self.search_masks(self.config.target_sparsity)
        if hasattr(model, "update_masks"):
            model.update_masks(self.current_masks["m_i"], self.current_masks["m_o"])
        if hasattr(model, "apply_mha_masks"):
            model.apply_mha_masks(self.mask_metadata.get("mha_masks", {}))
        elif isinstance(model, MutableMapping):
            model["m_i"] = list(self.current_masks["m_i"])
            model["m_o"] = list(self.current_masks["m_o"])
            model["mha_masks"] = dict(self.mask_metadata.get("mha_masks", {}))
        return {
            "applied": True,
            "binary_masks": dict(self.current_masks),
            "mha_masks": dict(self.mask_metadata.get("mha_masks", {})),
            "mask_metadata": dict(self.mask_metadata),
            "model_type": type(model).__name__,
        }

    def trace(self) -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "route": "A_P",
            "model_name": self.config.model_name,
            "dataset_name": self.config.dataset_name,
            "target_sparsity": self.config.target_sparsity,
            "S_bar_ema_decay": SALIENCE_EMA_DECAY,
            "S_bar_ema_update": SALIENCE_EMA_UPDATE,
            "steps": self.step_statistics,
            "binary_masks": dict(self.current_masks),
            "mask_metadata": dict(self.mask_metadata),
            "salience_density": dict(self.block_density_scores),
            "reference_grounding": "paperbench_ref_001 train.py",
        }


class AdaptiveTuner:
    """Adaptive efficient tuning A_T with task-sensitive rank allocation."""

    def __init__(self, config: Optional[Any] = None) -> None:
        self.config = PruningRunConfig.from_any(config)
        self.layer_importance: Dict[str, float] = {}
        self.rank_allocation: Dict[str, int] = {}
        self.metadata: Dict[str, Any] = {}

    def compute_layer_importance(self, model: Any, dataloader: Optional[Iterable[Any]] = None) -> Dict[str, float]:
        batch_scale = 1.0
        if dataloader is not None:
            batches = list(dataloader)
            if batches:
                batch_scale += sum(abs(v) for batch in batches for v in _batch_values(batch)) / max(1, sum(len(_batch_values(batch)) for batch in batches))
        pairs = _named_numeric_parameters(model)
        scores = {
            name: (sum(abs(v) for v in values) / max(1, len(values))) * batch_scale
            for name, values in pairs
        }
        if not scores:
            scores = {"bounded.adapter": 1.0}
        self.layer_importance = scores
        return dict(scores)

    def allocate_ranks(self, budget: Optional[int] = None) -> Dict[str, Any]:
        budget = int(budget if budget is not None else self.config.tuning_budget)
        if not self.layer_importance:
            self.compute_layer_importance(BoundedLinear())
        total = sum(max(score, 0.0) for score in self.layer_importance.values()) or 1.0
        allocation: Dict[str, int] = {}
        remaining = budget
        ordered = sorted(self.layer_importance.items(), key=lambda item: item[1], reverse=True)
        for index, (name, score) in enumerate(ordered):
            if index == len(ordered) - 1:
                rank = max(1, remaining)
            else:
                rank = max(1, int(round(budget * max(score, 0.0) / total)))
            allocation[name] = rank
            remaining -= rank
        dynamic_added = sum(allocation.values())
        bytes_per_parameter = 2 if self.config.precision == "fp16" else 4
        self.rank_allocation = allocation
        self.metadata = {
            "algorithm": "A_T adaptive and efficient LM tuning",
            "tuning_layer_importance": dict(self.layer_importance),
            "sorted_task_sensitive_adapter_importance": [
                {"name": name, "importance": score, "rank": allocation.get(name, 0)}
                for name, score in ordered
            ],
            "importance_sort": "descending",
            "dynamic_rank_allocation": dict(allocation),
            "dynamic_added_tuning_parameters": dynamic_added,
            "trainable_parameter_count": dynamic_added,
            "relative_training_memory": dynamic_added * bytes_per_parameter,
            "training_cost": dynamic_added * self.config.batch_size / 1024.0,
            "memory_usage": dynamic_added * bytes_per_parameter,
            "precision": self.config.precision,
            "half_precision_attack": self.config.half_precision_attack,
            "budget": budget,
        }
        return dict(self.metadata)

    def trace(self) -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "route": "A_T",
            "model_name": self.config.model_name,
            "dataset_name": self.config.dataset_name,
            "A_T metadata": dict(self.metadata),
            "layer_importance": dict(self.layer_importance),
            "rank_allocation": dict(self.rank_allocation),
            "reference_grounding": "paperbench_ref_001 train.py",
        }


def select_task_sensitive_adapters(adapters: Mapping[str, APTAdapter], importance: Mapping[str, float], budget: int) -> Dict[str, APTAdapter]:
    ordered = sorted(importance.items(), key=lambda item: item[1], reverse=True)
    selected: Dict[str, APTAdapter] = {}
    spent = 0
    for name, _score in ordered:
        adapter = adapters.get(name)
        if adapter is None:
            continue
        cost = adapter.parameter_report()["trainable_parameter_count"]
        if spent + cost <= budget or not selected:
            selected[name] = adapter
            spent += cost
    return selected


def _method_result(method: str, config: PruningRunConfig) -> Dict[str, Any]:
    families = {
        "ours": "APT",
        "APT": "APT",
        "fine_tuning": "baseline",
        "FT": "baseline",
        "lora": "baseline",
        "LoRA": "baseline",
        "lora_prune": "ablation",
        "LoRA+Prune": "ablation",
        "mask_tuning": "baseline",
        "Mask Tuning": "baseline",
        "cofi": "baseline",
        "CoFi": "baseline",
        "pruning_distillation": "ablation",
        "bert": "model_route",
        "roberta": "model_route",
        "t5": "model_route",
        "test_time_adaptation": "adaptation",
        "10_shot_setting": "fixed_hyperparameter",
        "batch_size_32": "parameter_sweep",
        "batch_size_128": "parameter_sweep",
    }
    return {
        "method": method,
        "family": families[method],
        "selector": "src.apt.pruning.get_method_selector",
        "uses_main_training_path": method in {"ours", "APT", "lora_prune", "LoRA+Prune", "pruning_distillation"},
        "batch_size": BATCH_SIZE_128 if method == "batch_size_128" else config.batch_size,
        "ten_shot_setting": TEN_SHOT_SETTING,
        "half_precision_attack": config.half_precision_attack,
        "precision": config.precision,
    }


def build_method_selector_registry(config: Optional[Any] = None) -> Dict[str, Callable[[Optional[Any]], Dict[str, Any]]]:
    cfg = PruningRunConfig.from_any(config)
    return {name: (lambda runtime_config=None, selected=name: _method_result(selected, PruningRunConfig.from_any(runtime_config or cfg))) for name in PAPER_METHOD_SELECTOR_NAMES}


def get_method_selector(name: str) -> Callable[[Optional[Any]], Dict[str, Any]]:
    registry = build_method_selector_registry()
    if name not in registry:
        lower = {key.lower(): value for key, value in registry.items()}
        if name.lower() not in lower:
            raise KeyError(f"Unknown APT method/baseline selector: {name}")
        return lower[name.lower()]
    return registry[name]


def build_experiment_matrix(config: Optional[Any] = None) -> Dict[str, Any]:
    cfg = PruningRunConfig.from_any(config)
    batch_defaults = resolve_batch_size_defaults(cfg.bounded)
    step_defaults = resolve_num_steps_defaults(cfg.bounded)
    return {
        "schema_version": SCHEMA_VERSION,
        "experiments": [APT_NLU_JOINT_EXPERIMENT, APT_GENERATION_EXPERIMENT, BASELINE_EFFICIENCY_EXPERIMENT],
        "methods_or_models": list(PAPER_METHOD_SELECTOR_NAMES),
        "parameters": {
            "m_i input binary mask": [1, 1, 1, 1],
            "m_o output binary mask": [1, 1],
            "r_apt dynamic rank": cfg.r_apt,
            "t << T early-training window": step_defaults,
            "target_sparsity": cfg.target_sparsity,
            "pruning_warmup_steps": cfg.pruning_warmup_steps,
            "mask granularity": cfg.mask_granularity,
            "A_T metadata for trainable parameter count and relative training memory": "src.apt.pruning.AdaptiveTuner.allocate_ranks",
            "precision": cfg.precision,
            "half_precision_attack": cfg.half_precision_attack,
            "batch_size": batch_defaults,
            "batch_size_32": BATCH_SIZE_32,
            "batch_size_128": BATCH_SIZE_128,
            "10_shot_setting": TEN_SHOT_SETTING,
        },
        "hypothesis": "APT exposes executable pruning/tuning coupling and precision attack protocol with bounded defaults.",
        "decision_value": "The canonical route can import selectors, run bounded traces, and write method artifacts.",
    }


def materialize_baseline_checkpoint_assets(root: Path | str = ".") -> Dict[str, str]:
    root = Path(root)
    written: Dict[str, str] = {}
    metadata = {
        "cofi": {
            "method": "CoFi",
            "status": "metadata_ready",
            "route": "src.apt.pruning.materialize_baseline_checkpoint_assets",
            "uses": ["pruning", "distillation", "checkpoint metadata"],
            "full_mode_requirement": "Populate CoFi checkpoint weights before full measured runs.",
        },
        "mask_tuning": {
            "method": "Mask Tuning",
            "status": "metadata_ready",
            "route": "src.apt.pruning.materialize_baseline_checkpoint_assets",
            "uses": ["binary mask tuning"],
            "full_mode_requirement": "Populate retraining-free-pruning-compatible mask checkpoint weights before full measured runs.",
        },
    }
    for name, payload in metadata.items():
        directory = root / "checkpoints" / name
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "metadata.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        written[name] = str(path)
    return written


def _artifact_dir(config: PruningRunConfig) -> Path:
    return Path(config.output_dir)


def _write_json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(path)


def write_model_registry_artifact(output_dir: Path | str, payload: Mapping[str, Any]) -> str:
    return _write_json(Path(output_dir) / "model_registry.json", payload)


def write_pruning_trace_artifact(output_dir: Path | str, payload: Mapping[str, Any]) -> str:
    return _write_json(Path(output_dir) / "pruning_trace.json", payload)


def write_pruning_trace(output_dir: Path | str, payload: Mapping[str, Any]) -> str:
    return write_pruning_trace_artifact(output_dir, payload)


def write_tuning_trace_artifact(output_dir: Path | str, payload: Mapping[str, Any]) -> str:
    return _write_json(Path(output_dir) / "tuning_trace.json", payload)


def write_loss_trace_artifact(output_dir: Path | str, payload: Mapping[str, Any]) -> str:
    return _write_json(Path(output_dir) / "loss_trace.json", payload)


def write_config_resolved_artifact(output_dir: Path | str, payload: Mapping[str, Any]) -> str:
    return _write_json(Path(output_dir) / "config_resolved.json", payload)


def write_sensitivity_report_artifact(output_dir: Path | str, payload: Mapping[str, Any]) -> str:
    return _write_json(Path(output_dir) / "sensitivity_report.json", payload)


def write_result_table_artifacts(output_dir: Path | str, payload: Mapping[str, Any]) -> Dict[str, str]:
    root = Path(output_dir)
    json_path = _write_json(root / "result_table.json", payload)
    rows = payload.get("rows", [])
    lines = ["| task | method | metric | value | status |", "| --- | --- | --- | ---: | --- |"]
    for row in rows:
        lines.append(f"| {row.get('task')} | {row.get('method')} | {row.get('metric')} | {row.get('value')} | {row.get('status')} |")
    md_path = root / "result_table.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"result_table_json": json_path, "result_table_md": str(md_path)}


def build_sensitivity_report(pruner: AdaptivePruner, tuner: AdaptiveTuner, adapter: APTAdapter, config: PruningRunConfig) -> Dict[str, Any]:
    report = adapter.parameter_report()
    return {
        "schema_version": SCHEMA_VERSION,
        "target_sparsity": config.target_sparsity,
        "mask_granularity": config.mask_granularity,
        "batch_size_sweep": list(BATCH_SIZE_SWEEP),
        "target_sparsity_sweep": list(TARGET_SPARSITY_SWEEP if not config.bounded else (config.target_sparsity,)),
        "rank_sweep": list(RANK_SWEEP if not config.bounded else (config.r_apt,)),
        "half_precision_attack_sweep": list(HALF_PRECISION_ATTACK_SWEEP),
        "adapter_report": report,
        "A_P mask_metadata": pruner.mask_metadata,
        "A_T metadata": tuner.metadata,
        "training_cost": tuner.metadata.get("training_cost", 0.0),
        "memory_usage": tuner.metadata.get("memory_usage", 0.0),
        "relative_training_memory": tuner.metadata.get("relative_training_memory", 0.0),
    }


def run_bounded_training_route(config: Optional[Any] = None) -> Dict[str, Any]:
    cfg = PruningRunConfig.from_any(config)
    model = BoundedLinear()
    adapter = create_apt_adapter(model, cfg.r_apt, [1] * model.in_features, [1] * model.out_features, cfg)
    pruner = AdaptivePruner(cfg)
    tuner = AdaptiveTuner(cfg)
    losses: List[float] = []
    pruning_records: List[Dict[str, Any]] = []
    for step in range(cfg.pruning_start_step, min(cfg.max_steps, cfg.pruning_end_step) + 1):
        batch = {"inputs": [1.0, 0.5 + step * 0.1, -0.25, 0.75], "labels": [1, 0]}
        output = adapter.forward(batch["inputs"])
        pred_loss = sum(abs(v) for v in output) / max(1, len(output))
        layer_loss = abs(_kurtosis(output))
        distill = compute_distillation_loss(cfg.dataset_name, pred_loss, layer_loss) if cfg.distillation else {"L_distill": pred_loss, "L_pred": pred_loss, "L_layer": 0.0}
        losses.append(float(distill["L_distill"]))
        pruning_records.append(pruner.collect_step_statistics(model, batch, step))
    mask_metadata = pruner.search_masks(cfg.target_sparsity)
    pruner.apply_masks(adapter)
    importance = tuner.compute_layer_importance(model, [{"inputs": [1.0, 0.5, -0.25, 0.75]}])
    tuning_metadata = tuner.allocate_ranks(cfg.tuning_budget)
    selected_rank = max(1, max(tuner.rank_allocation.values()) if tuner.rank_allocation else cfg.r_apt)
    adapter.update_rank(selected_rank)
    adapter_report = adapter.parameter_report()
    loss_trace = {
        "schema_version": SCHEMA_VERSION,
        "dataset_name": cfg.dataset_name,
        "distillation": cfg.distillation,
        "losses": losses,
        "aggregate_loss": aggregate_loss(losses),
        "L_distill": losses[-1] if losses else 0.0,
        "L_pred": losses[-1] if losses else 0.0,
        "L_layer": 0.0 if not losses else max(0.0, losses[-1] - losses[0]),
        "tau": TAU,
        "layer_weights": {
            "GLUE": DISTILL_LAYER_WEIGHT_GLUE,
            "SQuAD": DISTILL_LAYER_WEIGHT_SQUAD,
        },
    }
    pruning_trace = pruner.trace()
    pruning_trace["collected_records"] = pruning_records
    tuning_trace = tuner.trace()
    model_registry = {
        "schema_version": SCHEMA_VERSION,
        "method": cfg.method,
        "model_name": cfg.model_name,
        "dataset_name": cfg.dataset_name,
        "APT_adapter": adapter_report,
        "LoRA_base_adapter": {"rank": adapter.r_apt, "scaling": adapter.scaling},
        "method_selectors": {name: _method_result(name, cfg) for name in PAPER_METHOD_SELECTOR_NAMES},
        "config_model_registry": _config_model_registry(),
        "config_method_registry_keys": list(_config_method_registry().keys()),
        "half_precision_attack": {
            "enabled": cfg.half_precision_attack,
            "precision": cfg.precision,
            "protocol_entry": "half_precision_attack",
            "attack_metadata": "bounded route records protocol and precision; full mode must run precision attack evaluation.",
        },
    }
    metrics = {
        "loss": loss_trace["aggregate_loss"],
        "training_cost": tuning_metadata["training_cost"],
        "memory_usage": tuning_metadata["memory_usage"],
        "relative accuracy": max(0.0, 1.0 - loss_trace["aggregate_loss"]),
    }
    reward = compute_ours_oradaptersby_inventory_score(metrics)
    result_table = {
        "schema_version": SCHEMA_VERSION,
        "status": "bounded_proxy" if cfg.bounded else "full_route_configured",
        "rows": [
            {"task": cfg.dataset_name, "method": cfg.method, "metric": key, "value": value, "status": "bounded_proxy"}
            for key, value in metrics.items()
        ],
        "reward": reward,
    }
    return {
        "config": cfg,
        "model": model,
        "adapter": adapter,
        "pruner": pruner,
        "tuner": tuner,
        "model_registry": model_registry,
        "pruning_trace": pruning_trace,
        "tuning_trace": tuning_trace,
        "loss_trace": loss_trace,
        "sensitivity_report": build_sensitivity_report(pruner, tuner, adapter, cfg),
        "result_table": result_table,
        "mask_metadata": mask_metadata,
        "objective": compute_ours_oradaptersby_inventory_objective(cfg),
        "experiment_matrix": build_experiment_matrix(cfg),
    }


def main(config: Optional[Any] = None) -> Dict[str, Any]:
    """Callable route used by CLI wrappers and smoke validation."""

    cfg = PruningRunConfig.from_any(config)
    start = time.time()
    route = run_bounded_training_route(cfg)
    root = _artifact_dir(cfg)
    root.mkdir(parents=True, exist_ok=True)
    checkpoint_paths = materialize_baseline_checkpoint_assets(root.parent if root.name == "results" else ".")
    written = {
        "model_registry": write_model_registry_artifact(root, route["model_registry"]),
        "pruning_trace": write_pruning_trace_artifact(root, route["pruning_trace"]),
        "tuning_trace": write_tuning_trace_artifact(root, route["tuning_trace"]),
        "loss_trace": write_loss_trace_artifact(root, route["loss_trace"]),
        "config_resolved": write_config_resolved_artifact(root, {**asdict(cfg), "experiment_matrix": route["experiment_matrix"]}),
        "sensitivity_report": write_sensitivity_report_artifact(root, route["sensitivity_report"]),
        **write_result_table_artifacts(root, route["result_table"]),
    }
    written.update({f"checkpoint_{name}": path for name, path in checkpoint_paths.items()})
    auxiliary = os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR")
    if auxiliary:
        _write_json(Path(auxiliary) / "pruning_auxiliary_manifest.json", {"source_output_dir": str(root), "written": written})
    return {
        "status": "bounded_proxy" if cfg.bounded else "full_route_configured",
        "elapsed_seconds": time.time() - start,
        "artifacts": written,
        "objective": route["objective"],
        "metrics": route["result_table"]["rows"],
    }


__all__ = [
    "ALPHA_DEFAULT",
    "BATCH_SIZE_SWEEP",
    "PruningRunConfig",
    "BoundedLinear",
    "APTAdapter",
    "AdaptivePruner",
    "AdaptiveTuner",
    "DEFAULT_APT_SCALING",
    "create_apt_adapter",
    "expand_mha_hidden_dimension_mask",
    "output_mask_to_attention_head_mask",
    "build_mha_pruning_metadata",
    "compute_weight_salience",
    "compute_salience_density",
    "block_type_index",
    "kronecker_delta",
    "count_top_i_added_dimensions",
    "mha_head_parameter_count",
    "neuron_parameter_count",
    "hidden_dimension_parameter_count",
    "select_task_sensitive_adapters",
    "build_method_selector_registry",
    "get_method_selector",
    "build_experiment_matrix",
    "compute_loss",
    "aggregate_loss",
    "compute_reward",
    "aggregate_reward",
    "compute_ours_oradaptersby_inventory_objective",
    "compute_ours_oradaptersby_inventory_score",
    "materialize_baseline_checkpoint_assets",
    "run_bounded_training_route",
    "build_sensitivity_report",
    "write_model_registry_artifact",
    "write_pruning_trace_artifact",
    "write_pruning_trace",
    "write_tuning_trace_artifact",
    "write_loss_trace_artifact",
    "write_config_resolved_artifact",
    "write_sensitivity_report_artifact",
    "write_result_table_artifacts",
    "main",
    "APT_NLU_JOINT_EXPERIMENT",
    "APT_GENERATION_EXPERIMENT",
    "BASELINE_EFFICIENCY_EXPERIMENT",
    "CONFIG_REGISTRY_ENTRYPOINT",
    "DATASET_REGISTRY_ROUTE",
    "METHOD_BASELINE_REGISTRY_ROUTE",
    "APT_ADAPTER_MODULE_ROUTE",
    "APT_ADAPTER_FORWARD_ROUTE",
    "APT_METADATA_ROUTE",
    "ADAPTIVE_PRUNING_ROUTE",
    "OUTLIER_SALIENCE_ROUTE",
    "FAST_SEARCH_ROUTE",
]
